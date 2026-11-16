/**
 * context-dump: capture the exact JSON payload sent to the LLM provider.
 *
 * The latest request of each session is written next to the session file, overwriting the
 * previous one: ~/.pi/agent/sessions/--<path>--/<timestamp>_<session-id>.context.json
 * (sessions without a file go to ~/.pi/agent/context-dumps/<session-id>.context.json). It contains:
 *   - meta (provider, model, time, context window, compaction reserve)
 *   - headers (auth values redacted)
 *   - payload (the full provider request body: system prompt, tools, messages, params)
 *   - usage (the provider's token report for this request, from the assistant message)
 *   - response status/headers
 *
 * Commands:
 *   /context            Explore the last request in a collapsible tree panel
 *                       Arrow keys or vim keys: ↑↓/jk move, ←→/hl close/open, ⇧↑↓/JK sibling,
 *                       / ? n N search. Arrays longer than 20 items are grouped.
 *   /context raw        View the last request as raw JSON in an editor (read-only)
 *   /context path       Show where the last dump was written
 *   /context off|on     Toggle capture
 *
 * The tree explorer is structure-agnostic: it walks whatever JSON the provider got.
 * The usage summary maps the payload onto Claude Code-style categories via the rules in
 * BUCKETS / SECTION_BUCKETS / *_KEYS below; anything unrecognised is counted as "Other".
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	copyToClipboard,
	getAgentDir,
	type ExtensionContext,
	type ExtensionAPI,
	type ExtensionCommandContext,
	type Theme,
} from "@earendil-works/pi-coding-agent";
import {
	type Component,
	decodeKittyPrintable,
	matchesKey,
	type TUI,
	truncateToWidth,
	visibleWidth,
	wrapTextWithAnsi,
} from "@earendil-works/pi-tui";

const SECRET = /auth|key|token|secret|cookie/i;

// ── Explorer hints (the only "schema knowledge"; unknown shapes still render fine) ──
/** String fields used to label an object, e.g. `[3] user`, `[0] tool_use · bash`. */
const KIND_KEYS = ["role", "type"];
/** Name fields; also searched one level down (e.g. OpenAI tools: `{ type, function: { name } }`). */
const NAME_KEYS = ["name", "title"];
/** Fallback label key when none of LABEL_KEYS exist. */
const FALLBACK_LABEL_KEYS = ["id"];
/** Array fields whose item `type`s are summarized in the parent label, e.g. `[text, tool_use]`. */
const BLOCK_KEYS = ["content", "parts"];
/** Strings longer than this (or multi-line) become expandable text. */
const INLINE_MAX = 40;
/** Strings at least this long that look like base64 are collapsed to a size badge. */
const BLOB_MIN = 256;
const CHARS_PER_TOKEN = 4;
/** Arrays longer than this are split into collapsible ranges (nested for very long arrays). */
const GROUP_SIZE = 20;
const MAX_MATCHES = 10_000;

/** Smallest power of GROUP_SIZE that splits `n` items into at most GROUP_SIZE groups. */
function groupSpan(n: number): number {
	let g = GROUP_SIZE;
	while (Math.ceil(n / g) > GROUP_SIZE) g *= GROUP_SIZE;
	return g;
}

type Path = (string | number)[];
type Dump = Record<string, unknown>;

type Entry = [string | number, unknown];

interface Row {
	id: string;
	/** For nodes: their own path. For text lines: the owning string. For groups: the array. */
	path: Path;
	depth: number;
	kind: "node" | "text" | "group";
	key?: string | number;
	value?: unknown;
	text?: string;
	expandable?: boolean;
	/** Group rows: [start, end) positions in the (possibly sorted) entry list, and those entries. */
	range?: [number, number];
	items?: Entry[];
}

interface Match {
	/** Tree-order position of the matched node. */
	seq: number;
	target: string;
	/** Ancestor node and group ids to expand to reveal the match. */
	expand: string[];
	/** For long strings: which occurrence within the string (to land on the right text line). */
	occurrence?: number;
}

interface ExplorerState {
	dump: Dump;
	expanded: Set<string>;
	cursorId?: string;
	scroll: number;
	sort: boolean;
	summary: boolean;
	/** Last committed search, kept across reopen like vim's last pattern. */
	query?: string;
	searchDir?: 1 | -1;
}

// ── Generic JSON helpers ──

const isObj = (v: unknown): v is Record<string, unknown> | unknown[] => typeof v === "object" && v !== null;
const pathId = (p: Path) => p.join("\u0001");
const fmtPath = (p: Path) =>
	p.map((s, i) => (typeof s === "number" ? `[${s}]` : i === 0 ? s : `.${s}`)).join("") || "(root)";

function entries(v: Record<string, unknown> | unknown[]): Entry[] {
	return Array.isArray(v) ? v.map((x, i) => [i, x]) : Object.entries(v);
}

function blobLabel(s: string): string | undefined {
	if (s.length < BLOB_MIN) return undefined;
	const kb = `${Math.round((s.length * 3) / 4 / 1024)} KB`;
	const dataUrl = /^data:([\w/+.-]+);base64,/.exec(s);
	if (dataUrl) return `[${dataUrl[1]} ${kb}]`;
	const sample = s.slice(0, 512);
	const looksB64 =
		/^[A-Za-z0-9+/=_-]+$/.test(sample) &&
		/[A-Z]/.test(sample) &&
		/[a-z]/.test(sample) &&
		/[0-9]/.test(sample) &&
		new Set(sample).size > 20;
	if (looksB64) return `[base64 ${kb}]`;
	return undefined;
}

function isExpandable(v: unknown): boolean {
	if (typeof v === "string") return !blobLabel(v) && (v.length > INLINE_MAX || v.includes("\n"));
	return isObj(v) && entries(v).length > 0;
}

function shortString(v: unknown): string | undefined {
	return typeof v === "string" && v.length > 0 && v.length <= 60 && !v.includes("\n") ? v : undefined;
}

function preview(s: string, max = 80): string {
	const flat = s.replace(/\s+/g, " ").trim();
	return `"${flat.length > max ? `${flat.slice(0, max)}…` : flat}"`;
}

/** Build a human label for an object from whatever identifying fields it has. */
function describe(obj: Record<string, unknown>): string {
	const pick = (o: Record<string, unknown>, keys: string[]) =>
		keys.map((k) => shortString(o[k])).filter((s): s is string => !!s);
	let name = pick(obj, NAME_KEYS)[0];
	if (!name) {
		for (const v of Object.values(obj)) {
			if (isObj(v) && !Array.isArray(v) && (name = pick(v, NAME_KEYS)[0])) break;
		}
	}
	let parts = [...new Set([...pick(obj, KIND_KEYS), ...(name ? [name] : [])])].slice(0, 2);
	if (parts.length === 0) parts = pick(obj, FALLBACK_LABEL_KEYS).slice(0, 1);
	let label = parts.join(" · ");
	for (const k of BLOCK_KEYS) {
		const c = obj[k];
		if (typeof c === "string") label += ` ${preview(c, 50)}`;
		else if (Array.isArray(c)) {
			const types = [
				...new Set(c.map((b) => (isObj(b) && !Array.isArray(b) ? (shortString(b.type) ?? "") : "")).filter(Boolean)),
			];
			if (types.length) label += ` [${types.join(", ")}]`;
		}
	}
	return label.trim();
}

/** Approximate token cost of one image/binary blob (Anthropic: ~1.6k tokens for a ~1MP image). */
const BLOB_TOKENS = 1600;

// ── Usage buckets (Claude Code-style categories) ──
// Each string in the payload is attributed to one bucket based on where it sits. Anything the
// rules don't recognise lands in "Other", so unknown payload shapes still add up correctly.

const BUCKETS = [
	{ name: "System prompt", color: "muted" },
	{ name: "System tools", color: "borderAccent" },
	{ name: "MCP tools", color: "accent" },
	{ name: "MCP server instructions", color: "success" },
	{ name: "Memory files", color: "syntaxNumber" },
	{ name: "Skills", color: "warning" },
	{ name: "Messages", color: "syntaxKeyword" },
	{ name: "Compaction summary", color: "syntaxString" },
	{ name: "Other", color: "dim" },
] as const;
type Bucket = (typeof BUCKETS)[number]["name"];

/** Top-level payload keys by role (Anthropic, OpenAI chat/responses, Gemini). */
const SYSTEM_KEYS = /^(system|systemInstruction|system_instruction|instructions)$/;
const TOOL_KEYS = /^(tools|functions)$/;
const MESSAGE_KEYS = /^(messages|contents|input)$/;
/** Message roles that carry system prompt text inside the message list. */
const SYSTEM_ROLES = new Set(["system", "developer"]);
/** pi system prompt sections (`<name>…</name>`) that get their own bucket. */
const SECTION_BUCKETS: Record<string, Bucket> = {
	project_context: "Memory files",
	skills: "Skills",
	mcp_servers: "MCP server instructions",
};
const isMcpTool = (name: string) =>
	name.startsWith("mcp__") || ["list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"].includes(name);
/** Prefixes pi uses for compaction / branch summaries injected into the conversation. */
const COMPACTION_MARKERS = [
	"The conversation history before this point was compacted",
	"The following is a summary of a branch",
];

interface BucketStats {
	chars: Map<Bucket, number>;
	counts: Map<Bucket, { n: number; unit: string }>;
}

function classify(payload: unknown, size: (v: unknown) => number): BucketStats {
	const chars = new Map<Bucket, number>();
	const counts = new Map<Bucket, { n: number; unit: string }>();
	const add = (b: Bucket, n: number) => chars.set(b, (chars.get(b) ?? 0) + n);
	const count = (b: Bucket, n: number, unit: string) => {
		if (n > 0) counts.set(b, { n: (counts.get(b)?.n ?? 0) + n, unit });
	};
	const keyCost = (k: string | number) => (typeof k === "string" ? k.length + 4 : 1);

	/** System prompt text: carve out pi's tagged sections into their own buckets. */
	const addSystemText = (s: string) => {
		let rest = s.length + 2;
		for (const m of s.matchAll(/<([a-z][a-z0-9_-]*)>\n[\s\S]*?\n<\/\1>/g)) {
			const b = SECTION_BUCKETS[m[1]!];
			if (!b) continue;
			add(b, m[0].length);
			rest -= m[0].length;
			if (b === "Skills") count(b, m[0].match(/<skill>/g)?.length ?? 0, "skills");
			if (b === "Memory files") count(b, m[0].match(/<project_instructions /g)?.length ?? 0, "files");
		}
		add("System prompt", rest);
	};

	type Mode = "system" | "message" | "plain";
	const walk = (v: unknown, bucket: Bucket, mode: Mode) => {
		if (typeof v === "string") {
			if (mode === "system" && !blobLabel(v)) return addSystemText(v);
			if (mode === "message" && COMPACTION_MARKERS.some((p) => v.slice(0, 200).includes(p)))
				return add("Compaction summary", size(v));
			return add(bucket, size(v));
		}
		if (!isObj(v)) return add(bucket, size(v));
		// A system/developer message inside the message list counts as system prompt.
		if (mode === "message" && !Array.isArray(v) && SYSTEM_ROLES.has(v.role as string)) {
			bucket = "System prompt";
			mode = "system";
		}
		let overhead = 2;
		for (const [k, x] of entries(v)) {
			overhead += keyCost(k);
			walk(x, bucket, mode);
		}
		add(bucket, overhead);
	};

	/** Tools: one bucket per tool, MCP vs built-in by name. Handles Gemini's nested declarations. */
	const walkTools = (tools: unknown) => {
		if (!Array.isArray(tools)) return walk(tools, "System tools", "plain");
		add("System tools", 2 + tools.length);
		for (const t of tools) {
			const obj = isObj(t) && !Array.isArray(t) ? t : {};
			const nested = Object.values(obj).find(
				(x) => Array.isArray(x) && x.length > 0 && x.every((d) => isObj(d) && typeof (d as any).name === "string"),
			);
			if (!obj.name && nested) {
				walkTools(nested);
				continue;
			}
			const name =
				shortString(obj.name) ?? shortString((obj.function as Record<string, unknown> | undefined)?.name) ?? "";
			const b: Bucket = isMcpTool(name) ? "MCP tools" : "System tools";
			add(b, size(t));
			count(b, 1, "tools");
		}
	};

	if (!isObj(payload) || Array.isArray(payload)) {
		walk(payload, "Other", "plain");
		return { chars, counts };
	}
	add("Other", 2);
	for (const [k, v] of Object.entries(payload)) {
		add("Other", keyCost(k));
		if (TOOL_KEYS.test(k)) walkTools(v);
		else if (SYSTEM_KEYS.test(k)) walk(v, "System prompt", "system");
		else if (MESSAGE_KEYS.test(k)) {
			walk(v, "Messages", "message");
			if (Array.isArray(v)) {
				const n = v.filter((m) => !(isObj(m) && SYSTEM_ROLES.has((m as any).role))).length;
				count("Messages", n, "messages");
			}
		} else walk(v, "Other", "plain");
	}
	return { chars, counts };
}

function fmtTok(t: number): string {
	if (t < 1000) return `${Math.round(t)}`;
	if (t < 100_000) return `${(t / 1000).toFixed(1)}k`;
	if (t < 999_500) return `${Math.round(t / 1000)}k`;
	return `${+(t / 1_000_000).toFixed(1)}M`;
}

function sanitize(s: string): string {
	return s.replace(/\t/g, "  ").replace(/[\x00-\x08\x0b-\x1f\x7f]|\x1b/g, "�");
}

// ── Explorer component ──

class ContextExplorer implements Component {
	private sizes = new WeakMap<object, number>();
	/** The node whose children are the "sections" (payload if present, else the whole dump). */
	private root: unknown;
	private rootChars: number;
	/** Measured prompt tokens from the provider's usage report, if the response has arrived. */
	private measured?: number;
	private contextWindow?: number;
	private compactReserve?: number;
	private modelName?: string;
	private tokPerChar: number;
	private rows: Row[] = [];
	private cursor = 0;
	private builtWidth = -1;
	private lastWidth = 80;
	private summaryHeight = 0;
	private centerNext = false;
	private flash?: { msg: string; color: "success" | "warning" | "error" };
	private flashTimer?: ReturnType<typeof setTimeout>;
	/** Active `/` or `?` prompt; `origin` restores the view on Esc. */
	private input?: {
		text: string;
		dir: 1 | -1;
		origin: { cursorId?: string; expanded: Set<string>; scroll: number };
	};
	private matches: Match[] = [];
	private matchIdx = -1;
	/** Tree-order sequence number per node id (filled by findMatches). */
	private seqs = new Map<string, number>();
	private tui: TUI;
	private theme: Theme;
	private state: ExplorerState;
	private file: string;
	private done: () => void;

	constructor(tui: TUI, theme: Theme, state: ExplorerState, file: string, done: () => void) {
		this.tui = tui;
		this.theme = theme;
		this.state = state;
		this.file = file;
		this.done = done;
		const dump = state.dump;
		this.root = isObj(dump.payload) ? dump.payload : dump;
		this.rootChars = Math.max(1, this.size(this.root));
		const meta = (isObj(dump.meta) ? dump.meta : {}) as Record<string, unknown>;
		const usage = (isObj(dump.usage) ? dump.usage : undefined) as Record<string, unknown> | undefined;
		const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
		if (usage) this.measured = num(usage.input) + num(usage.cacheRead) + num(usage.cacheWrite) || undefined;
		this.contextWindow = num(meta.contextWindow) || undefined;
		this.compactReserve = num(meta.compactReserve) || undefined;
		this.modelName = typeof meta.model === "string" ? meta.model : undefined;
		// Calibrate char-based estimates against the real token count when we have it.
		this.tokPerChar = this.measured ? this.measured / this.rootChars : 1 / CHARS_PER_TOKEN;
	}

	private tok(v: unknown): number {
		return this.size(v) * this.tokPerChar;
	}

	private get usedTokens(): number {
		return this.measured ?? this.rootChars * this.tokPerChar;
	}

	/** Claude Code-style usage summary: cell grid on the left, legend on the right. */
	private renderSummary(inner: number): string[] {
		const th = this.theme;
		const used = this.usedTokens;
		const win = this.contextWindow;

		const stats = classify(this.root, (v) => this.size(v));
		const colored = BUCKETS.map((b) => ({
			...b,
			tokens: (stats.chars.get(b.name) ?? 0) * this.tokPerChar,
			count: stats.counts.get(b.name),
		})).filter((b) => b.tokens >= 1);
		const reserve = win ? Math.min(this.compactReserve ?? 0, Math.max(0, win - used)) : 0;
		const denom = win ?? used;
		const pct = (t: number) => {
			const p = (t / Math.max(1, denom)) * 100;
			return p > 0 && p < 0.1 ? "<0.1%" : `${p.toFixed(1)}%`;
		};

		const legend: string[] = [];
		legend.push(th.bold(this.modelName ?? "unknown model"));
		const usedStr = win
			? `${fmtTok(used)}/${fmtTok(win)} tokens (${Math.round((used / win) * 100)}%)`
			: `${fmtTok(used)} tokens`;
		legend.push(`${usedStr} ${th.fg("dim", this.measured ? "measured" : "estimated")}`);
		legend.push("");
		legend.push(th.fg("muted", th.italic(`${this.measured ? "Measured total, estimated" : "Estimated"} usage by category`)));
		for (const s of colored) {
			const cnt = s.count ? th.fg("dim", ` · ${s.count.n} ${s.count.n === 1 ? s.count.unit.replace(/s$/, "") : s.count.unit}`) : "";
			legend.push(`${th.fg(s.color, "■")} ${s.name}: ${th.fg("muted", `${fmtTok(s.tokens)} (${pct(s.tokens)})`)}${cnt}`);
		}
		if (reserve > 0) {
			legend.push(`${th.fg("dim", "▨")} Compact buffer: ${th.fg("muted", `${fmtTok(reserve)} (${pct(reserve)})`)}`);
		}
		if (win) {
			const free = Math.max(0, win - used - reserve);
			legend.push(`${th.fg("dim", "□")} Free space: ${th.fg("muted", `${fmtTok(free)} (${pct(free)})`)}`);
		}

		// Grid: each cell is 1/N of the window (or of used tokens if the window is unknown).
		const gridCols = 10;
		const gridW = gridCols * 2;
		if (inner < gridW + 30) return legend.map((l) => ` ${l}`);
		const gridRows = Math.max(4, Math.min(legend.length, 8));
		const n = gridCols * gridRows;
		const cells: string[] = [];
		for (const s of colored) {
			const k = Math.max(1, Math.round((s.tokens / Math.max(1, denom)) * n));
			for (let i = 0; i < k && cells.length < n; i++) cells.push(th.fg(s.color, "■"));
		}
		const reserveCells = reserve > 0 ? Math.max(1, Math.round((reserve / denom) * n)) : 0;
		while (cells.length < n - reserveCells) cells.push(th.fg("dim", "□"));
		while (cells.length < n) cells.push(th.fg("dim", "▨"));
		const lines: string[] = [];
		for (let r = 0; r < Math.max(gridRows, legend.length); r++) {
			const grid = r < gridRows ? cells.slice(r * gridCols, (r + 1) * gridCols).join(" ") : " ".repeat(gridW - 1);
			lines.push(` ${grid}  ${legend[r] ?? ""}`);
		}
		return lines;
	}

	private size(v: unknown): number {
		if (typeof v === "string") return (blobLabel(v) ? BLOB_TOKENS * CHARS_PER_TOKEN : v.length) + 2;
		if (!isObj(v)) return String(v).length;
		const cached = this.sizes.get(v);
		if (cached !== undefined) return cached;
		let n = 2;
		for (const [k, x] of entries(v)) n += (typeof k === "string" ? k.length + 4 : 1) + this.size(x);
		this.sizes.set(v, n);
		return n;
	}

	private sortedEntries(v: Record<string, unknown> | unknown[]) {
		const e = entries(v);
		return this.state.sort ? e.sort((a, b) => this.size(b[1]) - this.size(a[1])) : e;
	}

	private groupId(path: Path, start: number, end: number) {
		return `${pathId(path)}\u0002${this.state.sort ? "s" : ""}${start}-${end}`;
	}

	/** Ids of the (nested) groups that contain position `pos` of an array of length `len`. */
	private groupChain(path: Path, len: number, pos: number): string[] {
		const ids: string[] = [];
		let s = 0;
		let e = len;
		while (e - s > GROUP_SIZE) {
			const g = groupSpan(e - s);
			const gs = s + Math.floor((pos - s) / g) * g;
			const ge = Math.min(gs + g, e);
			ids.push(this.groupId(path, gs, ge));
			s = gs;
			e = ge;
		}
		return ids;
	}

	// Flatten visible tree into rows. Depends on width because text rows are wrapped.
	private build(width: number) {
		const inner = Math.max(10, width - 2);
		const rows: Row[] = [];
		const expanded = this.state.expanded;
		const emitRange = (list: Entry[], path: Path, depth: number, start: number, end: number) => {
			if (end - start <= GROUP_SIZE) {
				for (let i = start; i < end; i++) visit(list[i]![0], list[i]![1], [...path, list[i]![0]], depth);
				return;
			}
			const g = groupSpan(end - start);
			for (let s = start; s < end; s += g) {
				const e = Math.min(s + g, end);
				const id = this.groupId(path, s, e);
				rows.push({ id, path, depth, kind: "group", expandable: true, range: [s, e], items: list.slice(s, e) });
				if (expanded.has(id)) emitRange(list, path, depth + 1, s, e);
			}
		};
		const visit = (key: string | number, value: unknown, path: Path, depth: number) => {
			const id = pathId(path);
			const expandable = isExpandable(value);
			rows.push({ id, path, depth, kind: "node", key, value, expandable });
			if (!expandable || !expanded.has(id)) return;
			if (typeof value === "string") {
				const indent = (depth + 1) * 2 + 2;
				const w = Math.max(10, inner - indent);
				let n = 0;
				for (const line of sanitize(value).split("\n")) {
					for (const wrapped of line ? wrapTextWithAnsi(line, w) : [""]) {
						rows.push({ id: `${id}#${n++}`, path, depth: depth + 1, kind: "text", text: wrapped });
					}
				}
			} else if (isObj(value)) {
				const list = this.sortedEntries(value);
				if (Array.isArray(value)) emitRange(list, path, depth + 1, 0, list.length);
				else for (const [k, v] of list) visit(k, v, [...path, k], depth + 1);
			}
		};
		for (const [k, v] of this.sortedEntries(this.state.dump)) visit(k, v, [k], 0);
		this.rows = rows;
		this.builtWidth = width;
		const idx = this.state.cursorId ? rows.findIndex((r) => r.id === this.state.cursorId) : -1;
		this.cursor = Math.max(0, idx >= 0 ? idx : Math.min(this.cursor, rows.length - 1));
	}

	private ensureBuilt() {
		if (this.builtWidth !== this.lastWidth) this.build(this.lastWidth);
	}

	private rebuild() {
		this.builtWidth = -1;
		this.tui.requestRender();
	}

	private setCursor(i: number) {
		this.cursor = Math.max(0, Math.min(this.rows.length - 1, i));
		this.state.cursorId = this.rows[this.cursor]?.id;
		this.tui.requestRender();
	}

	private showFlash(msg: string, color: "success" | "warning" | "error" = "success") {
		this.flash = { msg, color };
		clearTimeout(this.flashTimer);
		this.flashTimer = setTimeout(() => {
			this.flash = undefined;
			this.tui.requestRender();
		}, 2500);
		this.tui.requestRender();
	}

	private get bodyHeight() {
		return Math.max(3, this.tui.terminal.rows - 6 - this.summaryHeight);
	}

	/** Depth used for structural navigation: text lines belong to their string node. */
	private navDepth(r: Row) {
		return r.kind === "text" ? r.depth - 1 : r.depth;
	}

	/** Index of the nearest row above `i` that is shallower than `depth` (the parent). */
	private parentIndex(i: number, depth: number) {
		for (let j = i - 1; j >= 0; j--) if (this.rows[j]!.depth < depth) return j;
		return -1;
	}

	/** The node/group row that owns row `i` (text rows → their string node). */
	private nodeAt(i: number): Row | undefined {
		const r = this.rows[i];
		if (!r || r.kind !== "text") return r;
		return this.rows[this.parentIndex(i, r.depth)];
	}

	/** Move to next/previous row at the same or shallower depth, skipping expanded content. */
	private jumpSibling(dir: 1 | -1) {
		const row = this.rows[this.cursor];
		if (!row) return;
		const d = this.navDepth(row);
		for (let j = this.cursor + dir; j >= 0 && j < this.rows.length; j += dir) {
			const r = this.rows[j]!;
			if (r.kind !== "text" && r.depth <= d) return this.setCursor(j);
		}
	}

	// ── Search (vim-style) ──

	private get activeQuery(): string | undefined {
		return (this.input ? this.input.text : this.state.query) || undefined;
	}

	/** Enumerate every match in tree order (respecting sort), with the ids needed to reveal it. */
	private findMatches(query: string): Match[] {
		const ci = query === query.toLowerCase();
		const norm = (s: string) => (ci ? s.toLowerCase() : s);
		const q = norm(query);
		const out: Match[] = [];
		this.seqs = new Map();
		let seq = 0;
		const walk = (key: string | number, value: unknown, path: Path, ancestors: string[]) => {
			if (out.length >= MAX_MATCHES) return;
			const id = pathId(path);
			const mySeq = seq++;
			this.seqs.set(id, mySeq);
			const keyHit = typeof key === "string" && norm(key).includes(q);
			if (keyHit) out.push({ seq: mySeq, target: id, expand: ancestors });
			if (typeof value === "string") {
				if (blobLabel(value)) return;
				const s = norm(value);
				let i = s.indexOf(q);
				if (i < 0) return;
				if (!isExpandable(value)) {
					if (!keyHit) out.push({ seq: mySeq, target: id, expand: ancestors });
					return;
				}
				for (let k = 0; i >= 0 && out.length < MAX_MATCHES; k++, i = s.indexOf(q, i + q.length)) {
					out.push({ seq: mySeq, target: id, expand: ancestors, occurrence: k });
				}
			} else if (isObj(value)) {
				const list = this.sortedEntries(value);
				const inner = [...ancestors, id];
				list.forEach(([k, v], pos) => {
					const groups = Array.isArray(value) ? this.groupChain(path, list.length, pos) : [];
					walk(k, v, [...path, k], groups.length ? [...inner, ...groups] : inner);
				});
			} else if (!keyHit && value !== undefined && norm(String(value)).includes(q)) {
				out.push({ seq: mySeq, target: id, expand: ancestors });
			}
		};
		for (const [k, v] of this.sortedEntries(this.state.dump)) walk(k, v, [k], []);
		return out;
	}

	/** Tree-order position of the cursor, for "next match after cursor". */
	private cursorSeq(): number {
		const node = this.nodeAt(this.cursor);
		if (!node) return -1;
		if (node.kind === "group") {
			const first = node.items?.[0];
			return first ? (this.seqs.get(pathId([...node.path, first[0]])) ?? 0) - 0.5 : -1;
		}
		const s = this.seqs.get(node.id) ?? -1;
		return this.rows[this.cursor]?.kind === "text" ? s + 0.25 : s;
	}

	/** Pick the match after (dir=1) or before (dir=-1) the cursor, wrapping like vim. */
	private nextMatchFromCursor(dir: 1 | -1): { idx: number; wrapped: boolean } | undefined {
		const n = this.matches.length;
		if (!n) return undefined;
		// Still sitting on the current match? Step from it, so multiple hits in one string work.
		const cur = this.matches[this.matchIdx];
		const here = this.nodeAt(this.cursor);
		if (cur && here?.id === cur.target) {
			const idx = this.matchIdx + dir;
			return { idx: (idx + n) % n, wrapped: idx < 0 || idx >= n };
		}
		const at = this.cursorSeq();
		if (dir === 1) {
			const idx = this.matches.findIndex((m) => m.seq > at);
			return idx >= 0 ? { idx, wrapped: false } : { idx: 0, wrapped: true };
		}
		for (let i = n - 1; i >= 0; i--) if (this.matches[i]!.seq < at) return { idx: i, wrapped: false };
		return { idx: n - 1, wrapped: true };
	}

	/** Reveal match `idx` (expand ancestors/groups/string) and put the cursor on it. */
	private gotoMatch(idx: number) {
		const m = this.matches[idx];
		if (!m) return;
		this.matchIdx = idx;
		for (const id of m.expand) this.state.expanded.add(id);
		if (m.occurrence !== undefined) this.state.expanded.add(m.target);
		this.state.cursorId = m.target;
		this.build(this.lastWidth);
		let target = this.rows.findIndex((r) => r.id === m.target);
		if (target >= 0 && m.occurrence !== undefined && this.activeQuery) {
			// Find the wrapped text line that holds the k-th occurrence.
			const q = this.activeQuery;
			const ci = q === q.toLowerCase();
			let seen = 0;
			for (let j = target + 1; j < this.rows.length && this.rows[j]!.kind === "text"; j++) {
				const t = ci ? (this.rows[j]!.text ?? "").toLowerCase() : (this.rows[j]!.text ?? "");
				const nq = ci ? q.toLowerCase() : q;
				for (let i = t.indexOf(nq); i >= 0; i = t.indexOf(nq, i + nq.length)) seen++;
				if (seen > m.occurrence) {
					target = j;
					break;
				}
			}
		}
		this.centerNext = true;
		this.setCursor(Math.max(0, target));
	}

	private startSearch(dir: 1 | -1) {
		this.input = {
			text: "",
			dir,
			origin: { cursorId: this.state.cursorId, expanded: new Set(this.state.expanded), scroll: this.state.scroll },
		};
		this.tui.requestRender();
	}

	/** Incremental search: restore the starting view, then jump to the first hit from there. */
	private updateIncremental() {
		const input = this.input!;
		this.state.expanded = new Set(input.origin.expanded);
		this.state.cursorId = input.origin.cursorId;
		this.state.scroll = input.origin.scroll;
		this.build(this.lastWidth);
		this.matches = input.text ? this.findMatches(input.text) : [];
		this.matchIdx = -1;
		const next = this.nextMatchFromCursor(input.dir);
		if (next) this.gotoMatch(next.idx);
		else this.tui.requestRender();
	}

	private handleSearchInput(data: string, ch: string | undefined) {
		const input = this.input!;
		if (matchesKey(data, "escape") || matchesKey(data, "ctrl+c")) {
			this.state.expanded = input.origin.expanded;
			this.state.cursorId = input.origin.cursorId;
			this.state.scroll = input.origin.scroll;
			this.input = undefined;
			this.matches = this.state.query ? this.findMatches(this.state.query) : [];
			return this.rebuild();
		}
		if (matchesKey(data, "return")) {
			this.input = undefined;
			// Empty `/` + Enter repeats the last search, like vim.
			const query = input.text || this.state.query;
			if (!query) return this.tui.requestRender();
			this.state.query = query;
			this.state.searchDir = input.dir;
			if (!input.text) {
				this.matches = this.findMatches(query);
				const next = this.nextMatchFromCursor(input.dir);
				if (next) this.gotoMatch(next.idx);
			}
			if (!this.matches.length) return this.showFlash(`Pattern not found: ${query}`, "error");
			return this.showFlash(`${input.dir === 1 ? "/" : "?"}${query}  [${this.matchIdx + 1}/${this.matches.length}]`);
		}
		if (matchesKey(data, "backspace")) {
			if (!input.text) return this.handleSearchInput("\x1b", undefined); // vim: backspace on empty cancels
			input.text = input.text.slice(0, -1);
			return this.updateIncremental();
		}
		if (ch && !/[\x00-\x1f\x7f]/.test(ch)) {
			input.text += ch;
			return this.updateIncremental();
		}
	}

	private searchStep(reverse: boolean) {
		const query = this.state.query;
		if (!query) return this.showFlash("No previous search", "warning");
		this.ensureBuilt();
		if (!this.matches.length) this.matches = this.findMatches(query);
		const dir = ((this.state.searchDir ?? 1) * (reverse ? -1 : 1)) as 1 | -1;
		const next = this.nextMatchFromCursor(dir);
		if (!next) return this.showFlash(`Pattern not found: ${query}`, "error");
		this.gotoMatch(next.idx);
		const wrap = next.wrapped ? (dir === 1 ? "  (hit BOTTOM, continuing at TOP)" : "  (hit TOP, continuing at BOTTOM)") : "";
		this.showFlash(`${dir === 1 ? "/" : "?"}${query}  [${next.idx + 1}/${this.matches.length}]${wrap}`, next.wrapped ? "warning" : "success");
	}

	handleInput(data: string): void {
		const ch = decodeKittyPrintable(data) ?? (data.length > 0 && !data.startsWith("\x1b") ? data : undefined);
		this.ensureBuilt(); // keys can arrive faster than renders
		if (this.input) return this.handleSearchInput(data, ch);
		const row = this.rows[this.cursor];
		const page = this.bodyHeight - 1;
		if (matchesKey(data, "escape") || ch === "q" || matchesKey(data, "ctrl+c")) return this.done();
		if (ch === "/") return this.startSearch(1);
		if (ch === "?") return this.startSearch(-1);
		if (ch === "n") return this.searchStep(false);
		if (ch === "N") return this.searchStep(true);
		if (matchesKey(data, "shift+up") || ch === "K" || ch === "{") return this.jumpSibling(-1);
		if (matchesKey(data, "shift+down") || ch === "J" || ch === "}") return this.jumpSibling(1);
		if (matchesKey(data, "up") || ch === "k") return this.setCursor(this.cursor - 1);
		if (matchesKey(data, "down") || ch === "j") return this.setCursor(this.cursor + 1);
		if (matchesKey(data, "pageUp") || matchesKey(data, "ctrl+u")) return this.setCursor(this.cursor - page);
		if (matchesKey(data, "pageDown") || matchesKey(data, "ctrl+d")) return this.setCursor(this.cursor + page);
		if (matchesKey(data, "home") || ch === "g") return this.setCursor(0);
		if (matchesKey(data, "end") || ch === "G") return this.setCursor(this.rows.length - 1);
		if (!row) return;

		if (matchesKey(data, "right") || matchesKey(data, "return") || ch === "l") {
			if (row.kind === "text" || !row.expandable) return;
			if (!this.state.expanded.has(row.id)) {
				this.state.expanded.add(row.id);
				this.state.cursorId = row.id;
				return this.rebuild();
			}
			return this.setCursor(this.cursor + 1); // already open → step into first child
		}
		if (matchesKey(data, "space")) {
			const node = this.nodeAt(this.cursor);
			if (!node?.expandable) return;
			if (this.state.expanded.has(node.id)) this.state.expanded.delete(node.id);
			else this.state.expanded.add(node.id);
			this.state.cursorId = node.id;
			return this.rebuild();
		}
		if (matchesKey(data, "left") || ch === "h") {
			if (row.kind !== "text" && this.state.expanded.has(row.id)) {
				this.state.expanded.delete(row.id);
				return this.rebuild();
			}
			const p = this.parentIndex(this.cursor, row.depth);
			if (p >= 0) this.setCursor(p);
			return;
		}
		if (ch === "u") {
			this.state.summary = !this.state.summary;
			return this.rebuild();
		}
		if (ch === "c") {
			this.state.expanded.clear();
			let top = this.cursor;
			while (top > 0 && this.rows[top]!.depth > 0) top--;
			this.state.cursorId = this.rows[top]?.id;
			return this.rebuild();
		}
		if (ch === "s") {
			this.state.sort = !this.state.sort;
			this.state.cursorId = this.nodeAt(this.cursor)?.id;
			if (this.state.query) this.matches = this.findMatches(this.state.query);
			this.matchIdx = -1;
			this.showFlash(this.state.sort ? "Sorted by size" : "Original order");
			return this.rebuild();
		}
		const node = this.nodeAt(this.cursor);
		if (!node) return;
		if (ch === "y" || ch === "Y") {
			const value = node.kind === "group" ? node.items?.map(([, v]) => v) : node.value;
			const text =
				ch === "Y" ? this.rowPath(node) : typeof value === "string" ? value : JSON.stringify(value, null, 2);
			copyToClipboard(text).then(
				() => this.showFlash(`Copied ${ch === "Y" ? "path" : this.rowPath(node)}`),
				(e) => this.showFlash(`Copy failed: ${e?.message ?? e}`, "error"),
			);
		}
	}

	private rowPath(r: Row): string {
		if (r.kind !== "group") return fmtPath(r.path);
		return `${fmtPath(r.path)}${this.groupLabel(r)}`;
	}

	private groupLabel(r: Row): string {
		const [s, e] = r.range!;
		if (this.state.sort) return e - s === 1 ? `[#${e} by size]` : `[#${s + 1}…#${e} by size]`;
		const first = r.items?.[0]?.[0];
		const last = r.items?.[r.items.length - 1]?.[0];
		return first === last ? `[${first}]` : `[${first}…${last}]`;
	}

	private stats(chars: number, width: number): string {
		const th = this.theme;
		const pct = (chars / this.rootChars) * 100;
		const tok = `~${fmtTok(chars * this.tokPerChar)}`.padStart(7);
		const pctStr = (pct >= 1 ? `${Math.round(pct)}%` : "").padStart(4);
		if (width < 40) return th.fg("dim", tok);
		const cells = 6;
		const filled = (pct / 100) * cells;
		const full = Math.floor(filled);
		const partials = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
		const part = partials[Math.round((filled - full) * 7)] ?? "";
		const bar = ("█".repeat(full) + part).padEnd(cells, " ").slice(0, cells);
		const barColor = pct >= 25 ? "warning" : "accent";
		return `${th.fg("dim", tok)} ${th.fg("muted", pctStr)} ${th.fg(barColor, bar)}`;
	}

	/** Style `raw` with `style`, highlighting occurrences of the active search query. */
	private hl(raw: string, style: (s: string) => string): string {
		const q = this.activeQuery;
		if (!q) return style(raw);
		const ci = q === q.toLowerCase();
		const hay = ci ? raw.toLowerCase() : raw;
		const nq = ci ? q.toLowerCase() : q;
		let out = "";
		let i = 0;
		for (let j = hay.indexOf(nq); j >= 0; j = hay.indexOf(nq, i)) {
			out += style(raw.slice(i, j)) + this.theme.bg("searchMatchBg", style(raw.slice(j, j + nq.length)));
			i = j + nq.length;
		}
		return out + style(raw.slice(i));
	}

	private renderRow(row: Row, inner: number, selected: boolean): string {
		const th = this.theme;
		const indent = "  ".repeat(row.depth);
		let line: string;
		if (row.kind === "text") {
			line = `${indent}${th.fg("borderMuted", "│")} ${this.hl(row.text ?? "", (s) => th.fg("text", s))}`;
			line = truncateToWidth(line, inner, "…", true);
		} else if (row.kind === "group") {
			const open = this.state.expanded.has(row.id);
			const chars = (row.items ?? []).reduce((n, [, v]) => n + this.size(v), 0);
			const left = `${indent}${th.fg("accent", open ? "▾" : "▸")} ${th.fg("muted", this.groupLabel(row))} ${th.fg("dim", `(${row.items?.length ?? 0})`)}`;
			const stats = this.stats(chars, inner);
			line = `${truncateToWidth(left, Math.max(4, inner - visibleWidth(stats) - 1), "…", true)} ${stats}`;
		} else {
			const v = row.value;
			const open = this.state.expanded.has(row.id);
			const marker = row.expandable ? th.fg("accent", open ? "▾" : "▸") : th.fg("dim", "·");
			const keyStr =
				typeof row.key === "number"
					? th.fg("muted", `[${row.key}]`)
					: row.depth === 0
						? this.hl(String(row.key), (s) => th.bold(th.fg("accent", s)))
						: this.hl(String(row.key), (s) => th.fg("text", s));
			let summary = "";
			let showStats = true;
			if (Array.isArray(v)) summary = th.fg("dim", `[${v.length}]`);
			else if (isObj(v)) {
				const label = describe(v);
				const count = th.fg("dim", `{${Object.keys(v).length}}`);
				summary = label ? `${th.fg("syntaxType", label)} ${count}` : count;
			} else if (typeof v === "string") {
				const blob = blobLabel(v);
				summary = blob
					? th.fg("warning", blob)
					: this.hl(row.expandable ? preview(v) : JSON.stringify(v), (s) => th.fg("syntaxString", s));
				showStats = row.expandable || !!blob;
			} else {
				summary = this.hl(String(v), (s) => th.fg("syntaxNumber", s));
				showStats = false;
			}
			const sep = row.expandable || isObj(v) ? " " : th.fg("dim", ": ");
			const left = `${indent}${marker} ${keyStr}${sep}${summary}`;
			if (showStats) {
				const stats = this.stats(this.size(v), inner);
				const leftW = Math.max(4, inner - visibleWidth(stats) - 1);
				line = `${truncateToWidth(left, leftW, "…", true)} ${stats}`;
			} else {
				line = truncateToWidth(left, inner, "…", true);
			}
		}
		if (!selected) return line;
		// Re-apply the selection background after any inner background resets (search highlights).
		return th.bg("selectedBg", line.replaceAll("\x1b[49m", th.getBgAnsi("selectedBg")));
	}

	render(width: number): string[] {
		const th = this.theme;
		this.lastWidth = width;
		if (width !== this.builtWidth) this.build(width);
		const inner = Math.max(10, width - 2);
		const b = (s: string) => th.fg("border", s);
		const row = (content: string) => b("│") + truncateToWidth(content, inner, "…", true) + b("│");
		const lines: string[] = [];

		// Header
		const used = this.usedTokens;
		const usedStr = this.contextWindow
			? `${fmtTok(used)}/${fmtTok(this.contextWindow)} (${Math.round((used / this.contextWindow) * 100)}%)`
			: `${this.measured ? "" : "~"}${fmtTok(used)} tok`;
		const title = ` Context · ${this.modelName ?? "?"} · ${usedStr}${this.state.sort ? " · by size" : ""} `;
		const t = truncateToWidth(title, inner - 2, "…");
		lines.push(b(`╭─`) + th.fg("accent", t) + b(`${"─".repeat(Math.max(0, inner - 1 - visibleWidth(t)))}╮`));

		// Usage summary (toggle with `u`)
		if (this.state.summary) {
			const summary = this.renderSummary(inner);
			for (const l of summary) lines.push(row(l));
			lines.push(b(`├${"─".repeat(inner)}┤`));
			this.summaryHeight = summary.length + 1;
		} else {
			this.summaryHeight = 0;
		}

		// Breadcrumb
		const cur = this.nodeAt(this.cursor);
		let crumbs = (cur?.path ?? []).map((p) => (typeof p === "number" ? `[${p}]` : p));
		if (cur?.kind === "group") crumbs.push(this.groupLabel(cur));
		let crumb = crumbs.join(" › ");
		while (crumbs.length > 1 && visibleWidth(crumb) > inner - 2) {
			crumbs = crumbs.slice(1);
			crumb = `… › ${crumbs.join(" › ")}`;
		}
		lines.push(row(` ${th.fg("muted", crumb || "(root)")}`));
		lines.push(b(`├${"─".repeat(inner)}┤`));

		// Body
		const h = this.bodyHeight;
		let scroll = this.state.scroll;
		if (this.centerNext) {
			scroll = this.cursor - Math.floor(h / 2);
			this.centerNext = false;
		}
		if (this.cursor < scroll) scroll = this.cursor;
		if (this.cursor >= scroll + h) scroll = this.cursor - h + 1;
		scroll = Math.max(0, Math.min(scroll, Math.max(0, this.rows.length - h)));
		this.state.scroll = scroll;
		for (let i = 0; i < h; i++) {
			const r = this.rows[scroll + i];
			lines.push(r ? b("│") + this.renderRow(r, inner, scroll + i === this.cursor) + b("│") : row(""));
		}

		// Footer: search prompt, flash message, or key hints
		lines.push(b(`├${"─".repeat(inner)}┤`));
		const pos = th.fg("dim", ` ${this.cursor + 1}/${this.rows.length}`);
		let footer: string;
		if (this.input) {
			const count = !this.input.text
				? ""
				: this.matches.length
					? th.fg("dim", `  [${this.matchIdx + 1}/${this.matches.length}]`)
					: th.fg("error", "  no match");
			footer = ` ${this.input.dir === 1 ? "/" : "?"}${this.input.text}${th.inverse(" ")}${count}`;
		} else if (this.flash) {
			footer = th.fg(this.flash.color, ` ${this.flash.msg}`);
		} else {
			footer = th.fg(
				"dim",
				" ↑↓ move  ⇧↑↓ sibling  →← open/close  / search  n/N next  s sort  u usage  c collapse  y copy  q quit",
			);
		}
		lines.push(row(`${truncateToWidth(footer, Math.max(0, inner - visibleWidth(pos)), "…", true)}${pos}`));
		const fileLabel = truncateToWidth(` ${this.file.split("/").pop()} `, inner - 2, "…");
		lines.push(b(`╰─`) + th.fg("dim", fileLabel) + b(`${"─".repeat(Math.max(0, inner - 1 - visibleWidth(fileLabel)))}╯`));
		return lines;
	}

	invalidate(): void {
		this.builtWidth = -1;
	}

	dispose(): void {
		clearTimeout(this.flashTimer);
	}
}

// ── Extension ──

export default function (pi: ExtensionAPI) {
	let enabled = true;
	let counter = 0;
	let pendingHeaders: Record<string, unknown> | undefined;
	let last: { file: string; dump: Dump } | undefined;
	let explorer: ExplorerState | undefined;

	/** Tokens pi keeps free before auto-compacting (model override → setting → default), 0 if disabled. */
	const compactReserve = (model?: { provider: string; id: string }) => {
		try {
			const c = (pi.getSettings() as { compaction?: Record<string, any> }).compaction ?? {};
			if (c.enabled === false) return 0;
			const override = model ? c.modelOverrides?.[`${model.provider}/${model.id}`]?.reserveTokens : undefined;
			return override ?? c.reserveTokens ?? 16384;
		} catch {
			return undefined;
		}
	};

	/** One dump per session, stored beside the session's .jsonl (pi only lists *.jsonl, so it is ignored). */
	const dumpFile = (ctx: ExtensionContext) => {
		const sm = ctx.sessionManager;
		const sessionFile = sm.getSessionFile();
		return sessionFile
			? sessionFile.replace(/\.jsonl$/, "") + ".context.json"
			: join(getAgentDir(), "context-dumps", `${sm.getSessionId()}.context.json`);
	};

	const save = () => {
		if (!last) return;
		mkdirSync(dirname(last.file), { recursive: true });
		writeFileSync(last.file, JSON.stringify(last.dump, null, 2), "utf8");
	};

	/** On resume, pick up the dump left by a previous run of this session. */
	const loadExisting = (ctx: ExtensionContext) => {
		if (last) return;
		const file = dumpFile(ctx);
		if (!existsSync(file)) return;
		try {
			last = { file, dump: JSON.parse(readFileSync(file, "utf8")) as Dump };
		} catch {
			// ignore unreadable/partial dumps
		}
	};

	pi.on("before_provider_headers", (event) => {
		if (!enabled) return;
		pendingHeaders = Object.fromEntries(
			Object.entries(event.headers).map(([k, v]) => [k, SECRET.test(k) && v ? "[redacted]" : v]),
		);
	});

	pi.on("before_provider_request", (event, ctx) => {
		if (!enabled) return; // return undefined => payload untouched
		counter++;
		last = {
			file: dumpFile(ctx),
			dump: {
				meta: {
					time: new Date().toISOString(),
					sessionId: ctx.sessionManager.getSessionId(),
					request: counter,
					provider: ctx.model?.provider,
					model: ctx.model?.id,
					api: ctx.model?.api,
					contextWindow: ctx.model?.contextWindow,
					compactReserve: compactReserve(ctx.model),
				},
				headers: pendingHeaders,
				payload: structuredClone(event.payload),
			},
		};
		pendingHeaders = undefined;
		save();
		ctx.ui.setStatus("context-dump", `ctx #${counter}`);
	});

	// The assistant message carries the provider's real token usage for the request we just dumped.
	pi.on("message_end", (event) => {
		const msg = event.message as { role?: string; usage?: unknown };
		if (!enabled || !last || msg.role !== "assistant" || !isObj(msg.usage) || last.dump.usage) return;
		const { cost: _cost, ...usage } = msg.usage as Record<string, unknown>;
		last.dump.usage = usage;
		save();
	});

	pi.on("after_provider_response", (event) => {
		if (!enabled || !last) return;
		last.dump.response = { status: event.status, headers: event.headers };
		save();
	});

	const explore = async (ctx: ExtensionCommandContext, file: string, dump: Dump) => {
		// Keep expansion/cursor/search across reopen; reset when a new dump arrives.
		if (!explorer || explorer.dump !== dump) {
			explorer = {
				dump,
				expanded: new Set(["payload"]),
				scroll: 0,
				sort: false,
				summary: true,
				query: explorer?.query,
				searchDir: explorer?.searchDir,
			};
		}
		const state = explorer;
		let tui: TUI | undefined;
		await ctx.ui.custom<void>(
			(t, theme, _kb, done) => {
				tui = t;
				return new ContextExplorer(t, theme, state, file, done);
			},
			{
				overlay: true,
				overlayOptions: () => {
					const cols = tui?.terminal.columns ?? process.stdout.columns ?? 120;
					return {
						anchor: "right-center",
						width: cols < 110 ? "100%" : "50%",
						minWidth: 50,
						maxHeight: "100%",
					};
				},
			},
		);
	};

	pi.registerCommand("context", {
		description: "Explore the JSON payload last sent to the LLM (/context [raw|path|on|off])",
		handler: async (args, ctx) => {
			const arg = args.trim().toLowerCase();
			if (arg === "on" || arg === "off") {
				enabled = arg === "on";
				ctx.ui.setStatus("context-dump", enabled ? "ctx dump" : undefined);
				ctx.ui.notify(`Context capture ${enabled ? "enabled" : "disabled"}`, "info");
				return;
			}
			loadExisting(ctx);
			if (!last) {
				ctx.ui.notify("No request captured yet — send a message first.", "warning");
				return;
			}
			if (arg === "path") {
				ctx.ui.notify(last.file, "info");
				return;
			}
			if (arg === "raw" || ctx.mode !== "tui") {
				await ctx.ui.editor(`LLM request → ${last.file}`, JSON.stringify(last.dump, null, 2));
				return;
			}
			await explore(ctx, last.file, last.dump);
		},
	});
}
