# pi-context-dump

A [pi](https://pi.dev) extension that saves the exact JSON request sent to the LLM provider, so you can see what's actually in your context.

The latest request for each session is written next to the session file as `<session>.context.json`. It includes the payload (system prompt, tools, messages, params), headers with auth redacted, token usage, and the response status.

## Install

```bash
pi install git:github.com/liama28/pi-context-dump
```

Or copy `extensions/context-dump.ts` into `~/.pi/agent/extensions/` or `.pi/extensions/`.

## Usage

| Command | Description |
|---|---|
| `/context` | Browse the last request in a collapsible tree (`↑↓`/`jk` move, `←→`/`hl` collapse/expand, `/` search) |
| `/context raw` | View the raw JSON (read-only) |
| `/context path` | Show where the last dump was written |
| `/context off` \| `on` | Turn capture off or on |
