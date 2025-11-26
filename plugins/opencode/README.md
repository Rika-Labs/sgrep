# Sgrep OpenCode Plugin

What it does
- `sgrepSearch`: semantic code search for the current project (uses `sgrep search --json`, optional `--limit`). Returns ranked snippets (path, lines, score, snippet).

Install
```bash
cd plugins/opencode
bun install
```

Use in OpenCode
Normal: if published/linked as `sgrep-opencode`, add by name:
```json
{
  "plugins": ["sgrep-opencode"]
}
```

Dev (local checkout): point to the full path, e.g.
```json
{
  "plugins": ["file:///Users/jenslys/Code/sgrep/plugins/opencode"]
}
```

Develop
```bash
cd plugins/opencode
bun run index.ts   # type-check / quick load test
```
