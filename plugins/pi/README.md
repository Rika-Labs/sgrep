# sgrep Pi Plugin

Integrates [sgrep](https://github.com/rika-labs/sgrep) semantic code search with [Pi](https://github.com/badlogic/pi-mono).

## Install

In Pi, run:

```
/install https://github.com/rika-labs/sgrep
```

Or add to `~/.pi/agent/settings.json`:

```json
{
  "packages": [
    "https://github.com/rika-labs/sgrep"
  ]
}
```

The Pi manifest is at the repo root (`package.json`), pointing to:
- **Skill** at `plugins/pi/skills/sgrep/SKILL.md`
- **Extension** at `plugins/pi/extensions/sgrep-watch/index.ts` — auto-indexes on startup, watches for changes, cleans up on shutdown
