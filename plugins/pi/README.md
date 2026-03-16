# sgrep Pi Plugin

Integrates [sgrep](https://github.com/rika-labs/sgrep) semantic code search with [Pi](https://github.com/badlogic/pi-mono).

## Install

In Pi, run:

```
/install https://github.com/rika-labs/sgrep --subdir plugins/pi
```

Or add to `~/.pi/agent/settings.json`:

```json
{
  "packages": [
    "https://github.com/rika-labs/sgrep?subdir=plugins/pi"
  ]
}
```

The plugin provides:
- **Skill** at `skills/sgrep/SKILL.md` — teaches Pi how to use sgrep commands
- **Extension** at `extensions/sgrep-watch/index.ts` — auto-indexes on startup, watches for changes, cleans up on shutdown
