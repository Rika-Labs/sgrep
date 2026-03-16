# sgrep Pi Plugin

Integrates [sgrep](https://github.com/rika-labs/sgrep) semantic code search with [Pi](https://github.com/pi-ai/pi).

## Install

```bash
./scripts/install-agents.sh pi
```

This installs:
- Skill file at `~/.pi/agent/skills/sgrep/README.md` (no YAML frontmatter)
- Extension at `~/.pi/extensions/sgrep-watch.ts`

## Watch Lifecycle

Pi has full lifecycle support via extensions (`session_start` and `session_shutdown` events).

The extension:

1. On `session_start`: runs `sgrep index`, then spawns `sgrep watch` in detached mode
2. On `session_shutdown`: kills the watch process by stored PID

## Manual watch

```bash
# Terminal 1
sgrep watch

# Terminal 2
pi

# When done, Ctrl-C the watch in Terminal 1
```
