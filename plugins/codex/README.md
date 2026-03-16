# sgrep Codex CLI Plugin

Integrates [sgrep](https://github.com/rika-labs/sgrep) semantic code search with [Codex CLI](https://github.com/openai/codex).

## Install

```bash
./scripts/install-agents.sh codex
```

This installs:
- Skill file at `~/.agents/skills/sgrep/SKILL.md`
- Session-start hook at `~/.codex/hooks/session-start.sh`

## Watch Lifecycle

Codex has `SessionStart` hooks (experimental) but **no `SessionEnd` event**. The session-start hook runs `sgrep index` but cannot start `sgrep watch` without a cleanup path.

### Automatic watch with wrapper script

For full watch lifecycle, use the wrapper script:

```bash
./plugins/codex/sgrep-watch.sh [codex args...]
```

This script:

1. Runs `sgrep index` to build/refresh the index
2. Spawns `sgrep watch` in the background
3. Starts `codex` (passing through all arguments)
4. Kills the `sgrep watch` process on exit via `trap EXIT`

### Manual watch

```bash
# Terminal 1
sgrep watch

# Terminal 2
codex

# When done, Ctrl-C the watch in Terminal 1
```
