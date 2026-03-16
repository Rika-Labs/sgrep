# Agent integrations

`sgrep` integrates with four coding agents: **Claude Code**, **Codex CLI**, **Pi**, and **OpenCode**. Each integration provides the sgrep skill and watch lifecycle management.

## Quick install

No repo clone needed — install via curl:

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s claude
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s codex
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s pi
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s all
```

Or from a local clone:

```bash
./scripts/install-agents.sh claude
```

## What gets installed

| Agent | Skill | Watch Start | Watch Stop |
|---|---|---|---|
| Claude Code | `~/.claude/skills/sgrep/SKILL.md` | `SessionStart` hook | `Stop` hook |
| Codex CLI | `~/.agents/skills/sgrep/SKILL.md` | `~/.codex/hooks/session-start.sh` | Wrapper script (`trap EXIT`) |
| Pi | Pi package (skill + extension) | Extension `default()` | Extension `shutdown()` |
| OpenCode | TypeScript plugin | Plugin load | `SIGINT`/`SIGTERM` handlers |

## Claude Code plugin

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

The plugin manages `sgrep watch` and surfaces local search results to the agent. Details: [plugins/sgrep/README.md](../plugins/sgrep/README.md).

## Codex CLI plugin

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s codex
```

Codex has no session-end event, so watch cleanup requires a wrapper script:

```bash
./plugins/codex/sgrep-watch.sh [codex args...]
```

Details: [plugins/codex/README.md](../plugins/codex/README.md).

## Pi plugin

Install as a Pi package (in Pi, run):

```
/install https://github.com/rika-labs/sgrep
```

Or via the installer (adds the package to `~/.pi/agent/settings.json`):

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s pi
```

The package provides a skill and an extension that auto-indexes on startup and watches for changes. Details: [plugins/pi/README.md](../plugins/pi/README.md).

## OpenCode plugin

Add to your OpenCode configuration:

```json
{
  "plugins": ["sgrep-opencode"]
}
```

Details: [plugins/opencode/README.md](../plugins/opencode/README.md).

## Factory skill

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-skill.sh | sh
```

Installs to `~/.factory/skills/sgrep/`. Restart Factory after install. Details: [.factory/skills/sgrep/SKILL.md](../.factory/skills/sgrep/SKILL.md).

## Roll your own

- ensure `sgrep` is on `PATH`
- start a watcher for the repo you care about: `sgrep watch`
- query with JSON output: `sgrep search --json "find the auth middleware"`
- add `--context` when the full chunk text is needed
- use `--offline` when you want to forbid network fetches for model downloads
