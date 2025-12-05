# Agent integrations

sgrep is built for programmatic use: every command supports JSON output (`--json`), and `sgrep watch` keeps indexes hot for repeated queries.

## Claude Code plugin

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

The plugin manages `sgrep watch` and surfaces search results to the agent. Details: `plugins/sgrep/README.md`.

## Factory skill

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-skill.sh | sh
```

Installs to `~/.factory/skills/sgrep/`. Restart Factory after install. Details: `.factory/skills/sgrep/SKILL.md`.

## OpenCode plugin

Add to your OpenCode configuration:

```json
{
  "plugins": ["sgrep-opencode"]
}
```

Details: `plugins/opencode/README.md`.

## Roll your own

- Ensure `sgrep` is on `PATH`.
- Start a watcher for the repo you care about: `sgrep watch`.
- Query with JSON output: `sgrep search --json "find the auth middleware"`.
- Add `--context` when the full chunk text is needed.
