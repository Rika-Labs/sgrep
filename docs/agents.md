# Agent integrations

`sgrep` is built for local agent workflows: build an index once, search locally, and keep it warm with `sgrep watch`.

## Claude Code plugin

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

The plugin manages `sgrep watch` and surfaces local search results to the agent. Details: [plugins/sgrep/README.md](../plugins/sgrep/README.md).

## Factory skill

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-skill.sh | sh
```

Installs to `~/.factory/skills/sgrep/`. Restart Factory after install. Details: [.factory/skills/sgrep/SKILL.md](../.factory/skills/sgrep/SKILL.md).

## OpenCode plugin

Add to your OpenCode configuration:

```json
{
  "plugins": ["sgrep-opencode"]
}
```

Details: [plugins/opencode/README.md](../plugins/opencode/README.md).

## Roll your own

- ensure `sgrep` is on `PATH`
- start a watcher for the repo you care about: `sgrep watch`
- query with JSON output: `sgrep search --json "find the auth middleware"`
- add `--context` when the full chunk text is needed
- use `--offline` when you want to forbid network fetches for model downloads
