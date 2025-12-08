# Agent integrations

sgrep is built for programmatic use: every command supports JSON output (`--json`), and `sgrep watch` keeps indexes hot for repeated queries.

## Claude Code plugin

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

The plugin manages `sgrep watch` and surfaces search results to the agent. Details: [plugins/sgrep/README.md](../plugins/sgrep/README.md).

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

## Cloud offload for agents

For cloud-based agents or when local GPU isn't available, use [Modal.dev](https://modal.com) for GPU-accelerated embeddings (authenticate via `modal token new` or set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`):

```bash
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."
sgrep search --offload --json "find authentication logic"
```

This auto-deploys a Modal service with:
- **Qwen3-Embedding-8B**: 8K context window, outputs truncated to 384 dimensions for local compatibility

See [configuration.md](configuration.md) for GPU tier options and full configuration.

## Roll your own

- Ensure `sgrep` is on `PATH`.
- Start a watcher for the repo you care about: `sgrep watch`.
- Query with JSON output: `sgrep search --json "find the auth middleware"`.
- Add `--context` when the full chunk text is needed.
- Add `--offload` to use Modal.dev GPUs instead of local embeddings.
