<div align="center">

  <h1>sgrep</h1>
  <p><em>Fast, private, local semantic code search.</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>

</div>

Fast, private, local semantic code search for developers and coding agents. 

- Local-first: ONNX embeddings run on your machine; `--offline` blocks network calls.
- Hybrid ranking: tree-sitter chunks plus dense vectors, BM25F keyword scoring, and optional code-graph context.
- Agent-ready: JSON output and a background watcher keep results fresh.

## Three commands

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
sgrep search "where do we handle authentication?"
sgrep watch
```

Want source builds? `cargo install --path .`

## What happens on first run

- Downloads `jina-embeddings-v2-base-code` once to the fastembed cache (honors `HTTP(S)_PROXY`).
- Indexes each repo into `SGREP_HOME` (default: OS data dir such as `~/.local/share/sgrep` or `~/.sgrep`).
- `--offline` or `SGREP_OFFLINE=1` forbids network access and fails fast if the model is missing.

## Platforms

- macOS and Linux (arm64 and x86_64) via the install script.
- Windows: use WSL or a Linux container.

## Learn more

- Quickstart: `docs/quickstart.md`
- Flags and config: `docs/configuration.md`
- Offline and airgapped installs: `docs/offline.md`
- Agent integrations: `docs/agents.md`

## License

Apache License, Version 2.0. See `LICENSE`.
