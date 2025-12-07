<div align="center">

  <h1>sgrep</h1>
  <p><em>Fast, private, local semantic code search.</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>

</div>

Fast, private, local semantic code search for developers and coding agents.

- **Local-first**: ONNX embeddings run on your machine; `--offline` blocks network calls
- **Hybrid ranking**: tree-sitter chunks, dense vectors, BM25F keyword scoring, cross-encoder reranking
- **Cloud-optional**: Offload to [Modal.dev](https://modal.com) GPUs or store indexes in [Turbopuffer](https://turbopuffer.com)
- **Agent-ready**: JSON output and background watcher keep results fresh

## Three commands

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
sgrep index
sgrep watch
sgrep search "where do we handle authentication?"
```

Run `sgrep index` from the repo you want searched; it writes the index to your sgrep data dir.

Prefer source builds? Run `cargo install --path .` from the repo root.

## What happens on first run

- Downloads `jina-embeddings-v2-base-code` once to the fastembed cache (honors `HTTP(S)_PROXY`).
- Indexes the repo you run it in into `SGREP_HOME` (default: OS data dir such as `~/.local/share/sgrep` or `~/.sgrep`).
- `--offline` or `SGREP_OFFLINE=1` forbids network access and fails fast if the model is missing.

## Platforms

- macOS and Linux (arm64 and x86_64) via the install script.
- Windows: use WSL or a Linux container.

## Integrations

| Integration | Description |
|-------------|-------------|
| [Claude Code Plugin](plugins/sgrep/README.md) | Automatic index management and search skill for Claude Code |
| [OpenCode Plugin](plugins/opencode/README.md) | MCP tool for OpenCode |
| [Modal.dev](https://modal.com) | GPU-accelerated embeddings with Qwen3-Embedding-8B (outputs truncated to 384 dims for local compatibility) |
| [Turbopuffer](https://turbopuffer.com) | Serverless vector storage for remote indexes (Pinecone also supported via config) |

### Cloud offload (optional)

Run embeddings on Modal.dev GPUs instead of locally (authenticate via `modal token new` or set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`):

```bash
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."
sgrep search --offload "where do we handle auth?"
```

Auto-deploys a Modal service with Qwen3-Embedding-8B (384-dim outputs) and Qwen3-Reranker-8B. See [docs/configuration.md](docs/configuration.md) for GPU tier options.

## Learn more

- [Quickstart](docs/quickstart.md)
- [Flags and config](docs/configuration.md)
- [Offline and airgapped installs](docs/offline.md)
- [Agent integrations](docs/agents.md)

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).
