<div align="center">

  <h1>sgrep</h1>
  <p><em>Semantic code search that scales from private local processing to cloud-scale GPU acceleration</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>

</div>

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
sgrep index
sgrep search "where do we handle authentication?"
```

## Deploy Your Way

sgrep adapts to your privacy, performance, and scale needs:

```
┌─────────────────┬───────────────────┬───────────────────────┐
│                 │   Store Locally   │   Store Remotely      │
│                 │   (default)       │   (Pinecone/Turbo)    │
├─────────────────┼───────────────────┼───────────────────────┤
│ Process Locally │ ✓ Fully Private   │ ✓ Scale Storage       │
│ (default)       │   Zero Cloud      │   Private Embeddings  │
├─────────────────┼───────────────────┼───────────────────────┤
│ Offload to GPU  │ ✓ Fast Indexing   │ ✓ Full Cloud Scale    │
│ (Modal.dev)     │   Data Stays Local│   Maximum Speed       │
└─────────────────┴───────────────────┴───────────────────────┘
```

Start local and private. Scale up when you need to. [Learn more →](docs/deployment.md)

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/quickstart.md) | Get searching in 2 minutes |
| [Deployment Options](docs/deployment.md) | Local, remote, and hybrid setups |
| [Configuration](docs/configuration.md) | Flags, env vars, and config files |
| [Agent Integrations](docs/agents.md) | Claude Code, Factory, OpenCode |
| [Offline Mode](docs/offline.md) | Airgapped and proxy environments |

## Integrations

- **[Claude Code Plugin](plugins/sgrep/README.md)** — Automatic index management and search skill
- **[OpenCode Plugin](plugins/opencode/README.md)** — MCP tool integration
- **[Modal.dev](https://modal.com)** — GPU-accelerated embeddings (Qwen3-Embedding-8B)
- **[Turbopuffer](https://turbopuffer.com) / [Pinecone](https://pinecone.io)** — Serverless vector storage

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).
