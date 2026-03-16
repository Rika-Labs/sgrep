<div align="center">

  <h1>sgrep</h1>
  <p><em>Fast local semantic code search for private repositories</em></p>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>

</div>

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
sgrep index
sgrep search "where do we handle authentication?"
```

## What sgrep does

- builds a local semantic index for your repository
- searches code with natural-language queries
- keeps the index fresh with `sgrep watch`
- works well for private and offline-friendly workflows once the model cache is warm

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/quickstart.md) | Get searching in a few minutes |
| [Deployment](docs/deployment.md) | Running `sgrep` locally |
| [Configuration](docs/configuration.md) | Flags, env vars, and config files |
| [Agent Integrations](docs/agents.md) | Claude Code, Factory, OpenCode |
| [Offline Mode](docs/offline.md) | Airgapped and proxy environments |
| [Architecture](docs/architecture.md) | How indexing and search work |
| [Troubleshooting](docs/troubleshooting.md) | Common local issues |

## Integrations

- **[Claude Code Plugin](plugins/sgrep/README.md)** — automatic index management and search skill
- **[OpenCode Plugin](plugins/opencode/README.md)** — MCP tool integration

## License

Apache License, Version 2.0. See [LICENSE](LICENSE).
