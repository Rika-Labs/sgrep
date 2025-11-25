<div align="center">

  <h1>sgrep</h1>

  <p><em>High-performance local-first semantic code search engine.</em></p>

  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a><br>

</div>

Natural-language search mechanism operating on local codebases. Designed for integration with autonomous coding agents and developer workflows.

- **Vector + Graph Architecture:** Synthesizes dense vector embeddings with structural code graph analysis for high-fidelity retrieval.
- **Local & Private:** All processing occurs locally using optimized quantization and ONNX runtime execution.
- **Auto-Isolated:** Repository indexes are strictly compartmentalized by cryptographic hash.
- **Adaptive Resource Management:** Dynamic concurrency scaling based on real-time system telemetry.
- **Agent-Ready:** Provides stable CLI interfaces and structured JSON output schemas for machine consumption.

## Quick Start

1. **Install**

   ```bash
   curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
   ```

   Environment variables `INSTALL_DIR` and `SGREP_REPO` may be defined prior to execution to customize installation paths.

   **From source:**

   ```bash
   cargo install --path .
   ```

2. **Search**

   ```bash
   cd my-repo
   sgrep search "where do we handle authentication?"
   ```

   Initialization occurs automatically upon the first execution within a repository. Subsequent queries utilize the persisted index.

## Coding Agent Integration

### Claude Code Plugin

sgrep integrates with Claude Code via the official plugin interface:

```bash
/plugin marketplace add rika-labs/sgrep
/plugin install sgrep
```

The plugin manages the `sgrep watch` process lifecycle and exposes search capabilities to the agent environment. See [plugins/sgrep/README.md](plugins/sgrep/README.md) for technical specifications.

### Factory Droids Integration

Native support is provided for Factory Droids:

1. **Enable Custom Droids**: via `/settings` in Factory.
2. **Deploy Droid**:
   ```bash
   # Project-scope
   cp -r .factory/droids/ your-project/.factory/droids/

   # User-scope
   cp .factory/droids/sgrep.md ~/.factory/droids/
   ```

### Generic Agent Integration

For custom agent implementations:

1. Ensure `sgrep` is available in the system `PATH`.
2. Initialize the watcher process: `sgrep watch`.
3. Execute queries with structured output: `sgrep search --json "query"`.

## Commands

### `sgrep search`

Executes a semantic search query against the current repository index.

```bash
sgrep search "database connection pooling strategy"
```

**Options:**

| Flag | Description | Default |
| --- | --- | --- |
| `-n, --limit <n>` | Maximum number of results to return. | `10` |
| `-c, --context` | Return full chunk content. | `false` |
| `-p, --path <dir>` | Target repository root. | `.` |
| `--glob <pattern>` | Glob pattern for file inclusion. | — |
| `--filters key=value` | Metadata filters (language, path). | — |
| `--json` | Output results in JSON format. | `false` |

**JSON Schema Specification:**

```jsonc
{
  "query": "retry logic",
  "limit": 10,
  "duration_ms": 42,
  "index": {
    "repo_path": "/path/to/repo",
    "repo_hash": "<blake3>",
    "vector_dim": 384,
    "indexed_at": "2025-11-23T05:00:00Z",
    "total_files": 123,
    "total_chunks": 456
  },
  "results": [
    {
      "path": "src/lib.rs",
      "start_line": 10,
      "end_line": 42,
      "language": "rust",
      "score": 0.92,
      "semantic_score": 0.88,
      "keyword_score": 0.55,
      "snippet": "fn retry(...) { … }"
    }
  ]
}
```

### `sgrep index`

Forces a manual re-indexing operation.

- **Filtering:** Adheres to `.gitignore` and internal heuristics for binary/minified file detection.
- **Throttling:** Automatically adjusts thread count based on system load.

```bash
sgrep index                 # Index current directory
sgrep index ../other-repo   # Index specific path
sgrep index --force         # Force full rebuild
```

### `sgrep watch`

Initiates a filesystem watcher to perform incremental updates on the index.

```bash
sgrep watch                         # Watch current directory
sgrep watch ../service --debounce-ms 200
```

### `sgrep config`

Manages configuration settings.

```bash
sgrep config          # Display current configuration
sgrep config --init   # Initialize default configuration
```

## Architecture

sgrep utilizes a proprietary engine architecture optimized for code retrieval:

1.  **Vector + Graph Fusion:** The search algorithm integrates dense vector embeddings with a code property graph. This allows the engine to leverage both semantic meaning and structural relationships (e.g., function calls, class hierarchies) during retrieval.
2.  **Local Embedding Pipeline:** Utilizes quantized local models (e.g., mxbai-embed-xsmall-v1) running on ONNX Runtime.
3.  **Structural Chunking:** Code is parsed via `tree-sitter` to respect syntactic boundaries, ensuring embeddings correspond to logical code units.
4.  **Deduplication:** Content-addressable storage eliminates redundant processing of identical code blocks.

## Configuration

Configuration is managed via `~/.sgrep/config.toml`.

**Default Configuration:**

```toml
[embedding]
provider = "local"
```

### Environment Variables

- `SGREP_HOME`: Data directory override (default: `~/.sgrep/`).
- `SGREP_CONFIG`: Config file path override.
- `SGREP_DEVICE`: Accelerator selection (`cpu`, `cuda`, `coreml`).
- `SGREP_BATCH_SIZE`: Inference batch size.
- `SGREP_EMBEDDER_POOL_SIZE`: Model instance count.
- `RUST_LOG`: Tracing level (e.g., `sgrep=debug`).
- `RAYON_NUM_THREADS`: Thread pool limit.

## Development

```bash
cargo build --release
cargo test
cargo fmt
cargo clippy
```

## Troubleshooting

- **Stale Index:** Execute `sgrep index` to synchronize.
- **Cache Reset:** Remove `~/.sgrep/indexes/<repo>` to force a clean state.
- **Offline Operation:** Use `sgrep --offline index` for air-gapped environments. Ensure model artifacts are cached at `~/.sgrep/cache/fastembed`.

## Attribution

Architectural concepts derived from research into semantic code retrieval and local-first search engines like osgrep, mgrep, and ripgrep. 

## License

Apache License, Version 2.0.
See [LICENSE](LICENSE).
