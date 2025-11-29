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

### OpenCode Plugin

Native OpenCode plugin for semantic code search. Add to OpenCode configuration:

```json
{
  "plugins": ["sgrep-opencode"]
}
```

See [plugins/opencode/README.md](plugins/opencode/README.md) for setup and usage.

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

| Flag                  | Description                          | Default |
| --------------------- | ------------------------------------ | ------- |
| `-n, --limit <n>`     | Maximum number of results to return. | `10`    |
| `-c, --context`       | Return full chunk content.           | `false` |
| `-p, --path <dir>`    | Target repository root.              | `.`     |
| `--glob <pattern>`    | Glob pattern for file inclusion.     | —       |
| `--filters key=value` | Metadata filters (language, path).   | —       |
| `--json`              | Output results in JSON format.       | `false` |

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
    "total_chunks": 456,
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
      "snippet": "fn retry(...) { … }",
    },
  ],
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
- `SGREP_INIT_TIMEOUT_SECS`: Model initialization timeout (default: 120).
- `SGREP_MAX_THREADS`: Maximum threads for parallel operations.
- `SGREP_CPU_PRESET`: CPU usage preset (`auto`, `low`, `medium`, `high`, `background`).
- `HTTP_PROXY` / `HTTPS_PROXY`: Proxy for model downloads.
- `RUST_LOG`: Tracing level (e.g., `sgrep=debug`).
- `RAYON_NUM_THREADS`: Thread pool limit (overrides `SGREP_MAX_THREADS`).

### Thread Control

sgrep provides fine-grained control over CPU usage to prevent system slowdown during indexing:

```bash
sgrep --threads 4 index              # Limit to 4 threads
sgrep --cpu-preset low index         # Use 25% of CPU cores
sgrep --cpu-preset background watch  # Low-impact background mode
```

**Available presets:**

| Preset       | CPU Usage | Use Case                        |
| ------------ | --------- | ------------------------------- |
| `auto`       | 75%       | Default, balanced performance   |
| `low`        | 25%       | Laptop-friendly, battery saving |
| `medium`     | 50%       | Multi-tasking                   |
| `high`       | 100%      | Maximum performance             |
| `background` | 25%       | Watch/daemon mode               |

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
- **Model Download Failed:** See [Offline Installation](#offline-installation) below.
- **Initialization Timeout:** Increase with `SGREP_INIT_TIMEOUT_SECS=600`.

## Offline Installation

If HuggingFace is blocked in your region (e.g., China), use one of these methods:

### Using HTTP Proxy

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
sgrep search "your query"
```

### Manual Model Download

1. **Find model directory:**

   ```bash
   sgrep config --show-model-dir
   ```

2. **Download files** from https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1:
   - `onnx/model_quantized.onnx`
   - `tokenizer.json`
   - `config.json`
   - `special_tokens_map.json`
   - `tokenizer_config.json`

3. **Place files** in the model directory and verify:
   ```bash
   sgrep config --verify-model
   ```

### China Mirror Script

```bash
MODEL_DIR=$(sgrep config --show-model-dir)
mkdir -p "$MODEL_DIR"
BASE="https://hf-mirror.com/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main"

curl -L "$BASE/onnx/model_quantized.onnx" -o "$MODEL_DIR/model_quantized.onnx"
curl -L "$BASE/tokenizer.json" -o "$MODEL_DIR/tokenizer.json"
curl -L "$BASE/config.json" -o "$MODEL_DIR/config.json"
curl -L "$BASE/special_tokens_map.json" -o "$MODEL_DIR/special_tokens_map.json"
curl -L "$BASE/tokenizer_config.json" -o "$MODEL_DIR/tokenizer_config.json"

sgrep config --verify-model
```

## Attribution

Architectural concepts derived from research into semantic code retrieval and local-first search engines like osgrep, mgrep, and ripgrep.

## License

Apache License, Version 2.0.
See [LICENSE](LICENSE).
