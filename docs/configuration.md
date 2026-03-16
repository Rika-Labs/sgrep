# Configuration

`sgrep` is now documented as a local-only semantic search tool.

## Global flags

- `--device <cpu|cuda|coreml>` (or `SGREP_DEVICE`)
- `--offline` (or `SGREP_OFFLINE=1`) to forbid downloads and fail fast if the model is missing
- `--threads <n>` (or `SGREP_MAX_THREADS`) to bound parallelism
- `--cpu-preset <auto|low|medium|high|background>` (or `SGREP_CPU_PRESET`)

## Commands

### search

- `--path` (default `.`)
- `--limit` (default `10`)
- `--context` to return chunk bodies
- `--glob <pattern>` (repeatable)
- `--filters key=value` (repeatable)
- `--json` for structured output
- `--debug` to surface scores and timings

### index

- `--force` for a full rebuild
- `--batch-size` (or `SGREP_BATCH_SIZE`) to override embedder batch size
- `--profile` to print per-phase timings
- `--stats` to print index statistics without rebuilding
- `--json` to emit stats as JSON (only with `--stats`)
- `--detach` to run indexing in the background (not compatible with `--stats`)
- `path` argument optional; defaults to the current directory

### watch

- `path` argument optional; defaults to the current directory
- `--debounce-ms` (default `500`)
- `--batch-size` (or `SGREP_BATCH_SIZE`)
- `--detach` to run watch in the background

### config

- `--init` to write a default config file
- `--show-model-dir` to print the embedding cache location
- `--verify-model` to assert required model files exist

## Configuration file

`SGREP_CONFIG` overrides the path; otherwise `sgrep` uses `SGREP_HOME/config.toml`, defaulting to `~/.sgrep/config.toml`.

```toml
[embedding]
model = "mxbai"
provider = "local"
```

## Environment variables

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_HOME` | `~/.sgrep` | Index and config directory |
| `SGREP_CONFIG` | `$SGREP_HOME/config.toml` | Config file path override |
| `FASTEMBED_CACHE_DIR` | `$SGREP_HOME/cache/fastembed` | Model weights cache |

### Runtime flags

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `coreml` |
| `SGREP_OFFLINE` | `0` | Block network calls (`1` or `true`) |
| `SGREP_BATCH_SIZE` | auto | Embedding batch size |
| `SGREP_INIT_TIMEOUT_SECS` | `120` | Model initialization timeout |

### Threading

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_MAX_THREADS` | all cores | Maximum total threads |
| `SGREP_CPU_PRESET` | `auto` | Preset: `auto`, `background`, `low`, `medium`, `high` |
| `RAYON_NUM_THREADS` | auto | Rayon parallel pool size |

### ONNX Runtime (advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `ORT_INTRA_OP_NUM_THREADS` | 4 | Threads within operators |
| `ORT_INTER_OP_NUM_THREADS` | auto | Threads between operators |
| `ORT_NUM_THREADS` | auto | Legacy thread setting |

### BLAS backends (advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | auto | OpenMP threads |
| `MKL_NUM_THREADS` | auto | Intel MKL threads |
| `OPENBLAS_NUM_THREADS` | auto | OpenBLAS threads |
| `VECLIB_MAXIMUM_THREADS` | auto | macOS Veclib threads |

### Network

| Variable | Description |
|----------|-------------|
| `HTTP_PROXY` | HTTP proxy for model downloads |
| `HTTPS_PROXY` | HTTPS proxy for model downloads |

### Debugging

| Variable | Example | Description |
|----------|---------|-------------|
| `RUST_LOG` | `sgrep=debug` | Log level |

## JSON output schema

Use `--json` for machine-readable output.

### Search results

```bash
sgrep search --json "your query"
```

### Index statistics

```bash
sgrep index --stats --json
```
