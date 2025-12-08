# Configuration

## Global flags

- `--device <cpu|cuda|coreml>` (or `SGREP_DEVICE`)
- `--offline` (or `SGREP_OFFLINE=1`) to forbid downloads and fail fast if the model is missing
- `--threads <n>` (or `SGREP_MAX_THREADS`) to bound parallelism
- `--cpu-preset <auto|low|medium|high|background>` (or `SGREP_CPU_PRESET`)
  
Command-specific flags (offload, remote, detach) are listed per subcommand below. If `--offload` is omitted, the default comes from `[embedding].provider` in the config (default `local`).

## Commands

### search

- `--path` (default `.`)
- `--limit` (default `10`)
- `--context` to return chunk bodies
- `--glob <pattern>` (repeatable)
- `--filters key=value` (repeatable) for metadata filters like `lang=rust`
- `--json` for structured output
- `--debug` to surface scores and timings
- `--offload` (or `SGREP_OFFLOAD`) to use [Modal.dev](https://modal.com) for embeddings; omit to follow config, pass `--offload=false` to force local
- `--remote` (or `SGREP_REMOTE=1`) to query a configured remote vector store (Pinecone/Turbopuffer)

### index

- `--force` for a full rebuild
- `--batch-size` (or `SGREP_BATCH_SIZE`) to override embedder batch size
- `--profile` to print per-phase timings
- `--stats` to print index statistics without rebuilding
- `--json` to emit stats as JSON (only with `--stats`)
- `--offload` (or `SGREP_OFFLOAD`) to embed via Modal.dev GPUs; omit to follow config, pass `--offload=false` to force local
- `--remote` (or `SGREP_REMOTE=1`) to write to a configured remote vector store
- `--detach` to run indexing in the background (not compatible with `--stats`)
- `path` argument optional; defaults to the current directory

### watch

- `path` argument optional; defaults to the current directory
- `--debounce-ms` (default `500`)
- `--batch-size` (or `SGREP_BATCH_SIZE`)
- `--offload` (or `SGREP_OFFLOAD`) to embed updates via Modal.dev GPUs; omit to follow config, pass `--offload=false` to force local
- `--remote` (or `SGREP_REMOTE=1`) to mirror index updates to a configured remote store (supported for watch)
- `--detach` to run watch in the background

### config

- `--init` to write a default config file
- `--show-model-dir` to print the embedding cache location
- `--verify-model` to assert required model files exist

## Configuration file

`SGREP_CONFIG` overrides the path; otherwise it uses `SGREP_HOME/config.toml`, defaulting to `~/.sgrep/config.toml`. Example:

```toml
[embedding]
# Default provider for embeddings when --offload is not set
provider = "local" # or "modal"
```

## Modal.dev offload

Offload embeddings to [Modal.dev](https://modal.com) GPUs for faster processing on large codebases.

```toml
[modal]
# CLI authentication (for deployment)
token_id = "ak-..."              # Modal API token ID from https://modal.com/settings
token_secret = "as-..."          # Modal API token secret from https://modal.com/settings

# Endpoint authentication (for HTTP requests)
proxy_token_id = "wk-..."        # Modal proxy auth token ID from https://modal.com/settings
proxy_token_secret = "ws-..."    # Modal proxy auth token secret from https://modal.com/settings

gpu_tier = "high"                # budget (T4), balanced (A10G), high (L40S)
batch_size = 128                 # texts per request
endpoint = "https://..."         # auto-populated after first deploy
```

**Note:** Embedding dimension is fixed at 384 to match the local embedder. This ensures you can switch between local and Modal embeddings seamlessly.

**Authentication:**

Modal requires two types of tokens:

1. **API tokens** (`ak-`/`as-` prefix) - Used for Modal CLI deployment
   - Get from [Modal Settings](https://modal.com/settings) under "API Tokens"
   - Or run `modal token new` to authenticate via browser

2. **Proxy auth tokens** (`wk-`/`ws-` prefix) - Used for endpoint authentication
   - Get from [Modal Settings](https://modal.com/settings) under "Proxy Auth Tokens"
   - These secure your endpoints so only you can call them

**GPU tiers:**

| Tier | GPU | Cost | Best for |
|------|-----|------|----------|
| `budget` | T4 | ~$0.25/hr | Cost-sensitive workloads |
| `balanced` | A10G | ~$0.45/hr | Good balance of speed/cost |
| `high` | L40S | ~$1.10/hr | Maximum performance (default) |

**Models used:**
- Embeddings: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (8K context, truncated to 384 dimensions for local compatibility)

First run auto-deploys the service; subsequent runs use cached endpoints.

## Remote storage (flattened config)

Choose a provider by configuring its section; optionally set `remote_provider` to disambiguate if multiple are present. Enable with `--remote` or `SGREP_REMOTE=1`.

**Pinecone (example):**
```toml
[pinecone]
api_key = "your-key"                       # or set PINECONE_API_KEY
endpoint = "https://YOUR-INDEX.svc.region.pinecone.io"
namespace = "optional-namespace"           # defaults to repo hash
```

**Turbopuffer (example):**
```toml
[turbopuffer]
api_key = "tpuf_your_key"                  # or set TURBOPUFFER_API_KEY
region = "gcp-us-central1"
namespace_prefix = "sgrep"
```

**Optional selector:**
```toml
remote_provider = "pinecone"  # or "turbopuffer"; inferred if only one is configured
```

## Environment variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_HOME` | `~/.sgrep` | Index and config directory |
| `SGREP_CONFIG` | `$SGREP_HOME/config.toml` | Config file path override |
| `FASTEMBED_CACHE_DIR` | `$SGREP_HOME/cache/fastembed` | Model weights cache |

### Runtime Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `coreml` |
| `SGREP_OFFLINE` | `0` | Block network calls (`1` or `true`) |
| `SGREP_OFFLOAD` | `0` | Enable Modal.dev GPU offload |
| `SGREP_REMOTE` | `0` | Enable remote vector storage |
| `SGREP_BATCH_SIZE` | auto | Embedding batch size |
| `SGREP_INIT_TIMEOUT_SECS` | `120` | Model initialization timeout |

### Threading

| Variable | Default | Description |
|----------|---------|-------------|
| `SGREP_MAX_THREADS` | all cores | Maximum total threads |
| `SGREP_CPU_PRESET` | `auto` | Preset: `auto`, `background`, `low`, `medium`, `high` |
| `RAYON_NUM_THREADS` | auto | Rayon parallel pool size |

### ONNX Runtime (Advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `ORT_INTRA_OP_NUM_THREADS` | 4 | Threads within operators |
| `ORT_INTER_OP_NUM_THREADS` | auto | Threads between operators |
| `ORT_NUM_THREADS` | auto | Legacy thread setting |

### BLAS Backends (Advanced)

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | auto | OpenMP threads |
| `MKL_NUM_THREADS` | auto | Intel MKL threads |
| `OPENBLAS_NUM_THREADS` | auto | OpenBLAS threads |
| `VECLIB_MAXIMUM_THREADS` | auto | macOS Veclib threads |

### Authentication

| Variable | Description |
|----------|-------------|
| `MODAL_TOKEN_ID` | Modal CLI token ID (`ak-...`) |
| `MODAL_TOKEN_SECRET` | Modal CLI token secret (`as-...`) |
| `PINECONE_API_KEY` | Pinecone API key |
| `TURBOPUFFER_API_KEY` | Turbopuffer API key |

### Network

| Variable | Description |
|----------|-------------|
| `HTTP_PROXY` | HTTP proxy for model downloads |
| `HTTPS_PROXY` | HTTPS proxy for model downloads |

### Debugging

| Variable | Example | Description |
|----------|---------|-------------|
| `RUST_LOG` | `sgrep=debug` | Log level (trace, debug, info, warn, error) |

---

## JSON Output Schema

Use `--json` for machine-readable output.

### Search Results

```bash
sgrep search --json "your query"
```

**Response:**

```json
{
  "query": "your query",
  "elapsed_ms": 123,
  "index_metadata": {
    "total_chunks": 1000,
    "total_files": 50,
    "embedding_dim": 384,
    "has_graph": true
  },
  "matches": [
    {
      "path": "src/auth/handler.rs",
      "start_line": 10,
      "end_line": 25,
      "content": "fn authenticate(user: &User) -> Result<Token> {...}",
      "score": 0.85
    }
  ]
}
```

**With `--debug`:**

```json
{
  "matches": [
    {
      "path": "src/auth/handler.rs",
      "start_line": 10,
      "end_line": 25,
      "content": "...",
      "score": 0.85,
      "semantic_score": 0.82,
      "bm25_score": 0.88
    }
  ]
}
```

### Index Statistics

```bash
sgrep index --stats --json
```

**Response:**

```json
{
  "path": "/path/to/repo",
  "total_chunks": 1234,
  "total_files": 56,
  "total_directories": 12,
  "embedding_dim": 384,
  "graph_symbols": 789,
  "graph_edges": 456,
  "index_size_bytes": 12345678
}
```
