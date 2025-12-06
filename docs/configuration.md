# Configuration

## Global flags

- `--device <cpu|cuda|coreml>` (or `SGREP_DEVICE`)
- `--offline` (or `SGREP_OFFLINE=1`) to forbid downloads and fail fast if the model is missing
- `--threads <n>` (or `SGREP_MAX_THREADS`) to bound parallelism
- `--cpu-preset <auto|low|medium|high|background>` (or `SGREP_CPU_PRESET`)
- `--offload` (or `SGREP_OFFLOAD=1`) to use [Modal.dev](https://modal.com) for GPU-accelerated embeddings
- `--remote` (or `SGREP_REMOTE=1`) to use [Turbopuffer](https://turbopuffer.com) for remote vector storage

## Commands

### search

- `--path` (default `.`)
- `--limit` (default `10`)
- `--context` to return chunk bodies
- `--glob <pattern>` (repeatable)
- `--filters key=value` (repeatable) for metadata filters like `lang=rust`
- `--json` for structured output
- `--debug` to surface scores and timings

### index

- `--force` for a full rebuild
- `--batch-size` (or `SGREP_BATCH_SIZE`) to override embedder batch size
- `--profile` to print per-phase timings
- `path` argument optional; defaults to the current directory

### watch

- `path` argument optional; defaults to the current directory
- `--debounce-ms` (default `500`)
- `--batch-size` (or `SGREP_BATCH_SIZE`)

### config

- `--init` to write a default config file
- `--show-model-dir` to print the embedding cache location
- `--verify-model` to assert required model files exist

## Configuration file

`SGREP_CONFIG` overrides the path; otherwise it uses `SGREP_HOME/config.toml`, defaulting to `~/.sgrep/config.toml`. Example:

```toml
[embedding]
provider = "local"
```

## Modal.dev offload

Offload embeddings and reranking to [Modal.dev](https://modal.com) GPUs for faster processing on large codebases.

```toml
[modal]
# CLI authentication (for deployment)
token_id = "ak-..."              # Modal API token ID from https://modal.com/settings
token_secret = "as-..."          # Modal API token secret from https://modal.com/settings

# Endpoint authentication (for HTTP requests)
proxy_token_id = "wk-..."        # Modal proxy auth token ID from https://modal.com/settings
proxy_token_secret = "ws-..."    # Modal proxy auth token secret from https://modal.com/settings

gpu_tier = "high"                # budget (T4), balanced (A10G), high (L40S)
dimension = 4096                 # embedding dimension
batch_size = 32                  # texts per request
endpoint = "https://..."         # auto-populated after first deploy
```

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
- Embeddings: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (8K context, 4096 dimensions)
- Reranking: [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

First run auto-deploys the service; subsequent runs use cached endpoints.

## Turbopuffer remote storage

Store indexes in [Turbopuffer](https://turbopuffer.com) serverless vector database for remote access.

```toml
[turbopuffer]
api_key = "tpuf_your_key"        # or set TURBOPUFFER_API_KEY
region = "gcp-us-central1"       # default region
namespace_prefix = "sgrep"       # namespace prefix for indexes
```

## Environment variables

- `SGREP_HOME` to relocate indexes and config (default OS data dir such as `~/.local/share/sgrep`)
- `FASTEMBED_CACHE_DIR` to relocate the embedding cache (default OS cache dir such as `~/.local/share/sgrep/cache/fastembed`)
- `SGREP_INIT_TIMEOUT_SECS` to extend model startup (default `120`)
- `MODAL_TOKEN_ID` for Modal CLI authentication (token ID from modal.com/settings)
- `MODAL_TOKEN_SECRET` for Modal CLI authentication (token secret from modal.com/settings)
- `SGREP_OFFLOAD` to enable Modal.dev offload (`1` or `true`)
- `SGREP_REMOTE` to enable Turbopuffer remote storage (`1` or `true`)
- `HTTP_PROXY` / `HTTPS_PROXY` for model downloads
- `RUST_LOG` for tracing (e.g., `sgrep=debug`)
- `RAYON_NUM_THREADS` to hard-cap the Rayon pool
