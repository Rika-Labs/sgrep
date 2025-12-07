# Deployment Options

sgrep gives you complete control over where your code is processed and where your index is stored. Mix and match to fit your privacy, performance, and scale requirements.

## The Flexibility Matrix

```
┌─────────────────┬───────────────────────────┬───────────────────────────┐
│                 │   Store Locally           │   Store Remotely          │
│                 │   (default)               │   (Pinecone/Turbopuffer)  │
├─────────────────┼───────────────────────────┼───────────────────────────┤
│ Process Locally │ Fully Private             │ Scale Storage             │
│ (default)       │ Zero network calls        │ Keep embeddings private   │
│                 │ Works offline             │ Index multiple repos      │
├─────────────────┼───────────────────────────┼───────────────────────────┤
│ Offload to GPU  │ Fast Indexing             │ Full Cloud Scale          │
│ (Modal.dev)     │ 10-50x faster embedding   │ Maximum performance       │
│                 │ Index stays local         │ Serverless scaling        │
└─────────────────┴───────────────────────────┴───────────────────────────┘
```

## Mode 1: Fully Private (Default)

**Best for:** Individual developers, sensitive codebases, air-gapped environments

Everything runs on your machine. No data leaves your computer.

```bash
# Just run it - this is the default
sgrep index
sgrep search "error handling"
```

**What happens:**
- Embeddings generated locally using ONNX (`jina-embeddings-v2-base-code`)
- Index stored in `~/.sgrep/indexes/` (or `SGREP_HOME`)
- First run downloads the model (~400MB), then works fully offline

**Lock down with offline mode:**
```bash
sgrep index --offline        # Fails if model not cached
SGREP_OFFLINE=1 sgrep search "auth"
```

---

## Mode 2: Local Processing + Remote Storage

**Best for:** Multi-repo search, team sharing, scaling beyond local disk

Generate embeddings locally (keeping your code private), store the index in the cloud.

```bash
# Configure remote storage (one-time)
sgrep config --init
# Edit ~/.sgrep/config.toml with your provider

# Index with remote storage
sgrep index --remote
sgrep search --remote "database connection"
```

**Configuration (Turbopuffer):**
```toml
[turbopuffer]
api_key = "tpuf_your_key"      # or set TURBOPUFFER_API_KEY
region = "gcp-us-central1"
namespace_prefix = "sgrep"
```

**Configuration (Pinecone):**
```toml
[pinecone]
api_key = "your-key"           # or set PINECONE_API_KEY
endpoint = "https://YOUR-INDEX.svc.region.pinecone.io"
namespace = "my-project"       # defaults to repo hash
```

**Privacy note:** Your code is processed locally. Only the embeddings (384-dimensional vectors) are sent to the remote store. The vectors cannot be reversed to recover your source code.

---

## Mode 3: GPU Offload + Local Storage

**Best for:** Large codebases, fast initial indexing, teams with GPU budgets

Offload embedding to Modal.dev GPUs (10-50x faster), keep the index on your machine.

```bash
# Set Modal credentials
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."

# Index with GPU acceleration
sgrep index --offload
sgrep search --offload "authentication middleware"
```

**Configuration:**
```toml
[modal]
token_id = "ak-..."
token_secret = "as-..."
gpu_tier = "high"              # budget (T4), balanced (A10G), high (L40S)
batch_size = 128
```

**First run:** Auto-deploys a Modal service with Qwen3-Embedding-8B. Subsequent runs use the cached endpoint.

**GPU tiers:**

| Tier | GPU | Cost | Speed |
|------|-----|------|-------|
| `budget` | T4 | ~$0.25/hr | Good for small repos |
| `balanced` | A10G | ~$0.45/hr | Sweet spot |
| `high` | L40S | ~$1.10/hr | Maximum throughput |

---

## Mode 4: Full Cloud Scale

**Best for:** Enterprise teams, massive codebases, maximum performance

GPU-accelerated processing + serverless vector storage.

```bash
# Set all credentials
export MODAL_TOKEN_ID="ak-..."
export MODAL_TOKEN_SECRET="as-..."

# Configure remote storage in ~/.sgrep/config.toml

# Full cloud operation
sgrep index --offload --remote
sgrep search --offload --remote "distributed tracing"
```

**Combined configuration:**
```toml
[modal]
token_id = "ak-..."
token_secret = "as-..."
gpu_tier = "high"

[turbopuffer]
api_key = "tpuf_your_key"
region = "gcp-us-central1"
namespace_prefix = "sgrep"
```

**Benefits:**
- Fastest possible indexing (GPU acceleration)
- Unlimited index size (cloud storage)
- Query from anywhere (remote index)
- Automatic scaling

---

## Choosing Your Mode

| Need | Recommended Mode |
|------|------------------|
| Maximum privacy | Mode 1: Fully Private |
| Offline/air-gapped | Mode 1 with `--offline` |
| Index multiple repos | Mode 2: Remote Storage |
| Share index with team | Mode 2: Remote Storage |
| Large codebase (>100k files) | Mode 3 or 4: GPU Offload |
| Fast initial setup | Mode 3: GPU Offload |
| Enterprise scale | Mode 4: Full Cloud |

## Mixing Modes

You can switch between modes at any time:

```bash
# Start local, go remote later
sgrep index                    # Local processing, local storage
sgrep index --remote           # Push to remote storage
sgrep search --remote "query"  # Search remote index

# Different modes per repo
cd ~/work/private-project
sgrep index                    # Keep this one local

cd ~/work/team-project
sgrep index --remote           # Share this one
```

## Environment Variables

Quick reference for deployment-related settings:

| Variable | Purpose |
|----------|---------|
| `SGREP_OFFLOAD=1` | Enable GPU offload (same as `--offload`) |
| `SGREP_REMOTE=1` | Enable remote storage (same as `--remote`) |
| `SGREP_OFFLINE=1` | Block all network calls |
| `MODAL_TOKEN_ID` | Modal.dev authentication |
| `MODAL_TOKEN_SECRET` | Modal.dev authentication |
| `PINECONE_API_KEY` | Pinecone authentication |
| `TURBOPUFFER_API_KEY` | Turbopuffer authentication |

See [Configuration](configuration.md) for all options.
