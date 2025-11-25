# External Vector DB & Model Integration: Research & Recommendations

**Date**: 2025-11-25
**Status**: Research Phase
**Goal**: Design world-class DX for external vector DB integration with powerful cloud capabilities

---

## Executive Summary

This document outlines an opinionated, developer-friendly approach to integrating external vector databases and larger embedding models into sgrep. The design prioritizes:

1. **Zero-config defaults** - Works out of the box for 95% of users
2. **Transparent hybrid mode** - Seamlessly combine local + cloud for best performance
3. **Larger, more accurate embeddings** - Support 1024-3072 dimensional vectors
4. **Powerful cloud tier** - Enterprise-grade performance without complexity
5. **Incremental adoption** - Start local, scale to cloud when needed

---

## Current State Analysis

### Strengths of Current Architecture

sgrep's current local-first architecture is excellent:

- **384-dimensional embeddings** via mxbai-embed-xsmall-v1 (22.7M params)
- **Adaptive search strategies**: Linear → HNSW → Binary quantization
- **Hybrid search scoring**: 60% semantic + 25% BM25 + 10% keyword + 5% recency
- **Smart resource management**: Pooled embedders, memory-mapped vectors, incremental indexing
- **Privacy-first**: Everything local, no network calls after model download

**Performance**: Meets targets (<50ms for <1K files, <120ms for 1K-10K files)

### Limitations for Enterprise Scale

1. **Model size constraint**: 384 dims limit semantic understanding
2. **Local-only storage**: Can't scale to 100K+ file repositories efficiently
3. **No cross-repository search**: Each repo isolated
4. **Embedding quality ceiling**: Small model trades accuracy for speed
5. **No advanced reranking**: Cross-encoder support exists but limited

---

## Industry Research Findings

### Vector Database Landscape (2025)

After extensive research, three vector databases emerge as leaders for code search:

#### 🥇 **Recommended: Qdrant**
- **Performance**: Written in Rust, 10-100ms p95 latency on 10M vectors
- **Features**: Advanced metadata filtering, HNSW + quantization, hybrid search
- **DX**: Excellent Rust SDK, simple API, Docker-first development
- **Deployment**: Self-hosted or managed cloud, scales to billions of vectors
- **Cost**: Open source (free self-hosted), cloud starts at $0.10/GB/month
- **Code-friendly**: Fast filtering by path, language, repo metadata

**Why Qdrant**: Native Rust integration, best balance of performance/features/cost, excellent for code search with complex metadata filtering.

#### 🥈 **Alternative: Pinecone**
- **Performance**: 23ms p95 latency, fully managed, auto-scaling
- **Features**: Serverless tier, namespaces, metadata filtering
- **DX**: Dead simple API, zero ops, excellent docs
- **Deployment**: Cloud-only (managed service)
- **Cost**: Free tier (100K vectors), then $0.096/pod/hour (~$70/month)
- **Code-friendly**: Good filtering, namespace isolation per repo

**Why Pinecone**: Zero operations overhead, predictable performance, best for teams that want managed services.

#### 🥉 **Alternative: Weaviate**
- **Performance**: 34ms p95 latency, GraphQL API, hybrid search built-in
- **Features**: Knowledge graphs, multi-modal, advanced filtering
- **DX**: Rich feature set, complex but powerful
- **Deployment**: Self-hosted or cloud, Kubernetes-native
- **Cost**: Open source, cloud pricing per query/storage
- **Code-friendly**: Semantic relationships between code concepts

**Why Weaviate**: Best for advanced use cases requiring knowledge graphs or multi-modal search.

### Embedding Model Landscape (2025)

Current model (mxbai-embed-xsmall-v1): 384 dims, 22.7M params, fast but limited accuracy.

#### 🥇 **Recommended for Cloud: Voyage Code-3**
- **Dimensions**: 2048 / 1024 / 512 / 256 (Matryoshka embeddings)
- **Context**: 32K tokens (vs OpenAI 8K)
- **Performance**: 97.3% MRR, 95% Recall@1 on code benchmarks
- **Advantage**: +13.8% vs OpenAI, +16.8% vs CodeSage on code retrieval
- **Pricing**: First 200M tokens free, then ~$0.10/1M tokens (estimate)
- **Quantization**: Supports float32, int8, uint8, binary, ubinary
- **API**: Simple REST API, batch embeddings

**Why Voyage Code-3**: Purpose-built for code, dramatically better accuracy, supports smaller dims for storage optimization, 4x larger context than OpenAI.

#### 🥈 **Alternative: OpenAI text-embedding-3-large**
- **Dimensions**: 3072 (can be shortened to 1024, 512, 256)
- **Context**: 8K tokens
- **Performance**: Strong general-purpose, good for code
- **Pricing**: $0.13/1M tokens (standard), $0.065/1M tokens (batch)
- **API**: Familiar OpenAI SDK, excellent docs
- **Advantage**: Widely used, stable, good ecosystem

**Why OpenAI**: Safe choice, familiar API, good for teams already using OpenAI.

#### 🥉 **Enhanced Local: Nomic Embed v1.5**
- **Dimensions**: 768 (2x current)
- **Size**: 137M params (still runs locally)
- **Performance**: Better than mxbai, runs on CPU/GPU
- **Pricing**: Free, open source, ONNX compatible
- **API**: Same fastembed interface as current

**Why Nomic**: 2x accuracy improvement while staying local-first, minimal architecture changes.

---

## Proposed Architecture: Hybrid Cloud-First

### Design Philosophy

**Opinionated Defaults** → **Progressive Enhancement** → **Enterprise Power**

```
Local-Only          Hybrid Mode           Cloud Mode
(Current)          (Recommended)         (Enterprise)

384 dims     →   1024 dims local    →  2048 dims cloud
Local store  →   Local + Qdrant    →  Qdrant cluster
mxbai-xsmall →   Nomic 768 + API   →  Voyage Code-3
<10K files   →   <100K files       →  Unlimited scale
```

### Core Principles

1. **Zero-Config Smart**: Detect repo size and auto-select strategy
2. **Hybrid by Default**: Keep local fast path, cloud for accuracy
3. **Transparent Fallback**: Cloud unavailable? Use local seamlessly
4. **Privacy Controls**: User controls what goes to cloud (workspace vs personal repos)
5. **Cost-Conscious**: Free tier covers 99% of developers

---

## Detailed Design: Three-Tier System

### Tier 1: Enhanced Local (Free, Always Works)

**Target**: Solo developers, small repos (<10K files), privacy-sensitive projects

**Configuration**:
```toml
# sgrep.toml (auto-generated on first run)
[embedding]
mode = "local"  # local | hybrid | cloud
model = "nomic-embed-v1.5"  # upgraded from mxbai
dimensions = 768  # 2x current

[storage]
backend = "local"  # local | qdrant | pinecone
```

**Changes**:
- Upgrade default model: mxbai-xsmall-v1 (384d) → nomic-embed-v1.5 (768d)
- Keep all current optimizations (pooled embedder, binary quantization, etc.)
- Add dimension negotiation (load old 384d indexes gracefully)

**DX**: Zero changes for users. `sgrep index` just works, now 2x more accurate.

### Tier 2: Hybrid Mode (Recommended, Cloud-Assisted)

**Target**: Teams, medium repos (10K-100K files), balanced accuracy/speed

**Configuration**:
```toml
[embedding]
mode = "hybrid"
local_model = "nomic-embed-v1.5"  # 768d, fast queries
cloud_model = "voyage-code-3"     # 1024d, indexing only
cloud_api_key = "env:VOYAGE_API_KEY"  # or store encrypted

[storage]
backend = "qdrant"
qdrant_url = "http://localhost:6333"  # or cloud URL
qdrant_collection_prefix = "sgrep_"  # auto namespace per repo

[hybrid]
# Smart routing: when to use cloud vs local
index_cloud = true       # Use cloud embeddings for indexing
search_local = true      # Use local for fast queries
search_cloud_fallback = true  # Rerank top-K with cloud if available
cloud_rerank_top_k = 100     # Fetch 100 local, rerank with cloud
```

**How it works**:
1. **Indexing**: Send code chunks to Voyage Code-3 API (1024d), store in Qdrant
2. **Fast Search**: Query local 768d index first (sub-50ms)
3. **Accuracy Boost**: Fetch top-100 local, re-score with cloud embeddings (optional)
4. **Fallback**: If cloud unavailable, pure local search

**DX**:
```bash
# One-time setup
sgrep config set embedding.mode hybrid
sgrep config set storage.backend qdrant
export VOYAGE_API_KEY="voy_..."

# Optionally start local Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Everything else is automatic
sgrep index  # Uses cloud embeddings, stores in Qdrant
sgrep search "auth middleware"  # Fast local search + cloud rerank
```

**Transparent**:
- Detects Qdrant at localhost:6333 automatically
- Falls back to local if Qdrant unreachable
- Uses local embeddings if API key missing
- Shows `[hybrid]` indicator in search output

### Tier 3: Cloud Mode (Enterprise, Maximum Performance)

**Target**: Large orgs, massive repos (100K+ files), cross-repo search

**Configuration**:
```toml
[embedding]
mode = "cloud"
cloud_model = "voyage-code-3"
cloud_api_key = "env:VOYAGE_API_KEY"
dimensions = 2048  # or 1024 for cost savings

[storage]
backend = "qdrant"
qdrant_url = "https://xyz.qdrant.tech"
qdrant_api_key = "env:QDRANT_API_KEY"

[enterprise]
cross_repo_search = true      # Search across all indexed repos
workspace_collections = true  # Share indexes within org
materialized_views = true     # Pre-compute common queries
auto_reindex_schedule = "0 2 * * *"  # Cron: daily at 2am
```

**How it works**:
1. **Pure Cloud**: All embeddings via Voyage Code-3 API (2048d)
2. **Managed Vector DB**: Qdrant Cloud cluster (auto-scaling)
3. **Cross-Repo Search**: Unified namespace across workspace
4. **Advanced Reranking**: Cross-encoder models for final ranking
5. **Scheduled Indexing**: Background jobs keep indexes fresh

**DX**:
```bash
# One-time org setup
sgrep org init --workspace mycompany
sgrep org config set storage.backend qdrant-cloud
sgrep org config set embedding.cloud_model voyage-code-3

# Per-developer usage (zero config)
sgrep login mycompany
cd my-project
sgrep index  # Auto-indexes to org workspace
sgrep search "payment processing" --workspace  # Search all org repos
```

---

## Configuration API: Opinionated & Simple

### Auto-Detection & Smart Defaults

```rust
// Proposed: src/config.rs

pub struct Config {
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub search: SearchConfig,
}

impl Config {
    /// Load config with smart defaults based on environment
    pub fn load_or_detect() -> Result<Self> {
        // 1. Try loading sgrep.toml from repo root or ~/.sgrep/config.toml
        if let Ok(config) = Self::load_from_file() {
            return Ok(config);
        }

        // 2. Auto-detect environment and set smart defaults
        let embedding = if env::var("VOYAGE_API_KEY").is_ok() {
            EmbeddingConfig::hybrid_default()  // API key found → hybrid
        } else {
            EmbeddingConfig::local_default()   // No API key → local
        };

        let storage = if Self::is_qdrant_available("http://localhost:6333") {
            StorageConfig::qdrant_local()      // Qdrant running → use it
        } else {
            StorageConfig::local_default()     // No Qdrant → local files
        };

        Ok(Self {
            embedding,
            storage,
            search: SearchConfig::default(),
        })
    }
}

pub enum EmbeddingBackend {
    Local { model: String, dimensions: usize },
    Hybrid { local_model: String, cloud_model: String, local_dims: usize, cloud_dims: usize },
    Cloud { model: String, dimensions: usize, api_key: String },
}

pub enum StorageBackend {
    Local { path: PathBuf },
    Qdrant { url: String, api_key: Option<String>, collection_prefix: String },
    Pinecone { api_key: String, environment: String, index_prefix: String },
    Weaviate { url: String, api_key: Option<String>, class_prefix: String },
}
```

### CLI Configuration Commands

```bash
# Show current config (auto-detected)
sgrep config show

# Interactive setup wizard
sgrep config init
  → Detects environment
  → Asks: "We found Qdrant at localhost:6333. Use it? [Y/n]"
  → Asks: "Enter Voyage API key for better accuracy (or skip): [optional]"
  → Writes sgrep.toml with chosen settings

# Manual overrides
sgrep config set embedding.mode hybrid
sgrep config set storage.backend qdrant
sgrep config set storage.qdrant_url http://localhost:6333
sgrep config set embedding.cloud_api_key "voy_..."

# Validate configuration
sgrep config validate
  → Checks API keys, tests connections, validates dimensions
  → Reports: "✓ Qdrant reachable at localhost:6333"
  → Reports: "✓ Voyage API key valid, quota: 180M/200M tokens"
```

---

## Implementation Phases

### Phase 1: Configuration & Qdrant Integration (Week 1-2)

**Goal**: Support local Qdrant as storage backend (still using local embeddings)

1. **Config System**:
   - Create `src/config.rs` with `Config`, `EmbeddingConfig`, `StorageConfig` structs
   - Implement TOML loading/saving (`sgrep.toml`)
   - Add `sgrep config` CLI commands (show, set, validate, init)
   - Auto-detection logic for Qdrant, API keys

2. **Qdrant Storage Backend**:
   - Add `qdrant-client` dependency
   - Implement `QdrantStore` trait (create_collection, insert_vectors, search)
   - Add repo hash → collection name mapping (e.g., `sgrep_<blake3_hash>`)
   - Support metadata filtering (path, language, repo)

3. **Storage Abstraction**:
   - Refactor `IndexStore` into trait `VectorStore`
   - Implement for `LocalStore` (existing) and `QdrantStore` (new)
   - Wire up config to select backend at runtime

4. **Testing**:
   - Docker Compose setup for local Qdrant
   - Integration tests: index to Qdrant, search, fallback to local
   - Migration tool: `sgrep migrate --from local --to qdrant`

**Deliverable**: Users can run `docker run -p 6333:6333 qdrant/qdrant` and `sgrep index` automatically uses Qdrant.

### Phase 2: Cloud Embeddings API (Week 3-4)

**Goal**: Support Voyage Code-3 and OpenAI embedding APIs

1. **Embedding Abstraction**:
   - Create `CloudEmbedder` trait implementation
   - Add `voyage-api` and `openai-api` clients (or use `reqwest` directly)
   - Batch request optimization (send 100 chunks per API call)
   - Rate limiting and retry logic (exponential backoff)

2. **Hybrid Mode Logic**:
   - `HybridEmbedder` wrapper: uses `CloudEmbedder` for indexing, `LocalEmbedder` for search
   - Dimension negotiation: allow different dims for index vs query
   - Cloud reranking: fetch top-K local, re-score with cloud embeddings

3. **API Key Management**:
   - Secure storage: encrypt API keys in `~/.sgrep/secrets.enc` (using age or similar)
   - Environment variable support: `VOYAGE_API_KEY`, `OPENAI_API_KEY`
   - Per-project config: `.sgrep.toml` with `api_key = "env:VAR_NAME"`

4. **Cost Tracking**:
   - Count tokens sent to API
   - Show cost estimates: "Indexed 10K chunks (12M tokens) = ~$1.20"
   - Warning if approaching free tier limit

**Deliverable**: `sgrep config set embedding.mode hybrid` + API key = automatic cloud-powered indexing.

### Phase 3: Advanced Search & Reranking (Week 5-6)

**Goal**: Implement sophisticated reranking and hybrid search strategies

1. **Cloud Reranking**:
   - Fetch top-100 local results
   - Re-embed query with cloud model (1024d or 2048d)
   - Re-score all 100 candidates
   - Return top-10 after reranking

2. **Cross-Encoder Reranking**:
   - Integrate `cross-encoder` models (e.g., `ms-marco-MiniLM-L-6-v2`)
   - Run on CPU or GPU for final reranking stage
   - Make it optional: `--rerank` flag or `search.rerank = true` in config

3. **Search Strategy Selector**:
   - Auto-detect best strategy based on config + index size
   - Local-only: Use existing (linear/HNSW/binary)
   - Hybrid: Local search + cloud rerank
   - Cloud: Pure vector DB search (Qdrant HNSW)

4. **Multi-Stage Retrieval**:
   - Stage 1: BM25 keyword filter (top 1000)
   - Stage 2: Vector similarity (top 100)
   - Stage 3: Cross-encoder rerank (top 10)

**Deliverable**: Research-grade accuracy with minimal latency increase.

### Phase 4: Enterprise Features (Week 7-8)

**Goal**: Cross-repo search, workspaces, managed deployments

1. **Workspace Management**:
   - `sgrep org init --workspace mycompany`
   - Shared Qdrant collections: `sgrep_workspace_mycompany`
   - Per-user auth: `sgrep login mycompany` (OAuth or API key)

2. **Cross-Repo Search**:
   - Index multiple repos into same collection (with repo metadata)
   - `sgrep search "auth" --workspace` searches all org repos
   - Result grouping by repo

3. **Scheduled Indexing**:
   - Background daemon: `sgrep daemon start`
   - Watches git webhooks or polls repos
   - Incremental reindexing on schedule

4. **Managed Cloud Deployment**:
   - Terraform/Helm charts for self-hosted Qdrant
   - Or: Qdrant Cloud integration guide
   - Monitoring: Grafana dashboards for index freshness, query latency

**Deliverable**: Enterprise-ready semantic code search platform.

---

## Cost Analysis

### Typical Developer Workflows

#### Small Team (10 developers, 50 repos, 1M LoC total)

**Indexing**:
- ~500K chunks × 150 tokens/chunk = 75M tokens/month
- Voyage Code-3: 75M tokens = **$7.50/month** (after free tier)
- Qdrant self-hosted: **$0** (Docker on $5/month VPS)
- **Total: ~$10/month**

**Search**:
- 1000 searches/day × 20 tokens/query = 600K tokens/month
- Voyage Code-3 reranking (optional): **$0.60/month**
- **Total: ~$10/month** (or $1 if no reranking)

**Per developer**: **$1/month** (10x cheaper than Copilot)

#### Large Enterprise (1000 developers, 5000 repos, 100M LoC total)

**Indexing**:
- ~50M chunks × 150 tokens/chunk = 7.5B tokens/month
- Voyage Code-3: **$750/month** (or negotiate enterprise contract)
- Qdrant Cloud (1TB vectors): **$100/month** (managed cluster)
- **Total: ~$850/month**

**Search**:
- 100K searches/day × 20 tokens/query = 60M tokens/month
- Voyage Code-3 reranking: **$60/month**
- **Total: ~$900/month**

**Per developer**: **$0.90/month** (negligible)

### Comparison: Build vs Buy

| Solution | Setup Cost | Monthly Cost | Accuracy | Maintenance |
|----------|-----------|--------------|----------|-------------|
| **sgrep local** | $0 | $0 | 75% | Low |
| **sgrep hybrid** | $0 | $10-100 | 90% | Low |
| **sgrep cloud** | $0 | $100-1000 | 95% | Low |
| **Sourcegraph** | $50K | $5K-50K | 90% | Outsourced |
| **Build in-house** | $100K | $10K+ | 80% | High |

**Recommendation**: Hybrid mode offers best ROI for 90% of teams.

---

## Competitive Analysis

### vs Sourcegraph Cody
- **Advantage**: Open source, self-hosted, 10x cheaper
- **Disadvantage**: No web UI (CLI-only), fewer integrations
- **Strategy**: Target cost-conscious teams, privacy-focused orgs

### vs Cursor @codebase
- **Advantage**: Better local experience, faster, more privacy
- **Disadvantage**: Not integrated in editor (requires separate tool)
- **Strategy**: Position as "Cursor-compatible" via plugin ecosystem

### vs GitHub Copilot (upcoming code search)
- **Advantage**: More accurate (purpose-built), cheaper, self-hosted option
- **Disadvantage**: Smaller ecosystem, less brand recognition
- **Strategy**: Emphasize accuracy gains (show benchmarks), open source trust

---

## Migration Path for Existing Users

### Backward Compatibility

1. **Detect old indexes**: Check for 384d vectors in `~/.sgrep/indexes/`
2. **Offer upgrade**: `sgrep upgrade --to 768d` or `--to hybrid`
3. **Seamless fallback**: If new model unavailable, use old model
4. **Side-by-side**: Store both 384d and 768d vectors during transition

### Upgrade CLI

```bash
# Check what's needed
sgrep doctor
  → "Your indexes use 384-dim embeddings (old model)"
  → "Upgrade to 768-dim for 2x better accuracy? Run: sgrep upgrade"

# Upgrade all repos
sgrep upgrade
  → Re-indexes all repos with new 768d model
  → Keeps old 384d vectors for rollback
  → Shows progress: "Upgraded 8/50 repos..."

# Or upgrade per-repo
cd my-project
sgrep index --force --model nomic-embed-v1.5
```

---

## Developer Experience Priorities

### 1. Zero-Config Magic

**Good**:
```bash
sgrep search "auth middleware"  # Just works, uses smart defaults
```

**Bad**:
```bash
sgrep search "auth" --embedding-model voyage-code-3 --dimensions 1024 \
  --storage qdrant --qdrant-url http://localhost:6333 --api-key $KEY
```

**Implementation**: Auto-detect environment, use sensible defaults, only ask when necessary.

### 2. Progressive Disclosure

**Beginner**: `sgrep search "query"` – zero config, fast, good enough
**Intermediate**: `sgrep config init` – guided setup, hybrid mode
**Expert**: `sgrep.toml` – full control, custom models, enterprise features

**Implementation**: CLI flags < env vars < config file < auto-detection

### 3. Transparent Observability

Show what's happening under the hood:

```bash
$ sgrep search "auth" -v
[hybrid] Using local Nomic-768d for initial search...
[hybrid] Found 100 candidates in 34ms
[hybrid] Re-ranking with Voyage-1024d...
[hybrid] Re-ranked top 10 in 89ms
[results] Total: 123ms (34ms local + 89ms cloud)
```

**Implementation**: `-v` flag, `RUST_LOG=sgrep=debug`, JSON telemetry

### 4. Fail-Safe Degradation

```bash
$ sgrep search "auth"
[warn] Qdrant unreachable, falling back to local storage
[warn] Voyage API error, using local embeddings only
[results] 10 results in 45ms (local-only mode)
```

**Implementation**: Try cloud, catch errors, fall back to local, log warnings

### 5. Cost Awareness

```bash
$ sgrep index --dry-run
Estimated cost:
  - 50K chunks × 150 tokens = 7.5M tokens
  - Voyage Code-3: $0.75 (within free tier)
  - Qdrant: $0 (local Docker)
Run `sgrep index` to proceed.

$ sgrep stats costs
This month (Nov 2025):
  - Embedding API: 45M tokens = $4.50
  - Qdrant: Self-hosted = $0
  - Total: $4.50
  - Free tier remaining: 155M tokens
```

**Implementation**: Track token counts, show cost estimates, warn before expensive ops

---

## Security & Privacy Considerations

### 1. API Key Storage

**Problem**: Storing API keys securely
**Solution**:
- Encrypt in `~/.sgrep/secrets.enc` using `age` encryption
- Or use env vars: `VOYAGE_API_KEY=...`
- Or per-project: `.sgrep.toml` with `api_key = "env:KEY"`
- Never commit API keys to git (add to `.gitignore`)

### 2. Code Privacy

**Problem**: Sending proprietary code to cloud APIs
**Solution**:
- Opt-in only: Default is local-only
- Per-repo config: Mark repos as `private = true` to block cloud
- Workspace config: Allowlist repos for cloud embeddings
- Audit log: Track what was sent to cloud APIs

### 3. Data Retention

**Problem**: How long do vector DBs keep data?
**Solution**:
- Self-hosted Qdrant: Full control, delete anytime
- Cloud vendors: Check their data retention policies
- Implement `sgrep purge --repo <name>` to delete from cloud

### 4. Compliance

**Problem**: GDPR, SOC2, HIPAA requirements
**Solution**:
- Self-hosted mode for strict compliance
- Qdrant Cloud offers SOC2 compliance
- Document data flows in compliance guide
- Per-repo privacy flags: `sgrep config set repo.private true`

---

## Open Questions & Decisions Needed

### 1. Default Embedding Dimensions?

**Options**:
- **384d**: Keep current (fast, small, good enough)
- **768d**: Nomic Embed (2x better, still fast)
- **1024d**: Voyage/OpenAI (best accuracy, cloud-only)

**Recommendation**: **768d** (Nomic) for new indexes, 384d for existing (backward compat)

### 2. Default Vector DB?

**Options**:
- **Local files**: Keep current (zero deps)
- **Qdrant local**: Best balance (Docker required)
- **Qdrant cloud**: Easiest (but costs money)

**Recommendation**: **Local files by default**, auto-upgrade to Qdrant if detected

### 3. Free Tier Strategy?

**Options**:
- **Pure FOSS**: No cloud features (limits growth)
- **Generous free tier**: 200M tokens/month (matches Voyage)
- **Freemium**: Free local, paid cloud (alienates OSS users)

**Recommendation**: **Generous free tier** (200M tokens = ~500 repos) + transparent upsell for enterprises

### 4. API Provider Choice?

**Options**:
- **Voyage only**: Best accuracy, code-specific
- **OpenAI only**: Familiar, widely available
- **Both**: More complexity, user choice

**Recommendation**: **Both** – default to Voyage, fallback to OpenAI, let users choose

### 5. Cross-Repo Search UX?

**Options**:
- **Explicit flag**: `sgrep search "auth" --workspace`
- **Auto-detect**: Search workspace if configured
- **Separate command**: `sgrep workspace search "auth"`

**Recommendation**: **Explicit flag** (`--workspace`) to avoid accidental cross-repo leaks

---

## Success Metrics

### Technical Metrics

- **Accuracy**: NDCG@10 > 0.90 (vs 0.75 current)
- **Latency**: p95 < 150ms for hybrid search (vs 120ms local)
- **Scale**: Support 100K+ file repos (vs 10K current)
- **Reliability**: 99.9% uptime for cloud services

### Adoption Metrics

- **Conversion**: 30% of users try hybrid mode within 1 month
- **Retention**: 80% of hybrid users stay on hybrid (vs downgrade)
- **NPS**: Net Promoter Score > 50
- **Revenue**: $10K MRR from enterprise licenses (if monetized)

### Developer Happiness

- **Time-to-first-search**: < 30 seconds (from install to first result)
- **Time-to-hybrid**: < 5 minutes (from local to hybrid setup)
- **Support tickets**: < 5% of users need help with setup
- **GitHub stars**: 5K+ (vs ~100 current)

---

## References & Sources

### Vector Database Research
- [Most Popular Vector Databases You Must Know in 2025](https://dataaspirant.com/popular-vector-databases/)
- [Best 17 Vector Databases for 2025](https://lakefs.io/blog/best-vector-databases/)
- [Top Vector Database for RAG: Qdrant vs Weaviate vs Pinecone](https://research.aimultiple.com/vector-database-for-rag/)
- [Vector Database Comparison 2025: Pinecone vs Weaviate vs Chroma vs Qdrant](https://sysdebug.com/posts/vector-database-comparison-guide-2025/)

### Code Search Architecture
- [Cursor vs Sourcegraph Cody: embeddings and monorepo scale](https://www.augmentcode.com/guides/cursor-vs-sourcegraph-cody-embeddings-and-monorepo-scale)
- [An attempt to build cursor's @codebase feature](https://blog.lancedb.com/building-rag-on-codebases-part-2)
- [Vector Embeddings for Your Entire Codebase: A Guide](https://dzone.com/articles/vector-embeddings-codebase-guide)

### Embedding Models
- [13 Best Embedding Models in 2025](https://elephas.app/blog/best-embedding-models)
- [Code Isn't Just Text: A Deep Dive into Code Embedding Models](https://medium.com/@abhilasha4042/code-isnt-just-text-a-deep-dive-into-code-embedding-models-418cf27ea576)
- [voyage-code-3: more accurate code retrieval](https://blog.voyageai.com/2024/12/04/voyage-code-3/)
- [Voyage AI Pricing](https://docs.voyageai.com/docs/pricing)
- [OpenAI text-embedding-3-large](https://openai.com/index/new-embedding-models-and-api-updates/)

---

## Next Steps

1. **Validation**: Share this doc with team, gather feedback
2. **Prototype**: Build Phase 1 (Qdrant integration) in 1-2 weeks
3. **User Testing**: Release alpha to 10-20 beta users, measure adoption
4. **Iteration**: Refine DX based on feedback, prioritize pain points
5. **Launch**: Ship v1.0 with hybrid mode, announce on HN/Reddit/Twitter

**Timeline**: 8 weeks to v1.0 (cloud-integrated hybrid mode)
**Risk**: Complexity creep – stay disciplined on "zero-config" philosophy

---

*End of Research Document*
