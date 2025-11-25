# External Vector DB Integration: Implementation Examples

This document provides concrete code examples and configuration samples for the proposed external vector DB integration.

---

## Configuration File Examples

### Example 1: Local-Only (Default)

```toml
# .sgrep.toml (or ~/.sgrep/config.toml)
version = "1.0"

[embedding]
mode = "local"
model = "nomic-embed-v1.5"
dimensions = 768
# Device selection: cpu, cuda, coreml
device = "auto"
pool_size = 8  # Number of parallel embedding instances

[storage]
backend = "local"
path = "~/.sgrep/indexes"

[search]
# Hybrid search weights
semantic_weight = 0.60
bm25_weight = 0.25
keyword_weight = 0.10
recency_weight = 0.05
# Default result limit
default_limit = 10
# Enable cross-encoder reranking (optional, slower but more accurate)
rerank = false
```

### Example 2: Hybrid Mode (Recommended)

```toml
version = "1.0"

[embedding]
mode = "hybrid"
# Local model for fast queries
local_model = "nomic-embed-v1.5"
local_dimensions = 768
# Cloud model for accurate indexing
cloud_model = "voyage-code-3"
cloud_dimensions = 1024
# API key (use env var for security)
cloud_api_key = "env:VOYAGE_API_KEY"
# Batch size for cloud API (reduces API calls)
cloud_batch_size = 100

[storage]
backend = "qdrant"
qdrant_url = "http://localhost:6333"
# Optional: for Qdrant Cloud
# qdrant_api_key = "env:QDRANT_API_KEY"
# Collection naming: sgrep_<repo_hash>
collection_prefix = "sgrep_"

[hybrid]
# Use cloud embeddings for indexing
index_cloud = true
# Use local embeddings for initial search (fast)
search_local = true
# Re-rank top results with cloud embeddings (optional)
search_cloud_fallback = true
# How many results to re-rank
cloud_rerank_top_k = 100

[privacy]
# Prevent specific repos from using cloud
private_repos = [
    "~/work/client-a/*",
    "~/personal/secrets-*"
]
# Require explicit opt-in for cloud per repo
require_cloud_consent = false
```

### Example 3: Cloud Mode (Enterprise)

```toml
version = "1.0"

[embedding]
mode = "cloud"
cloud_model = "voyage-code-3"
cloud_dimensions = 2048  # Max accuracy
cloud_api_key = "env:VOYAGE_API_KEY"
cloud_batch_size = 250
# Timeout for API calls (seconds)
cloud_timeout = 30
# Retry configuration
cloud_max_retries = 3
cloud_retry_delay = 1.0

[storage]
backend = "qdrant"
qdrant_url = "https://xyz-cluster.qdrant.tech:6333"
qdrant_api_key = "env:QDRANT_API_KEY"
collection_prefix = "acme_corp_"

[enterprise]
# Enable cross-repository search
cross_repo_search = true
# Workspace identifier (shared collections)
workspace_id = "acme-engineering"
# Auto-reindex on schedule (cron format)
auto_reindex_schedule = "0 2 * * *"  # 2 AM daily
# Keep multiple index versions
versioned_indexes = true
# Number of versions to retain
max_index_versions = 3

[search]
# Advanced reranking with cross-encoder
rerank = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Multi-stage retrieval
multi_stage = true
multi_stage_config = { bm25_top_k = 1000, vector_top_k = 100, rerank_top_k = 10 }

[observability]
# Enable detailed telemetry
telemetry = true
# Log all searches to analytics
log_searches = true
# Prometheus metrics endpoint
metrics_port = 9090
# OpenTelemetry endpoint
otlp_endpoint = "http://localhost:4317"
```

---

## Rust Implementation Examples

### Example 1: Vector Store Trait

```rust
// src/vector_store.rs

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub repo_hash: String,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub chunk_hash: String,
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub limit: usize,
    pub filters: Vec<MetadataFilter>,
}

#[derive(Debug, Clone)]
pub enum MetadataFilter {
    Repo(String),
    Language(String),
    PathGlob(String),
    TimeRange { from: chrono::DateTime<chrono::Utc>, to: chrono::DateTime<chrono::Utc> },
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Initialize the vector store (create collections, check connectivity)
    async fn init(&mut self) -> Result<()>;

    /// Insert a batch of vectors
    async fn insert_batch(&self, records: Vec<VectorRecord>) -> Result<()>;

    /// Search for similar vectors
    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>>;

    /// Delete vectors by repo hash
    async fn delete_by_repo(&self, repo_hash: &str) -> Result<usize>;

    /// Get index statistics
    async fn stats(&self) -> Result<IndexStats>;

    /// Health check
    async fn health(&self) -> Result<bool>;
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: VectorMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    pub total_vectors: usize,
    pub total_repos: usize,
    pub dimension: usize,
    pub storage_bytes: u64,
}
```

### Example 2: Qdrant Implementation

```rust
// src/stores/qdrant_store.rs

use super::vector_store::*;
use anyhow::{Context, Result};
use qdrant_client::{
    client::QdrantClient,
    qdrant::{
        CreateCollection, Distance, PointStruct, SearchPoints, VectorParams,
        VectorsConfig, Filter, Condition, FieldCondition, Match,
    },
};
use std::collections::HashMap;

pub struct QdrantStore {
    client: QdrantClient,
    collection_prefix: String,
    dimension: usize,
}

impl QdrantStore {
    pub async fn new(url: &str, api_key: Option<String>, collection_prefix: String, dimension: usize) -> Result<Self> {
        let client = if let Some(key) = api_key {
            QdrantClient::new(Some(url), Some(&key), None, None, None)?
        } else {
            QdrantClient::new(Some(url), None, None, None, None)?
        };

        Ok(Self {
            client,
            collection_prefix,
            dimension,
        })
    }

    fn collection_name(&self, repo_hash: &str) -> String {
        format!("{}{}", self.collection_prefix, repo_hash)
    }

    async fn ensure_collection(&self, repo_hash: &str) -> Result<()> {
        let collection_name = self.collection_name(repo_hash);

        // Check if collection exists
        let collections = self.client.list_collections().await?;
        if collections.collections.iter().any(|c| c.name == collection_name) {
            return Ok(());
        }

        // Create collection with HNSW index
        self.client.create_collection(&CreateCollection {
            collection_name: collection_name.clone(),
            vectors_config: Some(VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::Params(VectorParams {
                    size: self.dimension as u64,
                    distance: Distance::Cosine.into(),
                    hnsw_config: Some(qdrant_client::qdrant::HnswConfigDiff {
                        m: Some(16),  // Number of edges per node
                        ef_construct: Some(100),  // Construction time accuracy
                        ..Default::default()
                    }),
                    quantization_config: None,  // Could enable scalar or binary quantization
                    on_disk: Some(false),  // Keep in memory for speed
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await
        .with_context(|| format!("Failed to create collection {}", collection_name))?;

        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn init(&mut self) -> Result<()> {
        // Test connectivity
        self.client.health_check().await
            .context("Qdrant health check failed")?;
        Ok(())
    }

    async fn insert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Group records by repo
        let mut by_repo: HashMap<String, Vec<VectorRecord>> = HashMap::new();
        for record in records {
            by_repo.entry(record.metadata.repo_hash.clone())
                .or_insert_with(Vec::new)
                .push(record);
        }

        // Insert into each repo's collection
        for (repo_hash, repo_records) in by_repo {
            self.ensure_collection(&repo_hash).await?;
            let collection_name = self.collection_name(&repo_hash);

            let points: Vec<PointStruct> = repo_records
                .into_iter()
                .map(|record| {
                    let mut payload = HashMap::new();
                    payload.insert("repo_hash".to_string(), record.metadata.repo_hash.into());
                    payload.insert("file_path".to_string(), record.metadata.file_path.into());
                    payload.insert("start_line".to_string(), (record.metadata.start_line as i64).into());
                    payload.insert("end_line".to_string(), (record.metadata.end_line as i64).into());
                    payload.insert("language".to_string(), record.metadata.language.into());
                    payload.insert("chunk_hash".to_string(), record.metadata.chunk_hash.into());
                    payload.insert("indexed_at".to_string(), record.metadata.indexed_at.to_rfc3339().into());

                    PointStruct {
                        id: Some(record.id.into()),
                        vectors: Some(record.vector.into()),
                        payload,
                    }
                })
                .collect();

            self.client.upsert_points(&collection_name, points, None).await
                .with_context(|| format!("Failed to insert into collection {}", collection_name))?;
        }

        Ok(())
    }

    async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // Extract repo filter
        let repo_hash = query.filters.iter()
            .find_map(|f| match f {
                MetadataFilter::Repo(hash) => Some(hash.clone()),
                _ => None,
            })
            .context("Repo hash required for search")?;

        let collection_name = self.collection_name(&repo_hash);

        // Build filters
        let mut conditions = Vec::new();
        for filter in &query.filters {
            match filter {
                MetadataFilter::Language(lang) => {
                    conditions.push(Condition::Field(FieldCondition {
                        key: "language".to_string(),
                        r#match: Some(Match::from(lang.clone())),
                        ..Default::default()
                    }));
                }
                MetadataFilter::PathGlob(pattern) => {
                    // Note: Qdrant doesn't support glob matching directly
                    // Would need to implement client-side filtering or use text match
                    // For now, skip or implement as prefix match
                }
                _ => {}
            }
        }

        let filter = if conditions.is_empty() {
            None
        } else {
            Some(Filter {
                must: conditions,
                ..Default::default()
            })
        };

        // Search
        let search_result = self.client.search_points(&SearchPoints {
            collection_name,
            vector: query.vector,
            limit: query.limit as u64,
            filter,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await
        .context("Search failed")?;

        // Convert results
        let results = search_result.result.into_iter()
            .map(|point| {
                let metadata = VectorMetadata {
                    repo_hash: point.payload.get("repo_hash")
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default(),
                    file_path: point.payload.get("file_path")
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default(),
                    start_line: point.payload.get("start_line")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0) as usize,
                    end_line: point.payload.get("end_line")
                        .and_then(|v| v.as_integer())
                        .unwrap_or(0) as usize,
                    language: point.payload.get("language")
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default(),
                    chunk_hash: point.payload.get("chunk_hash")
                        .and_then(|v| v.as_str().map(String::from))
                        .unwrap_or_default(),
                    indexed_at: point.payload.get("indexed_at")
                        .and_then(|v| v.as_str())
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&chrono::Utc))
                        .unwrap_or_else(chrono::Utc::now),
                };

                SearchResult {
                    id: point.id.unwrap().to_string(),
                    score: point.score,
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    async fn delete_by_repo(&self, repo_hash: &str) -> Result<usize> {
        let collection_name = self.collection_name(repo_hash);

        // Delete entire collection
        self.client.delete_collection(&collection_name).await?;

        Ok(0)  // Qdrant doesn't return count
    }

    async fn stats(&self) -> Result<IndexStats> {
        let collections = self.client.list_collections().await?;

        let mut total_vectors = 0;
        let total_repos = collections.collections.iter()
            .filter(|c| c.name.starts_with(&self.collection_prefix))
            .count();

        // Sum up vectors across all collections
        for collection in collections.collections {
            if collection.name.starts_with(&self.collection_prefix) {
                let info = self.client.collection_info(&collection.name).await?;
                if let Some(result) = info.result {
                    total_vectors += result.points_count.unwrap_or(0) as usize;
                }
            }
        }

        Ok(IndexStats {
            total_vectors,
            total_repos,
            dimension: self.dimension,
            storage_bytes: 0,  // Not easily available in Qdrant
        })
    }

    async fn health(&self) -> Result<bool> {
        self.client.health_check().await.map(|_| true)
    }
}
```

### Example 3: Cloud Embedder

```rust
// src/embedders/cloud_embedder.rs

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Clone)]
pub struct VoyageEmbedder {
    client: Client,
    api_key: String,
    model: String,
    dimension: usize,
    batch_size: usize,
}

#[derive(Serialize)]
struct VoyageRequest {
    input: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<bool>,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbedding>,
    usage: VoyageUsage,
}

#[derive(Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct VoyageUsage {
    total_tokens: usize,
}

impl VoyageEmbedder {
    pub fn new(api_key: String, model: String, dimension: usize, batch_size: usize) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            model,
            dimension,
            batch_size,
        }
    }

    async fn embed_batch_internal(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = VoyageRequest {
            input: texts.to_vec(),
            model: self.model.clone(),
            input_type: Some("document".to_string()),
            truncation: Some(true),
        };

        let response = self.client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Voyage AI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Voyage AI API error {}: {}", status, body);
        }

        let result: VoyageResponse = response.json().await
            .context("Failed to parse Voyage AI response")?;

        // Sort by index (API might return out of order)
        let mut embeddings: Vec<_> = result.data.into_iter()
            .map(|e| (e.index, e.embedding))
            .collect();
        embeddings.sort_by_key(|(idx, _)| *idx);

        let embeddings: Vec<Vec<f32>> = embeddings.into_iter()
            .map(|(_, emb)| emb)
            .collect();

        // Track usage (could send to metrics/logging)
        tracing::info!(
            tokens = result.usage.total_tokens,
            chunks = embeddings.len(),
            "Voyage AI embeddings generated"
        );

        Ok(embeddings)
    }

    async fn embed_batch_with_retry(&self, texts: &[String], max_retries: usize) -> Result<Vec<Vec<f32>>> {
        let mut last_error = None;

        for attempt in 0..max_retries {
            match self.embed_batch_internal(texts).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < max_retries - 1 {
                        let delay = Duration::from_secs(2u64.pow(attempt as u32));
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = max_retries,
                            delay_secs = delay.as_secs(),
                            "Voyage AI request failed, retrying..."
                        );
                        sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }
}

#[async_trait::async_trait]
impl super::BatchEmbedder for VoyageEmbedder {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Split into batches
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let embeddings = self.embed_batch_with_retry(chunk, 3).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
```

---

## CLI Command Examples

### Basic Usage

```bash
# Install sgrep
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh

# First search (auto-indexes)
cd my-project
sgrep search "authentication middleware"

# Manual indexing
sgrep index

# Watch mode (incremental updates)
sgrep watch
```

### Configuration

```bash
# Show current configuration
sgrep config show

# Interactive setup wizard
sgrep config init
# → Walks through: embedding mode, storage backend, API keys

# Set specific values
sgrep config set embedding.mode hybrid
sgrep config set storage.backend qdrant
sgrep config set storage.qdrant_url http://localhost:6333

# Validate configuration
sgrep config validate
# → Tests: Qdrant connectivity, API keys, model availability

# Edit config file directly
sgrep config edit
# → Opens ~/.sgrep/config.toml in $EDITOR
```

### Hybrid Mode Setup

```bash
# 1. Start local Qdrant (one-time)
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 2. Get Voyage API key (sign up at voyageai.com)
export VOYAGE_API_KEY="voy_..."

# 3. Configure sgrep for hybrid mode
sgrep config set embedding.mode hybrid
sgrep config set embedding.cloud_model voyage-code-3
sgrep config set storage.backend qdrant

# 4. Re-index with cloud embeddings
sgrep index --force
# → "Indexing with Voyage Code-3 (1024d)..."
# → "Stored 1,234 chunks in Qdrant"
# → "Cost: 185K tokens = $0.19"

# 5. Search (fast local, cloud-reranked)
sgrep search "retry logic with exponential backoff"
# → "[hybrid] Local search: 87ms, Cloud rerank: 124ms"
# → "10 results (total: 211ms)"
```

### Advanced Features

```bash
# Dry-run (estimate cost without indexing)
sgrep index --dry-run
# → "Estimated: 50K chunks, 7.5M tokens = $0.75"

# Show usage stats
sgrep stats
# → Total repos: 12
# → Total chunks: 45,678
# → Storage: 234 MB (local) + 1.2 GB (Qdrant)
# → API usage this month: 89M tokens = $8.90

# Show cost breakdown
sgrep stats costs
# → Nov 2025:
# →   Embedding API: 89M tokens = $8.90
# →   Qdrant: Self-hosted = $0
# →   Total: $8.90 / $10 budget
# →   Free tier remaining: 111M tokens

# Multi-repo search (workspace mode)
sgrep search "payment processing" --workspace
# → Searches all repos in workspace
# → Groups results by repo

# Delete repo from index
sgrep delete --repo ~/work/old-project
# → Removes from local + Qdrant

# Upgrade embeddings
sgrep upgrade
# → "Found 8 repos with 384d embeddings"
# → "Upgrade to 768d? [Y/n]"
# → Re-indexes all repos with new model

# Export/import index
sgrep export --output my-index.tar.gz
sgrep import --input my-index.tar.gz --to ~/new-machine/.sgrep/
```

---

## Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  # Optional: Prometheus for monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  # Optional: Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  qdrant_storage:
  prometheus_data:
  grafana_data:
```

Usage:
```bash
# Start Qdrant (and optionally monitoring stack)
docker-compose up -d qdrant

# Configure sgrep to use it
sgrep config set storage.backend qdrant
sgrep config set storage.qdrant_url http://localhost:6333

# Index and search
sgrep index
sgrep search "your query"
```

---

## Environment Variables

```bash
# Embedding Configuration
export VOYAGE_API_KEY="voy_..."              # Voyage AI API key
export OPENAI_API_KEY="sk-..."               # OpenAI API key (alternative)
export SGREP_EMBEDDING_MODE="hybrid"         # local | hybrid | cloud
export SGREP_CLOUD_MODEL="voyage-code-3"     # Cloud embedding model
export SGREP_LOCAL_MODEL="nomic-embed-v1.5"  # Local embedding model

# Storage Configuration
export SGREP_STORAGE_BACKEND="qdrant"        # local | qdrant | pinecone
export QDRANT_URL="http://localhost:6333"    # Qdrant connection URL
export QDRANT_API_KEY="..."                  # Qdrant Cloud API key (optional)

# Performance Tuning
export SGREP_BATCH_SIZE="256"                # Embedding batch size
export SGREP_EMBEDDER_POOL_SIZE="8"          # Parallel embedder instances
export SGREP_CLOUD_BATCH_SIZE="100"          # Cloud API batch size
export SGREP_DEVICE="auto"                   # cpu | cuda | coreml | auto

# Privacy & Security
export SGREP_PRIVATE_REPOS="~/work/client-*,~/personal/secrets-*"
export SGREP_REQUIRE_CLOUD_CONSENT="false"   # Require explicit opt-in

# Observability
export RUST_LOG="sgrep=info,qdrant_client=warn"
export SGREP_TELEMETRY="true"                # Enable telemetry
export SGREP_METRICS_PORT="9090"             # Prometheus metrics port

# Cost Controls
export SGREP_COST_LIMIT_MONTHLY="50.00"      # Max monthly spend (USD)
export SGREP_WARN_AT_PERCENT="80"            # Warn at 80% of limit
```

---

## Testing & Validation

### Integration Tests

```bash
# Test local mode
sgrep config set embedding.mode local
sgrep index tests/fixtures/sample-repo
sgrep search "test query" --path tests/fixtures/sample-repo
# → Verify: Results returned, no network calls

# Test Qdrant connectivity
docker run -d -p 6333:6333 qdrant/qdrant
sgrep config set storage.backend qdrant
sgrep config validate
# → Verify: "✓ Qdrant reachable at localhost:6333"

# Test hybrid mode (requires API key)
export VOYAGE_API_KEY="voy_..."
sgrep config set embedding.mode hybrid
sgrep index tests/fixtures/sample-repo --force
# → Verify: "Using Voyage Code-3 for indexing"
sgrep search "test query"
# → Verify: "[hybrid] Local search + cloud rerank"

# Test fallback behavior
docker stop <qdrant-container-id>
sgrep search "test query"
# → Verify: "[warn] Qdrant unreachable, falling back to local"

# Test cost estimation
sgrep index tests/fixtures/large-repo --dry-run
# → Verify: Shows token count and estimated cost
```

### Performance Benchmarks

```bash
# Benchmark indexing speed
time sgrep index --force

# Benchmark search latency
for i in {1..100}; do
  time sgrep search "random query $i" > /dev/null
done | awk '{sum+=$2; count++} END {print "Avg:", sum/count, "ms"}'

# Benchmark hybrid vs local
sgrep config set embedding.mode local
hyperfine --warmup 3 'sgrep search "auth middleware"'

sgrep config set embedding.mode hybrid
hyperfine --warmup 3 'sgrep search "auth middleware"'
```

---

## Migration Guide

### From Local to Hybrid

```bash
# 1. Backup existing indexes
cp -r ~/.sgrep/indexes ~/.sgrep/indexes.backup

# 2. Start Qdrant
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 3. Configure hybrid mode
export VOYAGE_API_KEY="voy_..."
sgrep config set embedding.mode hybrid
sgrep config set storage.backend qdrant

# 4. Migrate indexes
sgrep migrate --from local --to qdrant
# → Re-embeds all repos with cloud model
# → Uploads to Qdrant
# → Keeps local backup for 30 days

# 5. Verify
sgrep search "test query"
# → Should show "[hybrid]" indicator

# 6. Clean up old indexes (after testing)
rm -rf ~/.sgrep/indexes.backup
```

### Rollback from Hybrid to Local

```bash
# 1. Switch back to local mode
sgrep config set embedding.mode local
sgrep config set storage.backend local

# 2. Re-index locally (if needed)
sgrep index --force

# 3. Stop Qdrant (optional)
docker stop <qdrant-container-id>
```

---

## Monitoring & Observability

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sgrep'
    static_configs:
      - targets: ['localhost:9090']
```

Exposed metrics:
- `sgrep_searches_total` - Total searches performed
- `sgrep_search_duration_seconds` - Search latency histogram
- `sgrep_embedding_tokens_total` - Total tokens sent to API
- `sgrep_embedding_cost_usd_total` - Total API cost (USD)
- `sgrep_index_size_bytes` - Index size per repo
- `sgrep_qdrant_operations_total` - Qdrant operations (insert, search, delete)

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "sgrep Monitoring",
    "panels": [
      {
        "title": "Search Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, sgrep_search_duration_seconds_bucket)"
        }]
      },
      {
        "title": "API Cost (Daily)",
        "targets": [{
          "expr": "increase(sgrep_embedding_cost_usd_total[1d])"
        }]
      },
      {
        "title": "Searches per Minute",
        "targets": [{
          "expr": "rate(sgrep_searches_total[1m])"
        }]
      }
    ]
  }
}
```

---

*End of Implementation Examples*
