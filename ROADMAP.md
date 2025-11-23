# sgrep Product Roadmap

**Vision:** The fastest, most accurate local semantic code search tool, purpose-built for developers and AI coding agents.

**Current Version:** 0.1.6
**Target 1.0:** Q2 2025

---

## Version 1.0 - Production Ready (Target: Q2 2025)

**Goal:** Battle-tested, production-ready semantic code search with comprehensive documentation and stability guarantees.

### 1.0.0 - Core Stability ✅ (Released)
- ✅ Real semantic embeddings (BGE-small-en-v1.5-q, 384-dim)
- ✅ SIMD-accelerated cosine similarity (SimSIMD)
- ✅ Batch embedding generation (256 chunks/batch)
- ✅ Built-in ignore patterns (.git, node_modules, etc.)
- ✅ Tree-sitter based code chunking (15+ languages)
- ✅ Hybrid search (70% semantic, 20% keyword, 10% recency)
- ✅ Repository auto-isolation with hash-based indexing
- ✅ File system watcher with incremental updates

### 1.0.1 - Performance & Quality
**Status:** 🔄 In Progress

#### Indexing Performance
- [ ] **Parallel Batch Processing** - Process multiple batches concurrently
  - Current: Sequential batch processing
  - Target: 3-5x speedup with parallel ONNX sessions

- [ ] **GPU Acceleration** (Optional) - CUDA/Metal support for embedding generation
  - Detect available GPU via ONNX Runtime
  - Fallback to CPU if unavailable
  - Target: 10-20x speedup on GPU

- [ ] **Incremental Indexing** - Only re-embed changed files
  - Track file hashes and modification times
  - Skip unchanged files during re-index
  - Target: 95% reduction in re-index time

#### Search Performance
- [ ] **Memory-Mapped Index Loading** - Zero-copy index access via `mmap`
  - Use `memmap2` + `rkyv` for instant index loading
  - Current: ~100ms load time
  - Target: <1ms load time

- [ ] **Index Warming** - Pre-load frequently accessed indexes
  - Background index loading on startup
  - Smart caching of recent searches

### 1.0.2 - Developer Experience
**Status:** 📋 Planned (Q1 2025)

- [ ] **JSON Output** - Structured output for agent integration
  ```bash
  sgrep search "auth logic" --json
  ```

- [ ] **Automatic Indexing** - Index on first search (no manual `sgrep index`)

- [ ] **Progress Improvements**
  - Show embedding progress (X/Y embeddings generated)
  - Estimated time remaining
  - Network progress for model downloads

- [ ] **Better Error Messages**
  - Actionable suggestions ("Run `sgrep index` first")
  - Index corruption auto-recovery
  - Model download retry with exponential backoff

### 1.0.3 - Configuration System
**Status:** 📋 Planned (Q1 2025)

- [ ] **Configuration File** - `.sgrep.toml` for persistent settings
  ```toml
  [search]
  default_limit = 20
  semantic_weight = 0.7

  [embedding]
  model = "BGESmallENV15Q"
  batch_size = 256

  [indexing]
  exclude_patterns = ["*.lock", "dist/"]
  max_chunk_size = 200
  ```

- [ ] **Per-Repository Config** - `.sgrepignore` files

- [ ] **Environment Variables** - Full env var support
  - `SGREP_MODEL` - Override default embedding model
  - `SGREP_THREADS` - Control parallelism
  - `SGREP_CACHE_SIZE` - Embedding cache size

---

## Version 1.1 - Advanced Search (Target: Q2 2025)

**Goal:** Enterprise-grade search capabilities with advanced filtering and ranking.

### 1.1.0 - Query Language
- [ ] **Advanced Query Syntax**
  ```bash
  sgrep search "auth AND (jwt OR oauth)" --lang rust
  sgrep search "function:login path:auth/"
  sgrep search "NOT test" --exclude-path tests/
  ```

- [ ] **Regex Support** - Combine semantic + regex
  ```bash
  sgrep search "login" --regex "fn \w+_login"
  ```

- [ ] **Fuzzy Search** - Typo-tolerant search

- [ ] **Multi-Query Search** - OR multiple semantic queries

### 1.1.1 - Ranking Improvements
- [ ] **BM25 Keyword Scoring** - Replace simple keyword matching
  - TF-IDF with document frequency
  - Length normalization
  - Target: 20-30% better keyword recall

- [ ] **Learning to Rank** - User feedback loop
  - Track clicked results
  - Adjust weights based on usage
  - Personalized ranking over time

- [ ] **Custom Ranking Weights** - User-defined scoring
  ```bash
  sgrep search "auth" --semantic 0.8 --keyword 0.1 --recency 0.1
  ```

### 1.1.2 - Search Features
- [ ] **Cross-Repository Search** - Search multiple repos at once
  ```bash
  sgrep search "auth" --repos "~/work/*"
  ```

- [ ] **Search History** - Recent searches with caching

- [ ] **Search Suggestions** - "Did you mean...?" suggestions

- [ ] **Related Results** - "Similar to this result" exploration

---

## Version 1.2 - Massive Scale (Target: Q3 2025)

**Goal:** Handle repositories with millions of lines of code efficiently.

### 1.2.0 - HNSW Vector Index
- [ ] **Approximate Nearest Neighbor Search** - Logarithmic search time
  - Integrate `hnswlib-rs` for large indexes (>10K chunks)
  - Automatic fallback to linear search for small indexes
  - Target: <100ms search on 1M chunks

- [ ] **Quantized Vectors** - Compress embeddings for storage
  - f32 → f16: 50% size reduction, minimal accuracy loss
  - f32 → i8: 75% size reduction, <5% accuracy loss
  - Product quantization for 90%+ compression

- [ ] **Disk-Based Index** - Support indexes larger than RAM
  - Use `mmap` for on-demand page loading
  - LRU cache for hot chunks
  - Target: 10GB+ indexes with 1GB RAM

### 1.2.1 - Distributed Indexing
- [ ] **Parallel Model Instances** - Multiple ONNX sessions
  - One session per core for maximum throughput
  - Lock-free batch queue
  - Target: 10-20 chunks/second per core

- [ ] **Streaming Indexing** - Process while indexing
  - Start search before indexing completes
  - Incremental index updates
  - Live progress with partial results

### 1.2.2 - Index Optimization
- [ ] **Automatic Index Pruning** - Remove stale/deleted files

- [ ] **Index Defragmentation** - Optimize storage layout

- [ ] **Multi-Version Indexes** - Keep historical indexes
  - Git commit-based snapshots
  - Time-travel search ("Find in version X")

---

## Version 1.3 - AI Agent Integration (Target: Q4 2025)

**Goal:** Best-in-class integration with AI coding agents (Cursor, Copilot, Claude Code, etc.)

### 1.3.0 - API Server
- [ ] **HTTP API Server** - `sgrep serve`
  ```bash
  sgrep serve --port 8080
  # GET /search?q=auth&limit=10
  # POST /index
  # GET /health
  ```

- [ ] **WebSocket Support** - Real-time search streaming

- [ ] **gRPC API** - High-performance RPC interface

- [ ] **Rate Limiting** - Protect against abuse

### 1.3.1 - Agent Protocol
- [ ] **MCP Server** - Model Context Protocol integration
  - Native Cursor/Claude integration
  - Tool definitions for search, index, watch

- [ ] **LSP Extension** - Language Server Protocol support
  - Inline semantic search in editors
  - Hover definitions with semantic context

- [ ] **OpenAPI Spec** - Full API documentation
  - Swagger UI for testing
  - Client library generation

### 1.3.2 - Agent Features
- [ ] **Contextual Search** - Search with conversation context
  - Accept conversation history
  - Personalized results based on context

- [ ] **Code Explanation** - Semantic code documentation
  - "Explain this function"
  - Generate docstrings from code

- [ ] **Diff-Aware Search** - Search in uncommitted changes
  - Index working directory changes
  - Find modified code semantically

---

## Version 1.4 - Multi-Modal Search (Target: Q1 2026)

**Goal:** Search beyond code - documentation, images, diagrams.

### 1.4.0 - Documentation Search
- [ ] **Markdown Embedding** - Semantic search in docs
  - README, wikis, documentation sites
  - Code-doc alignment (link functions to docs)

- [ ] **Comment Extraction** - Index inline comments
  - Docstrings, JSDoc, Rustdoc
  - Link comments to code semantically

### 1.4.1 - Visual Search
- [ ] **Image Search** - Find diagrams, screenshots
  - Use CLIP for image-text embeddings
  - "Find architecture diagram"

- [ ] **Code Screenshot OCR** - Index code from images
  - Extract code from screenshots
  - Useful for tutorial/book content

### 1.4.2 - Cross-Modal Search
- [ ] **Unified Search** - Query across all modalities
  - "Show me auth code and its documentation"
  - "Find diagram explaining this function"

---

## Version 2.0 - Enterprise Features (Target: Q2 2026)

**Goal:** Team collaboration, security, and compliance features.

### 2.0.0 - Team Features
- [ ] **Shared Indexes** - Team-wide index sharing
  - Central index server
  - Push/pull index updates
  - Conflict resolution

- [ ] **Access Control** - Permission-based search
  - Respect repository permissions
  - Filter results by access level

- [ ] **Analytics** - Search usage tracking
  - Popular queries
  - Slow searches
  - Index health metrics

### 2.0.1 - Security
- [ ] **Secret Detection** - Never index secrets
  - Detect API keys, tokens, passwords
  - Redact from embeddings
  - Alert on sensitive code patterns

- [ ] **Compliance** - GDPR, SOC2 support
  - Data retention policies
  - Right to deletion
  - Audit logs

### 2.0.2 - Cloud Integration
- [ ] **Cloud Sync** - Sync indexes to S3/GCS
  - Backup and restore
  - Cross-device sync

- [ ] **Managed Service** - Hosted sgrep (optional)
  - No local setup required
  - Always-on indexing
  - API access from anywhere

---

## Performance Targets

| Repository Size | Index Time | Search Latency | Memory Usage |
|-----------------|------------|----------------|--------------|
| Small (<1K files) | <10s | <50ms | <100MB |
| Medium (1K-10K) | <1min | <100ms | <500MB |
| Large (10K-100K) | <10min | <200ms | <2GB |
| Massive (100K+) | <30min | <500ms | <5GB |

## Model Roadmap

| Version | Model | Size | Speed | Quality | Status |
|---------|-------|------|-------|---------|--------|
| 0.1.6 | BGE-small-en-v1.5-q | 24MB | Fast | Good | ✅ Current |
| 1.1.0 | Nomic-embed-code | 140MB | Medium | Excellent | 📋 Code-specific |
| 1.2.0 | Custom-trained | 50MB | Fast | Excellent | 🔍 Research |
| 1.4.0 | CLIP (multi-modal) | 350MB | Medium | Good | 📋 Images+code |

## Architecture Evolution

### Current (0.1.6)
```
File → Tree-sitter → Chunks → Batch Embed → Index → Linear Search
```

### Near Future (1.0.x)
```
File → Tree-sitter → Chunks → Parallel Batch Embed → mmap Index → SIMD Search
```

### Advanced (1.2.x)
```
File → Tree-sitter → Chunks → GPU Batch Embed → HNSW Index → ANN Search
```

### Enterprise (2.0.x)
```
File → Multi-modal Parse → Distributed Embed → Sharded Index → Cloud Sync
```

---

## Contributing

We welcome contributions! Priority areas:

1. **Performance** - Make indexing and search faster
2. **Model Research** - Test and benchmark new embedding models
3. **Language Support** - Add tree-sitter parsers
4. **Agent Integration** - Build agent protocol implementations
5. **Testing** - Expand test coverage

## Community Feedback

Have ideas? Open an issue or discussion:
- **Feature Requests:** [GitHub Issues](https://github.com/rika-labs/sgrep/issues)
- **Performance Ideas:** Tag with `performance`
- **Integration Requests:** Tag with `agents` or `api`

---

**Last Updated:** November 2025
**Next Review:** Q1 2025
