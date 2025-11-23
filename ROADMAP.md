# sgrep Development Roadmap

This document provides a comprehensive overview of implemented features and planned enhancements for sgrep. It serves as a reference for contributors, users, and stakeholders to understand the current state and future direction of the project.

## Status Legend

- ✅ **Implemented** - Feature is complete and available in the current release
- 🔄 **In Progress** - Feature is currently under active development
- 📋 **Planned** - Feature is documented and scheduled for future implementation
- 🔍 **Under Consideration** - Feature is being evaluated for inclusion

## Core Commands

### Implemented ✅

- [x] **`sgrep search`** - Semantic code search engine supporting natural language queries with hybrid ranking (semantic similarity, keyword matching, recency)
- [x] **`sgrep index`** - Manual repository indexing with full re-index capability via `--force` flag
- [x] **`sgrep watch`** - File system watcher with configurable debounce for incremental index updates

### Planned 📋

- [ ] **`sgrep serve`** - HTTP API endpoint providing sub-50ms search responses for integration with external tools and services
- [ ] **`sgrep list`** - Repository management command displaying indexed repositories, index sizes, and last update timestamps
- [ ] **`sgrep doctor`** - Diagnostic utility for verifying embedding model availability, index integrity, and system health

## Search Capabilities

### Implemented ✅

- [x] **Semantic Search** - Vector similarity search using cosine similarity on embedded query and code chunks
- [x] **Keyword Matching** - Full-text search with intelligent stopword filtering and relevance scoring
- [x] **Recency Boost** - Temporal ranking factor prioritizing recently modified files
- [x] **Hybrid Scoring** - Weighted combination algorithm (60% semantic, 30% keyword, 10% recency)
- [x] **Glob Pattern Filtering** - File path filtering via `--glob` flag with support for multiple patterns
- [x] **Metadata Filters** - Language and path-based filtering using `--filters` flag (e.g., `lang=rust`, `path=src`)
- [x] **Result Pagination** - Configurable result limit via `-n, --limit` flag
- [x] **Context Display** - Full chunk content rendering via `-c, --context` flag

### Planned 📋

- [ ] **JSON Output Format** - Structured JSON response format via `--json` flag for programmatic consumption and agent integration
- [ ] **Extended Metadata** - Enhanced result structure including comprehensive chunk metadata (path, line ranges, language, modification timestamps)

## Indexing & Code Analysis

### Implemented ✅

- [x] **Tree-sitter Integration** - Syntax-aware code chunking using tree-sitter parsers for:
  - Rust
  - Python
  - JavaScript
  - TypeScript
  - TSX
  - Go
- [x] **Fallback Chunking** - Line-based chunking strategy for unsupported languages and parsing failures
- [x] **Semantic Boundary Detection** - Intelligent code segmentation at function, class, and module boundaries
- [x] **Chunk Size Constraints** - Enforced limits (200 lines, 2048 characters) to optimize embedding quality
- [x] **Git Integration** - Respects `.gitignore` patterns for intelligent file exclusion
- [x] **Repository Isolation** - Automatic per-repository index isolation using path-based hashing
- [x] **Progress Reporting** - Real-time indexing progress indicators with file and chunk counts
- [x] **Force Re-index** - Complete index rebuild capability via `--force` flag

### Planned 📋

- [ ] **Automatic Indexing** - Transparent index creation on first search operation (currently requires explicit `sgrep index` invocation)
- [ ] **Content Deduplication** - Detection and single-embedding of identical code blocks to reduce storage and improve performance
- [ ] **Intelligent File Filtering** - Automatic exclusion of binary files, lockfiles, and minified assets during indexing
- [ ] **Incremental Indexing** - Differential index updates processing only modified files since last index

## Embedding System

### Implemented ✅

- [x] **Hashed Embeddings** - Deterministic, transformer-inspired embedding generation using cryptographic hashing
- [x] **Vector Dimensions** - Fixed 512-dimensional embedding vectors
- [x] **Embedding Cache** - In-memory cache supporting up to 50,000 entries for performance optimization
- [x] **Vector Normalization** - L2-normalized vectors ensuring consistent similarity calculations

### Planned 📋

- [ ] **ONNX Backend** - Optional ONNX runtime integration for production-grade embedding models (feature flag exists in `Cargo.toml`, implementation pending)
- [ ] **Configurable Dimensions** - Runtime-configurable embedding vector dimensions
- [ ] **Multi-Backend Support** - Pluggable embedding backend architecture supporting multiple embedding providers

## Performance & Resource Management

### Implemented ✅

- [x] **Parallel Processing** - Rayon-powered multi-threaded indexing and search operations
- [x] **Index Compression** - Zstandard (zstd) compression for efficient index storage
- [x] **Binary Serialization** - Bincode serialization format for fast index I/O operations
- [x] **Concurrency Control** - Environment variable (`RAYON_NUM_THREADS`) for manual thread pool sizing

### Planned 📋

- [ ] **Adaptive Throttling** - Dynamic concurrency adjustment based on system resource utilization (RAM, CPU)
- [ ] **Memory Pressure Monitoring** - Real-time memory usage tracking with automatic backpressure mechanisms
- [ ] **Thermal Management** - CPU temperature-aware throttling for mobile and laptop devices
- [ ] **Index Optimization** - Storage efficiency improvements through advanced compression and indexing strategies

## Configuration & Customization

### Implemented ✅

- [x] **Environment Variables** - Configuration via `SGREP_HOME`, `RUST_LOG`, and `RAYON_NUM_THREADS`
- [x] **Data Directory Detection** - Automatic discovery of user data directory (`~/.sgrep/` by default)
- [x] **Repository Auto-detection** - Automatic repository path resolution from current working directory

### Planned 📋

- [ ] **Configuration File** - `sgrep.toml` configuration file for persistent settings
- [ ] **Exclusion Patterns** - User-defined file and path exclusion patterns
- [ ] **Embedding Backend Selection** - Configuration-driven embedding provider selection
- [ ] **Concurrency Limits** - Per-operation concurrency limit configuration
- [ ] **Custom Index Locations** - Repository-specific index path overrides

## Storage & Index Management

### Implemented ✅

- [x] **Index Storage** - Hierarchical index storage under `~/.sgrep/indexes/<hash>/`
- [x] **Compressed Index Format** - Zstandard-compressed binary index files (`index.bin.zst`)
- [x] **Index Metadata** - Comprehensive metadata tracking (version, repository path, hash, indexing timestamps, statistics)
- [x] **Repository Isolation** - Hash-based repository identification ensuring index separation

### Planned 📋

- [ ] **Index Versioning** - Version-aware index format with automatic migration capabilities
- [ ] **Index Maintenance** - Utilities for index cleanup, pruning, and optimization
- [ ] **Health Monitoring** - Index integrity checks and validation tools
- [ ] **Cross-Repository Search** - Unified search across multiple indexed repositories

## Developer Experience

### Implemented ✅

- [x] **Command-Line Interface** - Intuitive CLI with descriptive error messages and help text
- [x] **Progress Indicators** - Visual progress bars and status updates during long-running operations
- [x] **Colored Output** - Syntax-highlighted terminal output for improved readability
- [x] **Observability** - Comprehensive tracing and logging support via `RUST_LOG` environment variable

### Planned 📋

- [ ] **Enhanced Error Messages** - Context-aware error messages with actionable suggestions
- [ ] **Index Validation Tools** - Utilities for detecting and repairing corrupted indexes
- [ ] **Performance Benchmarks** - Built-in benchmarking tools for performance regression testing
- [ ] **Integration Examples** - Sample code and documentation for coding agent integrations

## Testing & Quality Assurance

### Implemented ✅

- [x] **Unit Tests** - Comprehensive unit test coverage for core chunking logic
- [x] **Embedding Tests** - Unit tests verifying embedding normalization and cache behavior
- [x] **Search Tests** - Unit tests for keyword extraction and filter logic

### Planned 📋

- [ ] **Integration Tests** - End-to-end integration tests covering full indexing and search workflows
- [ ] **Fuzz Testing** - Fuzz testing for robust input handling
- [ ] **Performance Regression Tests** - Automated performance tracking in CI

