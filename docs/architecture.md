# Architecture

How sgrep works under the hood.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         sgrep                                    │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   Indexer    │   Searcher   │   Watcher    │   CLI             │
├──────────────┴──────────────┴──────────────┴───────────────────┤
│                     Core Components                              │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   Chunker    │   Embedder   │   Graph      │   Store           │
│ (tree-sitter)│   (ONNX)     │  (symbols)   │   (local/remote)  │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

## Indexing Pipeline

### 1. File Discovery

```
Repository → Git-aware walker → File list
```

- Respects `.gitignore`
- Detects git worktrees and inherits parent indexes
- Parallel directory traversal (up to 8 threads)

### 2. Chunking

```
Files → Tree-sitter parsing → Semantic chunks
```

**Strategy:**
- Parse each file with language-specific tree-sitter grammar
- Split at logical boundaries (functions, classes, blocks)
- Target chunk size: ~512 tokens
- Preserve context overlap between chunks

**Supported Languages:**
Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, C#, Ruby, Markdown, JSON, YAML, TOML, HTML, CSS, Bash

### 3. Symbol Extraction

```
AST → Symbol extractor → Code graph
```

**Extracted:**
- Symbol definitions (functions, classes, variables)
- Symbol relationships (calls, imports, extends)
- File-to-symbol mappings

### 4. Embedding

```
Chunks → Embedder → 384-dim vectors
```

**Local (default):**
- Model: `jina-embeddings-v2-base-code`
- Runtime: ONNX with CPU/CoreML/CUDA backends
- Batch size: Configurable (default auto-tuned)

**Modal.dev (optional):**
- Model: `Qwen3-Embedding-8B`
- Output: Truncated to 384 dims for compatibility
- GPU acceleration: T4, A10G, or L40S

### 5. Storage

```
Vectors + Metadata → Store → Index files
```

**Local storage:**
- Binary format with optional mmap
- Hierarchical index (chunks + directories)
- Location: `~/.sgrep/indexes/{repo-hash}/`

**Remote storage:**
- Pinecone or Turbopuffer
- Namespace per repository
- 384-dim vectors with metadata

---

## Search Pipeline

### 1. Query Processing

```
Query → Tokenize → Embed → Query vector
```

Optional query expansion with Qwen2.5:
- Analyzes query intent
- Generates related terms
- Expands search scope

### 2. Candidate Retrieval

Three strategies based on index size:

**Small indexes (< 500 vectors):**
```
Query vector → Linear scan → All similarities
```

**Medium indexes (500-1000 vectors):**
```
Query vector → HNSW index → Top-k candidates
```

**Large indexes (> 1000 vectors):**
```
Query vector → Binary quantization → Shortlist → Exact scoring
```

### 3. Hybrid Scoring

```
Candidates → BM25F + Semantic → Fused scores
```

**BM25F (keyword):**
- Term frequency with field boosts
- Fields: content, filename, path, symbols
- Normalized to [0, 1]

**Semantic (vector):**
- Cosine similarity
- Already in [0, 1]

**Fusion:**
- Adaptive weighting based on query characteristics
- Short queries → more weight on semantics
- Exact matches → more weight on BM25F

### 4. Pseudo-Relevance Feedback

```
Top-10 results → Extract terms → Expand query → Re-score
```

Adds up to 5 terms from top results to capture domain vocabulary.

### 5. Deduplication

```
Results → Near-duplicate detection → Unique results
```

Suppresses similar chunks from same or different files.

---

## Watch Mode

### Event Loop

```
Filesystem events → Debounce → Incremental update
```

1. **notify** crate watches for changes
2. Events debounced (default 500ms)
3. Dirty set tracks changed files
4. Incremental re-index on dirty files only

### Process Management

- Can run detached (`--detach`)
- PID tracking in `~/.sgrep/watch-pids/`
- Graceful shutdown on SIGTERM

---

## Storage Format

### Local Index Structure

```
~/.sgrep/
├── config.toml           # Global configuration
├── indexes/
│   └── {repo-hash}/
│       ├── index.bin     # Chunk embeddings + metadata
│       ├── hierarchy.bin # Directory embeddings
│       └── graph.bin     # Symbol graph
└── cache/
    └── fastembed/        # Model weights
```

### Index Binary Format

```
Header:
  magic: u32
  version: u32
  chunk_count: u64
  embedding_dim: u32

Chunks:
  [for each chunk]
    path: String
    start_line: u32
    end_line: u32
    content: String (optional)
    embedding: [f32; 384]
```

### Remote Storage

**Pinecone:**
- Vectors stored with metadata
- Namespace = repo hash (configurable)
- Queries use metadata filtering

**Turbopuffer:**
- Similar schema to Pinecone
- Region-specific endpoints
- Namespace prefix for multi-repo

---

## Memory Management

### Embedding Model

- Loaded once, shared across operations
- ONNX runtime manages GPU memory
- Optional pooled embedder for concurrency

### Index Loading

**Small indexes:** Fully loaded to memory
**Large indexes:** Memory-mapped (mmap)
- OS manages page cache
- Zero-copy access
- Reduced memory footprint

### Caching

| Cache | Scope | Purpose |
|-------|-------|---------|
| BM25F index | Per-search | Avoid rebuilding keyword index |
| HNSW index | Per-session | Reuse approximate index |
| Embedding model | Global | One-time load |

---

## Threading Model

```
Main thread
├── Rayon pool (bulk parallelism)
│   ├── File walking
│   ├── Chunking
│   └── Batch processing
├── ONNX threads (model inference)
│   └── Capped at 4 threads
└── Walker threads (filesystem)
    └── Capped at 8 threads
```

### CPU Presets

| Preset | Rayon threads |
|--------|---------------|
| auto | Available cores |
| background | 10% of cores |
| low | 25% of cores |
| medium | 50% of cores |
| high | 75% of cores |

---

## Error Handling

### Graceful Degradation

- Missing symbols → Search works, BM25F less accurate
- Query expander unavailable → Standard search
- Remote timeout → Retry with backoff

### Recovery

- Corrupted index → `sgrep index --force`
- Missing model → Auto-download on next run
- Stale watch → Restart watch process

---

## Extension Points

### Embedding Providers

Trait: `BatchEmbedder`
- `embed_batch(texts) -> Vec<Vec<f32>>`

Implementations:
- `LocalEmbedder` (ONNX)
- `ModalEmbedder` (HTTP)

### Storage Backends

Trait: `VectorStore`
- `upsert(vectors, metadata)`
- `query(vector, k) -> results`
- `delete(ids)`

Implementations:
- `LocalStore`
- `PineconeStore`
- `TurbopufferStore`

