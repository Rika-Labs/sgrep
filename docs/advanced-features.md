# Advanced Features

sgrep includes several sophisticated features that work behind the scenes to improve search quality. This guide explains what they do and how to leverage them.

## Hybrid Ranking Pipeline

sgrep doesn't just do vector search. It combines multiple ranking signals:

```
Query → BM25F (keywords) ─┐
      → Embeddings ───────┼→ Adaptive Fusion → Reranker → Results
      → Symbol Boost ─────┘
```

### 1. BM25F Keyword Scoring

Traditional keyword search with field boosting. Terms found in filenames and symbol names get higher weights than terms in code bodies.

**What it catches:** Exact function names, variable names, specific error messages.

### 2. Semantic Embeddings

Dense vector representations that capture meaning beyond exact matches.

**What it catches:** Conceptual queries like "authentication" finding OAuth, JWT, and login code.

### 3. Symbol Boost

Extracted symbol names (functions, classes, variables) get additional weight in BM25F scoring.

**What it catches:** Queries for "handleClick" finding the actual handler even in large files.

### 4. Cross-Encoder Reranking

A secondary model that deeply analyzes query-document pairs for final ranking.

**What it catches:** Subtle relevance that simpler models miss.

```bash
# See all score components
sgrep search --debug "your query"
```

---

## Code Graph Extraction

sgrep builds a graph of symbols and their relationships during indexing.

### Extracted Information

**Symbol Types:**
- Functions, Methods
- Classes, Structs, Interfaces
- Modules, Enums
- Variables, Constants, Types

**Relationships:**
- Calls (function A calls function B)
- Implements (class implements interface)
- Extends (class extends parent)
- Imports (file imports module)
- Returns, UsesType

### Supported Languages

Full tree-sitter parsing for:

| Language | Extensions | Symbol Extraction |
|----------|------------|-------------------|
| Rust | `.rs` | Full |
| Python | `.py` | Full |
| JavaScript | `.js`, `.jsx` | Full |
| TypeScript | `.ts`, `.tsx` | Full |
| Go | `.go` | Full |
| Java | `.java` | Full |
| C | `.c`, `.h` | Full |
| C++ | `.cpp`, `.hpp`, `.cc` | Full |
| C# | `.cs` | Full |
| Ruby | `.rb` | Full |
| Markdown | `.md` | Headings |
| JSON | `.json` | Keys |
| YAML | `.yaml`, `.yml` | Keys |
| TOML | `.toml` | Keys |
| HTML | `.html` | Tags |
| CSS | `.css` | Selectors |
| Bash | `.sh`, `.bash` | Functions |

### Viewing Graph Stats

```bash
sgrep index --stats
# Output includes:
#   Graph symbols: 1,234
#   Graph edges: 5,678
```

---

## Search Strategy Selection

sgrep automatically picks the best algorithm based on your index size:

| Index Size | Strategy | Why |
|------------|----------|-----|
| < 500 vectors | Linear scan | HNSW overhead not worth it |
| 500 - 1000 | HNSW index | Approximate nearest neighbors |
| > 1000 | Binary quantization | 32x memory reduction, ~5% recall loss |

### Binary Quantization

For large codebases, sgrep compresses embeddings to 1-bit representations:
- 32x memory reduction
- 10x faster initial shortlisting
- Followed by exact scoring on top candidates

This activates automatically. No configuration needed.

---

## Query Expansion (Experimental)

sgrep can expand your queries using a local LLM to improve recall.

### How It Works

1. Analyzes your query for intent
2. Generates related terms and synonyms
3. Expands the search to include related concepts

### Model

Uses [Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) (~400MB download on first use).

### Activation

Query expansion activates automatically when:
- The model is cached locally
- Not running in offline mode

---

## Pseudo-Relevance Feedback (PRF)

After initial results, sgrep extracts terms from top hits to refine the query.

**Parameters (internal):**
- Top 10 results analyzed
- Up to 5 expansion terms added

This happens automatically on every search.

---

## Deduplication

sgrep suppresses near-duplicate results from:
- Similar chunks in the same file
- Copy-pasted code across files

This prevents results from being dominated by repeated patterns.

---

## Hierarchical Indexing

Beyond chunk-level embeddings, sgrep builds directory-level representations.

**Benefits:**
- Faster navigation to relevant directories
- Better understanding of project structure
- Improved ranking for broad queries

---

## Threading & Performance

### CPU Presets

Control resource usage:

```bash
sgrep index --cpu-preset auto       # Detect and use available cores
sgrep index --cpu-preset low        # ~25% of cores
sgrep index --cpu-preset medium     # ~50% of cores
sgrep index --cpu-preset high       # ~75% of cores
sgrep index --cpu-preset background # ~10% of cores
```

### Thread Distribution

sgrep manages multiple thread pools:
- **Rayon** - Parallel iteration (bulk of work)
- **ONNX** - Model inference (capped at 4 threads)
- **Walker** - Filesystem traversal (capped at 8 threads)

### Environment Variables

Fine-grained control:

```bash
# Core threading
export SGREP_MAX_THREADS=4
export RAYON_NUM_THREADS=4

# ONNX runtime
export ORT_INTRA_OP_NUM_THREADS=2
export ORT_INTER_OP_NUM_THREADS=2

# BLAS backends (if used)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

---

## Caching

### BM25F Index Cache

The keyword index is cached per-repository to avoid rebuilding on every search.

### Embedding Cache

Model weights cached in `~/.sgrep/cache/fastembed/`.

### HNSW Index

Built on first search, reused until index changes.

---

## JSON Output Schema

For agent integration, use `--json`:

```bash
sgrep search --json "query"
```

**Response structure:**

```json
{
  "query": "your query",
  "elapsed_ms": 123,
  "index_metadata": {
    "total_chunks": 1000,
    "total_files": 50,
    "embedding_dim": 384
  },
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

**Score fields (with `--debug`):**
- `score` - Final combined score
- `semantic_score` - Vector similarity (0-1)
- `bm25_score` - Keyword relevance (normalized)

---

## Index Statistics

Detailed stats without rebuilding:

```bash
sgrep index --stats
```

**Output includes:**
- Total chunks, files, directories
- Graph symbols and edges
- Index size on disk
- Embedding dimensions

Machine-readable:

```bash
sgrep index --stats --json
```

---

## Profiling

See where time goes during indexing:

```bash
sgrep index --profile
```

**Output:**
```
Phase timings:
  walk:   234ms
  chunk:  567ms
  embed:  3.2s
  graph:  456ms
  write:  123ms

Cache hit rate: 78%
```

---

## Memory-Mapped Indexes

For large indexes, sgrep uses memory mapping for zero-copy access:
- Faster startup (no full load)
- Lower memory footprint
- OS manages caching

This activates automatically for large indexes.

---

## Glob Patterns

Fine-grained file filtering:

```bash
# Include patterns
sgrep search "query" --glob "src/**/*.rs"
sgrep search "query" --glob "*.ts" --glob "*.tsx"

# Exclude patterns (prefix with !)
sgrep search "query" --glob "!node_modules/**"
sgrep search "query" --glob "!**/*.test.js"

# Combine
sgrep search "query" --glob "src/**" --glob "!**/*.spec.ts"
```

---

## Metadata Filters

Filter by extracted metadata:

```bash
# By language
sgrep search "query" --filters lang=rust

# Multiple filters
sgrep search "query" --filters lang=python --filters type=function
```

Available filters depend on extracted metadata.
