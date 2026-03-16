# Architecture

How `sgrep` works under the hood.

## Overview

`sgrep` is a local semantic search pipeline with four major stages:

1. parse and chunk source files
2. embed chunks with a local model
3. persist the local index
4. rank results with semantic + keyword signals

## Main modules

- `src/chunker/*` — chunk generation and parser reuse
- `src/embedding/*` — local embedding runtime and provider selection
- `src/indexer/*` — full and incremental indexing
- `src/search/*` — local search, scoring, dedup, and result rendering
- `src/watch/mod.rs` — watch mode for incremental updates

## Indexing pipeline

### 1. File discovery

Repository → git-aware walker → file list

### 2. Chunking

Files → tree-sitter parsing → semantic chunks

### 3. Symbol extraction

AST → symbol extractor → code graph

### 4. Embedding

Chunks → local embedder → 384-dimensional vectors

### 5. Storage

Vectors + metadata → local index files under `~/.sgrep/indexes/{repo-hash}/`

## Search pipeline

- embed the query locally
- retrieve candidates from the local index
- combine semantic and keyword scores
- deduplicate nearby results
- render snippets or JSON output

## Watch mode

Filesystem events are debounced and turned into incremental local re-indexes.

## Memory and threading

- the embedding model is loaded locally and shared across operations
- large indexes can use memory-mapped access
- threading is controlled via `SGREP_MAX_THREADS` and `SGREP_CPU_PRESET`
- ONNX and BLAS thread counts are constrained to avoid oversubscription
