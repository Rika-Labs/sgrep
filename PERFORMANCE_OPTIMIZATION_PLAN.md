# SGREP Performance Optimization Plan
## Goal: Index 50k Files in Under 60 Seconds

**Target Throughput:** 833+ files/second
**Current Performance:** ~333 files/second for large repos (10K-100K files)
**Required Improvement:** 2.5x speedup

---

## Executive Summary

Based on codebase analysis and industry research, embedding generation consumes 60-80% of indexing time. To achieve sub-60-second indexing for 50K files, we need aggressive optimizations across:

1. **Embedding Pipeline** (highest impact): Larger batches, GPU acceleration, better parallelization
2. **Chunking Strategy** (medium impact): Skip expensive parsing for simple files
3. **I/O & Serialization** (low impact): Memory-mapped I/O, faster compression
4. **Caching** (medium impact): Persistent cross-repo cache, better deduplication

---

## Current Performance Profile

### Bottleneck Analysis (from src/indexer.rs profiling)

| Phase | % of Time | Current Strategy | Parallelized? |
|-------|-----------|------------------|---------------|
| **Embedding** | 60-80% | PooledEmbedder (8 instances max) + rayon batches | ✅ Yes |
| **Chunking** | 5-15% | Tree-sitter + rayon parallel iteration | ✅ Yes |
| **File Walking** | 1-5% | `ignore` crate parallel walk | ✅ Yes |
| **Serialization** | 1-3% | Bincode + zstd compression | ❌ No |

### Current Optimization Features

✅ **Already Implemented:**
- Pooled embedders (one per CPU core, max 8)
- Rayon parallel chunking and batch processing
- BLAKE3 hash-based deduplication
- Moka LRU cache (50K entries)
- Adaptive batch sizing (256 CPU, 512 GPU)
- Token budget control (~6K tokens/batch)
- BGE-small-en-v1.5-q model (384-dim, 33M params, 24MB)

---

## Optimization Strategy

### Phase 1: Aggressive Embedding Optimization (Target: 1.5-2x speedup)

#### 1.1 Increase Default Batch Sizes
**Current:** 256 (CPU), 512 (GPU)
**Target:** 512 (CPU), 2048 (GPU)

**Rationale:** Modern embedding models benefit from larger batches. GitHub's 2025 embedding model update "doubles throughput speed" through better batching. BGE-small is lightweight enough to handle larger batches.

**Implementation:**
- `src/indexer.rs:734-769`: Update `DEFAULT_CPU_BATCH_SIZE` and `DEFAULT_GPU_BATCH_SIZE`
- Add adaptive logic: detect available VRAM for GPU, RAM for CPU
- Add `--aggressive-batching` flag for power users

**Risk:** Higher memory usage. Mitigate with RAM checks.

---

#### 1.2 Optimize Pooled Embedder Distribution
**Current:** Round-robin with atomic counter
**Target:** Work-stealing queue with load balancing

**Rationale:** Some batches have fewer tokens (faster), others max out. Round-robin can lead to idle workers while one worker processes a large batch.

**Implementation:**
- Replace `AtomicUsize` counter in `src/embedding.rs:169-172` with `crossbeam::deque::WorkStealing`
- Each embedder instance pulls from shared work queue when idle
- Benchmark with criterion to validate improvement

**Expected Gain:** 10-20% reduction in embedding time for uneven batch distributions

---

#### 1.3 GPU Batch Pipelining
**Current:** Sequential batch processing (wait for batch N before starting batch N+1)
**Target:** Overlap CPU prep with GPU inference

**Rationale:** While GPU computes embeddings for batch N, CPU can tokenize batch N+1.

**Implementation:**
- Add async batch preparation in `src/indexer.rs:268-307`
- Use `tokio` or `async-std` to pipeline:
  1. Thread 1: Prepare batch N+1 (tokenization)
  2. Thread 2: GPU inference on batch N
  3. Thread 3: Post-process batch N-1 (cache writes)
- Only activate when `SGREP_DEVICE=cuda|coreml`

**Expected Gain:** 15-25% reduction in embedding time on GPU

---

#### 1.4 Persistent Embedding Cache
**Current:** In-memory Moka cache (lost on restart)
**Target:** On-disk cache with LRU eviction

**Rationale:** Common code patterns (e.g., `if __name__ == "__main__"`, boilerplate) appear across repos. Persistent cache enables cross-repo deduplication.

**Implementation:**
- Use `cacache` or `sled` for persistent KV store
- Key: `BLAKE3(chunk_text)` (already computed)
- Value: `Vec<f32>` (embedding vector)
- Location: `~/.sgrep/cache/embeddings.db`
- Max size: 1GB (configurable via `SGREP_CACHE_SIZE_MB`)
- Eviction: LRU based on access time

**Expected Gain:** 20-40% speedup on repos with common patterns (web frameworks, etc.)

---

### Phase 2: Chunking Optimizations (Target: 0.3-0.5x speedup)

#### 2.1 Fast-Path Chunking for Simple Files
**Current:** Always run tree-sitter parser
**Target:** Detect simple files and skip parsing

**Rationale:** Files like JSON, YAML, Markdown don't benefit from semantic parsing. Tree-sitter overhead (~1-5ms/file) is wasted.

**Implementation:**
- `src/chunker.rs:70-106`: Add pre-check for "simple" languages
- Simple languages: JSON, YAML, TOML, Markdown, CSS
- Use line-based chunking directly (existing fallback logic)
- Add heuristic: if file <100 lines, skip tree-sitter

**Expected Gain:** 5-10% reduction in chunking time

---

#### 2.2 Chunk Size Tuning
**Current:** Max 200 lines OR 2048 chars
**Target:** Optimize for embedding model context window

**Rationale:** BGE-small has a 512 token context window. Average code is ~4 chars/token. Current 2048 char limit = ~512 tokens (perfect!). But 200 line limit might be too restrictive.

**Implementation:**
- Increase line limit to 300 lines (keep 2048 char limit)
- Add telemetry to track chunk size distribution
- Benchmark different limits (150, 200, 250, 300 lines)

**Expected Gain:** 5-15% fewer chunks = faster embedding

---

### Phase 3: I/O & Serialization (Target: 0.1-0.2x speedup)

#### 3.1 Parallel Compression
**Current:** Single-threaded zstd compression
**Target:** Multi-threaded compression

**Implementation:**
- `src/store.rs:48`: Use `zstd::stream::Encoder::new()` with `.multithread(num_cpus::get())`
- Already supported by `zstd` crate!

**Expected Gain:** 2-5% reduction in write time

---

#### 3.2 Memory-Mapped Index Loading
**Current:** Read entire `index.bin.zst` into memory, decompress, deserialize
**Target:** Use `memmap2` for lazy loading

**Rationale:** Search only needs metadata + vectors. Chunks can be lazily loaded.

**Implementation:**
- Split index into two files:
  - `index_meta.bin` (metadata + vectors) - small, always loaded
  - `index_chunks.bin` (chunk text) - large, mmap'd
- Update `src/store.rs:58-77` to load separately
- Search reads from mmap only when showing results

**Expected Gain:** Faster search startup (not indexing, but improves UX)

---

### Phase 4: Advanced Optimizations (Target: 0.2-0.4x speedup)

#### 4.1 SIMD-Optimized Vector Operations
**Current:** SimSIMD integrated (commit 8a4d9d2) but unclear if actively used
**Target:** Explicitly use SimSIMD for all vector ops

**Implementation:**
- Audit `src/search.rs:50-107` (cosine similarity)
- Replace manual dot product with `simsimd::SpatialSimilarity::cosine()`
- Benchmark AVX-512 vs AVX2 vs scalar

**Expected Gain:** 5-10% faster search (not indexing)

---

#### 4.2 Incremental Indexing Parallelization
**Current:** `build_incremental` processes files sequentially
**Target:** Parallelize with rayon (like `build_full`)

**Implementation:**
- `src/indexer.rs:368-593`: Refactor incremental path
- Use `.par_iter()` on modified file list
- Challenge: Need to carefully handle index updates (thread-safe writes)

**Expected Gain:** 2-3x faster incremental updates (not initial indexing)

---

#### 4.3 Pre-Warming Model Instances
**Current:** Models loaded lazily on first batch
**Target:** Pre-load all pooled embedders in parallel

**Implementation:**
- `src/embedding.rs:151-167`: Add `.par_iter()` to model initialization
- Warm up ONNX runtime with dummy batch (some frameworks optimize after first run)

**Expected Gain:** 1-2 seconds saved on first indexing run

---

## Benchmark Suite Design

### Test Repositories (Verifiable, Reproducible)

#### Small Repo (1K files)
- **Source:** Clone `actix/actix-web` (Rust web framework)
- **Files:** ~1,200 Rust files
- **Target:** <5 seconds
- **Test:** `time sgrep index --force`

#### Medium Repo (10K files)
- **Source:** Clone `microsoft/vscode` (TypeScript)
- **Files:** ~9,800 TS/JS files
- **Target:** <20 seconds
- **Test:** `time sgrep index --force`

#### Large Repo (50K files)
- **Source:** Clone `chromium/chromium` (mirror) or generate synthetic repo
- **Files:** 50,000 files
- **Target:** <60 seconds ⭐
- **Test:** `time sgrep index --force`

#### Synthetic Benchmark (Controlled)
- **Generator:** Script to create N files of varying sizes/languages
- **Files:** Configurable (1K, 10K, 50K, 100K)
- **Languages:** Mix of Rust (30%), Python (30%), JS (20%), Go (10%), other (10%)
- **Purpose:** Eliminate network/git variance

---

### Benchmark Infrastructure

#### Directory Structure
```
benches/
├── bench_indexing.rs          # Criterion benchmarks
├── bench_search.rs            # Search benchmarks
├── repos/
│   ├── download_repos.sh      # Clone test repos
│   └── generate_synthetic.py  # Generate synthetic codebases
└── results/
    └── baseline.json          # Baseline measurements
```

#### Benchmark Harness (Criterion)

**File:** `benches/bench_indexing.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use sgrep::indexer::Indexer;
use std::path::PathBuf;

fn benchmark_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");

    for size in [1_000, 10_000, 50_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let repo_path = PathBuf::from(format!("benches/repos/synthetic_{}", size));
                b.iter(|| {
                    let indexer = Indexer::new(&repo_path);
                    indexer.build_full(/* ... */);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_indexing);
criterion_main!(benches);
```

#### CI/CD Integration

**GitHub Actions Workflow:** `.github/workflows/benchmark.yml`

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest-8core  # Consistent hardware
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Download test repos
        run: cd benches && ./repos/download_repos.sh

      - name: Run benchmarks
        run: cargo bench --bench bench_indexing

      - name: Compare to baseline
        run: |
          cargo install critcmp
          critcmp --export baseline < benches/results/baseline.json
          cargo bench --bench bench_indexing
          critcmp new baseline

      - name: Assert 50K target
        run: |
          # Extract 50K indexing time
          TIME=$(grep "synthetic_50000" target/criterion/indexing/*/new/estimates.json | jq .mean.point_estimate)
          # Assert < 60 seconds (60_000_000_000 ns)
          if (( $(echo "$TIME > 60000000000" | bc -l) )); then
            echo "❌ Failed: 50K indexing took ${TIME}ns (>60s)"
            exit 1
          fi
          echo "✅ Passed: 50K indexing took ${TIME}ns (<60s)"
```

---

### Benchmark Metrics to Track

#### Primary Metrics (Guarantee in README)
1. **Indexing Throughput:** Files/second for 1K, 10K, 50K repos
2. **Absolute Time:** Total seconds for 50K files (must be <60s)
3. **Search Latency (P95):** 95th percentile search time

#### Secondary Metrics (Internal)
4. **Embedding Time:** % of total time spent in embedding
5. **Cache Hit Rate:** % of chunks served from cache
6. **Memory Peak:** Max RSS during indexing
7. **CPU Utilization:** Average % across cores
8. **GPU Utilization:** If CUDA/CoreML enabled

#### Regression Detection
- Run benchmarks on every PR
- Compare to `main` baseline
- Fail CI if >10% regression on primary metrics
- Use `critcmp` for statistical significance testing

---

## Verification & Testing Strategy

### Unit Tests (Correctness)

**File:** `src/indexer.rs` (expand existing tests)

```rust
#[test]
fn test_large_repo_correctness() {
    // Index 10K files, verify all chunks present
    let repo = create_synthetic_repo(10_000);
    let index = build_full(&repo);
    assert_eq!(index.chunks.len(), expected_chunks);
    assert_eq!(index.vectors.len(), index.chunks.len());
}

#[test]
fn test_aggressive_batching_equivalence() {
    // Same results with batch_size=256 vs 2048
    let repo = create_test_repo();
    let index_small = build_with_batch_size(&repo, 256);
    let index_large = build_with_batch_size(&repo, 2048);
    assert_embeddings_equivalent(&index_small, &index_large, epsilon=0.01);
}

#[test]
fn test_persistent_cache_correctness() {
    // Verify cache hits produce identical embeddings
    let chunk = "fn main() {}";
    let embedding1 = embed_with_cache(chunk);
    clear_memory_cache();
    let embedding2 = embed_with_cache(chunk); // Should hit disk cache
    assert_eq!(embedding1, embedding2);
}
```

### Integration Tests (End-to-End)

**File:** `tests/integration_benchmarks.rs`

```rust
#[test]
fn test_50k_files_under_60s() {
    let repo = download_or_generate_50k_repo();
    let start = Instant::now();

    Command::new("sgrep")
        .args(&["index", "--force", repo.path()])
        .status()
        .expect("indexing failed");

    let duration = start.elapsed();
    assert!(
        duration.as_secs() < 60,
        "Indexing 50K files took {}s (expected <60s)",
        duration.as_secs()
    );
}

#[test]
fn test_search_quality_after_optimization() {
    // Ensure optimizations don't degrade search quality
    index_test_repo();
    let results = search_query("authentication middleware");
    assert!(results[0].score > 0.8);
    assert!(results[0].path.contains("auth"));
}
```

### Performance Test Suite

**File:** `benches/throughput_test.rs`

```rust
// Not a benchmark, a test that asserts performance SLAs
#[test]
fn assert_throughput_sla() {
    for (file_count, max_seconds) in [(1_000, 5), (10_000, 20), (50_000, 60)] {
        let repo = get_or_generate_repo(file_count);
        let duration = time_full_index(&repo);

        assert!(
            duration.as_secs() <= max_seconds,
            "{} files: {}s (SLA: {}s)",
            file_count, duration.as_secs(), max_seconds
        );
    }
}
```

---

## README Marketing Plan

### Performance Section Update

**Before:**
```markdown
| Repo Size | Index Time | Search P95 |
|-----------|------------|------------|
| <1K files | <5s | <50 ms |
| 1K–10K | <30s | <120 ms |
| 10K–100K | <5m | <250 ms |
```

**After:**
```markdown
## ⚡ Performance Benchmarks

**Guaranteed Throughput:** 833+ files/second
**Verified on:** Ubuntu 22.04, 8-core Intel i7, 16GB RAM

| Repo Size | Index Time | Throughput | Search P95 | Verified |
|-----------|------------|------------|-----------|----------|
| 1K files | <3s | 333+ files/s | <30 ms | ✅ [actix-web](benches/repos/actix-web) |
| 10K files | <15s | 666+ files/s | <80 ms | ✅ [vscode](benches/repos/vscode) |
| **50K files** | **<60s** | **833+ files/s** | <200 ms | ✅ [chromium-mirror](benches/repos/chromium) |
| 100K files | <2m | 833+ files/s | <300 ms | ✅ Synthetic benchmark |

> 💡 **Tip:** Performance scales with CPU cores. Results from 8-core system; 16-core systems see proportionally faster indexing.

### How We Achieved This

- **Aggressive Batching:** Process 2048 chunks/batch on GPU, 512 on CPU
- **Work-Stealing Parallelism:** Dynamic load balancing across embedder pool
- **Persistent Caching:** Cross-repo deduplication saves 20-40% on common code
- **Smart Chunking:** Skip expensive parsing for simple files (JSON, YAML)
- **SIMD Everything:** AVX-512 accelerated vector operations
- **Zero-Copy I/O:** Memory-mapped index loading

See [PERFORMANCE_OPTIMIZATION_PLAN.md](PERFORMANCE_OPTIMIZATION_PLAN.md) for full technical details.

### Run the Benchmarks Yourself

```bash
# Clone sgrep
git clone https://github.com/rika-labs/sgrep
cd sgrep

# Download test repositories
cd benches && ./repos/download_repos.sh && cd ..

# Run full benchmark suite
cargo bench --bench bench_indexing

# Quick check: 50K file SLA
cargo test --release test_50k_files_under_60s
```

All benchmarks run in CI on every commit: [View Latest Results](https://github.com/rika-labs/sgrep/actions/workflows/benchmark.yml)
```

---

## Implementation Roadmap

### Sprint 1: Foundational Optimizations (Week 1-2)
1. ✅ Increase batch sizes (1.1)
2. ✅ Parallel compression (3.1)
3. ✅ Fast-path chunking (2.1)
4. ✅ Set up benchmark infrastructure
5. ✅ Create baseline measurements

**Deliverable:** 1.3-1.5x speedup, benchmark harness ready

---

### Sprint 2: Advanced Parallelism (Week 3-4)
1. ✅ Work-stealing queue for embedders (1.2)
2. ✅ GPU batch pipelining (1.3)
3. ✅ Pre-warming models (4.3)
4. ✅ Chunk size tuning experiments (2.2)

**Deliverable:** 1.8-2.2x cumulative speedup

---

### Sprint 3: Caching & Polish (Week 5-6)
1. ✅ Persistent embedding cache (1.4)
2. ✅ Memory-mapped index loading (3.2)
3. ✅ Incremental indexing parallelization (4.2)
4. ✅ SIMD audit and optimization (4.1)
5. ✅ Final benchmark runs on 50K repo

**Deliverable:** 2.5x+ cumulative speedup, <60s for 50K files

---

### Sprint 4: Testing & Documentation (Week 7)
1. ✅ Comprehensive test suite (unit + integration)
2. ✅ CI/CD benchmark automation
3. ✅ README updates with verified benchmarks
4. ✅ Performance tuning guide for users
5. ✅ Blog post: "How We Made sgrep Index 50K Files in Under a Minute"

**Deliverable:** Production-ready release, marketing materials

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Memory pressure from large batches** | High | Add RAM checks, adaptive batch sizing, OOM fallback |
| **GPU not available on all systems** | Medium | Keep CPU path optimized, auto-detect hardware |
| **Cache corruption** | Low | Versioned cache with integrity checks, auto-rebuild |
| **Regression in search quality** | High | Integration tests asserting search quality metrics |
| **Platform-specific issues** | Medium | Test on Linux, macOS (Intel + ARM), Windows in CI |

---

## Success Criteria

### Must-Have (Release Blockers)
- ✅ 50K files indexed in <60 seconds on 8-core system
- ✅ All existing tests pass
- ✅ No regression in search quality (measured by integration tests)
- ✅ Memory usage <4GB for 50K file indexing
- ✅ CI benchmarks run on every PR

### Nice-to-Have (Post-Release)
- 100K files in <2 minutes
- GPU acceleration documentation
- Distributed indexing (multiple machines)
- Incremental search (search while indexing)

---

## References

### Research Sources
- [ripgrep performance techniques](https://burntsushi.net/ripgrep/) - SIMD, literal optimization, adaptive strategies
- [GitHub 2025 embedding model](https://www.infoq.com/news/2025/10/github-embedding-model/) - 37.6% retrieval improvement, 2x throughput, Matryoshka learning
- [LlamaIndex parallel processing](https://milvus.io/ai-quick-reference/how-does-llamaindex-support-parallel-processing-for-largescale-indexing) - Async operations, distributed computing, data chunking
- [Qodo-Embed-1](https://www.qodo.ai/blog/qodo-embed-1-code-embedding-code-retrieval/) - State-of-the-art code embeddings, batch processing optimization
- [Modal: Best Code Embedding Models](https://modal.com/blog/6-best-code-embedding-models-compared) - Comprehensive comparison of code embedding approaches

### Internal Documentation
- `src/indexer.rs:143-165` - Profiling infrastructure
- `src/embedding.rs:127-201` - PooledEmbedder implementation
- Recent commits: d0aae55 (pooling), 7fcc256 (BGE-small), 8a4d9d2 (SIMD)

---

**Last Updated:** 2025-11-24
**Owner:** Performance Team
**Status:** Ready for Implementation
