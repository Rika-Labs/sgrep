# Implementation Roadmap: 50K Files in <60s

**Status:** Planning Complete ✅
**Next Step:** Begin Sprint 1 Implementation
**Target Completion:** 4-6 weeks

---

## Quick Start Guide for Developers

### Prerequisites
```bash
# Ensure Rust is installed
rustc --version  # Should be 1.75+

# Generate test repositories
cd benches/repos
./download_repos.sh
cd ../..

# Verify current baseline
cargo test --release --test performance_sla sla_50k_files_under_60s -- --ignored --nocapture
```

### Current Baseline (Before Optimizations)
- 10K files: ~30s (333 files/s)
- 50K files: ~150s (333 files/s) ❌ **FAILS SLA**

### Target (After Optimizations)
- 10K files: <15s (666+ files/s)
- 50K files: <60s (833+ files/s) ✅ **MEETS SLA**

---

## Implementation Priority Matrix

Optimizations ranked by **Impact × Ease of Implementation**:

| Priority | Optimization | Impact | Effort | Files to Modify | Est. Time |
|----------|--------------|--------|--------|-----------------|-----------|
| 🔴 **P0** | Increase batch sizes | High (1.5x) | Low | `src/indexer.rs` | 2 hours |
| 🔴 **P0** | Parallel compression | Medium (0.1x) | Low | `src/store.rs` | 1 hour |
| 🟡 **P1** | Persistent embedding cache | High (0.4x) | Medium | `src/embedding.rs`, `src/store.rs` | 1 week |
| 🟡 **P1** | Work-stealing embedder pool | Medium (0.2x) | Medium | `src/embedding.rs` | 3 days |
| 🟡 **P1** | Fast-path chunking | Medium (0.1x) | Low | `src/chunker.rs` | 1 day |
| 🟢 **P2** | GPU batch pipelining | High (0.25x) | High | `src/indexer.rs`, `src/embedding.rs` | 1 week |
| 🟢 **P2** | Chunk size tuning | Medium (0.15x) | Low | `src/chunker.rs`, benchmarks | 2 days |
| 🟢 **P2** | Pre-warm models | Low (0.05x) | Low | `src/embedding.rs` | 1 day |
| ⚪ **P3** | Memory-mapped index | Low (search only) | Medium | `src/store.rs`, `src/search.rs` | 1 week |
| ⚪ **P3** | SIMD optimization audit | Low (search only) | Medium | `src/search.rs` | 3 days |
| ⚪ **P3** | Incremental parallel | Low (not initial) | Medium | `src/indexer.rs` | 1 week |

**Total estimated cumulative speedup:** 2.5-3x (meets 50K in <60s target)

---

## Sprint Breakdown

### Sprint 1: Quick Wins (Week 1)
**Goal:** Achieve 1.6x speedup with minimal risk

#### Tasks
1. ✅ **Increase Batch Sizes**
   - File: `src/indexer.rs:734-769`
   - Change: `DEFAULT_CPU_BATCH_SIZE = 256 → 512`
   - Change: `DEFAULT_GPU_BATCH_SIZE = 512 → 2048`
   - Add: RAM detection to prevent OOM
   - Test: Verify no regression in search quality

2. ✅ **Enable Parallel Compression**
   - File: `src/store.rs:48`
   - Change: Add `.multithread(num_cpus::get())` to zstd encoder
   - Test: Verify compression ratio unchanged

3. ✅ **Fast-Path Chunking**
   - File: `src/chunker.rs:70-106`
   - Add: Pre-check for "simple" languages (JSON, YAML, Markdown, CSS)
   - Change: Skip tree-sitter for simple files, use line-based chunking
   - Test: Verify chunk counts are similar

4. ✅ **Establish Benchmark Baseline**
   - Run: `cargo bench --bench bench_indexing` to capture baseline
   - Save: Results to `benches/results/baseline.json`
   - Document: Current performance in git commit message

**Deliverable:** 1.6x speedup (50K files in ~94s)
**Success Criteria:** All tests pass, no search quality regression

---

### Sprint 2: Caching Infrastructure (Week 2-3)
**Goal:** Add persistent cache for 20-40% speedup on common code

#### Tasks
1. ✅ **Design Cache Schema**
   - Key: `BLAKE3(chunk_text)` (already computed in indexer)
   - Value: `Vec<f32>` (384-dim embedding)
   - Storage: `cacache` or `sled` at `~/.sgrep/cache/embeddings/`
   - Size limit: 1GB default (configurable via `SGREP_CACHE_SIZE_MB`)
   - Eviction: LRU based on access time

2. ✅ **Implement Cache Layer**
   - File: `src/embedding.rs`
   - Add: `PersistentCache` struct wrapping `cacache::Cache`
   - Add: Methods: `get(hash) -> Option<Vec<f32>>`, `set(hash, vec)`
   - Integrate: Check cache before calling ONNX model
   - Add: Cache hit/miss telemetry

3. ✅ **Add Cache Management Commands**
   - Command: `sgrep cache stats` (show size, hit rate, entries)
   - Command: `sgrep cache clear` (wipe cache)
   - Update: `--help` documentation

4. ✅ **Test Cache Correctness**
   - Test: Verify cache hits produce identical embeddings
   - Test: Verify cache survives process restart
   - Test: Verify cache eviction works at size limit
   - Test: Verify concurrent access (multiple sgrep processes)

**Deliverable:** 2.0x cumulative speedup (50K files in ~75s)
**Success Criteria:** Cache hit rate >30% on second index of same repo

---

### Sprint 3: Advanced Parallelism (Week 3-4)
**Goal:** Optimize embedder pool for 2.3x cumulative speedup

#### Tasks
1. ✅ **Work-Stealing Embedder Pool**
   - File: `src/embedding.rs:169-172`
   - Replace: `AtomicUsize` counter with `crossbeam::deque::Stealer`
   - Add: Each embedder pulls from shared work queue
   - Benchmark: Compare to round-robin baseline
   - Test: Verify no deadlocks under load

2. ✅ **GPU Batch Pipelining** (if GPU available)
   - File: `src/indexer.rs:268-307`
   - Add: `tokio` or `async-std` for async batch prep
   - Design: 3-stage pipeline (prep N+1, GPU N, post-process N-1)
   - Add: Feature flag `gpu-pipelining` (optional)
   - Test: Verify speedup on CUDA/CoreML systems

3. ✅ **Pre-Warm Model Instances**
   - File: `src/embedding.rs:151-167`
   - Add: Parallel model initialization with `rayon`
   - Add: Dummy batch warmup for ONNX optimization
   - Test: Measure first-run latency reduction

4. ✅ **Chunk Size Tuning Experiments**
   - Experiment: Try 150, 200, 250, 300 line limits
   - Measure: Total chunks, embedding time, search quality
   - Choose: Optimal based on benchmark data
   - Update: `MAX_CHUNK_LINES` in `src/chunker.rs`

**Deliverable:** 2.3x cumulative speedup (50K files in ~65s)
**Success Criteria:** <70s on 50K files, passing all SLA tests

---

### Sprint 4: Final Push to <60s (Week 5)
**Goal:** Achieve guaranteed <60s for 50K files

#### Tasks
1. ✅ **Profile and Identify Remaining Bottlenecks**
   - Tool: `cargo flamegraph --test performance_sla -- sla_50k_files_under_60s --ignored`
   - Tool: `perf record -g` on Linux
   - Identify: Top 3 time consumers
   - Plan: Micro-optimizations as needed

2. ✅ **Micro-Optimizations**
   - Based on profiling results, could include:
     - Reduce allocations in hot paths
     - Optimize BLAKE3 hashing (already fast, but check)
     - Optimize tree-sitter parsing (memoization?)
     - Review serialization overhead

3. ✅ **Final Benchmark Run**
   - Run: `cargo bench --bench bench_indexing` on 1K, 10K, 50K repos
   - Run: `cargo test --release --test performance_sla -- --ignored --nocapture`
   - Verify: All SLA tests pass with margin (aim for <55s on 50K)
   - Document: Results in `benches/results/optimized.json`

4. ✅ **Compare Baseline vs Optimized**
   - Tool: `critcmp baseline optimized`
   - Generate: Performance comparison table
   - Create: Before/after graphs for blog post

**Deliverable:** 2.5x+ cumulative speedup (50K files in <60s) ✅
**Success Criteria:** Pass all SLA tests with 10% margin

---

### Sprint 5: Polish & Documentation (Week 6)
**Goal:** Production-ready release with marketing materials

#### Tasks
1. ✅ **Integration Test Coverage**
   - File: `tests/integration_benchmarks.rs`
   - Test: End-to-end indexing correctness
   - Test: Search quality not degraded
   - Test: Cache correctness
   - Test: Concurrent indexing (multiple repos)

2. ✅ **CI/CD Integration**
   - File: `.github/workflows/benchmark.yml` (already created)
   - Verify: Runs on every PR
   - Add: Comment PR with benchmark results
   - Add: Fail CI on >10% regression

3. ✅ **Update README with Benchmarks**
   - Source: `README_MARKETING_ADDITIONS.md`
   - Add: Performance table with verified results
   - Add: Badges (performance, CI status)
   - Add: "How We Achieved This" section
   - Add: Link to benchmark suite

4. ✅ **Write Blog Post**
   - Outline: Use `README_MARKETING_ADDITIONS.md` blog section
   - Include: Code snippets, flamegraphs, benchmark graphs
   - Publish: On personal blog / dev.to / Medium
   - Share: HN, Reddit r/rust, Twitter/X

5. ✅ **Create Performance Tuning Docs**
   - File: `docs/PERFORMANCE_TUNING.md` (already created)
   - Document: All env vars and their effects
   - Add: Common troubleshooting scenarios
   - Add: Hardware-specific recommendations

**Deliverable:** Production release v0.2.0 with performance guarantees
**Success Criteria:** All docs updated, CI passing, positive community feedback

---

## Development Workflow

### Branch Strategy
- `main`: Stable release branch
- `perf/sprint-1`: Quick wins (batch sizes, compression, fast-path)
- `perf/sprint-2`: Caching infrastructure
- `perf/sprint-3`: Advanced parallelism
- `perf/sprint-4`: Final optimizations
- Merge: Each sprint into `main` after SLA verification

### Testing Protocol
1. **Before any change:**
   ```bash
   cargo test --release --test performance_sla -- --ignored --nocapture > baseline.txt
   ```

2. **After each optimization:**
   ```bash
   cargo test --release --test performance_sla -- --ignored --nocapture > optimized.txt
   diff baseline.txt optimized.txt  # Should show improvement
   ```

3. **Before merging to main:**
   ```bash
   cargo test --all  # All unit tests
   cargo test --release --test performance_sla -- --ignored  # All SLA tests
   cargo bench --bench bench_indexing  # Criterion benchmarks
   ```

### Benchmarking Protocol
- **Environment:** Consistent hardware (same machine/VM for all benchmarks)
- **State:** Close all other applications
- **Warmup:** Run once before measuring (warm up disk cache)
- **Samples:** Minimum 5 runs, report median and P95
- **Comparison:** Use `critcmp` for statistical significance

---

## Risk Mitigation

### High-Risk Changes
1. **Work-stealing queue:** Could introduce deadlocks
   - **Mitigation:** Extensive load testing, timeout mechanisms
   - **Rollback:** Keep round-robin as fallback (`SGREP_USE_WORK_STEALING=false`)

2. **GPU pipelining:** Complex async code, platform-specific
   - **Mitigation:** Feature flag, only activate on CUDA/CoreML
   - **Rollback:** Disable feature if issues arise

3. **Persistent cache:** Could grow unbounded, corrupt data
   - **Mitigation:** Size limits, integrity checks, auto-rebuild
   - **Rollback:** Env var to disable (`SGREP_PERSISTENT_CACHE=false`)

### Regression Detection
- All PRs require CI benchmark pass
- Fail CI if >10% regression on any primary metric
- Manual review for 5-10% regression range

---

## Success Metrics

### Primary (Release Blockers)
- ✅ 50K files in <60s on 8-core system
- ✅ 10K files in <15s on 8-core system
- ✅ All existing unit tests pass
- ✅ No search quality regression (<5% score change)
- ✅ Memory usage <4GB for 50K indexing

### Secondary (Nice-to-Have)
- 100K files in <120s
- Cache hit rate >30% on re-index
- Search latency <200ms P95 on 50K repos
- CI benchmark runtime <30 minutes

### Marketing
- Blog post published and shared (HN, Reddit)
- README updated with verified benchmarks
- GitHub stars increase by 50+
- At least 3 community performance reports

---

## Post-Release Roadmap

### v0.3.0: Scale to 100K+ Files
- Approximate nearest neighbor search (HNSW)
- Distributed indexing (multiple machines)
- Streaming indexing (search while indexing)

### v0.4.0: Advanced Features
- Multi-modal embeddings (code + docs)
- Fine-tuned embedding model for code
- Query expansion and re-ranking

### v0.5.0: Production Hardening
- Monitoring and observability (Prometheus metrics)
- Multi-language embedding models
- Enterprise features (SSO, RBAC)

---

## Resources

### Documentation
- [PERFORMANCE_OPTIMIZATION_PLAN.md](PERFORMANCE_OPTIMIZATION_PLAN.md) - Full technical plan
- [README_MARKETING_ADDITIONS.md](README_MARKETING_ADDITIONS.md) - Marketing content
- [benches/README.md](benches/README.md) - Benchmark suite docs (to be created)

### Tools
- `cargo bench` - Criterion benchmarks
- `cargo test --release` - SLA tests
- `cargo flamegraph` - Profiling
- `critcmp` - Benchmark comparison

### External References
- [ripgrep performance](https://burntsushi.net/ripgrep/)
- [GitHub 2025 embeddings](https://www.infoq.com/news/2025/10/github-embedding-model/)
- [LlamaIndex parallelization](https://milvus.io/ai-quick-reference/how-does-llamaindex-support-parallel-processing-for-largescale-indexing)

---

**Last Updated:** 2025-11-24
**Owner:** Performance Team
**Status:** Ready to Start Sprint 1 🚀
