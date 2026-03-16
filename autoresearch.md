# Autoresearch: search quality

## Objective
Improve local search accuracy and result quality for `sgrep`, with the primary focus on returning the most useful files/chunks for natural-language code search queries.

This session targets **quality first**, not latency. The benchmark re-indexes the current worktree and then runs a query set against the `sgrep` repository itself so index-affecting changes are exercised too.

The benchmark mixes:
- code-intent queries that should land on implementation files
- documentation-intent queries that should land on docs/plugin guides
- broad architecture queries and lower-level implementation queries

## Metrics
- **Primary**: `quality_mrr` (higher is better)
- **Secondary**: `hit_at_1`, `hit_at_3`, `hit_at_10`, `code_mrr`, `docs_mrr`, `median_search_ms`

## How to Run
`./autoresearch.sh`

It rebuilds the release binary, force-reindexes the current repo in offline mode, runs the search-quality benchmark, and prints `METRIC name=value` lines.

## Files in Scope
- `src/search/engine.rs` — core search orchestration and ranking pipeline
- `src/search/scoring.rs` — adaptive weighting and score combination
- `src/search/file_type.rs` — path-based file type prioritization
- `src/search/config.rs` — search tuning constants
- `src/search/dedup.rs` — duplicate suppression, if ranking interactions matter
- `src/fts/mod.rs` — keyword extraction and BM25/BM25F helpers
- `src/graph/*` — graph-aware retrieval/boosting if needed
- `src/output/mod.rs` — only if result serialization/debugging support is needed
- `benchmarks/search_quality/*` — benchmark dataset/evaluator maintenance only when fixing benchmark correctness, not to game results
- `autoresearch.sh` / `autoresearch.checks.sh` — benchmark/check harness maintenance

## Off Limits
- No hardcoded query → path special cases in production code
- No benchmark-only branches, hidden env checks, or score hacks that detect benchmark queries
- Do not weaken or narrow the benchmark to manufacture wins
- Do not remove correctness checks to increase throughput
- Do not optimize exclusively for this repo's exact file names if the change is unlikely to generalize

## Constraints
- Keep search behavior general-purpose for local semantic code search
- Avoid benchmark overfitting; prefer changes that plausibly help other repos and larger indexes
- `cargo fmt -- --check`, `cargo clippy -- -D warnings`, and `cargo test` must pass for kept results
- Prefer minimal, explainable ranking changes over brittle complexity

## Benchmark Dataset
- `benchmarks/search_quality/queries.json`
- 25 queries total
- Code answers: source files under `src/`
- Docs answers: files under `docs/` and `plugins/sgrep/`
- Metric: mean reciprocal rank over accepted files in top 10

## What's Been Tried
- Initial benchmark harness created for quality-first autoresearch
- Candidate improvement areas identified before baseline:
  - query-intent-aware weighting instead of fixed semantic/BM25 weights
  - query-aware file-type prioritization so code-seeking queries prefer implementation files while docs-seeking queries still surface docs
  - stronger path/symbol overlap signals for module/file discovery queries
  - lexical candidate blending for large indexes so ANN shortlists do not hide strong exact/path matches
