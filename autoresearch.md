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
- Search space is intentionally restricted to `src/**/*`, `docs/**/*`, and `plugins/**/*` so the benchmark is not polluted by its own harness files (`benchmarks/search_quality/*`, `autoresearch*`)
- Metric: mean reciprocal rank over accepted files in top 10

## What's Been Tried
- Initial benchmark harness created for quality-first autoresearch
- Corrected the benchmark search space to `src/**/*`, `docs/**/*`, and `plugins/**/*` after the first run revealed benchmark/evaluator files polluting results
- Candidate improvement areas identified before baseline:
  - query-intent-aware weighting instead of fixed semantic/BM25 weights
  - query-aware file-type prioritization so code-seeking queries prefer implementation files while docs-seeking queries still surface docs
  - stronger path/symbol overlap signals for module/file discovery queries
  - lexical candidate blending for large indexes so ANN shortlists do not hide strong exact/path matches
- Kept: replaced hardcoded English phrase checks with generic PRF gating based on query structure plus ranking confidence, and inferred file-type intent from top candidates instead of string-matching the query. This preserved the best quality at 79.33 while making ranking less brittle across repos.
- Discarded: broad lexical shifts hurt quality or latency, including query-adaptive fusion weights, larger rerank windows, full-content identifier subword expansion, and naive multi-chunk file-evidence bonuses.
- Kept: expanded BM25F identifier tokenization only for high-signal lexical fields (filename/path/symbols), not all content. This improved module-discovery style code queries and raised quality_mrr to 81.33 without introducing benchmark-specific strings.
- Discarded: copying top-level file symbols into every chunk's lexical document over-broadcast file vocabulary and hurt docs even though it helped some code queries.
- Kept: added a small graph-aware local symbol-overlap bonus during reranking. This rewards chunks whose own extracted symbol identifiers structurally match the query terms and raised quality_mrr to 82.00 without relying on hardcoded query phrases.
