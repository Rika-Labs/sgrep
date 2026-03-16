# Changelog

All notable changes to `sgrep` will be documented in this file.

## 2.0.8 - 2026-03-16

### Added
- Added a checked-in `benchmarks/search_quality` evaluation harness so search-quality work can be measured against a stable set of code and documentation queries.

### Changed
- Improved local search ranking quality for natural-language code search with safer PRF expansion gating, candidate-driven file-type intent inference, and cleaner lexical query tokenization.
- Improved exact file/module/symbol retrieval with high-signal BM25F identifier tokenization, chunk-local symbol overlap, lexical confirmation, rare path/symbol bonuses, and documentation heading bonuses.
- Fixed two review-driven ranking edge cases: restored query stem variants are now re-filtered against query stopwords, and prose queries ending in sentence punctuation no longer disable PRF expansion.

### Performance
- Search-quality benchmark improved from the corrected baseline `quality_mrr=71.1111` to `quality_mrr=89.3333`.
- Final kept benchmark also reached `hit_at_1=88.0`, `hit_at_3=92.0`, `hit_at_10=92.0`, `code_mrr=87.8788`, and `docs_mrr=100.0`.

## 2.0.7 - 2026-03-16

### Added
- Added a checked-in changelog for release tracking.

### Changed
- Improved local indexing performance substantially for the local-only workflow.
- Lowered the local embedding `max_length` for `index` and `watch` to `17` while keeping `search` at `40` for quality.
- Kept the corrected worktree-snapshot benchmark harness used during optimization so benchmark results reflect the code under test.

### Removed
- Removed the remaining Modal/offload implementation modules.
- Removed the unused `flate2` dependency and related Modal/offload plumbing.
- Stopped advertising Modal/offload support in the README and Claude Code plugin docs that changed with the implementation removal.

### Performance
- Historical indexing improvement across the full optimization effort: `29366.27ms` → `953.69ms`.
- Final apples-to-apples corrected-harness median: `1396.38ms` → `953.69ms` (~31.7% faster).
- Final corrected-harness search median: `772.32ms` → `702.45ms` with quality held at `6/6`.
