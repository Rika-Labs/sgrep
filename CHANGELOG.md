# Changelog

All notable changes to `sgrep` will be documented in this file.

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
