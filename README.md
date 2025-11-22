# sgrep

Semantic grep for your codebase — local-first, fast, agent-friendly.

## What you get
- Natural-language search that feels like `grep`.
- Local-first indexing + BM25 + semantic ranking; remote is opt-in.
- Auto-index on first search; `--sync` to refresh; `sgrep watch` to keep it hot.
- JSON output for agents; simple flags for scores, content, limits.

## Install
```bash
cargo build --release
```
Binary: `target/release/sgrep` (put it on your PATH if you like).

## Use in 30 seconds
```bash
# 1) search (auto-indexes if missing)
sgrep search "where do we handle authentication?"

# 2) keep it fresh while you code
sgrep watch --add .     # add current repo once
sgrep watch             # start watching and auto-reindex on changes
sgrep watch --list      # see watched paths

# 3) optional: remote backend + embeddings
SGREP_REMOTE_URL=http://localhost:6333 sgrep search "payment flow" --remote
SGREP_EMBEDDING_URL=http://localhost:8080/embed sgrep index --sync
```

Common flags:
- `--sync` force re-index before searching.
- `--content` include full chunk text; `--scores` show scoring.
- `-m/--max` total matches, `--per-file` per-file cap.
- `--json` agent-ready output.

## Commands at a glance
- `sgrep search <query>` – semantic + keyword search (auto-sync if missing).
- `sgrep index [--path DIR] [--force=false] [--remote]` – build/refresh index.
- `sgrep watch [--add PATH|--remove PATH|--list|--clear] [--debounce-ms N]` – auto-reindex on file changes.
- `sgrep list` – show known stores.
- `sgrep setup` – ensure data dir exists.
- `sgrep doctor` – basic health info.

## Config & data
- Data: `~/.sgrep/stores/<store-id>/...` (override with `SGREP_DATA_DIR=/path/to/data`).
- Watch list: `~/.sgrep/watch.jsonl` (or under `SGREP_DATA_DIR`).
- Remote vector DB: `SGREP_REMOTE_URL`, `SGREP_REMOTE_API_KEY`, `SGREP_REMOTE_COLLECTION`.
- Remote embeddings: `SGREP_EMBEDDING_URL` (POST `{ "text": "..." }` → `{ "embedding": [...] }`).

## How it works (short)
- Per-repo store keyed by canonical path.
- Chunking with overlaps; skips oversized files; dedup by chunk hash.
- Hybrid ranking: BM25 (Tantivy) + semantic cosine; candidate pruning for speed.
- Optional remote search via Qdrant; same output schema for local/remote.

## Tips
- Use `--sync` after large refactors; otherwise searches reuse the index.
- `sgrep index --remote` pushes to your remote vector DB after local build.
- Set `SGREP_DATA_DIR` in CI or sandboxes to keep state isolated.
