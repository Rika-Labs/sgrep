# sgrep

Semantic search for your codebase. Fast, local-first, agent-ready.

- Semantic: Finds concepts (“auth logic”), not just strings.
- Local & Private: Default embeddings and indexes stay on your machine.
- Auto-Isolated: Each repo gets its own store automatically.
- Agent-Ready: Stable `--json` output and streaming-friendly.
- Adaptive: Auto-index on first search; `--sync` or `watch` keeps it fresh.
- Precision-first: Code/docs weighting, path/language filters, and identifier boosts for better relevance.

## Quick Start
Install
```bash
cargo install --git https://github.com/dallenpyrah/sgrep --locked
```
or build locally:
```bash
cargo build --release
cp target/release/sgrep /usr/local/bin/sgrep
```

Search
```bash
cd my-repo
sgrep search "where do we handle authentication?"
# narrow to code and paths
sgrep search "auth middleware" --lang rs,ts --paths "src/,convex/" --ignore "docs/,test/,data/"
```
First search auto-indexes. Switching repos just works—per-repo stores are automatic.

Keep Fresh (Watch)
```bash
sgrep watch --add .   # add current repo once
sgrep watch           # start watching and auto-reindex on changes
sgrep watch --list    # show watched paths
```

Remote (optional)
```bash
SGREP_REMOTE_URL=http://localhost:6333 sgrep search "payment flow" --remote
SGREP_EMBEDDING_URL=http://localhost:8080/embed sgrep index --sync
```

## Commands
`sgrep search` (default)
- Example: `sgrep search "how is the database connection pooled?"`
- Flags: `-m <n>` max results (25), `--per-file <n>` (1), `--content` full chunks, `--scores`, `--sync` force reindex, `--json`, `--remote`.
- Precision flags: `--lang rs,ts,py`, `--paths "src/,convex/"`, `--ignore "docs/,data/,test/"`.

`sgrep index`
- Manually (re)build index. Respects `.gitignore`.
- Examples: `sgrep index`, `sgrep index --dry-run`, `sgrep index --force=false`, `sgrep index --remote`.
- Markdown: `--include-md=false` to skip docs when building an index (default: true).

`sgrep watch`
- Auto-reindex on file changes.
- Examples: `sgrep watch`, `sgrep watch --add path`, `sgrep watch --remove path`, `sgrep watch --list`, `sgrep watch --clear`.

`sgrep list`
- Show indexed repos and locations.

`sgrep doctor`
- Basic health check (data dir, stores).

`sgrep setup`
- Ensure data dir exists.

## Configuration
- Data location: `~/.sgrep/stores/...` (override with `SGREP_DATA_DIR=/path/to/data`).
- Watch list: `~/.sgrep/watch.jsonl` (or under `SGREP_DATA_DIR`).
- Remote vector DB: `SGREP_REMOTE_URL`, `SGREP_REMOTE_API_KEY`, `SGREP_REMOTE_COLLECTION`.
- Remote embeddings: `SGREP_EMBEDDING_URL` (POST `{ "text": "..." }` → `{ "embedding": [...] }`).

## Performance & Architecture
- Hybrid ranking: BM25 (Tantivy) + semantic cosine; candidate pruning for speed; identifier and path/file-type boosts to lift code over docs/tests.
- Chunking: tree-sitter powered function/class chunks for supported languages; fallback line chunks with overlap; skips oversized files; dedup by chunk hash.
- Indexing: blake3-based incremental reindex (reuse unchanged files); file-watching with debounce.
- Local-first; optional remote search via Qdrant with the same output schema.

## Troubleshooting
- Index stale? `sgrep index --sync`.
- Weird results? `sgrep doctor`.
- Clean slate? Delete the store under `~/.sgrep` (or `SGREP_DATA_DIR`) and reindex.
