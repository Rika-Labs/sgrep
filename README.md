# sgrep

Semantic grep for your codebase.

`sgrep` is a **Rust-based, local-first semantic code search tool** that aims to feel as immediate and ergonomic as `grep`, while giving you high-quality, natural-language results and great agent integration.

- **Natural language first** – Ask "where do we handle authentication?" instead of remembering exact symbols.
- **Blazing fast by default** – Parallel semantic scoring + BM25 full-text search over a per-repo index.
- **Local-first & private** – Everything runs on your machine; remote backends are opt-in.
- **Agent-ready** – Stable `--json` output format that coding agents can consume directly.
- **Auto-indexed** – Searches will transparently build an index if missing; `--sync` forces a refresh when you want it.
- **Drop-dead simple watching** – `sgrep watch` keeps your index hot while you code; manage watched paths with list/add/remove/clear.

## Quick start

### Install

From source:

```bash
cargo build --release
```

This produces the binary at `target/release/sgrep`.

You can then put it on your `PATH`, or call it directly.

### One-time setup (optional)

```bash
./target/release/sgrep setup
```

This creates the base data directory used to store per-repo indexes:

- `~/.sgrep/stores/<store-id>/index.jsonl` – semantic index
- `~/.sgrep/stores/<store-id>/fts/` – BM25 full-text index (Tantivy)

### Index a repository

From the repository root:

```bash
./target/release/sgrep index
```

This will:

1. Walk the repo (respecting `.gitignore`).
2. Chunk code/config files into small logical blocks.
3. Embed each chunk.
4. Persist both the semantic index and full-text index.

You can also index another directory explicitly:

```bash
./target/release/sgrep index --path /path/to/repo
```

### Search

From a repo that has been indexed (or will be lazily indexed on first search):

```bash
./target/release/sgrep search "where do we handle authentication?"
```

Examples:

```bash
# General concept search
sgrep search "API rate limiting logic"

# Show more matches per file
sgrep search "error handling" --per-file 5

# JSON output for agents
sgrep search "user validation" --json -m 50

# Force a re-index before searching
sgrep search "queue backpressure" --sync

# Emit full chunk content instead of snippets and include scoring
sgrep search "metrics exporter" --content --scores
```

By default, `sgrep search`:

- Uses BM25 to find strong keyword matches.
- Uses semantic embeddings to score conceptual similarity.
- Fuses both scores, then limits results globally and per file.

## Commands

### `sgrep search`

Search the current repository using semantic + keyword meaning.

```bash
sgrep search <query>
```

Flags:

- `--json` – emit structured JSON results.
- `-m, --max <n>` – max total matches (default: `25`).
- `--per-file <n>` – max matches per file (default: `1`).
- `--remote` – send the query to the remote Qdrant backend instead of local search.

JSON schema (simplified):

```json
{
  "query": "string",
  "total": 0,
  "matches": [
    {
      "path": "path/to/file.rs",
      "start_line": 10,
      "end_line": 24,
      "score": 0.93,
      "semantic_score": 0.88,
      "keyword_score": 0.12,
      "snippet": "string snippet..."
    }
  ]
}
```

### `sgrep index`

Build or refresh the index for a repository.

```bash
sgrep index                 # index current dir
sgrep index --path ./other  # index another directory
sgrep index --remote        # also push to remote vector DB
sgrep index --dry-run       # show what would be indexed
sgrep index --force=false   # only index if missing
```

`sgrep index`:

- Writes a JSONL index with chunk metadata + embeddings.
- Builds a Tantivy BM25 index over chunk contents.
- When `--remote` is set, also upserts chunks into the configured Qdrant instance.
- Rebuilds the index by default; pass `--force=false` if you only want to index when a store is missing.

### `sgrep watch`

Keep your index fresh as you edit files (local-first).

```bash
sgrep watch               # watch current repo (or your saved list) and auto-reindex
sgrep watch --list        # show watched paths
sgrep watch --add path/   # add a path to the watch list
sgrep watch --remove path # remove a path from the watch list
sgrep watch --clear       # clear the watch list
```

Watching:

- Debounced file watching to avoid thrashing.
- Forces a re-index for each watched path after changes settle.
- Persists the watch list under `~/.sgrep/watch.jsonl` so `sgrep watch` works with no flags after you add paths once.

### `sgrep setup`

Initialize the data directory:

```bash
sgrep setup
```

This ensures `~/.sgrep/stores` exists and prints its path.

### `sgrep list`

List all known stores (per-repo indexes):

```bash
sgrep list
```

Shows store IDs and index paths.

### `sgrep doctor`

Print basic health information:

```bash
sgrep doctor
```

Shows the data directory and number of stores. Over time this can expand to cover embedding and remote backend diagnostics.

## Embeddings

`sgrep` separates **how** embeddings are computed from **how** they are used.

### Local default (no configuration)

By default, `sgrep` uses a fast, deterministic hash-based embedding of tokens into a fixed-size vector.

- No network calls.
- No model downloads.
- Works out-of-the-box on any machine that can build Rust.

This is combined with BM25, so you still get high-quality results even without a heavy semantic model.

### Remote embedding backend (recommended for best results)

To get the best possible semantic quality, you can point `sgrep` at a remote embedding service via an environment variable:

- `SGREP_EMBEDDING_URL` – HTTP endpoint that accepts a POST with `{ "text": "..." }` and returns `{ "embedding": [f32, ...] }`.

When this is set:

- Indexing uses the remote model to embed chunks.
- Local search uses the remote model to embed queries.
- Remote Qdrant search uses the remote model to embed queries as well.

If the remote embedding call fails, `sgrep` will return an error rather than silently degrading.

This design lets you:

- Start simple (no configuration).
- Later plug in your own high-quality embedding service or a managed provider without changing how you call `sgrep`.

## Remote vector DB (Qdrant)

`sgrep` can optionally push and query embeddings from a remote Qdrant instance for even more scalability and offloaded compute.

### Environment variables

- `SGREP_REMOTE_URL` – base URL of Qdrant (`http://localhost:6333` by default).
- `SGREP_REMOTE_API_KEY` – optional API key, sent as `api-key` header.
- `SGREP_REMOTE_COLLECTION` – optional collection name; by default it derives a per-repo name from the repository path.

### Remote indexing

```bash
sgrep index --remote
```

This will:

1. Build the local index (if needed).
2. Ensure a Qdrant collection exists with the right vector size and cosine distance.
3. Upsert all local chunks into Qdrant with payloads (`path`, `start_line`, `end_line`, `text`).

### Remote search

```bash
sgrep search "auth logic" --remote
sgrep search "auth logic" --remote --json
```

This calls Qdrant’s search API and maps results back into the same JSON schema used by local search, so agents and tools do not need to know whether results are coming from local or remote.

## Data layout

By default, `sgrep` stores its data under:

- `~/.sgrep/stores/<store-id>/index.jsonl` – semantic chunks and embeddings.
- `~/.sgrep/stores/<store-id>/fts/` – Tantivy index files.
- Override the base data directory with `SGREP_DATA_DIR=/custom/path` if you want to keep state in a repo-local or sandboxed location.

Store IDs are derived from the canonical path of the repository directory.

Deleting a store is as simple as removing that directory; the next `sgrep index` will rebuild it.

## Design and performance

- **Rust single binary** – easy distribution, predictable performance.
- **Chunk-based indexing** – indexes functions / logical blocks instead of whole files.
- **Parallel search** – semantic scoring uses Rayon to fully utilize available CPU cores.
- **Hybrid ranking** – BM25 (Tantivy) + semantic cosine similarity for high precision and recall.
- **Local-first** – remote dependencies are strictly opt-in.
- **Index efficiency** – overlaps chunks to capture context, skips oversized files by default, deduplicates chunk IDs, and prunes search candidates using BM25 before semantic scoring.

## Status

`sgrep` is an early, fast-moving project. Expect:

- Some rough edges in configuration and diagnostics.
- Rapid iteration on ranking heuristics and integration points.

Feedback, issues, and ideas for improving the developer experience are very welcome.
# sgrep
