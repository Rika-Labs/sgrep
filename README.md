# sgrep — Lightning-Fast Local-First Semantic Code Search

**sgrep** is a Rust CLI that brings semantic awareness to the familiar `grep` workflow. Ask natural-language questions such as “where do we refresh OAuth tokens?” and get highlighted, syntax-aware matches instantly—no cloud calls, no vendor lock-in.

- **Semantic** – Understands concepts like “auth middleware” or “retry logic,” not just substrings.
- **Local & Private** – Deterministic embeddings generated on-device with transformer-inspired hashing (optional ONNX models planned).
- **Auto-Isolated** – Every repository transparently receives its own index under `~/.sgrep/indexes/<hash>`.
- **Adaptive** – Rayon-powered chunking/indexing automatically scales across cores while keeping laptops cool.
- **Agent-Ready** – Designed for coding agents: stable CLI surface, structured JSON output coming soon.

License: **Apache 2.0**

---

## Quick Start

### Install

**Zero-clone install (fetches the latest GitHub Release):**

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
```

Customize with `INSTALL_DIR=$HOME/.local/bin` or `SGREP_REPO=<fork>/sgrep` before the pipe if you need alternate locations.

Releases are generated automatically whenever `main` is updated, so the script always installs the freshest build for macOS (arm64) and Linux (x86_64).

**From source:**

```bash
cargo install --path .
# or, once published:
cargo install sgrep
```

### One-Minute Setup

```bash
sgrep index          # Indexes the current repository (no "." needed)
sgrep search "auth logic"
```

The first search automatically indexes if needed. Subsequent queries are instant thanks to cached embeddings and memory-mapped indexes.

### Watch Mode

```bash
sgrep watch          # Tails the repo, re-indexes on change
```

Need to pre-download larger ONNX models? Place them in `~/.sgrep/models/` and enable the `onnx-embedder` feature (coming soon).

---

## Coding Agent Integration (Preview)

1. Install `sgrep` globally and add it to your agent’s PATH.
2. Run `sgrep watch` before starting the agent session.
3. Teach the agent to run `sgrep search --json "query"` (JSON output lands in v0.2.0).

Agents benefit from deterministic result ordering, score telemetry, and chunk-level metadata (path, lines, language, timestamp).

---

## Commands

### `sgrep search`
Semantic search with hybrid keyword re-ranking.

| Flag | Description | Default |
|------|-------------|---------|
| `query` | Natural-language prompt (required) | — |
| `-p, --path <dir>` | Repository root to search | `.` |
| `-n, --limit <n>` | Max matches | `10` |
| `-c, --context` | Show full chunk instead of snippet | `false` |
| `--glob <pattern>` | Include-only glob (repeatable) | — |
| `--filters key=value` | Language/path filters (repeatable) | — |

Examples:

```bash
sgrep search "API rate limiting"
sgrep search "error handling" --glob "src/**/*.rs" -n 20
sgrep search "database pooling" --filters lang=rust --context
```

### `sgrep index`
Builds or refreshes the local index. If you omit the path, `sgrep` automatically uses the current directory. Respects `.gitignore` and aggressively skips binaries/minified assets.

```bash
sgrep index                 # index current repo
sgrep index ../other-repo   # index specific path
sgrep index --force         # rebuild from scratch
```

### `sgrep watch`
Runs a notify-based watcher that incrementally re-embeds touched files with a smart debounce window.

```bash
sgrep watch                 # watch current repo
sgrep watch ../service --debounce-ms 200
```

### Upcoming
- `sgrep serve` – hot HTTP endpoint for <50 ms responses.
- `sgrep list` – view indexed repos, sizes, last update.
- `sgrep doctor` – verify model paths and index health.

---

## Performance & Architecture

- **Chunking** – tree-sitter extracts functions/classes per language with fallback windowed chunking.
- **Embeddings** – Default hashed embedder offers zero-download semantics; opt-in ONNX models unlock transformer quality without leaving the machine.
- **Storage** – Bincode + zstd keeps repositories compact; indexes live under `~/.sgrep/indexes/`.
- **Hybrid Ranking** – Cosine similarity + keyword match + recency boost deliver precise results without sacrificing recall.
- **Watching** – Notify + Rayon perform incremental re-indexing without blocking your shell.

Target metrics:

| Repo Size | Index Time | Search P95 |
|-----------|------------|------------|
| <1K files | <5s | <50 ms |
| 1K–10K | <30s | <120 ms |
| 10K–100K | <5m | <250 ms |

---

## Configuration

- Models & indexes live under `~/.sgrep/` (configurable via `SGREP_HOME`).
- `RUST_LOG=sgrep=debug` unlocks tracing spans for chunking, embedding, and storage.
- Upcoming `sgrep.toml` will let you pin exclusions, embedding backends, and concurrency limits.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| “No index found” | Run `sgrep index` (auto-detects repo path if omitted). |
| Slow indexing | Set `RAYON_NUM_THREADS=4` to limit concurrency on thermally constrained laptops. |
| Odd matches | Clear `~/.sgrep/indexes/<repo>` and re-index to reset caches. |

---

## Attribution

Inspired by tools such as ripgrep, osgrep, and mgrep. `sgrep` rethinks them with a fully local, Rust-first architecture.

---

## License

Licensed under the [Apache License, Version 2.0](./LICENSE). Contributions are welcome via pull requests.
