<div align="center">

  <h1>sgrep</h1>

  <p><em>Lightning-fast local-first semantic code search.</em></p>

  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a><br>

</div>

Natural-language search that works like `grep`. Fast, local, and works with coding agents.

- **Semantic:** Finds concepts ("auth middleware", "retry logic"), not just strings.
- **Local & Private:** Real ML embeddings powered by BGE-small-en-v1.5-q (quantized, 384-dim, runs locally via ONNX).
- **Auto-Isolated:** Every repository transparently receives its own index under `~/.sgrep/indexes/<hash>`.
- **Adaptive:** Rayon-powered chunking/indexing automatically scales across cores while keeping laptops cool.
- **Agent-Ready:** Designed for coding agents: stable CLI surface, structured JSON output coming soon.

## Quick Start

1. **Install**

   ```bash
   curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
   ```

   Customize with `INSTALL_DIR=$HOME/.local/bin` or `SGREP_REPO=<fork>/sgrep` before the pipe if you need alternate locations.

   **From source:**

   ```bash
   cargo install --path .
   ```

2. **Search**

   ```bash
   cd my-repo
   sgrep search "where do we handle authentication?"
   ```

   **Your first search will automatically index the repository.** Each repository is automatically isolated with its own index. Switching between repos "just works" — no manual configuration needed.

## Coding Agent Integration

1. Install `sgrep` globally and add it to your agent's PATH.
2. Run `sgrep watch` before starting the agent session.
3. Teach the agent to run `sgrep search --json "query"` (JSON output lands in v0.2.0).

Agents benefit from consistent result ordering, semantic understanding, score telemetry, and chunk-level metadata (path, lines, language, timestamp).

## Commands

### `sgrep search`

The default command. Searches the current directory using semantic meaning.

```bash
sgrep search "how is the database connection pooled?"
```

**Options:**

| Flag | Description | Default |
| --- | --- | --- |
| `-n, --limit <n>` | Max matches to return. | `10` |
| `-c, --context` | Show full chunk content instead of snippet. | `false` |
| `-p, --path <dir>` | Repository root to search. | `.` |
| `--glob <pattern>` | Include-only glob (repeatable). | — |
| `--filters key=value` | Language/path filters (repeatable). | — |

**Examples:**

```bash
sgrep search "API rate limiting"
sgrep search "error handling" --glob "src/**/*.rs" -n 20
sgrep search "database pooling" --filters lang=rust --context
```

### `sgrep index`

Manually indexes the repository. Useful if you want to pre-warm the cache or if you've made massive changes outside of the editor.

- Respects `.gitignore` and aggressively skips binaries/minified assets.
- **Smart Indexing:** Only embeds code and config files. Skips binaries, lockfiles, and minified assets.
- **Adaptive Throttling:** Monitors your RAM and CPU usage. If your system gets hot, indexing slows down automatically.

```bash
sgrep index              # Index current dir
sgrep index ../other-repo   # Index specific path
sgrep index --force         # Rebuild from scratch
```

### `sgrep watch`

Runs a notify-based watcher that incrementally re-embeds touched files with a smart debounce window.

```bash
sgrep watch                 # Watch current repo
sgrep watch ../service --debounce-ms 200
```

## Performance & Architecture

sgrep is designed to be a "good citizen" on your machine:

1. **Real Embeddings:** Uses BGE-small-en-v1.5-q, a quantized SOTA model (33M params) that runs locally via FastEmbed + ONNX. First run downloads the model (~24MB) once.
2. **The Thermostat:** Indexing adjusts concurrency in real-time based on memory pressure and CPU speed. It won't freeze your laptop.
3. **Smart Chunking:** Uses `tree-sitter` to split code by function/class boundaries, ensuring embeddings capture complete logical blocks.
4. **Deduplication:** Identical code blocks (boilerplate, license headers) are embedded once and cached, saving space and time.
5. **Hybrid Search:** 70% semantic similarity + 20% keyword matching (path/filename boosted) + 10% recency for optimal results.

**Target metrics:**

| Repo Size | Index Time | Search P95 |
|-----------|------------|------------|
| <1K files | <5s | <50 ms |
| 1K–10K | <30s | <120 ms |
| 10K–100K | <5m | <250 ms |

## Configuration

### Automatic Repository Isolation

sgrep automatically creates a unique index for each repository based on repository paths. Indexes live under `~/.sgrep/indexes/<hash>`.

**Examples:**

```bash
cd ~/work/myproject        # Auto-detected and indexed
sgrep search "API handlers"

cd ~/personal/utils        # Auto-detected and indexed
sgrep search "helper functions"
```

Stores are isolated automatically — no manual configuration needed!

### Manual Configuration

- **Data location:** `~/.sgrep/` (configurable via `SGREP_HOME`)
- **Model cache:** `~/.cache/fastembed/` (embedding models downloaded once on first use)
- **Env Vars:**
  - `SGREP_HOME`: Override default data directory
  - `RUST_LOG=sgrep=debug`: Enable tracing spans for chunking, embedding, and storage
  - `RAYON_NUM_THREADS=4`: Limit concurrency on thermally constrained laptops

Upcoming `sgrep.toml` will let you pin exclusions and concurrency limits.

## Development

```bash
cargo build --release
cargo test
cargo fmt
cargo clippy
```

## Troubleshooting

- **Index feels stale?** Run `sgrep index` to refresh.
- **Weird results?** Clear `~/.sgrep/indexes/<repo>` and re-index to reset caches.
- **Slow indexing?** Set `RAYON_NUM_THREADS=4` to limit concurrency on thermally constrained laptops.
- **No index found?** Run `sgrep index` (auto-detects repo path if omitted).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for a comprehensive list of implemented and planned features.

## Attribution

Inspired by tools such as ripgrep, osgrep, and mgrep. `sgrep` rethinks them with a fully local, Rust-first architecture.

## License

Licensed under the Apache License, Version 2.0.  
See [LICENSE](LICENSE) and [Apache-2.0](https://opensource.org/licenses/Apache-2.0) for details.
