# Quickstart

## Prerequisites

- macOS or Linux on arm64 or x86_64.
- For source builds: Rust 1.75+ (`cargo install --path .`).

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh
```

Environment overrides:
- `INSTALL_DIR` for the binary destination (default `/usr/local/bin`)
- `SGREP_REPO` to pin a fork or branch

## First search

```bash
cd path/to/repo
sgrep search "where do we handle authentication?"
```

The first run downloads `jina-embeddings-v2-base-code`, builds an index, and caches everything locally. Add `--json` when feeding results to an agent.

## Keep the index fresh

- Continuous: `sgrep watch [path]`
- Manual rebuild: `sgrep index [path]` or `sgrep index --force` for a clean slate

## Where files live

- Model cache: `FASTEMBED_CACHE_DIR` or the OS cache dir (e.g., `~/.local/share/sgrep/cache/fastembed`), falling back to `~/.sgrep/cache/fastembed`.
- Indexes and data: `SGREP_HOME` or the OS data dir (e.g., `~/.local/share/sgrep`), falling back to `~/.sgrep`.
- Config: `SGREP_CONFIG` or `SGREP_HOME/config.toml` (default `~/.sgrep/config.toml`).

## Verify

```bash
sgrep --version
sgrep search --json "hello world"
```
