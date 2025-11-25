# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` wires the CLI; `search.rs` orchestrates hybrid semantic + keyword search; `indexer.rs` builds and refreshes indexes; `chunker.rs` handles tree-sitter chunking; `embedding.rs` manages the embedding pool and providers (local mxbai + Voyage API); `config.rs` handles config file parsing; `store.rs` persists metadata; `fts.rs` covers keyword scoring; `watch.rs` handles incremental re-indexing.
- `plugins/sgrep/` contains the Claude Code plugin (hooks, skills). Update `plugins/sgrep/README.md` when changing agent behavior.
- `scripts/install.sh` is the installer; `default-ignore.txt` lists default exclusions; `target/` holds build artifacts (use `cargo clean` to reset).

## Build, Test, and Development Commands
- `cargo build` (debug) / `cargo build --release` (ship-ready binary).
- `cargo run -- --help` for a smoke test; `cargo run -- search "query"` to exercise the CLI locally.
- `cargo fmt` enforces formatting; `cargo clippy -- -D warnings` lints and fails on warnings.
- `cargo test` runs all unit tests (module-local); narrow scope with `cargo test search::tests::`.
- Useful diagnostics: `RUST_LOG=sgrep=debug cargo run -- search "query"` for tracing.
- Offline indexing: `sgrep --offline index` (or `SGREP_OFFLINE=1`) disables network fetches. Ensure the model cache exists at `~/.sgrep/cache/fastembed` (run once with network or copy the model) or you'll get a clear error up front.
- Config management: `sgrep config` shows current provider; `sgrep config --init` creates default config file. Config lives at `~/.sgrep/config.toml`.

## Coding Style & Naming Conventions
- Rust 2021, toolchain `rust-version = 1.75`; 4-space indentation. Always run `cargo fmt`.
- Prefer `anyhow::Result<T>` for fallible functions; use `thiserror` for structured error types.
- Modules/files stay snake_case; types are PascalCase; functions snake_case; CLI flags are kebab-case via Clap.
- Keep the JSON output stable; gate breaking schema or flag changes behind a new flag and document updates in README and plugin docs.

## Testing Guidelines
- Unit tests live beside implementations (`mod tests` blocks). Add regression tests near the code you touch.
- Use deterministic fixtures; avoid real index dirs—prefer temp dirs (`tempdir`/`std::env::temp_dir`).
- Concurrency/indexing tests should be serialized with `serial_test` to avoid global state races.
- Expected pre-PR checks: `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo test`.

## Commit & Pull Request Guidelines
- Commits: short, imperative, ~50 chars (e.g., "Improve debounce in watch loop"); group related changes together.
- PRs: provide a succinct summary of behavior change, verification steps (commands/output), linked issue if any, and note schema/flag changes. Update README and plugin docs when user-facing behavior shifts.
- Include screenshots or sample CLI output when altering UX or logging; call out performance impact if touching indexing or embedding paths.
