# Configuration

## Global flags

- `--device <cpu|cuda|coreml>` (or `SGREP_DEVICE`)
- `--offline` (or `SGREP_OFFLINE=1`) to forbid downloads and fail fast if the model is missing
- `--threads <n>` (or `SGREP_MAX_THREADS`) to bound parallelism
- `--cpu-preset <auto|low|medium|high|background>` (or `SGREP_CPU_PRESET`)

## Commands

### search

- `--path` (default `.`)
- `--limit` (default `10`)
- `--context` to return chunk bodies
- `--glob <pattern>` (repeatable)
- `--filters key=value` (repeatable) for metadata filters like `lang=rust`
- `--json` for structured output
- `--debug` to surface scores and timings

### index

- `--force` for a full rebuild
- `--batch-size` (or `SGREP_BATCH_SIZE`) to override embedder batch size
- `--profile` to print per-phase timings
- `path` argument optional; defaults to the current directory

### watch

- `path` argument optional; defaults to the current directory
- `--debounce-ms` (default `500`)
- `--batch-size` (or `SGREP_BATCH_SIZE`)

### config

- `--init` to write a default config file
- `--show-model-dir` to print the embedding cache location
- `--verify-model` to assert required model files exist

## Configuration file

`SGREP_CONFIG` overrides the path; otherwise it uses `SGREP_HOME/config.toml`, defaulting to `~/.sgrep/config.toml`. Example:

```toml
[embedding]
provider = "local"
```

## Environment variables

- `SGREP_HOME` to relocate indexes and config (default OS data dir such as `~/.local/share/sgrep`)
- `FASTEMBED_CACHE_DIR` to relocate the embedding cache (default OS cache dir such as `~/.local/share/sgrep/cache/fastembed`)
- `SGREP_INIT_TIMEOUT_SECS` to extend model startup (default `120`)
- `HTTP_PROXY` / `HTTPS_PROXY` for model downloads
- `RUST_LOG` for tracing (e.g., `sgrep=debug`)
- `RAYON_NUM_THREADS` to hard-cap the Rayon pool
