# sgrep Roadmap

This is a living plan to make sgrep the fastest, most developer-friendly semantic search tool.

## Near Term (0.1.x)
- Speed: SIMD everywhere, tighter chunk pruning, PCA/quantization toggle for embeddings, smarter cache hits on partial diffs.
- DX: richer `sgrep doctor` (env, model, store health), friendlier errors, progress bars for index/search, quieter logs by default.
- Watch mode polish: selective reindex on changed files, backoff when system is hot, clearer status output.
- Agents: streaming `--json` output, HTTP/gRPC daemon mode for editors/agents, sample client snippets.
- Packaging: prebuilt binaries (macOS/Linux), `brew`/`scoop` formulas, versioned `cargo install --locked`.
- Tests/bench: reproducible benchmark suite (`sgrep bench`) vs osgrep/ripgrep+fzf; snapshot tests for ranking.

## Mid Term (0.2.x)
- Embeddings: pluggable local models (candle/ggml/gguf) with auto device (CPU/GPU/Metal); on-disk model cache + warmup.
- Ranking: lightweight reranker, identifier/comment boosting, code-structure signals (imports/defs/refs) to bias scoring.
- Indexing: tree-sitter chunking per language, incremental indexer driven by git diff/watchman, better binary/minified detection.
- Remote: adapters for Qdrant/Weaviate/Pinecone; streaming search; hybrid local+remote fallback.
- Multi-repo: federated search across multiple stores; opt-in global cache for common dependencies.
- Tooling: VS Code/JetBrains extensions; CLI daemon keeps hot caches; auth-aware ignores for monorepos.

## Longer Term (0.3.x+)
- Code graph: symbol-aware search (“where is this used?”), call/dep graph traversal, context windows stitched from graph walks.
- Query understanding: rewrite/suggest queries, intent detection (lexical vs semantic vs symbol), auto-filter by language/path.
- Offline quality: curated eval sets, gold labels for ranking; “why” view explaining scores and evidence chunks.
- Team/CI: headless mode for CI gating (stale index check), pre-warm indexes in pipelines, shared remote stores with TTL.
- Ergonomics: zero-config installers, one-liner setup scripts, first-run profiling to auto-tune concurrency and memory budgets.

## Benchmarks & Quality Bar
- Goals: beat osgrep and common stacks on index throughput (files/s), query latency p50/p95, recall@k on code-search benchmarks.
- Ship `sgrep bench` with canned corpora + real-world repo samples; publish default configs; make results reproducible locally.

## Principles
- Local-first by default; remote is explicit and transparent.
- Fast by default; provide knobs but auto-tune when possible.
- Agent-ready: stable schemas, streaming, deterministic ordering.
- Trust: explainable results, clear errors, easy cleanup/reset.***
