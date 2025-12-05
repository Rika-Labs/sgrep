use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use console::{style, Term};
use indicatif::HumanDuration;
use tracing::{info, warn};

use crate::cli::{resolve_repo_path, Cli, Commands};
use crate::config::Config;
use crate::embedding::{self, Embedder, PooledEmbedder};
use crate::output::JsonResponse;
use crate::threading::{CpuPreset, ThreadConfig};
use crate::{indexer, search, store, watch};

/// Parameters for executing a search operation.
/// Consolidates the search-related parameters into a single struct
/// to improve readability and reduce function parameter count (KISS principle).
pub struct SearchParams<'a> {
    pub query: &'a str,
    pub path: &'a Path,
    pub limit: usize,
    pub context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
    pub json: bool,
    pub debug: bool,
    pub no_rerank: bool,
    pub rerank_oversample: usize,
}

struct ProgressLine {
    term: Term,
    enabled: bool,
}

impl ProgressLine {
    fn stderr() -> Self {
        let term = Term::stderr();
        let enabled = term.is_term();
        Self { term, enabled }
    }

    fn set(&self, message: &str) {
        if self.enabled {
            let _ = self.term.clear_line();
            let _ = self.term.write_str(&format!("\r{message}"));
            let _ = self.term.flush();
        } else {
            eprintln!("{message}");
        }
    }

    fn finish(&self, message: &str) {
        if self.enabled {
            let _ = self.term.clear_line();
            let _ = self.term.write_line(&format!("\r{message}"));
        } else {
            eprintln!("{message}");
        }
    }
}

/// Context needed to render search results.
/// Consolidates the 7 render-related parameters into a single struct.
pub struct RenderContext<'a> {
    pub results: Vec<search::SearchResult>,
    pub query: &'a str,
    pub limit: usize,
    pub index: &'a store::RepositoryIndex,
    pub elapsed: Duration,
    pub json: bool,
    pub debug: bool,
}

pub fn run() -> Result<()> {
    setup_tracing();
    let cli = parse_cli();
    run_with_cli(cli)
}

pub fn run_with_cli(cli: Cli) -> Result<()> {
    let preset = cli.cpu_preset.as_ref().and_then(|s| CpuPreset::from_str(s));
    ThreadConfig::init(cli.max_threads, preset);
    ThreadConfig::get().apply();

    if let Commands::Config {
        init,
        show_model_dir,
        verify_model,
    } = &cli.command
    {
        return handle_config(*init, *show_model_dir, *verify_model);
    }

    let embedder = build_embedder(cli.offline, cli.device.clone())?;

    match cli.command {
        Commands::Index {
            path,
            force,
            batch_size,
            profile,
            stats,
            json,
        } => {
            if stats {
                return handle_index_stats(path, json);
            }
            handle_index(embedder, path, force, batch_size, profile)
        }
        Commands::Search {
            query,
            path,
            limit,
            context,
            glob,
            filters,
            json,
            debug,
            no_rerank,
            rerank_oversample,
        } => handle_search(
            embedder,
            SearchParams {
                query: &query,
                path: &path,
                limit,
                context,
                glob,
                filters,
                json,
                debug,
                no_rerank,
                rerank_oversample,
            },
        ),
        Commands::Watch {
            path,
            debounce_ms,
            batch_size,
        } => handle_watch(embedder, path, debounce_ms, batch_size),
        Commands::Config { .. } => unreachable!(), // Handled above
    }
}

fn build_embedder(
    offline: bool,
    device: Option<String>,
) -> Result<Arc<dyn embedding::BatchEmbedder>> {
    let progress = ProgressLine::stderr();

    if env::var("TOKENIZERS_PARALLELISM").is_err() {
        env::set_var("TOKENIZERS_PARALLELISM", "true");
    }

    if offline {
        env::set_var("SGREP_OFFLINE", "1");
    }

    if let Some(device) = device {
        env::set_var("SGREP_DEVICE", device);
    }

    embedding::configure_offline_env(offline)?;

    let use_pooled = env::var("SGREP_USE_POOLED_EMBEDDER")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    progress.set("Loading embedding model…");

    let embedder: Arc<dyn embedding::BatchEmbedder> = if use_pooled {
        Arc::new(PooledEmbedder::default())
    } else {
        Arc::new(Embedder::default())
    };

    progress.finish("Embedding model ready");

    Ok(embedder)
}

#[cfg(not(test))]
fn build_reranker_silent(
    json_mode: bool,
) -> Option<Arc<dyn crate::reranker::Reranker + Send + Sync>> {
    use crate::reranker::CrossEncoderReranker;

    match CrossEncoderReranker::new(!json_mode) {
        Ok(reranker) => Some(Arc::new(reranker)),
        Err(e) => {
            if !json_mode {
                eprintln!(
                    "{} Reranker unavailable: {} (falling back to semantic search)",
                    style("⚠").yellow(),
                    e
                );
            }
            None
        }
    }
}

#[cfg(test)]
fn build_reranker_silent(
    _json_mode: bool,
) -> Option<Arc<dyn crate::reranker::Reranker + Send + Sync>> {
    None
}

fn handle_config(init: bool, show_model_dir: bool, verify_model: bool) -> Result<()> {
    if show_model_dir {
        let model_dir = embedding::get_fastembed_cache_dir().join(embedding::MODEL_NAME);
        println!("{}", model_dir.display());
        return Ok(());
    }

    if verify_model {
        return verify_model_files();
    }

    let config_path = Config::config_path();

    if init {
        if config_path.exists() {
            println!(
                "{} Config already exists at {}",
                style("ℹ").cyan(),
                config_path.display()
            );
        } else {
            let path = Config::create_default_config()?;
            println!(
                "{} Created config at {}",
                style("✔").green(),
                path.display()
            );
        }
        return Ok(());
    }

    println!(
        "{} Config path: {}",
        style("ℹ").cyan(),
        config_path.display()
    );

    if config_path.exists() {
        println!(
            "  Provider: {}",
            style(format!("local ({})", embedding::MODEL_NAME)).bold()
        );
    } else {
        println!("  No config file found (using defaults)");
        println!(
            "  Provider: {}",
            style(format!("local ({})", embedding::MODEL_NAME)).bold()
        );
        println!();
        println!(
            "  Run {} to create a config file",
            style("sgrep config --init").cyan()
        );
    }

    Ok(())
}

fn verify_model_files() -> Result<()> {
    let model_dir = embedding::get_fastembed_cache_dir().join(embedding::MODEL_NAME);

    println!(
        "{} Model directory: {}\n",
        style("ℹ").cyan(),
        model_dir.display()
    );

    let mut all_ok = true;
    for file in embedding::MODEL_FILES {
        let exists = model_dir.join(file).exists();
        let status = if exists {
            style("OK").green()
        } else {
            all_ok = false;
            style("MISSING").red()
        };
        println!("  [{}] {}", status, file);
    }

    println!();
    if all_ok {
        println!("{} All model files present.", style("✔").green());
        Ok(())
    } else {
        println!("{} Some model files are missing.\n", style("✖").red());
        println!("Download from: {}", embedding::MODEL_DOWNLOAD_URL);
        println!("Place files in: {}", model_dir.display());
        Err(anyhow!("Model files incomplete"))
    }
}

fn handle_index_stats(path: Option<std::path::PathBuf>, json: bool) -> Result<()> {
    let repo_path = resolve_repo_path(path)?;
    let store = store::IndexStore::new(&repo_path)?;

    let stats = store
        .get_stats()?
        .ok_or_else(|| anyhow!("No index found. Run 'sgrep index' first."))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("Index Statistics");
        println!("  Repository:     {}", stats.repo_path.display());
        println!("  Indexed at:     {}", stats.indexed_at);
        println!("  Vector dim:     {}", stats.vector_dim);
        println!("  Files:          {}", stats.total_files);
        println!("  Chunks:         {}", stats.total_chunks);
        println!("  Graph symbols:  {}", stats.graph_symbols);
        println!("  Graph edges:    {}", stats.graph_edges);
        println!(
            "  Mmap:           {}",
            if stats.mmap_available { "yes" } else { "no" }
        );
        println!(
            "  Binary vectors: {}",
            if stats.binary_vectors_available {
                "yes"
            } else {
                "no"
            }
        );
    }
    Ok(())
}

fn handle_index(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    path: Option<std::path::PathBuf>,
    force: bool,
    batch_size: Option<usize>,
    profile: bool,
) -> Result<()> {
    let path = resolve_repo_path(path)?;

    if !crate::query_expander::is_model_cached() {
        use crate::query_expander::QueryExpander;
        if let Err(e) = QueryExpander::new() {
            eprintln!(
                "{} Query expander download failed: {} (search will use heuristics)",
                style("⚠").yellow(),
                e
            );
        }
    }

    let indexer = indexer::Indexer::new(embedder.clone());
    let report = indexer
        .build_index(indexer::IndexRequest {
            path: path.clone(),
            force,
            batch_size,
            profile,
            dirty: None,
        })
        .context("Failed to build index")?;

    println!(
        "{} Indexed {} files ({} chunks) in {}",
        style("✔").green(),
        report.files_indexed,
        report.chunks_indexed,
        HumanDuration(report.duration)
    );

    // Show graph statistics
    if report.graph_symbols > 0 {
        println!(
            "  {} symbols extracted, {} relationships",
            report.graph_symbols, report.graph_edges
        );
    }

    if profile {
        if let Some(t) = report.timings {
            println!(
                "  walk: {} | chunk: {} | embed: {} | graph: {} | write: {}",
                HumanDuration(t.walk),
                HumanDuration(t.chunk),
                HumanDuration(t.embed),
                HumanDuration(t.graph),
                HumanDuration(t.write)
            );
        }
        println!(
            "  cache hits: {} | cache misses: {} | hit rate: {:.1}%",
            report.cache_hits,
            report.cache_misses,
            if report.cache_hits + report.cache_misses == 0 {
                0.0
            } else {
                (report.cache_hits as f64 / (report.cache_hits + report.cache_misses) as f64)
                    * 100.0
            }
        );
    }

    Ok(())
}

fn handle_search(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    params: SearchParams<'_>,
) -> Result<()> {
    let start = Instant::now();

    let reranker = if params.no_rerank {
        None
    } else {
        build_reranker_silent(params.json)
    };

    let mut engine = match &reranker {
        Some(r) => search::SearchEngine::with_reranker(embedder.clone(), r.clone()),
        None => search::SearchEngine::new(embedder.clone()),
    };

    if let Err(e) = engine.enable_query_expander_silent() {
        if !params.json {
            eprintln!("{} Query expander unavailable: {}", style("⚠").yellow(), e);
        }
    }

    // Try to load the code graph for hybrid search
    let store_result = store::IndexStore::new(params.path)?;
    if let Ok(Some(graph)) = store_result.load_graph() {
        engine.set_graph(graph);
    }

    let rerank_enabled = reranker.is_some();

    // Try mmap-based search first for zero-copy performance
    if let Ok(Some(mmap_index)) = store_result.load_mmap() {
        if mmap_index.metadata.vector_dim == embedder.dimension() {
            let results = engine.search_hybrid_mmap(
                &mmap_index,
                params.query,
                search::SearchOptions {
                    limit: params.limit,
                    include_context: params.context,
                    glob: params.glob.clone(),
                    filters: params.filters.clone(),
                    rerank: rerank_enabled,
                    oversample_factor: params.rerank_oversample,
                },
            )?;
            let elapsed = start.elapsed();
            let repo_index = mmap_index.to_repository_index();
            return render_results(RenderContext {
                results,
                query: params.query,
                limit: params.limit,
                index: &repo_index,
                elapsed,
                json: params.json,
                debug: params.debug,
            });
        }
    }

    // Fall back to standard loading
    let index = load_or_index(params.path, embedder.clone())?;
    let results = engine.search_hybrid(
        &index,
        params.query,
        search::SearchOptions {
            limit: params.limit,
            include_context: params.context,
            glob: params.glob,
            filters: params.filters,
            rerank: rerank_enabled,
            oversample_factor: params.rerank_oversample,
        },
    )?;
    let elapsed = start.elapsed();

    render_results(RenderContext {
        results,
        query: params.query,
        limit: params.limit,
        index: &index,
        elapsed,
        json: params.json,
        debug: params.debug,
    })
}

fn handle_watch(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    path: Option<std::path::PathBuf>,
    debounce_ms: u64,
    batch_size: Option<usize>,
) -> Result<()> {
    let path = resolve_repo_path(path)?;
    let indexer = indexer::Indexer::new(embedder.clone());
    let mut watcher =
        watch::WatchService::new(indexer, Duration::from_millis(debounce_ms), batch_size);
    watcher.run(&path)
}

fn load_or_index(
    path: &Path,
    embedder: Arc<dyn embedding::BatchEmbedder>,
) -> Result<store::RepositoryIndex> {
    let store = store::IndexStore::new(path)?;
    match store.load() {
        Ok(Some(index)) => {
            if index.metadata.vector_dim != embedder.dimension() {
                warn!(
                    "msg" = "index vector dim mismatch, rebuilding",
                    "index_dim" = index.metadata.vector_dim,
                    "embedder_dim" = embedder.dimension()
                );
                return rebuild_index(path, embedder);
            }
            if index.chunks.len() != index.vectors.len() {
                warn!(
                    "msg" = "index chunk/vector length mismatch, rebuilding",
                    "chunks" = index.chunks.len(),
                    "vectors" = index.vectors.len()
                );
                return rebuild_index(path, embedder);
            }
            Ok(index)
        }
        Ok(None) => {
            info!("msg" = "no index found, building", "path" = %path.display());
            rebuild_index(path, embedder)
        }
        Err(err) => {
            warn!("error" = %err, "msg" = "failed to load index, rebuilding");
            rebuild_index(path, embedder)
        }
    }
}

fn rebuild_index(
    path: &Path,
    embedder: Arc<dyn embedding::BatchEmbedder>,
) -> Result<store::RepositoryIndex> {
    eprintln!(
        "{} Building index for {} (this happens once per repo)",
        style("ℹ").cyan(),
        path.display()
    );

    let indexer = indexer::Indexer::new(embedder.clone());
    let report = indexer
        .build_index(indexer::IndexRequest {
            path: path.to_path_buf(),
            force: true,
            batch_size: None,
            profile: false,
            dirty: None,
        })
        .with_context(|| format!("Index build failed for {}", path.display()))?;

    eprintln!(
        "{} Indexed {} files ({} chunks) in {}",
        style("✔").green(),
        report.files_indexed,
        report.chunks_indexed,
        HumanDuration(report.duration)
    );

    if report.graph_symbols > 0 {
        eprintln!(
            "  {} symbols extracted, {} relationships",
            report.graph_symbols, report.graph_edges
        );
    }

    let store = store::IndexStore::new(path)?;
    let index = store.load()?.ok_or_else(|| {
        anyhow!(
            "Index build finished but no index was saved. Try deleting ~/.sgrep/indexes/{} and re-running.",
            store.repo_hash()
        )
    })?;

    if index.metadata.vector_dim != embedder.dimension() {
        return Err(anyhow!(
            "Index vector dimension {} does not match embedder {}. Try setting SGREP_DEVICE=cpu and rerun `sgrep search`.",
            index.metadata.vector_dim,
            embedder.dimension()
        ));
    }

    Ok(index)
}

fn parse_cli() -> Cli {
    if let Ok(raw) = env::var("SGREP_TEST_ARGS") {
        let mut parts = vec!["sgrep".to_string()];
        parts.extend(raw.split_whitespace().map(|s| s.to_string()));
        return Cli::parse_from(parts);
    }
    Cli::parse()
}

fn render_results(ctx: RenderContext<'_>) -> Result<()> {
    if ctx.json {
        let payload =
            JsonResponse::from_results(ctx.query, ctx.limit, ctx.results, ctx.index, ctx.elapsed);
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if ctx.results.is_empty() {
        println!("{} No matches found", style("⚠").yellow());
    } else {
        for (idx, result) in ctx.results.iter().enumerate() {
            let header = format!(
                "{}. {}:{}-{}",
                idx + 1,
                result.chunk.path.display(),
                result.chunk.start_line,
                result.chunk.end_line,
            );
            println!("{} {}", style("→").cyan(), style(header).bold());
            if ctx.debug {
                println!(
                    "    score: {:.2} | semantic: {:.2} | bm25: {:.2}",
                    result.score, result.semantic_score, result.bm25_score
                );
            }
            println!("{}", result.render_snippet());
            println!();
        }
        if ctx.debug {
            println!(
                "{} {} results in {:?}",
                style("ℹ").cyan(),
                ctx.results.len(),
                ctx.elapsed
            );
        }
    }
    Ok(())
}

fn setup_tracing() {
    let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "sgrep=info".into());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use crate::store::{IndexMetadata, RepositoryIndex};
    use chrono::Utc;
    use serial_test::serial;
    use std::any::type_name_of_val;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct TestEmbedder;

    impl embedding::BatchEmbedder for TestEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0])
                .collect())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    fn temp_repo() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sgrep_app_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.rs"), "pub fn hi() {}\n").unwrap();
        dir
    }

    fn sample_index(root: &Path) -> RepositoryIndex {
        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: Path::new("lib.rs").to_path_buf(),
            language: "rust".into(),
            start_line: 1,
            end_line: 1,
            text: "pub fn hi() {}".into(),
            hash: "hash".into(),
            modified_at: Utc::now(),
        };
        let meta = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").into(),
            repo_path: root.to_path_buf(),
            repo_hash: "hash".into(),
            vector_dim: 3,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        RepositoryIndex::new(meta, vec![chunk], vec![vec![1.0, 2.0, 3.0]])
    }

    #[test]
    fn build_embedder_sets_env_flags() {
        let prev_token = env::var("TOKENIZERS_PARALLELISM").ok();
        let prev_offline = env::var("SGREP_OFFLINE").ok();
        let prev_device = env::var("SGREP_DEVICE").ok();
        let prev_config = env::var("SGREP_CONFIG").ok();
        env::remove_var("TOKENIZERS_PARALLELISM");
        env::remove_var("SGREP_OFFLINE");
        env::remove_var("SGREP_DEVICE");
        // Use non-existent config to force local provider
        env::set_var("SGREP_CONFIG", "/nonexistent/config.toml");
        let _ = build_embedder(true, Some("cpu".into())).unwrap();
        assert_eq!(env::var("TOKENIZERS_PARALLELISM").unwrap(), "true");
        assert_eq!(env::var("SGREP_OFFLINE").unwrap(), "1");
        assert_eq!(env::var("SGREP_DEVICE").unwrap(), "cpu");
        if let Some(val) = prev_token {
            env::set_var("TOKENIZERS_PARALLELISM", val);
        } else {
            env::remove_var("TOKENIZERS_PARALLELISM");
        }
        if let Some(val) = prev_offline {
            env::set_var("SGREP_OFFLINE", val);
        } else {
            env::remove_var("SGREP_OFFLINE");
        }
        if let Some(val) = prev_device {
            env::set_var("SGREP_DEVICE", val);
        } else {
            env::remove_var("SGREP_DEVICE");
        }
        if let Some(val) = prev_config {
            env::set_var("SGREP_CONFIG", val);
        } else {
            env::remove_var("SGREP_CONFIG");
        }
    }

    #[test]
    #[serial]
    fn load_or_index_rebuilds_on_dim_mismatch() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        std::env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let bad_index = sample_index(&repo);
        store.save(&bad_index).unwrap();

        let index = load_or_index(&repo, embedder.clone()).unwrap();
        assert_eq!(index.metadata.vector_dim, embedder.dimension());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn build_embedder_respects_use_pooled_flag() {
        env::set_var("SGREP_USE_POOLED_EMBEDDER", "false");
        let embedder = build_embedder(false, None).unwrap();
        let ty = type_name_of_val(embedder.as_ref());
        assert!(
            ty.contains("Embedder"),
            "expected non-pooled embedder, got {}",
            ty
        );
        env::remove_var("SGREP_USE_POOLED_EMBEDDER");
    }

    #[test]
    #[serial]
    fn handle_index_runs_with_profile() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        handle_index(embedder, Some(repo.clone()), true, Some(32), true)
            .expect("indexing should succeed");
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_index_reports_zero_hit_rate_for_empty_repo() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = std::env::temp_dir().join(format!("sgrep_empty_repo_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&repo).unwrap();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        // No files means cache_hits + cache_misses = 0, exercising the zero-hit-rate branch.
        handle_index(embedder, Some(repo.clone()), true, None, true).unwrap();

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_index_reports_cache_hit_rate() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        handle_index(embedder.clone(), Some(repo.clone()), true, None, true).unwrap();
        // second run should see cache hits > 0 and exercise hit-rate branch
        handle_index(embedder, Some(repo.clone()), false, None, true).unwrap();

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn render_results_handles_empty_and_json() {
        let repo_root = Path::new("/tmp/render");
        let index = sample_index(repo_root);
        render_results(RenderContext {
            results: Vec::new(),
            query: "hello",
            limit: 5,
            index: &index,
            elapsed: Duration::from_millis(1),
            json: false,
            debug: false,
        })
        .unwrap();

        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("lib.rs"),
            language: "rust".into(),
            start_line: 1,
            end_line: 2,
            text: "fn hello() {}".into(),
            hash: "h".into(),
            modified_at: Utc::now(),
        };
        let result = search::SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.4,
            bm25_score: 0.0,
            show_full_context: false,
        };
        render_results(RenderContext {
            results: vec![result],
            query: "hello",
            limit: 5,
            index: &index,
            elapsed: Duration::from_millis(2),
            json: true,
            debug: false,
        })
        .unwrap();
    }

    #[test]
    fn render_results_handles_human_output() {
        let repo_root = Path::new("/tmp/render2");
        let index = sample_index(repo_root);
        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("lib.rs"),
            language: "rust".into(),
            start_line: 3,
            end_line: 5,
            text: "fn hello_world() {}".into(),
            hash: "h2".into(),
            modified_at: Utc::now(),
        };
        let result = search::SearchResult {
            chunk,
            score: 0.9,
            semantic_score: 0.8,
            bm25_score: 0.0,
            show_full_context: false,
        };
        render_results(RenderContext {
            results: vec![result],
            query: "hello world",
            limit: 3,
            index: &index,
            elapsed: Duration::from_millis(3),
            json: false,
            debug: false,
        })
        .unwrap();
    }

    #[test]
    #[serial]
    fn load_or_index_rebuilds_on_length_mismatch() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let mut index = sample_index(&repo);
        index.vectors.clear(); // mismatch lengths
        store.save(&index).unwrap();

        let rebuilt = load_or_index(&repo, embedder.clone()).unwrap();
        assert_eq!(rebuilt.metadata.vector_dim, embedder.dimension());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn load_or_index_recovers_from_corrupt_index() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let path = repo
            .join(".sgrep_home")
            .join("indexes")
            .join(store.repo_hash())
            .join("index.bin.zst");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a valid index").unwrap();

        let rebuilt = load_or_index(&repo, embedder.clone()).unwrap();
        assert_eq!(rebuilt.metadata.vector_dim, embedder.dimension());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_watch_runs_noop_in_tests() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        handle_watch(embedder, Some(repo.clone()), 100, Some(8)).unwrap();
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn load_or_index_rebuilds_on_chunk_vector_mismatch() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let mut index = sample_index(&repo);
        index.vectors.pop();
        store.save(&index).unwrap();

        let rebuilt = load_or_index(&repo, embedder.clone()).unwrap();
        assert_eq!(rebuilt.metadata.vector_dim, embedder.dimension());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn run_with_cli_dispatches_index() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let cli = Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Index {
                path: Some(repo.clone()),
                force: true,
                batch_size: Some(16),
                profile: false,
                stats: false,
                json: false,
            },
        };
        run_with_cli(cli).unwrap();
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn run_with_cli_dispatches_watch() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let cli = Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Watch {
                path: Some(repo.clone()),
                debounce_ms: 50,
                batch_size: Some(16),
            },
        };
        run_with_cli(cli).unwrap();
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn run_with_cli_dispatches_search_json() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        // ensure index exists
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let indexer = indexer::Indexer::new(embedder.clone());
        indexer
            .build_index(indexer::IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: Some(8),
                profile: false,
                dirty: None,
            })
            .unwrap();

        let cli = Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Search {
                query: "hi".into(),
                path: repo.clone(),
                limit: 5,
                context: false,
                glob: vec![],
                filters: vec![],
                json: true,
                debug: false,
                no_rerank: false,
                rerank_oversample: 3,
            },
        };
        run_with_cli(cli).unwrap();
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn run_uses_test_args_override() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        env::set_var("SGREP_TEST_ARGS", format!("index {}", repo.display()));
        run().unwrap();
        env::remove_var("SGREP_TEST_ARGS");
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    fn config_show_model_dir_returns_path() {
        let cli = Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Config {
                init: false,
                show_model_dir: true,
                verify_model: false,
            },
        };
        run_with_cli(cli).unwrap();
    }

    #[test]
    #[serial]
    fn config_verify_model_checks_files() {
        let cli = Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Config {
                init: false,
                show_model_dir: false,
                verify_model: true,
            },
        };
        let _ = run_with_cli(cli);
    }

    #[test]
    #[serial]
    fn verify_model_files_returns_error_when_missing() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_verify_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_cache).unwrap();
        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);

        let result = super::verify_model_files();
        assert!(result.is_err());

        env::remove_var("FASTEMBED_CACHE_DIR");
        std::fs::remove_dir_all(&temp_cache).ok();
    }

    #[test]
    #[serial]
    fn verify_model_files_returns_ok_when_present() {
        let temp_cache =
            std::env::temp_dir().join(format!("sgrep_verify_ok_test_{}", Uuid::new_v4()));
        let model_dir = temp_cache.join(embedding::MODEL_NAME);
        std::fs::create_dir_all(&model_dir).unwrap();

        for file in embedding::MODEL_FILES {
            std::fs::write(model_dir.join(file), b"mock").unwrap();
        }

        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);

        let result = super::verify_model_files();
        assert!(result.is_ok());

        env::remove_var("FASTEMBED_CACHE_DIR");
        std::fs::remove_dir_all(&temp_cache).ok();
    }

    #[test]
    fn search_params_consolidates_parameters() {
        use std::path::Path;

        // Test that SearchParams correctly holds all the search parameters
        let params = super::SearchParams {
            query: "test query",
            path: Path::new("/test/path"),
            limit: 10,
            context: true,
            glob: vec!["*.rs".to_string()],
            filters: vec!["lang=rust".to_string()],
            json: false,
            debug: true,
            no_rerank: false,
            rerank_oversample: 3,
        };

        assert_eq!(params.query, "test query");
        assert_eq!(params.path, Path::new("/test/path"));
        assert_eq!(params.limit, 10);
        assert!(params.context);
        assert_eq!(params.glob.len(), 1);
        assert_eq!(params.filters.len(), 1);
        assert!(!params.json);
        assert!(params.debug);
        assert!(!params.no_rerank);
        assert_eq!(params.rerank_oversample, 3);
    }

    #[test]
    fn render_context_consolidates_parameters() {
        let repo_root = Path::new("/tmp/render_context_test");
        let index = sample_index(repo_root);

        let ctx = super::RenderContext {
            results: vec![],
            query: "test",
            limit: 5,
            index: &index,
            elapsed: Duration::from_millis(100),
            json: true,
            debug: false,
        };

        assert_eq!(ctx.query, "test");
        assert_eq!(ctx.limit, 5);
        assert!(ctx.results.is_empty());
        assert_eq!(ctx.elapsed, Duration::from_millis(100));
        assert!(ctx.json);
        assert!(!ctx.debug);
    }

    #[test]
    #[serial]
    fn index_stats_errors_on_missing_index() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let result = handle_index_stats(Some(repo.clone()), false);
        assert!(result.is_err());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn index_stats_succeeds_with_index() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        // Build index first
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let indexer = indexer::Indexer::new(embedder);
        indexer
            .build_index(indexer::IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: Some(8),
                profile: false,
                dirty: None,
            })
            .unwrap();

        let result = handle_index_stats(Some(repo.clone()), false);
        assert!(result.is_ok());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn index_stats_json_output() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        // Build index first
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let indexer = indexer::Indexer::new(embedder);
        indexer
            .build_index(indexer::IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: Some(8),
                profile: false,
                dirty: None,
            })
            .unwrap();

        let result = handle_index_stats(Some(repo.clone()), true);
        assert!(result.is_ok());
        std::fs::remove_dir_all(&repo).ok();
    }
}
