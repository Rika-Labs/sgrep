use once_cell::sync::OnceCell;
use std::env;
use std::ffi::OsString;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use console::{style, Term};
use indicatif::HumanDuration;

fn format_info(msg: &str) -> String {
    format!("{} {}", style("[info]").blue().bold(), msg)
}

fn format_ok(msg: &str) -> String {
    format!("{} {}", style("[ok]").green().bold(), msg)
}

fn format_warn(msg: &str) -> String {
    format!("{} {}", style("[warn]").yellow().bold(), msg)
}

pub fn log_info(msg: &str) {
    eprintln!("{}", format_info(msg));
}

pub fn log_ok(msg: &str) {
    println!("{}", format_ok(msg));
}

pub fn log_warn(msg: &str) {
    eprintln!("{}", format_warn(msg));
}

use crate::chunker;
use crate::cli::{resolve_repo_path, Cli, Commands};
use crate::config::{Config, EmbeddingProviderType, RemoteProviderType};
use crate::embedding::{self, Embedder, EmbeddingModel, PooledEmbedder};
use crate::fts;
use crate::modal::{ModalDeployer, ModalEmbedder};
use crate::output::JsonResponse;
use crate::remote;
use crate::remote::{push_remote_index, RemoteFactory, RemoteVectorStore};
use crate::threading::{CpuPreset, ThreadConfig};
use crate::{indexer, search, store, watch};

pub struct SearchParams<'a> {
    pub query: &'a str,
    pub path: &'a Path,
    pub limit: usize,
    pub context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
    pub json: bool,
    pub debug: bool,
    pub remote: Option<Arc<dyn RemoteVectorStore>>,
}

#[allow(dead_code)]
struct ProgressLine {
    term: Term,
    enabled: bool,
}

#[allow(dead_code)]
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

pub struct RenderContext<'a> {
    pub results: Vec<search::SearchResult>,
    pub query: &'a str,
    pub limit: usize,
    pub index: &'a store::RepositoryIndex,
    pub elapsed: Duration,
    pub json: bool,
    pub debug: bool,
}

fn spawn_detached_without_detach_flag() -> Result<u32> {
    let exe = env::current_exe()?;
    if env::var("SGREP_DETACH_TEST").is_ok() {
        return Ok(0);
    }

    let args: Vec<OsString> = sanitize_detach_args(env::args_os().skip(1));

    let child = Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn detached process")?;

    Ok(child.id())
}

fn sanitize_detach_args<I: Iterator<Item = OsString>>(args: I) -> Vec<OsString> {
    args.filter(|arg| arg != "-d" && arg != "--detach")
        .collect()
}

fn maybe_detach(cli: &Cli) -> Result<Option<(&'static str, u32)>> {
    match &cli.command {
        Commands::Index { stats, detach, .. } => {
            if *detach {
                if *stats {
                    return Err(anyhow!("--detach cannot be used with --stats"));
                }
                return spawn_detached_without_detach_flag().map(|pid| Some(("index", pid)));
            }
        }
        Commands::Watch { detach, .. } => {
            if *detach {
                return spawn_detached_without_detach_flag().map(|pid| Some(("watch", pid)));
            }
        }
        _ => {}
    }
    Ok(None)
}

fn resolve_embedding_model(config: &Config, _cli_model: Option<&str>) -> EmbeddingModel {
    config.embedding.model
}

fn resolve_embedding_provider(
    config: &Config,
    offload_flag: Option<bool>,
) -> EmbeddingProviderType {
    match offload_flag {
        Some(true) => EmbeddingProviderType::Modal,
        Some(false) => EmbeddingProviderType::Local,
        None => config.embedding.provider.clone(),
    }
}

fn resolve_remote_provider(config: &Config, remote_flag: Option<bool>) -> bool {
    match remote_flag {
        Some(true) => true,
        Some(false) => false,
        None => match &config.remote_provider {
            Some(RemoteProviderType::Turbopuffer) | Some(RemoteProviderType::Pinecone) => true,
            Some(RemoteProviderType::None) | None => false,
        },
    }
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

    if let Some((command, pid)) = maybe_detach(&cli)? {
        println!("Detached {command} (pid {pid})");
        return Ok(());
    }

    let offload_flag = match &cli.command {
        Commands::Index { offload, .. } => *offload,
        Commands::Search { offload, .. } => *offload,
        Commands::Watch { offload, .. } => *offload,
        Commands::Config { .. } => None,
    };

    let config = Config::load().unwrap_or_default();
    let model = resolve_embedding_model(&config, None);
    let provider = resolve_embedding_provider(&config, offload_flag);
    let offload = matches!(provider, EmbeddingProviderType::Modal);

    let embedder = build_embedder(
        model,
        cli.offline,
        cli.device.clone(),
        provider.clone(),
        &config,
    )?;

    match cli.command {
        Commands::Index {
            path,
            force,
            batch_size,
            profile,
            stats,
            json,
            offload: _offload,
            remote,
            detach: _,
        } => {
            if stats {
                return handle_index_stats(path, json);
            }
            let use_remote = resolve_remote_provider(&config, remote);
            handle_index(
                embedder, path, force, batch_size, profile, use_remote, None, offload,
            )
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
            offload: _offload,
            remote,
        } => {
            let use_remote = resolve_remote_provider(&config, remote);
            let remote_store = build_remote_store(use_remote, &path)?;
            handle_search(
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
                    remote: remote_store,
                },
            )
        }
        Commands::Watch {
            path,
            debounce_ms,
            batch_size,
            offload: _offload,
            remote,
            detach: _,
        } => {
            let use_remote = resolve_remote_provider(&config, remote);
            handle_watch(
                embedder,
                path,
                debounce_ms,
                batch_size,
                use_remote,
                None,
                offload,
            )
        }
        Commands::Config { .. } => unreachable!(), // Handled above
    }
}

fn build_embedder(
    model: EmbeddingModel,
    offline: bool,
    device: Option<String>,
    provider: EmbeddingProviderType,
    config: &Config,
) -> Result<Arc<dyn embedding::BatchEmbedder>> {
    let _progress = ProgressLine::stderr();

    if env::var("TOKENIZERS_PARALLELISM").is_err() {
        env::set_var("TOKENIZERS_PARALLELISM", "true");
    }

    if offline {
        env::set_var("SGREP_OFFLINE", "1");
    }

    if let Some(device) = device {
        env::set_var("SGREP_DEVICE", device);
    }

    if matches!(provider, EmbeddingProviderType::Modal) {
        return build_modal_embedder(model, config);
    }

    embedding::configure_offline_env(offline)?;

    let use_pooled = env::var("SGREP_USE_POOLED_EMBEDDER")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    #[cfg(not(test))]
    let embedder: Arc<dyn embedding::BatchEmbedder> = if use_pooled {
        Arc::new(PooledEmbedder::new(model, 1, 100_000))
    } else {
        Arc::new(Embedder::new(model, 100_000))
    };

    #[cfg(test)]
    let embedder: Arc<dyn embedding::BatchEmbedder> = if use_pooled {
        Arc::new(PooledEmbedder::default())
    } else {
        Arc::new(Embedder::default())
    };

    Ok(embedder)
}

const MODAL_MAX_TEXTS_PER_REQUEST: usize = 1000;

static MODAL_ENDPOINTS: OnceCell<(String, bool)> = OnceCell::new();

fn resolve_modal_endpoints(config: &Config, json_mode: bool) -> Result<(String, bool)> {
    if let Some(existing) = MODAL_ENDPOINTS.get() {
        return Ok(existing.clone());
    }

    let deployer = ModalDeployer::new(
        config.modal.token_id.clone(),
        config.modal.token_secret.clone(),
    );

    let (embed_url, cached) = deployer.ensure_deployed()?;

    if !json_mode {
        if cached {
            log_info("Using cached Modal endpoints");
        } else {
            log_info("Deployed Modal endpoints");
        }
    }

    let _ = MODAL_ENDPOINTS.set((embed_url.clone(), cached));
    Ok((embed_url, cached))
}

fn build_modal_embedder(
    model: EmbeddingModel,
    config: &Config,
) -> Result<Arc<dyn embedding::BatchEmbedder>> {
    let model_config = model.config();
    let dimension = model_config.output_dim;
    let batch_size = resolve_modal_batch_size(config);
    let concurrency = resolve_modal_concurrency(config);
    let use_gzip = env::var("SGREP_MODAL_GZIP")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let (embed_endpoint, _cached) =
        resolve_modal_endpoints(config, false).context("Modal embedder unavailable")?;

    log_info(&format!(
        "Using Modal embedder (model: {}, GPU: A10G, dim: {}, batch: {}, concurrency: {})",
        model_config.display_name, dimension, batch_size, concurrency
    ));

    let endpoint = if let Some(endpoint) = config.modal.endpoint.clone() {
        log_info(&format!("Using cached endpoint: {}", endpoint));
        endpoint
    } else {
        embed_endpoint
    };

    let embedder = ModalEmbedder::new(
        endpoint,
        dimension,
        config.modal.proxy_token_id.clone(),
        config.modal.proxy_token_secret.clone(),
    )
    .with_batch_size(batch_size)
    .with_concurrency(concurrency)
    .with_gzip(use_gzip);

    Ok(Arc::new(embedder))
}

fn resolve_modal_batch_size(config: &Config) -> usize {
    let env_override = env::var("SGREP_MODAL_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let configured = if config.modal.batch_size == 0 {
        None
    } else {
        Some(config.modal.batch_size)
    };

    let batch_size = env_override.or(configured).unwrap_or(128);
    batch_size.clamp(16, MODAL_MAX_TEXTS_PER_REQUEST)
}

fn resolve_modal_concurrency(config: &Config) -> usize {
    let env_override = env::var("SGREP_MODAL_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());

    let default = default_modal_concurrency();
    let value = env_override.or(config.modal.concurrency).unwrap_or(default);

    value.clamp(1, 64)
}

fn default_modal_concurrency() -> usize {
    let cpus = num_cpus::get();
    cpus.clamp(16, 48)
}

fn handle_config(init: bool, show_model_dir: bool, verify_model: bool) -> Result<()> {
    if show_model_dir {
        let model_dir = embedding::get_fastembed_cache_dir()
            .join(embedding::EmbeddingModel::default().config().name);
        println!("{}", model_dir.display());
        return Ok(());
    }

    if verify_model {
        return verify_model_files();
    }

    let config_path = Config::config_path();

    if init {
        if config_path.exists() {
            log_info(&format!(
                "Config already exists at {}",
                config_path.display()
            ));
        } else {
            let path = Config::create_default_config()?;
            log_ok(&format!("Created config at {}", path.display()));
        }
        return Ok(());
    }

    log_info(&format!("Config path: {}", config_path.display()));

    if config_path.exists() {
        println!(
            "  Provider: {}",
            style(format!(
                "local ({})",
                embedding::EmbeddingModel::default().config().name
            ))
            .bold()
        );
    } else {
        println!("  No config file found (using defaults)");
        println!(
            "  Provider: {}",
            style(format!(
                "local ({})",
                embedding::EmbeddingModel::default().config().name
            ))
            .bold()
        );
        println!();
        println!("  Run 'sgrep config --init' to create a config file");
    }

    Ok(())
}

fn verify_model_files() -> Result<()> {
    let model_dir = embedding::get_fastembed_cache_dir()
        .join(embedding::EmbeddingModel::default().config().name);

    log_info(&format!("Model directory: {}\n", model_dir.display()));

    let config = embedding::EmbeddingModel::default().config();
    let mut all_ok = true;
    for (_, local_name) in config.files {
        let exists = model_dir.join(local_name).exists();
        let status = if exists {
            "OK"
        } else {
            all_ok = false;
            "MISSING"
        };
        println!("  [{}] {}", status, local_name);
    }

    println!();
    if all_ok {
        log_ok("All model files present.");
        Ok(())
    } else {
        println!("[error] Some model files are missing.\n");
        println!("Download from: {}", config.download_base_url);
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

#[allow(clippy::too_many_arguments)]
fn handle_index(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    path: Option<std::path::PathBuf>,
    force: bool,
    batch_size: Option<usize>,
    profile: bool,
    remote_flag: bool,
    remote_override: Option<Arc<dyn RemoteVectorStore>>,
    remote_embedding: bool,
) -> Result<()> {
    let path = resolve_repo_path(path)?;

    let remote_store = if remote_flag {
        if let Some(override_store) = remote_override {
            Some(override_store)
        } else {
            build_remote_store(true, &path)?
        }
    } else {
        None
    };

    let indexer = if remote_embedding {
        indexer::Indexer::with_remote_embedding(embedder.clone(), true)
    } else {
        indexer::Indexer::new(embedder.clone())
    };
    let report = indexer
        .build_index(indexer::IndexRequest {
            path: path.clone(),
            force,
            batch_size,
            profile,
            dirty: None,
        })
        .context("Failed to build index")?;
    let index_duration = report.duration;

    let mut upload_duration = None;

    if let Some(remote) = remote_store.as_ref() {
        let upload_start = Instant::now();

        push_remote_index(&path, remote, true)?;
        log_ok(&format!("Remote index pushed to {}", remote.name()));

        let store_inst = store::IndexStore::new(&path)?;
        if let Ok(Some(graph)) = store_inst.load_graph() {
            remote::graph_blob::push_graph_blob(remote.as_ref(), &graph, embedder.dimension())?;
            log_ok("Graph blob uploaded");
        }

        upload_duration = Some(upload_start.elapsed());
    }

    log_ok(&format!(
        "Indexed {} files ({} chunks) in {}",
        report.files_indexed,
        report.chunks_indexed,
        HumanDuration(report.duration)
    ));

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

    log_info(&format!(
        "Timers: indexed {}, uploaded {}",
        HumanDuration(index_duration),
        upload_duration
            .map(HumanDuration)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    ));

    Ok(())
}

fn handle_search(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    params: SearchParams<'_>,
) -> Result<()> {
    let start = Instant::now();

    if let Some(remote) = params.remote.clone() {
        return handle_remote_search(embedder, params, remote, start);
    }

    let mut engine = search::SearchEngine::new(embedder.clone());

    let store_result = store::IndexStore::new(params.path)?;
    match store_result.build_state() {
        store::BuildState::InProgress(_) => {
            return Err(anyhow!(
                "Index is currently building. Wait for `sgrep index` to finish."
            ));
        }
        store::BuildState::Interrupted(_) => {
            return Err(anyhow!(
                "Indexing was interrupted; rerun `sgrep index --force` to rebuild."
            ));
        }
        _ => {}
    }

    if let Ok(Some(graph)) = store_result.load_graph() {
        engine.set_graph(graph);
    }

    if let Some(mmap_index) = store_result.load_mmap()? {
        if mmap_index.metadata.vector_dim != embedder.dimension() {
            return Err(anyhow!(
                "Index vector dimension {} does not match embedder {}. Re-run `sgrep index --force`.",
                mmap_index.metadata.vector_dim,
                embedder.dimension()
            ));
        }

        store::validate_index_model(
            &mmap_index.metadata.embedding_model,
            embedding::EmbeddingModel::default().config().name,
        )?;

        let results = engine.search_mmap(
            &mmap_index,
            params.query,
            search::SearchOptions {
                limit: params.limit,
                include_context: params.context,
                glob: params.glob.clone(),
                filters: params.filters.clone(),
                dedup: search::DedupOptions::default(),
                file_type_priority: search::FileTypePriority::default(),
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

    let index = load_index_for_search(params.path, &store_result, embedder.clone())?;
    let results = engine.search(
        &index,
        params.query,
        search::SearchOptions {
            limit: params.limit,
            include_context: params.context,
            glob: params.glob,
            filters: params.filters,
            dedup: search::DedupOptions::default(),
            file_type_priority: search::FileTypePriority::default(),
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

fn handle_remote_search(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    params: SearchParams<'_>,
    remote: Arc<dyn RemoteVectorStore>,
    start: Instant,
) -> Result<()> {
    let graph = remote::graph_blob::fetch_graph_blob(remote.as_ref(), embedder.dimension())?;

    let query_vec = embedder.embed(params.query)?;
    let oversample = params.limit * 4;
    let hits = remote.query(&query_vec, oversample.max(20))?;

    if hits.is_empty() {
        let store_inst = store::IndexStore::new(params.path)?;
        let metadata = store::IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: params.path.to_path_buf(),
            repo_hash: store_inst.repo_hash().to_string(),
            vector_dim: embedder.dimension(),
            indexed_at: chrono::Utc::now(),
            total_files: 0,
            total_chunks: 0,
            embedding_model: embedding::EmbeddingModel::default()
                .config()
                .name
                .to_string(),
        };
        let index = store::RepositoryIndex::new(metadata, Vec::new(), Vec::new());
        return render_results(RenderContext {
            results: Vec::new(),
            query: params.query,
            limit: params.limit,
            index: &index,
            elapsed: start.elapsed(),
            json: params.json,
            debug: params.debug,
        });
    }

    let chunks: Vec<chunker::CodeChunk> = hits
        .iter()
        .map(|h| chunker::CodeChunk {
            id: uuid::Uuid::new_v4(),
            path: std::path::Path::new(&h.path).to_path_buf(),
            language: h.language.clone(),
            start_line: h.start_line,
            end_line: h.end_line,
            text: h.content.clone(),
            hash: h.id.clone(),
            modified_at: chrono::Utc::now(),
        })
        .collect();

    let chunk_refs: Vec<&chunker::CodeChunk> = chunks.iter().collect();
    let symbols: Vec<Vec<String>> = hits.iter().map(|h| h.symbols.clone()).collect();
    let bm25_index = fts::build_bm25f_index(&chunk_refs, Some(&symbols));

    let mut results: Vec<search::SearchResult> = hits
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            let semantic_score = h.score;
            let bm25_score = bm25_index.score(params.query, idx);
            let bm25_normalized = (bm25_score / 10.0).min(1.0);
            let combined = 0.85 * semantic_score + 0.15 * bm25_normalized;

            search::SearchResult {
                chunk: chunks[idx].clone(),
                score: combined,
                semantic_score,
                bm25_score,
                show_full_context: params.context,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(ref g) = graph {
        for result in &mut results {
            let chunk_text_lower = result.chunk.text.to_lowercase();
            let query_lower = params.query.to_lowercase();
            for word in query_lower.split_whitespace() {
                if let Some(symbols) = g.name_index.get(word) {
                    if !symbols.is_empty() && chunk_text_lower.contains(word) {
                        result.score *= 1.1;
                    }
                }
            }
        }
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let file_type_priority = search::FileTypePriority::default();
    for result in &mut results {
        let multiplier = file_type_priority.multiplier(search::classify_path(&result.chunk.path));
        result.score *= multiplier;
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(params.limit);

    let store_inst = store::IndexStore::new(params.path)?;
    let metadata = store::IndexMetadata {
        version: env!("CARGO_PKG_VERSION").to_string(),
        repo_path: params.path.to_path_buf(),
        repo_hash: store_inst.repo_hash().to_string(),
        vector_dim: embedder.dimension(),
        indexed_at: chrono::Utc::now(),
        total_files: 0,
        total_chunks: results.len(),
        embedding_model: embedding::EmbeddingModel::default()
            .config()
            .name
            .to_string(),
    };
    let index = store::RepositoryIndex::new(metadata, Vec::new(), Vec::new());

    render_results(RenderContext {
        results,
        query: params.query,
        limit: params.limit,
        index: &index,
        elapsed: start.elapsed(),
        json: params.json,
        debug: params.debug,
    })
}

fn build_remote_store(use_remote: bool, path: &Path) -> Result<Option<Arc<dyn RemoteVectorStore>>> {
    if !use_remote {
        return Ok(None);
    }

    let mut config = Config::load().unwrap_or_default();

    let store_inst = store::IndexStore::new(path)?;
    let repo_hash = store_inst.repo_hash().to_string();

    if config.remote_provider == Some(RemoteProviderType::None) {
        config.remote_provider = None;
    }

    let remote = RemoteFactory::build_from_config(&config, &repo_hash)?;

    if remote.is_none() {
        return Err(anyhow!(
            "Remote storage requested but no provider configured.\n\
             Add to config.toml:\n\n\
             remote_provider = \"turbopuffer\"  # or \"pinecone\"\n\n\
             [turbopuffer]\n\
             api_key = \"your-api-key\"\n\n\
             Or use --remote=false to disable."
        ));
    }

    Ok(remote)
}

fn handle_watch(
    embedder: Arc<dyn embedding::BatchEmbedder>,
    path: Option<std::path::PathBuf>,
    debounce_ms: u64,
    batch_size: Option<usize>,
    remote_flag: bool,
    remote_override: Option<Arc<dyn RemoteVectorStore>>,
    remote_embedding: bool,
) -> Result<()> {
    let path = resolve_repo_path(path)?;
    let remote_store = if remote_flag {
        if let Some(override_store) = remote_override {
            Some(override_store)
        } else {
            build_remote_store(true, &path)?
        }
    } else {
        None
    };
    let indexer = if remote_embedding {
        indexer::Indexer::with_remote_embedding(embedder.clone(), true)
    } else {
        indexer::Indexer::new(embedder.clone())
    };
    let mut watcher = watch::WatchService::new(
        indexer,
        Duration::from_millis(debounce_ms),
        batch_size,
        remote_store,
    );
    watcher.run(&path)
}

fn load_index_for_search(
    path: &Path,
    store: &store::IndexStore,
    embedder: Arc<dyn embedding::BatchEmbedder>,
) -> Result<store::RepositoryIndex> {
    let index = match store.load() {
        Ok(Some(index)) => index,
        Ok(None) => {
            return Err(anyhow!(
                "No index found for {}. Run `sgrep index` first.",
                path.display()
            ))
        }
        Err(err) => {
            return Err(anyhow!(
                "Failed to load index for {}: {err}. Re-run `sgrep index --force`.",
                path.display()
            ))
        }
    };

    if index.metadata.vector_dim != embedder.dimension() {
        return Err(anyhow!(
            "Index vector dimension {} does not match embedder {}. Re-run `sgrep index --force`.",
            index.metadata.vector_dim,
            embedder.dimension()
        ));
    }

    // Validate embedding model matches
    store::validate_index_model(
        &index.metadata.embedding_model,
        embedding::EmbeddingModel::default().config().name,
    )?;

    if index.chunks.len() != index.vectors.len() {
        return Err(anyhow!(
            "Index chunks ({}) do not match vectors ({}). Re-run `sgrep index --force`.",
            index.chunks.len(),
            index.vectors.len()
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
        log_warn("No matches found");
    } else {
        for (idx, result) in ctx.results.iter().enumerate() {
            let header = format!(
                "{}. {}:{}-{}",
                idx + 1,
                result.chunk.path.display(),
                result.chunk.start_line,
                result.chunk.end_line,
            );
            println!("-> {}", style(header).bold());
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
            log_info(&format!(
                "{} results in {:?}",
                ctx.results.len(),
                ctx.elapsed
            ));
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
    use crate::remote::{RemoteChunk, RemoteSearchHit};
    use crate::store::utils::data_dir;
    use crate::store::{IndexMetadata, RepositoryIndex};
    use chrono::Utc;
    use serial_test::serial;
    use std::any::type_name_of_val;
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Mutex, Once};
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

    #[derive(Clone, Default)]
    struct WideTestEmbedder;

    impl embedding::BatchEmbedder for WideTestEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
        }

        fn dimension(&self) -> usize {
            384
        }
    }

    fn temp_repo() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("sgrep_app_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lib.rs"), "pub fn hi() {}\n").unwrap();
        dir
    }

    static DETACH_ENV: Once = Once::new();

    fn clear_detach_env() {
        env::remove_var("SGREP_DETACH_TEST");
    }

    fn cli_for_index_detach(detach: bool, stats: bool) -> Cli {
        Cli {
            device: None,
            offline: false,
            max_threads: None,
            cpu_preset: None,
            command: Commands::Index {
                path: None,
                force: false,
                batch_size: None,
                profile: false,
                stats,
                json: false,
                offload: Some(false),
                remote: None,
                detach,
            },
        }
    }

    #[test]
    fn sanitize_detach_args_strips_flags() {
        let args = vec![
            OsString::from("index"),
            OsString::from("-d"),
            OsString::from("--detach"),
            OsString::from("path"),
        ];
        let filtered = sanitize_detach_args(args.into_iter());
        assert_eq!(
            filtered,
            vec![OsString::from("index"), OsString::from("path")]
        );
    }

    #[test]
    fn maybe_detach_rejects_stats() {
        let cli = cli_for_index_detach(true, true);
        let result = maybe_detach(&cli);
        assert!(result.is_err());
    }

    #[test]
    fn maybe_detach_returns_pid_when_env_set() {
        DETACH_ENV.call_once(clear_detach_env);
        env::set_var("SGREP_DETACH_TEST", "1");
        let cli = cli_for_index_detach(true, false);
        let result = maybe_detach(&cli).unwrap();
        env::remove_var("SGREP_DETACH_TEST");
        assert_eq!(result, Some(("index", 0)));
    }

    #[test]
    #[serial]
    fn modal_batch_size_defaults_and_env_override() {
        env::remove_var("SGREP_MODAL_BATCH_SIZE");

        let mut config = Config::default();
        config.modal.batch_size = 0;
        assert_eq!(resolve_modal_batch_size(&config), 128);

        config.modal.batch_size = 2000;
        assert_eq!(
            resolve_modal_batch_size(&config),
            MODAL_MAX_TEXTS_PER_REQUEST
        );

        env::set_var("SGREP_MODAL_BATCH_SIZE", "32");
        config.modal.batch_size = 0;
        assert_eq!(resolve_modal_batch_size(&config), 32);
        env::remove_var("SGREP_MODAL_BATCH_SIZE");
    }

    #[test]
    #[serial]
    fn modal_concurrency_prefers_env_and_clamps() {
        env::remove_var("SGREP_MODAL_CONCURRENCY");

        let mut config = Config::default();
        config.modal.concurrency = Some(100);
        assert_eq!(resolve_modal_concurrency(&config), 64);

        config.modal.concurrency = Some(4);
        assert_eq!(resolve_modal_concurrency(&config), 4);

        env::set_var("SGREP_MODAL_CONCURRENCY", "1");
        assert_eq!(resolve_modal_concurrency(&config), 1);

        env::set_var("SGREP_MODAL_CONCURRENCY", "100");
        assert_eq!(resolve_modal_concurrency(&config), 64);
        env::remove_var("SGREP_MODAL_CONCURRENCY");
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
            embedding_model: embedding::EmbeddingModel::default()
                .config()
                .name
                .to_string(),
        };
        RepositoryIndex::new(meta, vec![chunk], vec![vec![1.0, 2.0, 3.0]])
    }

    #[derive(Default)]
    struct MockRemoteStore {
        upserts: Mutex<Vec<Vec<RemoteChunk>>>,
        queries: Mutex<Vec<(Vec<f32>, usize)>>,
        query_hits: Mutex<Vec<RemoteSearchHit>>,
    }

    impl MockRemoteStore {
        fn with_hits(hits: Vec<RemoteSearchHit>) -> Self {
            Self {
                query_hits: Mutex::new(hits),
                ..Default::default()
            }
        }

        fn upsert_called(&self) -> bool {
            !self.upserts.lock().unwrap().is_empty()
        }

        fn last_upsert_count(&self) -> Option<usize> {
            self.upserts.lock().unwrap().last().map(|c| c.len())
        }

        fn query_called(&self) -> bool {
            !self.queries.lock().unwrap().is_empty()
        }
    }

    impl RemoteVectorStore for MockRemoteStore {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()> {
            self.upserts.lock().unwrap().push(chunks.to_vec());
            Ok(())
        }

        fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<RemoteSearchHit>> {
            self.queries.lock().unwrap().push((vector.to_vec(), top_k));

            let hits = self.query_hits.lock().unwrap().clone();
            Ok(hits.into_iter().take(top_k).collect())
        }

        fn delete_namespace(&self) -> Result<()> {
            Ok(())
        }
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
        let _ = build_embedder(
            EmbeddingModel::default(),
            true,
            Some("cpu".into()),
            EmbeddingProviderType::Local,
            &Config::default(),
        )
        .unwrap();
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
    fn build_embedder_respects_use_pooled_flag() {
        env::set_var("SGREP_USE_POOLED_EMBEDDER", "false");
        let embedder = build_embedder(
            EmbeddingModel::default(),
            false,
            None,
            EmbeddingProviderType::Local,
            &Config::default(),
        )
        .unwrap();
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
        handle_index(
            embedder,
            Some(repo.clone()),
            true,
            Some(32),
            true,
            false,
            None,
            false,
        )
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
        handle_index(
            embedder,
            Some(repo.clone()),
            true,
            None,
            true,
            false,
            None,
            false,
        )
        .unwrap();

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_index_reports_cache_hit_rate() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        handle_index(
            embedder.clone(),
            Some(repo.clone()),
            true,
            None,
            true,
            false,
            None,
            false,
        )
        .unwrap();
        // second run should see cache hits > 0 and exercise hit-rate branch
        handle_index(
            embedder,
            Some(repo.clone()),
            false,
            None,
            true,
            false,
            None,
            false,
        )
        .unwrap();

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn search_uses_remote_store_when_flagged() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));

        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());

        let indexer = indexer::Indexer::new(embedder.clone());
        indexer
            .build_index(indexer::IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: Some(4),
                profile: false,
                dirty: None,
            })
            .unwrap();

        let hits = vec![RemoteSearchHit {
            id: "test-chunk-1".to_string(),
            score: 0.95,
            path: "lib.rs".to_string(),
            start_line: 1,
            end_line: 1,
            content: "pub fn hi() {}".to_string(),
            language: "rust".to_string(),
            symbols: Vec::new(),
        }];

        let remote = Arc::new(MockRemoteStore::with_hits(hits));

        handle_search(
            embedder,
            SearchParams {
                query: "hi",
                path: repo.as_path(),
                limit: 1,
                context: false,
                glob: vec![],
                filters: vec![],
                json: true,
                debug: false,
                remote: Some(remote.clone()),
            },
        )
        .unwrap();

        assert!(remote.query_called());
        std::fs::remove_dir_all(&repo).ok();
        env::remove_var("SGREP_HOME");
    }

    #[test]
    #[serial]
    fn search_uses_local_index_when_remote_not_requested() {
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());

        let indexer = indexer::Indexer::new(embedder.clone());
        indexer
            .build_index(indexer::IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: Some(4),
                profile: false,
                dirty: None,
            })
            .unwrap();

        handle_search(
            embedder,
            SearchParams {
                query: "hi",
                path: repo.as_path(),
                limit: 1,
                context: false,
                glob: vec![],
                filters: vec![],
                json: true,
                debug: false,
                remote: None,
            },
        )
        .unwrap();

        let store = store::IndexStore::new(repo.as_path()).unwrap();
        assert!(store.load().unwrap().is_some());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn index_pushes_remote_when_flag_true() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let remote = Arc::new(MockRemoteStore::default());

        handle_index(
            embedder,
            Some(repo.clone()),
            true,
            Some(16),
            false,
            true,
            Some(remote.clone()),
            false,
        )
        .unwrap();

        assert!(remote.upsert_called());
        assert!(remote.last_upsert_count().unwrap_or_default() > 0);
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn index_saves_locally_when_remote_flag_absent() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let remote = Arc::new(MockRemoteStore::default());

        handle_index(
            embedder,
            Some(repo.clone()),
            true,
            None,
            false,
            false,
            Some(remote.clone()),
            false,
        )
        .unwrap();

        assert!(!remote.upsert_called());
        let store = store::IndexStore::new(repo.as_path()).unwrap();
        assert!(store.load().unwrap().is_some());
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
    fn load_index_for_search_errors_on_missing_index() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();

        let err = load_index_for_search(&repo, &store, embedder.clone()).unwrap_err();
        assert!(format!("{err}").contains("No index found"));

        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn load_index_for_search_errors_on_dim_mismatch() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let bad_index = sample_index(&repo);
        store.save(&bad_index).unwrap();

        let err = load_index_for_search(&repo, &store, embedder.clone()).unwrap_err();
        assert!(format!("{err}").contains("vector dimension"));
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn load_index_for_search_errors_on_length_mismatch() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let mut index = sample_index(&repo);
        index.vectors.clear();
        index.metadata.vector_dim = embedder.dimension();
        store.save(&index).unwrap();

        let err = load_index_for_search(&repo, &store, embedder.clone()).unwrap_err();
        assert!(format!("{err}").contains("Re-run `sgrep index --force`"));
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn load_index_for_search_errors_on_corrupt_index() {
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

        let err = load_index_for_search(&repo, &store, embedder.clone()).unwrap_err();
        assert!(format!("{err}").contains("Failed to load index"));
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_watch_runs_noop_in_tests() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        handle_watch(
            embedder,
            Some(repo.clone()),
            100,
            Some(8),
            false,
            None,
            false,
        )
        .unwrap();
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_watch_pushes_remote_index() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let remote = Arc::new(MockRemoteStore::default());

        handle_watch(
            embedder,
            Some(repo.clone()),
            50,
            Some(4),
            true,
            Some(remote.clone()),
            false,
        )
        .unwrap();

        assert!(remote.upsert_called());
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_search_errors_when_build_in_progress() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();
        let _guard = store.start_build_guard().unwrap();

        let err = handle_search(
            embedder,
            SearchParams {
                query: "hi",
                path: repo.as_path(),
                limit: 1,
                context: false,
                glob: vec![],
                filters: vec![],
                json: false,
                debug: false,
                remote: None,
            },
        )
        .unwrap_err();

        assert!(
            format!("{err}").contains("Index is currently building"),
            "expected in-progress message, got: {err}"
        );
        std::fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn handle_search_errors_when_indexing_interrupted() {
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(TestEmbedder::default());
        let repo = temp_repo();
        env::set_var("SGREP_HOME", repo.join(".sgrep_home"));
        let store = store::IndexStore::new(&repo).unwrap();

        let mut marker_path = data_dir();
        marker_path.push("indexes");
        marker_path.push(store.repo_hash());
        marker_path.push("index.building");

        std::fs::write(&marker_path, b"building").unwrap();

        let err = handle_search(
            embedder,
            SearchParams {
                query: "hi",
                path: repo.as_path(),
                limit: 1,
                context: false,
                glob: vec![],
                filters: vec![],
                json: false,
                debug: false,
                remote: None,
            },
        )
        .unwrap_err();

        assert!(
            format!("{err}")
                .contains("Indexing was interrupted; rerun `sgrep index --force` to rebuild."),
            "expected interrupted message, got: {err}"
        );
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
                offload: Some(false),
                remote: None,
                detach: false,
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
                offload: Some(false),
                remote: None,
                detach: false,
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
        let embedder: Arc<dyn embedding::BatchEmbedder> = Arc::new(WideTestEmbedder::default());
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
                offload: Some(false),
                remote: None,
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
        let model_dir = temp_cache.join(embedding::EmbeddingModel::default().config().name);
        std::fs::create_dir_all(&model_dir).unwrap();

        let config = embedding::EmbeddingModel::default().config();
        for (_, local_name) in config.files {
            std::fs::write(model_dir.join(local_name), b"mock").unwrap();
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

        let params = super::SearchParams {
            query: "test query",
            path: Path::new("/test/path"),
            limit: 10,
            context: true,
            glob: vec!["*.rs".to_string()],
            filters: vec!["lang=rust".to_string()],
            json: false,
            debug: true,
            remote: None,
        };

        assert_eq!(params.query, "test query");
        assert_eq!(params.path, Path::new("/test/path"));
        assert_eq!(params.limit, 10);
        assert!(params.context);
        assert_eq!(params.glob.len(), 1);
        assert_eq!(params.filters.len(), 1);
        assert!(!params.json);
        assert!(params.debug);
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

    #[test]
    fn log_formatters_include_prefix_and_message() {
        let msg = "hello";
        let info = format_info(msg);
        let ok = format_ok(msg);
        let warn = format_warn(msg);

        assert!(info.contains("[info]") && info.contains(msg));
        assert!(ok.contains("[ok]") && ok.contains(msg));
        assert!(warn.contains("[warn]") && warn.contains(msg));
    }

    #[test]
    fn resolve_remote_provider_flag_true_overrides_config() {
        let mut config = Config::default();
        config.remote_provider = Some(RemoteProviderType::None);
        assert!(resolve_remote_provider(&config, Some(true)));
    }

    #[test]
    fn resolve_remote_provider_flag_false_overrides_config() {
        let mut config = Config::default();
        config.remote_provider = Some(RemoteProviderType::Turbopuffer);
        assert!(!resolve_remote_provider(&config, Some(false)));
    }

    #[test]
    fn resolve_remote_provider_uses_config_turbopuffer() {
        let mut config = Config::default();
        config.remote_provider = Some(RemoteProviderType::Turbopuffer);
        assert!(resolve_remote_provider(&config, None));
    }

    #[test]
    fn resolve_remote_provider_uses_config_pinecone() {
        let mut config = Config::default();
        config.remote_provider = Some(RemoteProviderType::Pinecone);
        assert!(resolve_remote_provider(&config, None));
    }

    #[test]
    fn resolve_remote_provider_config_none_returns_false() {
        let mut config = Config::default();
        config.remote_provider = Some(RemoteProviderType::None);
        assert!(!resolve_remote_provider(&config, None));
    }

    #[test]
    fn resolve_remote_provider_missing_returns_false() {
        let config = Config::default();
        assert!(!resolve_remote_provider(&config, None));
    }
}
