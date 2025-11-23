use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueHint};
use console::style;
use indicatif::HumanDuration;
use serde::Serialize;
use tracing::{info, warn};

mod chunker;
mod embedding;
mod fts;
mod indexer;
mod search;
mod store;
mod watch;

use embedding::Embedder;

#[derive(Serialize)]
struct JsonResponse {
    query: String,
    limit: usize,
    duration_ms: u128,
    results: Vec<JsonMatch>,
    index: JsonIndexMeta,
}

#[derive(Serialize)]
struct JsonMatch {
    path: String,
    start_line: usize,
    end_line: usize,
    language: String,
    score: f32,
    semantic_score: f32,
    keyword_score: f32,
    snippet: String,
}

#[derive(Serialize)]
struct JsonIndexMeta {
    repo_path: String,
    repo_hash: String,
    vector_dim: usize,
    indexed_at: String,
    total_files: usize,
    total_chunks: usize,
}

impl JsonResponse {
    fn from_results(
        query: &str,
        limit: usize,
        results: Vec<search::SearchResult>,
        index: &store::RepositoryIndex,
        duration: Duration,
    ) -> Self {
        let matches = results
            .into_iter()
            .map(|r| JsonMatch {
                path: r.chunk.path.to_string_lossy().to_string(),
                start_line: r.chunk.start_line,
                end_line: r.chunk.end_line,
                language: r.chunk.language.clone(),
                score: r.score,
                semantic_score: r.semantic_score,
                keyword_score: r.keyword_score,
                snippet: r.render_snippet(),
            })
            .collect();

        let index_meta = &index.metadata;
        Self {
            query: query.to_string(),
            limit,
            duration_ms: duration.as_millis(),
            results: matches,
            index: JsonIndexMeta {
                repo_path: index_meta.repo_path.to_string_lossy().to_string(),
                repo_hash: index_meta.repo_hash.clone(),
                vector_dim: index_meta.vector_dim,
                indexed_at: index_meta.indexed_at.to_rfc3339(),
                total_files: index_meta.total_files,
                total_chunks: index_meta.total_chunks,
            },
        }
    }
}

fn main() -> Result<()> {
    setup_tracing();
    let cli = Cli::parse();

    if let Some(device) = &cli.device {
        env::set_var("SGREP_DEVICE", device);
    }

    let embedder = Arc::new(Embedder::default());

    match cli.command {
        Commands::Index {
            path,
            force,
            batch_size,
        } => {
            let path = resolve_repo_path(path)?;
            let indexer = indexer::Indexer::new(embedder.clone());
            let report = indexer
                .build_index(indexer::IndexRequest {
                    path: path.clone(),
                    force,
                    batch_size,
                })
                .context("Failed to build index")?;

            println!(
                "{} Indexed {} files ({} chunks) in {}",
                style("✔").green(),
                report.files_indexed,
                report.chunks_indexed,
                HumanDuration(report.duration)
            );
        }
        Commands::Search {
            query,
            path,
            limit,
            context,
            glob,
            filters,
            json,
        } => {
            let start = Instant::now();
            let index = load_or_index(&path, embedder.clone())?;
            let engine = search::SearchEngine::new(embedder.clone());
            let results = engine.search(
                &index,
                &query,
                search::SearchOptions {
                    limit,
                    include_context: context,
                    glob,
                    filters,
                },
            )?;
            let elapsed = start.elapsed();

            if json {
                let payload = JsonResponse::from_results(&query, limit, results, &index, elapsed);
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                if results.is_empty() {
                    println!("{} No matches found", style("⚠").yellow());
                } else {
                    for (idx, result) in results.iter().enumerate() {
                        let header = format!(
                            "{}. {}:{}-{} ({:.2})",
                            idx + 1,
                            result.chunk.path.display(),
                            result.chunk.start_line,
                            result.chunk.end_line,
                            result.score
                        );
                        println!("{} {}", style("→").cyan(), style(header).bold());
                        println!(
                            "    semantic: {:.2} | keyword: {:.2}",
                            result.semantic_score, result.keyword_score
                        );
                        println!("{}", result.render_snippet());
                        println!();
                    }
                }
            }
        }
        Commands::Watch {
            path,
            debounce_ms,
            batch_size,
        } => {
            let path = resolve_repo_path(path)?;
            let indexer = indexer::Indexer::new(embedder.clone());
            let mut watcher =
                watch::WatchService::new(indexer, Duration::from_millis(debounce_ms), batch_size);
            watcher.run(&path)?;
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

#[derive(Parser)]
#[command(name = "sgrep", version, about = "Lightning-fast semantic code search")]
struct Cli {
    /// Preferred device for embeddings (cpu|cuda|coreml). Also reads SGREP_DEVICE.
    #[arg(global = true, long, env = "SGREP_DEVICE")]
    device: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for code using natural language
    Search {
        /// Query such as "where do we handle authentication?"
        query: String,
        /// Repository path (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Maximum results to return
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// Show extra context around matches
        #[arg(short, long)]
        context: bool,
        /// Restrict to globs (repeatable)
        #[arg(long)]
        glob: Vec<String>,
        /// Simple metadata filters like lang=rust
        #[arg(long)]
        filters: Vec<String>,
        /// Emit structured JSON output for agents
        #[arg(long)]
        json: bool,
    },
    /// Index a repository for semantic search
    Index {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Force full re-index
        #[arg(short, long)]
        force: bool,
        /// Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.
        #[arg(long, env = "SGREP_BATCH_SIZE", value_parser = clap::value_parser!(usize), help = "Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.")]
        batch_size: Option<usize>,
    },
    /// Watch a repository and keep the index fresh
    Watch {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Debounce window in milliseconds
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
        /// Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.
        #[arg(long, env = "SGREP_BATCH_SIZE", value_parser = clap::value_parser!(usize), help = "Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.")]
        batch_size: Option<usize>,
    },
}

fn resolve_repo_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(p) => Ok(p),
        None => std::env::current_dir().context("Failed to determine current directory"),
    }
}

fn load_or_index(path: &Path, embedder: Arc<Embedder>) -> Result<store::RepositoryIndex> {
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

fn rebuild_index(path: &Path, embedder: Arc<Embedder>) -> Result<store::RepositoryIndex> {
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
        })
        .with_context(|| format!("Index build failed for {}", path.display()))?;

    eprintln!(
        "{} Indexed {} files ({} chunks) in {}",
        style("✔").green(),
        report.files_indexed,
        report.chunks_indexed,
        HumanDuration(report.duration)
    );

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
