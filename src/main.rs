use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueHint};
use console::style;
use indicatif::HumanDuration;

mod chunker;
mod embedding;
mod fts;
mod indexer;
mod search;
mod store;
mod watch;

use embedding::Embedder;

fn main() -> Result<()> {
    setup_tracing();
    let cli = Cli::parse();
    let embedder = Arc::new(Embedder::default());

    match cli.command {
        Commands::Index { path, force } => {
            let path = resolve_repo_path(path)?;
            let indexer = indexer::Indexer::new(embedder.clone());
            let report = indexer
                .build_index(indexer::IndexRequest {
                    path: path.clone(),
                    force,
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
        } => {
            let store = store::IndexStore::new(&path)?;
            let index = store.load()?.ok_or_else(|| {
                anyhow!(
                    "No index found for {}. Run `sgrep index {}` first.",
                    path.display(),
                    path.display()
                )
            })?;
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
        Commands::Watch { path, debounce_ms } => {
            let path = resolve_repo_path(path)?;
            let indexer = indexer::Indexer::new(embedder.clone());
            let mut watcher = watch::WatchService::new(indexer, Duration::from_millis(debounce_ms));
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
    },
    /// Index a repository for semantic search
    Index {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Force full re-index
        #[arg(short, long)]
        force: bool,
    },
    /// Watch a repository and keep the index fresh
    Watch {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Debounce window in milliseconds
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
    },
}

fn resolve_repo_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(p) => Ok(p),
        None => std::env::current_dir().context("Failed to determine current directory"),
    }
}
