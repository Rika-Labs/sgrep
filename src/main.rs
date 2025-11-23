use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Args, Parser, Subcommand};

mod chunker;
mod embedding;
mod fts;
mod indexer;
mod remote;
mod search;
mod store;
mod watch;

#[derive(Parser)]
#[command(name = "sgrep")]
#[command(about = "Semantic grep for your codebase", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Search {
        query: String,
        #[arg(long)]
        json: bool,
        #[arg(short = 'm', long = "max", default_value_t = 25)]
        max: usize,
        #[arg(long = "per-file", default_value_t = 1)]
        per_file: usize,
        #[arg(long)]
        remote: bool,
        #[arg(long)]
        sync: bool,
        #[arg(long)]
        content: bool,
        #[arg(long)]
        scores: bool,
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        paths: Option<String>,
        #[arg(long)]
        ignore: Option<String>,
    },
    Index {
        path: Option<PathBuf>,
        #[arg(long)]
        remote: bool,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        force: bool,
        #[arg(long = "dry-run")]
        dry_run: bool,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        include_md: bool,
        #[arg(long = "no-md", action = ArgAction::SetTrue)]
        no_md: bool,
    },
    Watch(WatchArgs),
    Setup,
    List,
    Doctor,
}

#[derive(Args)]
struct WatchArgs {
    #[arg(long)]
    list: bool,
    #[arg(long)]
    add: Option<PathBuf>,
    #[arg(long)]
    remove: Option<PathBuf>,
    #[arg(long)]
    clear: bool,
    #[arg(long)]
    path: Option<PathBuf>,
    #[arg(long, default_value_t = 750)]
    debounce_ms: u64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:?}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let embedder = embedding::Embedder::from_env()?;
    match cli.command {
        Command::Search {
            query,
            json,
            max,
            per_file,
            remote,
            sync,
            content,
            scores,
            lang,
            paths,
            ignore,
        } => handle_search(
            query, json, max, per_file, remote, sync, content, scores, lang, paths, ignore,
            &embedder,
        )?,
        Command::Index {
            path,
            remote,
            force,
            dry_run,
            include_md,
            no_md,
        } => handle_index(path, remote, force, dry_run, include_md, no_md, &embedder)?,
        Command::Watch(args) => handle_watch(args, &embedder)?,
        Command::Setup => handle_setup()?,
        Command::List => handle_list()?,
        Command::Doctor => handle_doctor()?,
    }
    Ok(())
}

fn handle_search(
    query: String,
    json: bool,
    max: usize,
    per_file: usize,
    remote_backend: bool,
    sync_index: bool,
    include_content: bool,
    show_scores: bool,
    lang: Option<String>,
    paths: Option<String>,
    ignore: Option<String>,
    embedder: &embedding::Embedder,
) -> Result<()> {
    if query.trim().is_empty() {
        return Ok(());
    }
    let root = std::env::current_dir()?;
    if !remote_backend {
        let options = indexer::IndexOptions {
            root: root.clone(),
            force_reindex: sync_index,
            dry_run: false,
            include_markdown: true,
        };
        let _ = indexer::index_repository(embedder, options)?;
    }
    let response = if remote_backend {
        remote::remote_search(
            &root,
            embedder,
            search::SearchConfig {
                query,
                max_results: max,
                per_file,
                include_content,
                lang_filter: lang,
                path_filter: paths,
                ignore_filter: ignore,
            },
        )?
    } else {
        search::search(
            &root,
            embedder,
            search::SearchConfig {
                query,
                max_results: max,
                per_file,
                include_content,
                lang_filter: lang,
                path_filter: paths,
                ignore_filter: ignore,
            },
        )?
    };
    if response.total == 0 {
        return Ok(());
    }
    if json {
        let value = serde_json::to_string_pretty(&response)?;
        println!("{}", value);
        return Ok(());
    }
    for item in response.matches.iter() {
        if show_scores {
            println!(
                "{}:{}-{} [{:.3} s:{:.3} k:{:.3}]: {}",
                item.path,
                item.start_line,
                item.end_line,
                item.score,
                item.semantic_score,
                item.keyword_score,
                item.snippet.replace('\n', " "),
            );
        } else {
            println!(
                "{}:{}-{}: {}",
                item.path,
                item.start_line,
                item.end_line,
                item.snippet.replace('\n', " "),
            );
        }
    }
    Ok(())
}

fn handle_index(
    path: Option<PathBuf>,
    remote_backend: bool,
    force_reindex: bool,
    dry_run: bool,
    include_md: bool,
    no_md: bool,
    embedder: &embedding::Embedder,
) -> Result<()> {
    let target = match path {
        Some(p) => p,
        None => std::env::current_dir()?,
    };
    let should_force = force_reindex || !store::index_exists(&target)?;
    let options = indexer::IndexOptions {
        root: target.clone(),
        force_reindex: should_force,
        dry_run,
        include_markdown: if no_md { false } else { include_md },
    };
    let stats = indexer::index_repository(embedder, options)?;
    if stats.skipped_existing {
        println!("index already exists, skipping (use --force to rebuild)");
    } else if dry_run {
        println!("would index {}", target.display());
    } else {
        println!(
            "indexed {} chunks across {} files",
            stats.chunks, stats.files
        );
    }
    if remote_backend {
        let remote_count = remote::remote_index(&target)?;
        println!("remote indexed {} chunks", remote_count);
    }
    Ok(())
}

fn handle_watch(args: WatchArgs, embedder: &embedding::Embedder) -> Result<()> {
    if args.list {
        let watched = watch::list()?;
        if watched.is_empty() {
            println!("no watched paths");
        } else {
            for path in watched {
                println!("{}", path.display());
            }
        }
        return Ok(());
    }
    if args.clear {
        watch::clear()?;
        println!("cleared watch list");
        return Ok(());
    }
    if let Some(path) = args.add {
        let added = watch::add(&path)?;
        if added {
            println!("added {}", path.display());
        } else {
            println!("already watching {}", path.display());
        }
        return Ok(());
    }
    if let Some(path) = args.remove {
        let removed = watch::remove(&path)?;
        if removed {
            println!("removed {}", path.display());
        } else {
            println!("not watching {}", path.display());
        }
        return Ok(());
    }
    watch::run_watch(embedder, args.path, Some(args.debounce_ms))
}

fn handle_setup() -> Result<()> {
    let dir = store::stores_dir()?;
    std::fs::create_dir_all(&dir)?;
    println!("data directory: {}", dir.display());
    Ok(())
}

fn handle_list() -> Result<()> {
    let stores = store::list_stores()?;
    if stores.is_empty() {
        return Ok(());
    }
    for store in stores.iter() {
        println!("{} -> {}", store.id, store.index_path.display());
    }
    Ok(())
}

fn handle_doctor() -> Result<()> {
    let dir = store::stores_dir()?;
    let stores = store::list_stores()?;
    println!("data directory: {}", dir.display());
    println!("stores: {}", stores.len());
    Ok(())
}
