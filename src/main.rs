use anyhow::Result;

mod app;
mod chunker;
mod cli;
mod config;
mod embedding;
mod fts;
mod graph;
mod indexer;
mod output;
mod reranker;
mod search;
mod store;
mod watch;

fn main() -> Result<()> {
    app::run()
}
