use anyhow::Result;

mod app;
mod chunker;
mod cli;
mod config;
mod embedding;
mod fts;
mod graph;
mod indexer;
mod modal;
mod output;
mod query_expander;
mod reranker;
mod search;
mod store;
mod threading;
mod turbopuffer;
mod watch;

fn main() -> Result<()> {
    app::run()
}
