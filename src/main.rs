use anyhow::Result;

mod app;
mod chunker;
mod cli;
mod embedding;
mod fts;
mod indexer;
mod output;
mod search;
mod store;
mod watch;

fn main() -> Result<()> {
    app::run()
}
