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
mod remote;
mod search;
mod store;
mod threading;
mod turbopuffer;
mod watch;

fn main() {
    let code = match app::run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            2
        }
    };

    std::process::exit(code);
}
