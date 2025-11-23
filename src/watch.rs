use std::path::Path;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{info, warn};

use crate::indexer::{IndexRequest, Indexer};

pub struct WatchService {
    indexer: Indexer,
    debounce: Duration,
}

impl WatchService {
    pub fn new(indexer: Indexer, debounce: Duration) -> Self {
        Self { indexer, debounce }
    }

    pub fn run(&mut self, path: &Path) -> Result<()> {
        let (tx, rx) = channel();
        let mut watcher = RecommendedWatcher::new(tx, Config::default())?;
        watcher.watch(path, RecursiveMode::Recursive)?;
        info!("watching" = %path.display());
        let mut last_index = Instant::now() - self.debounce;
        loop {
            match rx.recv() {
                Ok(Ok(event)) => {
                    if should_reindex(&event.kind) && last_index.elapsed() >= self.debounce {
                        info!("event" = ?event.kind, "msg" = "re-indexing");
                        self.indexer.build_index(IndexRequest {
                            path: path.to_path_buf(),
                            force: true,
                        })?;
                        last_index = Instant::now();
                    }
                }
                Ok(Err(err)) => {
                    warn!("error" = %err, "msg" = "notify error");
                }
                Err(err) => {
                    warn!("error" = %err, "msg" = "watch dropped");
                    return Err(err.into());
                }
            }
        }
    }
}

fn should_reindex(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any
    )
}
