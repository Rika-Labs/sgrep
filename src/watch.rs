use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
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
        
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        ctrlc::set_handler(move || {
            shutdown_clone.store(true, Ordering::Relaxed);
        })?;
        
        let mut last_index = Instant::now() - self.debounce;
        loop {
            if shutdown.load(Ordering::Relaxed) {
                info!("msg" = "shutting down");
                break;
            }
            
            match rx.recv_timeout(Duration::from_millis(100)) {
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
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    warn!("msg" = "watch channel disconnected");
                    break;
                }
            }
        }
        
        Ok(())
    }
}

fn should_reindex(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any
    )
}
