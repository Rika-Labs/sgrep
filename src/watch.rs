use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{info, warn};

use crate::indexer::{IndexRequest, Indexer};

pub struct WatchService {
    indexer: Indexer,
    debounce: Duration,
    batch_size: Option<usize>,
}

impl WatchService {
    pub fn new(indexer: Indexer, debounce: Duration, batch_size: Option<usize>) -> Self {
        Self {
            indexer,
            debounce,
            batch_size,
        }
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
        let mut indexing_thread: Option<thread::JoinHandle<()>> = None;
        let (index_done_tx, index_done_rx) = channel();

        loop {
            if shutdown.load(Ordering::Relaxed) {
                info!("msg" = "shutting down");
                if let Some(handle) = indexing_thread.take() {
                    warn!("msg" = "waiting for indexing to complete");
                    let _ = handle.join();
                }
                break;
            }

            if indexing_thread.is_some() {
                match index_done_rx.try_recv() {
                    Ok(Ok(_)) => {
                        indexing_thread = None;
                    }
                    Ok(Err(e)) => {
                        warn!("error" = %e, "msg" = "indexing failed");
                        indexing_thread = None;
                    }
                    Err(_) => {}
                }
            }

            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(event)) => {
                    if should_reindex(&event.kind)
                        && last_index.elapsed() >= self.debounce
                        && indexing_thread.is_none()
                    {
                        info!("event" = ?event.kind, "msg" = "re-indexing");
                        let indexer = self.indexer.clone();
                        let path = path.to_path_buf();
                        let shutdown_clone = shutdown.clone();
                        let done_tx = index_done_tx.clone();
                        let batch_size = self.batch_size;
                        indexing_thread = Some(thread::spawn(move || {
                            let result = indexer.build_index(IndexRequest {
                                path,
                                force: true,
                                batch_size,
                            });
                            if shutdown_clone.load(Ordering::Relaxed) {
                                warn!("msg" = "indexing cancelled");
                                let _ = done_tx.send(Ok(()));
                                return;
                            }
                            let _ = done_tx.send(result.map(|_| ()));
                        }));
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
