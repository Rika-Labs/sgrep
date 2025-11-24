use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tracing::{info, warn};

use crate::indexer::{DirtySet, IndexRequest, Indexer};

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

        let mut last_event: Option<Instant> = None;
        let mut indexing_thread: Option<thread::JoinHandle<()>> = None;
        let (index_done_tx, index_done_rx) = channel();
        let mut dirty_paths: HashSet<std::path::PathBuf> = HashSet::new();
        let mut deleted_paths: HashSet<std::path::PathBuf> = HashSet::new();

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
                    if should_reindex(&event.kind) {
                        categorize_event(&event, &mut dirty_paths, &mut deleted_paths);
                        last_event = Some(Instant::now());
                    }
                }
                Ok(Err(err)) => {
                    warn!("error" = %err, "msg" = "notify error");
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    warn!("msg" = "watch channel disconnected");
                    break;
                }
            }

            let ready = last_event
                .map(|t| t.elapsed() >= self.debounce)
                .unwrap_or(false)
                && indexing_thread.is_none()
                && (!dirty_paths.is_empty() || !deleted_paths.is_empty());

            if ready {
                info!(
                    "msg" = "incremental re-indexing",
                    "dirty" = dirty_paths.len(),
                    "deleted" = deleted_paths.len()
                );
                let indexer = self.indexer.clone();
                let path = path.to_path_buf();
                let shutdown_clone = shutdown.clone();
                let done_tx = index_done_tx.clone();
                let batch_size = self.batch_size;
                let dirty_set = DirtySet {
                    touched: dirty_paths.drain().collect(),
                    deleted: deleted_paths.drain().collect(),
                };

                indexing_thread = Some(thread::spawn(move || {
                    let result = indexer.build_index(IndexRequest {
                        path,
                        force: false,
                        batch_size,
                        profile: false,
                        dirty: Some(dirty_set),
                    });
                    if shutdown_clone.load(Ordering::Relaxed) {
                        warn!("msg" = "indexing cancelled");
                        let _ = done_tx.send(Ok(()));
                        return;
                    }
                    let _ = done_tx.send(result.map(|_| ()));
                }));
                last_event = None;
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

fn categorize_event(
    event: &notify::Event,
    dirty: &mut HashSet<std::path::PathBuf>,
    deleted: &mut HashSet<std::path::PathBuf>,
) {
    match &event.kind {
        EventKind::Remove(_) => {
            for path in &event.paths {
                deleted.insert(path.clone());
            }
        }
        EventKind::Modify(modify) => {
            // Rename events carry both old and new paths. Prefer to treat first as deleted and last as touched.
            if let notify::event::ModifyKind::Name(_) = modify {
                if let Some((first, rest)) = event.paths.split_first() {
                    deleted.insert(first.clone());
                    for path in rest {
                        dirty.insert(path.clone());
                    }
                    return;
                }
            }
            for path in &event.paths {
                dirty.insert(path.clone());
            }
        }
        EventKind::Create(_) | EventKind::Any => {
            for path in &event.paths {
                dirty.insert(path.clone());
            }
        }
        _ => {}
    }
}
