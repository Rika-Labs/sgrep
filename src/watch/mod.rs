use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use notify::EventKind;

#[cfg(test)]
use crate::indexer::Indexer;
#[cfg(not(test))]
use crate::indexer::{DirtySet, IndexRequest, Indexer};
#[cfg(not(test))]
use crate::store::IndexStore;
#[cfg(not(test))]
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(test))]
use std::sync::mpsc::{channel, RecvTimeoutError};
#[cfg(not(test))]
use std::sync::Arc;
#[cfg(not(test))]
use std::thread;
#[cfg(not(test))]
use std::time::Instant;
#[cfg(not(test))]
use tracing::{info, warn};

#[cfg_attr(test, allow(dead_code))]
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

    #[cfg(test)]
    pub fn run(&mut self, _path: &Path) -> Result<()> {
        Ok(())
    }

    #[cfg(not(test))]
    pub fn run(&mut self, path: &Path) -> Result<()> {
        let store = IndexStore::new(path)?;
        if store.load()?.is_none() {
            info!("Creating initial index: {}", path.display());
            let result = self.indexer.build_index(crate::indexer::IndexRequest {
                path: path.to_path_buf(),
                force: false,
                batch_size: self.batch_size,
                profile: false,
                dirty: None,
            });
            match result {
                Ok(report) => {
                    info!(
                        "Initial index created: {} files, {} chunks in {:.1}s",
                        report.files_indexed,
                        report.chunks_indexed,
                        report.duration.as_secs_f64()
                    );
                }
                Err(e) => {
                    warn!("error" = %e, "msg" = "failed to create initial index, will retry on first change");
                }
            }
        } else {
            info!(
                "Using existing index for incremental updates: {}",
                path.display()
            );
        }

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
                    "Updating index: {} changed, {} deleted",
                    dirty_paths.len(),
                    deleted_paths.len()
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

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{CreateKind, ModifyKind, RemoveKind};
    use notify::Event;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn should_reindex_positive_cases() {
        assert!(should_reindex(&EventKind::Create(CreateKind::File)));
        assert!(should_reindex(&EventKind::Modify(ModifyKind::Data(
            notify::event::DataChange::Any
        ))));
        assert!(should_reindex(&EventKind::Remove(RemoveKind::File)));
        assert!(should_reindex(&EventKind::Any));
    }

    #[test]
    fn should_reindex_negative_case() {
        assert!(!should_reindex(&EventKind::Access(
            notify::event::AccessKind::Read
        )));
    }

    #[test]
    fn categorize_event_handles_remove() {
        let event = Event {
            kind: EventKind::Remove(RemoveKind::File),
            paths: vec![PathBuf::from("deleted.rs")],
            attrs: Default::default(),
        };
        let mut dirty = HashSet::new();
        let mut deleted = HashSet::new();
        categorize_event(&event, &mut dirty, &mut deleted);
        assert!(deleted.contains(&PathBuf::from("deleted.rs")));
        assert!(dirty.is_empty());
    }

    #[test]
    fn categorize_event_handles_create_and_modify() {
        let create = Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![PathBuf::from("new.rs")],
            attrs: Default::default(),
        };
        let modify = Event {
            kind: EventKind::Modify(ModifyKind::Data(notify::event::DataChange::Any)),
            paths: vec![PathBuf::from("changed.rs")],
            attrs: Default::default(),
        };

        let mut dirty = HashSet::new();
        let mut deleted = HashSet::new();
        categorize_event(&create, &mut dirty, &mut deleted);
        categorize_event(&modify, &mut dirty, &mut deleted);

        assert!(dirty.contains(&PathBuf::from("new.rs")));
        assert!(dirty.contains(&PathBuf::from("changed.rs")));
        assert!(deleted.is_empty());
    }

    #[test]
    fn categorize_event_handles_rename_dual_paths() {
        let rename = Event {
            kind: EventKind::Modify(ModifyKind::Name(notify::event::RenameMode::Both)),
            paths: vec![PathBuf::from("old.rs"), PathBuf::from("new.rs")],
            attrs: Default::default(),
        };

        let mut dirty = HashSet::new();
        let mut deleted = HashSet::new();
        categorize_event(&rename, &mut dirty, &mut deleted);

        assert!(deleted.contains(&PathBuf::from("old.rs")));
        assert!(dirty.contains(&PathBuf::from("new.rs")));
    }

    #[test]
    fn categorize_event_ignores_unhandled_kind() {
        let event = Event {
            kind: EventKind::Access(notify::event::AccessKind::Read),
            paths: vec![PathBuf::from("ignore.rs")],
            attrs: Default::default(),
        };
        let mut dirty = HashSet::new();
        let mut deleted = HashSet::new();
        categorize_event(&event, &mut dirty, &mut deleted);
        assert!(dirty.is_empty());
        assert!(deleted.is_empty());
    }

    #[test]
    fn watch_service_new_and_run_noop_in_tests() {
        let indexer = Indexer::new(Arc::new(crate::embedding::Embedder::default()));
        let mut svc = WatchService::new(indexer, Duration::from_millis(200), Some(32));
        let root = Path::new(".");
        svc.run(root).expect("test run should be a no-op");
    }
}
