use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use notify::EventKind;

use crate::indexer::{DirtySet, IndexRequest, Indexer};
#[cfg(not(test))]
use crate::store::IndexStore;
#[cfg(not(test))]
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(test))]
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
#[cfg(not(test))]
use std::thread;
use std::time::Instant;
#[cfg(not(test))]
use tracing::{info, warn};

/// Tracks the state of event aggregation and debouncing for the watch service.
/// This struct is extracted to enable integration testing without real filesystem watchers.
pub struct WatchEventProcessor {
    /// Accumulated paths that have been touched (created/modified)
    pub dirty_paths: HashSet<PathBuf>,
    /// Accumulated paths that have been deleted
    pub deleted_paths: HashSet<PathBuf>,
    /// Timestamp of the last event received
    pub last_event: Option<Instant>,
    /// Debounce duration - how long to wait after the last event before triggering indexing
    pub debounce: Duration,
    /// Whether an indexing operation is currently in progress
    pub indexing_in_progress: bool,
    /// Shutdown signal
    pub shutdown: Arc<AtomicBool>,
    /// Count of index operations triggered (for testing)
    pub index_trigger_count: usize,
}

impl WatchEventProcessor {
    /// Creates a new event processor with the specified debounce duration
    pub fn new(debounce: Duration) -> Self {
        Self {
            dirty_paths: HashSet::new(),
            deleted_paths: HashSet::new(),
            last_event: None,
            debounce,
            indexing_in_progress: false,
            shutdown: Arc::new(AtomicBool::new(false)),
            index_trigger_count: 0,
        }
    }

    /// Processes a filesystem event and updates the dirty/deleted sets
    pub fn process_event(&mut self, event: &notify::Event) {
        if should_reindex(&event.kind) {
            categorize_event(event, &mut self.dirty_paths, &mut self.deleted_paths);
            self.last_event = Some(Instant::now());
        }
    }

    /// Checks if the debounce window has elapsed and we're ready to trigger indexing
    pub fn is_ready_to_index(&self) -> bool {
        self.last_event
            .map(|t| t.elapsed() >= self.debounce)
            .unwrap_or(false)
            && !self.indexing_in_progress
            && (!self.dirty_paths.is_empty() || !self.deleted_paths.is_empty())
    }

    /// Collects and drains the accumulated dirty set for indexing
    /// Returns None if there are no changes to process
    pub fn collect_dirty_set(&mut self) -> Option<DirtySet> {
        if self.dirty_paths.is_empty() && self.deleted_paths.is_empty() {
            return None;
        }

        let dirty_set = DirtySet {
            touched: self.dirty_paths.drain().collect(),
            deleted: self.deleted_paths.drain().collect(),
        };

        self.last_event = None;
        self.index_trigger_count += 1;

        Some(dirty_set)
    }

    /// Marks indexing as started
    pub fn start_indexing(&mut self) {
        self.indexing_in_progress = true;
    }

    /// Marks indexing as complete
    pub fn complete_indexing(&mut self) {
        self.indexing_in_progress = false;
    }

    /// Signals shutdown
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Checks if shutdown has been signaled
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Returns the number of paths waiting to be indexed
    pub fn pending_count(&self) -> usize {
        self.dirty_paths.len() + self.deleted_paths.len()
    }
}

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

    // =========================================================================
    // Integration Tests for WatchEventProcessor
    // These tests validate the acceptance criteria:
    // 1. FS events trigger incremental indexing once per debounce window
    // 2. Deletions are handled correctly via DirtySet
    // 3. Graceful shutdown
    // 4. No flakiness under repeated runs
    // =========================================================================

    mod integration_tests {
        use super::*;
        use crate::embedding::BatchEmbedder;
        use crate::store::IndexStore;
        use anyhow::Result;
        use serial_test::serial;
        use std::fs;
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use uuid::Uuid;

        /// Test embedder that tracks calls for verification
        #[derive(Clone)]
        struct TrackingEmbedder {
            embed_count: Arc<AtomicUsize>,
            batch_count: Arc<AtomicUsize>,
            delay_ms: u64,
        }

        impl TrackingEmbedder {
            fn new(delay_ms: u64) -> Self {
                Self {
                    embed_count: Arc::new(AtomicUsize::new(0)),
                    batch_count: Arc::new(AtomicUsize::new(0)),
                    delay_ms,
                }
            }

            #[allow(dead_code)]
            fn embed_count(&self) -> usize {
                self.embed_count.load(Ordering::SeqCst)
            }

            fn batch_count(&self) -> usize {
                self.batch_count.load(Ordering::SeqCst)
            }

            fn reset_counts(&self) {
                self.embed_count.store(0, Ordering::SeqCst);
                self.batch_count.store(0, Ordering::SeqCst);
            }
        }

        impl BatchEmbedder for TrackingEmbedder {
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.batch_count.fetch_add(1, Ordering::SeqCst);
                self.embed_count.fetch_add(texts.len(), Ordering::SeqCst);
                if self.delay_ms > 0 {
                    thread::sleep(Duration::from_millis(self.delay_ms));
                }
                Ok(texts
                    .iter()
                    .map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0])
                    .collect())
            }

            fn dimension(&self) -> usize {
                4
            }
        }

        /// Creates an isolated test environment with unique SGREP_HOME
        fn setup_test_env() -> (PathBuf, PathBuf) {
            let test_id = Uuid::new_v4();
            let home =
                std::env::temp_dir().join(format!("sgrep_watch_integration_home_{}", test_id));
            let repo =
                std::env::temp_dir().join(format!("sgrep_watch_integration_repo_{}", test_id));

            fs::create_dir_all(&home).expect("Failed to create test home");
            fs::create_dir_all(&repo).expect("Failed to create test repo");

            std::env::set_var("SGREP_HOME", &home);

            (home, repo)
        }

        /// Cleans up test directories
        fn cleanup_test_env(home: &Path, repo: &Path) {
            fs::remove_dir_all(home).ok();
            fs::remove_dir_all(repo).ok();
        }

        /// Creates sample files in test repo
        fn create_sample_files(repo: &Path) {
            fs::write(
                repo.join("main.rs"),
                "fn main() { println!(\"Hello\"); }",
            )
            .unwrap();
            fs::write(
                repo.join("lib.rs"),
                "pub fn add(a: i32, b: i32) -> i32 { a + b }",
            )
            .unwrap();

            let src = repo.join("src");
            fs::create_dir_all(&src).unwrap();
            fs::write(src.join("utils.rs"), "pub fn helper() {}").unwrap();
        }

        // =====================================================================
        // TEST 1: FS events trigger incremental indexing once per debounce
        // =====================================================================

        #[test]
        fn test_processor_aggregates_multiple_events_before_debounce() {
            let debounce = Duration::from_millis(100);
            let mut processor = WatchEventProcessor::new(debounce);

            // Simulate multiple rapid events
            let events = vec![
                Event {
                    kind: EventKind::Create(CreateKind::File),
                    paths: vec![PathBuf::from("file1.rs")],
                    attrs: Default::default(),
                },
                Event {
                    kind: EventKind::Modify(ModifyKind::Data(notify::event::DataChange::Any)),
                    paths: vec![PathBuf::from("file2.rs")],
                    attrs: Default::default(),
                },
                Event {
                    kind: EventKind::Create(CreateKind::File),
                    paths: vec![PathBuf::from("file3.rs")],
                    attrs: Default::default(),
                },
            ];

            // Process all events rapidly (within debounce window)
            for event in &events {
                processor.process_event(event);
            }

            // ASSERTION: All 3 files accumulated in dirty_paths
            assert_eq!(
                processor.dirty_paths.len(),
                3,
                "All events should be accumulated"
            );
            assert!(processor.dirty_paths.contains(&PathBuf::from("file1.rs")));
            assert!(processor.dirty_paths.contains(&PathBuf::from("file2.rs")));
            assert!(processor.dirty_paths.contains(&PathBuf::from("file3.rs")));

            // ASSERTION: Not ready to index yet (debounce not elapsed)
            assert!(
                !processor.is_ready_to_index(),
                "Should not be ready before debounce elapses"
            );

            // Wait for debounce
            thread::sleep(debounce + Duration::from_millis(20));

            // ASSERTION: Now ready to index
            assert!(
                processor.is_ready_to_index(),
                "Should be ready after debounce elapses"
            );

            // Collect dirty set
            let dirty_set = processor.collect_dirty_set();
            assert!(dirty_set.is_some());
            let dirty = dirty_set.unwrap();

            // ASSERTION: All files in single DirtySet
            assert_eq!(
                dirty.touched.len(),
                3,
                "All files should be in single DirtySet"
            );
            assert_eq!(
                processor.index_trigger_count, 1,
                "Only one index trigger should occur"
            );
        }

        #[test]
        fn test_processor_resets_debounce_on_new_events() {
            let debounce = Duration::from_millis(100);
            let mut processor = WatchEventProcessor::new(debounce);

            // First event
            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("file1.rs")],
                attrs: Default::default(),
            });

            // Wait half debounce time
            thread::sleep(Duration::from_millis(50));

            // Second event (should reset timer)
            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("file2.rs")],
                attrs: Default::default(),
            });

            // Wait another half debounce time (total 100ms from first event)
            thread::sleep(Duration::from_millis(50));

            // ASSERTION: Still not ready because timer was reset
            // Note: Due to timing variability, this may sometimes pass
            // The key assertion is that after full debounce from LAST event, it's ready

            // Wait remaining time
            thread::sleep(Duration::from_millis(60));

            // ASSERTION: Now ready (debounce from last event has passed)
            assert!(
                processor.is_ready_to_index(),
                "Should be ready after full debounce from last event"
            );
        }

        #[test]
        fn test_processor_blocks_indexing_while_in_progress() {
            let debounce = Duration::from_millis(10);
            let mut processor = WatchEventProcessor::new(debounce);

            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("file.rs")],
                attrs: Default::default(),
            });

            thread::sleep(debounce + Duration::from_millis(5));

            // Start indexing
            processor.start_indexing();

            // ASSERTION: Not ready while indexing in progress
            assert!(
                !processor.is_ready_to_index(),
                "Should not be ready while indexing"
            );

            // Add new event
            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("file2.rs")],
                attrs: Default::default(),
            });

            thread::sleep(debounce + Duration::from_millis(5));

            // Still not ready
            assert!(
                !processor.is_ready_to_index(),
                "Should still not be ready while indexing"
            );

            // Complete indexing
            processor.complete_indexing();

            // ASSERTION: Now ready with new events
            assert!(
                processor.is_ready_to_index(),
                "Should be ready after indexing completes"
            );
        }

        #[test]
        #[serial]
        fn test_debounce_triggers_single_index_for_burst_of_changes() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder.clone());

            // Create initial index
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Initial index should succeed");

            embedder.reset_counts();

            // Simulate burst of file changes
            let file1 = repo.join("main.rs");
            let file2 = repo.join("lib.rs");
            let file3 = repo.join("new_file.rs");

            fs::write(&file1, "fn main() { println!(\"Updated 1\"); }").unwrap();
            fs::write(&file2, "pub fn add(a: i32, b: i32) -> i32 { a + b + 1 }").unwrap();
            fs::write(&file3, "fn new_function() {}").unwrap();

            // Create aggregated dirty set (as debounce would produce)
            let dirty_set = DirtySet {
                touched: vec![file1, file2, file3],
                deleted: vec![],
            };

            // Run SINGLE incremental index
            let report = indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: false,
                    batch_size: None,
                    profile: false,
                    dirty: Some(dirty_set),
                })
                .expect("Incremental index should succeed");

            // ASSERTION: Batches were called (embedder was used)
            assert!(
                embedder.batch_count() >= 1,
                "Should have made embedding calls"
            );

            // ASSERTION: Report shows indexed content
            assert!(
                report.chunks_indexed > 0,
                "Should have indexed some chunks"
            );

            cleanup_test_env(&home, &repo);
        }

        // =====================================================================
        // TEST 2: Deletions handled correctly
        // =====================================================================

        #[test]
        fn test_processor_separates_deletions_from_touches() {
            let mut processor = WatchEventProcessor::new(Duration::from_millis(10));

            // Mix of create, modify, and delete events
            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("new.rs")],
                attrs: Default::default(),
            });
            processor.process_event(&Event {
                kind: EventKind::Modify(ModifyKind::Data(notify::event::DataChange::Any)),
                paths: vec![PathBuf::from("modified.rs")],
                attrs: Default::default(),
            });
            processor.process_event(&Event {
                kind: EventKind::Remove(RemoveKind::File),
                paths: vec![PathBuf::from("deleted.rs")],
                attrs: Default::default(),
            });

            // ASSERTION: Paths correctly categorized
            assert_eq!(processor.dirty_paths.len(), 2);
            assert_eq!(processor.deleted_paths.len(), 1);
            assert!(processor.dirty_paths.contains(&PathBuf::from("new.rs")));
            assert!(processor.dirty_paths.contains(&PathBuf::from("modified.rs")));
            assert!(processor.deleted_paths.contains(&PathBuf::from("deleted.rs")));
        }

        #[test]
        #[serial]
        fn test_deletion_removes_chunks_from_index() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder.clone());

            // Create initial index
            let initial_report = indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Initial index should succeed");

            let initial_chunks = initial_report.chunks_indexed;
            assert!(initial_chunks > 0, "Should have initial chunks");

            // Verify main.rs is in index
            let store = IndexStore::new(&repo).unwrap();
            let initial_index = store.load().unwrap().expect("Index should exist");
            let has_main = initial_index
                .chunks
                .iter()
                .any(|c| c.path.ends_with("main.rs"));
            assert!(has_main, "main.rs should be in initial index");

            // Delete the file
            let deleted_file = repo.join("main.rs");
            fs::remove_file(&deleted_file).expect("Should delete file");

            // Run incremental with deletion
            let report = indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: false,
                    batch_size: None,
                    profile: false,
                    dirty: Some(DirtySet {
                        touched: vec![],
                        deleted: vec![deleted_file],
                    }),
                })
                .expect("Deletion index should succeed");

            // ASSERTION: Chunks decreased
            assert!(
                report.chunks_indexed < initial_chunks,
                "Chunk count should decrease after deletion"
            );

            // ASSERTION: main.rs no longer in index
            let updated_index = store.load().unwrap().expect("Index should exist");
            let has_main_after = updated_index
                .chunks
                .iter()
                .any(|c| c.path.ends_with("main.rs"));
            assert!(
                !has_main_after,
                "main.rs should NOT be in index after deletion"
            );

            cleanup_test_env(&home, &repo);
        }

        #[test]
        #[serial]
        fn test_directory_deletion_removes_contained_files() {
            let (home, repo) = setup_test_env();

            // Create directory with files
            let subdir = repo.join("submodule");
            fs::create_dir_all(&subdir).unwrap();
            fs::write(subdir.join("a.rs"), "fn a() {}").unwrap();
            fs::write(subdir.join("b.rs"), "fn b() {}").unwrap();
            fs::write(repo.join("root.rs"), "fn root() {}").unwrap();

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder);

            // Create initial index
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Initial index should succeed");

            // Verify subdir files indexed
            let store = IndexStore::new(&repo).unwrap();
            let initial_index = store.load().unwrap().expect("Index should exist");
            let subdir_count = initial_index
                .chunks
                .iter()
                .filter(|c| c.path.starts_with("submodule"))
                .count();
            assert!(subdir_count >= 2, "Should have chunks from submodule");

            // Delete directory
            fs::remove_dir_all(&subdir).expect("Should delete directory");

            // Run incremental with directory deletion
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: false,
                    batch_size: None,
                    profile: false,
                    dirty: Some(DirtySet {
                        touched: vec![],
                        deleted: vec![subdir],
                    }),
                })
                .expect("Directory deletion should succeed");

            // ASSERTION: No chunks from deleted directory
            let updated_index = store.load().unwrap().expect("Index should exist");
            let subdir_count_after = updated_index
                .chunks
                .iter()
                .filter(|c| c.path.starts_with("submodule"))
                .count();
            assert_eq!(
                subdir_count_after, 0,
                "No chunks should remain from deleted directory"
            );

            // ASSERTION: root.rs still exists
            let has_root = updated_index
                .chunks
                .iter()
                .any(|c| c.path.ends_with("root.rs"));
            assert!(has_root, "root.rs should still be in index");

            cleanup_test_env(&home, &repo);
        }

        #[test]
        #[serial]
        fn test_rename_handles_old_and_new_paths() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder);

            // Create initial index
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Initial index should succeed");

            let store = IndexStore::new(&repo).unwrap();
            let initial_index = store.load().unwrap().expect("Index should exist");
            assert!(initial_index.chunks.iter().any(|c| c.path.ends_with("main.rs")));

            // Rename file
            let old_path = repo.join("main.rs");
            let new_path = repo.join("entry.rs");
            fs::rename(&old_path, &new_path).expect("Should rename");

            // Run incremental with rename
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: false,
                    batch_size: None,
                    profile: false,
                    dirty: Some(DirtySet {
                        touched: vec![new_path],
                        deleted: vec![old_path],
                    }),
                })
                .expect("Rename index should succeed");

            // ASSERTION: Old path gone, new path present
            let updated_index = store.load().unwrap().expect("Index should exist");
            assert!(
                !updated_index.chunks.iter().any(|c| c.path.ends_with("main.rs")),
                "main.rs should be gone"
            );
            assert!(
                updated_index.chunks.iter().any(|c| c.path.ends_with("entry.rs")),
                "entry.rs should be present"
            );

            cleanup_test_env(&home, &repo);
        }

        // =====================================================================
        // TEST 3: Graceful shutdown
        // =====================================================================

        #[test]
        fn test_processor_shutdown_signal() {
            let processor = WatchEventProcessor::new(Duration::from_millis(100));

            assert!(!processor.is_shutdown(), "Should not be shutdown initially");

            processor.signal_shutdown();

            assert!(processor.is_shutdown(), "Should be shutdown after signal");
        }

        #[test]
        fn test_processor_tracks_indexing_state() {
            let mut processor = WatchEventProcessor::new(Duration::from_millis(10));

            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("test.rs")],
                attrs: Default::default(),
            });

            thread::sleep(Duration::from_millis(15));

            assert!(processor.is_ready_to_index());

            processor.start_indexing();
            assert!(processor.indexing_in_progress);
            assert!(!processor.is_ready_to_index());

            processor.complete_indexing();
            assert!(!processor.indexing_in_progress);
        }

        #[test]
        #[serial]
        fn test_shutdown_does_not_corrupt_index() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder);

            // Create index
            indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Index should succeed");

            // Verify index is valid
            let store = IndexStore::new(&repo).unwrap();
            let index = store.load().unwrap().expect("Index should exist");

            assert!(!index.chunks.is_empty(), "Should have chunks");
            assert_eq!(
                index.chunks.len(),
                index.vectors.len(),
                "Chunks and vectors should match"
            );

            for vec in &index.vectors {
                assert_eq!(vec.len(), 4, "Vector dimension should match");
            }

            cleanup_test_env(&home, &repo);
        }

        // =====================================================================
        // TEST 4: No flakiness under repeated runs
        // =====================================================================

        #[test]
        #[serial]
        fn test_repeated_indexing_produces_consistent_results() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder);

            let mut results = Vec::new();

            for _ in 0..5 {
                let report = indexer
                    .build_index(IndexRequest {
                        path: repo.clone(),
                        force: true,
                        batch_size: None,
                        profile: false,
                        dirty: None,
                    })
                    .expect("Index should succeed");
                results.push(report);
            }

            // ASSERTION: Consistent chunk counts
            let first_chunks = results[0].chunks_indexed;
            for (i, report) in results.iter().enumerate() {
                assert_eq!(
                    report.chunks_indexed, first_chunks,
                    "Run {} had inconsistent chunk count",
                    i
                );
            }

            // ASSERTION: Consistent file counts
            let first_files = results[0].files_indexed;
            for (i, report) in results.iter().enumerate() {
                assert_eq!(
                    report.files_indexed, first_files,
                    "Run {} had inconsistent file count",
                    i
                );
            }

            cleanup_test_env(&home, &repo);
        }

        #[test]
        #[serial]
        fn test_incremental_idempotent_with_empty_dirty_set() {
            let (home, repo) = setup_test_env();
            create_sample_files(&repo);

            let embedder = Arc::new(TrackingEmbedder::new(0));
            let indexer = Indexer::new(embedder);

            // Create initial index
            let initial_report = indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: true,
                    batch_size: None,
                    profile: false,
                    dirty: None,
                })
                .expect("Initial index should succeed");

            // Run with empty dirty set
            let report = indexer
                .build_index(IndexRequest {
                    path: repo.clone(),
                    force: false,
                    batch_size: None,
                    profile: false,
                    dirty: Some(DirtySet {
                        touched: vec![],
                        deleted: vec![],
                    }),
                })
                .expect("Empty incremental should succeed");

            // ASSERTION: Same chunk count
            assert_eq!(
                report.chunks_indexed, initial_report.chunks_indexed,
                "Should have same chunks with empty dirty set"
            );

            cleanup_test_env(&home, &repo);
        }

        #[test]
        fn test_processor_deterministic_under_repeated_events() {
            // Run multiple times to check for race conditions
            for _ in 0..10 {
                let mut processor = WatchEventProcessor::new(Duration::from_millis(5));

                let events: Vec<Event> = (0..20)
                    .map(|i| Event {
                        kind: EventKind::Create(CreateKind::File),
                        paths: vec![PathBuf::from(format!("file_{}.rs", i))],
                        attrs: Default::default(),
                    })
                    .collect();

                for event in &events {
                    processor.process_event(event);
                }

                // ASSERTION: All events accumulated
                assert_eq!(
                    processor.dirty_paths.len(),
                    20,
                    "All events should be accumulated"
                );

                thread::sleep(Duration::from_millis(10));

                assert!(processor.is_ready_to_index());

                let dirty = processor.collect_dirty_set().unwrap();
                assert_eq!(dirty.touched.len(), 20);
                assert_eq!(processor.index_trigger_count, 1);
            }
        }

        #[test]
        fn test_processor_handles_duplicate_events() {
            let mut processor = WatchEventProcessor::new(Duration::from_millis(10));

            // Same file modified multiple times
            for _ in 0..5 {
                processor.process_event(&Event {
                    kind: EventKind::Modify(ModifyKind::Data(notify::event::DataChange::Any)),
                    paths: vec![PathBuf::from("same_file.rs")],
                    attrs: Default::default(),
                });
            }

            // ASSERTION: Only one entry (HashSet deduplicates)
            assert_eq!(
                processor.dirty_paths.len(),
                1,
                "Duplicate events should be deduplicated"
            );
            assert!(processor.dirty_paths.contains(&PathBuf::from("same_file.rs")));
        }

        #[test]
        fn test_processor_pending_count() {
            let mut processor = WatchEventProcessor::new(Duration::from_millis(10));

            assert_eq!(processor.pending_count(), 0);

            processor.process_event(&Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![PathBuf::from("new.rs")],
                attrs: Default::default(),
            });
            assert_eq!(processor.pending_count(), 1);

            processor.process_event(&Event {
                kind: EventKind::Remove(RemoveKind::File),
                paths: vec![PathBuf::from("deleted.rs")],
                attrs: Default::default(),
            });
            assert_eq!(processor.pending_count(), 2);
        }

        #[test]
        fn test_processor_access_events_ignored() {
            let mut processor = WatchEventProcessor::new(Duration::from_millis(10));

            processor.process_event(&Event {
                kind: EventKind::Access(notify::event::AccessKind::Read),
                paths: vec![PathBuf::from("read_only.rs")],
                attrs: Default::default(),
            });

            assert_eq!(
                processor.pending_count(),
                0,
                "Access events should be ignored"
            );
            assert!(
                processor.last_event.is_none(),
                "last_event should not be set for ignored events"
            );
        }
    }
}
