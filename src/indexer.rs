use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;

const DEFAULT_IGNORE: &str = include_str!("../default-ignore.txt");
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::chunker::{self, CodeChunk};
use crate::embedding::{BatchEmbedder, Embedder};
use crate::store::{IndexMetadata, IndexStore, RepositoryIndex};

pub struct IndexRequest {
    pub path: PathBuf,
    pub force: bool,
    pub batch_size: Option<usize>,
}

pub struct IndexReport {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub duration: Duration,
}

#[derive(Clone)]
pub struct Indexer {
    embedder: Arc<dyn BatchEmbedder>,
}

impl Indexer {
    pub fn new<E: BatchEmbedder + 'static>(embedder: Arc<E>) -> Self {
        Self { embedder }
    }

    pub fn build_index(&self, request: IndexRequest) -> Result<IndexReport> {
        let root = canonical(&request.path);
        info!("indexing" = %root.display(), "force" = request.force);
        let files = collect_files(&root);

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancelled.clone();
        if let Err(err) = ctrlc::set_handler(move || {
            cancel_flag.store(true, Ordering::SeqCst);
        }) {
            warn!("ctrlc_handler" = %err, "msg" = "could not install; continuing");
        }

        let pb =
            ProgressBar::with_draw_target(Some(files.len() as u64), ProgressDrawTarget::stderr());
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg} (eta {eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("chunking");

        let start = Instant::now();
        let chunks: Vec<CodeChunk> = files
            .par_iter()
            .map(|path| {
                if cancelled.load(Ordering::SeqCst) {
                    return Vec::new();
                }
                match chunker::chunk_file(path, &root) {
                    Ok(ch) => {
                        pb.inc(1);
                        ch
                    }
                    Err(err) => {
                        warn!("path" = %path.display(), "error" = %err, "msg" = "skipping");
                        pb.inc(1);
                        Vec::new()
                    }
                }
            })
            .flat_map_iter(|chunks| chunks.into_iter())
            .collect();

        pb.set_length(chunks.len() as u64);
        pb.set_position(0);
        pb.reset_elapsed();
        pb.set_message("embedding");

        let batch_size = determine_batch_size(request.batch_size);
        let mut vectors = Vec::with_capacity(chunks.len());

        for batch_start in (0..chunks.len()).step_by(batch_size) {
            if cancelled.load(Ordering::SeqCst) {
                pb.finish_and_clear();
                return Err(anyhow!("Indexing cancelled by user"));
            }

            let batch_end = (batch_start + batch_size).min(chunks.len());
            let batch_texts: Vec<String> = chunks[batch_start..batch_end]
                .iter()
                .map(|chunk| chunk.text.clone())
                .collect();

            match self.embed_batch_with_retry(&batch_texts) {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(err) => {
                    pb.finish_and_clear();
                    return Err(anyhow!(
                        "Embedding failed for batch starting at {}: {}",
                        batch_start,
                        err
                    ));
                }
            }

            pb.inc((batch_end - batch_start) as u64);
        }

        pb.finish_with_message("embedding complete");

        let store = IndexStore::new(&root)?;
        let metadata = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: root.clone(),
            repo_hash: store.repo_hash().to_string(),
            vector_dim: self.embedder.dimension(),
            indexed_at: chrono::Utc::now(),
            total_files: files.len(),
            total_chunks: chunks.len(),
        };

        let repository_index = RepositoryIndex::new(metadata, chunks, vectors);
        store.save(&repository_index)?;

        Ok(IndexReport {
            files_indexed: files.len(),
            chunks_indexed: repository_index.chunks.len(),
            duration: start.elapsed(),
        })
    }

    fn embed_batch_with_retry(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        const MAX_ATTEMPTS: usize = 3;
        let mut last_err = None;

        for attempt in 1..=MAX_ATTEMPTS {
            match self.embedder.embed_batch(texts) {
                Ok(vectors) => return Ok(vectors),
                Err(err) => {
                    last_err = Some(err);
                    if attempt < MAX_ATTEMPTS {
                        let backoff = Duration::from_millis((attempt * 150) as u64);
                        warn!(
                            "attempt" = attempt,
                            "msg" = "retrying embed batch",
                            "backoff_ms" = backoff.as_millis()
                        );
                        thread::sleep(backoff);
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("embedding failed for batch")))
    }
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let default_excludes = build_default_excludes();

    WalkBuilder::new(root)
        .hidden(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .follow_links(false)
        .add_custom_ignore_filename(".sgrepignore")
        .build()
        .filter_map(|entry| match entry {
            Ok(e) => {
                if !e.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    return None;
                }
                let path = e.path();
                let relative_path = path.strip_prefix(root).unwrap_or(path);

                if default_excludes.is_match(relative_path) {
                    return None;
                }

                Some(e.into_path())
            }
            _ => None,
        })
        .collect()
}

fn build_default_excludes() -> globset::GlobSet {
    let mut builder = GlobSetBuilder::new();

    for line in DEFAULT_IGNORE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Ok(glob) = Glob::new(line) {
            builder.add(glob);
        }

        let pattern_without_slash = line.trim_end_matches('/');
        if pattern_without_slash != line {
            if let Ok(glob) = Glob::new(&format!("**/{}", pattern_without_slash)) {
                builder.add(glob);
            }
            if let Ok(glob) = Glob::new(&format!("{}/**", pattern_without_slash)) {
                builder.add(glob);
            }
        }
    }

    builder
        .build()
        .unwrap_or_else(|_| GlobSetBuilder::new().build().unwrap())
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn determine_batch_size(override_val: Option<usize>) -> usize {
    if let Some(v) = override_val {
        return v.clamp(16, 2048);
    }

    if let Ok(value) = env::var("SGREP_BATCH_SIZE") {
        if let Ok(parsed) = value.parse::<usize>() {
            return parsed.clamp(16, 2048);
        }
    }

    match env::var("SGREP_DEVICE")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "cuda" | "coreml" => 512,
        _ => 256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;

    fn set_test_home() -> PathBuf {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_test_home_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_var("SGREP_HOME", &temp_dir);
        temp_dir
    }

    fn create_test_repo() -> PathBuf {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_indexer_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        fs::write(
            temp_dir.join("test.rs"),
            "fn hello() { println!(\"Hello\"); }",
        )
        .unwrap();
        fs::write(temp_dir.join("test.py"), "def hello():\n    print('Hello')").unwrap();

        let src_dir = temp_dir.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

        temp_dir
    }

    #[test]
    fn indexer_creates_report() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
        };

        let report = indexer.build_index(request).unwrap();

        assert!(report.files_indexed > 0);
        assert!(report.chunks_indexed > 0);
        assert!(report.duration.as_secs() < 60);

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    fn indexer_saves_to_store() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
        };

        indexer.build_index(request).unwrap();

        let store = IndexStore::new(&test_repo).unwrap();
        let loaded = store.load().unwrap();

        assert!(loaded.is_some());
        let index = loaded.unwrap();
        assert!(index.chunks.len() > 0);
        assert_eq!(index.chunks.len(), index.vectors.len());

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    fn collect_files_respects_gitignore() {
        let test_repo = create_test_repo();

        std::process::Command::new("git")
            .args(&["init"])
            .current_dir(&test_repo)
            .output()
            .ok();

        fs::write(test_repo.join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(test_repo.join("ignored.txt"), "this should be ignored").unwrap();
        fs::write(test_repo.join("visible.txt"), "this should be visible").unwrap();

        let files = collect_files(&test_repo);
        let file_names: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();

        assert!(
            !file_names.contains(&"ignored.txt".to_string()),
            "ignored.txt should not be collected, found files: {:?}",
            file_names
        );
        assert!(file_names.contains(&"visible.txt".to_string()));

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    fn index_metadata_has_correct_fields() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
        };

        indexer.build_index(request).unwrap();

        let store = IndexStore::new(&test_repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(index.metadata.vector_dim, embedder.dimension());
        assert!(index.metadata.total_files > 0);
        assert!(index.metadata.total_chunks > 0);
        assert_eq!(index.metadata.total_chunks, index.chunks.len());

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    fn canonical_handles_nonexistent_paths() {
        let nonexistent = PathBuf::from("/this/path/does/not/exist");
        let result = canonical(&nonexistent);
        assert_eq!(result, nonexistent);
    }

    #[test]
    fn determine_batch_size_respects_env() {
        env::set_var("SGREP_BATCH_SIZE", "1024");
        assert_eq!(determine_batch_size(None), 1024);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[test]
    fn determine_batch_size_prefers_override() {
        env::set_var("SGREP_BATCH_SIZE", "64");
        assert_eq!(determine_batch_size(Some(512)), 512);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[derive(Clone, Default)]
    struct FailingEmbedder;

    impl BatchEmbedder for FailingEmbedder {
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Err(anyhow!("intentional embed failure"))
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    fn indexer_aborts_after_repeated_embedding_failures() {
        let embedder = Arc::new(FailingEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
        };

        let result = indexer.build_index(request);
        assert!(result.is_err());
        let msg = format!("{:?}", result.err().unwrap());
        assert!(msg.contains("Embedding failed"));

        fs::remove_dir_all(&test_repo).ok();
    }
}
