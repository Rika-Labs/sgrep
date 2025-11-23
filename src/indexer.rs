use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use globset::{Glob, GlobSetBuilder};
use ignore::WalkBuilder;

const DEFAULT_IGNORE: &str = include_str!("../default-ignore.txt");
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::chunker::{self, CodeChunk};
use crate::embedding::Embedder;
use crate::store::{IndexMetadata, IndexStore, RepositoryIndex};

pub struct IndexRequest {
    pub path: PathBuf,
    pub force: bool,
}

pub struct IndexReport {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub duration: Duration,
}

pub struct Indexer {
    embedder: Arc<Embedder>,
}

impl Indexer {
    pub fn new(embedder: Arc<Embedder>) -> Self {
        Self { embedder }
    }

    pub fn build_index(&self, request: IndexRequest) -> Result<IndexReport> {
        let root = canonical(&request.path);
        info!("indexing" = %root.display(), "force" = request.force);
        let files = collect_files(&root);
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("chunking");

        let start = Instant::now();
        let chunks: Vec<CodeChunk> = files
            .par_iter()
            .map(|path| match chunker::chunk_file(path, &root) {
                Ok(ch) => {
                    pb.inc(1);
                    ch
                }
                Err(err) => {
                    warn!("path" = %path.display(), "error" = %err, "msg" = "skipping");
                    pb.inc(1);
                    Vec::new()
                }
            })
            .flat_map_iter(|chunks| chunks.into_iter())
            .collect();
        pb.finish_with_message("embedding");

        const BATCH_SIZE: usize = 256;
        let mut vectors = Vec::with_capacity(chunks.len());

        for batch_start in (0..chunks.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(chunks.len());
            let batch_texts: Vec<String> = chunks[batch_start..batch_end]
                .iter()
                .map(|chunk| chunk.text.clone())
                .collect();

            match self.embedder.embed_batch(&batch_texts) {
                Ok(batch_vectors) => vectors.extend(batch_vectors),
                Err(_) => {
                    vectors.extend(vec![vec![0.0; self.embedder.dimension()]; batch_texts.len()]);
                }
            }
        }

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

    builder.build().unwrap_or_else(|_| {
        GlobSetBuilder::new().build().unwrap()
    })
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use uuid::Uuid;

    fn create_test_repo() -> PathBuf {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_indexer_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        
        fs::write(temp_dir.join("test.rs"), "fn hello() { println!(\"Hello\"); }").unwrap();
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
        
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
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
        
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
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
        
        assert!(!file_names.contains(&"ignored.txt".to_string()), 
            "ignored.txt should not be collected, found files: {:?}", file_names);
        assert!(file_names.contains(&"visible.txt".to_string()));
        
        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    fn index_metadata_has_correct_fields() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new(embedder.clone());
        
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
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
}
