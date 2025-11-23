use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ignore::WalkBuilder;
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

        let vectors: Vec<Vec<f32>> = chunks
            .par_iter()
            .map(|chunk| {
                self.embedder
                    .embed(&chunk.text)
                    .unwrap_or_else(|_| vec![0.0; self.embedder.dimension()])
            })
            .collect();

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
    WalkBuilder::new(root)
        .hidden(false)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .follow_links(false)
        .build()
        .filter_map(|entry| match entry {
            Ok(e) if e.file_type().map(|ft| ft.is_file()).unwrap_or(false) => Some(e.into_path()),
            _ => None,
        })
        .collect()
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}
