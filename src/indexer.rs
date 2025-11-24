use std::collections::{HashMap, HashSet};
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
use crate::embedding::BatchEmbedder;
use crate::store::{IndexMetadata, IndexStore, RepositoryIndex};

const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024; // skip very large assets
const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "svg", "ico", "webp", "avif", "psd", "mp4", "mov", "avi",
    "mp3", "wav", "flac", "ogg", "pdf", "zip", "gz", "bz2", "7z", "rar", "tar", "exe", "dll", "so",
    "a", "bin", "class", "wasm", "woff", "woff2", "ttf", "otf",
];
pub struct IndexRequest {
    pub path: PathBuf,
    pub force: bool,
    pub batch_size: Option<usize>,
    pub profile: bool,
    pub dirty: Option<DirtySet>,
}

pub struct IndexReport {
    pub files_indexed: usize,
    pub chunks_indexed: usize,
    pub duration: Duration,
    pub timings: Option<IndexTimings>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DirtySet {
    pub touched: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
}

impl DirtySet {
    pub fn is_empty(&self) -> bool {
        self.touched.is_empty() && self.deleted.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct IndexTimings {
    pub walk: Duration,
    pub chunk: Duration,
    pub embed: Duration,
    pub write: Duration,
}

#[derive(Clone)]
pub struct Indexer {
    embedder: Arc<dyn BatchEmbedder>,
}

impl Indexer {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self { embedder }
    }
    
    pub fn new_concrete<E: BatchEmbedder + 'static>(embedder: Arc<E>) -> Self {
        Self { embedder }
    }

    pub fn build_index(&self, request: IndexRequest) -> Result<IndexReport> {
        let total_start = Instant::now();
        let root = canonical(&request.path);
        if request.force {
            info!("Full indexing: {}", root.display());
        } else {
            info!("Indexing: {}", root.display());
        }
        let store = IndexStore::new(&root)?;
        let existing_index = if request.force {
            None
        } else {
            store.load().ok().flatten()
        };

        if let (Some(dirty), Some(index)) = (&request.dirty, existing_index.clone()) {
            let reusable = index.metadata.vector_dim == self.embedder.dimension()
                && index.metadata.version == env!("CARGO_PKG_VERSION");
            if reusable && !dirty.is_empty() {
                info!(
                    "Incremental indexing: {} changed, {} deleted",
                    dirty.touched.len(),
                    dirty.deleted.len()
                );
                return self.build_incremental(
                    &root,
                    index,
                    dirty.clone(),
                    &store,
                    &request,
                    total_start,
                );
            } else if !dirty.is_empty() {
                if !reusable {
                    warn!(
                        "Index incompatible (dim: {} vs {}, version: {} vs {}), doing full reindex",
                        index.metadata.vector_dim,
                        self.embedder.dimension(),
                        index.metadata.version,
                        env!("CARGO_PKG_VERSION")
                    );
                } else {
                    info!("No changes detected, skipping incremental");
                }
            }
        }

        self.build_full(&root, existing_index, &store, request, total_start)
    }

    fn build_full(
        &self,
        root: &Path,
        existing_index: Option<RepositoryIndex>,
        store: &IndexStore,
        request: IndexRequest,
        total_start: Instant,
    ) -> Result<IndexReport> {
        let walk_start = Instant::now();
        let files = collect_files(root);
        let walk_duration = walk_start.elapsed();

        let cancelled = Arc::new(AtomicBool::new(false));

        let pb =
            ProgressBar::with_draw_target(Some(files.len() as u64), ProgressDrawTarget::stderr());
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("chunking");

        let chunk_start = Instant::now();
        let chunks: Vec<CodeChunk> = files
            .par_iter()
            .map(|path| {
                if cancelled.load(Ordering::SeqCst) {
                    return Vec::new();
                }
                match chunker::chunk_file(path, root) {
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
        let chunk_duration = chunk_start.elapsed();

        pb.set_length(chunks.len() as u64);
        pb.set_position(0);
        pb.reset_elapsed();
        pb.set_message("embedding");

        let base_batch_size = determine_batch_size(request.batch_size);
        let batch_size = adjust_batch_size_for_progress(base_batch_size, chunks.len());
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];

        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        if let Some(index) = existing_index {
            if index.metadata.vector_dim == self.embedder.dimension()
                && index.metadata.version == env!("CARGO_PKG_VERSION")
            {
                for (chunk, vector) in index.chunks.iter().zip(index.vectors.iter()) {
                    cache.insert(chunk.hash.clone(), vector.clone());
                }
            }
        }

        let embed_start = Instant::now();
        let token_budget = determine_token_budget();

        struct Batch {
            indices: Vec<usize>,
            texts: Vec<String>,
            hashes: Vec<String>,
        }

        let mut batches: Vec<Batch> = Vec::new();
        let mut batch_start = 0usize;

        while batch_start < chunks.len() {
            if cancelled.load(Ordering::SeqCst) {
                pb.finish_and_clear();
                return Err(anyhow!("Indexing cancelled by user"));
            }

            let mut batch_end = batch_start;
            let mut token_count = 0usize;
            while batch_end < chunks.len() && (batch_end - batch_start) < batch_size {
                let est_tokens = estimate_tokens(&chunks[batch_end].text);
                if batch_end > batch_start && token_count + est_tokens > token_budget {
                    break;
                }
                token_count += est_tokens;
                batch_end += 1;
            }

            if batch_end == batch_start {
                batch_end = (batch_start + batch_size).min(chunks.len());
            }

            let mut batch_texts: Vec<String> = Vec::new();
            let mut batch_indices: Vec<usize> = Vec::new();
            let mut batch_hashes: Vec<String> = Vec::new();

            for idx in batch_start..batch_end {
                if cache.get(&chunks[idx].hash).is_some() {
                    continue;
                }
                batch_indices.push(idx);
                batch_texts.push(chunks[idx].text.clone());
                batch_hashes.push(chunks[idx].hash.clone());
            }

            if !batch_texts.is_empty() {
                batches.push(Batch {
                    indices: batch_indices,
                    texts: batch_texts,
                    hashes: batch_hashes,
                });
            }

            batch_start = batch_end;
        }

        let cache_hits = chunks.len() - batches.iter().map(|b| b.indices.len()).sum::<usize>();
        let cache_misses: usize = batches.iter().map(|b| b.indices.len()).sum();

        for idx in 0..chunks.len() {
            if let Some(vec) = cache.get(&chunks[idx].hash) {
                vectors[idx] = Some(vec.clone());
            }
        }

        pb.set_position(cache_hits.min(chunks.len()) as u64);

        let cache = Arc::new(std::sync::Mutex::new(cache));
        let embedder = self.embedder.clone();
        let cancelled_clone = cancelled.clone();

        let batch_results: Result<Vec<(Vec<usize>, Vec<Vec<f32>>)>> = batches
            .into_par_iter()
            .map(|batch| {
                if cancelled_clone.load(Ordering::SeqCst) {
                    return Err(anyhow!("Indexing cancelled"));
                }

                let texts: Vec<String> = batch.texts;
                let hashes = batch.hashes;
                match embedder.embed_batch(&texts) {
                    Ok(batch_vectors) => {
                        let mut cache_guard = cache.lock().unwrap();
                        for (vec, hash) in batch_vectors.iter().zip(hashes.iter()) {
                            cache_guard.insert(hash.clone(), vec.clone());
                        }
                        Ok((batch.indices, batch_vectors))
                    }
                    Err(err) => {
                        let mut last_err = err;
                        for attempt in 2..=3 {
                            if cancelled_clone.load(Ordering::SeqCst) {
                                return Err(anyhow!("Indexing cancelled"));
                            }
                            thread::sleep(Duration::from_millis((attempt * 150) as u64));
                            match embedder.embed_batch(&texts) {
                                Ok(batch_vectors) => {
                                    let mut cache_guard = cache.lock().unwrap();
                                    for (vec, hash) in batch_vectors.iter().zip(hashes.iter()) {
                                        cache_guard.insert(hash.clone(), vec.clone());
                                    }
                                    return Ok((batch.indices, batch_vectors));
                                }
                                Err(e) => last_err = e,
                            }
                        }
                        Err(anyhow!("Embedding failed after retries: {}", last_err))
                    }
                }
            })
            .collect();

        let batch_results = batch_results?;

        let mut processed_count = cache_hits;
        for (indices, batch_vectors) in batch_results {
            for (vec, idx) in batch_vectors.into_iter().zip(indices.iter()) {
                vectors[*idx] = Some(vec);
                processed_count += 1;
                pb.set_position(processed_count.min(chunks.len()) as u64);
            }
        }

        let vectors: Vec<Vec<f32>> = vectors
            .into_iter()
            .enumerate()
            .map(|(idx, maybe_vec)| {
                maybe_vec.ok_or_else(|| {
                    anyhow!(
                        "Missing vector for chunk {} (hash {})",
                        idx,
                        chunks[idx].hash
                    )
                })
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;

        let embed_duration = embed_start.elapsed();

        pb.finish_with_message("embedding complete");

        let write_start = Instant::now();
        let metadata = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: root.to_path_buf(),
            repo_hash: store.repo_hash().to_string(),
            vector_dim: self.embedder.dimension(),
            indexed_at: chrono::Utc::now(),
            total_files: files.len(),
            total_chunks: chunks.len(),
        };

        let repository_index = RepositoryIndex::new(metadata, chunks, vectors);
        store.save(&repository_index)?;
        let write_duration = write_start.elapsed();

        Ok(IndexReport {
            files_indexed: files.len(),
            chunks_indexed: repository_index.chunks.len(),
            duration: total_start.elapsed(),
            timings: request.profile.then_some(IndexTimings {
                walk: walk_duration,
                chunk: chunk_duration,
                embed: embed_duration,
                write: write_duration,
            }),
            cache_hits,
            cache_misses,
        })
    }

    fn build_incremental(
        &self,
        root: &Path,
        existing: RepositoryIndex,
        dirty: DirtySet,
        store: &IndexStore,
        request: &IndexRequest,
        total_start: Instant,
    ) -> Result<IndexReport> {
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        for (chunk, vector) in existing.chunks.iter().zip(existing.vectors.iter()) {
            cache.insert(chunk.hash.clone(), vector.clone());
        }

        let mut by_path: HashMap<PathBuf, Vec<(CodeChunk, Vec<f32>)>> = HashMap::new();
        for (chunk, vector) in existing
            .chunks
            .into_iter()
            .zip(existing.vectors.into_iter())
        {
            let key = normalize_to_relative(&root.join(&chunk.path), root);
            by_path.entry(key).or_default().push((chunk, vector));
        }

        let default_excludes = build_default_excludes();
        let mut cache_hits: usize = 0;
        let mut cache_misses: usize = 0;

        // Handle deletions up front
        let mut _deleted_files = 0usize;
        for deleted in dirty.deleted.iter() {
            let rel = normalize_to_relative(deleted, root);

            // Remove exact file matches.
            if by_path.remove(&rel).is_some() {
                _deleted_files += 1;
                continue;
            }

            // If the deleted path is a directory (or prefix), remove all entries under it.
            let to_remove: Vec<PathBuf> = by_path
                .keys()
                .filter(|p| p.starts_with(&rel))
                .cloned()
                .collect();
            if !to_remove.is_empty() {
                for key in to_remove {
                    if by_path.remove(&key).is_some() {
                        _deleted_files += 1;
                    }
                }
            }
        }

        let mut touched_paths: Vec<PathBuf> = Vec::new();
        for touched in dirty.touched.iter() {
            let full = if touched.is_absolute() {
                touched.clone()
            } else {
                root.join(touched)
            };
            if full.is_dir() {
                touched_paths.extend(collect_files(&full));
            } else {
                touched_paths.push(full);
            }
        }

        let base_batch_size = determine_batch_size(request.batch_size);
        let token_budget = determine_token_budget();

        let mut chunk_duration = Duration::ZERO;
        let mut embed_duration = Duration::ZERO;

        for full_path in touched_paths {
            let rel = normalize_to_relative(&full_path, root);

            if is_probably_binary(&full_path) {
                continue;
            }

            if let Ok(meta) = std::fs::metadata(&full_path) {
                if meta.len() > MAX_FILE_BYTES {
                    continue;
                }
            }

            if default_excludes.is_match(rel.as_path()) {
                continue;
            }

            if !full_path.exists() {
                if by_path.remove(&rel).is_some() {
                    _deleted_files += 1;
                }
                continue;
            }

            let chunk_start = Instant::now();
            let chunks = match chunker::chunk_file(&full_path, root) {
                Ok(ch) => ch,
                Err(err) => {
                    warn!("path" = %full_path.display(), "error" = %err, "msg" = "skipping");
                    continue;
                }
            };
            chunk_duration += chunk_start.elapsed();

            by_path.remove(&rel); // clear any old entries for this path

            if chunks.is_empty() {
                continue;
            }

            let mut file_results: Vec<Option<(CodeChunk, Vec<f32>)>> = vec![None; chunks.len()];
            let mut batch_texts: Vec<String> = Vec::new();
            let mut batch_indices: Vec<usize> = Vec::new();
            let mut batch_chunks: Vec<CodeChunk> = Vec::new();

            for (idx, chunk) in chunks.into_iter().enumerate() {
                if let Some(vec) = cache.get(&chunk.hash) {
                    cache_hits += 1;
                    file_results[idx] = Some((chunk, vec.clone()));
                } else {
                    cache_misses += 1;
                    batch_indices.push(idx);
                    batch_texts.push(chunk.text.clone());
                    batch_chunks.push(chunk);
                }
            }

            if !batch_texts.is_empty() {
                let embed_start = Instant::now();
                let mut start = 0usize;
                while start < batch_texts.len() {
                    let mut end = start;
                    let mut tokens = 0usize;
                    while end < batch_texts.len() && (end - start) < base_batch_size {
                        let est = estimate_tokens(&batch_texts[end]);
                        if end > start && tokens + est > token_budget {
                            break;
                        }
                        tokens += est;
                        end += 1;
                    }
                    if end == start {
                        end = (start + base_batch_size).min(batch_texts.len());
                    }

                    let slice = &batch_texts[start..end];
                    let vectors = self.embed_batch_with_retry(slice)?;
                    for (vec, idx_in_batch) in vectors.into_iter().zip(start..end) {
                        let global_idx = batch_indices[idx_in_batch];
                        let chunk = batch_chunks[idx_in_batch].clone();
                        cache.insert(chunk.hash.clone(), vec.clone());
                        file_results[global_idx] = Some((chunk, vec));
                    }

                    start = end;
                }
                embed_duration += embed_start.elapsed();
            }

            let ready: Vec<(CodeChunk, Vec<f32>)> = file_results
                .into_iter()
                .enumerate()
                .map(|(idx, maybe)| {
                    maybe.ok_or_else(|| {
                        anyhow!(
                            "Missing vector for chunk {} in {}",
                            idx,
                            full_path.display()
                        )
                    })
                })
                .collect::<Result<_>>()?;

            by_path.insert(rel, ready);
        }

        // Flatten and sort by path for determinism
        let mut entries: Vec<(PathBuf, Vec<(CodeChunk, Vec<f32>)>)> = by_path.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut chunks: Vec<CodeChunk> = Vec::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for (_path, pairs) in entries.into_iter() {
            for (chunk, vector) in pairs {
                chunks.push(chunk);
                vectors.push(vector);
            }
        }

        let write_start = Instant::now();
        let metadata = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: root.to_path_buf(),
            repo_hash: store.repo_hash().to_string(),
            vector_dim: self.embedder.dimension(),
            indexed_at: chrono::Utc::now(),
            total_files: chunks
                .iter()
                .map(|c| c.path.clone())
                .collect::<HashSet<_>>()
                .len(),
            total_chunks: chunks.len(),
        };

        let repository_index = RepositoryIndex::new(metadata, chunks, vectors);
        store.save(&repository_index)?;
        let write_duration = write_start.elapsed();

        Ok(IndexReport {
            files_indexed: repository_index.metadata.total_files,
            chunks_indexed: repository_index.metadata.total_chunks,
            duration: total_start.elapsed(),
            timings: request.profile.then_some(IndexTimings {
                walk: Duration::ZERO,
                chunk: chunk_duration,
                embed: embed_duration,
                write: write_duration,
            }),
            cache_hits,
            cache_misses,
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

                if is_probably_binary(e.path()) {
                    return None;
                }

                if let Ok(meta) = e.metadata() {
                    if meta.len() > MAX_FILE_BYTES {
                        return None;
                    }
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

fn is_probably_binary(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("min") || BINARY_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
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

fn normalize_to_relative(path: &Path, root: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix(root) {
        return stripped.to_path_buf();
    }

    if !path.is_absolute() {
        return path.to_path_buf();
    }

    // Try canonicalizing the full path first; this will fail for deleted files.
    if let Ok(canonicalized) = path.canonicalize() {
        if let Ok(stripped) = canonicalized.strip_prefix(root) {
            return stripped.to_path_buf();
        }
        return canonicalized;
    }

    // Deleted paths can't be canonicalized; normalize via their existing parent.
    if let (Some(parent), Some(file_name)) = (path.parent(), path.file_name()) {
        if let Ok(parent_canonical) = parent.canonicalize() {
            let recomposed = parent_canonical.join(file_name);
            if let Ok(stripped) = recomposed.strip_prefix(root) {
                return stripped.to_path_buf();
            }
            return recomposed;
        }
    }

    path.to_path_buf()
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
        "cuda" | "coreml" => 2048,
        _ => 512,
    }
}

fn adjust_batch_size_for_progress(base: usize, total_chunks: usize) -> usize {
    if total_chunks == 0 {
        return base;
    }

    let estimated_batches = (total_chunks + base - 1) / base;
    if estimated_batches >= 4 {
        return base;
    }

    let desired_batches = total_chunks.min(4);
    let progress_friendly = (total_chunks + desired_batches - 1) / desired_batches;

    progress_friendly.max(1).min(base)
}

fn determine_token_budget() -> usize {
    env::var("SGREP_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(512, 20_000))
        .unwrap_or(6_000)
}

fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, text.len() / 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedder;
    use std::fs;
    use uuid::Uuid;
    use serial_test::serial;

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
    #[serial]
    fn indexer_creates_report() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new_concrete(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
            profile: false,
            dirty: None,
        };

        let report = indexer.build_index(request).unwrap();

        assert!(report.files_indexed > 0);
        assert!(report.chunks_indexed > 0);
        assert!(report.duration.as_secs() < 60);

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    #[serial]
    fn indexer_saves_to_store() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new_concrete(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
            profile: false,
            dirty: None,
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
    #[serial]
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
    #[serial]
    fn index_metadata_has_correct_fields() {
        let embedder = Arc::new(Embedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
            profile: false,
            dirty: None,
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
    #[serial]
    fn canonical_handles_nonexistent_paths() {
        let nonexistent = PathBuf::from("/this/path/does/not/exist");
        let result = canonical(&nonexistent);
        assert_eq!(result, nonexistent);
    }

    #[test]
    #[serial]
    fn determine_batch_size_respects_env() {
        env::set_var("SGREP_BATCH_SIZE", "1024");
        assert_eq!(determine_batch_size(None), 1024);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[test]
    #[serial]
    fn determine_batch_size_prefers_override() {
        env::set_var("SGREP_BATCH_SIZE", "64");
        assert_eq!(determine_batch_size(Some(512)), 512);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[test]
    #[serial]
    fn adjust_batch_size_reduces_single_batch_case() {
        // 95 chunks with a 256 base batch size would normally be a single batch.
        // We want to split this so the progress bar can show movement.
        let adjusted = adjust_batch_size_for_progress(256, 95);
        assert!(adjusted < 256);
        let batches = (95 + adjusted - 1) / adjusted;
        assert!(batches >= 2);
    }

    #[test]
    #[serial]
    fn adjust_batch_size_keeps_large_jobs_intact() {
        // When we already have plenty of batches, don't shrink the size.
        let adjusted = adjust_batch_size_for_progress(256, 10_000);
        assert_eq!(adjusted, 256);
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
    #[serial]
    fn indexer_aborts_after_repeated_embedding_failures() {
        let embedder = Arc::new(FailingEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let test_repo = create_test_repo();
        let request = IndexRequest {
            path: test_repo.clone(),
            force: true,
            batch_size: None,
            profile: false,
            dirty: None,
        };

        let result = indexer.build_index(request);
        assert!(result.is_err());
        let msg = format!("{:?}", result.err().unwrap());
        assert!(msg.contains("Embedding failed"));

        fs::remove_dir_all(&test_repo).ok();
    }

    #[derive(Clone, Default)]
    struct StubEmbedder;

    impl BatchEmbedder for StubEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn incremental_updates_dirty_paths_only() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let rust_path = repo.join("test.rs");
        std::fs::write(&rust_path, "fn hello_updated() { println!(\"Hi\"); }\n").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![rust_path.clone()],
                    deleted: Vec::new(),
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        let rust_chunk = index
            .chunks
            .iter()
            .find(|c| c.path.ends_with(std::path::Path::new("test.rs")))
            .unwrap();
        assert!(rust_chunk.text.contains("hello_updated"));

        let py_chunk = index
            .chunks
            .iter()
            .find(|c| c.path.ends_with(std::path::Path::new("test.py")))
            .unwrap();
        assert!(py_chunk.text.contains("def hello"));
    }

    #[test]
    #[serial]
    fn incremental_handles_deleted_files() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let py_path = repo.join("test.py");
        std::fs::remove_file(&py_path).unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![py_path.clone()],
                }),
            })
            .unwrap();

        assert!(report.files_indexed >= 1);

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 2);
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("test.py"))));
    }

    #[test]
    #[serial]
    fn incremental_handles_deleted_files_with_relative_paths() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        // Remove file before passing its relative path to the incremental run.
        std::fs::remove_file(repo.join("test.py")).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![PathBuf::from("test.py")],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 2);
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("test.py"))));
    }

    #[test]
    #[serial]
    fn incremental_handles_deleted_nested_file() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        let nested = repo.join("src").join("extra.rs");
        fs::write(&nested, "pub fn extra() {}").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        fs::remove_file(&nested).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![PathBuf::from("src/extra.rs")],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 3);
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("src/extra.rs"))));
    }

    #[test]
    #[serial]
    fn incremental_touched_directory_adds_new_files() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let new_file = repo.join("src").join("new.rs");
        fs::write(&new_file, "pub fn brand_new() {}").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![repo.join("src")],
                    deleted: Vec::new(),
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 4);
        assert!(index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("src/new.rs"))));
    }

    #[test]
    #[serial]
    fn incremental_mixed_dirty_set_updates_and_deletes() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let rust_path = repo.join("test.rs");
        fs::write(&rust_path, "fn hello() { println!(\"Updated\"); }").unwrap();
        let py_path = repo.join("test.py");
        fs::remove_file(&py_path).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![rust_path.clone()],
                    deleted: vec![py_path.clone()],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 2);
        let rust_chunk = index
            .chunks
            .iter()
            .find(|c| c.path.ends_with(std::path::Path::new("test.rs")))
            .unwrap();
        assert!(rust_chunk.text.contains("Updated"));
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("test.py"))));
    }

    #[test]
    #[serial]
    fn incremental_handles_deleted_directory() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        let old_dir = repo.join("old");
        fs::create_dir_all(&old_dir).unwrap();
        fs::write(old_dir.join("old.rs"), "pub fn legacy() {}").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        fs::remove_dir_all(&old_dir).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![PathBuf::from("old")],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.total_files, 3);
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("old.rs"))));
    }

    #[test]
    #[serial]
    fn token_budget_clamps_min_and_max() {
        env::set_var("SGREP_TOKEN_BUDGET", "128");
        assert_eq!(determine_token_budget(), 512);

        env::set_var("SGREP_TOKEN_BUDGET", "50000");
        assert_eq!(determine_token_budget(), 20_000);

        env::remove_var("SGREP_TOKEN_BUDGET");
    }

    #[test]
    #[serial]
    fn determine_batch_size_respects_device_env() {
        env::set_var("SGREP_DEVICE", "coreml");
        assert_eq!(determine_batch_size(None), 512);
        env::set_var("SGREP_DEVICE", "cuda");
        assert_eq!(determine_batch_size(None), 512);
        env::remove_var("SGREP_DEVICE");
    }

    #[test]
    #[serial]
    fn is_probably_binary_catches_minified_and_common_binary_exts() {
        assert!(is_probably_binary(&PathBuf::from("app.min")));
        assert!(is_probably_binary(&PathBuf::from("image.png")));
        assert!(!is_probably_binary(&PathBuf::from("main.rs")));
    }

    #[test]
    #[serial]
    fn collect_files_skips_binary_and_large_assets() {
        let temp_dir =
            std::env::temp_dir().join(format!("sgrep_collect_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let keep = temp_dir.join("keep.rs");
        fs::write(&keep, "fn keep() {}").unwrap();

        let binary = temp_dir.join("bundle.min.js");
        fs::write(&binary, "function minified(){}").unwrap();

        let big = temp_dir.join("big.txt");
        let large_contents = vec![b'a'; (MAX_FILE_BYTES + 1) as usize];
        fs::write(&big, large_contents).unwrap();

        let files = collect_files(&temp_dir);

        assert!(files.iter().any(|p| p.ends_with("keep.rs")));
        assert!(!files.iter().any(|p| p.ends_with("bundle.min.js")));
        assert!(!files.iter().any(|p| p.ends_with("big.txt")));

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn normalize_to_relative_handles_deleted_absolute_path() {
        let repo = create_test_repo();
        let missing = repo.join("ghost.rs");
        // Ensure the file truly does not exist
        let _ = std::fs::remove_file(&missing);

        let rel = normalize_to_relative(&missing, &repo);
        assert_eq!(rel, PathBuf::from("ghost.rs"));

        let nested_missing = repo.join("src").join("ghost_nested.rs");
        let _ = std::fs::remove_file(&nested_missing);
        let rel_nested = normalize_to_relative(&nested_missing, &repo);
        assert_eq!(rel_nested, PathBuf::from("src/ghost_nested.rs"));
    }
}
