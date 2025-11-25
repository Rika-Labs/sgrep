use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use globset::{Glob, GlobSetBuilder};
use ignore::{WalkBuilder, WalkState};

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

    #[allow(dead_code)]
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

        if files.is_empty() {
            return Ok(IndexReport {
                files_indexed: 0,
                chunks_indexed: 0,
                duration: total_start.elapsed(),
                timings: request.profile.then_some(IndexTimings {
                    walk: walk_duration,
                    chunk: Duration::ZERO,
                    embed: Duration::ZERO,
                    write: Duration::ZERO,
                }),
                cache_hits: 0,
                cache_misses: 0,
            });
        }

        let cancelled = Arc::new(AtomicBool::new(false));

        let pb =
            ProgressBar::with_draw_target(Some(files.len() as u64), ProgressDrawTarget::stderr());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓▒░  "),
        );
        pb.set_message("parsing files...");

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

        let mut file_to_chunks: HashMap<PathBuf, Vec<usize>> = HashMap::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            file_to_chunks
                .entry(chunk.path.clone())
                .or_default()
                .push(idx);
        }
        let total_files = file_to_chunks.len();

        pb.set_length(total_files as u64);
        pb.set_position(0);
        pb.reset_elapsed();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} Indexing files ({pos}/{len}) • {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("starting...");
        pb.enable_steady_tick(Duration::from_millis(100));

        let base_batch_size = determine_batch_size(request.batch_size);
        let batch_size = adjust_batch_size_for_progress(base_batch_size, chunks.len());
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];

        let cache: Arc<DashMap<String, Vec<f32>>> = Arc::new(DashMap::new());
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

            for (idx, chunk) in chunks.iter().enumerate().take(batch_end).skip(batch_start) {
                if cache.contains_key(&chunk.hash) {
                    continue;
                }
                batch_indices.push(idx);
                batch_texts.push(chunk.text.clone());
            }

            if !batch_texts.is_empty() {
                batches.push(Batch {
                    indices: batch_indices,
                    texts: batch_texts,
                });
            }

            batch_start = batch_end;
        }

        let cache_hits = chunks.len() - batches.iter().map(|b| b.indices.len()).sum::<usize>();
        let cache_misses: usize = batches.iter().map(|b| b.indices.len()).sum();

        let mut embedded_chunks: HashSet<usize> = HashSet::new();
        for idx in 0..chunks.len() {
            if let Some(vec) = cache.get(&chunks[idx].hash) {
                vectors[idx] = Some(vec.value().clone());
                embedded_chunks.insert(idx);
            }
        }

        let mut completed_files: HashSet<PathBuf> = HashSet::new();
        let mut files_completed = 0usize;

        for (file_path, chunk_indices) in &file_to_chunks {
            if chunk_indices
                .iter()
                .all(|idx| embedded_chunks.contains(idx))
            {
                completed_files.insert(file_path.clone());
                files_completed += 1;
            }
        }
        pb.set_position(files_completed as u64);
        if files_completed > 0 {
            pb.set_message(format!("indexed {} (cached)", files_completed));
        }

        let mut pending_chunks: Vec<(usize, String)> = Vec::new();
        for batch in batches.into_iter() {
            for (idx, text) in batch.indices.into_iter().zip(batch.texts.into_iter()) {
                pending_chunks.push((idx, text));
            }
        }

        if pending_chunks.is_empty() {
            pb.set_position(total_files as u64);
            pb.finish_with_message("all cached");
        } else {
            pb.set_message("loading AI model...");
            // Warm up the model with a single embedding
            let _ = self.embedder.embed(&pending_chunks[0].1);

            // Process in batches for efficiency, but update progress per-chunk for UX
            // Batch size tuned for optimal throughput while maintaining responsive progress
            let embed_batch_size = env::var("SGREP_EMBED_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32)
                .clamp(8, 128);

            let mut chunk_idx = 0;
            while chunk_idx < pending_chunks.len() {
                if cancelled.load(Ordering::SeqCst) {
                    return Err(anyhow!("Indexing cancelled"));
                }

                // Collect a batch of texts to embed
                let batch_end = (chunk_idx + embed_batch_size).min(pending_chunks.len());
                let batch_texts: Vec<String> = pending_chunks[chunk_idx..batch_end]
                    .iter()
                    .map(|(_, text)| text.clone())
                    .collect();

                // Embed the entire batch at once (much faster than one-at-a-time)
                let batch_vectors = self.embedder.embed_batch(&batch_texts)?;

                // Process results and update progress sequentially for UX
                for (i, vec) in batch_vectors.into_iter().enumerate() {
                    let (global_idx, _) = &pending_chunks[chunk_idx + i];
                    vectors[*global_idx] = Some(vec);
                    embedded_chunks.insert(*global_idx);

                    // Update file progress - files appear to complete sequentially
                    for (file_path, chunk_indices) in &file_to_chunks {
                        if !completed_files.contains(file_path)
                            && chunk_indices
                                .iter()
                                .all(|&idx| embedded_chunks.contains(&idx))
                        {
                            completed_files.insert(file_path.clone());
                            files_completed += 1;
                            pb.set_position(files_completed as u64);
                            let display_path =
                                file_path.strip_prefix(root).unwrap_or(file_path).display();
                            pb.set_message(format!("{}", display_path));
                        }
                    }
                    pb.tick();
                }

                chunk_idx = batch_end;
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

        for full_path in &touched_paths {
            let rel = normalize_to_relative(full_path, root);
            if !full_path.exists() {
                if by_path.remove(&rel).is_some() {
                    _deleted_files += 1;
                }
            } else {
                by_path.remove(&rel);
            }
        }

        let base_batch_size = determine_batch_size(request.batch_size);
        let token_budget = determine_token_budget();

        let cache = Arc::new(std::sync::Mutex::new(cache));
        let embedder = self.embedder.clone();
        let embed_timeout = determine_embed_timeout();
        let root = root.to_path_buf();

        struct FileResult {
            rel_path: PathBuf,
            chunks: Vec<(CodeChunk, Vec<f32>)>,
            cache_hits: usize,
            cache_misses: usize,
            chunk_duration: Duration,
            embed_duration: Duration,
        }

        let file_results: Result<Vec<FileResult>> = touched_paths
            .par_iter()
            .filter_map(|full_path| {
                let rel = normalize_to_relative(full_path, &root);

                if is_probably_binary(full_path) {
                    return None;
                }

                if let Ok(meta) = std::fs::metadata(full_path) {
                    if meta.len() > MAX_FILE_BYTES {
                        return None;
                    }
                }

                if default_excludes.is_match(rel.as_path()) {
                    return None;
                }

                if !full_path.exists() {
                    return None;
                }

                let chunk_start = Instant::now();
                let chunks = match chunker::chunk_file(full_path, &root) {
                    Ok(ch) => ch,
                    Err(err) => {
                        warn!("path" = %full_path.display(), "error" = %err, "msg" = "skipping");
                        return None;
                    }
                };
                let chunk_duration = chunk_start.elapsed();

                if chunks.is_empty() {
                    return Some(Ok(FileResult {
                        rel_path: rel,
                        chunks: Vec::new(),
                        cache_hits: 0,
                        cache_misses: 0,
                        chunk_duration,
                        embed_duration: Duration::ZERO,
                    }));
                }

                let mut file_results: Vec<Option<(CodeChunk, Vec<f32>)>> = vec![None; chunks.len()];
                let mut batch_texts: Vec<String> = Vec::new();
                let mut batch_indices: Vec<usize> = Vec::new();
                let mut batch_chunks: Vec<CodeChunk> = Vec::new();
                let mut local_cache_hits = 0usize;
                let mut local_cache_misses = 0usize;

                {
                    let cache_guard = cache.lock().unwrap();
                    for (idx, chunk) in chunks.into_iter().enumerate() {
                        if let Some(vec) = cache_guard.get(&chunk.hash) {
                            local_cache_hits += 1;
                            file_results[idx] = Some((chunk, vec.clone()));
                        } else {
                            local_cache_misses += 1;
                            batch_indices.push(idx);
                            batch_texts.push(chunk.text.clone());
                            batch_chunks.push(chunk);
                        }
                    }
                }

                let embed_duration = if !batch_texts.is_empty() {
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
                        let vectors = {
                            let mut attempt = 1;
                            let max_attempts = 3;
                            loop {
                                match embed_batch_with_timeout(
                                    embedder.clone(),
                                    slice.to_vec(),
                                    embed_timeout,
                                ) {
                                    Ok(v) => break v,
                                    Err(_err) if attempt < max_attempts => {
                                        thread::sleep(Duration::from_millis(
                                            (attempt * 200) as u64,
                                        ));
                                        attempt += 1;
                                    }
                                    Err(err) => {
                                        return Some(Err(anyhow!(
                                            "Embedding failed after {} attempts: {}",
                                            attempt,
                                            err
                                        )))
                                    }
                                }
                            }
                        };

                        {
                            let mut cache_guard = cache.lock().unwrap();
                            for (vec, idx_in_batch) in vectors.into_iter().zip(start..end) {
                                let global_idx = batch_indices[idx_in_batch];
                                let chunk = batch_chunks[idx_in_batch].clone();
                                cache_guard.insert(chunk.hash.clone(), vec.clone());
                                file_results[global_idx] = Some((chunk, vec));
                            }
                        }

                        start = end;
                    }
                    embed_start.elapsed()
                } else {
                    Duration::ZERO
                };

                let ready: Result<Vec<(CodeChunk, Vec<f32>)>> = file_results
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
                    .collect();

                match ready {
                    Ok(chunks) => Some(Ok(FileResult {
                        rel_path: rel,
                        chunks,
                        cache_hits: local_cache_hits,
                        cache_misses: local_cache_misses,
                        chunk_duration,
                        embed_duration,
                    })),
                    Err(e) => Some(Err(e)),
                }
            })
            .collect();

        let file_results = file_results?;

        let mut chunk_duration = Duration::ZERO;
        let mut embed_duration = Duration::ZERO;

        for result in file_results {
            cache_hits += result.cache_hits;
            cache_misses += result.cache_misses;
            chunk_duration += result.chunk_duration;
            embed_duration += result.embed_duration;
            if !result.chunks.is_empty() {
                by_path.insert(result.rel_path, result.chunks);
            }
        }

        // Flatten and sort by path for determinism
        #[allow(clippy::type_complexity)]
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

    #[allow(dead_code)]
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
    use std::sync::Mutex;

    let default_excludes = build_default_excludes();
    let files = Arc::new(Mutex::new(Vec::new()));
    let root_arc = Arc::new(root.to_path_buf());
    let excludes_arc = Arc::new(default_excludes);

    WalkBuilder::new(root)
        .hidden(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .parents(true)
        .follow_links(false)
        .add_custom_ignore_filename(".sgrepignore")
        .threads(num_cpus::get().min(8))
        .build_parallel()
        .run(|| {
            let files = Arc::clone(&files);
            let root = Arc::clone(&root_arc);
            let excludes = Arc::clone(&excludes_arc);

            Box::new(move |entry| {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => return WalkState::Continue,
                };

                if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    return WalkState::Continue;
                }

                if is_probably_binary(entry.path()) {
                    return WalkState::Continue;
                }

                if let Ok(meta) = entry.metadata() {
                    if meta.len() > MAX_FILE_BYTES {
                        return WalkState::Continue;
                    }
                }

                let path = entry.path();
                let relative_path = path.strip_prefix(root.as_path()).unwrap_or(path);

                if excludes.is_match(relative_path) {
                    return WalkState::Continue;
                }

                files.lock().unwrap().push(entry.into_path());
                WalkState::Continue
            })
        });

    Arc::try_unwrap(files)
        .expect("All references should be dropped")
        .into_inner()
        .unwrap()
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
        "cuda" | "coreml" => 128,
        _ => 64, // Larger batches for better ONNX efficiency
    }
}

fn adjust_batch_size_for_progress(base: usize, total_chunks: usize) -> usize {
    if total_chunks == 0 {
        return base;
    }

    let estimated_batches = total_chunks.div_ceil(base);
    if estimated_batches >= 4 {
        return base;
    }

    let desired_batches = total_chunks.min(4);
    let progress_friendly = total_chunks.div_ceil(desired_batches);

    progress_friendly.max(1).min(base)
}

fn determine_token_budget() -> usize {
    env::var("SGREP_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(512, 20_000))
        .unwrap_or(6_000)
}

fn determine_embed_timeout() -> Duration {
    env::var("SGREP_EMBED_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(120))
}

fn embed_batch_with_timeout(
    embedder: Arc<dyn BatchEmbedder>,
    texts: Vec<String>,
    timeout: Duration,
) -> Result<Vec<Vec<f32>>> {
    let text_len = texts.len();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = embedder.embed_batch(&texts);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(res) => res,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!(
            "embedding batch timed out after {:?} ({} items)",
            timeout,
            text_len
        )),
        Err(err) => Err(anyhow!("embedding worker failed: {}", err)),
    }
}

fn estimate_tokens(text: &str) -> usize {
    let mut token_count = 0usize;
    let mut in_word = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            if !in_word {
                token_count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if is_operator_or_punctuation(ch) {
                token_count += 1;
            }
        }
    }

    token_count.max(1)
}

fn is_operator_or_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '<'
            | '>'
            | ';'
            | ':'
            | ','
            | '.'
            | '='
            | '+'
            | '-'
            | '*'
            | '/'
            | '%'
            | '!'
            | '&'
            | '|'
            | '^'
            | '~'
            | '?'
            | '@'
            | '#'
            | '$'
            | '\\'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration as StdDuration;
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct DeterministicEmbedder;

    impl BatchEmbedder for DeterministicEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0])
                .collect())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

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
        let embedder = Arc::new(DeterministicEmbedder::default());
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
        let embedder = Arc::new(DeterministicEmbedder::default());
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
    fn collect_files_respects_root_prefixed_gitignore() {
        let test_repo = create_test_repo();

        std::process::Command::new("git")
            .args(&["init"])
            .current_dir(&test_repo)
            .output()
            .ok();

        fs::write(
            test_repo.join(".gitignore"),
            "/node_modules\n/dist\n.vercel\n",
        )
        .unwrap();

        let node_modules = test_repo.join("node_modules");
        fs::create_dir_all(&node_modules).unwrap();
        fs::write(node_modules.join("dep.js"), "module.exports = {}").unwrap();

        let dist = test_repo.join("dist");
        fs::create_dir_all(&dist).unwrap();
        fs::write(dist.join("bundle.js"), "bundled").unwrap();

        let vercel = test_repo.join(".vercel");
        fs::create_dir_all(&vercel).unwrap();
        fs::write(vercel.join("output.js"), "output").unwrap();

        fs::write(test_repo.join("src.js"), "real code").unwrap();

        let files = collect_files(&test_repo);
        let file_names: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();

        eprintln!("Collected files: {:?}", file_names);

        assert!(
            !file_names.contains(&"dep.js".to_string()),
            "node_modules/dep.js should be ignored"
        );
        assert!(
            !file_names.contains(&"bundle.js".to_string()),
            "dist/bundle.js should be ignored"
        );
        assert!(
            !file_names.contains(&"output.js".to_string()),
            ".vercel/output.js should be ignored"
        );
        assert!(file_names.contains(&"src.js".to_string()));

        fs::remove_dir_all(&test_repo).ok();
    }

    #[test]
    #[serial]
    fn index_metadata_has_correct_fields() {
        let embedder = Arc::new(DeterministicEmbedder::default());
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
    fn indexer_aborts_on_embedding_failure() {
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
        assert!(msg.contains("intentional embed failure"));

        fs::remove_dir_all(&test_repo).ok();
    }

    #[derive(Clone, Default)]
    struct IncompleteEmbedder;

    impl BatchEmbedder for IncompleteEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            // Return no vectors to leave holes in the results.
            let _ = texts;
            Ok(Vec::new())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn indexer_errors_when_vectors_missing() {
        let embedder = Arc::new(IncompleteEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        let result = indexer.build_index(IndexRequest {
            path: repo.clone(),
            force: true,
            batch_size: Some(2),
            profile: false,
            dirty: None,
        });

        assert!(result.is_err());
        let msg = format!("{:?}", result.err().unwrap());
        assert!(
            msg.contains("No embedding generated") || msg.contains("Missing vector"),
            "Expected error about missing embeddings, got: {}",
            msg
        );

        fs::remove_dir_all(&repo).ok();
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
    fn incremental_incompatible_index_with_dirty_triggers_full_reindex() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();

        // seed an incompatible index (different vector_dim and version)
        let chunk = chunker::chunk_file(&repo.join("test.rs"), &repo)
            .unwrap()
            .pop()
            .unwrap();
        let bad_meta = IndexMetadata {
            version: "0.0.0".into(),
            repo_path: repo.clone(),
            repo_hash: "hash".into(),
            vector_dim: 3,
            indexed_at: chrono::Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        let bad_index =
            RepositoryIndex::new(bad_meta, vec![chunk.clone()], vec![vec![0.1, 0.2, 0.3]]);
        let store = IndexStore::new(&repo).unwrap();
        store.save(&bad_index).unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![repo.join("test.rs")],
                    deleted: vec![repo.join("missing.rs")],
                }),
            })
            .unwrap();

        assert!(report.cache_hits + report.cache_misses > 0);
        std::fs::remove_dir_all(&repo).ok();
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
    fn incremental_removes_deleted_directory_tree() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder.clone());

        let _home = set_test_home();
        let repo = create_test_repo();
        let nested_dir = repo.join("nested_dir");
        fs::create_dir_all(&nested_dir).unwrap();
        fs::write(nested_dir.join("nested.rs"), "fn inner() {}\n").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        fs::remove_dir_all(&nested_dir).unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![nested_dir.clone()],
                }),
            })
            .unwrap();

        assert!(report.files_indexed >= 1);

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();
        assert!(!index
            .chunks
            .iter()
            .any(|c| c.path.to_string_lossy().contains("nested_dir")));
        fs::remove_dir_all(&repo).ok();
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
        assert_eq!(determine_batch_size(None), 128);
        env::set_var("SGREP_DEVICE", "cuda");
        assert_eq!(determine_batch_size(None), 128);
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
        let temp_dir = std::env::temp_dir().join(format!("sgrep_collect_test_{}", Uuid::new_v4()));
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

    #[derive(Clone)]
    struct SlowEmbedder {
        delay: StdDuration,
    }

    impl BatchEmbedder for SlowEmbedder {
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            std::thread::sleep(self.delay);
            Ok(vec![vec![1.0, 0.0, 0.0, 0.0]])
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn embed_batch_with_timeout_times_out() {
        let embedder = Arc::new(SlowEmbedder {
            delay: StdDuration::from_millis(200),
        });
        let start = Instant::now();
        let result =
            embed_batch_with_timeout(embedder, vec!["slow".into()], StdDuration::from_millis(50));
        assert!(result.is_err());
        assert!(start.elapsed() >= StdDuration::from_millis(50));
    }

    #[test]
    #[serial]
    fn embed_batch_with_timeout_succeeds() {
        let embedder = Arc::new(SlowEmbedder {
            delay: StdDuration::from_millis(5),
        });
        let result =
            embed_batch_with_timeout(embedder, vec!["ok".into()], StdDuration::from_millis(100));
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn determine_embed_timeout_reads_env() {
        env::set_var("SGREP_EMBED_TIMEOUT_SECS", "2");
        let timeout = determine_embed_timeout();
        assert_eq!(timeout, Duration::from_secs(2));
        env::remove_var("SGREP_EMBED_TIMEOUT_SECS");
    }

    #[test]
    #[serial]
    fn build_index_handles_empty_repository() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_empty_repo_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: true,
                dirty: None,
            })
            .unwrap();

        assert_eq!(report.files_indexed, 0);
        assert_eq!(report.chunks_indexed, 0);
        assert!(report.timings.is_some());

        fs::remove_dir_all(&repo).ok();
    }

    #[derive(Clone, Default)]
    struct RetryingEmbedder {
        attempts: Arc<std::sync::Mutex<usize>>,
    }

    impl BatchEmbedder for RetryingEmbedder {
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut guard = self.attempts.lock().unwrap();
            *guard += 1;
            if *guard < 2 {
                Err(anyhow!("temporary failure"))
            } else {
                Ok(vec![vec![1.0, 0.0, 0.0, 0.0]])
            }
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[derive(Clone, Default)]
    struct AlwaysFailRetryEmbedder;

    impl BatchEmbedder for AlwaysFailRetryEmbedder {
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Err(anyhow!("permanent failure"))
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn embed_batch_with_retry_recovers_after_initial_error() {
        let embedder = Arc::new(RetryingEmbedder::default());
        let indexer = Indexer::new(embedder);

        let vectors = indexer
            .embed_batch_with_retry(&vec!["hello".into()])
            .expect("should recover on second attempt");
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0], vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[serial]
    fn embed_batch_with_retry_fails_after_max_attempts() {
        let embedder = Arc::new(AlwaysFailRetryEmbedder::default());
        let indexer = Indexer::new(embedder);

        let result = indexer.embed_batch_with_retry(&vec!["fail".into()]);
        assert!(result.is_err());
    }

    #[test]
    fn estimate_tokens_counts_words_and_operators() {
        assert_eq!(estimate_tokens("fn hello() {}"), 6);
        assert_eq!(estimate_tokens("let x = 5 + 3;"), 7);
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("   "), 1);
    }

    #[test]
    fn estimate_tokens_handles_complex_code() {
        let code = "fn calculate(x: i32, y: i32) -> i32 { x + y }";
        let tokens = estimate_tokens(code);
        assert!(tokens > 10);
        assert!(tokens < 30);
    }

    #[test]
    fn estimate_tokens_counts_identifiers_once() {
        assert_eq!(estimate_tokens("hello_world"), 1);
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens("hello123world"), 1);
    }

    #[test]
    fn is_operator_or_punctuation_covers_common_operators() {
        assert!(is_operator_or_punctuation('('));
        assert!(is_operator_or_punctuation(')'));
        assert!(is_operator_or_punctuation('{'));
        assert!(is_operator_or_punctuation('}'));
        assert!(is_operator_or_punctuation(';'));
        assert!(is_operator_or_punctuation('='));
        assert!(is_operator_or_punctuation('+'));
        assert!(!is_operator_or_punctuation('a'));
        assert!(!is_operator_or_punctuation(' '));
        assert!(!is_operator_or_punctuation('\n'));
    }

    #[derive(Clone)]
    struct ProgressTrackingEmbedder {
        call_count: Arc<AtomicUsize>,
        delay_ms: u64,
    }

    impl BatchEmbedder for ProgressTrackingEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.delay_ms > 0 {
                std::thread::sleep(StdDuration::from_millis(self.delay_ms));
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn chunks_embedded_in_batches() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(ProgressTrackingEmbedder {
            call_count: call_count.clone(),
            delay_ms: 1,
        });
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_batched_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        for i in 0..5 {
            fs::write(
                repo.join(format!("file{}.rs", i)),
                format!("fn func{}() {{}}", i),
            )
            .unwrap();
        }

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        assert!(report.chunks_indexed >= 5);
        let calls = call_count.load(Ordering::SeqCst);
        // With batching, we should have fewer calls than chunks (batches are more efficient)
        // At minimum we need 1 call for warmup + at least 1 batch call
        assert!(
            calls >= 1 && calls <= report.chunks_indexed,
            "Expected batched calls (1 to {}), got {}",
            report.chunks_indexed,
            calls
        );

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn file_progress_tracks_completed_files() {
        let embedder = Arc::new(DeterministicEmbedder::default());
        let indexer = Indexer::new_concrete(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_file_progress_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("a.rs"), "fn a() {}").unwrap();
        fs::write(repo.join("b.rs"), "fn b() {}").unwrap();
        fs::write(repo.join("c.rs"), "fn c() {}").unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        assert_eq!(report.files_indexed, 3);
        assert!(report.chunks_indexed >= 3);

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn cache_hits_skip_embedding() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(ProgressTrackingEmbedder {
            call_count: call_count.clone(),
            delay_ms: 0,
        });
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_cache_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("test.rs"), "fn test() {}").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let first_calls = call_count.load(Ordering::SeqCst);
        assert!(first_calls >= 1);

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        assert!(report.cache_hits >= 1);
        assert_eq!(report.cache_misses, 0);

        fs::remove_dir_all(&repo).ok();
    }
}
