mod batch;
mod files;
mod hierarchy;

pub use batch::{
    adjust_batch_size_for_progress, determine_batch_size, determine_embed_timeout,
    determine_token_budget, embed_batch_with_timeout, estimate_tokens,
};
pub use files::{
    build_default_excludes, canonical, collect_files, detect_language_for_graph,
    is_probably_binary, normalize_to_relative, MAX_FILE_BYTES,
};
pub use hierarchy::build_hierarchical_index;
#[cfg(test)]
pub(crate) use hierarchy::compute_directory_embeddings;

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use console::style;
use dashmap::DashMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use tracing::{debug, warn};

use crate::chunker::{self, CodeChunk};
use crate::embedding::{BatchEmbedder, EmbedProgress, ProgressCallback};
use crate::graph::{CodeGraph, SymbolExtractor};
use crate::store::{IndexMetadata, IndexStore, RepositoryIndex};

const PARSE_TEMPLATE: &str = "{prefix} Parsing files ({pos}/{len}, {percent}%)";
const INDEX_TEMPLATE: &str = "{prefix} Indexing files ({pos}/{len}, {percent}%) • {msg}";
const EMBED_TEMPLATE: &str = "{prefix} Embedding chunks ({pos}/{len}, {percent}%)";

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
    pub graph_symbols: usize,
    pub graph_edges: usize,
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
    pub graph: Duration,
    pub write: Duration,
}

fn make_progress_callback(pb: ProgressBar, offset: usize, total: usize) -> ProgressCallback {
    Box::new(move |progress: EmbedProgress| {
        let mut position = offset.saturating_add(progress.completed);
        if position > total {
            position = total;
        }

        pb.set_position(position as u64);

        if let Some(msg) = &progress.message {
            pb.set_message(msg.clone());
        }
    })
}

#[derive(Clone)]
pub struct Indexer {
    embedder: Arc<dyn BatchEmbedder>,
}

impl Indexer {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self { embedder }
    }

    pub fn embedder_dimension(&self) -> usize {
        self.embedder.dimension()
    }

    #[allow(dead_code)]
    pub fn new_concrete<E: BatchEmbedder + 'static>(embedder: Arc<E>) -> Self {
        Self { embedder }
    }

    pub fn with_remote_embedding(
        embedder: Arc<dyn BatchEmbedder>,
        _remote_embedding: bool,
    ) -> Self {
        Self { embedder }
    }

    pub fn warmup(&self) -> Result<()> {
        let _ = self.embedder.embed("warmup")?;
        Ok(())
    }

    pub fn build_index(&self, request: IndexRequest) -> Result<IndexReport> {
        let total_start = Instant::now();
        let root = canonical(&request.path);
        if request.force {
            debug!("Full indexing: {}", root.display());
        } else {
            debug!("Indexing: {}", root.display());
        }
        let store = IndexStore::new(&root)?;
        if store.is_building() {
            return Err(anyhow!(
                "Index build already in progress for {}",
                root.display()
            ));
        }
        let _build_guard = store.start_build_guard()?;
        let existing_index = if request.force {
            None
        } else {
            store.load().ok().flatten()
        };

        if let (Some(dirty), Some(index)) = (&request.dirty, existing_index.clone()) {
            let reusable = index.metadata.vector_dim == self.embedder.dimension()
                && index.metadata.version == env!("CARGO_PKG_VERSION");
            if reusable && !dirty.is_empty() {
                debug!(
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
                        "msg" = "Index incompatible, doing full reindex",
                        "index_dim" = index.metadata.vector_dim,
                        "embedder_dim" = self.embedder.dimension()
                    );
                } else {
                    debug!("No changes detected, skipping incremental");
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
                    graph: Duration::ZERO,
                    write: Duration::ZERO,
                }),
                cache_hits: 0,
                cache_misses: 0,
                graph_symbols: 0,
                graph_edges: 0,
            });
        }

        let cancelled = Arc::new(AtomicBool::new(false));

        let info_prefix = style("[info]").blue().bold().to_string();
        let pb =
            ProgressBar::with_draw_target(Some(files.len() as u64), ProgressDrawTarget::stderr());
        pb.set_prefix(info_prefix.clone());
        pb.set_style(
            ProgressStyle::with_template(PARSE_TEMPLATE)
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("parsing files");

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
            ProgressStyle::with_template(INDEX_TEMPLATE)
                .unwrap_or_else(|_| ProgressStyle::default_bar()),
        );
        pb.set_message("starting");

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

        let pending_batches: Vec<Batch> = batches;

        if pending_batches.is_empty() {
            pb.set_position(total_files as u64);
            pb.finish_with_message(format!("{} all cached", info_prefix));
        } else {
            let total_pending = pending_batches
                .iter()
                .map(|b| b.indices.len())
                .sum::<usize>();

            pb.set_style(
                ProgressStyle::with_template(EMBED_TEMPLATE)
                    .unwrap_or_else(|_| ProgressStyle::default_bar()),
            );
            pb.set_length(total_pending as u64);
            pb.set_position(0);

            let texts: Vec<String> = pending_batches
                .iter()
                .flat_map(|b| b.texts.iter().cloned())
                .collect();
            let indices: Vec<usize> = pending_batches
                .iter()
                .flat_map(|b| b.indices.iter().copied())
                .collect();

            let progress_callback = make_progress_callback(pb.clone(), 0, total_pending);

            let all_embeddings = self
                .embedder
                .embed_batch_with_progress(&texts, Some(&progress_callback))?;

            for (i, vec) in all_embeddings.into_iter().enumerate() {
                let idx = indices[i];
                vectors[idx] = Some(vec);
                embedded_chunks.insert(idx);
            }

            for (file_path, chunk_indices) in &file_to_chunks {
                if chunk_indices.iter().all(|&i| embedded_chunks.contains(&i)) {
                    completed_files.insert(file_path.clone());
                    files_completed += 1;
                }
            }

            pb.set_position(total_pending as u64);
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

        pb.finish_with_message(format!("{} embedding complete", info_prefix));

        let graph_start = Instant::now();
        let graph = self.build_graph(root, &files);
        let graph_stats = graph.stats();
        let graph_duration = graph_start.elapsed();

        let hierarchy = build_hierarchical_index(&chunks, &vectors);
        let hier_stats = hierarchy.stats();
        debug!(
            "Built hierarchical index: {} files, {} directories",
            hier_stats.file_count, hier_stats.directory_count
        );

        let write_start = Instant::now();
        let metadata = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: root.to_path_buf(),
            repo_hash: store.repo_hash().to_string(),
            vector_dim: self.embedder.dimension(),
            indexed_at: chrono::Utc::now(),
            total_files: files.len(),
            total_chunks: chunks.len(),
            embedding_model: crate::embedding::EmbeddingModel::default()
                .config()
                .name
                .to_string(),
        };

        let repository_index = RepositoryIndex::new(metadata, chunks, vectors);
        store.save(&repository_index)?;
        store.save_graph(&graph)?;
        store.save_hierarchy(&hierarchy)?;
        let write_duration = write_start.elapsed();

        Ok(IndexReport {
            files_indexed: files.len(),
            chunks_indexed: repository_index.chunks.len(),
            duration: total_start.elapsed(),
            timings: request.profile.then_some(IndexTimings {
                walk: walk_duration,
                chunk: chunk_duration,
                embed: embed_duration,
                graph: graph_duration,
                write: write_duration,
            }),
            cache_hits,
            cache_misses,
            graph_symbols: graph_stats.total_symbols,
            graph_edges: graph_stats.total_edges,
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

        let mut _deleted_files = 0usize;
        for deleted in dirty.deleted.iter() {
            let rel = normalize_to_relative(deleted, root);

            if by_path.remove(&rel).is_some() {
                _deleted_files += 1;
                continue;
            }

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

        let graph_start = Instant::now();
        let all_files: Vec<PathBuf> = chunks
            .iter()
            .map(|c| root.join(&c.path))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let graph = self.build_graph(&root, &all_files);
        let graph_stats = graph.stats();
        let graph_duration = graph_start.elapsed();

        let hierarchy = build_hierarchical_index(&chunks, &vectors);
        let hier_stats = hierarchy.stats();
        debug!(
            "Built hierarchical index: {} files, {} directories",
            hier_stats.file_count, hier_stats.directory_count
        );

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
            embedding_model: crate::embedding::EmbeddingModel::default()
                .config()
                .name
                .to_string(),
        };

        let repository_index = RepositoryIndex::new(metadata, chunks, vectors);
        store.save(&repository_index)?;
        store.save_graph(&graph)?;
        store.save_hierarchy(&hierarchy)?;
        let write_duration = write_start.elapsed();

        Ok(IndexReport {
            files_indexed: repository_index.metadata.total_files,
            chunks_indexed: repository_index.metadata.total_chunks,
            duration: total_start.elapsed(),
            timings: request.profile.then_some(IndexTimings {
                walk: Duration::ZERO,
                chunk: chunk_duration,
                embed: embed_duration,
                graph: graph_duration,
                write: write_duration,
            }),
            cache_hits,
            cache_misses,
            graph_symbols: graph_stats.total_symbols,
            graph_edges: graph_stats.total_edges,
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

    fn build_graph(&self, root: &Path, files: &[PathBuf]) -> CodeGraph {
        let mut graph = CodeGraph::new();
        let mut extractor = SymbolExtractor::new();

        for path in files {
            let language = detect_language_for_graph(path);
            if language.is_none() {
                continue;
            }
            let lang = language.unwrap();

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let relative_path = path.strip_prefix(root).unwrap_or(path);

            match extractor.extract_from_file(relative_path, &source, lang) {
                Ok((symbols, edges, imports)) => {
                    for symbol in symbols {
                        graph.add_symbol(symbol);
                    }
                    for edge in edges {
                        graph.add_edge(edge);
                    }
                    for import in imports {
                        graph.add_import(import);
                    }
                }
                Err(e) => {
                    warn!("path" = %path.display(), "error" = %e, "msg" = "failed to extract symbols");
                }
            }
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indicatif::ProgressBar;
    use serial_test::serial;
    use std::fs;
    use std::sync::atomic::AtomicUsize;
    use uuid::Uuid;

    #[test]
    fn progress_templates_are_consistent() {
        assert_eq!(
            PARSE_TEMPLATE,
            "{prefix} Parsing files ({pos}/{len}, {percent}%)"
        );
        assert_eq!(
            INDEX_TEMPLATE,
            "{prefix} Indexing files ({pos}/{len}, {percent}%) • {msg}"
        );
        assert_eq!(
            EMBED_TEMPLATE,
            "{prefix} Embedding chunks ({pos}/{len}, {percent}%)"
        );
    }

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
    fn warmup_executes() {
        let embedder = Arc::new(DeterministicEmbedder::default());
        let indexer = Indexer::new_concrete(embedder);
        assert!(indexer.warmup().is_ok());
    }

    #[test]
    fn progress_callback_applies_offset() {
        let pb = ProgressBar::hidden();
        pb.set_length(10);

        let callback = make_progress_callback(pb.clone(), 4, 10);

        callback(EmbedProgress {
            completed: 3,
            total: 10,
            message: None,
        });
        assert_eq!(pb.position(), 7);

        callback(EmbedProgress {
            completed: 20,
            total: 10,
            message: None,
        });
        assert_eq!(pb.position(), 10);
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
        assert!(!index.chunks.is_empty());
        assert_eq!(index.chunks.len(), index.vectors.len());

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

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn incremental_rebuilds_hierarchy_after_deletion() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = create_test_repo();
        let deleted = repo.join("test.py");

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        std::fs::remove_file(&deleted).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: Vec::new(),
                    deleted: vec![deleted.clone()],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let hierarchy = store.load_hierarchy().unwrap().unwrap();

        assert!(
            hierarchy
                .find_file_by_path(std::path::Path::new("test.py"))
                .is_none(),
            "deleted files must be removed from hierarchy after incremental reindex"
        );

        fs::remove_dir_all(&repo).ok();
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
                std::thread::sleep(Duration::from_millis(self.delay_ms));
            }
            Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }

        fn dimension(&self) -> usize {
            4
        }
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

    #[test]
    #[serial]
    fn embedding_reuse_verified_via_counters() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let text_count = Arc::new(AtomicUsize::new(0));

        #[derive(Clone)]
        struct CountingEmbedder {
            call_count: Arc<AtomicUsize>,
            text_count: Arc<AtomicUsize>,
        }

        impl BatchEmbedder for CountingEmbedder {
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                self.text_count.fetch_add(texts.len(), Ordering::SeqCst);
                Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
            }
            fn dimension(&self) -> usize {
                4
            }
        }

        let embedder = Arc::new(CountingEmbedder {
            call_count: call_count.clone(),
            text_count: text_count.clone(),
        });
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_reuse_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("main.rs"), "fn main() { println!(\"hello\"); }").unwrap();
        fs::write(
            repo.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }",
        )
        .unwrap();
        fs::write(repo.join("utils.rs"), "pub fn helper() -> bool { true }").unwrap();

        let first_report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let first_call_count = call_count.load(Ordering::SeqCst);
        let first_text_count = text_count.load(Ordering::SeqCst);

        assert!(first_call_count >= 1);
        assert!(first_text_count >= 3);
        assert!(first_report.cache_misses > 0);

        call_count.store(0, Ordering::SeqCst);
        text_count.store(0, Ordering::SeqCst);

        let second_report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let second_call_count = call_count.load(Ordering::SeqCst);
        let second_text_count = text_count.load(Ordering::SeqCst);

        assert_eq!(second_text_count, 0);
        assert!(second_report.cache_hits >= first_report.chunks_indexed);
        assert_eq!(second_report.cache_misses, 0);
        assert_eq!(second_call_count, 0);

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn version_mismatch_forces_full_reindex() {
        let call_count = Arc::new(AtomicUsize::new(0));

        #[derive(Clone)]
        struct VersionTestEmbedder {
            call_count: Arc<AtomicUsize>,
        }

        impl BatchEmbedder for VersionTestEmbedder {
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.call_count.fetch_add(texts.len(), Ordering::SeqCst);
                Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
            }
            fn dimension(&self) -> usize {
                4
            }
        }

        let embedder = Arc::new(VersionTestEmbedder {
            call_count: call_count.clone(),
        });
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_version_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("test.rs"), "fn test() { }").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let first_count = call_count.load(Ordering::SeqCst);
        assert!(first_count >= 1);

        let store = IndexStore::new(&repo).unwrap();
        let mut index = store.load().unwrap().unwrap();
        index.metadata.version = "0.0.0-fake".to_string();
        store.save(&index).unwrap();

        call_count.store(0, Ordering::SeqCst);

        fs::write(repo.join("new.rs"), "fn new() {}").unwrap();

        let report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![repo.join("new.rs")],
                    deleted: vec![],
                }),
            })
            .unwrap();

        let second_count = call_count.load(Ordering::SeqCst);

        assert!(second_count >= 2);
        assert!(report.chunks_indexed >= 2);

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn dimension_mismatch_forces_full_reindex() {
        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_dim_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("test.rs"), "fn test() { }").unwrap();

        let embedder4 = Arc::new(DeterministicEmbedder::default());
        let indexer4 = Indexer::new(embedder4);

        indexer4
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();
        assert_eq!(index.metadata.vector_dim, 4);

        #[derive(Clone)]
        struct Embedder8Dim;

        impl BatchEmbedder for Embedder8Dim {
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|t| vec![t.len() as f32; 8]).collect())
            }
            fn dimension(&self) -> usize {
                8
            }
        }

        let embedder8 = Arc::new(Embedder8Dim);
        let indexer8 = Indexer::new(embedder8);

        fs::write(repo.join("new.rs"), "fn new() {}").unwrap();

        let report = indexer8
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![repo.join("new.rs")],
                    deleted: vec![],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        assert_eq!(index.metadata.vector_dim, 8);
        assert!(index.vectors.iter().all(|v| v.len() == 8));
        assert!(report.chunks_indexed >= 2);

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn deleted_files_removed_from_index() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_delete_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        fs::write(repo.join("keep.rs"), "fn keep() { }").unwrap();
        fs::write(repo.join("delete_me.rs"), "fn delete_me() { }").unwrap();
        fs::write(repo.join("also_keep.rs"), "fn also_keep() { }").unwrap();

        let first_report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        assert_eq!(first_report.files_indexed, 3);

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();
        let has_deleted_file = index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("delete_me.rs")));
        assert!(has_deleted_file);

        fs::remove_file(repo.join("delete_me.rs")).unwrap();

        let _second_report = indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![],
                    deleted: vec![repo.join("delete_me.rs")],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        let still_has_deleted = index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("delete_me.rs")));
        assert!(!still_has_deleted);

        let has_keep = index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("keep.rs")));
        let has_also_keep = index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("also_keep.rs")));

        assert!(has_keep);
        assert!(has_also_keep);
        assert!(index.chunks.len() < first_report.chunks_indexed);

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn deleted_directory_removes_all_nested_files() {
        let embedder = Arc::new(StubEmbedder::default());
        let indexer = Indexer::new(embedder);

        let _home = set_test_home();
        let repo = std::env::temp_dir().join(format!("sgrep_dir_delete_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&repo).unwrap();

        let subdir = repo.join("subdir");
        fs::create_dir_all(&subdir).unwrap();
        fs::write(repo.join("root.rs"), "fn root() { }").unwrap();
        fs::write(subdir.join("nested1.rs"), "fn nested1() { }").unwrap();
        fs::write(subdir.join("nested2.rs"), "fn nested2() { }").unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: true,
                batch_size: None,
                profile: false,
                dirty: None,
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        let nested_count = index
            .chunks
            .iter()
            .filter(|c| {
                let path_str = c.path.to_string_lossy();
                path_str.contains("nested1") || path_str.contains("nested2")
            })
            .count();
        assert!(nested_count >= 2);

        fs::remove_dir_all(&subdir).unwrap();

        indexer
            .build_index(IndexRequest {
                path: repo.clone(),
                force: false,
                batch_size: None,
                profile: false,
                dirty: Some(DirtySet {
                    touched: vec![],
                    deleted: vec![subdir.clone()],
                }),
            })
            .unwrap();

        let store = IndexStore::new(&repo).unwrap();
        let index = store.load().unwrap().unwrap();

        let remaining_nested = index
            .chunks
            .iter()
            .filter(|c| {
                let path_str = c.path.to_string_lossy();
                path_str.contains("nested1") || path_str.contains("nested2")
            })
            .count();
        assert_eq!(remaining_nested, 0);

        let has_root = index
            .chunks
            .iter()
            .any(|c| c.path.ends_with(std::path::Path::new("root.rs")));
        assert!(has_root);

        fs::remove_dir_all(&repo).ok();
    }
}
