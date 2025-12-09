mod binary;
mod bm25_cache;
pub mod config;
mod dedup;
pub mod file_type;
mod hnsw;
mod results;
mod scoring;

pub use dedup::{suppress_near_duplicates, DedupOptions};
pub use file_type::{classify_path, FileType, FileTypePriority};
pub use results::{DirectorySearchResult, FileSearchResult, SearchResult};
pub use scoring::cosine_similarity;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::fts::{self, Bm25FDocument, Bm25FIndex};
use crate::graph::{CodeGraph, Symbol};
use crate::store::{HierarchicalIndex, MmapIndex, RepositoryIndex};

use bm25_cache::{Bm25CacheKey, Bm25FCache};

use binary::{binary_shortlist, binary_shortlist_precomputed, quantize_to_binary};
use config::{
    BINARY_QUANTIZATION_THRESHOLD, BINARY_SHORTLIST_FACTOR, HNSW_THRESHOLD, PRF_EXPANSION_TERMS,
    PRF_TOP_K,
};
use hnsw::{build_hnsw_index, search_hnsw_candidates};
use scoring::{normalize_bm25_scores, select_top_k, AdaptiveWeights as Weights};

/// Build a BM25F index from chunks with optional symbol information.
/// This implements the BM25F algorithm which boosts term frequencies for
/// important fields (filename, path, symbols) BEFORE the saturation function.
fn build_bm25f_from_chunks(chunks: &[CodeChunk], graph: Option<&CodeGraph>) -> Bm25FIndex {
    let documents: Vec<Bm25FDocument> = chunks
        .iter()
        .map(|chunk| {
            let mut doc = Bm25FDocument::new(&chunk.text, &chunk.path);

            // Extract symbol names from the graph if available
            if let Some(g) = graph {
                let symbols: Vec<String> = g
                    .symbols_in_file(&chunk.path)
                    .iter()
                    .filter(|s| s.start_line >= chunk.start_line && s.end_line <= chunk.end_line)
                    .map(|s| s.name.clone())
                    .collect();
                doc = doc.with_symbols(symbols);
            }

            doc
        })
        .collect();

    Bm25FIndex::build(&documents)
}

#[derive(Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub include_context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
    pub dedup: DedupOptions,
    pub file_type_priority: FileTypePriority,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            include_context: false,
            glob: vec![],
            filters: vec![],
            dedup: DedupOptions::default(),
            file_type_priority: FileTypePriority::default(),
        }
    }
}

pub struct SearchEngine {
    embedder: Arc<dyn BatchEmbedder>,
    graph: Option<CodeGraph>,
    bm25_cache: RefCell<Bm25FCache>,
}

#[allow(dead_code)]
impl SearchEngine {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self {
            embedder,
            graph: None,
            bm25_cache: RefCell::new(Bm25FCache::new()),
        }
    }

    pub fn set_graph(&mut self, graph: CodeGraph) {
        self.graph = Some(graph);
        // Invalidate cache since graph affects BM25F index
        self.bm25_cache.borrow_mut().clear();
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Get or build a BM25F index, using the cache when possible.
    /// This avoids rebuilding the expensive index on every search.
    fn get_or_build_bm25f(&self, chunks: &[CodeChunk], repo_hash: &str) -> Bm25FIndex {
        let key = Bm25CacheKey::new(repo_hash, chunks.len(), self.graph.is_some());

        let mut cache = self.bm25_cache.borrow_mut();
        if let Some(index) = cache.get(&key) {
            return index.clone();
        }

        // Cache miss - build the index
        let index = build_bm25f_from_chunks(chunks, self.graph.as_ref());
        cache.insert(key, index.clone());
        index
    }

    pub fn search(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let has_filters = !options.glob.is_empty() || !options.filters.is_empty();
        let num_vectors = index.vectors.len();

        if has_filters {
            return self.search_linear(index, query, options);
        }

        if num_vectors >= BINARY_QUANTIZATION_THRESHOLD {
            return self.search_binary_quantized(index, query, options);
        }

        if num_vectors >= HNSW_THRESHOLD {
            return self.search_hnsw(index, query, options);
        }

        self.search_linear(index, query, options)
    }

    pub fn search_mmap(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let has_filters = !options.glob.is_empty() || !options.filters.is_empty();
        let num_vectors = index.len();

        if has_filters {
            return self.search_mmap_linear(index, query, options);
        }

        if num_vectors >= BINARY_QUANTIZATION_THRESHOLD {
            return self.search_mmap_binary_quantized(index, query, options);
        }

        if num_vectors >= HNSW_THRESHOLD {
            return self.search_mmap_hnsw(index, query, options);
        }

        self.search_mmap_linear(index, query, options)
    }

    fn search_linear(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let globset = fts::build_globset(&options.glob);
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let candidates: Vec<(usize, &CodeChunk, &[f32], f32)> = index
            .chunks
            .iter()
            .zip(&index.vectors)
            .enumerate()
            .filter(|(_, (chunk, _))| fts::glob_matches(globset.as_ref(), &chunk.path))
            .filter(|(_, (chunk, _))| fts::matches_filters(&options.filters, chunk))
            .map(|(idx, (chunk, vector))| {
                let bm25_raw = bm25f_index.score(query, idx);
                (idx, chunk, vector.as_slice(), bm25_raw)
            })
            .collect();

        let bm25_raw_scores: Vec<f32> = candidates.iter().map(|(_, _, _, bm25)| *bm25).collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, (_, chunk, vector, bm25_raw))| {
                self.score_chunk(
                    chunk,
                    vector,
                    &query_vec,
                    query,
                    *bm25_raw,
                    bm25_normalized[i],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;

            let exp_candidates: Vec<(usize, &CodeChunk, &[f32], f32)> = index
                .chunks
                .iter()
                .zip(&index.vectors)
                .enumerate()
                .filter(|(_, (chunk, _))| fts::glob_matches(globset.as_ref(), &chunk.path))
                .filter(|(_, (chunk, _))| fts::matches_filters(&options.filters, chunk))
                .map(|(idx, (chunk, vector))| {
                    let bm25_raw = bm25f_index.score(&expanded_query, idx);
                    (idx, chunk, vector.as_slice(), bm25_raw)
                })
                .collect();

            let exp_bm25_raw: Vec<f32> =
                exp_candidates.iter().map(|(_, _, _, bm25)| *bm25).collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (i, (_, chunk, vector, bm25_raw)) in exp_candidates.iter().enumerate() {
                let result = self.score_chunk(
                    chunk,
                    vector,
                    &expanded_vec,
                    &expanded_query,
                    *bm25_raw,
                    exp_bm25_norm[i],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup(matches, index, &options);
        Ok(matches)
    }

    fn search_binary_quantized(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let query_binary = quantize_to_binary(&query_vec);
        let index_binary: Vec<Vec<u64>> = index
            .vectors
            .iter()
            .map(|v| quantize_to_binary(v))
            .collect();
        let shortlist_size =
            (PRF_TOP_K.max(fetch_limit) * BINARY_SHORTLIST_FACTOR).min(index.vectors.len());
        let candidates = binary_shortlist(&query_binary, &index_binary, shortlist_size);

        let bm25_raw_scores: Vec<f32> = candidates
            .iter()
            .map(|&idx| bm25f_index.score(query, idx))
            .collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    query,
                    bm25_raw_scores[i],
                    bm25_normalized[i],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_binary = quantize_to_binary(&expanded_vec);
            let expanded_candidates =
                binary_shortlist(&expanded_binary, &index_binary, shortlist_size);

            let exp_bm25_raw: Vec<f32> = expanded_candidates
                .iter()
                .map(|&idx| bm25f_index.score(&expanded_query, idx))
                .collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (i, idx) in expanded_candidates.iter().enumerate() {
                let result = self.score_chunk(
                    &index.chunks[*idx],
                    &index.vectors[*idx],
                    &expanded_vec,
                    &expanded_query,
                    exp_bm25_raw[i],
                    exp_bm25_norm[i],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup(matches, index, &options);
        Ok(matches)
    }

    fn search_hnsw(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let hnsw = build_hnsw_index(index.metadata.vector_dim, index.vectors.len())?;
        for (i, vector) in index.vectors.iter().enumerate() {
            hnsw.add(i as u64, vector)
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates = search_hnsw_candidates(
            &hnsw,
            &query_vec,
            PRF_TOP_K.max(fetch_limit),
            index.vectors.len(),
        )?;

        let valid_candidates: Vec<usize> = candidates
            .into_iter()
            .filter(|&idx| idx < index.chunks.len())
            .collect();

        let bm25_raw_scores: Vec<f32> = valid_candidates
            .iter()
            .map(|&idx| bm25f_index.score(query, idx))
            .collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = valid_candidates
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    query,
                    bm25_raw_scores[i],
                    bm25_normalized[i],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_candidates = search_hnsw_candidates(
                &hnsw,
                &expanded_vec,
                PRF_TOP_K.max(fetch_limit),
                index.vectors.len(),
            )?;

            let exp_valid: Vec<usize> = expanded_candidates
                .into_iter()
                .filter(|&idx| idx < index.chunks.len())
                .collect();

            let exp_bm25_raw: Vec<f32> = exp_valid
                .iter()
                .map(|&idx| bm25f_index.score(&expanded_query, idx))
                .collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (i, &idx) in exp_valid.iter().enumerate() {
                let result = self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &expanded_vec,
                    &expanded_query,
                    exp_bm25_raw[i],
                    exp_bm25_norm[i],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup(matches, index, &options);
        Ok(matches)
    }

    fn search_mmap_linear(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let globset = fts::build_globset(&options.glob);
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let candidates: Vec<(usize, f32)> = (0..index.len())
            .filter_map(|i| {
                let chunk = &index.chunks[i];
                if !fts::glob_matches(globset.as_ref(), &chunk.path) {
                    return None;
                }
                if !fts::matches_filters(&options.filters, chunk) {
                    return None;
                }
                let bm25_raw = bm25f_index.score(query, i);
                Some((i, bm25_raw))
            })
            .collect();

        let bm25_raw_scores: Vec<f32> = candidates.iter().map(|(_, bm25)| *bm25).collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .enumerate()
            .map(|(idx, (i, bm25_raw))| {
                self.score_chunk(
                    &index.chunks[*i],
                    index.get_vector(*i),
                    &query_vec,
                    query,
                    *bm25_raw,
                    bm25_normalized[idx],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;

            let exp_candidates: Vec<(usize, f32)> = (0..index.len())
                .filter_map(|i| {
                    let chunk = &index.chunks[i];
                    if !fts::glob_matches(globset.as_ref(), &chunk.path) {
                        return None;
                    }
                    if !fts::matches_filters(&options.filters, chunk) {
                        return None;
                    }
                    let bm25_raw = bm25f_index.score(&expanded_query, i);
                    Some((i, bm25_raw))
                })
                .collect();

            let exp_bm25_raw: Vec<f32> = exp_candidates.iter().map(|(_, bm25)| *bm25).collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (idx, (i, bm25_raw)) in exp_candidates.iter().enumerate() {
                let result = self.score_chunk(
                    &index.chunks[*i],
                    index.get_vector(*i),
                    &expanded_vec,
                    &expanded_query,
                    *bm25_raw,
                    exp_bm25_norm[idx],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup_mmap(matches, index, &options);
        Ok(matches)
    }

    fn search_mmap_hnsw(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let fresh_hnsw;
        let hnsw: &usearch::Index = if let Some(loaded) = index.get_hnsw() {
            loaded
        } else {
            fresh_hnsw = build_hnsw_index(index.metadata.vector_dim, index.len())?;
            for i in 0..index.len() {
                fresh_hnsw
                    .add(i as u64, index.get_vector(i))
                    .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
            }
            &fresh_hnsw
        };

        let candidates =
            search_hnsw_candidates(hnsw, &query_vec, PRF_TOP_K.max(fetch_limit), index.len())?;

        let valid_candidates: Vec<usize> = candidates
            .into_iter()
            .filter(|&idx| idx < index.len())
            .collect();

        let bm25_raw_scores: Vec<f32> = valid_candidates
            .iter()
            .map(|&idx| bm25f_index.score(query, idx))
            .collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = valid_candidates
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    query,
                    bm25_raw_scores[i],
                    bm25_normalized[i],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_candidates = search_hnsw_candidates(
                hnsw,
                &expanded_vec,
                PRF_TOP_K.max(fetch_limit),
                index.len(),
            )?;

            let exp_valid: Vec<usize> = expanded_candidates
                .into_iter()
                .filter(|&idx| idx < index.len())
                .collect();

            let exp_bm25_raw: Vec<f32> = exp_valid
                .iter()
                .map(|&idx| bm25f_index.score(&expanded_query, idx))
                .collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (i, &idx) in exp_valid.iter().enumerate() {
                let result = self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &expanded_vec,
                    &expanded_query,
                    exp_bm25_raw[i],
                    exp_bm25_norm[i],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup_mmap(matches, index, &options);
        Ok(matches)
    }

    fn search_mmap_binary_quantized(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = Weights::from_query(query);
        let (query_vec, fetch_limit) = self.prepare_search(query, &options)?;
        let shortlist_size =
            (PRF_TOP_K.max(fetch_limit) * BINARY_SHORTLIST_FACTOR).min(index.len());
        let bm25f_index = self.get_or_build_bm25f(&index.chunks, &index.metadata.repo_hash);

        let query_binary = quantize_to_binary(&query_vec);

        let (candidates, index_binary) = if index.has_binary_vectors() {
            (
                binary_shortlist_precomputed(&query_binary, index, shortlist_size),
                None,
            )
        } else {
            let idx_bin: Vec<Vec<u64>> = (0..index.len())
                .map(|i| quantize_to_binary(index.get_vector(i)))
                .collect();
            let cands = binary_shortlist(&query_binary, &idx_bin, shortlist_size);
            (cands, Some(idx_bin))
        };

        let bm25_raw_scores: Vec<f32> = candidates
            .iter()
            .map(|&idx| bm25f_index.score(query, idx))
            .collect();
        let bm25_normalized = normalize_bm25_scores(&bm25_raw_scores);

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    query,
                    bm25_raw_scores[i],
                    bm25_normalized[i],
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_binary = quantize_to_binary(&expanded_vec);

            let expanded_candidates = if index.has_binary_vectors() {
                binary_shortlist_precomputed(&expanded_binary, index, shortlist_size)
            } else if let Some(ref idx_bin) = index_binary {
                binary_shortlist(&expanded_binary, idx_bin, shortlist_size)
            } else {
                vec![]
            };

            let exp_bm25_raw: Vec<f32> = expanded_candidates
                .iter()
                .map(|&idx| bm25f_index.score(&expanded_query, idx))
                .collect();
            let exp_bm25_norm = normalize_bm25_scores(&exp_bm25_raw);

            for (i, idx) in expanded_candidates.iter().enumerate() {
                let result = self.score_chunk(
                    &index.chunks[*idx],
                    index.get_vector(*idx),
                    &expanded_vec,
                    &expanded_query,
                    exp_bm25_raw[i],
                    exp_bm25_norm[i],
                    options.include_context,
                    &weights,
                );
                Self::merge_result(&mut matches, result);
            }
        }

        Self::apply_file_type_priority(&mut matches, &options.file_type_priority);
        select_top_k(&mut matches, fetch_limit);
        let matches = self.apply_dedup_mmap(matches, index, &options);
        Ok(matches)
    }

    fn prepare_search(&self, query: &str, options: &SearchOptions) -> Result<(Vec<f32>, usize)> {
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        Ok((query_vec, limit))
    }

    fn merge_result(results: &mut Vec<SearchResult>, new_result: SearchResult) {
        if let Some(existing) = results
            .iter_mut()
            .find(|r| r.chunk.id == new_result.chunk.id)
        {
            if new_result.score > existing.score {
                *existing = new_result;
            }
        } else {
            results.push(new_result);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score_chunk(
        &self,
        chunk: &CodeChunk,
        vector: &[f32],
        query_vec: &[f32],
        _query: &str,
        bm25_raw: f32,
        bm25_normalized: f32,
        include_context: bool,
        weights: &Weights,
    ) -> SearchResult {
        let semantic = cosine_similarity(query_vec, vector);
        let score = weights.semantic * semantic + weights.bm25 * bm25_normalized;

        SearchResult {
            chunk: chunk.clone(),
            score,
            semantic_score: semantic,
            bm25_score: bm25_raw,
            show_full_context: include_context,
        }
    }

    fn apply_file_type_priority(results: &mut [SearchResult], priority: &FileTypePriority) {
        for result in results.iter_mut() {
            let multiplier = priority.multiplier(file_type::classify_path(&result.chunk.path));
            result.score *= multiplier;
        }
    }

    fn expand_query_with_prf(&self, original_query: &str, top_results: &[SearchResult]) -> String {
        if top_results.is_empty() {
            return original_query.to_string();
        }

        let original_keywords: std::collections::HashSet<String> =
            fts::extract_keywords(original_query).into_iter().collect();

        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for result in top_results.iter().take(PRF_TOP_K) {
            let tokens = fts::tokenize(&result.chunk.text);
            for token in tokens {
                if token.len() >= 3 && !original_keywords.contains(&token) {
                    *term_freq.entry(token).or_insert(0) += 1;
                }
            }
        }

        let mut terms: Vec<(String, usize)> = term_freq.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));

        let expansion_terms: Vec<String> = terms
            .into_iter()
            .take(PRF_EXPANSION_TERMS)
            .filter(|(_, count)| *count >= 2)
            .map(|(term, _)| term)
            .collect();

        if expansion_terms.is_empty() {
            return original_query.to_string();
        }

        format!("{} {}", original_query, expansion_terms.join(" "))
    }

    fn apply_dedup(
        &self,
        mut results: Vec<SearchResult>,
        index: &RepositoryIndex,
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        if !options.dedup.enabled || results.len() <= 1 {
            return results;
        }

        let chunk_id_to_idx: HashMap<uuid::Uuid, usize> = index
            .chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| (chunk.id, idx))
            .collect();

        let vectors: Vec<Vec<f32>> = results
            .iter()
            .filter_map(|r| {
                chunk_id_to_idx
                    .get(&r.chunk.id)
                    .and_then(|&idx| index.vectors.get(idx))
                    .cloned()
            })
            .collect();

        suppress_near_duplicates(&mut results, &vectors, &options.dedup);
        results
    }

    fn apply_dedup_mmap(
        &self,
        mut results: Vec<SearchResult>,
        index: &MmapIndex,
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        if !options.dedup.enabled || results.len() <= 1 {
            return results;
        }

        let chunk_id_to_idx: HashMap<uuid::Uuid, usize> = index
            .chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| (chunk.id, idx))
            .collect();

        let vectors: Vec<Vec<f32>> = results
            .iter()
            .filter_map(|r| {
                chunk_id_to_idx
                    .get(&r.chunk.id)
                    .map(|&idx| index.get_vector(idx).to_vec())
            })
            .collect();

        suppress_near_duplicates(&mut results, &vectors, &options.dedup);
        results
    }

    fn graph_results_to_search_results(
        &self,
        symbols: &[Symbol],
        index: &RepositoryIndex,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let query_vec = self.embedder.embed(query)?;

        for symbol in symbols.iter().take(options.limit) {
            for (i, chunk) in index.chunks.iter().enumerate() {
                if chunk.path == symbol.file_path
                    && chunk.start_line <= symbol.start_line
                    && chunk.end_line >= symbol.end_line
                {
                    let semantic = cosine_similarity(&query_vec, &index.vectors[i]);

                    results.push(SearchResult {
                        chunk: chunk.clone(),
                        score: 0.8 + semantic * 0.2,
                        semantic_score: semantic,
                        bm25_score: 0.0,
                        show_full_context: options.include_context,
                    });
                    break;
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    fn graph_results_to_search_results_mmap(
        &self,
        symbols: &[Symbol],
        index: &MmapIndex,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        let query_vec = self.embedder.embed(query)?;

        for symbol in symbols.iter().take(options.limit) {
            for (i, chunk) in index.chunks.iter().enumerate() {
                if chunk.path == symbol.file_path
                    && chunk.start_line <= symbol.start_line
                    && chunk.end_line >= symbol.end_line
                {
                    let semantic = cosine_similarity(&query_vec, index.get_vector(i));

                    results.push(SearchResult {
                        chunk: chunk.clone(),
                        score: 0.8 + semantic * 0.2,
                        semantic_score: semantic,
                        bm25_score: 0.0,
                        show_full_context: options.include_context,
                    });
                    break;
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    pub fn find_symbol(&self, name: &str) -> Vec<Symbol> {
        self.graph
            .as_ref()
            .map(|g| g.find_by_name(name).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn find_callers(&self, name: &str) -> Vec<Symbol> {
        self.graph
            .as_ref()
            .map(|g| {
                g.find_by_name(name)
                    .into_iter()
                    .flat_map(|s| g.find_callers(&s.id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_callees(&self, name: &str) -> Vec<Symbol> {
        self.graph
            .as_ref()
            .map(|g| {
                g.find_by_name(name)
                    .into_iter()
                    .flat_map(|s| g.find_callees(&s.id))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn graph_stats(&self) -> Option<crate::graph::GraphStats> {
        self.graph.as_ref().map(|g| g.stats())
    }

    pub fn search_files(
        &self,
        hier: &HierarchicalIndex,
        query: &str,
        limit: usize,
    ) -> Result<Vec<FileSearchResult>> {
        if hier.files.is_empty() || hier.file_vectors.is_empty() {
            return Ok(vec![]);
        }

        let query_vec = self.embedder.embed(query)?;

        let mut results: Vec<FileSearchResult> = hier
            .files
            .iter()
            .enumerate()
            .filter_map(|(idx, file_entry)| {
                let file_vec = hier.get_file_vector(idx)?;
                let score = cosine_similarity(&query_vec, file_vec);
                Some(FileSearchResult {
                    path: file_entry.path.clone(),
                    score,
                    chunk_count: file_entry.chunk_count(),
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    pub fn search_directories(
        &self,
        hier: &HierarchicalIndex,
        query: &str,
        limit: usize,
    ) -> Result<Vec<DirectorySearchResult>> {
        if hier.directories.is_empty() || hier.dir_vectors.is_empty() {
            return Ok(vec![]);
        }

        let query_vec = self.embedder.embed(query)?;

        let mut results: Vec<DirectorySearchResult> = hier
            .directories
            .iter()
            .enumerate()
            .filter_map(|(idx, dir_entry)| {
                let dir_vec = hier.get_dir_vector(idx)?;
                let score = cosine_similarity(&query_vec, dir_vec);

                let chunk_count: usize = dir_entry
                    .file_indices
                    .iter()
                    .filter_map(|&file_idx| hier.files.get(file_idx))
                    .map(|f| f.chunk_count())
                    .sum();

                Some(DirectorySearchResult {
                    path: dir_entry.path.clone(),
                    score,
                    file_count: dir_entry.file_count(),
                    chunk_count,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::dedup::DEFAULT_SEMANTIC_DEDUP_THRESHOLD;
    use crate::store::IndexMetadata;
    use chrono::Utc;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct MockEmbedder;

    impl BatchEmbedder for MockEmbedder {
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

    fn make_chunk(text: &str, language: &str, path: &str) -> CodeChunk {
        let hash = format!("{}_{}", path, text.len());
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: language.to_string(),
            start_line: 1,
            end_line: 10,
            text: text.to_string(),
            hash,
            modified_at: Utc::now(),
        }
    }

    fn make_index(chunks: Vec<CodeChunk>, vectors: Vec<Vec<f32>>) -> RepositoryIndex {
        let metadata = IndexMetadata {
            version: "0.1.0".to_string(),
            repo_path: PathBuf::from("/test"),
            repo_hash: "test123".to_string(),
            vector_dim: vectors.first().map(|v| v.len()).unwrap_or(0),
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: chunks.len(),
            embedding_model: "jina-embeddings-v2-base-code".to_string(),
        };
        RepositoryIndex::new(metadata, chunks, vectors)
    }

    #[test]
    fn search_respects_limit() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn foo() {}", "rust", "a.rs"),
            make_chunk("fn bar() {}", "rust", "b.rs"),
            make_chunk("fn baz() {}", "rust", "c.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "function",
                SearchOptions {
                    limit: 2,
                    dedup: DedupOptions {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_filters_by_glob() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn test1() {}", "rust", "src/auth.rs"),
            make_chunk("fn test2() {}", "rust", "tests/auth.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "test",
                SearchOptions {
                    glob: vec!["src/**/*.rs".to_string()],
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.path.to_string_lossy().contains("src"));
    }

    #[test]
    fn search_filters_by_language() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn test() {}", "rust", "test.rs"),
            make_chunk("def test():", "python", "test.py"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "test",
                SearchOptions {
                    filters: vec!["lang=rust".to_string()],
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.language, "rust");
    }

    #[test]
    fn select_top_k_returns_highest_scores() {
        let chunk = make_chunk("test", "rust", "test.rs");
        let mut matches = vec![
            SearchResult {
                chunk: chunk.clone(),
                score: 0.3,
                semantic_score: 0.3,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.6,
                semantic_score: 0.6,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.1,
                semantic_score: 0.1,
                bm25_score: 0.0,
                show_full_context: false,
            },
        ];

        select_top_k(&mut matches, 2);

        assert_eq!(matches.len(), 2);
        assert!((matches[0].score - 0.9).abs() < 1e-6);
        assert!((matches[1].score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn hnsw_search_produces_results_for_large_index() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks: Vec<CodeChunk> = (0..600)
            .map(|i| {
                make_chunk(
                    &format!("fn func{}() {{}}", i),
                    "rust",
                    &format!("file{}.rs", i),
                )
            })
            .collect();
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "function",
                SearchOptions {
                    dedup: DedupOptions {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn binary_quantization_search_produces_results() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks: Vec<CodeChunk> = (0..1100)
            .map(|i| {
                make_chunk(
                    &format!("fn func{}() {{}}", i),
                    "rust",
                    &format!("file{}.rs", i),
                )
            })
            .collect();
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "function",
                SearchOptions {
                    dedup: DedupOptions {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn search_files_returns_file_level_results() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let mut hier = HierarchicalIndex::new();
        hier.add_file(
            PathBuf::from("src/auth.rs"),
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 0.0],
        );
        hier.add_file(
            PathBuf::from("src/db.rs"),
            vec![2],
            vec![0.0, 1.0, 0.0, 0.0],
        );
        hier.add_file(
            PathBuf::from("src/api.rs"),
            vec![3, 4, 5],
            vec![0.5, 0.5, 0.0, 0.0],
        );

        let results = engine.search_files(&hier, "auth", 2).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        for result in &results {
            assert!(!result.path.as_os_str().is_empty());
        }
    }

    #[test]
    fn search_directories_returns_dir_level_results() {
        use crate::indexer::compute_directory_embeddings;

        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let mut hier = HierarchicalIndex::new();
        hier.add_file(
            PathBuf::from("src/auth/login.rs"),
            vec![0],
            vec![1.0, 0.0, 0.0, 0.0],
        );
        hier.add_file(
            PathBuf::from("src/auth/logout.rs"),
            vec![1],
            vec![0.8, 0.2, 0.0, 0.0],
        );
        hier.add_file(
            PathBuf::from("src/db/connection.rs"),
            vec![2],
            vec![0.0, 1.0, 0.0, 0.0],
        );

        compute_directory_embeddings(&mut hier);

        let results = engine
            .search_directories(&hier, "authentication", 2)
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());
        let index = make_index(vec![], vec![]);

        let results = engine
            .search(&index, "query", SearchOptions::default())
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn search_files_empty_hierarchy_returns_empty() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());
        let hier = HierarchicalIndex::new();

        let results = engine.search_files(&hier, "query", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_directories_empty_hierarchy_returns_empty() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());
        let hier = HierarchicalIndex::new();

        let results = engine.search_directories(&hier, "query", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn filter_by_language_matches_chunks() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn test() {}", "rust", "test.rs"),
            make_chunk("def test():", "python", "test.py"),
            make_chunk("function test() {}", "javascript", "test.js"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "test",
                SearchOptions {
                    filters: vec!["lang=python".to_string()],
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.language, "python");
    }

    #[test]
    fn multiple_glob_patterns_filter_correctly() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn test1() {}", "rust", "src/auth.rs"),
            make_chunk("fn test2() {}", "rust", "src/db.rs"),
            make_chunk("fn test3() {}", "rust", "tests/auth.rs"),
            make_chunk("fn test4() {}", "rust", "lib/utils.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "test",
                SearchOptions {
                    glob: vec!["src/**/*.rs".to_string()],
                    dedup: DedupOptions {
                        enabled: false,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.chunk.path.starts_with("src/"));
        }
    }

    #[test]
    fn find_symbol_returns_empty_without_graph() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder);

        let results = engine.find_symbol("test");
        assert!(results.is_empty());
    }

    #[test]
    fn find_callers_returns_empty_without_graph() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder);

        let results = engine.find_callers("test");
        assert!(results.is_empty());
    }

    #[test]
    fn find_callees_returns_empty_without_graph() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder);

        let results = engine.find_callees("test");
        assert!(results.is_empty());
    }

    #[test]
    fn graph_stats_returns_none_without_graph() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder);

        assert!(engine.graph_stats().is_none());
    }

    #[test]
    fn find_symbol_with_graph() {
        use crate::graph::{CodeGraph, Symbol, SymbolKind};

        let embedder = Arc::new(MockEmbedder);
        let mut engine = SearchEngine::new(embedder);

        let mut graph = CodeGraph::new();
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "authenticate".to_string(),
            qualified_name: "auth::authenticate".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/auth.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn authenticate()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        engine.set_graph(graph);

        let results = engine.find_symbol("authenticate");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "authenticate");
    }

    #[test]
    fn find_callers_with_graph() {
        use crate::graph::{CodeGraph, Edge, EdgeKind, Symbol, SymbolKind};

        let embedder = Arc::new(MockEmbedder);
        let mut engine = SearchEngine::new(embedder);

        let mut graph = CodeGraph::new();
        let callee_id = Uuid::new_v4();
        let caller_id = Uuid::new_v4();

        graph.add_symbol(Symbol {
            id: callee_id,
            name: "authenticate".to_string(),
            qualified_name: "auth::authenticate".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/auth.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn authenticate()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_symbol(Symbol {
            id: caller_id,
            name: "login".to_string(),
            qualified_name: "login::login".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/login.rs"),
            start_line: 1,
            end_line: 20,
            language: "rust".to_string(),
            signature: "fn login()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_edge(Edge {
            source_id: caller_id,
            target_id: callee_id,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        engine.set_graph(graph);

        let callers = engine.find_callers("authenticate");
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].name, "login");
    }

    #[test]
    fn find_callees_with_graph() {
        use crate::graph::{CodeGraph, Edge, EdgeKind, Symbol, SymbolKind};

        let embedder = Arc::new(MockEmbedder);
        let mut engine = SearchEngine::new(embedder);

        let mut graph = CodeGraph::new();
        let caller_id = Uuid::new_v4();
        let callee_id = Uuid::new_v4();

        graph.add_symbol(Symbol {
            id: caller_id,
            name: "login".to_string(),
            qualified_name: "login::login".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/login.rs"),
            start_line: 1,
            end_line: 20,
            language: "rust".to_string(),
            signature: "fn login()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_symbol(Symbol {
            id: callee_id,
            name: "authenticate".to_string(),
            qualified_name: "auth::authenticate".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/auth.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn authenticate()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_edge(Edge {
            source_id: caller_id,
            target_id: callee_id,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        engine.set_graph(graph);

        let callees = engine.find_callees("login");
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].name, "authenticate");
    }

    #[test]
    fn graph_stats_with_graph() {
        use crate::graph::{CodeGraph, Symbol, SymbolKind};

        let embedder = Arc::new(MockEmbedder);
        let mut engine = SearchEngine::new(embedder);

        let mut graph = CodeGraph::new();
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            qualified_name: "test::test".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/test.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn test()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        engine.set_graph(graph);

        let stats = engine.graph_stats();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_symbols, 1);
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let similarity = cosine_similarity(&v1, &v1);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&v1, &v2);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![-1.0, 0.0, 0.0, 0.0];
        let similarity = cosine_similarity(&v1, &v2);
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_with_include_context() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![make_chunk("fn test() {}", "rust", "test.rs")];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "test",
                SearchOptions {
                    include_context: true,
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].show_full_context);
    }

    #[test]
    fn select_top_k_handles_less_than_k() {
        let mut matches: Vec<SearchResult> = vec![];
        select_top_k(&mut matches, 5);
        assert!(matches.is_empty());
    }

    // Test for search consolidation (Phase 5) - verify SearchContext reduces duplication
    mod search_consolidation_tests {
        use super::*;

        #[test]
        fn search_context_calculates_fetch_limit() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());
            let options = SearchOptions {
                limit: 5,
                ..Default::default()
            };
            let (_, fetch_limit) = engine.prepare_search("test", &options).unwrap();
            assert_eq!(fetch_limit, 5);
        }

        #[test]
        fn merge_results_keeps_higher_scores() {
            let chunk = make_chunk("test", "rust", "test.rs");
            let mut existing = vec![SearchResult {
                chunk: chunk.clone(),
                score: 0.5,
                semantic_score: 0.5,
                bm25_score: 0.0,
                show_full_context: false,
            }];
            let new_result = SearchResult {
                chunk: chunk.clone(),
                score: 0.8,
                semantic_score: 0.8,
                bm25_score: 0.0,
                show_full_context: false,
            };

            SearchEngine::merge_result(&mut existing, new_result);

            assert_eq!(existing.len(), 1);
            assert!((existing[0].score - 0.8).abs() < 1e-6);
        }

        #[test]
        fn merge_results_adds_new_chunks() {
            let chunk1 = make_chunk("test1", "rust", "test1.rs");
            let chunk2 = make_chunk("test2", "rust", "test2.rs");
            let mut existing = vec![SearchResult {
                chunk: chunk1,
                score: 0.5,
                semantic_score: 0.5,
                bm25_score: 0.0,
                show_full_context: false,
            }];
            let new_result = SearchResult {
                chunk: chunk2,
                score: 0.6,
                semantic_score: 0.6,
                bm25_score: 0.0,
                show_full_context: false,
            };

            SearchEngine::merge_result(&mut existing, new_result);

            assert_eq!(existing.len(), 2);
        }
    }

    // Test to verify embedding is hoisted outside loops (Phase 2.1 performance fix)
    mod embedding_call_count_tests {
        use super::*;
        use crate::graph::{Symbol, SymbolKind};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CountingEmbedder {
            call_count: AtomicUsize,
        }

        impl CountingEmbedder {
            fn new() -> Self {
                Self {
                    call_count: AtomicUsize::new(0),
                }
            }

            fn get_call_count(&self) -> usize {
                self.call_count.load(Ordering::SeqCst)
            }
        }

        impl BatchEmbedder for CountingEmbedder {
            fn embed(&self, _text: &str) -> Result<Vec<f32>> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(vec![1.0, 0.5, 0.0, -0.5])
            }

            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.call_count.fetch_add(texts.len(), Ordering::SeqCst);
                Ok(texts.iter().map(|_| vec![1.0, 0.5, 0.0, -0.5]).collect())
            }

            fn dimension(&self) -> usize {
                4
            }
        }

        fn make_symbol(name: &str, path: &str, start: usize, end: usize) -> Symbol {
            Symbol {
                id: Uuid::new_v4(),
                name: name.to_string(),
                qualified_name: name.to_string(),
                kind: SymbolKind::Function,
                file_path: PathBuf::from(path),
                start_line: start,
                end_line: end,
                language: "rust".to_string(),
                signature: format!("fn {}()", name),
                parent_id: None,
                chunk_id: None,
            }
        }

        #[test]
        fn graph_results_to_search_results_embeds_query_once() {
            let embedder = Arc::new(CountingEmbedder::new());
            let engine = SearchEngine::new(embedder.clone());

            // Create multiple chunks and symbols
            let chunks = vec![
                make_chunk("fn foo() {}", "rust", "src/foo.rs"),
                make_chunk("fn bar() {}", "rust", "src/bar.rs"),
                make_chunk("fn baz() {}", "rust", "src/baz.rs"),
            ];
            let vectors: Vec<Vec<f32>> = chunks.iter().map(|_| vec![1.0, 0.5, 0.0, -0.5]).collect();
            let index = make_index(chunks, vectors);

            // Create symbols that match the chunks
            let symbols = vec![
                make_symbol("foo", "src/foo.rs", 1, 10),
                make_symbol("bar", "src/bar.rs", 1, 10),
                make_symbol("baz", "src/baz.rs", 1, 10),
            ];

            let options = SearchOptions {
                limit: 10,
                ..Default::default()
            };

            let initial_count = embedder.get_call_count();
            let _ =
                engine.graph_results_to_search_results(&symbols, &index, "test query", &options);
            let calls_made = embedder.get_call_count() - initial_count;
            assert_eq!(
                calls_made, 1,
                "Expected embed() to be called once, but was called {} times",
                calls_made
            );
        }
    }

    // Tests for BM25F caching (GitHub issue #19)
    mod bm25_cache_integration_tests {
        use super::*;

        #[test]
        fn cache_hit_on_multiple_searches_same_index() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());

            let chunks = vec![
                make_chunk("fn foo() {}", "rust", "a.rs"),
                make_chunk("fn bar() {}", "rust", "b.rs"),
            ];
            let vectors: Vec<Vec<f32>> = chunks
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();
            let index = make_index(chunks, vectors);

            // First search - cache miss
            let _ = engine
                .search(&index, "function", SearchOptions::default())
                .unwrap();

            // Check cache has entry
            assert_eq!(engine.bm25_cache.borrow().entry_count(), 1);
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 1);

            // Second search - should be cache hit
            let _ = engine
                .search(&index, "different query", SearchOptions::default())
                .unwrap();

            assert_eq!(engine.bm25_cache.borrow().hit_count(), 1);
        }

        #[test]
        fn cache_invalidated_on_set_graph() {
            use crate::graph::{CodeGraph, Symbol, SymbolKind};

            let embedder = Arc::new(MockEmbedder);
            let mut engine = SearchEngine::new(embedder.clone());

            let chunks = vec![make_chunk("fn test() {}", "rust", "test.rs")];
            let vectors: Vec<Vec<f32>> = chunks
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();
            let index = make_index(chunks, vectors);

            // First search - cache miss, stores entry
            let _ = engine
                .search(&index, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().entry_count(), 1);

            // Set graph - should invalidate cache
            let mut graph = CodeGraph::new();
            graph.add_symbol(Symbol {
                id: Uuid::new_v4(),
                name: "test".to_string(),
                qualified_name: "test::test".to_string(),
                kind: SymbolKind::Function,
                file_path: PathBuf::from("test.rs"),
                start_line: 1,
                end_line: 10,
                language: "rust".to_string(),
                signature: "fn test()".to_string(),
                parent_id: None,
                chunk_id: None,
            });
            engine.set_graph(graph);

            // Cache should be cleared
            assert_eq!(engine.bm25_cache.borrow().entry_count(), 0);

            // Next search should be cache miss
            let _ = engine
                .search(&index, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 2);
        }

        #[test]
        fn cache_miss_on_different_repo() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());

            let chunks1 = vec![make_chunk("fn foo() {}", "rust", "a.rs")];
            let vectors1: Vec<Vec<f32>> = chunks1
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();

            let mut index1 = make_index(chunks1, vectors1.clone());
            index1.metadata.repo_hash = "repo1".to_string();

            let chunks2 = vec![make_chunk("fn bar() {}", "rust", "b.rs")];
            let vectors2: Vec<Vec<f32>> = chunks2
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();

            let mut index2 = make_index(chunks2, vectors2);
            index2.metadata.repo_hash = "repo2".to_string();

            // Search on first index
            let _ = engine
                .search(&index1, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 1);

            // Search on second index - different repo, should be cache miss
            let _ = engine
                .search(&index2, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 2);

            // Cache should still have 1 entry (replaced)
            assert_eq!(engine.bm25_cache.borrow().entry_count(), 1);
        }

        #[test]
        fn cache_miss_on_different_chunk_count() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());

            let chunks1 = vec![make_chunk("fn foo() {}", "rust", "a.rs")];
            let vectors1: Vec<Vec<f32>> = chunks1
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();
            let index1 = make_index(chunks1, vectors1);

            let chunks2 = vec![
                make_chunk("fn foo() {}", "rust", "a.rs"),
                make_chunk("fn bar() {}", "rust", "b.rs"),
            ];
            let vectors2: Vec<Vec<f32>> = chunks2
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();

            let mut index2 = make_index(chunks2, vectors2);
            // Same repo_hash but different chunk count
            index2.metadata.repo_hash = index1.metadata.repo_hash.clone();

            // Search on first index
            let _ = engine
                .search(&index1, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 1);

            // Search on second index - different chunk count, should be cache miss
            let _ = engine
                .search(&index2, "test", SearchOptions::default())
                .unwrap();
            assert_eq!(engine.bm25_cache.borrow().miss_count(), 2);
        }
    }

    mod near_duplicate_suppression_tests {
        use super::*;

        fn make_chunk_with_hash(text: &str, path: &str, hash: &str) -> CodeChunk {
            CodeChunk {
                id: Uuid::new_v4(),
                path: PathBuf::from(path),
                language: "rust".to_string(),
                start_line: 1,
                end_line: 10,
                text: text.to_string(),
                hash: hash.to_string(),
                modified_at: Utc::now(),
            }
        }

        #[allow(dead_code)]
        fn make_result_with_score(
            chunk: CodeChunk,
            score: f32,
            vector: Vec<f32>,
        ) -> (SearchResult, Vec<f32>) {
            (
                SearchResult {
                    chunk,
                    score,
                    semantic_score: score,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                vector,
            )
        }

        #[test]
        fn suppress_identical_hash_chunks() {
            let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
            let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "hash123");
            let chunk3 = make_chunk_with_hash("fn bar() {}", "c.rs", "hash456");

            let mut results = vec![
                SearchResult {
                    chunk: chunk1,
                    score: 0.9,
                    semantic_score: 0.9,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk2,
                    score: 0.8,
                    semantic_score: 0.8,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk3,
                    score: 0.7,
                    semantic_score: 0.7,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
            ];

            suppress_near_duplicates(&mut results, &[], &DedupOptions::default());

            assert_eq!(results.len(), 2);
            assert!((results[0].score - 0.9).abs() < 1e-6);
            assert_eq!(results[0].chunk.hash, "hash123");
            assert_eq!(results[1].chunk.hash, "hash456");
        }

        #[test]
        fn suppress_semantically_similar_chunks() {
            let chunk1 =
                make_chunk_with_hash("fn authenticate() { check_password(); }", "auth.rs", "h1");
            let chunk2 = make_chunk_with_hash(
                "fn authenticate() { check_password(); verify(); }",
                "auth2.rs",
                "h2",
            );
            let chunk3 = make_chunk_with_hash("fn connect_to_database() {}", "db.rs", "h3");

            let vectors = vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.99, 0.14, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ];

            let mut results = vec![
                SearchResult {
                    chunk: chunk1,
                    score: 0.9,
                    semantic_score: 0.9,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk2,
                    score: 0.85,
                    semantic_score: 0.85,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk3,
                    score: 0.7,
                    semantic_score: 0.7,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
            ];

            let options = DedupOptions {
                semantic_threshold: 0.95,
                ..Default::default()
            };

            suppress_near_duplicates(&mut results, &vectors, &options);

            assert_eq!(results.len(), 2);
            assert!(results[0].chunk.path.to_string_lossy().contains("auth.rs"));
            assert!(results[1].chunk.path.to_string_lossy().contains("db.rs"));
        }

        #[test]
        fn preserve_similar_but_distinct_chunks() {
            let chunk1 = make_chunk_with_hash("fn process_user() {}", "a.rs", "h1");
            let chunk2 = make_chunk_with_hash("fn process_order() {}", "b.rs", "h2");

            let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.8, 0.6, 0.0, 0.0]];

            let mut results = vec![
                SearchResult {
                    chunk: chunk1,
                    score: 0.9,
                    semantic_score: 0.9,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk2,
                    score: 0.85,
                    semantic_score: 0.85,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
            ];

            let options = DedupOptions {
                semantic_threshold: 0.95,
                ..Default::default()
            };

            suppress_near_duplicates(&mut results, &vectors, &options);

            assert_eq!(results.len(), 2);
        }

        #[test]
        fn ordering_is_reproducible() {
            let chunk1 = make_chunk_with_hash("fn a() {}", "a.rs", "h1");
            let chunk2 = make_chunk_with_hash("fn b() {}", "b.rs", "h2");
            let chunk3 = make_chunk_with_hash("fn c() {}", "c.rs", "h3");

            let vectors = vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.5, 0.5, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ];

            let create_results = || {
                vec![
                    SearchResult {
                        chunk: chunk1.clone(),
                        score: 0.9,
                        semantic_score: 0.9,
                        bm25_score: 0.0,
                        show_full_context: false,
                    },
                    SearchResult {
                        chunk: chunk2.clone(),
                        score: 0.85,
                        semantic_score: 0.85,
                        bm25_score: 0.0,
                        show_full_context: false,
                    },
                    SearchResult {
                        chunk: chunk3.clone(),
                        score: 0.8,
                        semantic_score: 0.8,
                        bm25_score: 0.0,
                        show_full_context: false,
                    },
                ]
            };

            let mut results1 = create_results();
            let mut results2 = create_results();

            suppress_near_duplicates(&mut results1, &vectors, &DedupOptions::default());
            suppress_near_duplicates(&mut results2, &vectors, &DedupOptions::default());

            assert_eq!(results1.len(), results2.len());
            for (r1, r2) in results1.iter().zip(results2.iter()) {
                assert_eq!(r1.chunk.id, r2.chunk.id);
                assert!((r1.score - r2.score).abs() < 1e-6);
            }
        }

        #[test]
        fn dedup_options_default_values() {
            let options = DedupOptions::default();
            assert!((options.semantic_threshold - DEFAULT_SEMANTIC_DEDUP_THRESHOLD).abs() < 1e-6);
            assert!(options.enabled);
        }

        #[test]
        fn suppression_disabled_when_not_enabled() {
            let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
            let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "hash123"); // Same hash

            let mut results = vec![
                SearchResult {
                    chunk: chunk1,
                    score: 0.9,
                    semantic_score: 0.9,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk2,
                    score: 0.8,
                    semantic_score: 0.8,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
            ];

            let options = DedupOptions {
                enabled: false,
                ..Default::default()
            };

            suppress_near_duplicates(&mut results, &[], &options);

            assert_eq!(results.len(), 2);
        }

        #[test]
        fn empty_results_handled_gracefully() {
            let mut results: Vec<SearchResult> = vec![];
            suppress_near_duplicates(&mut results, &[], &DedupOptions::default());
            assert!(results.is_empty());
        }

        #[test]
        fn single_result_unchanged() {
            let chunk = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
            let mut results = vec![SearchResult {
                chunk,
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            }];

            let original_len = results.len();
            suppress_near_duplicates(
                &mut results,
                &[vec![1.0, 0.0, 0.0, 0.0]],
                &DedupOptions::default(),
            );
            assert_eq!(results.len(), original_len);
        }

        #[test]
        fn search_options_includes_dedup_setting() {
            let options = SearchOptions::default();
            assert!(options.dedup.enabled, "Dedup should be enabled by default");
        }

        #[test]
        fn search_applies_dedup_by_default() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());

            let chunks = vec![
                make_chunk_with_hash("fn duplicate() { same_code(); }", "a.rs", "same_hash"),
                make_chunk_with_hash("fn duplicate() { same_code(); }", "b.rs", "same_hash"),
                make_chunk_with_hash("fn unique() { different_code(); }", "c.rs", "diff_hash"),
            ];
            let vectors: Vec<Vec<f32>> = chunks
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();

            let results = engine
                .search(
                    &make_index(chunks, vectors),
                    "duplicate",
                    SearchOptions::default(),
                )
                .unwrap();

            let same_hash_count = results
                .iter()
                .filter(|r| r.chunk.hash == "same_hash")
                .count();
            assert_eq!(same_hash_count, 1);
        }

        #[test]
        fn dedup_can_be_disabled_in_search_options() {
            let embedder = Arc::new(MockEmbedder);
            let engine = SearchEngine::new(embedder.clone());

            let chunks = vec![
                make_chunk_with_hash("fn duplicate() { same_code(); }", "a.rs", "same_hash"),
                make_chunk_with_hash("fn duplicate() { same_code(); }", "b.rs", "same_hash"),
            ];
            let vectors: Vec<Vec<f32>> = chunks
                .iter()
                .map(|c| embedder.embed(&c.text).unwrap())
                .collect();

            let options = SearchOptions {
                dedup: DedupOptions {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            };

            let results = engine
                .search(&make_index(chunks, vectors), "duplicate", options)
                .unwrap();

            assert_eq!(results.len(), 2);
        }
    }
}
