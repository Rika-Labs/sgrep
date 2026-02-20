use super::dedup::{suppress_near_duplicates, DedupOptions};
use super::file_type::{self, FileTypePriority};
use super::results::{DirectorySearchResult, FileSearchResult, SearchResult};
use super::scoring::{
    cosine_similarity, normalize_bm25_scores, select_top_k, AdaptiveWeights as Weights,
};

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::fts::{self, Bm25FDocument, Bm25FIndex};
use crate::graph::{CodeGraph, Symbol};
use crate::store::{HierarchicalIndex, MmapIndex, RepositoryIndex};

use super::bm25_cache::{Bm25CacheKey, Bm25FCache};

use super::binary::{binary_shortlist, binary_shortlist_precomputed, quantize_to_binary};
use super::config::{
    BINARY_QUANTIZATION_THRESHOLD, BINARY_SHORTLIST_FACTOR, HNSW_THRESHOLD, PRF_EXPANSION_TERMS,
    PRF_TOP_K,
};
use super::hnsw::{build_hnsw_index, search_hnsw_candidates};

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
                    .file_symbols
                    .get(&chunk.path)
                    .into_iter()
                    .flat_map(|ids| ids.iter())
                    .filter_map(|id| g.symbols.get(id))
                    .filter(|s| s.start_line >= chunk.start_line && s.end_line <= chunk.end_line)
                    .map(|s| s.name.clone())
                    .collect();
                if !symbols.is_empty() {
                    doc = doc.with_symbols(symbols);
                }
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
            file_type_priority: FileTypePriority,
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

    pub(crate) fn prepare_search(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<(Vec<f32>, usize)> {
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        Ok((query_vec, limit))
    }

    pub(crate) fn merge_result(results: &mut Vec<SearchResult>, new_result: SearchResult) {
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

    pub(crate) fn apply_dedup_mmap(
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

    pub(crate) fn graph_results_to_search_results(
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

    pub(crate) fn graph_results_to_search_results_mmap(
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

    pub fn bm25_cache_counts(&self) -> (usize, usize, usize) {
        let cache = self.bm25_cache.borrow();
        (cache.entry_count(), cache.hit_count(), cache.miss_count())
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
