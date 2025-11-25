use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use simsimd::SpatialSimilarity;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::fts;
use crate::graph::{classify_query, CodeGraph, QueryType, Symbol};
use crate::reranker::Reranker;
use crate::store::{MmapIndex, RepositoryIndex};

const HNSW_THRESHOLD: usize = 500;
const BINARY_QUANTIZATION_THRESHOLD: usize = 1000;
const BINARY_SHORTLIST_FACTOR: usize = 10;
const HNSW_CONNECTIVITY: usize = 16;
const HNSW_EXPANSION_ADD: usize = 128;
const HNSW_EXPANSION_SEARCH: usize = 64;
const HNSW_OVERSAMPLE_FACTOR: usize = 4;
const RERANK_OVERSAMPLE_FACTOR: usize = 3;
const PRF_TOP_K: usize = 10;
const PRF_EXPANSION_TERMS: usize = 5;
const RECENCY_HALF_LIFE_HOURS: f32 = 48.0;

#[derive(Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub include_context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
    pub rerank: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            include_context: false,
            glob: vec![],
            filters: vec![],
            rerank: false,
        }
    }
}

#[derive(Clone, Copy)]
struct AdaptiveWeights {
    semantic: f32,
    bm25: f32,
    keyword: f32,
    recency: f32,
    file_type: f32,
}

impl AdaptiveWeights {
    fn from_query(query: &str) -> Self {
        let word_count = query.split_whitespace().count();
        let query_lower = query.to_lowercase();

        let is_short = word_count <= 2;
        let is_question = query_lower.starts_with("how ")
            || query_lower.starts_with("where ")
            || query_lower.starts_with("what ")
            || query_lower.starts_with("why ")
            || query_lower.starts_with("when ")
            || query_lower.starts_with("which ");
        let has_code_symbols = query.chars().any(|c| "(){}[]<>::->=>".contains(c));

        let mut semantic = 0.45;
        let mut bm25 = 0.20;
        let mut keyword = 0.15;
        let recency = 0.05;
        let mut file_type = 0.15;

        if is_question {
            semantic += 0.10;
            bm25 -= 0.05;
            file_type -= 0.05;
        }

        if is_short {
            bm25 += 0.10;
            semantic -= 0.05;
            file_type -= 0.05;
        }

        if has_code_symbols {
            keyword += 0.10;
            semantic -= 0.05;
            file_type -= 0.05;
        }

        let total = semantic + bm25 + keyword + recency + file_type;
        Self {
            semantic: semantic / total,
            bm25: bm25 / total,
            keyword: keyword / total,
            recency: recency / total,
            file_type: file_type / total,
        }
    }
}

pub struct SearchEngine {
    embedder: Arc<dyn BatchEmbedder>,
    reranker: Option<Arc<dyn Reranker>>,
    graph: Option<CodeGraph>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self {
            embedder,
            reranker: None,
            graph: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_reranker(embedder: Arc<dyn BatchEmbedder>, reranker: Arc<dyn Reranker>) -> Self {
        Self {
            embedder,
            reranker: Some(reranker),
            graph: None,
        }
    }

    /// Set the code graph for hybrid search
    pub fn set_graph(&mut self, graph: CodeGraph) {
        self.graph = Some(graph);
    }

    /// Check if graph is available
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
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
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let globset = fts::build_globset(&options.glob);

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

        let mut matches: Vec<SearchResult> = index
            .chunks
            .iter()
            .zip(&index.vectors)
            .enumerate()
            .filter(|(_, (chunk, _))| fts::glob_matches(globset.as_ref(), &chunk.path))
            .filter(|(_, (chunk, _))| fts::matches_filters(&options.filters, chunk))
            .map(|(idx, (chunk, vector))| {
                let bm25_score = bm25_index.score(query, idx);
                self.score_chunk(
                    chunk,
                    vector,
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;

            for (idx, (chunk, vector)) in index.chunks.iter().zip(&index.vectors).enumerate() {
                if !fts::glob_matches(globset.as_ref(), &chunk.path) {
                    continue;
                }
                if !fts::matches_filters(&options.filters, chunk) {
                    continue;
                }
                let bm25_score = bm25_index.score(&expanded_query, idx);
                let result = self.score_chunk(
                    chunk,
                    vector,
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn search_binary_quantized(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

        let query_binary = quantize_to_binary(&query_vec);
        let index_binary: Vec<Vec<u64>> = index
            .vectors
            .iter()
            .map(|v| quantize_to_binary(v))
            .collect();
        let shortlist_size =
            (PRF_TOP_K.max(fetch_limit) * BINARY_SHORTLIST_FACTOR).min(index.vectors.len());
        let candidates = binary_shortlist(&query_binary, &index_binary, shortlist_size);

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .map(|&idx| {
                let bm25_score = bm25_index.score(query, idx);
                self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_binary = quantize_to_binary(&expanded_vec);
            let expanded_candidates =
                binary_shortlist(&expanded_binary, &index_binary, shortlist_size);

            for idx in expanded_candidates {
                let bm25_score = bm25_index.score(&expanded_query, idx);
                let result = self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn search_hnsw(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

        let hnsw = self.build_hnsw_index(index.metadata.vector_dim, index.vectors.len())?;
        for (i, vector) in index.vectors.iter().enumerate() {
            hnsw.add(i as u64, vector)
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates = self.search_hnsw_candidates(
            &hnsw,
            &query_vec,
            PRF_TOP_K.max(fetch_limit),
            index.vectors.len(),
        )?;

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .filter_map(|&idx| {
                if idx >= index.chunks.len() {
                    return None;
                }
                let bm25_score = bm25_index.score(query, idx);
                Some(self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                ))
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_candidates = self.search_hnsw_candidates(
                &hnsw,
                &expanded_vec,
                PRF_TOP_K.max(fetch_limit),
                index.vectors.len(),
            )?;

            for idx in expanded_candidates {
                if idx >= index.chunks.len() {
                    continue;
                }
                let bm25_score = bm25_index.score(&expanded_query, idx);
                let result = self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn search_mmap_linear(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let globset = fts::build_globset(&options.glob);

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

        let mut matches: Vec<SearchResult> = (0..index.len())
            .filter_map(|i| {
                let chunk = &index.chunks[i];
                if !fts::glob_matches(globset.as_ref(), &chunk.path) {
                    return None;
                }
                if !fts::matches_filters(&options.filters, chunk) {
                    return None;
                }
                let vector = index.get_vector(i);
                let bm25_score = bm25_index.score(query, i);
                Some(self.score_chunk(
                    chunk,
                    vector,
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                ))
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;

            for i in 0..index.len() {
                let chunk = &index.chunks[i];
                if !fts::glob_matches(globset.as_ref(), &chunk.path) {
                    continue;
                }
                if !fts::matches_filters(&options.filters, chunk) {
                    continue;
                }
                let vector = index.get_vector(i);
                let bm25_score = bm25_index.score(&expanded_query, i);
                let result = self.score_chunk(
                    chunk,
                    vector,
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn search_mmap_hnsw(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

        let hnsw = self.build_hnsw_index(index.metadata.vector_dim, index.len())?;
        for i in 0..index.len() {
            hnsw.add(i as u64, index.get_vector(i))
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates = self.search_hnsw_candidates(
            &hnsw,
            &query_vec,
            PRF_TOP_K.max(fetch_limit),
            index.len(),
        )?;

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .filter_map(|&idx| {
                if idx >= index.len() {
                    return None;
                }
                let bm25_score = bm25_index.score(query, idx);
                Some(self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                ))
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_candidates = self.search_hnsw_candidates(
                &hnsw,
                &expanded_vec,
                PRF_TOP_K.max(fetch_limit),
                index.len(),
            )?;

            for idx in expanded_candidates {
                if idx >= index.len() {
                    continue;
                }
                let bm25_score = bm25_index.score(&expanded_query, idx);
                let result = self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn search_mmap_binary_quantized(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let weights = AdaptiveWeights::from_query(query);
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let shortlist_size =
            (PRF_TOP_K.max(fetch_limit) * BINARY_SHORTLIST_FACTOR).min(index.len());

        let doc_texts: Vec<&str> = index.chunks.iter().map(|c| c.text.as_str()).collect();
        let bm25_index = fts::Bm25Index::build(&doc_texts);

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

        let mut matches: Vec<SearchResult> = candidates
            .iter()
            .map(|&idx| {
                let bm25_score = bm25_index.score(query, idx);
                self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    &keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                )
            })
            .collect();

        select_top_k(&mut matches, PRF_TOP_K.max(fetch_limit));

        let expanded_query = self.expand_query_with_prf(query, &matches);
        if expanded_query != query {
            let expanded_keywords = fts::extract_keywords(&expanded_query);
            let expanded_vec = self.embedder.embed(&expanded_query)?;
            let expanded_binary = quantize_to_binary(&expanded_vec);

            let expanded_candidates = if index.has_binary_vectors() {
                binary_shortlist_precomputed(&expanded_binary, index, shortlist_size)
            } else if let Some(ref idx_bin) = index_binary {
                binary_shortlist(&expanded_binary, idx_bin, shortlist_size)
            } else {
                vec![]
            };

            for idx in expanded_candidates {
                let bm25_score = bm25_index.score(&expanded_query, idx);
                let result = self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &expanded_vec,
                    &expanded_keywords,
                    bm25_score,
                    options.include_context,
                    &weights,
                );
                if let Some(existing) = matches.iter_mut().find(|m| m.chunk.id == result.chunk.id) {
                    if result.score > existing.score {
                        *existing = result;
                    }
                } else {
                    matches.push(result);
                }
            }
        }

        select_top_k(&mut matches, fetch_limit);
        let matches = self.maybe_rerank(query, matches, &options);
        Ok(matches)
    }

    fn build_hnsw_index(&self, dimensions: usize, capacity: usize) -> Result<Index> {
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: HNSW_CONNECTIVITY,
            expansion_add: HNSW_EXPANSION_ADD,
            expansion_search: HNSW_EXPANSION_SEARCH,
            multi: false,
        };

        let hnsw =
            Index::new(&options).map_err(|e| anyhow::anyhow!("HNSW creation failed: {}", e))?;
        hnsw.reserve(capacity)
            .map_err(|e| anyhow::anyhow!("HNSW reserve failed: {}", e))?;
        Ok(hnsw)
    }

    fn search_hnsw_candidates(
        &self,
        hnsw: &Index,
        query_vec: &[f32],
        limit: usize,
        max_candidates: usize,
    ) -> Result<Vec<usize>> {
        let oversample = (limit * HNSW_OVERSAMPLE_FACTOR).min(max_candidates);
        let results = hnsw
            .search(query_vec, oversample)
            .map_err(|e| anyhow::anyhow!("HNSW search failed: {}", e))?;
        Ok(results.keys.iter().map(|&k| k as usize).collect())
    }

    fn score_chunk(
        &self,
        chunk: &CodeChunk,
        vector: &[f32],
        query_vec: &[f32],
        keywords: &[String],
        bm25_score: f32,
        include_context: bool,
        weights: &AdaptiveWeights,
    ) -> SearchResult {
        let semantic = cosine_similarity(query_vec, vector);
        let keyword = fts::keyword_score(keywords, &chunk.text, &chunk.path);
        let recency = recency_boost(chunk);
        let file_type = content_based_file_boost(chunk);

        let bm25_normalized = (bm25_score / 10.0).min(1.0);

        let score = weights.semantic * semantic
            + weights.bm25 * bm25_normalized
            + weights.keyword * keyword
            + weights.recency * recency
            + weights.file_type * file_type;

        SearchResult {
            chunk: chunk.clone(),
            score,
            semantic_score: semantic,
            bm25_score,
            keyword_score: keyword,
            show_full_context: include_context,
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

    /// Apply reranking to results if enabled and reranker is available
    fn maybe_rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        // Skip reranking if not enabled or no reranker available
        if !options.rerank {
            return results;
        }

        let reranker = match &self.reranker {
            Some(r) => r,
            None => return results,
        };

        // Skip if too few results
        if results.len() <= 1 {
            return results;
        }

        // Extract document texts for reranking
        let docs: Vec<&str> = results.iter().map(|r| r.chunk.text.as_str()).collect();

        // Rerank and reorder results
        match reranker.rerank(query, &docs) {
            Ok(reranked) => {
                let mut reordered: Vec<SearchResult> = reranked
                    .into_iter()
                    .filter_map(|(idx, rerank_score)| {
                        if idx < results.len() {
                            let mut result = results[idx].clone();
                            // Blend rerank score with original score (rerank has higher weight)
                            result.score = 0.3 * result.score + 0.7 * rerank_score;
                            Some(result)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Ensure results are sorted by the new blended score
                reordered.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Truncate to original limit
                reordered.truncate(options.limit);
                reordered
            }
            Err(_) => results, // Fall back to original results on error
        }
    }

    /// Hybrid search combining graph and vector approaches
    pub fn search_hybrid(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_type = classify_query(query);

        match query_type {
            QueryType::Structural => {
                // Try graph-first for structural queries
                if let Some(ref graph) = self.graph {
                    let graph_results = self.search_graph_only(graph, query, &options);
                    if !graph_results.is_empty() {
                        // Convert graph results to SearchResults using the index
                        return self.graph_results_to_search_results(
                            &graph_results,
                            index,
                            query,
                            &options,
                        );
                    }
                }
                // Fall back to vector search
                self.search(index, query, options)
            }
            QueryType::Semantic => {
                // Pure semantic - use vector search
                self.search(index, query, options)
            }
            QueryType::Hybrid => {
                // Combine both approaches
                let mut vector_results = self.search(index, query, options.clone())?;

                if let Some(ref graph) = self.graph {
                    // Boost results based on graph relationships
                    self.apply_graph_boost(&mut vector_results, graph, query);
                }

                // Re-sort after boost
                vector_results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                Ok(vector_results)
            }
        }
    }

    /// Hybrid search using mmap index
    pub fn search_hybrid_mmap(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_type = classify_query(query);

        match query_type {
            QueryType::Structural => {
                if let Some(ref graph) = self.graph {
                    let graph_results = self.search_graph_only(graph, query, &options);
                    if !graph_results.is_empty() {
                        return self.graph_results_to_search_results_mmap(
                            &graph_results,
                            index,
                            query,
                            &options,
                        );
                    }
                }
                self.search_mmap(index, query, options)
            }
            QueryType::Semantic => {
                self.search_mmap(index, query, options)
            }
            QueryType::Hybrid => {
                let mut vector_results = self.search_mmap(index, query, options.clone())?;

                if let Some(ref graph) = self.graph {
                    self.apply_graph_boost(&mut vector_results, graph, query);
                }

                vector_results.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                Ok(vector_results)
            }
        }
    }

    /// Search using only the graph
    fn search_graph_only(
        &self,
        graph: &CodeGraph,
        query: &str,
        _options: &SearchOptions,
    ) -> Vec<Symbol> {
        let query_lower = query.to_lowercase();

        // Parse structural queries
        if query_lower.contains("callers of") || query_lower.contains("who calls") {
            if let Some(name) = extract_symbol_name_from_query(query, &["callers of", "who calls", "what calls"]) {
                let symbols = graph.find_by_name(&name);
                return symbols
                    .into_iter()
                    .flat_map(|s| graph.find_callers(&s.id))
                    .cloned()
                    .collect();
            }
        }

        if query_lower.contains("calls to") || query_lower.contains("callees of") {
            if let Some(name) = extract_symbol_name_from_query(query, &["calls to", "callees of"]) {
                let symbols = graph.find_by_name(&name);
                return symbols
                    .into_iter()
                    .flat_map(|s| graph.find_callees(&s.id))
                    .cloned()
                    .collect();
            }
        }

        if query_lower.contains("definition of") || query_lower.contains("find definition") {
            if let Some(name) = extract_symbol_name_from_query(query, &["definition of", "find definition", "go to definition"]) {
                return graph.find_by_name(&name).into_iter().cloned().collect();
            }
        }

        if query_lower.contains("implementations of") || query_lower.contains("implementors of") {
            if let Some(name) = extract_symbol_name_from_query(query, &["implementations of", "implementors of"]) {
                let symbols = graph.find_by_name(&name);
                return symbols
                    .into_iter()
                    .flat_map(|s| graph.find_implementors(&s.id))
                    .cloned()
                    .collect();
            }
        }

        if query_lower.contains("imports") {
            if let Some(name) = extract_symbol_name_from_query(query, &["imports", "what imports"]) {
                // Find files that import the given module/file
                let path = std::path::PathBuf::from(&name);
                return graph
                    .find_importers(&path)
                    .into_iter()
                    .flat_map(|p| graph.symbols_in_file(p))
                    .cloned()
                    .collect();
            }
        }

        // Fall back to name search
        let words: Vec<_> = query.split_whitespace().collect();
        if words.len() <= 3 {
            for word in words {
                if word.len() >= 3 && !is_common_word(word) {
                    let results = graph.find_by_name(word);
                    if !results.is_empty() {
                        return results.into_iter().cloned().collect();
                    }
                    // Try prefix match
                    let prefix_results = graph.find_by_prefix(word);
                    if !prefix_results.is_empty() {
                        return prefix_results.into_iter().take(10).cloned().collect();
                    }
                }
            }
        }

        vec![]
    }

    /// Apply graph-based boost to vector search results
    fn apply_graph_boost(
        &self,
        results: &mut [SearchResult],
        graph: &CodeGraph,
        query: &str,
    ) {
        let query_words: Vec<_> = query
            .split_whitespace()
            .filter(|w| w.len() >= 3 && !is_common_word(w))
            .collect();

        for result in results.iter_mut() {
            let file_path = &result.chunk.path;

            // Check if file contains symbols matching query terms
            let file_symbols = graph.symbols_in_file(file_path);
            let mut boost = 0.0f32;

            for symbol in file_symbols {
                for word in &query_words {
                    let word_lower = word.to_lowercase();
                    if symbol.name.to_lowercase().contains(&word_lower) {
                        boost += 0.1;
                    }
                }

                // Boost files with more incoming references (important symbols)
                let incoming = graph.incoming_edges(&symbol.id).len();
                if incoming > 5 {
                    boost += 0.05;
                }
            }

            // Check import relationships
            let imports = graph.find_imports_of(file_path);
            if imports.len() > 0 {
                // Files that import others are likely implementation files
                boost += 0.02 * (imports.len() as f32).min(5.0);
            }

            // Apply boost (capped at 20% increase)
            result.score *= 1.0 + boost.min(0.2);
        }
    }

    /// Convert graph symbols to SearchResults
    fn graph_results_to_search_results(
        &self,
        symbols: &[Symbol],
        index: &RepositoryIndex,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        for symbol in symbols.iter().take(options.limit) {
            // Find matching chunk in the index
            for (i, chunk) in index.chunks.iter().enumerate() {
                if chunk.path == symbol.file_path
                    && chunk.start_line <= symbol.start_line
                    && chunk.end_line >= symbol.end_line
                {
                    let query_vec = self.embedder.embed(query)?;
                    let semantic = cosine_similarity(&query_vec, &index.vectors[i]);

                    results.push(SearchResult {
                        chunk: chunk.clone(),
                        score: 0.8 + semantic * 0.2, // High base score for graph matches
                        semantic_score: semantic,
                        bm25_score: 0.0,
                        keyword_score: 1.0,
                        show_full_context: options.include_context,
                    });
                    break;
                }
            }
        }

        // Sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Convert graph symbols to SearchResults (mmap version)
    fn graph_results_to_search_results_mmap(
        &self,
        symbols: &[Symbol],
        index: &MmapIndex,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        for symbol in symbols.iter().take(options.limit) {
            for (i, chunk) in index.chunks.iter().enumerate() {
                if chunk.path == symbol.file_path
                    && chunk.start_line <= symbol.start_line
                    && chunk.end_line >= symbol.end_line
                {
                    let query_vec = self.embedder.embed(query)?;
                    let semantic = cosine_similarity(&query_vec, index.get_vector(i));

                    results.push(SearchResult {
                        chunk: chunk.clone(),
                        score: 0.8 + semantic * 0.2,
                        semantic_score: semantic,
                        bm25_score: 0.0,
                        keyword_score: 1.0,
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

    /// Find symbols by name
    pub fn find_symbol(&self, name: &str) -> Vec<Symbol> {
        self.graph
            .as_ref()
            .map(|g| g.find_by_name(name).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Find callers of a symbol
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

    /// Find what a function calls
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

    /// Get graph statistics
    pub fn graph_stats(&self) -> Option<crate::graph::GraphStats> {
        self.graph.as_ref().map(|g| g.stats())
    }
}

/// Extract symbol name from a structural query
fn extract_symbol_name_from_query(query: &str, patterns: &[&str]) -> Option<String> {
    let query_lower = query.to_lowercase();

    for pattern in patterns {
        if let Some(idx) = query_lower.find(pattern) {
            let after = &query[idx + pattern.len()..];
            let name = after
                .trim()
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()?
                .trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    None
}

/// Check if a word is a common query word (not a symbol name)
fn is_common_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "this", "that", "into",
        "when", "what", "how", "why", "does", "are", "our", "your",
        "their", "then", "where", "find", "show", "get", "can", "will",
        "should", "would", "could", "function", "method", "class",
        "file", "code", "implement", "implementation", "definition",
        "callers", "callees", "imports", "exports",
    ];
    COMMON_WORDS.contains(&word.to_lowercase().as_str())
}

#[derive(Clone, Serialize)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
    pub semantic_score: f32,
    pub bm25_score: f32,
    pub keyword_score: f32,
    pub show_full_context: bool,
}

impl SearchResult {
    pub fn render_snippet(&self) -> String {
        if self.show_full_context {
            self.chunk.text.clone()
        } else {
            self.chunk
                .text
                .lines()
                .take(12)
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    match f32::cosine(lhs, rhs) {
        Some(distance) => ((1.0 - distance) as f32).clamp(-1.0, 1.0),
        None => cosine_similarity_scalar(lhs, rhs),
    }
}

fn cosine_similarity_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dot: f32 = lhs.iter().zip(rhs).map(|(a, b)| a * b).sum();
    let norm_l: f32 = lhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_r: f32 = rhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_l == 0.0 || norm_r == 0.0 {
        return 0.0;
    }
    (dot / (norm_l * norm_r)).clamp(-1.0, 1.0)
}

fn recency_boost(chunk: &CodeChunk) -> f32 {
    let age_hours = (Utc::now() - chunk.modified_at).num_hours().max(0) as f32;
    1.0 / (1.0 + age_hours / RECENCY_HALF_LIFE_HOURS)
}

fn content_based_file_boost(chunk: &CodeChunk) -> f32 {
    let content_lower = chunk.text.to_lowercase();
    let path_str = chunk.path.to_string_lossy().to_lowercase();
    let word_count = chunk.text.split_whitespace().count().max(1) as f32;

    let test_content_patterns = [
        "assert",
        "expect(",
        ".tobe(",
        ".toequal(",
        "should.",
        "mock",
        "stub",
        "fake",
        "fixture",
        "beforeeach",
        "aftereach",
        "beforeall",
        "afterall",
        "describe(",
        "it(\"",
        "it('",
        "test(\"",
        "test('",
        "@test",
        "#[test]",
        "def test_",
        "func test",
    ];

    let test_term_count: usize = test_content_patterns
        .iter()
        .map(|p| content_lower.matches(p).count())
        .sum();

    let content_test_density = (test_term_count as f32 / word_count).min(0.3);

    let path_test_indicators = [
        "test/",
        "/test/",
        "tests/",
        "/tests/",
        "__tests__/",
        "/__tests__/",
        "spec/",
        "/spec/",
        "specs/",
        "/specs/",
        "fixture/",
        "/fixture/",
        "fixtures/",
        "/fixtures/",
        "__testfixtures__/",
        "/__testfixtures__/",
        "testfixtures/",
        "/testfixtures/",
        "testdata/",
        "/testdata/",
        "test_data/",
        "/test_data/",
        "mock/",
        "/mock/",
        "mocks/",
        "/mocks/",
        "e2e/",
        "/e2e/",
        "integration/",
        "/integration/",
        "unit/",
        "/unit/",
        "unit_test/",
        "/unit_test/",
        "test_utils/",
        "/test_utils/",
        "testutils/",
        "/testutils/",
        "test_helpers/",
        "/test_helpers/",
        "testhelpers/",
        "/testhelpers/",
        "test_support/",
        "/test_support/",
        "testsupport/",
        "/testsupport/",
        "test_common/",
        "/test_common/",
        "testcommon/",
        "/testcommon/",
        "testlib/",
        "/testlib/",
        "test_lib/",
        "/test_lib/",
        "conftest",
        "pytest.ini",
        "jest.config",
        "vitest.config",
        ".test.",
        "_test.",
        ".spec.",
        "_spec.",
        ".test.ts",
        ".test.js",
        ".test.tsx",
        ".test.jsx",
        ".spec.ts",
        ".spec.js",
        ".spec.tsx",
        ".spec.jsx",
        "_test.go",
        "_test.py",
        "_test.rs",
        "_test.java",
        "_test.kt",
        "_test.swift",
        "_test.dart",
        "test_",
        "testcase",
        "test_case",
        "testhelper",
        "test_helper",
        "testutil",
        "test_util",
    ];

    let path_example_indicators = [
        "examples/",
        "/examples/",
        "example/",
        "/example/",
        "samples/",
        "/samples/",
        "sample/",
        "/sample/",
        "demo/",
        "/demo/",
        "demos/",
        "/demos/",
        "playground/",
        "/playground/",
        "playgrounds/",
        "/playgrounds/",
        "sandbox/",
        "/sandbox/",
        "sandboxes/",
        "/sandboxes/",
        "scratch/",
        "/scratch/",
        "scratchpad/",
        "/scratchpad/",
        "tutorial/",
        "/tutorial/",
        "tutorials/",
        "/tutorials/",
        "tut/",
        "/tut/",
        "tuts/",
        "/tuts/",
        "example_",
        "sample_",
        "demo_",
    ];

    let path_impl_indicators = [
        "/src/",
        "/lib/",
        "/pkg/",
        "/internal/",
        "/core/",
        "/server/",
        "/client/",
        "/api/",
        "/services/",
        "src/",
        "lib/",
        "pkg/",
        "internal/",
        "core/",
        "server/",
        "client/",
        "api/",
        "services/",
        "/app/",
        "/apps/",
        "/application/",
        "/applications/",
        "app/",
        "apps/",
        "application/",
        "applications/",
        "/components/",
        "/modules/",
        "/utils/",
        "/utilities/",
        "components/",
        "modules/",
        "utils/",
        "utilities/",
        "/common/",
        "/shared/",
        "/public/",
        "/private/",
        "common/",
        "shared/",
        "public/",
        "private/",
        "/main/",
        "/java/",
        "/scala/",
        "/python/",
        "main/",
        "java/",
        "scala/",
        "python/",
        "/include/",
        "/headers/",
        "/bin/",
        "/scripts/",
        "include/",
        "headers/",
        "bin/",
        "scripts/",
        "/domain/",
        "/business/",
        "/logic/",
        "/model/",
        "domain/",
        "business/",
        "logic/",
        "model/",
        "/controllers/",
        "/views/",
        "/models/",
        "/routes/",
        "controllers/",
        "views/",
        "models/",
        "routes/",
        "/handlers/",
        "/middleware/",
        "/providers/",
        "/repositories/",
        "handlers/",
        "middleware/",
        "providers/",
        "repositories/",
        "/entities/",
        "/interfaces/",
        "/types/",
        "/schemas/",
        "entities/",
        "interfaces/",
        "types/",
        "schemas/",
    ];

    let path_is_test = path_test_indicators.iter().any(|p| path_str.contains(p));
    let path_is_example = path_example_indicators.iter().any(|p| path_str.contains(p));
    let path_is_impl = path_impl_indicators.iter().any(|p| path_str.contains(p))
        && !path_is_test
        && !path_is_example;

    let ext = chunk
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_doc = matches!(ext.as_str(), "md" | "mdx" | "rst" | "txt" | "adoc");
    let is_error_doc = path_str.contains("/errors/");

    let score = if is_doc {
        if is_error_doc { 0.2 } else { 0.1 }
    } else if path_is_test {
        0.1
    } else if path_is_example {
        0.2
    } else if path_is_impl {
        1.0 - (content_test_density * 0.3)
    } else {
        0.6 - (content_test_density * 0.3)
    };

    score.clamp(0.05, 1.0)
}

fn select_top_k(matches: &mut Vec<SearchResult>, k: usize) {
    if matches.len() > k {
        matches.select_nth_unstable_by(k, |a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
    }
    sort_by_score(matches);
}

fn sort_by_score(matches: &mut [SearchResult]) {
    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn quantize_to_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = vector.len().div_ceil(64);
    let mut binary = vec![0u64; num_words];
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            binary[i / 64] |= 1u64 << (i % 64);
        }
    }
    binary
}

fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

fn binary_shortlist(
    query_binary: &[u64],
    index_binary: &[Vec<u64>],
    shortlist_size: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, u32)> = index_binary
        .iter()
        .enumerate()
        .map(|(i, v)| (i, hamming_distance(query_binary, v)))
        .collect();

    if distances.len() > shortlist_size {
        distances.select_nth_unstable_by_key(shortlist_size, |&(_, d)| d);
        distances.truncate(shortlist_size);
    }

    distances.into_iter().map(|(i, _)| i).collect()
}

fn binary_shortlist_precomputed(
    query_binary: &[u64],
    index: &MmapIndex,
    shortlist_size: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, u32)> = (0..index.len())
        .filter_map(|i| {
            index
                .get_binary_vector(i)
                .map(|v| (i, hamming_distance(query_binary, v)))
        })
        .collect();

    if distances.len() > shortlist_size {
        distances.select_nth_unstable_by_key(shortlist_size, |&(_, d)| d);
        distances.truncate(shortlist_size);
    }

    distances.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::IndexMetadata;
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
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: language.to_string(),
            start_line: 1,
            end_line: 10,
            text: text.to_string(),
            hash: "test_hash".to_string(),
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
        };
        RepositoryIndex::new(metadata, chunks, vectors)
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        let vec = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&vec, &vec) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&vec1, &vec2).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&vec1, &vec2) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_has_similarity_zero() {
        let zero = vec![0.0, 0.0, 0.0];
        let nonzero = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zero, &nonzero), 0.0);
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
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: false,
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
                    limit: 10,
                    include_context: false,
                    glob: vec!["src/**/*.rs".to_string()],
                    filters: vec![],
                    rerank: false,
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
                    limit: 10,
                    include_context: false,
                    glob: vec![],
                    filters: vec!["lang=rust".to_string()],
                    rerank: false,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.language, "rust");
    }

    #[test]
    fn snippet_truncates_without_context() {
        let chunk = make_chunk(
            &(1..=14)
                .map(|i| format!("line{}", i))
                .collect::<Vec<_>>()
                .join("\n"),
            "rust",
            "test.rs",
        );
        let result = SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
            bm25_score: 0.0,
            keyword_score: 0.0,
            show_full_context: false,
        };
        assert_eq!(result.render_snippet().lines().count(), 12);
    }

    #[test]
    fn snippet_shows_full_with_context() {
        let chunk = make_chunk(
            &(1..=14)
                .map(|i| format!("line{}", i))
                .collect::<Vec<_>>()
                .join("\n"),
            "rust",
            "test.rs",
        );
        let result = SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
            bm25_score: 0.0,
            keyword_score: 0.0,
            show_full_context: true,
        };
        assert_eq!(result.render_snippet().lines().count(), 14);
    }

    #[test]
    fn recency_boost_decays_over_time() {
        let recent = make_chunk("test", "rust", "test.rs");
        let mut old = make_chunk("test", "rust", "test.rs");
        old.modified_at = Utc::now() - chrono::Duration::days(30);

        let recent_boost = recency_boost(&recent);
        let old_boost = recency_boost(&old);

        assert!(recent_boost > old_boost);
        assert!(recent_boost <= 1.0);
        assert!(old_boost > 0.0);
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
                keyword_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                keyword_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.6,
                semantic_score: 0.6,
                bm25_score: 0.0,
                keyword_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk.clone(),
                score: 0.1,
                semantic_score: 0.1,
                bm25_score: 0.0,
                keyword_score: 0.0,
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
                    limit: 10,
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: false,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn quantize_to_binary_sets_bits_for_positive_values() {
        let vector = vec![1.0, -0.5, 0.3, -0.1, 0.0, 0.001];
        let binary = quantize_to_binary(&vector);

        assert_eq!(binary.len(), 1);
        assert_eq!(binary[0] & 0b000001, 1);
        assert_eq!(binary[0] & 0b000010, 0);
        assert_eq!(binary[0] & 0b000100, 0b000100);
        assert_eq!(binary[0] & 0b001000, 0);
        assert_eq!(binary[0] & 0b010000, 0);
        assert_eq!(binary[0] & 0b100000, 0b100000);
    }

    #[test]
    fn quantize_to_binary_handles_large_vectors() {
        let vector: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let binary = quantize_to_binary(&vector);

        assert_eq!(binary.len(), 2);
        for word in &binary {
            assert_eq!(word.count_ones(), 32);
        }
    }

    #[test]
    fn hamming_distance_identical_vectors() {
        let a = vec![0b1010101010u64];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_distance_completely_different() {
        let a = vec![0u64];
        let b = vec![u64::MAX];
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    #[test]
    fn hamming_distance_partial_difference() {
        let a = vec![0b1111_0000u64];
        let b = vec![0b1111_1111u64];
        assert_eq!(hamming_distance(&a, &b), 4);
    }

    #[test]
    fn binary_shortlist_returns_closest_candidates() {
        let query = vec![0b1111u64];
        let index_binary = vec![
            vec![0b1111u64],
            vec![0b0000u64],
            vec![0b1110u64],
            vec![0b0001u64],
        ];

        let shortlist = binary_shortlist(&query, &index_binary, 2);

        assert_eq!(shortlist.len(), 2);
        assert!(shortlist.contains(&0));
        assert!(shortlist.contains(&2));
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
                    limit: 10,
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: false,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn reranking_reorders_results_when_enabled() {
        use crate::reranker::MockReranker;

        let embedder = Arc::new(MockEmbedder);
        let reranker = Arc::new(MockReranker);
        let engine = SearchEngine::with_reranker(embedder.clone(), reranker);

        // Create chunks with different lengths - MockReranker prefers longer documents
        let chunks = vec![
            make_chunk("short", "rust", "a.rs"),
            make_chunk(
                "this is a much longer document that should be preferred",
                "rust",
                "b.rs",
            ),
            make_chunk("medium length", "rust", "c.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "query",
                SearchOptions {
                    limit: 3,
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: true,
                },
            )
            .unwrap();

        // MockReranker prefers longer documents, so the longest should be first
        assert_eq!(results.len(), 3);
        assert!(results[0].chunk.text.contains("much longer"));
    }

    #[test]
    fn reranking_skipped_when_disabled() {
        use crate::reranker::MockReranker;

        let embedder = Arc::new(MockEmbedder);
        let reranker = Arc::new(MockReranker);
        let engine = SearchEngine::with_reranker(embedder.clone(), reranker);

        let chunks = vec![
            make_chunk("fn foo() {}", "rust", "a.rs"),
            make_chunk("fn bar() {}", "rust", "b.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        // Search with rerank: false
        let results = engine
            .search(
                &make_index(chunks, vectors),
                "function",
                SearchOptions {
                    limit: 2,
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: false,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        // Results should be in semantic score order, not reranked
    }

    #[test]
    fn reranking_skipped_without_reranker() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone()); // No reranker

        let chunks = vec![
            make_chunk("fn foo() {}", "rust", "a.rs"),
            make_chunk("fn bar() {}", "rust", "b.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        // Even with rerank: true, should work without panicking
        let results = engine
            .search(
                &make_index(chunks, vectors),
                "function",
                SearchOptions {
                    limit: 2,
                    include_context: false,
                    glob: vec![],
                    filters: vec![],
                    rerank: true,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_options_default_has_rerank_disabled() {
        let options = SearchOptions::default();
        assert!(!options.rerank);
        assert_eq!(options.limit, 10);
    }
}
