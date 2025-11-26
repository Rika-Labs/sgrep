mod binary;
mod graph_hybrid;
mod hnsw;
mod results;
mod scoring;

pub use results::{DirectorySearchResult, FileSearchResult, SearchResult};
pub use scoring::{cosine_similarity, AdaptiveWeights};

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::fts::{self, Bm25FDocument, Bm25FIndex};
use crate::graph::{CodeGraph, QueryType, Symbol};
use crate::query_expander::{QueryAnalysis, QueryExpander};
use crate::reranker::Reranker;
use crate::store::{HierarchicalIndex, MmapIndex, RepositoryIndex};

use binary::{binary_shortlist, binary_shortlist_precomputed, quantize_to_binary};
use graph_hybrid::{apply_graph_boost, search_graph_only};
use hnsw::{build_hnsw_index, search_hnsw_candidates};
use scoring::{
    content_based_file_boost, directory_match_boost, filename_match_boost, implementation_boost,
    normalize_bm25_scores, recency_boost, reexport_file_penalty, select_top_k,
    AdaptiveWeights as Weights,
};

const HNSW_THRESHOLD: usize = 500;
const BINARY_QUANTIZATION_THRESHOLD: usize = 1000;
const BINARY_SHORTLIST_FACTOR: usize = 10;
const RERANK_OVERSAMPLE_FACTOR: usize = 3;
const PRF_TOP_K: usize = 10;
const PRF_EXPANSION_TERMS: usize = 5;

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

pub struct SearchEngine {
    embedder: Arc<dyn BatchEmbedder>,
    reranker: Option<Arc<dyn Reranker>>,
    graph: Option<CodeGraph>,
    query_expander: Option<QueryExpander>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self {
            embedder,
            reranker: None,
            graph: None,
            query_expander: None,
        }
    }

    #[allow(dead_code)]
    pub fn with_reranker(embedder: Arc<dyn BatchEmbedder>, reranker: Arc<dyn Reranker>) -> Self {
        Self {
            embedder,
            reranker: Some(reranker),
            graph: None,
            query_expander: None,
        }
    }

    /// Enable the query expander for intelligent query understanding.
    /// This downloads the Qwen2.5 model on first use (~400MB).
    pub fn enable_query_expander(&mut self) -> Result<()> {
        if self.query_expander.is_none() {
            self.query_expander = Some(QueryExpander::new()?);
        }
        Ok(())
    }

    pub fn set_graph(&mut self, graph: CodeGraph) {
        self.graph = Some(graph);
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Analyze a query using the LLM-based expander, or fall back to heuristics.
    fn analyze_query(&self, query: &str) -> QueryAnalysis {
        if let Some(ref expander) = self.query_expander {
            expander
                .analyze(query)
                .unwrap_or_else(|_| QueryAnalysis::from_heuristics(query))
        } else {
            QueryAnalysis::from_heuristics(query)
        }
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
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let globset = fts::build_globset(&options.glob);
        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

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
        let weights = Weights::from_query(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

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
        let weights = Weights::from_query(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

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
        let weights = Weights::from_query(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let globset = fts::build_globset(&options.glob);
        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

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
        let weights = Weights::from_query(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };

        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

        let hnsw = build_hnsw_index(index.metadata.vector_dim, index.len())?;
        for i in 0..index.len() {
            hnsw.add(i as u64, index.get_vector(i))
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates =
            search_hnsw_candidates(&hnsw, &query_vec, PRF_TOP_K.max(fetch_limit), index.len())?;

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
                &hnsw,
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
        let weights = Weights::from_query(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let fetch_limit = if options.rerank && self.reranker.is_some() {
            limit * RERANK_OVERSAMPLE_FACTOR
        } else {
            limit
        };
        let shortlist_size =
            (PRF_TOP_K.max(fetch_limit) * BINARY_SHORTLIST_FACTOR).min(index.len());

        let bm25f_index = build_bm25f_from_chunks(&index.chunks, self.graph.as_ref());

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

    fn score_chunk(
        &self,
        chunk: &CodeChunk,
        vector: &[f32],
        query_vec: &[f32],
        query: &str,
        bm25_raw: f32,
        bm25_normalized: f32,
        include_context: bool,
        weights: &Weights,
    ) -> SearchResult {
        let semantic = cosine_similarity(query_vec, vector);
        let recency = recency_boost(chunk);
        let file_type = content_based_file_boost(chunk);
        let dir_match = directory_match_boost(chunk, query);
        let reexport_penalty = reexport_file_penalty(chunk);
        let impl_boost = implementation_boost(chunk, self.graph.as_ref());
        let filename_boost = filename_match_boost(chunk, query);

        let score = weights.semantic * semantic
            + weights.bm25 * bm25_normalized
            + weights.recency * recency
            + weights.file_type * file_type
            + dir_match
            + reexport_penalty
            + impl_boost
            + filename_boost;

        SearchResult {
            chunk: chunk.clone(),
            score,
            semantic_score: semantic,
            bm25_score: bm25_raw,
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

    fn maybe_rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        if !options.rerank {
            return results;
        }

        let reranker = match &self.reranker {
            Some(r) => r,
            None => return results,
        };

        if results.len() <= 1 {
            return results;
        }

        let docs: Vec<&str> = results.iter().map(|r| r.chunk.text.as_str()).collect();

        match reranker.rerank(query, &docs) {
            Ok(reranked) => {
                let mut reordered: Vec<SearchResult> = reranked
                    .into_iter()
                    .filter_map(|(idx, rerank_score)| {
                        if idx < results.len() {
                            let mut result = results[idx].clone();
                            let rerank_boost = rerank_score * 0.4;
                            result.score = result.score + rerank_boost;
                            Some(result)
                        } else {
                            None
                        }
                    })
                    .collect();

                reordered.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                reordered.truncate(options.limit);
                reordered
            }
            Err(_) => results,
        }
    }

    pub fn search_hybrid(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let analysis = self.analyze_query(query);
        let query_type = analysis.to_query_type();

        match query_type {
            QueryType::Structural => {
                if let Some(ref graph) = self.graph {
                    let graph_results = search_graph_only(graph, query);
                    if !graph_results.is_empty() {
                        return self.graph_results_to_search_results(
                            &graph_results,
                            index,
                            query,
                            &options,
                        );
                    }
                }
                self.search_with_fallback(index, query, options, &analysis)
            }
            QueryType::Semantic => self.search_with_fallback(index, query, options, &analysis),
            QueryType::Hybrid => {
                let mut vector_results =
                    self.search_with_fallback(index, query, options.clone(), &analysis)?;

                if let Some(ref graph) = self.graph {
                    apply_graph_boost(&mut vector_results, graph, query);
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

    fn search_with_fallback(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut results = self.search(index, query, options.clone())?;

        let needs_fallback =
            results.is_empty() || results.first().map(|r| r.score < 0.3).unwrap_or(true);

        if needs_fallback && !analysis.expanded_queries.is_empty() {
            for expanded_query in analysis.expanded_queries.iter().take(3) {
                if expanded_query == query {
                    continue;
                }

                let expanded_results = self.search(index, expanded_query, options.clone())?;

                for result in expanded_results {
                    if !results.iter().any(|r| r.chunk.id == result.chunk.id) {
                        let mut discounted = result;
                        discounted.score *= 0.9;
                        results.push(discounted);
                    }
                }
            }

            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(options.limit);
        }

        Ok(results)
    }

    pub fn search_hybrid_mmap(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let analysis = self.analyze_query(query);
        let query_type = analysis.to_query_type();

        match query_type {
            QueryType::Structural => {
                if let Some(ref graph) = self.graph {
                    let graph_results = search_graph_only(graph, query);
                    if !graph_results.is_empty() {
                        return self.graph_results_to_search_results_mmap(
                            &graph_results,
                            index,
                            query,
                            &options,
                        );
                    }
                }
                self.search_mmap_with_fallback(index, query, options, &analysis)
            }
            QueryType::Semantic => self.search_mmap_with_fallback(index, query, options, &analysis),
            QueryType::Hybrid => {
                let mut vector_results =
                    self.search_mmap_with_fallback(index, query, options.clone(), &analysis)?;

                if let Some(ref graph) = self.graph {
                    apply_graph_boost(&mut vector_results, graph, query);
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

    fn search_mmap_with_fallback(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
        analysis: &QueryAnalysis,
    ) -> Result<Vec<SearchResult>> {
        let mut results = self.search_mmap(index, query, options.clone())?;

        let needs_fallback =
            results.is_empty() || results.first().map(|r| r.score < 0.3).unwrap_or(true);

        if needs_fallback && !analysis.expanded_queries.is_empty() {
            for expanded_query in analysis.expanded_queries.iter().take(3) {
                if expanded_query == query {
                    continue;
                }

                let expanded_results = self.search_mmap(index, expanded_query, options.clone())?;

                for result in expanded_results {
                    if !results.iter().any(|r| r.chunk.id == result.chunk.id) {
                        let mut discounted = result;
                        discounted.score *= 0.9;
                        results.push(discounted);
                    }
                }
            }

            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(options.limit);
        }

        Ok(results)
    }

    fn graph_results_to_search_results(
        &self,
        symbols: &[Symbol],
        index: &RepositoryIndex,
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
    fn reranking_skipped_without_reranker() {
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
                    limit: 10,
                    include_context: false,
                    glob: vec![],
                    filters: vec!["lang=python".to_string()],
                    rerank: false,
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
                    limit: 10,
                    include_context: false,
                    glob: vec!["src/**/*.rs".to_string()],
                    filters: vec![],
                    rerank: false,
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
                    limit: 10,
                    include_context: true,
                    glob: vec![],
                    filters: vec![],
                    rerank: false,
                },
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].show_full_context);
    }

    #[test]
    fn select_top_k_handles_less_than_k() {
        let chunk = make_chunk("test", "rust", "test.rs");
        let mut matches = vec![SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
            bm25_score: 0.0,
            show_full_context: false,
        }];

        select_top_k(&mut matches, 10);

        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn select_top_k_empty_input() {
        let mut matches: Vec<SearchResult> = vec![];
        select_top_k(&mut matches, 5);
        assert!(matches.is_empty());
    }
}
