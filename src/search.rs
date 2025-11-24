use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use simsimd::SpatialSimilarity;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::fts;
use crate::store::{MmapIndex, RepositoryIndex};

const HNSW_THRESHOLD: usize = 500;
const BINARY_QUANTIZATION_THRESHOLD: usize = 1000;
const BINARY_SHORTLIST_FACTOR: usize = 10;
const HNSW_CONNECTIVITY: usize = 16;
const HNSW_EXPANSION_ADD: usize = 128;
const HNSW_EXPANSION_SEARCH: usize = 64;
const HNSW_OVERSAMPLE_FACTOR: usize = 4;

const SEMANTIC_WEIGHT: f32 = 0.7;
const KEYWORD_WEIGHT: f32 = 0.2;
const RECENCY_WEIGHT: f32 = 0.1;
const RECENCY_HALF_LIFE_HOURS: f32 = 48.0;

pub struct SearchOptions {
    pub limit: usize,
    pub include_context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
}

pub struct SearchEngine {
    embedder: Arc<dyn BatchEmbedder>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<dyn BatchEmbedder>) -> Self {
        Self { embedder }
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
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let globset = fts::build_globset(&options.glob);

        let mut matches: Vec<SearchResult> = index
            .chunks
            .iter()
            .zip(&index.vectors)
            .filter(|(chunk, _)| fts::glob_matches(globset.as_ref(), &chunk.path))
            .filter(|(chunk, _)| fts::matches_filters(&options.filters, chunk))
            .map(|(chunk, vector)| {
                self.score_chunk(chunk, vector, &query_vec, &keywords, options.include_context)
            })
            .collect();

        select_top_k(&mut matches, limit);
        Ok(matches)
    }

    fn search_binary_quantized(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);

        let query_binary = quantize_to_binary(&query_vec);
        let index_binary: Vec<Vec<u64>> = index.vectors.iter().map(|v| quantize_to_binary(v)).collect();
        let shortlist_size = (limit * BINARY_SHORTLIST_FACTOR).min(index.vectors.len());
        let candidates = binary_shortlist(&query_binary, &index_binary, shortlist_size);

        let mut matches: Vec<SearchResult> = candidates
            .into_iter()
            .map(|idx| {
                self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    &keywords,
                    options.include_context,
                )
            })
            .collect();

        select_top_k(&mut matches, limit);
        Ok(matches)
    }

    fn search_hnsw(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);

        let hnsw = self.build_hnsw_index(index.metadata.vector_dim, index.vectors.len())?;
        for (i, vector) in index.vectors.iter().enumerate() {
            hnsw.add(i as u64, vector)
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates = self.search_hnsw_candidates(&hnsw, &query_vec, limit, index.vectors.len())?;

        let mut matches: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|idx| {
                if idx >= index.chunks.len() {
                    return None;
                }
                Some(self.score_chunk(
                    &index.chunks[idx],
                    &index.vectors[idx],
                    &query_vec,
                    &keywords,
                    options.include_context,
                ))
            })
            .collect();

        sort_by_score(&mut matches);
        matches.truncate(limit);
        Ok(matches)
    }

    fn search_mmap_linear(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let globset = fts::build_globset(&options.glob);

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
                Some(self.score_chunk(chunk, vector, &query_vec, &keywords, options.include_context))
            })
            .collect();

        select_top_k(&mut matches, limit);
        Ok(matches)
    }

    fn search_mmap_hnsw(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);

        let hnsw = self.build_hnsw_index(index.metadata.vector_dim, index.len())?;
        for i in 0..index.len() {
            hnsw.add(i as u64, index.get_vector(i))
                .map_err(|e| anyhow::anyhow!("HNSW add failed: {}", e))?;
        }

        let candidates = self.search_hnsw_candidates(&hnsw, &query_vec, limit, index.len())?;

        let mut matches: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|idx| {
                if idx >= index.len() {
                    return None;
                }
                Some(self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    &keywords,
                    options.include_context,
                ))
            })
            .collect();

        sort_by_score(&mut matches);
        matches.truncate(limit);
        Ok(matches)
    }

    fn search_mmap_binary_quantized(
        &self,
        index: &MmapIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);
        let shortlist_size = (limit * BINARY_SHORTLIST_FACTOR).min(index.len());

        let query_binary = quantize_to_binary(&query_vec);

        // Use pre-computed binary vectors if available, otherwise compute on the fly
        let candidates = if index.has_binary_vectors() {
            binary_shortlist_precomputed(&query_binary, index, shortlist_size)
        } else {
            let index_binary: Vec<Vec<u64>> = (0..index.len())
                .map(|i| quantize_to_binary(index.get_vector(i)))
                .collect();
            binary_shortlist(&query_binary, &index_binary, shortlist_size)
        };

        let mut matches: Vec<SearchResult> = candidates
            .into_iter()
            .map(|idx| {
                self.score_chunk(
                    &index.chunks[idx],
                    index.get_vector(idx),
                    &query_vec,
                    &keywords,
                    options.include_context,
                )
            })
            .collect();

        select_top_k(&mut matches, limit);
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

        let hnsw = Index::new(&options)
            .map_err(|e| anyhow::anyhow!("HNSW creation failed: {}", e))?;
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
        include_context: bool,
    ) -> SearchResult {
        let semantic = cosine_similarity(query_vec, vector);
        let keyword = fts::keyword_score(keywords, &chunk.text, &chunk.path);
        let recency = recency_boost(chunk);
        let score = SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * keyword + RECENCY_WEIGHT * recency;

        SearchResult {
            chunk: chunk.clone(),
            score,
            semantic_score: semantic,
            keyword_score: keyword,
            show_full_context: include_context,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub show_full_context: bool,
}

impl SearchResult {
    pub fn render_snippet(&self) -> String {
        if self.show_full_context {
            self.chunk.text.clone()
        } else {
            self.chunk.text.lines().take(12).collect::<Vec<_>>().join("\n")
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

fn select_top_k(matches: &mut Vec<SearchResult>, k: usize) {
    if matches.len() > k {
        matches.select_nth_unstable_by(k, |a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
    }
    sort_by_score(matches);
}

fn sort_by_score(matches: &mut [SearchResult]) {
    matches.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
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
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
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
            index.get_binary_vector(i).map(|v| (i, hamming_distance(query_binary, v)))
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
            Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0]).collect())
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
        let vectors: Vec<Vec<f32>> = chunks.iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine.search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions { limit: 2, include_context: false, glob: vec![], filters: vec![] },
        ).unwrap();

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
        let vectors: Vec<Vec<f32>> = chunks.iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine.search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                limit: 10,
                include_context: false,
                glob: vec!["src/**/*.rs".to_string()],
                filters: vec![],
            },
        ).unwrap();

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
        let vectors: Vec<Vec<f32>> = chunks.iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine.search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                limit: 10,
                include_context: false,
                glob: vec![],
                filters: vec!["lang=rust".to_string()],
            },
        ).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.language, "rust");
    }

    #[test]
    fn snippet_truncates_without_context() {
        let chunk = make_chunk(
            &(1..=14).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n"),
            "rust",
            "test.rs",
        );
        let result = SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
            keyword_score: 0.0,
            show_full_context: false,
        };
        assert_eq!(result.render_snippet().lines().count(), 12);
    }

    #[test]
    fn snippet_shows_full_with_context() {
        let chunk = make_chunk(
            &(1..=14).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n"),
            "rust",
            "test.rs",
        );
        let result = SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
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
            SearchResult { chunk: chunk.clone(), score: 0.3, semantic_score: 0.3, keyword_score: 0.0, show_full_context: false },
            SearchResult { chunk: chunk.clone(), score: 0.9, semantic_score: 0.9, keyword_score: 0.0, show_full_context: false },
            SearchResult { chunk: chunk.clone(), score: 0.6, semantic_score: 0.6, keyword_score: 0.0, show_full_context: false },
            SearchResult { chunk: chunk.clone(), score: 0.1, semantic_score: 0.1, keyword_score: 0.0, show_full_context: false },
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
            .map(|i| make_chunk(&format!("fn func{}() {{}}", i), "rust", &format!("file{}.rs", i)))
            .collect();
        let vectors: Vec<Vec<f32>> = chunks.iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine.search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions { limit: 10, include_context: false, glob: vec![], filters: vec![] },
        ).unwrap();

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
        let vector: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
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
            .map(|i| make_chunk(&format!("fn func{}() {{}}", i), "rust", &format!("file{}.rs", i)))
            .collect();
        let vectors: Vec<Vec<f32>> = chunks.iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine.search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions { limit: 10, include_context: false, glob: vec![], filters: vec![] },
        ).unwrap();

        assert_eq!(results.len(), 10);
    }
}
