use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use simsimd::SpatialSimilarity;

use crate::chunker::CodeChunk;
use crate::embedding::Embedder;
use crate::fts;
use crate::store::RepositoryIndex;

pub struct SearchOptions {
    pub limit: usize,
    pub include_context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
}

pub struct SearchEngine {
    embedder: Arc<Embedder>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<Embedder>) -> Self {
        Self { embedder }
    }

    pub fn search(
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
                let semantic = cosine_similarity(&query_vec, vector);
                let keyword = fts::keyword_score(&keywords, &chunk.text, &chunk.path);
                let recency = recency_boost(chunk);
                let final_score = 0.7 * semantic + 0.2 * keyword + 0.1 * recency;
                SearchResult {
                    chunk: chunk.clone(),
                    score: final_score,
                    semantic_score: semantic,
                    keyword_score: keyword,
                    show_full_context: options.include_context,
                }
            })
            .collect();

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }
}

#[derive(Clone)]
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
    (1.0 - f32::cosine(lhs, rhs).unwrap_or(1.0)) as f32
}

fn recency_boost(chunk: &CodeChunk) -> f32 {
    let age_hours = (Utc::now() - chunk.modified_at).num_hours().max(0) as f32;
    1.0 / (1.0 + age_hours / 48.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn create_test_chunk(text: &str, language: &str, path: &str) -> CodeChunk {
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

    fn create_test_index(chunks: Vec<CodeChunk>, vectors: Vec<Vec<f32>>) -> RepositoryIndex {
        use crate::store::IndexMetadata;
        
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
    fn cosine_similarity_identical_vectors() {
        let vec = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec, &vec);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![-1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vectors() {
        let vec1 = vec![0.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec1, &vec2);
        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn search_returns_limited_results() {
        let embedder = Arc::new(Embedder::default());
        let engine = SearchEngine::new(embedder.clone());
        
        let chunks = vec![
            create_test_chunk("fn authenticate() {}", "rust", "auth.rs"),
            create_test_chunk("fn validate() {}", "rust", "validate.rs"),
            create_test_chunk("fn login() {}", "rust", "login.rs"),
        ];
        let vectors = vec![
            embedder.embed("authenticate").unwrap(),
            embedder.embed("validate").unwrap(),
            embedder.embed("login").unwrap(),
        ];
        let index = create_test_index(chunks, vectors);
        
        let options = SearchOptions {
            limit: 2,
            include_context: false,
            glob: vec![],
            filters: vec![],
        };
        
        let results = engine.search(&index, "authentication", options).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_respects_glob_filters() {
        let embedder = Arc::new(Embedder::default());
        let engine = SearchEngine::new(embedder.clone());
        
        let chunks = vec![
            create_test_chunk("fn test1() {}", "rust", "src/auth.rs"),
            create_test_chunk("fn test2() {}", "rust", "tests/auth.rs"),
        ];
        let vectors = vec![
            embedder.embed("test").unwrap(),
            embedder.embed("test").unwrap(),
        ];
        let index = create_test_index(chunks, vectors);
        
        let options = SearchOptions {
            limit: 10,
            include_context: false,
            glob: vec!["src/**/*.rs".to_string()],
            filters: vec![],
        };
        
        let results = engine.search(&index, "test", options).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.path.to_string_lossy().contains("src"));
    }

    #[test]
    fn search_respects_language_filters() {
        let embedder = Arc::new(Embedder::default());
        let engine = SearchEngine::new(embedder.clone());
        
        let chunks = vec![
            create_test_chunk("fn test() {}", "rust", "test.rs"),
            create_test_chunk("def test():", "python", "test.py"),
        ];
        let vectors = vec![
            embedder.embed("test").unwrap(),
            embedder.embed("test").unwrap(),
        ];
        let index = create_test_index(chunks, vectors);
        
        let options = SearchOptions {
            limit: 10,
            include_context: false,
            glob: vec![],
            filters: vec!["lang=rust".to_string()],
        };
        
        let results = engine.search(&index, "test", options).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.language, "rust");
    }

    #[test]
    fn search_result_render_snippet_respects_context_flag() {
        let chunk = create_test_chunk(
            "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\nline11\nline12\nline13\nline14",
            "rust",
            "test.rs"
        );
        
        let result_no_context = SearchResult {
            chunk: chunk.clone(),
            score: 0.5,
            semantic_score: 0.5,
            keyword_score: 0.0,
            show_full_context: false,
        };
        
        let snippet = result_no_context.render_snippet();
        assert_eq!(snippet.lines().count(), 12);
        
        let result_with_context = SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.5,
            keyword_score: 0.0,
            show_full_context: true,
        };
        
        let full = result_with_context.render_snippet();
        assert_eq!(full.lines().count(), 14);
    }

    #[test]
    fn recency_boost_decreases_with_age() {
        let recent = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: "test".to_string(),
            hash: "hash".to_string(),
            modified_at: Utc::now(),
        };
        
        let old = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: "test".to_string(),
            hash: "hash".to_string(),
            modified_at: Utc::now() - chrono::Duration::days(30),
        };
        
        let recent_boost = recency_boost(&recent);
        let old_boost = recency_boost(&old);
        
        assert!(recent_boost > old_boost);
        assert!(recent_boost <= 1.0);
        assert!(old_boost > 0.0);
    }
}
