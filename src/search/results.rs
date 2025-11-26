use std::path::PathBuf;
use serde::Serialize;

use crate::chunker::CodeChunk;

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

#[derive(Clone, Debug, Serialize)]
pub struct FileSearchResult {
    pub path: PathBuf,
    pub score: f32,
    pub chunk_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DirectorySearchResult {
    pub path: PathBuf,
    pub score: f32,
    pub file_count: usize,
    pub chunk_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

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
    fn file_search_result_creation() {
        let result = FileSearchResult {
            path: PathBuf::from("src/auth.rs"),
            score: 0.85,
            chunk_count: 5,
        };

        assert_eq!(result.path, PathBuf::from("src/auth.rs"));
        assert!((result.score - 0.85).abs() < 1e-6);
        assert_eq!(result.chunk_count, 5);
    }

    #[test]
    fn directory_search_result_creation() {
        let result = DirectorySearchResult {
            path: PathBuf::from("src/auth"),
            score: 0.75,
            file_count: 3,
            chunk_count: 12,
        };

        assert_eq!(result.path, PathBuf::from("src/auth"));
        assert!((result.score - 0.75).abs() < 1e-6);
        assert_eq!(result.file_count, 3);
        assert_eq!(result.chunk_count, 12);
    }
}
