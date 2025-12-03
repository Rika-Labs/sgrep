use serde::Serialize;
use std::path::PathBuf;

use crate::chunker::CodeChunk;

#[derive(Clone, Serialize)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
    pub semantic_score: f32,
    pub bm25_score: f32,
    pub show_full_context: bool,
}

impl SearchResult {
    pub fn render_snippet(&self) -> String {
        let content_start = Self::find_content_start(&self.chunk.text);
        let content = &self.chunk.text[content_start..];

        if self.show_full_context {
            content.to_string()
        } else {
            content.lines().take(12).collect::<Vec<_>>().join("\n")
        }
    }

    fn find_content_start(text: &str) -> usize {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return 0;
        }

        let start_idx = if lines[0].starts_with("// File:") {
            1
        } else {
            0
        };

        let mut content_line = start_idx;
        for (i, line) in lines.iter().enumerate().skip(start_idx) {
            if line.trim().is_empty() {
                content_line = i + 1;
                break;
            }
        }

        if lines
            .get(content_line)
            .map(|l| l.starts_with("// Context:"))
            .unwrap_or(false)
        {
            content_line += 1;
        }

        if content_line >= lines.len() {
            return 0;
        }

        lines[..content_line]
            .iter()
            .map(|l| l.len() + 1)
            .sum::<usize>()
            .min(text.len())
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
