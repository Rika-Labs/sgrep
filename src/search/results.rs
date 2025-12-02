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
        let text = &self.chunk.text;

        // Skip file context (imports/use statements prepended for embedding)
        // Context ends with a blank line before the actual code
        let content_start = Self::find_content_start(text);
        let content = &text[content_start..];

        if self.show_full_context {
            content.to_string()
        } else {
            content.lines().take(12).collect::<Vec<_>>().join("\n")
        }
    }

    fn find_content_start(text: &str) -> usize {
        // Look for the pattern: context lines followed by blank line
        // Context lines are: "// File:", "use ", "import ", "package ", etc.
        let lines: Vec<&str> = text.lines().collect();
        let mut last_context_line = 0;
        let mut found_blank_after_context = false;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if last_context_line > 0 && i == last_context_line + 1 {
                    found_blank_after_context = true;
                }
                continue;
            }

            let is_context = trimmed.starts_with("// File:")
                || trimmed.starts_with("// Context:")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("package ")
                || trimmed.starts_with("require ")
                || trimmed.starts_with("#include")
                || trimmed.starts_with("using ");

            if is_context {
                last_context_line = i;
            } else if found_blank_after_context || last_context_line == 0 {
                // Found actual content - return byte offset to this line
                let offset: usize = lines[..i].iter().map(|l| l.len() + 1).sum();
                return offset.min(text.len());
            }
        }

        0
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
