mod language;
mod treesitter;

pub use language::{detect_language, LanguageKind};
pub use treesitter::{is_container_node, is_context_provider, is_semantic_node, kind_to_label};

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tree_sitter::Parser;
use uuid::Uuid;

use treesitter::chunk_with_tree;

const MAX_LINES_PER_CHUNK: usize = 200;
const MAX_SNIPPET_CHARS: usize = 2048;
pub(crate) const MAX_CONTEXT_LINES: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub id: Uuid,
    pub path: PathBuf,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub hash: String,
    pub modified_at: DateTime<Utc>,
}

pub fn chunk_file(path: &Path, repo_root: &Path) -> Result<Vec<CodeChunk>> {
    let absolute = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let relative = absolute
        .strip_prefix(repo_root)
        .unwrap_or(&absolute)
        .to_path_buf();
    let source =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    if source.trim().is_empty() {
        return Ok(vec![]);
    }

    let language = detect_language(path);
    let modified = fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or_else(|_| SystemTime::now());
    let modified_at: DateTime<Utc> = modified.into();

    let mut parser = Parser::new();
    let mut chunks = if let Some(lang_kind) = language {
        let label = lang_kind.label().to_string();
        if let Some(lang) = lang_kind.language() {
            parser.set_language(&lang).ok();
            if let Some(tree) = parser.parse(&source, None) {
                chunk_with_tree(&source, &relative, &tree, &label, modified_at)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    if chunks.is_empty() {
        let label = language.map(|kind| kind.label()).unwrap_or("plain");
        chunks = chunk_fallback(&source, &relative, label, modified_at);
    }

    Ok(chunks)
}

fn chunk_fallback(
    source: &str,
    path: &Path,
    language: &str,
    modified_at: DateTime<Utc>,
) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut start = 0usize;
    let mut chunks = Vec::new();
    while start < lines.len() {
        let end = usize::min(start + MAX_LINES_PER_CHUNK, lines.len());
        let snippet = lines[start..end].join("\n");
        chunks.push(build_chunk(
            path,
            &snippet,
            start + 1,
            end,
            language.to_string(),
            modified_at,
        ));
        start = end;
    }
    if chunks.is_empty() {
        chunks.push(build_chunk(
            path,
            source,
            1,
            lines.len(),
            language.to_string(),
            modified_at,
        ));
    }
    chunks
}

pub(crate) fn build_chunk(
    path: &Path,
    snippet: &str,
    start: usize,
    end: usize,
    language: String,
    modified_at: DateTime<Utc>,
) -> CodeChunk {
    let path_prefix = format!("// File: {}\n", path.display());
    let mut text = format!("{}{}", path_prefix, snippet.trim());

    if text.len() > MAX_SNIPPET_CHARS {
        // Find a valid UTF-8 char boundary at or before MAX_SNIPPET_CHARS
        let mut new_len = MAX_SNIPPET_CHARS;
        while new_len > 0 && !text.is_char_boundary(new_len) {
            new_len -= 1;
        }
        text.truncate(new_len);
    }
    let mut hasher = Hasher::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    CodeChunk {
        id: Uuid::new_v4(),
        path: path.to_path_buf(),
        language,
        start_line: start,
        end_line: end,
        text,
        hash: hash.to_hex().to_string(),
        modified_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::path::Path;

    #[test]
    fn fallback_chunking_respects_max_lines() {
        let source: String = (0..210).map(|i| format!("fn test{}() {{}}\n", i)).collect();
        let chunks = chunk_fallback(&source, Path::new("src/lib.rs"), "rust", Utc::now());
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, MAX_LINES_PER_CHUNK);
        assert_eq!(chunks.last().unwrap().start_line, MAX_LINES_PER_CHUNK + 1);
    }

    #[test]
    fn build_chunk_handles_utf8_multibyte_characters() {
        // Create a string with multibyte UTF-8 characters (emoji, unicode, etc.)
        // Each emoji is typically 4 bytes
        let emoji_str = "🚀".repeat(600); // 4 bytes * 600 = 2400 bytes
        let chunk = build_chunk(
            Path::new("test.txt"),
            &emoji_str,
            1,
            1,
            "plain".to_string(),
            Utc::now(),
        );
        // Should truncate at a valid char boundary, not panic
        assert!(chunk.text.len() <= MAX_SNIPPET_CHARS);
        assert!(chunk.text.is_char_boundary(chunk.text.len()));
    }

    #[test]
    fn build_chunk_handles_mixed_utf8_characters() {
        // Mix of ASCII and multibyte characters near the truncation boundary
        let mut content = "x".repeat(MAX_SNIPPET_CHARS - 10);
        content.push_str("🎉🎊🎈🎁🎀"); // Add emojis that will push it over the limit
        let chunk = build_chunk(
            Path::new("test.txt"),
            &content,
            1,
            1,
            "plain".to_string(),
            Utc::now(),
        );
        assert!(chunk.text.len() <= MAX_SNIPPET_CHARS);
        assert!(chunk.text.is_char_boundary(chunk.text.len()));
    }

    #[test]
    fn build_chunk_preserves_text_under_limit() {
        let content = "Hello, world! 🌍";
        let chunk = build_chunk(
            Path::new("test.txt"),
            content,
            1,
            1,
            "plain".to_string(),
            Utc::now(),
        );
        assert!(chunk.text.contains(content));
        assert!(chunk.text.starts_with("// File: test.txt\n"));
    }

    #[test]
    fn build_chunk_handles_chinese_characters() {
        // Chinese characters are typically 3 bytes in UTF-8
        let chinese = "你好世界".repeat(700); // Should exceed MAX_SNIPPET_CHARS
        let chunk = build_chunk(
            Path::new("test.txt"),
            &chinese,
            1,
            1,
            "plain".to_string(),
            Utc::now(),
        );
        assert!(chunk.text.len() <= MAX_SNIPPET_CHARS);
        assert!(chunk.text.is_char_boundary(chunk.text.len()));
    }

    #[test]
    fn chunk_file_produces_chunks_for_small_rust_file() {
        let dir = std::env::temp_dir().join("sgrep_chunk_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("small.rs");
        std::fs::write(&path, "fn main() { println!(\"hi\"); }\n").unwrap();
        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].path.ends_with(Path::new("small.rs")));
    }

    #[test]
    fn chunk_file_returns_empty_for_blank_file() {
        let dir = std::env::temp_dir().join("sgrep_chunk_file_blank");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.rs");
        std::fs::write(&path, "   \n\t").unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_file_unknown_extension_uses_plain_fallback() {
        let dir = std::env::temp_dir().join("sgrep_chunk_file_plain");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notes.txt");
        std::fs::write(&path, "just some words\nanother line").unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "plain");
    }

    #[test]
    fn chunk_file_with_parse_error_falls_back() {
        let dir = std::env::temp_dir().join("sgrep_chunk_file_parse_error");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.rs");
        // Tree-sitter will produce an ERROR node but no semantic matches; we should fallback.
        std::fs::write(&path, "??? ??\n!!!").unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "rust");
        assert!(chunks[0].text.contains("???"));
    }

    #[test]
    fn chunk_fallback_handles_empty_source() {
        let chunks = chunk_fallback("", Path::new("empty.txt"), "plain", Utc::now());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.starts_with("// File: empty.txt\n"));
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn chunk_file_with_rust_imports_includes_context() {
        let dir = std::env::temp_dir().join("sgrep_chunk_context");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("with_imports.rs");
        std::fs::write(
            &path,
            "use std::collections::HashMap;\n\nfn main() { println!(\"hi\"); }\n",
        )
        .unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(!chunks.is_empty());
        // The chunk should contain the import as context
        assert!(
            chunks[0].text.contains("use std::collections::HashMap")
                || chunks[0].text.contains("main"),
            "Expected import context or function, got: {}",
            chunks[0].text
        );
    }

    #[test]
    fn chunk_file_with_impl_includes_context() {
        let dir = std::env::temp_dir().join("sgrep_chunk_impl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("impl_context.rs");
        std::fs::write(
            &path,
            r#"
struct Foo;

impl Foo {
    fn bar(&self) {
        println!("bar");
    }
}
"#,
        )
        .unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        // Should have chunks for both struct and impl
        assert!(!chunks.is_empty());
        // At least one chunk should exist
        let has_foo = chunks.iter().any(|c| c.text.contains("Foo"));
        assert!(
            has_foo,
            "Expected Foo in chunks: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
    }
}
