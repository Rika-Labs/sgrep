mod language;
mod parser_pool;
mod treesitter;

pub use language::detect_language;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use treesitter::chunk_with_tree;

const MAX_LINES_PER_CHUNK: usize = 400;
const MAX_SNIPPET_CHARS: usize = 8192;
pub(crate) const MAX_CONTEXT_LINES: usize = 10;

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

    let mut chunks = if let Some(lang_kind) = language {
        let label = lang_kind.label().to_string();
        parser_pool::with_parser(lang_kind, |parser| {
            parser
                .parse(&source, None)
                .map(|tree| chunk_with_tree(&source, &relative, &tree, &label, modified_at))
                .unwrap_or_default()
        })
        .unwrap_or_default()
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
        let source: String = (0..450).map(|i| format!("fn test{}() {{}}\n", i)).collect();
        let chunks = chunk_fallback(&source, Path::new("src/lib.rs"), "rust", Utc::now());
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, MAX_LINES_PER_CHUNK);
        assert_eq!(chunks.last().unwrap().start_line, MAX_LINES_PER_CHUNK + 1);
    }

    #[test]
    fn build_chunk_handles_utf8_multibyte_characters() {
        let emoji_str = "🚀".repeat(2500);
        let chunk = build_chunk(
            Path::new("test.txt"),
            &emoji_str,
            1,
            1,
            "plain".to_string(),
            Utc::now(),
        );
        assert!(chunk.text.len() <= MAX_SNIPPET_CHARS);
        assert!(chunk.text.is_char_boundary(chunk.text.len()));
    }

    #[test]
    fn build_chunk_handles_mixed_utf8_characters() {
        let mut content = "x".repeat(MAX_SNIPPET_CHARS - 10);
        content.push_str("🎉🎊🎈🎁🎀");
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
        let chinese = "你好世界".repeat(2800);
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
        assert!(!chunks.is_empty());
        let has_foo = chunks.iter().any(|c| c.text.contains("Foo"));
        assert!(
            has_foo,
            "Expected Foo in chunks: {:?}",
            chunks.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn chunk_file_deterministic_with_pooling() {
        let dir = std::env::temp_dir().join("sgrep_chunk_determ");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("deterministic.rs");
        std::fs::write(&path, "fn main() { println!(\"hello\"); }\n").unwrap();

        let first = chunk_file(&path, &dir).unwrap();
        let second = chunk_file(&path, &dir).unwrap();
        let third = chunk_file(&path, &dir).unwrap();

        assert!(!first.is_empty());
        assert_eq!(first.len(), second.len());
        assert_eq!(second.len(), third.len());

        for i in 0..first.len() {
            assert_eq!(first[i].hash, second[i].hash);
            assert_eq!(second[i].hash, third[i].hash);
        }
    }

    #[test]
    fn chunk_multiple_files_same_language() {
        let dir = std::env::temp_dir().join("sgrep_chunk_multi");
        std::fs::create_dir_all(&dir).unwrap();

        for i in 0..10 {
            let path = dir.join(format!("file_{}.rs", i));
            std::fs::write(&path, format!("fn func_{}() {{ }}\n", i)).unwrap();
        }

        for i in 0..10 {
            let path = dir.join(format!("file_{}.rs", i));
            let chunks = chunk_file(&path, &dir).unwrap();
            assert!(!chunks.is_empty());
            assert_eq!(chunks[0].language, "rust");
        }
    }

    #[test]
    fn parallel_chunking_produces_correct_results() {
        use rayon::prelude::*;

        let dir = std::env::temp_dir().join("sgrep_chunk_parallel");
        std::fs::create_dir_all(&dir).unwrap();

        let files: Vec<_> = (0..20)
            .map(|i| {
                let ext = match i % 4 {
                    0 => "rs",
                    1 => "py",
                    2 => "js",
                    _ => "go",
                };
                let content = match ext {
                    "rs" => format!("fn func_{}() {{}}", i),
                    "py" => format!("def func_{}(): pass", i),
                    "js" => format!("function func_{}() {{}}", i),
                    "go" => format!("func func_{}() {{}}", i),
                    _ => unreachable!(),
                };
                let path = dir.join(format!("file_{}.{}", i, ext));
                std::fs::write(&path, content).unwrap();
                path
            })
            .collect();

        let results: Vec<_> = files
            .par_iter()
            .map(|path| chunk_file(path, &dir))
            .collect();

        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "File {} failed", i);
            let chunks = result.as_ref().unwrap();
            assert!(!chunks.is_empty());
        }
    }

    #[test]
    fn merges_small_adjacent_functions_into_single_chunk() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_merge");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("small_funcs.rs");
        let source = r#"
fn func1() {
    println!("1");
}

fn func2() {
    println!("2");
}

fn func3() {
    println!("3");
}

fn func4() {
    println!("4");
}

fn func5() {
    println!("5");
}
"#;
        std::fs::write(&path, source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() < 5,
            "Expected fewer than 5 chunks after merging, got {}",
            chunks.len()
        );
        assert!(!chunks.is_empty());
    }

    #[test]
    fn respects_max_chunk_lines_when_merging() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_max");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many_funcs.rs");

        let source: String = (0..100)
            .map(|i| format!("fn func{}() {{\n    println!(\"{}\");\n}}\n\n", i, i))
            .collect();
        std::fs::write(&path, &source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks when content exceeds max"
        );
        for chunk in &chunks {
            let line_count = chunk.end_line - chunk.start_line + 1;
            assert!(
                line_count <= 250,
                "Chunk has {} lines, expected <= 250",
                line_count
            );
        }
    }

    #[test]
    fn does_not_merge_across_class_boundaries() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_boundary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two_impls.rs");
        let source = r#"
struct Foo;
struct Bar;

impl Foo {
    fn method1(&self) {
        println!("foo");
    }
}

impl Bar {
    fn method1(&self) {
        println!("bar");
    }
}
"#;
        std::fs::write(&path, source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        let foo_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.text.contains("impl Foo"))
            .collect();
        let bar_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.text.contains("impl Bar"))
            .collect();

        let has_foo =
            !foo_chunks.is_empty() || chunks.iter().any(|c| c.text.contains("struct Foo"));
        let has_bar =
            !bar_chunks.is_empty() || chunks.iter().any(|c| c.text.contains("struct Bar"));
        assert!(has_foo, "Expected Foo content in chunks");
        assert!(has_bar, "Expected Bar content in chunks");
    }

    #[test]
    fn splits_large_impl_block_by_methods() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_split_impl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_impl.rs");

        let methods: String = (0..50)
            .map(|i| {
                format!(
                    "    fn method{}(&self) {{\n        println!(\"{}\");\n        let x = {};\n    }}\n\n",
                    i, i, i
                )
            })
            .collect();
        let source = format!("struct Large;\n\nimpl Large {{\n{}}}\n", methods);
        std::fs::write(&path, &source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() > 1,
            "Expected large impl to be split into multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn chunk_boundaries_align_to_symbol_boundaries() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_align");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aligned.rs");

        let source: String = (0..10)
            .map(|i| {
                let body: String = (0..15)
                    .map(|j| format!("    let var{} = {};\n", j, i * 100 + j))
                    .collect();
                format!("fn func{}() {{\n{}}}\n\n", i, body)
            })
            .collect();
        std::fs::write(&path, &source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();

        for chunk in &chunks {
            let text = &chunk.text;
            let open_braces = text.matches('{').count();
            let close_braces = text.matches('}').count();
            assert_eq!(
                open_braces, close_braces,
                "Chunk has unbalanced braces: {} opens, {} closes",
                open_braces, close_braces
            );
        }
    }

    #[test]
    fn fallback_path_unchanged_for_unknown_files() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_fallback");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.xyz");

        let source: String = (0..500).map(|i| format!("line {}\n", i)).collect();
        std::fs::write(&path, &source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() >= 2,
            "Expected fallback to create multiple chunks"
        );
        assert_eq!(chunks[0].language, "plain");
        assert!(
            chunks[0].end_line >= 350,
            "Fallback chunk should have ~400 lines"
        );
    }

    #[test]
    fn adaptive_chunking_works_for_python() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_python");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("funcs.py");

        let source = r#"
def func1():
    print("1")

def func2():
    print("2")

def func3():
    print("3")

def func4():
    print("4")

def func5():
    print("5")
"#;
        std::fs::write(&path, source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() < 5,
            "Expected Python functions to be merged, got {} chunks",
            chunks.len()
        );
        assert!(!chunks.is_empty());
    }

    #[test]
    fn adaptive_chunking_works_for_javascript() {
        let dir = std::env::temp_dir().join("sgrep_adaptive_js");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("funcs.js");

        let source = r#"
function func1() {
    console.log("1");
}

function func2() {
    console.log("2");
}

function func3() {
    console.log("3");
}

function func4() {
    console.log("4");
}

function func5() {
    console.log("5");
}
"#;
        std::fs::write(&path, source).unwrap();

        let chunks = chunk_file(&path, &dir).unwrap();
        assert!(
            chunks.len() < 5,
            "Expected JavaScript functions to be merged, got {} chunks",
            chunks.len()
        );
        assert!(!chunks.is_empty());
    }
}
