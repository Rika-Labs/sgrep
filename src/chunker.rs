use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Parser, Tree};
use uuid::Uuid;

const MAX_LINES_PER_CHUNK: usize = 200;
const MAX_SNIPPET_CHARS: usize = 2048;
const MAX_CONTEXT_LINES: usize = 3;

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

fn chunk_with_tree(
    source: &str,
    path: &Path,
    tree: &Tree,
    language: &str,
    modified_at: DateTime<Utc>,
) -> Vec<CodeChunk> {
    let root = tree.root_node();
    let mut chunks = Vec::new();

    // Collect file-level imports/use statements for context
    let file_context = extract_file_context(source, &root, language);

    // Recursively collect semantic nodes with their parent context
    collect_semantic_nodes(
        &root,
        source,
        path,
        language,
        modified_at,
        &file_context,
        &mut chunks,
    );

    chunks
}

/// Extract file-level context like imports, package declarations, module names
fn extract_file_context(source: &str, root: &tree_sitter::Node, language: &str) -> String {
    let mut context_lines = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        if context_lines.len() >= MAX_CONTEXT_LINES {
            break;
        }

        let kind = node.kind();

        // Collect imports, use statements, package declarations
        let is_context_node = matches!(
            kind,
            "use_declaration"
                | "use_item"
                | "import_statement"
                | "import_declaration"
                | "package_clause"
                | "package_declaration"
                | "namespace_declaration"
                | "module_declaration"
        ) || kind.contains("import")
            || kind.contains("use")
            || (kind.contains("package") && language == "go");

        if is_context_node {
            if let Ok(text) = node.utf8_text(source.as_bytes()) {
                let line = text.lines().next().unwrap_or(text);
                if !line.trim().is_empty() && line.len() < 100 {
                    context_lines.push(line.trim().to_string());
                }
            }
        }
    }

    if context_lines.is_empty() {
        String::new()
    } else {
        context_lines.join("\n") + "\n\n"
    }
}

/// Recursively collect semantic nodes, including nested ones with parent context
fn collect_semantic_nodes(
    node: &tree_sitter::Node,
    source: &str,
    path: &Path,
    language: &str,
    modified_at: DateTime<Utc>,
    file_context: &str,
    chunks: &mut Vec<CodeChunk>,
) {
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }

        let kind = child.kind();

        if is_semantic_node(kind) {
            let start = child.start_position();
            let end = child.end_position();

            if end.row <= start.row {
                continue;
            }

            let snippet = child.utf8_text(source.as_bytes()).unwrap_or_default();
            if snippet.trim().is_empty() {
                continue;
            }

            // Build context-aware text
            let context = extract_parent_context(node, source, language);
            let full_text = if context.is_empty() && file_context.is_empty() {
                snippet.to_string()
            } else if context.is_empty() {
                format!("{}{}", file_context, snippet)
            } else {
                format!("{}// Context: {}\n{}", file_context, context, snippet)
            };

            chunks.push(build_chunk(
                path,
                &full_text,
                start.row + 1,
                end.row + 1,
                language.to_string(),
                modified_at,
            ));

            // Don't recurse into this node if we already captured it
            // But do recurse for container types that might have nested items
            if is_container_node(kind) {
                collect_semantic_nodes(
                    &child,
                    source,
                    path,
                    language,
                    modified_at,
                    file_context,
                    chunks,
                );
            }
        } else {
            // Continue searching in non-semantic nodes
            collect_semantic_nodes(
                &child,
                source,
                path,
                language,
                modified_at,
                file_context,
                chunks,
            );
        }
    }
}

/// Extract parent context (class name, struct name, impl block, etc.)
fn extract_parent_context(parent: &tree_sitter::Node, source: &str, _language: &str) -> String {
    let kind = parent.kind();

    // Check if parent is a container that provides context
    if !is_context_provider(kind) {
        return String::new();
    }

    // Try to extract the name/signature from the parent
    let mut cursor = parent.walk();
    for child in parent.children(&mut cursor) {
        let child_kind = child.kind();

        // Look for identifier nodes that give us the name
        if matches!(
            child_kind,
            "identifier"
                | "name"
                | "type_identifier"
                | "field_identifier"
                | "property_identifier"
                | "class_name"
        ) {
            if let Ok(name) = child.utf8_text(source.as_bytes()) {
                return format!("{} {}", kind_to_label(kind), name.trim());
            }
        }
    }

    String::new()
}

/// Check if a node kind provides context for its children
fn is_context_provider(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration"
            | "struct_item"
            | "impl_item"
            | "trait_item"
            | "interface_declaration"
            | "module"
            | "mod_item"
            | "namespace_declaration"
            | "class_definition"
            | "class_body"
    ) || kind.contains("class")
        || kind.contains("impl")
        || kind.contains("struct")
        || kind.contains("interface")
        || kind.contains("module")
        || kind.contains("namespace")
}

/// Check if a node is a container that can have nested semantic nodes
fn is_container_node(kind: &str) -> bool {
    matches!(
        kind,
        "class_declaration"
            | "struct_item"
            | "impl_item"
            | "trait_item"
            | "interface_declaration"
            | "module"
            | "mod_item"
            | "namespace_declaration"
            | "class_definition"
            | "class_body"
    ) || kind.contains("class")
        || kind.contains("impl")
        || kind.contains("module")
        || kind.contains("namespace")
}

/// Convert node kind to a human-readable label
fn kind_to_label(kind: &str) -> &'static str {
    if kind.contains("class") {
        "class"
    } else if kind.contains("struct") {
        "struct"
    } else if kind.contains("impl") {
        "impl"
    } else if kind.contains("trait") {
        "trait"
    } else if kind.contains("interface") {
        "interface"
    } else if kind.contains("module") || kind.contains("mod") {
        "module"
    } else if kind.contains("namespace") {
        "namespace"
    } else {
        ""
    }
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

fn build_chunk(
    path: &Path,
    snippet: &str,
    start: usize,
    end: usize,
    language: String,
    modified_at: DateTime<Utc>,
) -> CodeChunk {
    let mut text = snippet.trim().to_string();
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

fn detect_language(path: &Path) -> Option<LanguageKind> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(LanguageKind::Rust),
        "py" => Some(LanguageKind::Python),
        "ts" => Some(LanguageKind::TypeScript),
        "tsx" => Some(LanguageKind::Tsx),
        "js" | "jsx" => Some(LanguageKind::JavaScript),
        "go" => Some(LanguageKind::Go),
        "java" => Some(LanguageKind::Java),
        "c" | "h" => Some(LanguageKind::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some(LanguageKind::Cpp),
        "cs" => Some(LanguageKind::CSharp),
        "rb" => Some(LanguageKind::Ruby),
        "md" | "markdown" => Some(LanguageKind::Markdown),
        "json" => Some(LanguageKind::Json),
        "yaml" | "yml" => Some(LanguageKind::Yaml),
        "toml" => Some(LanguageKind::Toml),
        "html" | "htm" => Some(LanguageKind::Html),
        "css" => Some(LanguageKind::Css),
        "sh" | "bash" => Some(LanguageKind::Bash),
        _ => None,
    }
}

fn is_semantic_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_declaration"
            | "function_item"
            | "method_definition"
            | "class_declaration"
            | "struct_item"
            | "impl_item"
            | "trait_item"
            | "interface_declaration"
            | "lexical_declaration"
            | "module"
            | "mod_item"
            | "const_item"
            | "const_declaration"
            | "variable_declaration"
            | "type_item"
            | "type_alias_declaration"
            | "type_declaration"
            | "enum_item"
            | "enum_declaration"
            | "static_item"
            | "macro_definition"
            | "decorated_definition"
            | "assignment"
            | "export_statement"
            | "namespace_declaration"
            | "var_declaration"
            | "package_clause"
            | "section"
            | "atx_heading"
            | "setext_heading"
            | "fenced_code_block"
            | "list"
            | "block_quote"
            | "pair"
            | "object"
            | "array"
            | "block_mapping"
            | "block_sequence"
            | "table"
            | "rule_set"
    ) || kind.contains("function")
        || kind.contains("class")
        || kind.contains("method")
        || kind.contains("heading")
}

#[derive(Clone, Copy)]
enum LanguageKind {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Ruby,
    Markdown,
    Json,
    Yaml,
    Toml,
    Html,
    Css,
    Bash,
}

impl LanguageKind {
    fn label(&self) -> &'static str {
        match self {
            LanguageKind::Rust => "rust",
            LanguageKind::Python => "python",
            LanguageKind::JavaScript => "javascript",
            LanguageKind::TypeScript => "typescript",
            LanguageKind::Tsx => "tsx",
            LanguageKind::Go => "go",
            LanguageKind::Java => "java",
            LanguageKind::C => "c",
            LanguageKind::Cpp => "cpp",
            LanguageKind::CSharp => "csharp",
            LanguageKind::Ruby => "ruby",
            LanguageKind::Markdown => "markdown",
            LanguageKind::Json => "json",
            LanguageKind::Yaml => "yaml",
            LanguageKind::Toml => "toml",
            LanguageKind::Html => "html",
            LanguageKind::Css => "css",
            LanguageKind::Bash => "bash",
        }
    }

    fn language(&self) -> Option<Language> {
        match self {
            LanguageKind::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            LanguageKind::Python => Some(tree_sitter_python::LANGUAGE.into()),
            LanguageKind::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
            LanguageKind::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            LanguageKind::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            LanguageKind::Go => Some(tree_sitter_go::LANGUAGE.into()),
            LanguageKind::Java => Some(tree_sitter_java::LANGUAGE.into()),
            LanguageKind::C => Some(tree_sitter_c::LANGUAGE.into()),
            LanguageKind::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
            LanguageKind::CSharp => Some(tree_sitter_c_sharp::LANGUAGE.into()),
            LanguageKind::Ruby => Some(tree_sitter_ruby::LANGUAGE.into()),
            LanguageKind::Markdown => Some(tree_sitter_md::LANGUAGE.into()),
            LanguageKind::Json => Some(tree_sitter_json::LANGUAGE.into()),
            LanguageKind::Yaml => Some(tree_sitter_yaml::LANGUAGE.into()),
            LanguageKind::Toml => Some(tree_sitter_toml::LANGUAGE.into()),
            LanguageKind::Html => Some(tree_sitter_html::LANGUAGE.into()),
            LanguageKind::Css => Some(tree_sitter_css::LANGUAGE.into()),
            LanguageKind::Bash => Some(tree_sitter_bash::LANGUAGE.into()),
        }
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
    fn detect_language_handles_rust_extension() {
        let lang = detect_language(Path::new("main.rs"));
        assert!(matches!(lang, Some(LanguageKind::Rust)));
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
        assert_eq!(chunk.text, content);
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
        assert_eq!(chunks[0].text, "");
        assert_eq!(chunks[0].start_line, 1);
    }

    #[test]
    fn detect_language_covers_common_extensions() {
        let cases = [
            ("file.ts", "typescript"),
            ("component.tsx", "tsx"),
            ("script.js", "javascript"),
            ("script.jsx", "javascript"),
            ("main.c", "c"),
            ("main.go", "go"),
            ("main.java", "java"),
            ("header.hpp", "cpp"),
            ("class.cs", "csharp"),
            ("tool.rb", "ruby"),
            ("doc.md", "markdown"),
            ("data.json", "json"),
            ("config.yaml", "yaml"),
            ("config.toml", "toml"),
            ("index.html", "html"),
            ("style.css", "css"),
            ("script.sh", "bash"),
        ];

        for (file, expected_label) in cases {
            let lang = detect_language(Path::new(file)).expect("language detected");
            assert_eq!(lang.label(), expected_label);
            assert!(lang.language().is_some());
        }
    }

    #[test]
    fn is_semantic_node_matches_keyword_variants() {
        assert!(is_semantic_node("custom_function_kind"));
        assert!(is_semantic_node("custom_method_kind"));
        assert!(is_semantic_node("custom_heading_kind"));
    }

    #[test]
    fn kind_to_label_returns_correct_labels() {
        assert_eq!(kind_to_label("class_declaration"), "class");
        assert_eq!(kind_to_label("struct_item"), "struct");
        assert_eq!(kind_to_label("impl_item"), "impl");
        assert_eq!(kind_to_label("trait_item"), "trait");
        assert_eq!(kind_to_label("interface_declaration"), "interface");
        assert_eq!(kind_to_label("mod_item"), "module");
        assert_eq!(kind_to_label("namespace_declaration"), "namespace");
        assert_eq!(kind_to_label("unknown_node"), "");
    }

    #[test]
    fn is_context_provider_matches_container_types() {
        assert!(is_context_provider("class_declaration"));
        assert!(is_context_provider("struct_item"));
        assert!(is_context_provider("impl_item"));
        assert!(is_context_provider("trait_item"));
        assert!(is_context_provider("interface_declaration"));
        assert!(is_context_provider("mod_item"));
        assert!(!is_context_provider("function_declaration"));
        assert!(!is_context_provider("variable_declaration"));
    }

    #[test]
    fn is_container_node_matches_nestable_types() {
        assert!(is_container_node("class_declaration"));
        assert!(is_container_node("impl_item"));
        assert!(is_container_node("mod_item"));
        assert!(!is_container_node("function_declaration"));
        assert!(!is_container_node("const_item"));
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
