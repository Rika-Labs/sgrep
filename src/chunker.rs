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
    let mut cursor = root.walk();
    let mut chunks = Vec::new();
    for node in root.children(&mut cursor) {
        if !node.is_named() {
            continue;
        }
        if !is_semantic_node(node.kind()) {
            continue;
        }
        let start = node.start_position();
        let end = node.end_position();
        if end.row <= start.row {
            continue;
        }
        let snippet = node.utf8_text(source.as_bytes()).unwrap_or_default();
        if snippet.trim().is_empty() {
            continue;
        }
        chunks.push(build_chunk(
            path,
            snippet,
            start.row + 1,
            end.row + 1,
            language.to_string(),
            modified_at,
        ));
    }
    chunks
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
        text.truncate(MAX_SNIPPET_CHARS);
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
            LanguageKind::Yaml => {
                let lang = tree_sitter_yaml::language();
                Some(unsafe { std::mem::transmute(lang) })
            }
            LanguageKind::Toml => {
                let lang = tree_sitter_toml::language();
                Some(unsafe { std::mem::transmute(lang) })
            }
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
}
