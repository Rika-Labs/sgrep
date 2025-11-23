use std::path::Path;

use anyhow::Result;
use tree_sitter::{Language, Node, Parser};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Code,
    Documentation,
    Config,
    Data,
    Other,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: u32,
    pub end_line: u32,
}

pub struct ChunkedFile {
    pub file_type: FileType,
    pub chunks: Vec<Chunk>,
}

fn language_for_extension(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        _ => None,
    }
}

fn classify_file(path: &Path) -> FileType {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    return match ext.as_str() {
        "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "java" | "cs" | "cpp" | "c" | "h" => {
            FileType::Code
        }
        "md" | "mdx" | "txt" => FileType::Documentation,
        "json" | "yaml" | "yml" | "toml" => FileType::Config,
        "csv" | "tsv" => FileType::Data,
        _ => FileType::Other,
    };
}

pub fn chunk_file(path: &Path, contents: &str) -> Result<ChunkedFile> {
    let file_type = classify_file(path);
    if let Some(lang) = path
        .extension()
        .and_then(|s| s.to_str())
        .and_then(|ext| language_for_extension(ext))
    {
        let chunks = tree_chunks(contents, lang);
        if !chunks.is_empty() {
            return Ok(ChunkedFile { file_type, chunks });
        }
    }
    Ok(ChunkedFile {
        file_type,
        chunks: fallback_line_chunks(contents, 40, 8),
    })
}

fn tree_chunks(contents: &str, language: Language) -> Vec<Chunk> {
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let tree = match parser.parse(contents, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let root = tree.root_node();
    let mut out = Vec::new();
    collect_nodes(root, contents, &mut out);
    if out.is_empty() {
        return Vec::new();
    }
    out
}

fn collect_nodes(node: Node, source: &str, out: &mut Vec<Chunk>) {
    let kind = node.kind();
    if is_definition_node(kind) {
        if let Some(chunk) = to_chunk(&node, source) {
            out.push(chunk);
        }
    }
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_nodes(child, source, out);
        }
    }
}

fn is_definition_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_item"
            | "impl_item"
            | "trait_item"
            | "struct_item"
            | "enum_item"
            | "function_declaration"
            | "method_definition"
            | "class_declaration"
            | "arrow_function"
            | "lexical_declaration"
            | "generator_function_declaration"
            | "function_definition"
            | "class_definition"
    )
}

fn to_chunk(node: &Node, source: &str) -> Option<Chunk> {
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    if start_byte >= end_byte {
        return None;
    }
    let prefix = &source[..start_byte.min(source.len())];
    let start_line = prefix.lines().count() as u32 + 1;
    let snippet = &source[..end_byte.min(source.len())];
    let end_line = snippet.lines().count() as u32;
    Some(Chunk {
        start_byte,
        end_byte,
        start_line,
        end_line,
    })
}

fn fallback_line_chunks(contents: &str, max_lines: usize, overlap: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = contents.lines().collect();
    let mut idx = 0usize;
    let mut start_byte = 0usize;
    while idx < lines.len() {
        let end = (idx + max_lines).min(lines.len());
        let start_line = idx as u32 + 1;
        let end_line = end as u32;
        let mut end_byte = start_byte;
        for l in lines.iter().skip(idx).take(max_lines) {
            end_byte += l.len() + 1;
        }
        chunks.push(Chunk {
            start_byte,
            end_byte: end_byte.min(contents.len()),
            start_line,
            end_line,
        });
        if end == lines.len() {
            break;
        }
        let overlap_lines = overlap.min(max_lines);
        let advance = max_lines.saturating_sub(overlap_lines);
        for l in lines.iter().skip(idx).take(advance) {
            start_byte += l.len() + 1;
        }
        idx += advance;
    }
    if chunks.is_empty() && !contents.is_empty() {
        chunks.push(Chunk {
            start_byte: 0,
            end_byte: contents.len(),
            start_line: 1,
            end_line: lines.len() as u32,
        });
    }
    chunks
}
