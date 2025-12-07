use std::path::Path;

use chrono::{DateTime, Utc};
use tree_sitter::Tree;

use super::{build_chunk, CodeChunk, MAX_CONTEXT_LINES};

const TARGET_CHUNK_LINES: usize = 50;
const MAX_CHUNK_LINES: usize = 200;

#[derive(Debug, Clone)]
struct SemanticSpan {
    start_line: usize,
    end_line: usize,
    is_container: bool,
    text: String,
}

impl SemanticSpan {
    fn line_count(&self) -> usize {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

fn collect_semantic_spans(
    node: &tree_sitter::Node,
    source: &str,
    language: &str,
    file_context: &str,
    spans: &mut Vec<SemanticSpan>,
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

            if end.row < start.row {
                continue;
            }

            let snippet = child.utf8_text(source.as_bytes()).unwrap_or_default();
            if snippet.trim().is_empty() {
                continue;
            }

            let context = extract_parent_context(node, source, language);
            let full_text = if context.is_empty() && file_context.is_empty() {
                snippet.to_string()
            } else if context.is_empty() {
                format!("{}{}", file_context, snippet)
            } else {
                format!("{}// Context: {}\n{}", file_context, context, snippet)
            };

            let is_container = is_container_node(kind);

            spans.push(SemanticSpan {
                start_line: start.row + 1,
                end_line: end.row + 1,
                is_container,
                text: full_text,
            });

            if is_container {
                collect_semantic_spans(&child, source, language, file_context, spans);
            }
        } else {
            collect_semantic_spans(&child, source, language, file_context, spans);
        }
    }
}

fn merge_small_spans(mut spans: Vec<SemanticSpan>, file_context: &str) -> Vec<SemanticSpan> {
    if spans.is_empty() {
        return spans;
    }

    spans.sort_by_key(|s| s.start_line);

    let mut filtered_spans: Vec<SemanticSpan> = Vec::new();
    for span in &spans {
        if span.is_container {
            let has_children = spans.iter().any(|other| {
                other.start_line > span.start_line
                    && other.end_line <= span.end_line
                    && !std::ptr::eq(span, other)
            });
            if !has_children {
                filtered_spans.push(span.clone());
            }
        } else {
            filtered_spans.push(span.clone());
        }
    }

    if filtered_spans.is_empty() {
        return filtered_spans;
    }

    let mut merged: Vec<SemanticSpan> = Vec::new();
    let mut current: Option<SemanticSpan> = None;

    for span in filtered_spans {
        match current.take() {
            None => current = Some(span),
            Some(mut c) => {
                let combined_lines = span.end_line.saturating_sub(c.start_line) + 1;
                let should_merge = combined_lines <= MAX_CHUNK_LINES
                    && c.line_count() < TARGET_CHUNK_LINES
                    && !span.is_container
                    && !c.is_container;

                if should_merge {
                    c.end_line = span.end_line;
                    let span_text_no_context =
                        if !file_context.is_empty() && span.text.starts_with(file_context) {
                            &span.text[file_context.len()..]
                        } else {
                            &span.text
                        };
                    c.text = format!("{}\n\n{}", c.text, span_text_no_context);
                    current = Some(c);
                } else {
                    merged.push(c);
                    current = Some(span);
                }
            }
        }
    }

    if let Some(c) = current {
        merged.push(c);
    }

    merged
}

pub fn chunk_with_tree(
    source: &str,
    path: &Path,
    tree: &Tree,
    language: &str,
    modified_at: DateTime<Utc>,
) -> Vec<CodeChunk> {
    let root = tree.root_node();
    let file_context = extract_file_context(source, &root, language);

    let mut spans = Vec::new();
    collect_semantic_spans(&root, source, language, &file_context, &mut spans);

    let merged_spans = merge_small_spans(spans, &file_context);

    merged_spans
        .into_iter()
        .map(|span| {
            build_chunk(
                path,
                &span.text,
                span.start_line,
                span.end_line,
                language.to_string(),
                modified_at,
            )
        })
        .collect()
}

fn extract_file_context(source: &str, root: &tree_sitter::Node, language: &str) -> String {
    let mut context_lines = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        if context_lines.len() >= MAX_CONTEXT_LINES {
            break;
        }

        let kind = node.kind();
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

fn extract_parent_context(parent: &tree_sitter::Node, source: &str, _language: &str) -> String {
    let kind = parent.kind();
    if !is_context_provider(kind) {
        return String::new();
    }

    let mut cursor = parent.walk();
    for child in parent.children(&mut cursor) {
        let child_kind = child.kind();
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

pub fn is_context_provider(kind: &str) -> bool {
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

pub fn is_container_node(kind: &str) -> bool {
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

pub fn kind_to_label(kind: &str) -> &'static str {
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

pub fn is_semantic_node(kind: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
