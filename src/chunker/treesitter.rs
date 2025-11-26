use std::path::Path;

use chrono::{DateTime, Utc};
use tree_sitter::Tree;

use super::{build_chunk, CodeChunk, MAX_CONTEXT_LINES};

pub fn chunk_with_tree(
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

/// Check if a node is a container that can have nested semantic nodes
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

/// Convert node kind to a human-readable label
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
