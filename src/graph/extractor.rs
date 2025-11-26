use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use tree_sitter::{Parser, Tree};
use uuid::Uuid;

use super::symbols::{Edge, EdgeKind, ImportRelation, Symbol, SymbolKind, is_container_kind};

pub struct SymbolExtractor {
    parsers: HashMap<String, Parser>,
}

impl SymbolExtractor {
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
        }
    }

    pub fn extract_from_file(
        &mut self,
        path: &Path,
        source: &str,
        language: &str,
    ) -> Result<(Vec<Symbol>, Vec<Edge>, Vec<ImportRelation>)> {
        let parser = self.get_or_create_parser(language)?;
        let tree = parser.parse(source, None);

        match tree {
            Some(tree) => self.extract_from_tree(path, source, language, &tree),
            None => Ok((vec![], vec![], vec![])),
        }
    }

    fn get_or_create_parser(&mut self, language: &str) -> Result<&mut Parser> {
        if !self.parsers.contains_key(language) {
            let mut parser = Parser::new();
            if let Some(lang) = get_tree_sitter_language(language) {
                parser.set_language(&lang).ok();
            }
            self.parsers.insert(language.to_string(), parser);
        }
        Ok(self.parsers.get_mut(language).unwrap())
    }

    fn extract_from_tree(
        &self,
        path: &Path,
        source: &str,
        language: &str,
        tree: &Tree,
    ) -> Result<(Vec<Symbol>, Vec<Edge>, Vec<ImportRelation>)> {
        let mut symbols = Vec::new();
        let mut edges = Vec::new();
        let mut imports = Vec::new();

        let root = tree.root_node();
        let mut cursor = root.walk();

        for node in root.children(&mut cursor) {
            if let Some(import) = self.extract_import(&node, source, path, language) {
                imports.push(import);
            }
        }

        self.collect_symbols(
            &root,
            source,
            path,
            language,
            None,
            &mut symbols,
            &mut edges,
        );

        Ok((symbols, edges, imports))
    }

    fn collect_symbols(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        language: &str,
        parent_id: Option<Uuid>,
        symbols: &mut Vec<Symbol>,
        edges: &mut Vec<Edge>,
    ) {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if !child.is_named() {
                continue;
            }

            let kind = child.kind();

            if let Some(symbol_kind) = classify_node_kind(kind, language) {
                if let Some(symbol) = self.extract_symbol(&child, source, path, language, symbol_kind, parent_id) {
                    let symbol_id = symbol.id;

                    if let Some(parent) = parent_id {
                        edges.push(Edge {
                            source_id: parent,
                            target_id: symbol_id,
                            kind: EdgeKind::Contains,
                            metadata: None,
                        });
                    }

                    symbols.push(symbol);

                    if is_container_kind(symbol_kind) {
                        self.collect_symbols(&child, source, path, language, Some(symbol_id), symbols, edges);
                    }

                    self.extract_calls(&child, source, symbol_id, edges);
                }
            } else {
                self.collect_symbols(&child, source, path, language, parent_id, symbols, edges);
            }
        }
    }

    fn extract_symbol(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        language: &str,
        kind: SymbolKind,
        parent_id: Option<Uuid>,
    ) -> Option<Symbol> {
        let name = self.extract_symbol_name(node, source, language)?;
        if name.is_empty() || name.len() > 200 {
            return None;
        }

        let start = node.start_position();
        let end = node.end_position();

        let qualified_name = if let Some(_parent) = parent_id {
            name.clone()
        } else {
            name.clone()
        };

        let signature = node
            .utf8_text(source.as_bytes())
            .ok()?
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .chars()
            .take(200)
            .collect();

        Some(Symbol {
            id: Uuid::new_v4(),
            name,
            qualified_name,
            kind,
            file_path: path.to_path_buf(),
            start_line: start.row + 1,
            end_line: end.row + 1,
            language: language.to_string(),
            signature,
            parent_id,
            chunk_id: None,
        })
    }

    fn extract_symbol_name(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        language: &str,
    ) -> Option<String> {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            let child_kind = child.kind();

            if matches!(
                child_kind,
                "identifier"
                    | "name"
                    | "type_identifier"
                    | "property_identifier"
                    | "field_identifier"
                    | "variable_name"
                    | "function_name"
                    | "class_name"
                    | "method_name"
                    | "constant_name"
            ) {
                if let Ok(name) = child.utf8_text(source.as_bytes()) {
                    let name = name.trim();
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }

            if child_kind == "call_expression" && language == "typescript" {
                continue;
            }
        }

        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            let first_line = text.lines().next()?;
            let patterns = [
                ("fn ", "("),
                ("function ", "("),
                ("def ", "("),
                ("func ", "("),
                ("class ", " "),
                ("class ", "{"),
                ("struct ", " "),
                ("struct ", "{"),
                ("interface ", " "),
                ("interface ", "{"),
                ("trait ", " "),
                ("trait ", "{"),
                ("enum ", " "),
                ("enum ", "{"),
                ("type ", " "),
                ("const ", " "),
                ("let ", " "),
                ("var ", " "),
            ];

            for (start, end) in patterns {
                if let Some(idx) = first_line.find(start) {
                    let after = &first_line[idx + start.len()..];
                    let name_end = after
                        .find(end)
                        .or_else(|| after.find('('))
                        .or_else(|| after.find(':'))
                        .or_else(|| after.find('<'))
                        .unwrap_or(after.len());
                    let name = after[..name_end].trim();
                    if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        return Some(name.to_string());
                    }
                }
            }
        }

        None
    }

    fn extract_import(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        path: &Path,
        language: &str,
    ) -> Option<ImportRelation> {
        let kind = node.kind();

        let is_import = matches!(
            kind,
            "use_declaration"
                | "use_item"
                | "import_statement"
                | "import_declaration"
                | "include_directive"
                | "require_expression"
        ) || kind.contains("import")
            || kind.contains("use")
            || kind.contains("require");

        if !is_import {
            return None;
        }

        let text = node.utf8_text(source.as_bytes()).ok()?;
        let target = self.extract_import_target(text, language)?;

        Some(ImportRelation {
            source_file: path.to_path_buf(),
            target_path: target,
            alias: self.extract_import_alias(text, language),
            line: node.start_position().row + 1,
            is_type_only: text.contains("import type") || text.contains("type "),
        })
    }

    fn extract_import_target(&self, text: &str, language: &str) -> Option<String> {
        let text = text.trim();

        match language {
            "rust" => {
                if text.starts_with("use ") {
                    let path = text
                        .trim_start_matches("use ")
                        .trim_end_matches(';')
                        .split("::")
                        .collect::<Vec<_>>()
                        .join("::");
                    return Some(path);
                }
            }
            "typescript" | "javascript" | "tsx" => {
                if let Some(from_idx) = text.find("from ") {
                    let after = &text[from_idx + 5..];
                    let path = after
                        .trim()
                        .trim_matches(|c| c == '\'' || c == '"' || c == ';');
                    return Some(path.to_string());
                }
                if let Some(req_idx) = text.find("require(") {
                    let after = &text[req_idx + 8..];
                    if let Some(end) = after.find(')') {
                        let path = after[..end].trim_matches(|c| c == '\'' || c == '"');
                        return Some(path.to_string());
                    }
                }
            }
            "python" => {
                if text.starts_with("from ") {
                    let parts: Vec<_> = text.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return Some(parts[1].to_string());
                    }
                } else if text.starts_with("import ") {
                    let module = text.trim_start_matches("import ").split_whitespace().next()?;
                    return Some(module.to_string());
                }
            }
            "go" => {
                if let Some(quote_start) = text.find('"') {
                    if let Some(quote_end) = text[quote_start + 1..].find('"') {
                        return Some(text[quote_start + 1..quote_start + 1 + quote_end].to_string());
                    }
                }
            }
            "java" => {
                if text.starts_with("import ") {
                    let path = text
                        .trim_start_matches("import ")
                        .trim_start_matches("static ")
                        .trim_end_matches(';')
                        .trim();
                    return Some(path.to_string());
                }
            }
            "c" | "cpp" => {
                if text.contains("#include") {
                    let header = text
                        .trim_start_matches("#include")
                        .trim()
                        .trim_matches(|c| c == '<' || c == '>' || c == '"');
                    return Some(header.to_string());
                }
            }
            _ => {}
        }

        None
    }

    fn extract_import_alias(&self, text: &str, language: &str) -> Option<String> {
        let text = text.trim();

        match language {
            "rust" => {
                if let Some(as_idx) = text.find(" as ") {
                    let after = &text[as_idx + 4..];
                    let alias = after.trim_end_matches(';').trim();
                    if !alias.is_empty() {
                        return Some(alias.to_string());
                    }
                }
            }
            "typescript" | "javascript" | "tsx" => {
                if let Some(as_idx) = text.find(" as ") {
                    let after = &text[as_idx + 4..];
                    let end = after.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(after.len());
                    let alias = after[..end].trim();
                    if !alias.is_empty() {
                        return Some(alias.to_string());
                    }
                }
            }
            "python" => {
                if let Some(as_idx) = text.find(" as ") {
                    let after = &text[as_idx + 4..];
                    let alias = after.split_whitespace().next()?;
                    return Some(alias.to_string());
                }
            }
            "go" => {
                let parts: Vec<_> = text.split_whitespace().collect();
                if parts.len() >= 2 && !parts[1].starts_with('"') {
                    return Some(parts[1].to_string());
                }
            }
            _ => {}
        }

        None
    }

    fn extract_calls(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        caller_id: Uuid,
        edges: &mut Vec<Edge>,
    ) {
        let _cursor = node.walk();
        self.collect_call_expressions(&node.walk().node(), source, caller_id, edges, &mut HashSet::new());
    }

    fn collect_call_expressions(
        &self,
        node: &tree_sitter::Node,
        source: &str,
        caller_id: Uuid,
        edges: &mut Vec<Edge>,
        seen: &mut HashSet<String>,
    ) {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            let kind = child.kind();

            if matches!(kind, "call_expression" | "method_call_expression" | "function_call") {
                if let Some(callee_name) = self.extract_callee_name(&child, source) {
                    if seen.insert(callee_name.clone()) {
                        edges.push(Edge {
                            source_id: caller_id,
                            target_id: Uuid::nil(),
                            kind: EdgeKind::Calls,
                            metadata: Some(callee_name),
                        });
                    }
                }
            }

            self.collect_call_expressions(&child, source, caller_id, edges, seen);
        }
    }

    fn extract_callee_name(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            let kind = child.kind();

            if matches!(
                kind,
                "identifier"
                    | "property_identifier"
                    | "field_identifier"
                    | "member_expression"
                    | "scoped_identifier"
            ) {
                if let Ok(name) = child.utf8_text(source.as_bytes()) {
                    let name = name.trim();
                    if !name.is_empty() && name.len() < 100 {
                        return Some(name.to_string());
                    }
                }
            }

            if kind == "member_expression" || kind == "field_expression" {
                return self.extract_callee_name(&child, source);
            }
        }

        None
    }
}

impl Default for SymbolExtractor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn classify_node_kind(kind: &str, _language: &str) -> Option<SymbolKind> {
    match kind {
        "function_declaration" | "function_item" | "function_definition" | "arrow_function" => {
            Some(SymbolKind::Function)
        }
        "method_definition" | "method_declaration" => Some(SymbolKind::Method),
        "class_declaration" | "class_definition" => Some(SymbolKind::Class),
        "struct_item" | "struct_declaration" | "struct_definition" => Some(SymbolKind::Struct),
        "interface_declaration" => Some(SymbolKind::Interface),
        "trait_item" | "trait_definition" => Some(SymbolKind::Trait),
        "enum_item" | "enum_declaration" => Some(SymbolKind::Enum),
        "type_alias_declaration" | "type_item" | "type_declaration" => Some(SymbolKind::Type),
        "const_item" | "const_declaration" => Some(SymbolKind::Constant),
        "variable_declaration" | "lexical_declaration" | "let_declaration" | "static_item" => {
            Some(SymbolKind::Variable)
        }
        "module" | "mod_item" | "module_declaration" => Some(SymbolKind::Module),
        "namespace_declaration" => Some(SymbolKind::Namespace),
        "package_clause" | "package_declaration" => Some(SymbolKind::Package),
        "impl_item" => Some(SymbolKind::Struct),
        _ => {
            if kind.contains("function") {
                Some(SymbolKind::Function)
            } else if kind.contains("method") {
                Some(SymbolKind::Method)
            } else if kind.contains("class") {
                Some(SymbolKind::Class)
            } else if kind.contains("struct") {
                Some(SymbolKind::Struct)
            } else if kind.contains("interface") {
                Some(SymbolKind::Interface)
            } else if kind.contains("trait") {
                Some(SymbolKind::Trait)
            } else if kind.contains("enum") {
                Some(SymbolKind::Enum)
            } else {
                None
            }
        }
    }
}

pub fn get_tree_sitter_language(language: &str) -> Option<tree_sitter::Language> {
    match language {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "csharp" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_symbol_extractor_rust() {
        let mut extractor = SymbolExtractor::new();
        let source = r#"
fn hello() {
    println!("hello");
}

struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn method(&self) -> i32 {
        self.field
    }
}
"#;

        let (symbols, _edges, _imports) = extractor
            .extract_from_file(Path::new("test.rs"), source, "rust")
            .unwrap();

        assert!(!symbols.is_empty());

        let function_names: Vec<_> = symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Function)
            .map(|s| s.name.as_str())
            .collect();

        assert!(function_names.contains(&"hello"));
    }

    #[test]
    fn test_import_extraction_rust() {
        let mut extractor = SymbolExtractor::new();
        let source = r#"
use std::collections::HashMap;
use crate::module::Item as MyItem;

fn main() {}
"#;

        let (_symbols, _edges, imports) = extractor
            .extract_from_file(Path::new("test.rs"), source, "rust")
            .unwrap();

        assert!(!imports.is_empty());
    }
}
