//! Knowledge Graph for Code Search
//!
//! This module provides a structural understanding layer on top of semantic vector search.
//! It extracts symbols, their relationships, and enables graph-aware query routing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tree_sitter::{Parser, Tree};
use uuid::Uuid;

/// Types of symbols that can be extracted from code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Interface,
    Trait,
    Enum,
    Type,
    Constant,
    Variable,
    Module,
    Namespace,
    Package,
    Field,
    Property,
}

impl SymbolKind {
    pub fn label(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Method => "method",
            SymbolKind::Class => "class",
            SymbolKind::Struct => "struct",
            SymbolKind::Interface => "interface",
            SymbolKind::Trait => "trait",
            SymbolKind::Enum => "enum",
            SymbolKind::Type => "type",
            SymbolKind::Constant => "constant",
            SymbolKind::Variable => "variable",
            SymbolKind::Module => "module",
            SymbolKind::Namespace => "namespace",
            SymbolKind::Package => "package",
            SymbolKind::Field => "field",
            SymbolKind::Property => "property",
        }
    }
}

/// Types of relationships between symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// File imports another file/module
    Imports,
    /// Symbol exports from module
    Exports,
    /// Function/method calls another function
    Calls,
    /// Symbol is defined in file/module
    DefinedIn,
    /// Symbol references another symbol
    References,
    /// Container contains symbol (class contains method)
    Contains,
    /// Type implements interface/trait
    Implements,
    /// Class extends another class
    Extends,
    /// Type parameter or generic constraint
    TypeOf,
}

/// A symbol extracted from code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub id: Uuid,
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub file_path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    /// Signature or declaration text
    pub signature: String,
    /// Parent symbol ID (for methods in classes, etc.)
    pub parent_id: Option<Uuid>,
    /// Associated chunk ID for vector search
    pub chunk_id: Option<Uuid>,
}

/// An edge in the code graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub kind: EdgeKind,
    /// Optional metadata (e.g., import alias, call site line)
    pub metadata: Option<String>,
}

/// Import relationship between files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportRelation {
    pub source_file: PathBuf,
    pub target_path: String,
    pub alias: Option<String>,
    pub line: usize,
    pub is_type_only: bool,
}

/// The code knowledge graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeGraph {
    /// All symbols indexed by ID
    pub symbols: HashMap<Uuid, Symbol>,
    /// Symbol name to IDs (for quick lookup)
    pub name_index: HashMap<String, Vec<Uuid>>,
    /// File to symbols mapping
    pub file_symbols: HashMap<PathBuf, Vec<Uuid>>,
    /// Edges (relationships)
    pub edges: Vec<Edge>,
    /// Import relationships
    pub imports: Vec<ImportRelation>,
    /// Outgoing edges by source ID
    outgoing: HashMap<Uuid, Vec<usize>>,
    /// Incoming edges by target ID
    incoming: HashMap<Uuid, Vec<usize>>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol to the graph
    pub fn add_symbol(&mut self, symbol: Symbol) {
        let id = symbol.id;
        let name = symbol.name.clone();
        let file = symbol.file_path.clone();

        self.symbols.insert(id, symbol);
        self.name_index.entry(name).or_default().push(id);
        self.file_symbols.entry(file).or_default().push(id);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) {
        let idx = self.edges.len();
        self.outgoing.entry(edge.source_id).or_default().push(idx);
        self.incoming.entry(edge.target_id).or_default().push(idx);
        self.edges.push(edge);
    }

    /// Add an import relationship
    pub fn add_import(&mut self, import: ImportRelation) {
        self.imports.push(import);
    }

    /// Get symbol by ID
    pub fn get_symbol(&self, id: &Uuid) -> Option<&Symbol> {
        self.symbols.get(id)
    }

    /// Find symbols by name
    pub fn find_by_name(&self, name: &str) -> Vec<&Symbol> {
        self.name_index
            .get(name)
            .map(|ids| ids.iter().filter_map(|id| self.symbols.get(id)).collect())
            .unwrap_or_default()
    }

    /// Find symbols by name prefix (for autocomplete)
    pub fn find_by_prefix(&self, prefix: &str) -> Vec<&Symbol> {
        let prefix_lower = prefix.to_lowercase();
        self.name_index
            .iter()
            .filter(|(name, _)| name.to_lowercase().starts_with(&prefix_lower))
            .flat_map(|(_, ids)| ids.iter().filter_map(|id| self.symbols.get(id)))
            .collect()
    }

    /// Find symbols in a file
    pub fn symbols_in_file(&self, path: &Path) -> Vec<&Symbol> {
        self.file_symbols
            .get(path)
            .map(|ids| ids.iter().filter_map(|id| self.symbols.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get outgoing edges from a symbol
    pub fn outgoing_edges(&self, symbol_id: &Uuid) -> Vec<&Edge> {
        self.outgoing
            .get(symbol_id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get incoming edges to a symbol
    pub fn incoming_edges(&self, symbol_id: &Uuid) -> Vec<&Edge> {
        self.incoming
            .get(symbol_id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Find callers of a function
    pub fn find_callers(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.incoming_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Calls)
            .filter_map(|e| self.symbols.get(&e.source_id))
            .collect()
    }

    /// Find callees of a function
    pub fn find_callees(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.outgoing_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Calls)
            .filter_map(|e| self.symbols.get(&e.target_id))
            .collect()
    }

    /// Find files that import a given file
    pub fn find_importers(&self, file_path: &Path) -> Vec<&PathBuf> {
        let path_str = file_path.to_string_lossy();
        self.imports
            .iter()
            .filter(|imp| {
                imp.target_path.contains(path_str.as_ref())
                    || path_str.contains(&imp.target_path)
            })
            .map(|imp| &imp.source_file)
            .collect()
    }

    /// Find files that a given file imports
    pub fn find_imports_of(&self, file_path: &Path) -> Vec<&ImportRelation> {
        self.imports
            .iter()
            .filter(|imp| imp.source_file == file_path)
            .collect()
    }

    /// Find implementors of a trait/interface
    pub fn find_implementors(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.incoming_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Implements)
            .filter_map(|e| self.symbols.get(&e.source_id))
            .collect()
    }

    /// Find what a type implements
    pub fn find_implementations(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.outgoing_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Implements)
            .filter_map(|e| self.symbols.get(&e.target_id))
            .collect()
    }

    /// Find definitions of a symbol name
    pub fn find_definitions(&self, name: &str, kind: Option<SymbolKind>) -> Vec<&Symbol> {
        self.find_by_name(name)
            .into_iter()
            .filter(|s| kind.map_or(true, |k| s.kind == k))
            .collect()
    }

    /// Get symbols by kind
    pub fn symbols_of_kind(&self, kind: SymbolKind) -> Vec<&Symbol> {
        self.symbols.values().filter(|s| s.kind == kind).collect()
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        let mut kind_counts: HashMap<SymbolKind, usize> = HashMap::new();
        for symbol in self.symbols.values() {
            *kind_counts.entry(symbol.kind).or_default() += 1;
        }

        let mut edge_counts: HashMap<EdgeKind, usize> = HashMap::new();
        for edge in &self.edges {
            *edge_counts.entry(edge.kind).or_default() += 1;
        }

        GraphStats {
            total_symbols: self.symbols.len(),
            total_edges: self.edges.len(),
            total_imports: self.imports.len(),
            total_files: self.file_symbols.len(),
            symbols_by_kind: kind_counts,
            edges_by_kind: edge_counts,
        }
    }

    /// Merge another graph into this one
    pub fn merge(&mut self, other: CodeGraph) {
        for (id, symbol) in other.symbols {
            self.add_symbol(Symbol { id, ..symbol });
        }
        for edge in other.edges {
            self.add_edge(edge);
        }
        for import in other.imports {
            self.add_import(import);
        }
    }

    /// Remove all symbols and edges for a file
    pub fn remove_file(&mut self, path: &Path) {
        if let Some(symbol_ids) = self.file_symbols.remove(path) {
            let ids_set: HashSet<_> = symbol_ids.iter().collect();

            // Remove edges involving these symbols
            self.edges.retain(|e| {
                !ids_set.contains(&e.source_id) && !ids_set.contains(&e.target_id)
            });

            // Remove symbols
            for id in &symbol_ids {
                if let Some(symbol) = self.symbols.remove(id) {
                    if let Some(names) = self.name_index.get_mut(&symbol.name) {
                        names.retain(|i| i != id);
                    }
                }
            }

            // Rebuild edge indices
            self.rebuild_edge_indices();
        }

        // Remove imports from this file
        self.imports.retain(|imp| imp.source_file != path);
    }

    fn rebuild_edge_indices(&mut self) {
        self.outgoing.clear();
        self.incoming.clear();
        for (idx, edge) in self.edges.iter().enumerate() {
            self.outgoing.entry(edge.source_id).or_default().push(idx);
            self.incoming.entry(edge.target_id).or_default().push(idx);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_symbols: usize,
    pub total_edges: usize,
    pub total_imports: usize,
    pub total_files: usize,
    pub symbols_by_kind: HashMap<SymbolKind, usize>,
    pub edges_by_kind: HashMap<EdgeKind, usize>,
}

/// Extract symbols and relationships from source code
pub struct SymbolExtractor {
    parsers: HashMap<String, Parser>,
}

impl SymbolExtractor {
    pub fn new() -> Self {
        Self {
            parsers: HashMap::new(),
        }
    }

    /// Extract symbols from a file
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

        // First pass: collect imports
        for node in root.children(&mut cursor) {
            if let Some(import) = self.extract_import(&node, source, path, language) {
                imports.push(import);
            }
        }

        // Second pass: collect symbols
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

                    // Add containment edge if there's a parent
                    if let Some(parent) = parent_id {
                        edges.push(Edge {
                            source_id: parent,
                            target_id: symbol_id,
                            kind: EdgeKind::Contains,
                            metadata: None,
                        });
                    }

                    symbols.push(symbol);

                    // Recurse for nested symbols (methods in classes, etc.)
                    if is_container_kind(symbol_kind) {
                        self.collect_symbols(&child, source, path, language, Some(symbol_id), symbols, edges);
                    }

                    // Extract call relationships
                    self.extract_calls(&child, source, symbol_id, edges);
                }
            } else {
                // Continue recursing for non-symbol nodes
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

        // Build qualified name
        let qualified_name = if let Some(parent) = parent_id {
            // Would need parent name lookup - simplified for now
            name.clone()
        } else {
            name.clone()
        };

        // Extract signature (first line or declaration)
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

        // Look for identifier nodes that give us the name
        for child in node.children(&mut cursor) {
            let child_kind = child.kind();

            // Common identifier patterns across languages
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

            // TypeScript/JavaScript function declarations
            if child_kind == "call_expression" && language == "typescript" {
                continue;
            }
        }

        // Fallback: try to find any identifier in the first line
        if let Ok(text) = node.utf8_text(source.as_bytes()) {
            let first_line = text.lines().next()?;
            // Extract name from common patterns like "fn name(", "function name(", "class Name"
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

        // Check if this is an import-like node
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
                // use std::collections::HashMap;
                // use crate::module::Item;
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
                // import { X } from 'module';
                // import X from 'module';
                // const X = require('module');
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
                // from module import X
                // import module
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
                // import "package"
                // import alias "package"
                if let Some(quote_start) = text.find('"') {
                    if let Some(quote_end) = text[quote_start + 1..].find('"') {
                        return Some(text[quote_start + 1..quote_start + 1 + quote_end].to_string());
                    }
                }
            }
            "java" => {
                // import com.example.Class;
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
                // #include <header.h>
                // #include "header.h"
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
                // use module::Item as Alias;
                if let Some(as_idx) = text.find(" as ") {
                    let after = &text[as_idx + 4..];
                    let alias = after.trim_end_matches(';').trim();
                    if !alias.is_empty() {
                        return Some(alias.to_string());
                    }
                }
            }
            "typescript" | "javascript" | "tsx" => {
                // import X as Alias from 'module';
                // import { X as Alias } from 'module';
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
                // import module as alias
                // from module import X as alias
                if let Some(as_idx) = text.find(" as ") {
                    let after = &text[as_idx + 4..];
                    let alias = after.split_whitespace().next()?;
                    return Some(alias.to_string());
                }
            }
            "go" => {
                // import alias "package"
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
        let mut cursor = node.walk();
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

            // Check for call expressions
            if matches!(kind, "call_expression" | "method_call_expression" | "function_call") {
                if let Some(callee_name) = self.extract_callee_name(&child, source) {
                    // Avoid duplicate edges for same callee
                    if seen.insert(callee_name.clone()) {
                        edges.push(Edge {
                            source_id: caller_id,
                            // Target is a placeholder - will be resolved later
                            target_id: Uuid::nil(),
                            kind: EdgeKind::Calls,
                            metadata: Some(callee_name),
                        });
                    }
                }
            }

            // Recurse
            self.collect_call_expressions(&child, source, caller_id, edges, seen);
        }
    }

    fn extract_callee_name(&self, node: &tree_sitter::Node, source: &str) -> Option<String> {
        let mut cursor = node.walk();

        // Look for the function/method being called
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

            // For method calls like obj.method(), get the method name
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

/// Classify tree-sitter node kind to SymbolKind
fn classify_node_kind(kind: &str, _language: &str) -> Option<SymbolKind> {
    match kind {
        // Functions
        "function_declaration" | "function_item" | "function_definition" | "arrow_function" => {
            Some(SymbolKind::Function)
        }
        // Methods
        "method_definition" | "method_declaration" => Some(SymbolKind::Method),
        // Classes
        "class_declaration" | "class_definition" => Some(SymbolKind::Class),
        // Structs
        "struct_item" | "struct_declaration" | "struct_definition" => Some(SymbolKind::Struct),
        // Interfaces
        "interface_declaration" => Some(SymbolKind::Interface),
        // Traits
        "trait_item" | "trait_definition" => Some(SymbolKind::Trait),
        // Enums
        "enum_item" | "enum_declaration" => Some(SymbolKind::Enum),
        // Types
        "type_alias_declaration" | "type_item" | "type_declaration" => Some(SymbolKind::Type),
        // Constants
        "const_item" | "const_declaration" => Some(SymbolKind::Constant),
        // Variables
        "variable_declaration" | "lexical_declaration" | "let_declaration" | "static_item" => {
            Some(SymbolKind::Variable)
        }
        // Modules
        "module" | "mod_item" | "module_declaration" => Some(SymbolKind::Module),
        // Namespaces
        "namespace_declaration" => Some(SymbolKind::Namespace),
        // Packages
        "package_clause" | "package_declaration" => Some(SymbolKind::Package),
        // Impl blocks (treat as containing methods)
        "impl_item" => Some(SymbolKind::Struct), // Impl blocks associate with structs
        _ => {
            // Check for common patterns
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

/// Check if a symbol kind can contain other symbols
fn is_container_kind(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Class
            | SymbolKind::Struct
            | SymbolKind::Interface
            | SymbolKind::Trait
            | SymbolKind::Module
            | SymbolKind::Namespace
    )
}

/// Get tree-sitter language for a language name
fn get_tree_sitter_language(language: &str) -> Option<tree_sitter::Language> {
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

/// Query classifier for routing to graph vs vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Structural query - use graph
    Structural,
    /// Semantic query - use vectors
    Semantic,
    /// Hybrid - use both
    Hybrid,
}

/// Classify a query to determine the best search approach
pub fn classify_query(query: &str) -> QueryType {
    let query_lower = query.to_lowercase();

    // Structural patterns
    let structural_patterns = [
        "who calls",
        "what calls",
        "callers of",
        "calls to",
        "imports",
        "imported by",
        "defined in",
        "definition of",
        "find definition",
        "go to definition",
        "implementations of",
        "implementors of",
        "extends",
        "inherits from",
        "usages of",
        "references to",
        "where is .* used",
        "where is .* defined",
        "what uses",
        "what imports",
        "dependencies of",
        "dependents of",
    ];

    // Check for structural patterns
    for pattern in structural_patterns {
        if query_lower.contains(pattern) {
            return QueryType::Structural;
        }
    }

    // Symbol-like queries (specific names)
    let has_symbol_chars = query.chars().any(|c| c == '.' || c == ':' || c == '_');
    let is_camel_case = query.chars().any(|c| c.is_uppercase())
        && query.chars().any(|c| c.is_lowercase())
        && !query.contains(' ');
    let is_snake_case = query.contains('_') && !query.contains(' ');

    if (has_symbol_chars || is_camel_case || is_snake_case) && query.len() < 50 {
        return QueryType::Hybrid;
    }

    // Question patterns (conceptual) - pure semantic
    let question_patterns = ["how does", "what is", "explain", "why", "describe"];
    for pattern in question_patterns {
        if query_lower.starts_with(pattern) {
            return QueryType::Semantic;
        }
    }

    // Default to hybrid for most queries
    QueryType::Hybrid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_kind_labels() {
        assert_eq!(SymbolKind::Function.label(), "function");
        assert_eq!(SymbolKind::Class.label(), "class");
        assert_eq!(SymbolKind::Trait.label(), "trait");
    }

    #[test]
    fn test_code_graph_add_and_find() {
        let mut graph = CodeGraph::new();

        let symbol = Symbol {
            id: Uuid::new_v4(),
            name: "my_function".to_string(),
            qualified_name: "module::my_function".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 10,
            end_line: 20,
            language: "rust".to_string(),
            signature: "fn my_function() -> i32".to_string(),
            parent_id: None,
            chunk_id: None,
        };

        graph.add_symbol(symbol);

        let found = graph.find_by_name("my_function");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "my_function");
    }

    #[test]
    fn test_code_graph_edges() {
        let mut graph = CodeGraph::new();

        let caller_id = Uuid::new_v4();
        let callee_id = Uuid::new_v4();

        let caller = Symbol {
            id: caller_id,
            name: "caller".to_string(),
            qualified_name: "caller".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn caller()".to_string(),
            parent_id: None,
            chunk_id: None,
        };

        let callee = Symbol {
            id: callee_id,
            name: "callee".to_string(),
            qualified_name: "callee".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 10,
            end_line: 15,
            language: "rust".to_string(),
            signature: "fn callee()".to_string(),
            parent_id: None,
            chunk_id: None,
        };

        graph.add_symbol(caller);
        graph.add_symbol(callee);
        graph.add_edge(Edge {
            source_id: caller_id,
            target_id: callee_id,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        let callees = graph.find_callees(&caller_id);
        assert_eq!(callees.len(), 1);
        assert_eq!(callees[0].name, "callee");

        let callers = graph.find_callers(&callee_id);
        assert_eq!(callers.len(), 1);
        assert_eq!(callers[0].name, "caller");
    }

    #[test]
    fn test_query_classification() {
        assert_eq!(classify_query("who calls authenticate"), QueryType::Structural);
        assert_eq!(classify_query("callers of render"), QueryType::Structural);
        assert_eq!(classify_query("what imports this module"), QueryType::Structural);

        assert_eq!(classify_query("how does authentication work"), QueryType::Semantic);
        assert_eq!(classify_query("explain the caching mechanism"), QueryType::Semantic);

        assert_eq!(classify_query("MyClassName"), QueryType::Hybrid);
        assert_eq!(classify_query("my_function_name"), QueryType::Hybrid);
    }

    #[test]
    fn test_graph_stats() {
        let mut graph = CodeGraph::new();

        for i in 0..5 {
            graph.add_symbol(Symbol {
                id: Uuid::new_v4(),
                name: format!("func_{}", i),
                qualified_name: format!("func_{}", i),
                kind: SymbolKind::Function,
                file_path: PathBuf::from("src/lib.rs"),
                start_line: i * 10,
                end_line: i * 10 + 5,
                language: "rust".to_string(),
                signature: format!("fn func_{}()", i),
                parent_id: None,
                chunk_id: None,
            });
        }

        let stats = graph.stats();
        assert_eq!(stats.total_symbols, 5);
        assert_eq!(stats.total_files, 1);
        assert_eq!(*stats.symbols_by_kind.get(&SymbolKind::Function).unwrap(), 5);
    }

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

        let (symbols, edges, _imports) = extractor
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

    #[test]
    fn test_graph_remove_file() {
        let mut graph = CodeGraph::new();
        let file1 = PathBuf::from("src/a.rs");
        let file2 = PathBuf::from("src/b.rs");

        let sym1 = Symbol {
            id: Uuid::new_v4(),
            name: "func_a".to_string(),
            qualified_name: "func_a".to_string(),
            kind: SymbolKind::Function,
            file_path: file1.clone(),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn func_a()".to_string(),
            parent_id: None,
            chunk_id: None,
        };

        let sym2 = Symbol {
            id: Uuid::new_v4(),
            name: "func_b".to_string(),
            qualified_name: "func_b".to_string(),
            kind: SymbolKind::Function,
            file_path: file2.clone(),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn func_b()".to_string(),
            parent_id: None,
            chunk_id: None,
        };

        graph.add_symbol(sym1);
        graph.add_symbol(sym2);

        assert_eq!(graph.symbols.len(), 2);

        graph.remove_file(&file1);

        assert_eq!(graph.symbols.len(), 1);
        assert!(graph.find_by_name("func_a").is_empty());
        assert!(!graph.find_by_name("func_b").is_empty());
    }
}
