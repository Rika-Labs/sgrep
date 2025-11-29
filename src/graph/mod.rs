mod extractor;
mod query;
mod symbols;

pub use extractor::SymbolExtractor;
pub use query::QueryType;
pub use symbols::{Edge, EdgeKind, ImportRelation, Symbol, SymbolKind};

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeGraph {
    pub symbols: HashMap<Uuid, Symbol>,
    pub name_index: HashMap<String, Vec<Uuid>>,
    pub file_symbols: HashMap<PathBuf, Vec<Uuid>>,
    pub edges: Vec<Edge>,
    pub imports: Vec<ImportRelation>,
    outgoing: HashMap<Uuid, Vec<usize>>,
    incoming: HashMap<Uuid, Vec<usize>>,
}

#[allow(dead_code)]
impl CodeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_symbol(&mut self, symbol: Symbol) {
        let id = symbol.id;
        let name = symbol.name.clone();
        let file = symbol.file_path.clone();

        self.symbols.insert(id, symbol);
        self.name_index.entry(name).or_default().push(id);
        self.file_symbols.entry(file).or_default().push(id);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        let idx = self.edges.len();
        self.outgoing.entry(edge.source_id).or_default().push(idx);
        self.incoming.entry(edge.target_id).or_default().push(idx);
        self.edges.push(edge);
    }

    pub fn add_import(&mut self, import: ImportRelation) {
        self.imports.push(import);
    }

    pub fn get_symbol(&self, id: &Uuid) -> Option<&Symbol> {
        self.symbols.get(id)
    }

    pub fn find_by_name(&self, name: &str) -> Vec<&Symbol> {
        self.name_index
            .get(name)
            .map(|ids| ids.iter().filter_map(|id| self.symbols.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn find_by_prefix(&self, prefix: &str) -> Vec<&Symbol> {
        let prefix_lower = prefix.to_lowercase();
        self.name_index
            .iter()
            .filter(|(name, _)| name.to_lowercase().starts_with(&prefix_lower))
            .flat_map(|(_, ids)| ids.iter().filter_map(|id| self.symbols.get(id)))
            .collect()
    }

    pub fn symbols_in_file(&self, path: &Path) -> Vec<&Symbol> {
        self.file_symbols
            .get(path)
            .map(|ids| ids.iter().filter_map(|id| self.symbols.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn outgoing_edges(&self, symbol_id: &Uuid) -> Vec<&Edge> {
        self.outgoing
            .get(symbol_id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn incoming_edges(&self, symbol_id: &Uuid) -> Vec<&Edge> {
        self.incoming
            .get(symbol_id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn find_callers(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.incoming_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Calls)
            .filter_map(|e| self.symbols.get(&e.source_id))
            .collect()
    }

    pub fn find_callees(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.outgoing_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Calls)
            .filter_map(|e| self.symbols.get(&e.target_id))
            .collect()
    }

    pub fn find_importers(&self, file_path: &Path) -> Vec<&PathBuf> {
        let path_str = file_path.to_string_lossy();
        self.imports
            .iter()
            .filter(|imp| {
                imp.target_path.contains(path_str.as_ref()) || path_str.contains(&imp.target_path)
            })
            .map(|imp| &imp.source_file)
            .collect()
    }

    pub fn find_imports_of(&self, file_path: &Path) -> Vec<&ImportRelation> {
        self.imports
            .iter()
            .filter(|imp| imp.source_file == file_path)
            .collect()
    }

    pub fn find_implementors(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.incoming_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Implements)
            .filter_map(|e| self.symbols.get(&e.source_id))
            .collect()
    }

    pub fn find_implementations(&self, symbol_id: &Uuid) -> Vec<&Symbol> {
        self.outgoing_edges(symbol_id)
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Implements)
            .filter_map(|e| self.symbols.get(&e.target_id))
            .collect()
    }

    pub fn find_definitions(&self, name: &str, kind: Option<SymbolKind>) -> Vec<&Symbol> {
        self.find_by_name(name)
            .into_iter()
            .filter(|s| kind.map_or(true, |k| s.kind == k))
            .collect()
    }

    pub fn symbols_of_kind(&self, kind: SymbolKind) -> Vec<&Symbol> {
        self.symbols.values().filter(|s| s.kind == kind).collect()
    }

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

    pub fn remove_file(&mut self, path: &Path) {
        if let Some(symbol_ids) = self.file_symbols.remove(path) {
            let ids_set: HashSet<_> = symbol_ids.iter().collect();

            self.edges
                .retain(|e| !ids_set.contains(&e.source_id) && !ids_set.contains(&e.target_id));

            for id in &symbol_ids {
                if let Some(symbol) = self.symbols.remove(id) {
                    if let Some(names) = self.name_index.get_mut(&symbol.name) {
                        names.retain(|i| i != id);
                    }
                }
            }

            self.rebuild_edge_indices();
        }

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            *stats.symbols_by_kind.get(&SymbolKind::Function).unwrap(),
            5
        );
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

    #[test]
    fn test_get_symbol() {
        let mut graph = CodeGraph::new();
        let id = Uuid::new_v4();
        graph.add_symbol(Symbol {
            id,
            name: "test".to_string(),
            qualified_name: "test".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn test()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        assert!(graph.get_symbol(&id).is_some());
        assert!(graph.get_symbol(&Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_find_by_prefix() {
        let mut graph = CodeGraph::new();
        for name in ["authenticate", "authorize", "auth_helper"] {
            graph.add_symbol(Symbol {
                id: Uuid::new_v4(),
                name: name.to_string(),
                qualified_name: name.to_string(),
                kind: SymbolKind::Function,
                file_path: PathBuf::from("src/lib.rs"),
                start_line: 1,
                end_line: 5,
                language: "rust".to_string(),
                signature: format!("fn {}()", name),
                parent_id: None,
                chunk_id: None,
            });
        }

        let found = graph.find_by_prefix("auth");
        assert_eq!(found.len(), 3);
    }

    #[test]
    fn test_find_implementors() {
        let mut graph = CodeGraph::new();
        let trait_id = Uuid::new_v4();
        let impl_id = Uuid::new_v4();

        graph.add_symbol(Symbol {
            id: trait_id,
            name: "MyTrait".to_string(),
            qualified_name: "MyTrait".to_string(),
            kind: SymbolKind::Trait,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "trait MyTrait".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_symbol(Symbol {
            id: impl_id,
            name: "MyStruct".to_string(),
            qualified_name: "MyStruct".to_string(),
            kind: SymbolKind::Struct,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 10,
            end_line: 15,
            language: "rust".to_string(),
            signature: "struct MyStruct".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_edge(Edge {
            source_id: impl_id,
            target_id: trait_id,
            kind: EdgeKind::Implements,
            metadata: None,
        });

        let implementors = graph.find_implementors(&trait_id);
        assert_eq!(implementors.len(), 1);
        assert_eq!(implementors[0].name, "MyStruct");
    }

    #[test]
    fn test_symbols_of_kind() {
        let mut graph = CodeGraph::new();
        for i in 0..3 {
            graph.add_symbol(Symbol {
                id: Uuid::new_v4(),
                name: format!("func_{}", i),
                qualified_name: format!("func_{}", i),
                kind: SymbolKind::Function,
                file_path: PathBuf::from("src/lib.rs"),
                start_line: i,
                end_line: i + 5,
                language: "rust".to_string(),
                signature: format!("fn func_{}()", i),
                parent_id: None,
                chunk_id: None,
            });
        }
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "MyStruct".to_string(),
            qualified_name: "MyStruct".to_string(),
            kind: SymbolKind::Struct,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 100,
            end_line: 110,
            language: "rust".to_string(),
            signature: "struct MyStruct".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        let functions = graph.symbols_of_kind(SymbolKind::Function);
        assert_eq!(functions.len(), 3);
        let structs = graph.symbols_of_kind(SymbolKind::Struct);
        assert_eq!(structs.len(), 1);
    }

    #[test]
    fn test_merge_graphs() {
        let mut graph1 = CodeGraph::new();
        graph1.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "func_a".to_string(),
            qualified_name: "func_a".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/a.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn func_a()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        let mut graph2 = CodeGraph::new();
        graph2.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "func_b".to_string(),
            qualified_name: "func_b".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/b.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn func_b()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph1.merge(graph2);
        assert_eq!(graph1.symbols.len(), 2);
    }

    #[test]
    fn test_find_definitions() {
        let mut graph = CodeGraph::new();
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "foo".to_string(),
            qualified_name: "foo".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn foo()".to_string(),
            parent_id: None,
            chunk_id: None,
        });
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "foo".to_string(),
            qualified_name: "foo".to_string(),
            kind: SymbolKind::Struct,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 10,
            end_line: 15,
            language: "rust".to_string(),
            signature: "struct foo".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        let all_foos = graph.find_definitions("foo", None);
        assert_eq!(all_foos.len(), 2);

        let func_foos = graph.find_definitions("foo", Some(SymbolKind::Function));
        assert_eq!(func_foos.len(), 1);
    }

    #[test]
    fn test_add_import() {
        let mut graph = CodeGraph::new();
        graph.add_import(ImportRelation {
            source_file: PathBuf::from("src/main.rs"),
            target_path: "src/auth".to_string(),
            alias: None,
            line: 1,
            is_type_only: false,
        });

        assert_eq!(graph.imports.len(), 1);
    }

    #[test]
    fn test_find_imports_of() {
        let mut graph = CodeGraph::new();
        let file = PathBuf::from("src/main.rs");
        graph.add_import(ImportRelation {
            source_file: file.clone(),
            target_path: "auth".to_string(),
            alias: None,
            line: 1,
            is_type_only: false,
        });
        graph.add_import(ImportRelation {
            source_file: file.clone(),
            target_path: "utils".to_string(),
            alias: None,
            line: 2,
            is_type_only: false,
        });

        let imports = graph.find_imports_of(&file);
        assert_eq!(imports.len(), 2);
    }
}
