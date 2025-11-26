use crate::graph::{CodeGraph, Symbol};

pub fn extract_symbol_name_from_query(query: &str, patterns: &[&str]) -> Option<String> {
    let query_lower = query.to_lowercase();

    for pattern in patterns {
        if let Some(idx) = query_lower.find(pattern) {
            let after = &query[idx + pattern.len()..];
            let name = after
                .trim()
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()?
                .trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }

    None
}

pub fn is_common_word(word: &str) -> bool {
    const COMMON_WORDS: &[&str] = &[
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "into",
        "when",
        "what",
        "how",
        "why",
        "does",
        "are",
        "our",
        "your",
        "their",
        "then",
        "where",
        "find",
        "show",
        "get",
        "can",
        "will",
        "should",
        "would",
        "could",
        "function",
        "method",
        "class",
        "file",
        "code",
        "implement",
        "implementation",
        "definition",
        "callers",
        "callees",
        "imports",
        "exports",
    ];
    COMMON_WORDS.contains(&word.to_lowercase().as_str())
}

pub fn search_graph_only(graph: &CodeGraph, query: &str) -> Vec<Symbol> {
    let query_lower = query.to_lowercase();

    if query_lower.contains("callers of") || query_lower.contains("who calls") {
        if let Some(name) =
            extract_symbol_name_from_query(query, &["callers of", "who calls", "what calls"])
        {
            let symbols = graph.find_by_name(&name);
            return symbols
                .into_iter()
                .flat_map(|s| graph.find_callers(&s.id))
                .cloned()
                .collect();
        }
    }

    if query_lower.contains("calls to") || query_lower.contains("callees of") {
        if let Some(name) = extract_symbol_name_from_query(query, &["calls to", "callees of"]) {
            let symbols = graph.find_by_name(&name);
            return symbols
                .into_iter()
                .flat_map(|s| graph.find_callees(&s.id))
                .cloned()
                .collect();
        }
    }

    if query_lower.contains("definition of") || query_lower.contains("find definition") {
        if let Some(name) = extract_symbol_name_from_query(
            query,
            &["definition of", "find definition", "go to definition"],
        ) {
            return graph.find_by_name(&name).into_iter().cloned().collect();
        }
    }

    if query_lower.contains("implementations of") || query_lower.contains("implementors of") {
        if let Some(name) =
            extract_symbol_name_from_query(query, &["implementations of", "implementors of"])
        {
            let symbols = graph.find_by_name(&name);
            return symbols
                .into_iter()
                .flat_map(|s| graph.find_implementors(&s.id))
                .cloned()
                .collect();
        }
    }

    if query_lower.contains("imports") {
        if let Some(name) = extract_symbol_name_from_query(query, &["imports", "what imports"]) {
            let path = std::path::PathBuf::from(&name);
            return graph
                .find_importers(&path)
                .into_iter()
                .flat_map(|p| graph.symbols_in_file(p))
                .cloned()
                .collect();
        }
    }

    let words: Vec<_> = query.split_whitespace().collect();
    if words.len() <= 3 {
        for word in words {
            if word.len() >= 3 && !is_common_word(word) {
                let results = graph.find_by_name(word);
                if !results.is_empty() {
                    return results.into_iter().cloned().collect();
                }
                let prefix_results = graph.find_by_prefix(word);
                if !prefix_results.is_empty() {
                    return prefix_results.into_iter().take(10).cloned().collect();
                }
            }
        }
    }

    vec![]
}

use super::SearchResult;

pub fn apply_graph_boost(results: &mut [SearchResult], graph: &CodeGraph, query: &str) {
    let query_words: Vec<_> = query
        .split_whitespace()
        .filter(|w| w.len() >= 3 && !is_common_word(w))
        .collect();

    for result in results.iter_mut() {
        let file_path = &result.chunk.path;

        let file_symbols = graph.symbols_in_file(file_path);
        let mut boost = 0.0f32;

        for symbol in file_symbols {
            for word in &query_words {
                let word_lower = word.to_lowercase();
                if symbol.name.to_lowercase().contains(&word_lower) {
                    boost += 0.1;
                }
            }

            let incoming = graph.incoming_edges(&symbol.id).len();
            if incoming > 5 {
                boost += 0.05;
            }
        }

        let imports = graph.find_imports_of(file_path);
        if imports.len() > 0 {
            boost += 0.02 * (imports.len() as f32).min(5.0);
        }

        result.score *= 1.0 + boost.min(0.2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use crate::graph::{Edge, EdgeKind, SymbolKind};
    use chrono::Utc;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn make_symbol(name: &str, file: &str) -> Symbol {
        Symbol {
            id: Uuid::new_v4(),
            name: name.to_string(),
            qualified_name: name.to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from(file),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: format!("fn {}()", name),
            parent_id: None,
            chunk_id: None,
        }
    }

    fn make_chunk(path: &str, content: &str) -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 5,
            text: content.to_string(),
            hash: "test".to_string(),
            modified_at: Utc::now(),
        }
    }

    #[test]
    fn extract_symbol_name_basic() {
        let name = extract_symbol_name_from_query("callers of authenticate", &["callers of"]);
        assert_eq!(name, Some("authenticate".to_string()));
    }

    #[test]
    fn extract_symbol_name_with_extra_words() {
        let name = extract_symbol_name_from_query("who calls process_data function", &["who calls"]);
        assert_eq!(name, Some("process_data".to_string()));
    }

    #[test]
    fn extract_symbol_name_no_match() {
        let name = extract_symbol_name_from_query("find something", &["callers of"]);
        assert_eq!(name, None);
    }

    #[test]
    fn extract_symbol_name_empty_after_pattern() {
        let name = extract_symbol_name_from_query("callers of ", &["callers of"]);
        assert_eq!(name, None);
    }

    #[test]
    fn is_common_word_returns_true() {
        assert!(is_common_word("the"));
        assert!(is_common_word("function"));
        assert!(is_common_word("implementation"));
        assert!(is_common_word("callers"));
    }

    #[test]
    fn is_common_word_returns_false() {
        assert!(!is_common_word("authenticate"));
        assert!(!is_common_word("process"));
        assert!(!is_common_word("foobar"));
    }

    #[test]
    fn is_common_word_case_insensitive() {
        assert!(is_common_word("THE"));
        assert!(is_common_word("Function"));
    }

    #[test]
    fn search_graph_callers_of() {
        let mut graph = CodeGraph::new();
        let callee = make_symbol("authenticate", "src/auth.rs");
        let caller = make_symbol("login", "src/login.rs");
        let callee_id = callee.id;
        let caller_id = caller.id;
        graph.add_symbol(callee);
        graph.add_symbol(caller);
        graph.add_edge(Edge {
            source_id: caller_id,
            target_id: callee_id,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        let results = search_graph_only(&graph, "callers of authenticate");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "login");
    }

    #[test]
    fn search_graph_callees_of() {
        let mut graph = CodeGraph::new();
        let caller = make_symbol("login", "src/login.rs");
        let callee = make_symbol("authenticate", "src/auth.rs");
        let caller_id = caller.id;
        let callee_id = callee.id;
        graph.add_symbol(caller);
        graph.add_symbol(callee);
        graph.add_edge(Edge {
            source_id: caller_id,
            target_id: callee_id,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        let results = search_graph_only(&graph, "callees of login");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "authenticate");
    }

    #[test]
    fn search_graph_definition_of() {
        let mut graph = CodeGraph::new();
        let sym = make_symbol("process_data", "src/processor.rs");
        graph.add_symbol(sym);

        let results = search_graph_only(&graph, "definition of process_data");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "process_data");
    }

    #[test]
    fn search_graph_implementors_of() {
        let mut graph = CodeGraph::new();
        let trait_sym = Symbol {
            kind: SymbolKind::Trait,
            ..make_symbol("Handler", "src/traits.rs")
        };
        let impl_sym = Symbol {
            kind: SymbolKind::Struct,
            ..make_symbol("MyHandler", "src/impl.rs")
        };
        let trait_id = trait_sym.id;
        let impl_id = impl_sym.id;
        graph.add_symbol(trait_sym);
        graph.add_symbol(impl_sym);
        graph.add_edge(Edge {
            source_id: impl_id,
            target_id: trait_id,
            kind: EdgeKind::Implements,
            metadata: None,
        });

        let results = search_graph_only(&graph, "implementations of Handler");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "MyHandler");
    }

    #[test]
    fn search_graph_short_query_finds_by_name() {
        let mut graph = CodeGraph::new();
        graph.add_symbol(make_symbol("authenticate", "src/auth.rs"));
        graph.add_symbol(make_symbol("other", "src/other.rs"));

        let results = search_graph_only(&graph, "authenticate");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "authenticate");
    }

    #[test]
    fn search_graph_short_query_finds_by_prefix() {
        let mut graph = CodeGraph::new();
        graph.add_symbol(make_symbol("authentication", "src/auth.rs"));

        let results = search_graph_only(&graph, "auth");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "authentication");
    }

    #[test]
    fn search_graph_empty_for_common_words_only() {
        let mut graph = CodeGraph::new();
        graph.add_symbol(make_symbol("authenticate", "src/auth.rs"));

        let results = search_graph_only(&graph, "the function");
        assert!(results.is_empty());
    }

    #[test]
    fn apply_graph_boost_increases_score_for_matching_symbols() {
        let mut graph = CodeGraph::new();
        let sym = make_symbol("authenticate", "src/auth.rs");
        graph.add_symbol(sym);

        let chunk = make_chunk("src/auth.rs", "fn authenticate() {}");

        let mut results = vec![SearchResult {
            chunk,
            score: 1.0,
            semantic_score: 0.5,
            bm25_score: 0.5,
            show_full_context: false,
        }];

        apply_graph_boost(&mut results, &graph, "authenticate function");
        assert!(results[0].score > 1.0);
    }

    #[test]
    fn apply_graph_boost_no_change_for_unrelated_query() {
        let mut graph = CodeGraph::new();
        let sym = make_symbol("process", "src/process.rs");
        graph.add_symbol(sym);

        let chunk = make_chunk("src/other.rs", "fn other() {}");

        let mut results = vec![SearchResult {
            chunk,
            score: 1.0,
            semantic_score: 0.5,
            bm25_score: 0.5,
            show_full_context: false,
        }];

        apply_graph_boost(&mut results, &graph, "something unrelated");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }
}
