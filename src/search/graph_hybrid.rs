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
        "the", "and", "for", "with", "from", "this", "that", "into",
        "when", "what", "how", "why", "does", "are", "our", "your",
        "their", "then", "where", "find", "show", "get", "can", "will",
        "should", "would", "could", "function", "method", "class",
        "file", "code", "implement", "implementation", "definition",
        "callers", "callees", "imports", "exports",
    ];
    COMMON_WORDS.contains(&word.to_lowercase().as_str())
}

pub fn search_graph_only(
    graph: &CodeGraph,
    query: &str,
) -> Vec<Symbol> {
    let query_lower = query.to_lowercase();

    if query_lower.contains("callers of") || query_lower.contains("who calls") {
        if let Some(name) = extract_symbol_name_from_query(query, &["callers of", "who calls", "what calls"]) {
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
        if let Some(name) = extract_symbol_name_from_query(query, &["definition of", "find definition", "go to definition"]) {
            return graph.find_by_name(&name).into_iter().cloned().collect();
        }
    }

    if query_lower.contains("implementations of") || query_lower.contains("implementors of") {
        if let Some(name) = extract_symbol_name_from_query(query, &["implementations of", "implementors of"]) {
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

pub fn apply_graph_boost(
    results: &mut [SearchResult],
    graph: &CodeGraph,
    query: &str,
) {
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
