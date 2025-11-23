use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;

use crate::embedding::{Embedder, cosine_similarity};
use crate::fts;
use crate::store::{ChunkRecord, load_index};

#[derive(Debug, Clone, Serialize)]
pub struct SearchMatch {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub total: usize,
    pub matches: Vec<SearchMatch>,
}

pub struct SearchConfig {
    pub query: String,
    pub max_results: usize,
    pub per_file: usize,
    pub include_content: bool,
    pub lang_filter: Option<String>,
    pub path_filter: Option<String>,
    pub ignore_filter: Option<String>,
}

pub fn search(root: &Path, embedder: &Embedder, config: SearchConfig) -> Result<SearchResponse> {
    if config.query.trim().is_empty() {
        return Ok(SearchResponse {
            query: config.query,
            total: 0,
            matches: Vec::new(),
        });
    }
    let chunks = load_index(root)?;
    if chunks.is_empty() {
        return Ok(SearchResponse {
            query: config.query,
            total: 0,
            matches: Vec::new(),
        });
    }
    let keyword_limit = config.max_results.saturating_mul(30).max(200);
    let bm25_scores = fts::keyword_scores(root, &config.query, keyword_limit)?;
    let query_embedding = embedder.embed(&config.query)?;
    let terms: Vec<String> = config
        .query
        .split_whitespace()
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let candidate_ids: HashSet<u64> = if bm25_scores.is_empty() {
        HashSet::new()
    } else {
        let mut pairs: Vec<(u64, f32)> = bm25_scores
            .iter()
            .map(|(id, score)| (*id, *score))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        pairs
            .into_iter()
            .take(keyword_limit)
            .map(|(id, _)| id)
            .collect()
    };
    let lang_filters = parse_list(&config.lang_filter);
    let path_filters = parse_list(&config.path_filter);
    let ignore_filters = parse_list(&config.ignore_filter);
    let query_tokens = extract_tokens(&config.query);
    let mut matches: Vec<SearchMatch> = chunks
        .par_iter()
        .filter_map(|chunk| {
            if chunk.text.is_empty() {
                return None;
            }
            if !lang_filters.is_empty() {
                if let Some(ext) = Path::new(&chunk.path).extension().and_then(|s| s.to_str()) {
                    if !lang_filters.iter().any(|f| f.eq_ignore_ascii_case(ext)) {
                        return None;
                    }
                }
            }
            if !path_filters.is_empty() {
                let p = &chunk.path;
                if !path_filters.iter().any(|f| p.contains(f)) {
                    return None;
                }
            }
            if !ignore_filters.is_empty() {
                let p = &chunk.path;
                if ignore_filters.iter().any(|f| p.contains(f)) {
                    return None;
                }
            }
            if !candidate_ids.is_empty() && !candidate_ids.contains(&chunk.id) {
                return None;
            }
            let semantic = cosine_similarity(&chunk.embedding, &query_embedding);
            let mut keyword_score = bm25_scores.get(&chunk.id).copied().unwrap_or(0.0);
            if keyword_score == 0.0 && !terms.is_empty() {
                let lower = chunk.text.to_ascii_lowercase();
                for term in terms.iter() {
                    if lower.contains(term) {
                        keyword_score += 1.0;
                    }
                }
            }
            let identifier_boost = identifier_match_boost(&chunk.text, &query_tokens);
            keyword_score *= identifier_boost;
            if semantic <= 0.0 && keyword_score == 0.0 {
                return None;
            }
            let file_type_boost = file_type_multiplier(&chunk.file_type, &chunk.path);
            let path_boost = path_multiplier(&chunk.path);
            let score = (semantic * 0.6 + keyword_score * 0.3 + identifier_boost * 0.1)
                * file_type_boost
                * path_boost;
            let snippet = snippet_for_chunk(chunk, config.include_content);
            let item = SearchMatch {
                path: chunk.path.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                score,
                semantic_score: semantic,
                keyword_score,
                snippet,
            };
            Some(item)
        })
        .collect();
    if matches.is_empty() {
        return Ok(SearchResponse {
            query: config.query,
            total: 0,
            matches,
        });
    }
    matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    let mut per_file_counts: HashMap<String, usize> = HashMap::new();
    let mut limited = Vec::new();
    for item in matches.into_iter() {
        if limited.len() >= config.max_results {
            break;
        }
        let count = per_file_counts.entry(item.path.clone()).or_insert(0usize);
        if *count >= config.per_file {
            continue;
        }
        *count += 1;
        limited.push(item);
    }
    let total = limited.len();
    Ok(SearchResponse {
        query: config.query,
        total,
        matches: limited,
    })
}

fn snippet_for_chunk(chunk: &ChunkRecord, include_full: bool) -> String {
    if chunk.text.is_empty() {
        return String::new();
    }
    if include_full {
        return chunk.text.clone();
    }
    let mut lines = Vec::new();
    for line in chunk.text.lines().take(8) {
        lines.push(line.to_string());
    }
    lines.join("\n")
}

fn parse_list(input: &Option<String>) -> Vec<String> {
    input
        .as_ref()
        .map(|s| {
            s.split(',')
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn extract_tokens(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn identifier_match_boost(text: &str, tokens: &[String]) -> f32 {
    if tokens.is_empty() {
        return 1.0;
    }
    let lower = text.to_ascii_lowercase();
    let mut boost = 1.0;
    for t in tokens {
        if lower.contains(t) {
            boost += 0.2;
        }
    }
    boost
}

fn file_type_multiplier(file_type: &str, path: &str) -> f32 {
    let explicit = match file_type {
        "code" => Some(1.5),
        "doc" => Some(0.6),
        "config" => Some(0.9),
        "data" => Some(0.9),
        _ => None,
    };
    if let Some(v) = explicit {
        return v;
    }
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "java" | "c" | "cpp" => 1.4,
        "md" | "mdx" | "txt" => 0.6,
        "json" | "yaml" | "yml" | "toml" => 0.9,
        _ => 1.0,
    }
}

fn path_multiplier(path: &str) -> f32 {
    if path.contains("src/") || path.contains("lib/") || path.contains("convex/") {
        1.3
    } else if path.contains("docs/")
        || path.contains("data/")
        || path.contains("test/")
        || path.contains("tests/")
    {
        0.4
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::{IndexOptions, index_repository};
    use serial_test::serial;
    use tempfile::tempdir;

    #[test]
    fn empty_query_returns_no_results() {
        let root = Path::new(".");
        let embedder = Embedder::from_env().unwrap();
        let response = search(
            root,
            &embedder,
            SearchConfig {
                query: String::new(),
                max_results: 10,
                per_file: 1,
                include_content: false,
                lang_filter: None,
                path_filter: None,
                ignore_filter: None,
            },
        )
        .unwrap();
        assert_eq!(response.total, 0);
    }

    #[test]
    #[serial]
    fn search_returns_full_content_when_requested() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("example.rs");
        std::fs::write(
            &file_path,
            r#"
fn hello() {
    println!("world");
}
"#,
        )
        .unwrap();
        let data_root = dir.path().join("data");
        let original_data = std::env::var("SGREP_DATA_DIR").ok();
        unsafe {
            std::env::set_var("SGREP_DATA_DIR", &data_root);
        }
        let embedder = Embedder::from_env().unwrap();
        let opts = IndexOptions {
            root: dir.path().to_path_buf(),
            force_reindex: true,
            dry_run: false,
            include_markdown: true,
        };
        index_repository(&embedder, opts).unwrap();

        let response = search(
            dir.path(),
            &embedder,
            SearchConfig {
                query: "hello world".to_string(),
                max_results: 10,
                per_file: 2,
                include_content: true,
                lang_filter: None,
                path_filter: None,
                ignore_filter: None,
            },
        )
        .unwrap();
        assert!(response.total > 0);
        let snippet = &response.matches[0].snippet;
        assert!(snippet.contains("fn hello"));
        assert!(snippet.contains("println!"));

        if let Some(data) = original_data {
            unsafe {
                std::env::set_var("SGREP_DATA_DIR", data);
            }
        } else {
            unsafe {
                std::env::remove_var("SGREP_DATA_DIR");
            }
        }
    }
}
