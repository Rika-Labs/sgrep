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
        let mut pairs: Vec<(u64, f32)> = bm25_scores.iter().map(|(id, score)| (*id, *score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        pairs
            .into_iter()
            .take(keyword_limit)
            .map(|(id, _)| id)
            .collect()
    };
    let mut matches: Vec<SearchMatch> = chunks
        .par_iter()
        .filter_map(|chunk| {
            if chunk.text.is_empty() {
                return None;
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
            if semantic <= 0.0 && keyword_score == 0.0 {
                return None;
            }
            let score = semantic * 0.7 + keyword_score * 0.3;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use serial_test::serial;
    use crate::indexer::{IndexOptions, index_repository};

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
        unsafe { std::env::set_var("SGREP_DATA_DIR", &data_root); }
        let embedder = Embedder::from_env().unwrap();
        let opts = IndexOptions {
            root: dir.path().to_path_buf(),
            force_reindex: true,
            dry_run: false,
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
            },
        )
        .unwrap();
        assert!(response.total > 0);
        let snippet = &response.matches[0].snippet;
        assert!(snippet.contains("fn hello"));
        assert!(snippet.contains("println!"));

        if let Some(data) = original_data {
            unsafe { std::env::set_var("SGREP_DATA_DIR", data); }
        } else {
            unsafe { std::env::remove_var("SGREP_DATA_DIR"); }
        }
    }
}
