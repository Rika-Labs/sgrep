use std::collections::HashSet;
use std::path::Path;

use globset::{Glob, GlobSet, GlobSetBuilder};
use once_cell::sync::Lazy;

use crate::chunker::CodeChunk;

static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "the", "and", "for", "but", "with", "from", "this", "that", "into", "when", "what", "how",
        "why", "does", "are", "our", "your", "their", "then", "where", "located", "find", "show",
        "get", "can", "will", "should", "would", "could",
    ]
    .into_iter()
    .collect()
});

pub fn extract_keywords(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter_map(|token| {
            let token = token.trim().to_lowercase();
            if token.len() < 3 || STOPWORDS.contains(token.as_str()) {
                None
            } else {
                Some(token)
            }
        })
        .collect()
}

pub fn keyword_score(keywords: &[String], text: &str, path: &Path) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }

    let haystack = text.to_lowercase();
    let path_str = path.to_string_lossy().to_lowercase();
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mut score = 0.0;
    for kw in keywords {
        let kw_lower = kw.as_str();
        if filename.contains(kw_lower) {
            score += 3.0;
        } else if path_str.contains(kw_lower) {
            score += 2.0;
        } else if haystack.contains(kw_lower) {
            score += 1.0;
        }
    }
    score / keywords.len() as f32
}

pub fn matches_filters(filters: &[String], chunk: &CodeChunk) -> bool {
    filters.iter().all(|filter| match filter.split_once('=') {
        Some((key, value)) => match key {
            "lang" | "language" => chunk.language.eq_ignore_ascii_case(value),
            "path" | "file" => chunk
                .path
                .to_string_lossy()
                .to_lowercase()
                .contains(&value.to_lowercase()),
            _ => true,
        },
        None => true,
    })
}

pub fn build_globset(patterns: &[String]) -> Option<GlobSet> {
    if patterns.is_empty() {
        return None;
    }
    let mut builder = GlobSetBuilder::new();
    let mut added = false;
    for pattern in patterns {
        if let Ok(glob) = Glob::new(pattern) {
            builder.add(glob);
            added = true;
        }
    }
    if !added {
        None
    } else {
        builder.build().ok()
    }
}

pub fn glob_matches(globset: Option<&GlobSet>, path: &Path) -> bool {
    globset.map(|set| set.is_match(path)).unwrap_or(true)
}

/// BM25 parameters (tuned for code search)
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Tokenize text for BM25 indexing
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter_map(|token| {
            let token = token.trim().to_lowercase();
            if token.len() >= 2 {
                Some(token)
            } else {
                None
            }
        })
        .collect()
}

/// BM25 index for a collection of documents
#[derive(Default, Clone)]
pub struct Bm25Index {
    /// Document frequency: term -> count of documents containing term
    pub doc_freq: std::collections::HashMap<String, usize>,
    /// Term frequencies per document: doc_idx -> (term -> count)
    pub term_freqs: Vec<std::collections::HashMap<String, usize>>,
    /// Document lengths (token counts)
    pub doc_lengths: Vec<usize>,
    /// Average document length
    pub avg_doc_len: f32,
    /// Total number of documents
    pub num_docs: usize,
}

impl Bm25Index {
    /// Build BM25 index from document texts
    pub fn build(documents: &[&str]) -> Self {
        use std::collections::HashMap;

        let num_docs = documents.len();
        if num_docs == 0 {
            return Self::default();
        }

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut term_freqs: Vec<HashMap<String, usize>> = Vec::with_capacity(num_docs);
        let mut doc_lengths: Vec<usize> = Vec::with_capacity(num_docs);
        let mut total_len = 0usize;

        for doc in documents {
            let tokens = tokenize(doc);
            let doc_len = tokens.len();
            doc_lengths.push(doc_len);
            total_len += doc_len;

            let mut tf: HashMap<String, usize> = HashMap::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            for token in tokens {
                *tf.entry(token.clone()).or_insert(0) += 1;
                if seen.insert(token.clone()) {
                    *doc_freq.entry(token).or_insert(0) += 1;
                }
            }
            term_freqs.push(tf);
        }

        let avg_doc_len = if num_docs > 0 {
            total_len as f32 / num_docs as f32
        } else {
            0.0
        };

        Self {
            doc_freq,
            term_freqs,
            doc_lengths,
            avg_doc_len,
            num_docs,
        }
    }

    /// Compute BM25 score for a query against a document
    pub fn score(&self, query: &str, doc_idx: usize) -> f32 {
        if doc_idx >= self.num_docs {
            return 0.0;
        }

        let query_tokens = tokenize(query);
        let doc_len = self.doc_lengths[doc_idx] as f32;
        let tf_map = &self.term_freqs[doc_idx];

        let mut score = 0.0;
        for term in &query_tokens {
            let tf = *tf_map.get(term).unwrap_or(&0) as f32;
            let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;

            if df == 0.0 || tf == 0.0 {
                continue;
            }

            // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            let idf = ((self.num_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            // TF component with length normalization
            let tf_norm = (tf * (BM25_K1 + 1.0))
                / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / self.avg_doc_len));

            score += idf * tf_norm;
        }
        score
    }

    /// Score all documents and return sorted indices by score (descending)
    #[allow(dead_code)]
    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = (0..self.num_docs)
            .map(|i| (i, self.score(query, i)))
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn sample_chunk(language: &str, path: &str) -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: language.to_string(),
            start_line: 1,
            end_line: 5,
            text: "fn auth_logic() {}".into(),
            hash: String::new(),
            modified_at: Utc::now(),
        }
    }

    // === BM25 Tests ===

    #[test]
    fn test_tokenize_extracts_words() {
        let tokens = tokenize("fn authenticate_user() { let x = 1; }");
        assert!(tokens.contains(&"fn".to_string()));
        assert!(tokens.contains(&"authenticate_user".to_string()));
        assert!(tokens.contains(&"let".to_string()));
    }

    #[test]
    fn test_tokenize_filters_short_tokens() {
        let tokens = tokenize("a b c ab cd");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"b".to_string()));
        assert!(tokens.contains(&"ab".to_string()));
        assert!(tokens.contains(&"cd".to_string()));
    }

    #[test]
    fn test_bm25_index_creation() {
        let docs = vec![
            "fn authenticate() { check_password(); }",
            "fn validate_input() { sanitize(); }",
            "fn authenticate_user() { authenticate(); validate(); }",
        ];
        let doc_refs: Vec<&str> = docs.iter().map(|s| &**s).collect();
        let index = Bm25Index::build(&doc_refs);

        assert_eq!(index.num_docs, 3);
        assert!(index.avg_doc_len > 0.0);
        assert!(index.doc_freq.get("authenticate").is_some());
        assert_eq!(*index.doc_freq.get("authenticate").unwrap(), 2); // appears in 2 docs
    }

    #[test]
    fn test_bm25_search_returns_relevant_docs() {
        let docs = vec![
            "fn authenticate() { check_password(); }",
            "fn validate_input() { sanitize(); }",
            "fn authenticate_user() { authenticate(); validate(); }",
        ];
        let doc_refs: Vec<&str> = docs.iter().map(|s| &**s).collect();
        let index = Bm25Index::build(&doc_refs);

        let results = index.search("authenticate", 10);

        // Doc 0 and 2 should have positive scores (contain "authenticate")
        assert!(!results.is_empty());
        let result_indices: Vec<usize> = results.iter().map(|(i, _)| *i).collect();
        assert!(result_indices.contains(&0) || result_indices.contains(&2));
    }

    #[test]
    fn test_bm25_score_zero_for_missing_terms() {
        let docs = vec!["fn foo() {}", "fn bar() {}"];
        let doc_refs: Vec<&str> = docs.iter().map(|s| &**s).collect();
        let index = Bm25Index::build(&doc_refs);

        let score = index.score("authenticate", 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_bm25_prefers_rare_terms() {
        let docs = vec![
            "fn common() { common(); common(); }", // "common" appears 3 times
            "fn common() { rare(); }",             // "rare" appears 1 time, "common" 1 time
            "fn common() {}",                      // "common" appears 1 time
        ];
        let doc_refs: Vec<&str> = docs.iter().map(|s| &**s).collect();
        let index = Bm25Index::build(&doc_refs);

        // "rare" should score higher than "common" for doc 1 since it's rarer
        let rare_score = index.score("rare", 1);
        let common_score = index.score("common", 1);
        assert!(rare_score > common_score);
    }

    #[test]
    fn test_bm25_empty_index() {
        let docs: Vec<&str> = vec![];
        let index = Bm25Index::build(&docs);

        assert_eq!(index.num_docs, 0);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn extracts_keywords_and_scores() {
        let keywords = extract_keywords("where is the auth logic?");
        assert!(keywords.contains(&"auth".into()));
        let score = keyword_score(&keywords, "auth logic lives here", Path::new("test.rs"));
        assert!(score > 0.0);
    }

    #[test]
    fn filters_match_language_and_path() {
        let chunk = sample_chunk("rust", "src/lib.rs");
        assert!(matches_filters(&["lang=rust".into()], &chunk));
        assert!(!matches_filters(&["lang=python".into()], &chunk));
        assert!(matches_filters(&["path=src".into()], &chunk));
    }

    #[test]
    fn glob_patterns_match_paths() {
        let globset = build_globset(&["src/**/*.rs".into()]);
        assert!(glob_matches(globset.as_ref(), Path::new("src/main.rs")));
        assert!(!glob_matches(globset.as_ref(), Path::new("tests/data.txt")));
    }

    #[test]
    fn keyword_score_zero_with_no_keywords() {
        let score = keyword_score(&[], "anything", Path::new("file.rs"));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn matches_filters_handles_invalid_format() {
        let chunk = sample_chunk("rust", "src/lib.rs");
        assert!(matches_filters(&["nonsense".into()], &chunk));
        assert!(matches_filters(&["keyvalue".into()], &chunk));
    }

    #[test]
    fn matches_filters_allows_unknown_keys() {
        let chunk = sample_chunk("rust", "src/lib.rs");
        assert!(matches_filters(&["owner=alice".into()], &chunk));
    }

    #[test]
    fn keyword_score_prefers_path_matches() {
        let keywords = vec!["auth".into()];
        let score = keyword_score(&keywords, "irrelevant body", Path::new("src/auth/mod.rs"));
        assert!(score >= 2.0);
    }

    #[test]
    fn build_globset_returns_none_for_invalid_patterns() {
        let globset = build_globset(&["[".into()]);
        assert!(globset.is_none());
    }

    #[test]
    fn glob_matches_defaults_to_true_without_globset() {
        assert!(glob_matches(None, Path::new("any/path.rs")));
    }
}
