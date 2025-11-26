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

/// BM25F field boost constants (tuned for code search)
/// These boosts are applied to term frequencies BEFORE the saturation function,
/// following the BM25F algorithm from "Simple BM25 extension to multiple weighted fields"
const BM25F_FILENAME_BOOST: usize = 5;
const BM25F_SYMBOL_BOOST: usize = 5;
const BM25F_PATH_BOOST: usize = 2;

/// A document with multiple fields for BM25F scoring.
/// Following Sourcegraph's approach, we treat all fields as one combined document
/// but boost term frequencies for important fields (filename, symbols) before
/// applying the BM25 saturation function.
#[derive(Debug, Clone, Default)]
pub struct Bm25FDocument {
    /// Main content text
    pub content: String,
    /// File path (for path-based boosting)
    pub path: String,
    /// Filename stem (highest priority for matching)
    pub filename: String,
    /// Symbol names in this chunk (function names, class names, etc.)
    pub symbols: Vec<String>,
}

impl Bm25FDocument {
    pub fn new(content: &str, path: &Path) -> Self {
        let path_str = path.to_string_lossy().to_string();
        let filename = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        Self {
            content: content.to_string(),
            path: path_str,
            filename,
            symbols: Vec::new(),
        }
    }

    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = symbols;
        self
    }
}

/// BM25F index that supports field-level term frequency boosting.
/// This implements the key insight from Sourcegraph's blog:
/// boost term frequencies BEFORE the saturation function, not after.
#[derive(Default, Clone)]
pub struct Bm25FIndex {
    /// Document frequency: term -> count of documents containing term
    pub doc_freq: std::collections::HashMap<String, usize>,
    /// Boosted term frequencies per document: doc_idx -> (term -> boosted_count)
    /// These frequencies already include field boosts
    pub term_freqs: Vec<std::collections::HashMap<String, usize>>,
    /// Document lengths (effective length after boosting)
    pub doc_lengths: Vec<usize>,
    /// Average document length
    pub avg_doc_len: f32,
    /// Total number of documents
    pub num_docs: usize,
}

impl Bm25FIndex {
    /// Build BM25F index from documents with field information.
    /// Term frequencies are boosted based on which field they appear in:
    /// - Filename matches: 5x boost
    /// - Symbol matches: 5x boost
    /// - Path matches: 2x boost
    /// - Content matches: 1x (no boost)
    pub fn build(documents: &[Bm25FDocument]) -> Self {
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
            let mut tf: HashMap<String, usize> = HashMap::new();
            let mut seen: HashSet<String> = HashSet::new();

            // Helper to add tokens with a boost factor
            let mut add_tokens = |text: &str, boost: usize| {
                for token in tokenize(text) {
                    *tf.entry(token.clone()).or_insert(0) += boost;
                    if seen.insert(token.clone()) {
                        *doc_freq.entry(token).or_insert(0) += 1;
                    }
                }
            };

            // Add content tokens with no boost (1x)
            add_tokens(&doc.content, 1);

            // Add path tokens with path boost
            add_tokens(&doc.path, BM25F_PATH_BOOST);

            // Add filename tokens with filename boost (highest priority)
            add_tokens(&doc.filename, BM25F_FILENAME_BOOST);

            // Add symbol tokens with symbol boost
            for symbol in &doc.symbols {
                add_tokens(symbol, BM25F_SYMBOL_BOOST);
            }

            // Effective document length is sum of all boosted term frequencies
            let doc_len: usize = tf.values().sum();
            doc_lengths.push(doc_len);
            total_len += doc_len;

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

    /// Compute BM25F score for a query against a document.
    /// This uses the pre-boosted term frequencies, so the saturation function
    /// is applied to the combined, boosted frequencies as intended by BM25F.
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
            // The key BM25F insight: tf here already includes field boosts,
            // so the saturation function naturally handles cross-field scoring
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

/// Build a BM25F index from chunks, extracting field information from each chunk.
/// This is a convenience function that creates Bm25FDocuments from CodeChunks.
pub fn build_bm25f_index(chunks: &[&CodeChunk], symbols_by_chunk: Option<&[Vec<String>]>) -> Bm25FIndex {
    let documents: Vec<Bm25FDocument> = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut doc = Bm25FDocument::new(&chunk.text, &chunk.path);
            if let Some(symbols) = symbols_by_chunk {
                if let Some(chunk_symbols) = symbols.get(i) {
                    doc = doc.with_symbols(chunk_symbols.clone());
                }
            }
            doc
        })
        .collect();

    Bm25FIndex::build(&documents)
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

    // === BM25F Tests ===

    #[test]
    fn test_bm25f_document_creation() {
        let doc = Bm25FDocument::new("fn main() {}", Path::new("src/main.rs"));
        assert_eq!(doc.content, "fn main() {}");
        assert_eq!(doc.filename, "main");
        assert_eq!(doc.path, "src/main.rs");
        assert!(doc.symbols.is_empty());
    }

    #[test]
    fn test_bm25f_document_with_symbols() {
        let doc = Bm25FDocument::new("fn authenticate() {}", Path::new("src/auth.rs"))
            .with_symbols(vec!["authenticate".to_string(), "login".to_string()]);
        assert_eq!(doc.symbols.len(), 2);
        assert!(doc.symbols.contains(&"authenticate".to_string()));
    }

    #[test]
    fn test_bm25f_index_creation() {
        let docs = vec![
            Bm25FDocument::new("fn foo() {}", Path::new("src/foo.rs")),
            Bm25FDocument::new("fn bar() {}", Path::new("src/bar.rs")),
        ];
        let index = Bm25FIndex::build(&docs);

        assert_eq!(index.num_docs, 2);
        assert!(index.avg_doc_len > 0.0);
    }

    #[test]
    fn test_bm25f_filename_boost() {
        // Doc 0: "auth" only in content
        // Doc 1: "auth" in filename (auth.rs)
        // BM25F should rank doc 1 higher due to filename boost
        let docs = vec![
            Bm25FDocument::new("fn auth() { check(); }", Path::new("src/login.rs")),
            Bm25FDocument::new("fn check() { verify(); }", Path::new("src/auth.rs")),
        ];
        let index = Bm25FIndex::build(&docs);

        let score_content_only = index.score("auth", 0);
        let score_filename = index.score("auth", 1);

        // Filename match should score higher than content-only match
        assert!(
            score_filename > score_content_only,
            "Filename match ({}) should score higher than content match ({})",
            score_filename,
            score_content_only
        );
    }

    #[test]
    fn test_bm25f_symbol_boost() {
        // Doc 0: "extract" only in content
        // Doc 1: "extract" as a symbol name (exact match)
        let docs = vec![
            Bm25FDocument::new("let result = extract();", Path::new("src/main.rs")),
            Bm25FDocument::new("fn helper() {}", Path::new("src/util.rs"))
                .with_symbols(vec!["extract".to_string()]),
        ];
        let index = Bm25FIndex::build(&docs);

        let score_content = index.score("extract", 0);
        let score_symbol = index.score("extract", 1);

        // Symbol match should score higher due to boost
        assert!(
            score_symbol > score_content,
            "Symbol match ({}) should score higher than content match ({})",
            score_symbol,
            score_content
        );
    }

    #[test]
    fn test_bm25f_combined_field_boost() {
        // Test the key BM25F insight: when a term appears in multiple fields,
        // the boosts are combined BEFORE saturation, not after
        let docs = vec![
            // Doc 0: "queue" appears in content only
            Bm25FDocument::new("process the queue item", Path::new("src/process.rs")),
            // Doc 1: "queue" appears in filename AND as a symbol (exact term match)
            Bm25FDocument::new("struct Item {}", Path::new("src/queue.rs"))
                .with_symbols(vec!["queue".to_string()]),
        ];
        let index = Bm25FIndex::build(&docs);

        let score_content = index.score("queue", 0);
        let score_multi_field = index.score("queue", 1);

        // Multi-field match should score significantly higher
        // (filename boost 5x + symbol boost 5x combined, but saturation limits the effect)
        // BM25's saturation function prevents linear scaling, so ~1.5x is expected
        assert!(
            score_multi_field > score_content * 1.5,
            "Multi-field match ({}) should be higher than content match ({}) by at least 1.5x",
            score_multi_field,
            score_content
        );
    }

    #[test]
    fn test_bm25f_respects_saturation() {
        // Test that term frequency saturation still works in BM25F
        // A document with 100 occurrences shouldn't be 100x better than 1 occurrence
        let doc_few = Bm25FDocument::new("auth auth auth", Path::new("src/a.rs"));
        let doc_many = Bm25FDocument::new(
            &"auth ".repeat(100),
            Path::new("src/b.rs"),
        );
        let index = Bm25FIndex::build(&[doc_few, doc_many]);

        let score_few = index.score("auth", 0);
        let score_many = index.score("auth", 1);

        // Score should increase but not proportionally to term count
        assert!(score_many > score_few, "More occurrences should score higher");
        assert!(
            score_many < score_few * 10.0,
            "Saturation should prevent linear scaling: {} vs {}",
            score_many,
            score_few
        );
    }

    #[test]
    fn test_bm25f_empty_index() {
        let docs: Vec<Bm25FDocument> = vec![];
        let index = Bm25FIndex::build(&docs);

        assert_eq!(index.num_docs, 0);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25f_search_ranking() {
        let docs = vec![
            Bm25FDocument::new("fn process() {}", Path::new("src/main.rs")),
            Bm25FDocument::new("fn validate() {}", Path::new("src/auth.rs"))
                .with_symbols(vec!["authenticate".to_string()]),
            Bm25FDocument::new("authenticate users here", Path::new("src/login.rs")),
        ];
        let index = Bm25FIndex::build(&docs);

        let results = index.search("authenticate", 10);

        // Should have 2 results (docs 1 and 2 match)
        assert_eq!(results.len(), 2);

        // Doc 1 (with symbol "authenticate") should rank first
        assert_eq!(
            results[0].0, 1,
            "Document with symbol match should rank first"
        );
    }

    #[test]
    fn test_build_bm25f_index_from_chunks() {
        let chunk1 = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("src/auth.rs"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: "fn authenticate() { verify(); }".to_string(),
            hash: "hash1".to_string(),
            modified_at: Utc::now(),
        };
        let chunk2 = CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("src/main.rs"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 5,
            text: "fn main() { auth(); }".to_string(),
            hash: "hash2".to_string(),
            modified_at: Utc::now(),
        };

        let chunks: Vec<&CodeChunk> = vec![&chunk1, &chunk2];
        let symbols = vec![
            vec!["authenticate".to_string()],
            vec!["main".to_string()],
        ];

        let index = build_bm25f_index(&chunks, Some(&symbols));

        assert_eq!(index.num_docs, 2);

        // "auth" query should prefer chunk1 (filename is auth.rs)
        let score1 = index.score("auth", 0);
        let score2 = index.score("auth", 1);
        assert!(
            score1 > score2,
            "auth.rs ({}) should score higher than main.rs ({}) for 'auth' query",
            score1,
            score2
        );
    }
}
