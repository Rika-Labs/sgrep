use chrono::Utc;
use simsimd::SpatialSimilarity;

use crate::chunker::CodeChunk;

pub const RECENCY_HALF_LIFE_HOURS: f32 = 48.0;

pub const MIN_SEMANTIC_WEIGHT: f32 = 0.60;

#[derive(Clone, Copy)]
pub struct AdaptiveWeights {
    pub semantic: f32,
    pub bm25: f32,
    pub recency: f32,
    pub file_type: f32,
}

impl AdaptiveWeights {
    pub fn from_query(query: &str) -> Self {
        let word_count = query.split_whitespace().count();
        let query_lower = query.to_lowercase();

        let is_short = word_count <= 2;
        let is_question = query_lower.starts_with("how ")
            || query_lower.starts_with("where ")
            || query_lower.starts_with("what ")
            || query_lower.starts_with("why ")
            || query_lower.starts_with("when ")
            || query_lower.starts_with("which ");
        let has_code_symbols = query.chars().any(|c| "(){}[]<>::->=>".contains(c));

        let mut semantic = 0.60;
        let mut bm25 = 0.20;
        let recency = 0.05;
        let mut file_type = 0.15;

        if is_question {
            semantic += 0.05;
            bm25 -= 0.05;
        }

        if is_short {
            bm25 += 0.05;
            file_type -= 0.05;
        }

        if has_code_symbols {
            bm25 += 0.05;
            file_type -= 0.05;
        }

        if semantic < MIN_SEMANTIC_WEIGHT {
            let deficit = MIN_SEMANTIC_WEIGHT - semantic;
            semantic = MIN_SEMANTIC_WEIGHT;
            bm25 = (bm25 - deficit).max(0.05);
        }

        let total = semantic + bm25 + recency + file_type;
        Self {
            semantic: semantic / total,
            bm25: bm25 / total,
            recency: recency / total,
            file_type: file_type / total,
        }
    }
}

pub fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    match f32::cosine(lhs, rhs) {
        Some(distance) => ((1.0 - distance) as f32).clamp(-1.0, 1.0),
        None => cosine_similarity_scalar(lhs, rhs),
    }
}

pub fn cosine_similarity_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dot: f32 = lhs.iter().zip(rhs).map(|(a, b)| a * b).sum();
    let norm_l: f32 = lhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_r: f32 = rhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_l == 0.0 || norm_r == 0.0 {
        return 0.0;
    }
    (dot / (norm_l * norm_r)).clamp(-1.0, 1.0)
}

pub fn recency_boost(chunk: &CodeChunk) -> f32 {
    let age_hours = (Utc::now() - chunk.modified_at).num_hours().max(0) as f32;
    1.0 / (1.0 + age_hours / RECENCY_HALF_LIFE_HOURS)
}

pub fn normalize_bm25_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }

    let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_score - min_score;

    if range < f32::EPSILON {
        return vec![0.5; scores.len()];
    }

    scores
        .iter()
        .map(|&s| ((s - min_score) / range).clamp(0.0, 1.0))
        .collect()
}

pub fn content_based_file_boost(chunk: &CodeChunk) -> f32 {
    let content_lower = chunk.text.to_lowercase();
    let path_str = chunk.path.to_string_lossy().to_lowercase();
    let word_count = chunk.text.split_whitespace().count().max(1) as f32;

    let test_content_patterns = [
        "assert",
        "expect(",
        ".tobe(",
        ".toequal(",
        "should.",
        "mock",
        "stub",
        "fake",
        "fixture",
        "beforeeach",
        "aftereach",
        "beforeall",
        "afterall",
        "describe(",
        "it(\"",
        "it('",
        "test(\"",
        "test('",
        "@test",
        "#[test]",
        "def test_",
        "func test",
    ];

    let test_term_count: usize = test_content_patterns
        .iter()
        .map(|p| content_lower.matches(p).count())
        .sum();

    let content_test_density = (test_term_count as f32 / word_count).min(0.3);

    let path_test_indicators = [
        "test/",
        "/test/",
        "tests/",
        "/tests/",
        "__tests__/",
        "/__tests__/",
        "spec/",
        "/spec/",
        "specs/",
        "/specs/",
        "fixture/",
        "/fixture/",
        "fixtures/",
        "/fixtures/",
        "__testfixtures__/",
        "/__testfixtures__/",
        "testfixtures/",
        "/testfixtures/",
        "testdata/",
        "/testdata/",
        "test_data/",
        "/test_data/",
        "mock/",
        "/mock/",
        "mocks/",
        "/mocks/",
        "e2e/",
        "/e2e/",
        "integration/",
        "/integration/",
        "unit/",
        "/unit/",
        "unit_test/",
        "/unit_test/",
        "test_utils/",
        "/test_utils/",
        "testutils/",
        "/testutils/",
        "test_helpers/",
        "/test_helpers/",
        "testhelpers/",
        "/testhelpers/",
        "test_support/",
        "/test_support/",
        "testsupport/",
        "/testsupport/",
        "test_common/",
        "/test_common/",
        "testcommon/",
        "/testcommon/",
        "testlib/",
        "/testlib/",
        "test_lib/",
        "/test_lib/",
        "conftest",
        "pytest.ini",
        "jest.config",
        "vitest.config",
        ".test.",
        "_test.",
        ".spec.",
        "_spec.",
        ".test.ts",
        ".test.js",
        ".test.tsx",
        ".test.jsx",
        ".spec.ts",
        ".spec.js",
        ".spec.tsx",
        ".spec.jsx",
        "_test.go",
        "_test.py",
        "_test.rs",
        "_test.java",
        "_test.kt",
        "_test.swift",
        "_test.dart",
        "test_",
        "testcase",
        "test_case",
        "testhelper",
        "test_helper",
        "testutil",
        "test_util",
    ];

    let path_example_indicators = [
        "examples/",
        "/examples/",
        "example/",
        "/example/",
        "samples/",
        "/samples/",
        "sample/",
        "/sample/",
        "demo/",
        "/demo/",
        "demos/",
        "/demos/",
        "playground/",
        "/playground/",
        "playgrounds/",
        "/playgrounds/",
        "sandbox/",
        "/sandbox/",
        "sandboxes/",
        "/sandboxes/",
        "scratch/",
        "/scratch/",
        "scratchpad/",
        "/scratchpad/",
        "tutorial/",
        "/tutorial/",
        "tutorials/",
        "/tutorials/",
        "tut/",
        "/tut/",
        "tuts/",
        "/tuts/",
        "example_",
        "sample_",
        "demo_",
    ];

    let path_impl_indicators = [
        "/src/",
        "/lib/",
        "/pkg/",
        "/internal/",
        "/core/",
        "/server/",
        "/client/",
        "/api/",
        "/services/",
        "src/",
        "lib/",
        "pkg/",
        "internal/",
        "core/",
        "server/",
        "client/",
        "api/",
        "services/",
        "/app/",
        "/apps/",
        "/application/",
        "/applications/",
        "app/",
        "apps/",
        "application/",
        "applications/",
        "/components/",
        "/modules/",
        "/utils/",
        "/utilities/",
        "components/",
        "modules/",
        "utils/",
        "utilities/",
        "/common/",
        "/shared/",
        "/public/",
        "/private/",
        "common/",
        "shared/",
        "public/",
        "private/",
        "/main/",
        "/java/",
        "/scala/",
        "/python/",
        "main/",
        "java/",
        "scala/",
        "python/",
        "/include/",
        "/headers/",
        "/bin/",
        "/scripts/",
        "include/",
        "headers/",
        "bin/",
        "scripts/",
        "/domain/",
        "/business/",
        "/logic/",
        "/model/",
        "domain/",
        "business/",
        "logic/",
        "model/",
        "/controllers/",
        "/views/",
        "/models/",
        "/routes/",
        "controllers/",
        "views/",
        "models/",
        "routes/",
        "/handlers/",
        "/middleware/",
        "/providers/",
        "/repositories/",
        "handlers/",
        "middleware/",
        "providers/",
        "repositories/",
        "/entities/",
        "/interfaces/",
        "/types/",
        "/schemas/",
        "entities/",
        "interfaces/",
        "types/",
        "schemas/",
    ];

    let path_is_test = path_test_indicators.iter().any(|p| path_str.contains(p));
    let path_is_example = path_example_indicators.iter().any(|p| path_str.contains(p));
    let path_is_impl = path_impl_indicators.iter().any(|p| path_str.contains(p))
        && !path_is_test
        && !path_is_example;

    let ext = chunk
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let is_doc = matches!(ext.as_str(), "md" | "mdx" | "rst" | "txt" | "adoc");
    let is_error_doc = path_str.contains("/errors/");

    let score = if is_doc {
        if is_error_doc {
            0.2
        } else {
            0.1
        }
    } else if path_is_test {
        0.1
    } else if path_is_example {
        0.2
    } else if path_is_impl {
        1.0 - (content_test_density * 0.3)
    } else {
        0.6 - (content_test_density * 0.3)
    };

    score.clamp(0.05, 1.0)
}

pub fn directory_match_boost(chunk: &CodeChunk, query: &str) -> f32 {
    let path_str = chunk.path.to_string_lossy().to_lowercase();
    let query_terms: Vec<&str> = query
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .collect();

    let dirs: Vec<&str> = path_str
        .split('/')
        .filter(|s| !s.is_empty() && !s.contains('.'))
        .collect();

    let mut boost: f32 = 0.0;
    for term in &query_terms {
        let term_lower = term.to_lowercase();
        for dir in &dirs {
            if dir.contains(&term_lower) || term_lower.contains(dir) {
                boost += 0.15;
            }
        }
    }

    boost.min(0.3)
}

pub fn reexport_file_penalty(chunk: &CodeChunk) -> f32 {
    let filename = chunk
        .path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");

    let is_reexport_file = filename == "mod.rs"
        || filename == "index.ts"
        || filename == "index.js"
        || filename == "index.tsx"
        || filename == "index.jsx"
        || filename == "__init__.py";

    if is_reexport_file {
        -0.25
    } else {
        0.0
    }
}

use crate::fts::stem_word;

pub fn filename_match_boost(chunk: &CodeChunk, query: &str) -> f32 {
    let filename = chunk
        .path
        .file_stem()
        .and_then(|f| f.to_str())
        .unwrap_or("")
        .to_lowercase();
    let filename_stemmed = stem_word(&filename);

    let query_terms: Vec<String> = query
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .map(|w| stem_word(&w.to_lowercase()))
        .collect();

    let mut boost: f32 = 0.0;
    for term in &query_terms {
        // Stemmed match: "ranking" -> "rank", "scoring" -> "scor"
        // allows morphological variants to match
        if filename_stemmed.contains(term) || term.contains(&filename_stemmed) {
            boost += 0.25;
        } else if filename.contains(term) || term.contains(&filename) {
            // Exact substring match (fallback)
            boost += 0.20;
        }
    }

    boost.min(0.35)
}

use crate::graph::{CodeGraph, SymbolKind};

pub fn implementation_boost(chunk: &CodeChunk, graph: Option<&CodeGraph>) -> f32 {
    let graph = match graph {
        Some(g) => g,
        None => return 0.0,
    };

    let symbols = graph.symbols_in_file(&chunk.path);
    let mut boost = 0.0f32;

    for symbol in &symbols {
        // Check if symbol is DEFINED in this chunk (not just used)
        if symbol.start_line >= chunk.start_line && symbol.start_line <= chunk.end_line {
            boost += match symbol.kind {
                SymbolKind::Function | SymbolKind::Method => 0.12,
                SymbolKind::Struct | SymbolKind::Class => 0.15,
                SymbolKind::Trait | SymbolKind::Interface => 0.12,
                _ => 0.08,
            };
        }
    }

    boost.clamp(0.0, 0.25)
}

use super::SearchResult;

pub fn select_top_k(matches: &mut Vec<SearchResult>, k: usize) {
    if matches.len() > k {
        matches.select_nth_unstable_by(k, |a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);
    }
    sort_by_score(matches);
}

pub fn sort_by_score(matches: &mut [SearchResult]) {
    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn make_chunk(text: &str, language: &str, path: &str) -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: language.to_string(),
            start_line: 1,
            end_line: 10,
            text: text.to_string(),
            hash: "test_hash".to_string(),
            modified_at: Utc::now(),
        }
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        let vec = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&vec, &vec) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&vec1, &vec2).abs() < 1e-6);
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&vec1, &vec2) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_has_similarity_zero() {
        let zero = vec![0.0, 0.0, 0.0];
        let nonzero = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&zero, &nonzero), 0.0);
    }

    #[test]
    fn recency_boost_decays_over_time() {
        let recent = make_chunk("test", "rust", "test.rs");
        let mut old = make_chunk("test", "rust", "test.rs");
        old.modified_at = Utc::now() - chrono::Duration::days(30);

        let recent_boost = recency_boost(&recent);
        let old_boost = recency_boost(&old);

        assert!(recent_boost > old_boost);
        assert!(recent_boost <= 1.0);
        assert!(old_boost > 0.0);
    }

    #[test]
    fn normalize_bm25_empty_returns_empty() {
        assert!(normalize_bm25_scores(&[]).is_empty());
    }

    #[test]
    fn normalize_bm25_single_value_returns_half() {
        let result = normalize_bm25_scores(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn normalize_bm25_equal_values_returns_half() {
        let result = normalize_bm25_scores(&[3.0, 3.0, 3.0]);
        assert_eq!(result.len(), 3);
        for v in result {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_bm25_min_max_normalization() {
        let result = normalize_bm25_scores(&[0.0, 5.0, 10.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_bm25_handles_negative_scores() {
        let result = normalize_bm25_scores(&[-10.0, 0.0, 10.0]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_weights_semantic_always_dominant_default() {
        let weights = AdaptiveWeights::from_query("find authentication logic");
        assert!(weights.semantic >= MIN_SEMANTIC_WEIGHT);
        assert!(weights.semantic > weights.bm25);
    }

    #[test]
    fn adaptive_weights_semantic_always_dominant_short_query() {
        let weights = AdaptiveWeights::from_query("HashMap");
        assert!(weights.semantic >= MIN_SEMANTIC_WEIGHT);
        assert!(weights.semantic > weights.bm25);
    }

    #[test]
    fn adaptive_weights_semantic_always_dominant_code_symbols() {
        let weights = AdaptiveWeights::from_query("fn()");
        assert!(weights.semantic >= MIN_SEMANTIC_WEIGHT);
        assert!(weights.semantic > weights.bm25);
    }

    #[test]
    fn adaptive_weights_semantic_always_dominant_short_with_code() {
        let weights = AdaptiveWeights::from_query("Vec<T>");
        assert!(weights.semantic >= MIN_SEMANTIC_WEIGHT);
        assert!(weights.semantic > weights.bm25);
    }

    #[test]
    fn adaptive_weights_question_boosts_semantic() {
        let question = AdaptiveWeights::from_query("how does authentication work");
        let normal = AdaptiveWeights::from_query("authentication implementation");
        assert!(question.semantic > normal.semantic);
    }

    #[test]
    fn adaptive_weights_sum_to_one() {
        let weights = AdaptiveWeights::from_query("test query");
        let sum = weights.semantic + weights.bm25 + weights.recency + weights.file_type;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn content_based_file_boost_doc_files() {
        let doc_chunk = make_chunk("# README", "markdown", "docs/README.md");
        let score = content_based_file_boost(&doc_chunk);
        assert!(score < 0.5, "Doc files should have low boost: {}", score);
    }

    #[test]
    fn content_based_file_boost_error_docs() {
        let error_doc = make_chunk("Error guide", "markdown", "docs/errors/E001.md");
        let score = content_based_file_boost(&error_doc);
        assert!(score > 0.1, "Error docs should have slightly higher boost: {}", score);
    }

    #[test]
    fn content_based_file_boost_test_files() {
        let test_chunk = make_chunk("assert!(true)", "rust", "tests/test_auth.rs");
        let score = content_based_file_boost(&test_chunk);
        assert!(score < 0.5, "Test files should have low boost: {}", score);
    }

    #[test]
    fn content_based_file_boost_example_files() {
        let example_chunk = make_chunk("fn main() {}", "rust", "examples/demo.rs");
        let score = content_based_file_boost(&example_chunk);
        assert!(score < 0.5, "Example files should have low boost: {}", score);
    }

    #[test]
    fn content_based_file_boost_impl_files() {
        let impl_chunk = make_chunk("fn authenticate() {}", "rust", "src/auth.rs");
        let score = content_based_file_boost(&impl_chunk);
        assert!(score > 0.5, "Impl files should have high boost: {}", score);
    }

    #[test]
    fn directory_match_boost_matches_dir() {
        let chunk = make_chunk("code", "rust", "src/graph/extractor.rs");
        let boost = directory_match_boost(&chunk, "graph extraction");
        assert!(boost > 0.0, "Should boost for matching directory: {}", boost);
    }

    #[test]
    fn directory_match_boost_no_match() {
        let chunk = make_chunk("code", "rust", "src/auth/login.rs");
        let boost = directory_match_boost(&chunk, "graph extraction");
        assert_eq!(boost, 0.0, "Should not boost for non-matching directory");
    }

    #[test]
    fn reexport_file_penalty_mod_rs() {
        let mod_chunk = make_chunk("pub mod auth;", "rust", "src/mod.rs");
        let penalty = reexport_file_penalty(&mod_chunk);
        assert!(penalty < 0.0, "mod.rs should have penalty: {}", penalty);
    }

    #[test]
    fn reexport_file_penalty_index_ts() {
        let index_chunk = make_chunk("export * from './auth'", "typescript", "src/index.ts");
        let penalty = reexport_file_penalty(&index_chunk);
        assert!(penalty < 0.0, "index.ts should have penalty: {}", penalty);
    }

    #[test]
    fn reexport_file_penalty_normal_file() {
        let normal_chunk = make_chunk("fn auth() {}", "rust", "src/auth.rs");
        let penalty = reexport_file_penalty(&normal_chunk);
        assert_eq!(penalty, 0.0, "Normal files should have no penalty");
    }

    #[test]
    fn filename_match_boost_exact_match() {
        let chunk = make_chunk("code", "rust", "src/scoring.rs");
        let boost = filename_match_boost(&chunk, "scoring function");
        assert!(boost > 0.0, "Should boost for filename match: {}", boost);
    }

    #[test]
    fn filename_match_boost_stemmed_match() {
        let chunk = make_chunk("code", "rust", "src/extractor.rs");
        let boost = filename_match_boost(&chunk, "extraction logic");
        assert!(boost > 0.0, "Should boost for stemmed match: {}", boost);
    }

    #[test]
    fn filename_match_boost_no_match() {
        let chunk = make_chunk("code", "rust", "src/auth.rs");
        let boost = filename_match_boost(&chunk, "scoring function");
        assert_eq!(boost, 0.0, "Should not boost for non-matching filename");
    }

    #[test]
    fn implementation_boost_without_graph() {
        let chunk = make_chunk("fn test() {}", "rust", "src/test.rs");
        let boost = implementation_boost(&chunk, None);
        assert_eq!(boost, 0.0, "Should return 0 without graph");
    }
}
