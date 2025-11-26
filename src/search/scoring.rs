use chrono::Utc;
use simsimd::SpatialSimilarity;

use crate::chunker::CodeChunk;

pub const RECENCY_HALF_LIFE_HOURS: f32 = 48.0;

#[derive(Clone, Copy)]
pub struct AdaptiveWeights {
    pub semantic: f32,
    pub bm25: f32,
    pub keyword: f32,
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

        let mut semantic = 0.45;
        let mut bm25 = 0.20;
        let mut keyword = 0.15;
        let recency = 0.05;
        let mut file_type = 0.15;

        if is_question {
            semantic += 0.10;
            bm25 -= 0.05;
            file_type -= 0.05;
        }

        if is_short {
            bm25 += 0.10;
            semantic -= 0.05;
            file_type -= 0.05;
        }

        if has_code_symbols {
            keyword += 0.10;
            semantic -= 0.05;
            file_type -= 0.05;
        }

        let total = semantic + bm25 + keyword + recency + file_type;
        Self {
            semantic: semantic / total,
            bm25: bm25 / total,
            keyword: keyword / total,
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
        if is_error_doc { 0.2 } else { 0.1 }
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
}
