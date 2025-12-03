//! File classification patterns for content-based scoring.
//!
//! Patterns are organized by category for easier maintenance and extension.
//! This module extracts hardcoded patterns from `content_based_file_boost()`
//! to improve maintainability (DRY principle).

/// Test file content patterns (case-insensitive matching).
/// Used to detect test-like content within source files.
pub const TEST_CONTENT_PATTERNS: &[&str] = &[
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

/// Test directory/file path patterns.
/// Used to identify test files by their path.
pub const PATH_TEST_INDICATORS: &[&str] = &[
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

/// Example directory/file path patterns.
/// Used to identify example/demo files.
pub const PATH_EXAMPLE_INDICATORS: &[&str] = &[
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

/// Implementation/source directory patterns.
/// Used to identify core implementation files.
pub const PATH_IMPL_INDICATORS: &[&str] = &[
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

/// Check if a path matches any of the given patterns.
#[inline]
pub fn path_matches_any(path_lower: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| path_lower.contains(p))
}

/// Count test term occurrences in content.
pub fn count_test_terms(content_lower: &str) -> usize {
    TEST_CONTENT_PATTERNS
        .iter()
        .map(|p| content_lower.matches(p).count())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_matches_any_finds_test_paths() {
        assert!(path_matches_any("/src/tests/foo.rs", PATH_TEST_INDICATORS));
        assert!(path_matches_any(
            "/app/__tests__/bar.ts",
            PATH_TEST_INDICATORS
        ));
        assert!(path_matches_any("foo_test.go", PATH_TEST_INDICATORS));
    }

    #[test]
    fn path_matches_any_finds_example_paths() {
        assert!(path_matches_any(
            "/examples/demo.rs",
            PATH_EXAMPLE_INDICATORS
        ));
        assert!(path_matches_any(
            "/playground/test.ts",
            PATH_EXAMPLE_INDICATORS
        ));
    }

    #[test]
    fn path_matches_any_finds_impl_paths() {
        assert!(path_matches_any("/src/lib.rs", PATH_IMPL_INDICATORS));
        assert!(path_matches_any(
            "/app/controllers/user.ts",
            PATH_IMPL_INDICATORS
        ));
    }

    #[test]
    fn path_matches_any_returns_false_for_no_match() {
        assert!(!path_matches_any("/random/path.txt", PATH_TEST_INDICATORS));
        assert!(!path_matches_any("/other/file.rs", PATH_EXAMPLE_INDICATORS));
    }

    #[test]
    fn count_test_terms_finds_patterns() {
        let content = "fn test_foo() { assert!(true); expect(1).to_be(1); }";
        let count = count_test_terms(&content.to_lowercase());
        assert!(count >= 2); // at least "assert" and "expect("
    }

    #[test]
    fn count_test_terms_returns_zero_for_no_patterns() {
        let content = "fn main() { println!(\"hello\"); }";
        let count = count_test_terms(&content.to_lowercase());
        assert_eq!(count, 0);
    }

    #[test]
    fn pattern_arrays_are_not_empty() {
        assert!(!TEST_CONTENT_PATTERNS.is_empty());
        assert!(!PATH_TEST_INDICATORS.is_empty());
        assert!(!PATH_EXAMPLE_INDICATORS.is_empty());
        assert!(!PATH_IMPL_INDICATORS.is_empty());
    }
}
