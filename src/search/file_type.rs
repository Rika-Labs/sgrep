use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileType {
    #[default]
    Implementation,
    Test,
    Documentation,
    Generated,
}

const IMPLEMENTATION_BOOST: f32 = 1.0;
const TEST_PENALTY: f32 = 0.8;
const DOC_PENALTY: f32 = 0.7;
const GENERATED_PENALTY: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct FileTypePriority;

impl Default for FileTypePriority {
    fn default() -> Self {
        Self
    }
}

impl FileTypePriority {
    pub fn multiplier(&self, file_type: FileType) -> f32 {
        match file_type {
            FileType::Implementation => IMPLEMENTATION_BOOST,
            FileType::Test => TEST_PENALTY,
            FileType::Documentation => DOC_PENALTY,
            FileType::Generated => GENERATED_PENALTY,
        }
    }
}

// Compiled regex patterns for classification (case-insensitive)
// Order of checking: Generated > Test > Documentation > Implementation (default)

static TEST_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        # Directory-based patterns
        (^|/)tests?/|
        (^|/)specs?/|
        (^|/)__tests__/|
        (^|/)e2e/|
        (^|/)integration[-_]?tests?/|
        # Suffix patterns by language
        _test\.(rs|go|py|ts|js|tsx|jsx|rb|ex|exs)$|
        _spec\.(rs|rb|ex|exs)$|
        \.test\.(ts|tsx|js|jsx|mjs|cjs)$|
        \.spec\.(ts|tsx|js|jsx|mjs|cjs|scala)$|
        # Prefix patterns
        (^|/)test_[^/]*\.py$|
        # Class-name patterns (Java/Kotlin/Scala)
        Test\.(java|kt|scala)$|
        Tests\.(java|kt|scala)$|
        Spec\.(java|kt|scala)$|
        # Test config files
        (^|/)jest\.config\.|
        (^|/)vitest\.config\.|
        (^|/)pytest\.ini$|
        (^|/)conftest\.py$
    ",
    )
    .expect("TEST_PATTERN regex should compile")
});

static DOC_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        # Directory-based patterns
        (^|/)docs?/|
        (^|/)documentation/|
        (^|/)examples?/|
        (^|/)samples?/|
        # File extensions
        \.(md|mdx|rst|adoc|txt)$|
        # Specific files (case insensitive)
        (^|/)readme|
        (^|/)changelog|
        (^|/)license|
        (^|/)contributing|
        (^|/)code_of_conduct|
        (^|/)security\.md$|
        (^|/)authors|
        (^|/)history
    ",
    )
    .expect("DOC_PATTERN regex should compile")
});

static GENERATED_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        # Directory-based patterns (vendor, generated output)
        (^|/)vendor/|
        (^|/)node_modules/|
        (^|/)dist/|
        (^|/)build/|
        (^|/)out/|
        (^|/)target/|
        (^|/)\.next/|
        (^|/)\.nuxt/|
        (^|/)coverage/|
        # Generated file patterns
        \.gen\.(rs|go|ts|js|py)$|
        \.generated\.|
        # Protobuf generated
        _pb2\.py$|
        _pb2_grpc\.py$|
        \.pb\.(go|rs|cc|h)$|
        # GraphQL generated
        \.graphql\.(ts|js)$|
        # OpenAPI/Swagger generated
        (^|/)openapi[-_]?client/|
        # Lock files (less relevant for code search)
        (^|/)package-lock\.json$|
        (^|/)yarn\.lock$|
        (^|/)Cargo\.lock$|
        (^|/)poetry\.lock$|
        (^|/)Gemfile\.lock$|
        # Minified files
        \.min\.(js|css)$|
        \.bundle\.(js|css)$
    ",
    )
    .expect("GENERATED_PATTERN regex should compile")
});

/// Classify a file path into a FileType category
pub fn classify_path(path: &Path) -> FileType {
    let path_str = path.to_string_lossy();

    // Order matters: check most specific/restrictive first
    // Generated files are often in vendor/node_modules which should take precedence
    if GENERATED_PATTERN.is_match(&path_str) {
        FileType::Generated
    } else if TEST_PATTERN.is_match(&path_str) {
        FileType::Test
    } else if DOC_PATTERN.is_match(&path_str) {
        FileType::Documentation
    } else {
        FileType::Implementation
    }
}

/// Apply file type prioritization to a score
pub fn apply_priority(score: f32, path: &Path, priority: &FileTypePriority) -> f32 {
    let file_type = classify_path(path);
    score * priority.multiplier(file_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_rust_test_files() {
        assert_eq!(
            classify_path(Path::new("src/auth_test.rs")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("src/auth_spec.rs")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("tests/integration.rs")),
            FileType::Test
        );
    }

    #[test]
    fn classify_js_ts_test_files() {
        assert_eq!(
            classify_path(Path::new("src/__tests__/auth.ts")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("src/auth.test.ts")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("src/auth.spec.tsx")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("e2e/login.test.ts")),
            FileType::Test
        );
    }

    #[test]
    fn classify_python_test_files() {
        assert_eq!(
            classify_path(Path::new("tests/test_auth.py")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("src/auth_test.py")),
            FileType::Test
        );
        assert_eq!(classify_path(Path::new("conftest.py")), FileType::Test);
    }

    #[test]
    fn classify_go_test_files() {
        assert_eq!(
            classify_path(Path::new("pkg/auth_test.go")),
            FileType::Test
        );
    }

    #[test]
    fn classify_java_test_files() {
        assert_eq!(
            classify_path(Path::new("src/AuthTest.java")),
            FileType::Test
        );
        assert_eq!(
            classify_path(Path::new("src/AuthSpec.scala")),
            FileType::Test
        );
    }

    #[test]
    fn classify_doc_files() {
        assert_eq!(
            classify_path(Path::new("docs/api.md")),
            FileType::Documentation
        );
        assert_eq!(
            classify_path(Path::new("README.md")),
            FileType::Documentation
        );
        assert_eq!(
            classify_path(Path::new("examples/basic.rs")),
            FileType::Documentation
        );
        assert_eq!(
            classify_path(Path::new("CHANGELOG.md")),
            FileType::Documentation
        );
        assert_eq!(
            classify_path(Path::new("doc/guide.rst")),
            FileType::Documentation
        );
    }

    #[test]
    fn classify_generated_files() {
        assert_eq!(
            classify_path(Path::new("vendor/lib/foo.rs")),
            FileType::Generated
        );
        assert_eq!(
            classify_path(Path::new("node_modules/lodash/index.js")),
            FileType::Generated
        );
        assert_eq!(
            classify_path(Path::new("src/schema.gen.rs")),
            FileType::Generated
        );
        assert_eq!(
            classify_path(Path::new("proto/api.pb.go")),
            FileType::Generated
        );
        assert_eq!(
            classify_path(Path::new("api/client_pb2.py")),
            FileType::Generated
        );
        assert_eq!(
            classify_path(Path::new("dist/bundle.min.js")),
            FileType::Generated
        );
    }

    #[test]
    fn classify_implementation_files() {
        assert_eq!(
            classify_path(Path::new("src/auth.rs")),
            FileType::Implementation
        );
        assert_eq!(
            classify_path(Path::new("lib/utils.py")),
            FileType::Implementation
        );
        assert_eq!(
            classify_path(Path::new("pkg/server/handler.go")),
            FileType::Implementation
        );
        assert_eq!(
            classify_path(Path::new("src/components/Button.tsx")),
            FileType::Implementation
        );
    }

    #[test]
    fn priority_applies_correct_multipliers() {
        let priority = FileTypePriority::default();

        let impl_score = apply_priority(1.0, Path::new("src/auth.rs"), &priority);
        assert!((impl_score - 1.0).abs() < 1e-6);

        let test_score = apply_priority(1.0, Path::new("tests/auth_test.rs"), &priority);
        assert!((test_score - 0.8).abs() < 1e-6);

        let doc_score = apply_priority(1.0, Path::new("docs/api.md"), &priority);
        assert!((doc_score - 0.7).abs() < 1e-6);

        let gen_score = apply_priority(1.0, Path::new("vendor/lib.rs"), &priority);
        assert!((gen_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn generated_takes_precedence_over_test() {
        assert_eq!(
            classify_path(Path::new("vendor/tests/foo_test.rs")),
            FileType::Generated
        );
    }

    #[test]
    fn test_takes_precedence_over_doc() {
        assert_eq!(
            classify_path(Path::new("examples/foo_test.rs")),
            FileType::Test
        );
    }
}
