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
