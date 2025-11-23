use std::collections::HashSet;
use std::path::Path;

use globset::{Glob, GlobSet, GlobSetBuilder};
use once_cell::sync::Lazy;

use crate::chunker::CodeChunk;

static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "the", "and", "for", "but", "with", "from", "this", "that", "into", "when", "what",
        "how", "why", "does", "are", "our", "your", "their", "then",
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

pub fn keyword_score(keywords: &[String], text: &str) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }
    let haystack = text.to_lowercase();
    let hits = keywords
        .iter()
        .filter(|kw| haystack.contains(kw.as_str()))
        .count();
    hits as f32 / keywords.len() as f32
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
        let score = keyword_score(&keywords, "auth logic lives here");
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
}
