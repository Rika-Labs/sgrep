use std::collections::HashSet;

use super::results::SearchResult;
use super::scoring::cosine_similarity;

pub const DEFAULT_SEMANTIC_DEDUP_THRESHOLD: f32 = 0.95;

#[derive(Clone, Debug)]
pub struct DedupOptions {
    pub enabled: bool,
    pub semantic_threshold: f32,
}

impl Default for DedupOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            semantic_threshold: DEFAULT_SEMANTIC_DEDUP_THRESHOLD,
        }
    }
}

pub fn suppress_near_duplicates(
    results: &mut Vec<SearchResult>,
    vectors: &[Vec<f32>],
    options: &DedupOptions,
) {
    if !options.enabled || results.len() <= 1 {
        return;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen_hashes: HashSet<&str> = HashSet::new();
    let mut indices_to_remove: Vec<usize> = Vec::new();

    for (idx, result) in results.iter().enumerate() {
        if seen_hashes.contains(result.chunk.hash.as_str()) {
            indices_to_remove.push(idx);
        } else {
            seen_hashes.insert(&result.chunk.hash);
        }
    }

    if !vectors.is_empty() && vectors.len() == results.len() && options.semantic_threshold < 1.0 {
        let hash_removed: HashSet<usize> = indices_to_remove.iter().cloned().collect();

        for i in 0..results.len() {
            if hash_removed.contains(&i) || indices_to_remove.contains(&i) {
                continue;
            }

            for j in 0..i {
                if hash_removed.contains(&j) || indices_to_remove.contains(&j) {
                    continue;
                }

                let similarity = cosine_similarity(&vectors[i], &vectors[j]);
                if similarity > options.semantic_threshold {
                    indices_to_remove.push(i);
                    break;
                }
            }
        }
    }

    indices_to_remove.sort_by(|a, b| b.cmp(a));
    indices_to_remove.dedup();
    for idx in indices_to_remove {
        if idx < results.len() {
            results.remove(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use chrono::Utc;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn make_chunk_with_hash(text: &str, path: &str, hash: &str) -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: text.to_string(),
            hash: hash.to_string(),
            modified_at: Utc::now(),
        }
    }

    fn make_result(chunk: CodeChunk, score: f32) -> SearchResult {
        SearchResult {
            chunk,
            score,
            semantic_score: score,
            bm25_score: 0.0,
            show_full_context: false,
        }
    }

    #[test]
    fn default_options_are_sensible() {
        let options = DedupOptions::default();
        assert!(options.enabled);
        assert!((options.semantic_threshold - DEFAULT_SEMANTIC_DEDUP_THRESHOLD).abs() < 1e-6);
    }

    #[test]
    fn hash_dedup_removes_exact_duplicates() {
        let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "same_hash");
        let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "same_hash");
        let chunk3 = make_chunk_with_hash("fn bar() {}", "c.rs", "different");

        let mut results = vec![
            make_result(chunk1, 0.9),
            make_result(chunk2, 0.8),
            make_result(chunk3, 0.7),
        ];

        suppress_near_duplicates(&mut results, &[], &DedupOptions::default());

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.hash, "same_hash");
        assert_eq!(results[1].chunk.hash, "different");
    }

    #[test]
    fn semantic_dedup_removes_similar_chunks() {
        let chunk1 = make_chunk_with_hash("fn auth() {}", "a.rs", "h1");
        let chunk2 = make_chunk_with_hash("fn auth2() {}", "b.rs", "h2");
        let chunk3 = make_chunk_with_hash("fn db() {}", "c.rs", "h3");

        let mut results = vec![
            make_result(chunk1, 0.9),
            make_result(chunk2, 0.85),
            make_result(chunk3, 0.7),
        ];

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.99, 0.14, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let options = DedupOptions {
            semantic_threshold: 0.95,
            ..Default::default()
        };

        suppress_near_duplicates(&mut results, &vectors, &options);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn disabled_option_preserves_all() {
        let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "same");
        let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "same");

        let mut results = vec![make_result(chunk1, 0.9), make_result(chunk2, 0.8)];

        let options = DedupOptions {
            enabled: false,
            ..Default::default()
        };

        suppress_near_duplicates(&mut results, &[], &options);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn empty_results_handled() {
        let mut results: Vec<SearchResult> = vec![];
        suppress_near_duplicates(&mut results, &[], &DedupOptions::default());
        assert!(results.is_empty());
    }

    #[test]
    fn single_result_unchanged() {
        let chunk = make_chunk_with_hash("fn foo() {}", "a.rs", "h1");
        let mut results = vec![make_result(chunk, 0.9)];

        suppress_near_duplicates(&mut results, &[vec![1.0, 0.0, 0.0, 0.0]], &DedupOptions::default());

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn ordering_is_deterministic() {
        let chunk1 = make_chunk_with_hash("fn a() {}", "a.rs", "h1");
        let chunk2 = make_chunk_with_hash("fn b() {}", "b.rs", "h2");

        let create_results = || {
            vec![
                make_result(chunk1.clone(), 0.9),
                make_result(chunk2.clone(), 0.8),
            ]
        };

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        let mut results1 = create_results();
        let mut results2 = create_results();

        suppress_near_duplicates(&mut results1, &vectors, &DedupOptions::default());
        suppress_near_duplicates(&mut results2, &vectors, &DedupOptions::default());

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert!((r1.score - r2.score).abs() < 1e-6);
        }
    }
}
