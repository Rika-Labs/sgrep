use anyhow::Result;

/// Trait for reranking search results
pub trait Reranker: Send + Sync {
    /// Rerank documents against a query, returning indices sorted by relevance (highest first)
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>>;
}

/// Configuration for reranker behavior
#[derive(Clone)]
#[allow(dead_code)]
pub struct RerankerConfig {
    /// Number of extra candidates to fetch before reranking (multiplier)
    pub oversample_factor: usize,
    /// Whether reranking is enabled
    pub enabled: bool,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            oversample_factor: 3,
            enabled: true,
        }
    }
}

#[cfg(not(test))]
#[allow(dead_code)]
mod real_impl {
    use super::*;
    use anyhow::anyhow;
    use fastembed::{TextRerank, RerankInitOptions, RerankerModel};
    use std::sync::Mutex;

    const RERANKER_MODEL: RerankerModel = RerankerModel::BGERerankerBase;

    /// Cross-encoder reranker using fastembed
    pub struct CrossEncoderReranker {
        model: Mutex<TextRerank>,
    }

    impl CrossEncoderReranker {
        pub fn new(show_download_progress: bool) -> Result<Self> {
            let options = RerankInitOptions::new(RERANKER_MODEL)
                .with_show_download_progress(show_download_progress);

            let model = TextRerank::try_new(options)
                .map_err(|e| anyhow!("Failed to initialize reranker: {}", e))?;

            Ok(Self {
                model: Mutex::new(model),
            })
        }
    }

    impl Reranker for CrossEncoderReranker {
        fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
            if documents.is_empty() {
                return Ok(vec![]);
            }

            let mut model = self.model.lock().unwrap();
            let results = model
                .rerank(query, documents.to_vec(), false, None)
                .map_err(|e| anyhow!("Reranking failed: {}", e))?;

            // Results come sorted by score descending
            Ok(results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect())
        }
    }
}

#[cfg(not(test))]
#[allow(unused_imports)]
pub use real_impl::CrossEncoderReranker;

/// Mock reranker for testing - just returns documents in original order with fake scores
#[derive(Default, Clone)]
pub struct MockReranker;

impl Reranker for MockReranker {
    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        // Mock: score based on document length (longer = more relevant)
        let mut scores: Vec<(usize, f32)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, doc.len() as f32 / 100.0))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }
}

/// No-op reranker that preserves original order
#[derive(Default, Clone)]
pub struct NoOpReranker;

impl Reranker for NoOpReranker {
    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        Ok(documents
            .iter()
            .enumerate()
            .map(|(i, _)| (i, 1.0 - (i as f32 / documents.len() as f32)))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_reranker_returns_all_documents() {
        let reranker = MockReranker;
        let docs = vec!["short", "medium length doc", "this is the longest document here"];

        let result = reranker.rerank("query", &docs).unwrap();

        assert_eq!(result.len(), 3);
    }

    #[test]
    fn mock_reranker_prefers_longer_documents() {
        let reranker = MockReranker;
        let docs = vec!["short", "this is the longest document here", "medium length"];

        let result = reranker.rerank("query", &docs).unwrap();

        // Longest doc should be first (index 1)
        assert_eq!(result[0].0, 1);
        // Shortest doc should be last (index 0)
        assert_eq!(result[2].0, 0);
    }

    #[test]
    fn mock_reranker_handles_empty_documents() {
        let reranker = MockReranker;
        let docs: Vec<&str> = vec![];

        let result = reranker.rerank("query", &docs).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn noop_reranker_preserves_order() {
        let reranker = NoOpReranker;
        let docs = vec!["first", "second", "third"];

        let result = reranker.rerank("query", &docs).unwrap();

        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 1);
        assert_eq!(result[2].0, 2);
    }

    #[test]
    fn noop_reranker_assigns_decreasing_scores() {
        let reranker = NoOpReranker;
        let docs = vec!["a", "b", "c"];

        let result = reranker.rerank("query", &docs).unwrap();

        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 > result[2].1);
    }

    #[test]
    fn reranker_config_has_sensible_defaults() {
        let config = RerankerConfig::default();

        assert_eq!(config.oversample_factor, 3);
        assert!(config.enabled);
    }
}
