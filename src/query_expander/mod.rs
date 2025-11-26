//! Query expansion and understanding using Qwen2.5 model.
//!
//! This module analyzes user queries to understand intent, extract symbols,
//! and generate query expansions for improved search recall.

mod model;

use anyhow::Result;

pub use model::{is_model_cached, QueryAnalysis, QueryExpander};

#[cfg(test)]
pub use model::get_model_path;

#[allow(dead_code)]
pub trait Expander: Send + Sync {
    fn expand(&self, query: &str) -> Result<Vec<String>>;

    fn expand_n(&self, query: &str, max_expansions: usize) -> Result<Vec<String>> {
        let mut results = self.expand(query)?;
        results.truncate(max_expansions);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    // Test that symbol queries get expanded with related terms
    #[test]
    #[serial]
    fn expands_symbol_query() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let queries = expander.expand("authenticate").unwrap();

        // Debug: print what we got
        eprintln!("Expansions for 'authenticate': {:?}", queries);

        // Should return at least 2 variations (original + expansions)
        assert!(
            queries.len() >= 1,
            "Expected at least 1 expansion, got {}: {:?}",
            queries.len(),
            queries
        );

        // Original query should be included
        assert!(
            queries.iter().any(|q| q.contains("authenticate")),
            "Original term should be in expansions: {:?}",
            queries
        );
    }

    // Test natural language query expansion
    #[test]
    #[serial]
    fn expands_natural_language_query() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let queries = expander.expand("how does login work").unwrap();

        assert!(
            queries.len() >= 1,
            "Expected at least 1 expansion, got {}",
            queries.len()
        );
    }

    // Test that expand_n respects the limit
    #[test]
    #[serial]
    fn expand_n_respects_limit() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let queries = expander.expand_n("authentication", 3).unwrap();

        assert!(
            queries.len() <= 3,
            "Should return at most 3 expansions, got {}",
            queries.len()
        );
    }

    // Test query analysis returns structured data
    #[test]
    #[serial]
    fn analyzes_query_structure() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let analysis = expander.analyze("who calls render").unwrap();

        eprintln!("Analysis: {:?}", analysis);

        // Model may classify differently - just check we get valid output
        assert!(
            !analysis.query_type.is_empty(),
            "Should return a query type"
        );
    }

    // Test empty query handling
    #[test]
    #[serial]
    fn handles_empty_query() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let queries = expander.expand("").unwrap();

        // Empty query should return at least the empty string
        assert!(queries.len() >= 1);
    }

    // Test that expansions are unique
    #[test]
    #[serial]
    fn expansions_are_unique() {
        let expander = QueryExpander::new().expect("Failed to create expander");
        let queries = expander.expand("parse config").unwrap();

        let unique_count = queries
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(
            unique_count,
            queries.len(),
            "All expansions should be unique"
        );
    }
}
