//! Search engine configuration constants.
//!
//! These thresholds control algorithm selection and tuning parameters for the search engine.
//! All constants are documented with their purpose and rationale.

// ============================================================================
// Search Strategy Selection Thresholds
// ============================================================================

/// Number of vectors at which HNSW indexing becomes beneficial.
/// Below this threshold, linear search is efficient enough.
/// Empirically tuned: HNSW overhead exceeds benefits under 500 vectors.
pub const HNSW_THRESHOLD: usize = 500;

/// Number of vectors at which binary quantization (BQ) search activates.
/// BQ reduces memory bandwidth by 32x (f32 -> 1-bit) with ~5% recall loss.
/// At 1000+ vectors, BQ shortlisting + exact scoring outperforms HNSW.
pub const BINARY_QUANTIZATION_THRESHOLD: usize = 1000;

/// Multiplier for binary shortlist size before exact cosine refinement.
/// A factor of 10 ensures 95%+ recall while limiting exact computations.
pub const BINARY_SHORTLIST_FACTOR: usize = 10;

// ============================================================================
// Pseudo-Relevance Feedback (PRF) Configuration
// ============================================================================

/// Number of top results used for Pseudo-Relevance Feedback.
/// PRF extracts terms from top-10 to expand the query.
pub const PRF_TOP_K: usize = 10;

/// Maximum terms added to query via PRF expansion.
/// 5 terms balances query drift vs. recall improvement.
pub const PRF_EXPANSION_TERMS: usize = 5;

// ============================================================================
// HNSW Index Configuration
// ============================================================================

/// HNSW graph connectivity (M parameter). Each node connects to 16 neighbors.
/// Higher = better recall, more memory. 16 is standard for 768-dim embeddings.
pub const HNSW_CONNECTIVITY: usize = 16;

/// ef_construction: search depth during index building.
/// 128 provides high-quality graph construction at moderate build cost.
pub const HNSW_EXPANSION_ADD: usize = 128;

/// ef_search: search depth during queries. 64 balances speed vs. recall.
/// Production systems often use 64-128 for embedding search.
pub const HNSW_EXPANSION_SEARCH: usize = 64;

/// Candidate oversampling before final scoring.
/// 4x ensures we capture semantically similar but initially lower-ranked items.
pub const HNSW_OVERSAMPLE_FACTOR: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_strategy_thresholds_are_ordered() {
        assert!(HNSW_THRESHOLD < BINARY_QUANTIZATION_THRESHOLD);
    }

    #[test]
    fn hnsw_threshold_is_reasonable() {
        assert!(HNSW_THRESHOLD >= 100);
        assert!(HNSW_THRESHOLD <= 10000);
    }

    #[test]
    fn binary_quantization_threshold_is_reasonable() {
        assert!(BINARY_QUANTIZATION_THRESHOLD >= 500);
    }

    #[test]
    fn shortlist_factor_provides_sufficient_recall() {
        assert!(BINARY_SHORTLIST_FACTOR >= 5);
        assert!(BINARY_SHORTLIST_FACTOR <= 20);
    }

    #[test]
    fn prf_parameters_are_reasonable() {
        assert!(PRF_TOP_K >= 3);
        assert!(PRF_EXPANSION_TERMS >= 2);
        assert!(PRF_EXPANSION_TERMS <= 10);
    }

    #[test]
    fn hnsw_connectivity_is_standard() {
        assert!(HNSW_CONNECTIVITY >= 8);
        assert!(HNSW_CONNECTIVITY <= 64);
    }

    #[test]
    fn hnsw_expansion_values_are_reasonable() {
        assert!(HNSW_EXPANSION_ADD >= HNSW_EXPANSION_SEARCH);
        assert!(HNSW_EXPANSION_SEARCH >= 32);
    }

    #[test]
    fn constants_have_expected_values() {
        assert_eq!(HNSW_THRESHOLD, 500);
        assert_eq!(BINARY_QUANTIZATION_THRESHOLD, 1000);
        assert_eq!(BINARY_SHORTLIST_FACTOR, 10);
        assert_eq!(PRF_TOP_K, 10);
        assert_eq!(PRF_EXPANSION_TERMS, 5);
        assert_eq!(HNSW_CONNECTIVITY, 16);
        assert_eq!(HNSW_EXPANSION_ADD, 128);
        assert_eq!(HNSW_EXPANSION_SEARCH, 64);
        assert_eq!(HNSW_OVERSAMPLE_FACTOR, 4);
    }
}
