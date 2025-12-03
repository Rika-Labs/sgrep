use simsimd::SpatialSimilarity;

use super::SearchResult;

#[derive(Clone, Copy)]
pub struct AdaptiveWeights {
    pub semantic: f32,
    pub bm25: f32,
}

impl AdaptiveWeights {
    #[allow(unused_variables)]
    pub fn from_query(query: &str) -> Self {
        Self {
            semantic: 0.85,
            bm25: 0.15,
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
    fn adaptive_weights_are_fixed() {
        let weights = AdaptiveWeights::from_query("any query");
        assert!((weights.semantic - 0.85).abs() < 1e-6);
        assert!((weights.bm25 - 0.15).abs() < 1e-6);
    }

    #[test]
    fn adaptive_weights_sum_to_one() {
        let weights = AdaptiveWeights::from_query("test query");
        let sum = weights.semantic + weights.bm25;
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
