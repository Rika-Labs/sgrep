use crate::store::MmapIndex;

pub fn quantize_to_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = vector.len().div_ceil(64);
    let mut binary = vec![0u64; num_words];
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            binary[i / 64] |= 1u64 << (i % 64);
        }
    }
    binary
}

pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

pub fn binary_shortlist(
    query_binary: &[u64],
    index_binary: &[Vec<u64>],
    shortlist_size: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, u32)> = index_binary
        .iter()
        .enumerate()
        .map(|(i, v)| (i, hamming_distance(query_binary, v)))
        .collect();

    if distances.len() > shortlist_size {
        distances.select_nth_unstable_by_key(shortlist_size, |&(_, d)| d);
        distances.truncate(shortlist_size);
    }

    distances.into_iter().map(|(i, _)| i).collect()
}

pub fn binary_shortlist_precomputed(
    query_binary: &[u64],
    index: &MmapIndex,
    shortlist_size: usize,
) -> Vec<usize> {
    let mut distances: Vec<(usize, u32)> = (0..index.len())
        .filter_map(|i| {
            index
                .get_binary_vector(i)
                .map(|v| (i, hamming_distance(query_binary, v)))
        })
        .collect();

    if distances.len() > shortlist_size {
        distances.select_nth_unstable_by_key(shortlist_size, |&(_, d)| d);
        distances.truncate(shortlist_size);
    }

    distances.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_to_binary_sets_bits_for_positive_values() {
        let vector = vec![1.0, -0.5, 0.3, -0.1, 0.0, 0.001];
        let binary = quantize_to_binary(&vector);

        assert_eq!(binary.len(), 1);
        assert_eq!(binary[0] & 0b000001, 1);
        assert_eq!(binary[0] & 0b000010, 0);
        assert_eq!(binary[0] & 0b000100, 0b000100);
        assert_eq!(binary[0] & 0b001000, 0);
        assert_eq!(binary[0] & 0b010000, 0);
        assert_eq!(binary[0] & 0b100000, 0b100000);
    }

    #[test]
    fn quantize_to_binary_handles_large_vectors() {
        let vector: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let binary = quantize_to_binary(&vector);

        assert_eq!(binary.len(), 2);
        for word in &binary {
            assert_eq!(word.count_ones(), 32);
        }
    }

    #[test]
    fn hamming_distance_identical_vectors() {
        let a = vec![0b1010101010u64];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_distance_completely_different() {
        let a = vec![0u64];
        let b = vec![u64::MAX];
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    #[test]
    fn hamming_distance_partial_difference() {
        let a = vec![0b1111_0000u64];
        let b = vec![0b1111_1111u64];
        assert_eq!(hamming_distance(&a, &b), 4);
    }

    #[test]
    fn binary_shortlist_returns_closest_candidates() {
        let query = vec![0b1111u64];
        let index_binary = vec![
            vec![0b1111u64],
            vec![0b0000u64],
            vec![0b1110u64],
            vec![0b0001u64],
        ];

        let shortlist = binary_shortlist(&query, &index_binary, 2);

        assert_eq!(shortlist.len(), 2);
        assert!(shortlist.contains(&0));
        assert!(shortlist.contains(&2));
    }
}
