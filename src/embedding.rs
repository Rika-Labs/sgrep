use std::convert::TryInto;
use std::sync::Arc;

use anyhow::Result;
use blake3::Hasher;
use moka::sync::Cache;

const DEFAULT_MAX_CACHE: u64 = 50_000;
pub const DEFAULT_VECTOR_DIM: usize = 512;

#[derive(Clone)]
pub struct Embedder {
    cache: Cache<u64, Arc<Vec<f32>>>,
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHE)
    }
}

impl Embedder {
    pub fn new(max_cache: u64) -> Self {
        Self {
            cache: Cache::builder().max_capacity(max_cache).build(),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = hash_key(text);
        if let Some(vec) = self.cache.get(&key) {
            return Ok(vec.as_ref().clone());
        }
        let vector = hashed_embedding(text.as_bytes());
        self.cache.insert(key, Arc::new(vector.clone()));
        Ok(vector)
    }

    pub fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

fn hashed_embedding(bytes: &[u8]) -> Vec<f32> {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut seed = u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap());
    let mut vector = vec![0.0f32; DEFAULT_VECTOR_DIM];
    for value in &mut vector {
        seed = splitmix64(seed);
        let sample = (seed as f64 / u64::MAX as f64) as f32;
        *value = sample * 2.0 - 1.0;
    }
    normalize(&mut vector);
    vector
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn hash_key(text: &str) -> u64 {
    let mut hasher = Hasher::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_normalized_to_expected_dimension() {
        let embedder = Embedder::default();
        let vec = embedder.embed("fn login() {}").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn identical_inputs_share_cache() {
        let embedder = Embedder::default();
        let v1 = embedder.embed("auth logic").unwrap();
        let v2 = embedder.embed("auth logic").unwrap();
        assert_eq!(v1, v2);
    }
}
