use std::sync::Arc;

use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use moka::sync::Cache;

const DEFAULT_MAX_CACHE: u64 = 50_000;
pub const DEFAULT_VECTOR_DIM: usize = 768;

#[derive(Clone)]
pub struct Embedder {
    cache: Cache<String, Arc<Vec<f32>>>,
    model: Arc<std::sync::Mutex<TextEmbedding>>,
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHE)
    }
}

impl Embedder {
    pub fn new(max_cache: u64) -> Self {
        tracing::info!("Initializing nomic-embed-text-v1.5 model...");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::NomicEmbedTextV15)
                .with_show_download_progress(true)
        ).expect("Failed to initialize embedding model");
        tracing::info!("✓ Embedding model loaded");
        Self {
            cache: Cache::builder().max_capacity(max_cache).build(),
            model: Arc::new(std::sync::Mutex::new(model)),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(vec) = self.cache.get(text) {
            return Ok(vec.as_ref().clone());
        }
        let mut model = self.model.lock().unwrap();
        let embeddings = model.embed(vec![text], None)?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;
        self.cache.insert(text.to_string(), Arc::new(vector.clone()));
        Ok(vector)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut uncached = Vec::new();
        let mut uncached_indices = Vec::new();
        let mut results = vec![Vec::new(); texts.len()];

        for (i, text) in texts.iter().enumerate() {
            if let Some(vec) = self.cache.get(text) {
                results[i] = vec.as_ref().clone();
            } else {
                uncached.push(text.as_str());
                uncached_indices.push(i);
            }
        }

        if !uncached.is_empty() {
            let mut model = self.model.lock().unwrap();
            let embeddings = model.embed(uncached.clone(), None)?;

            for (embedding, &idx) in embeddings.iter().zip(&uncached_indices) {
                results[idx] = embedding.clone();
                self.cache.insert(texts[idx].clone(), Arc::new(embedding.clone()));
            }
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_have_correct_dimension() {
        let embedder = Embedder::default();
        let vec = embedder.embed("fn login() {}").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
    }

    #[test]
    fn identical_inputs_use_cache() {
        let embedder = Embedder::default();
        let v1 = embedder.embed("auth logic").unwrap();
        let v2 = embedder.embed("auth logic").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn similar_code_has_high_similarity() {
        let embedder = Embedder::default();
        let v1 = embedder.embed("function authenticate user").unwrap();
        let v2 = embedder.embed("function to auth users").unwrap();
        let dot: f32 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|v| v * v).sum::<f32>().sqrt();
        let similarity = dot / (norm1 * norm2);
        assert!(similarity > 0.5, "Similar code should have similarity > 0.5, got {}", similarity);
    }
}
