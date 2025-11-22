use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::{Result, anyhow};
use reqwest::blocking::Client;
use serde::Deserialize;

pub const EMBEDDING_DIM: usize = 256;

#[derive(Debug, Clone)]
pub enum EmbeddingSource {
    LocalHash,
    RemoteHttp(String),
}

#[derive(Debug, Clone)]
pub struct Embedder {
    client: Option<Client>,
    source: EmbeddingSource,
}

#[derive(Deserialize)]
struct RemoteEmbeddingResponse {
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn from_env() -> Result<Self> {
        let remote = std::env::var("SGREP_EMBEDDING_URL").ok();
        Self::with_remote(remote)
    }

    pub fn with_remote(remote: Option<String>) -> Result<Self> {
        let source = match remote {
            Some(url) => EmbeddingSource::RemoteHttp(url),
            None => EmbeddingSource::LocalHash,
        };
        let client = match &source {
            EmbeddingSource::RemoteHttp(_) => Some(Client::builder().build()?),
            EmbeddingSource::LocalHash => None,
        };
        Ok(Self { client, source })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Ok(hash_embed(text));
        }
        match &self.source {
            EmbeddingSource::LocalHash => Ok(hash_embed(text)),
            EmbeddingSource::RemoteHttp(url) => {
                let vector = self.remote_embed(url, text)?;
                Ok(vector)
            }
        }
    }

    fn remote_embed(&self, url: &str, text: &str) -> Result<Vec<f32>> {
        let client = match &self.client {
            Some(c) => c,
            None => return Err(anyhow!("remote embedding client unavailable")),
        };
        let body = serde_json::json!({ "text": text });
        let response = client.post(url).json(&body).send()?;
        if !response.status().is_success() {
            return Err(anyhow!("remote embedding request failed"));
        }
        let parsed: RemoteEmbeddingResponse = response.json()?;
        if parsed.embedding.is_empty() {
            return Err(anyhow!("remote embedding vector is empty"));
        }
        Ok(parsed.embedding)
    }
}

fn hash_embed(text: &str) -> Vec<f32> {
    if text.is_empty() {
        return vec![0.0; EMBEDDING_DIM];
    }
    let mut values = vec![0.0f32; EMBEDDING_DIM];
    let mut count = 0usize;
    for token in text.split_whitespace() {
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let index = hash % EMBEDDING_DIM;
        values[index] += 1.0;
        count += 1;
        if count >= 1024 {
            break;
        }
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return values;
    }
    for value in &mut values {
        *value /= norm;
    }
    values
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    let len = a.len();
    let mut i = 0usize;
    while i < len {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
        i += 1;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_has_expected_length() {
        let embedder = Embedder::from_env().unwrap();
        let v = embedder.embed("fn main() {}").unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    #[test]
    fn cosine_similarity_of_identical_is_positive() {
        let embedder = Embedder::from_env().unwrap();
        let v = embedder.embed("test content").unwrap();
        let s = cosine_similarity(&v, &v);
        assert!(s > 0.0);
    }
}
