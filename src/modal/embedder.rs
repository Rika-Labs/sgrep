//! Modal embedder that calls the Modal.dev embedding endpoint.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::embedding::BatchEmbedder;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_BATCH_SIZE: usize = 32;
const MAX_RETRIES: usize = 3;

#[derive(Serialize)]
struct EmbedRequest {
    texts: Vec<String>,
    dimension: usize,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    dimension: usize,
}

pub struct ModalEmbedder {
    client: ureq::Agent,
    endpoint: String,
    token_id: Option<String>,
    token_secret: Option<String>,
    dimension: usize,
    batch_size: usize,
}

impl ModalEmbedder {
    pub fn new(
        endpoint: String,
        dimension: usize,
        token_id: Option<String>,
        token_secret: Option<String>,
    ) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build();

        Self {
            client,
            endpoint,
            token_id,
            token_secret,
            dimension,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn embed_chunk(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let request = EmbedRequest {
            texts: texts.to_vec(),
            dimension: self.dimension,
        };

        let mut req = self
            .client
            .post(&self.endpoint)
            .set("Content-Type", "application/json");

        // Add Modal proxy auth headers if credentials are available
        if let (Some(token_id), Some(token_secret)) = (&self.token_id, &self.token_secret) {
            req = req
                .set("Modal-Key", token_id)
                .set("Modal-Secret", token_secret);
        }

        let response = req.send_json(&request).map_err(|e| {
            if let ureq::Error::Status(status, _) = &e {
                match *status {
                    401 => anyhow!("Modal authentication failed. Check your token_id and token_secret."),
                    429 => anyhow!("Rate limited by Modal. Please wait and retry."),
                    500..=599 => anyhow!("Modal server error ({}). Please retry.", status),
                    _ => anyhow!("Modal request failed with status {}: {}", status, e),
                }
            } else {
                anyhow!("Failed to send request to Modal: {}", e)
            }
        })?;

        let embed_response: EmbedResponse = response
            .into_json()
            .context("Failed to parse Modal response")?;

        Ok(embed_response.embeddings)
    }
}

impl BatchEmbedder for ModalEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(self.batch_size) {
            let mut last_err = None;

            for attempt in 1..=MAX_RETRIES {
                match self.embed_chunk(chunk) {
                    Ok(embeddings) => {
                        all_embeddings.extend(embeddings);
                        last_err = None;
                        break;
                    }
                    Err(e) => {
                        if attempt < MAX_RETRIES {
                            eprintln!(
                                "Modal embed retry {}/{}: {}",
                                attempt, MAX_RETRIES, e
                            );
                            std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                            last_err = Some(e);
                        } else {
                            return Err(e);
                        }
                    }
                }
            }

            if let Some(e) = last_err {
                return Err(e);
            }
        }

        Ok(all_embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_embedder_has_correct_fields() {
        let embedder = ModalEmbedder::new(
            "https://embed.modal.run".to_string(),
            4096,
            Some("token-id".to_string()),
            Some("token-secret".to_string()),
        );
        assert_eq!(embedder.endpoint, "https://embed.modal.run");
        assert_eq!(embedder.dimension, 4096);
        assert_eq!(embedder.token_id, Some("token-id".to_string()));
        assert_eq!(embedder.token_secret, Some("token-secret".to_string()));
        assert_eq!(embedder.batch_size, DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn with_batch_size_sets_batch_size() {
        let embedder =
            ModalEmbedder::new("https://embed.modal.run".to_string(), 4096, None, None)
                .with_batch_size(64);
        assert_eq!(embedder.batch_size, 64);
    }

    #[test]
    fn dimension_returns_correct_value() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 1024, None, None);
        assert_eq!(embedder.dimension(), 1024);
    }

    #[test]
    fn embed_batch_returns_empty_for_empty_input() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 4096, None, None);
        let result = embedder.embed_batch(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn embed_request_serialization() {
        let request = EmbedRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
            dimension: 4096,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("4096"));
    }

    #[test]
    fn embed_response_deserialization() {
        let json = r#"{
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model": "Qwen/Qwen3-Embedding-8B",
            "dimension": 3
        }"#;
        let response: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0], vec![0.1, 0.2, 0.3]);
    }
}
