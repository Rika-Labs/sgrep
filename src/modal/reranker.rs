//! Modal reranker that calls the Modal.dev reranking endpoint.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::reranker::Reranker;

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_RETRIES: usize = 3;

#[derive(Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    top_k: usize,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    score: f32,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
    #[allow(dead_code)]
    model: String,
}

pub struct ModalReranker {
    client: ureq::Agent,
    endpoint: String,
    token_id: Option<String>,
    token_secret: Option<String>,
}

impl ModalReranker {
    pub fn new(
        endpoint: String,
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
        }
    }

    pub fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let request = RerankRequest {
            query: query.to_string(),
            documents: documents.to_vec(),
            top_k,
        };

        let mut last_err = None;

        for attempt in 1..=MAX_RETRIES {
            match self.do_rerank(&request) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        eprintln!(
                            "Modal rerank retry {}/{}: {}",
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

        Err(last_err.unwrap_or_else(|| anyhow!("Rerank failed")))
    }

    fn do_rerank(&self, request: &RerankRequest) -> Result<Vec<(usize, f32)>> {
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

        let response = req.send_json(request).map_err(|e| {
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

        let rerank_response: RerankResponse = response
            .into_json()
            .context("Failed to parse Modal rerank response")?;

        Ok(rerank_response
            .results
            .into_iter()
            .map(|r| (r.index, r.score))
            .collect())
    }
}

impl Reranker for ModalReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        let docs: Vec<String> = documents.iter().map(|s| s.to_string()).collect();
        ModalReranker::rerank(self, query, &docs, documents.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_reranker_has_correct_fields() {
        let reranker = ModalReranker::new(
            "https://rerank.modal.run".to_string(),
            Some("token-id".to_string()),
            Some("token-secret".to_string()),
        );
        assert_eq!(reranker.endpoint, "https://rerank.modal.run");
        assert_eq!(reranker.token_id, Some("token-id".to_string()));
        assert_eq!(reranker.token_secret, Some("token-secret".to_string()));
    }

    #[test]
    fn rerank_returns_empty_for_empty_documents() {
        let reranker = ModalReranker::new("https://rerank.modal.run".to_string(), None, None);
        let result = reranker.rerank("query", &[], 10);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn rerank_request_serialization() {
        let request = RerankRequest {
            query: "search query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            top_k: 5,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("search query"));
        assert!(json.contains("doc1"));
        assert!(json.contains("top_k"));
    }

    #[test]
    fn rerank_response_deserialization() {
        let json = r#"{
            "results": [
                {"index": 1, "score": 0.95},
                {"index": 0, "score": 0.82}
            ],
            "model": "Qwen/Qwen3-Reranker-8B"
        }"#;
        let response: RerankResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].index, 1);
        assert!((response.results[0].score - 0.95).abs() < f32::EPSILON);
    }
}
