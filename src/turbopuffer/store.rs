//! Turbopuffer vector store for remote index storage.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::remote::{RemoteChunk, RemoteSearchHit, RemoteVectorStore};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_RETRIES: usize = 3;

#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub id: String,
    #[serde(rename = "$dist")]
    pub distance: f32,
    pub path: Option<String>,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub content: Option<String>,
    pub language: Option<String>,
}

#[derive(Serialize)]
struct UpsertRow<'a> {
    id: &'a str,
    vector: &'a [f32],
    path: &'a str,
    start_line: usize,
    end_line: usize,
    content: &'a str,
    language: &'a str,
}

#[derive(Serialize)]
struct QueryRequest<'a> {
    rank_by: (String, String, &'a [f32]),
    top_k: usize,
    include_attributes: Vec<String>,
}

#[derive(Deserialize)]
struct QueryResponse {
    #[serde(default)]
    rows: Vec<SearchResult>,
}

#[derive(Serialize)]
struct WriteRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    upsert_rows: Option<Vec<UpsertRow<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    distance_metric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deletes: Option<Vec<String>>,
}

pub struct TurbopufferStore {
    client: ureq::Agent,
    api_key: String,
    namespace: String,
    region: String,
}

impl TurbopufferStore {
    pub fn new(api_key: String, namespace: String, region: String, timeout_secs: u64) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(if timeout_secs == 0 {
                DEFAULT_TIMEOUT_SECS
            } else {
                timeout_secs
            }))
            .build();

        Self {
            client,
            api_key,
            namespace,
            region,
        }
    }

    fn should_ignore_delete_status(status: u16) -> bool {
        status == 404
    }

    fn base_url(&self) -> String {
        format!(
            "https://api.turbopuffer.com/v2/namespaces/{}",
            self.namespace
        )
    }

    fn post_with_retry<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<R> {
        let mut last_err = None;

        for attempt in 1..=MAX_RETRIES {
            match self
                .client
                .post(url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .send_json(body)
            {
                Ok(response) => {
                    return response
                        .into_json()
                        .context("Failed to parse Turbopuffer response");
                }
                Err(e) => {
                    let err = if let ureq::Error::Status(status, _) = &e {
                        match *status {
                            401 => anyhow!("Authentication failed: invalid Turbopuffer API key"),
                            429 => anyhow!("Rate limited by Turbopuffer. Please wait and retry."),
                            500..=599 => {
                                anyhow!("Turbopuffer server error ({}). Please retry.", status)
                            }
                            _ => {
                                anyhow!("Turbopuffer request failed with status {}: {}", status, e)
                            }
                        }
                    } else {
                        anyhow!("Failed to send request to Turbopuffer: {}", e)
                    };

                    // Don't retry auth failures
                    if err.to_string().contains("Authentication failed") {
                        return Err(err);
                    }

                    if attempt < MAX_RETRIES {
                        eprintln!("Turbopuffer retry {}/{}: {}", attempt, MAX_RETRIES, err);
                        std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                        last_err = Some(err);
                    } else {
                        return Err(err);
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("Turbopuffer request failed")))
    }
}

impl RemoteVectorStore for TurbopufferStore {
    fn name(&self) -> &'static str {
        "turbopuffer"
    }

    fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let url = self.base_url();
        let rows: Vec<UpsertRow> = chunks
            .iter()
            .map(|c| UpsertRow {
                id: &c.id,
                vector: &c.vector,
                path: &c.path,
                start_line: c.start_line,
                end_line: c.end_line,
                content: &c.content,
                language: &c.language,
            })
            .collect();

        let request = WriteRequest {
            upsert_rows: Some(rows),
            distance_metric: Some("cosine_distance".to_string()),
            deletes: None,
        };

        self.post_with_retry::<_, serde_json::Value>(&url, &request)?;
        Ok(())
    }

    fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<RemoteSearchHit>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/query", self.base_url());
        let request = QueryRequest {
            rank_by: ("vector".to_string(), "ANN".to_string(), vector),
            top_k,
            include_attributes: vec![
                "path".to_string(),
                "start_line".to_string(),
                "end_line".to_string(),
                "content".to_string(),
                "language".to_string(),
            ],
        };

        let response: QueryResponse = self.post_with_retry(&url, &request)?;
        let hits = response
            .rows
            .into_iter()
            .map(|r| RemoteSearchHit {
                id: r.id,
                score: 1.0 - r.distance,
                path: r.path.unwrap_or_default(),
                start_line: r.start_line.unwrap_or(0),
                end_line: r.end_line.unwrap_or(0),
                content: r.content.unwrap_or_default(),
                language: r.language.unwrap_or_else(|| "plain".to_string()),
            })
            .collect();
        Ok(hits)
    }

    fn delete_namespace(&self) -> Result<()> {
        let url = self.base_url();

        for attempt in 1..=MAX_RETRIES {
            match self
                .client
                .delete(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .call()
            {
                Ok(_) => return Ok(()),
                Err(ureq::Error::Status(status, response))
                    if Self::should_ignore_delete_status(status) =>
                {
                    let _ = response.into_string();
                    return Ok(());
                }
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        eprintln!(
                            "Turbopuffer delete namespace retry {}/{}: {}",
                            attempt, MAX_RETRIES, e
                        );
                        std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                    } else {
                        return Err(anyhow!("Failed to delete namespace: {}", e));
                    }
                }
            }
        }

        Ok(())
    }
}

impl TurbopufferStore {
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn region(&self) -> &str {
        &self.region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_store_has_correct_fields() {
        let store = TurbopufferStore::new(
            "tpuf_test_key".to_string(),
            "sgrep-test".to_string(),
            "gcp-us-central1".to_string(),
            0,
        );
        assert_eq!(store.namespace(), "sgrep-test");
        assert_eq!(store.region(), "gcp-us-central1");
    }

    #[test]
    fn base_url_is_correct() {
        let store = TurbopufferStore::new(
            "tpuf_test_key".to_string(),
            "sgrep-abc123".to_string(),
            "gcp-us-central1".to_string(),
            0,
        );
        assert_eq!(
            store.base_url(),
            "https://api.turbopuffer.com/v2/namespaces/sgrep-abc123"
        );
    }

    #[test]
    fn upsert_empty_returns_ok() {
        let store = TurbopufferStore::new(
            "tpuf_test_key".to_string(),
            "sgrep-test".to_string(),
            "gcp-us-central1".to_string(),
            0,
        );
        let result = store.upsert(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn query_empty_vector_returns_empty() {
        let store = TurbopufferStore::new(
            "tpuf_test_key".to_string(),
            "sgrep-test".to_string(),
            "gcp-us-central1".to_string(),
            0,
        );
        let result = store.query(&[], 10);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn search_result_deserialization() {
        let json = r#"{
            "id": "chunk_123",
            "$dist": 0.15,
            "path": "src/main.rs",
            "start_line": 10,
            "end_line": 20,
            "content": "fn main() {}",
            "language": "rust"
        }"#;
        let result: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.id, "chunk_123");
        assert!((result.distance - 0.15).abs() < f32::EPSILON);
        assert_eq!(result.path, Some("src/main.rs".to_string()));
    }

    #[test]
    fn delete_not_found_status_is_ignored() {
        assert!(TurbopufferStore::should_ignore_delete_status(404));
        assert!(!TurbopufferStore::should_ignore_delete_status(500));
    }
}
