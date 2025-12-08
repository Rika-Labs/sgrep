//! Turbopuffer vector store for remote index storage.

use anyhow::{anyhow, Context, Result};
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::remote::{RemoteChunk, RemoteSearchHit, RemoteVectorStore};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_RETRIES: usize = 3;
const MAX_VECTORS_PER_REQUEST: usize = 1000;
const TP_ATTR_LIMIT_BYTES: usize = 3800; // under 4KB filterable attribute limit

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
struct UpsertRow {
    id: String,
    vector: Vec<f32>,
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
    language: String,
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
struct WriteRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    upsert_rows: Option<Vec<UpsertRow>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    distance_metric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deletes: Option<Vec<String>>,
}

#[derive(Clone)]
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

    fn truncate_attr(value: &str) -> String {
        let bytes = value.as_bytes();
        if bytes.len() <= TP_ATTR_LIMIT_BYTES {
            return value.to_owned();
        }

        let max_bytes = TP_ATTR_LIMIT_BYTES.saturating_sub(3); // leave room for ellipsis bytes
        let mut end = 0;
        for (idx, _) in value.char_indices() {
            if idx <= max_bytes {
                end = idx;
            } else {
                break;
            }
        }
        let mut truncated = value[..end].to_owned();
        truncated.push('…');
        truncated
    }

    fn dedup_rows(rows: Vec<UpsertRow>) -> Vec<UpsertRow> {
        let mut seen = HashSet::with_capacity(rows.len());
        let mut unique = Vec::with_capacity(rows.len());
        for row in rows.into_iter() {
            if seen.insert(row.id.clone()) {
                unique.push(row);
            }
        }
        unique
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
                    let err = match e {
                        ureq::Error::Status(status, response) => {
                            let body = response.into_string().unwrap_or_default();
                            match status {
                                401 => {
                                    anyhow!("Authentication failed: invalid Turbopuffer API key")
                                }
                                429 => {
                                    anyhow!("Rate limited by Turbopuffer. Please wait and retry.")
                                }
                                500..=599 => {
                                    if body.is_empty() {
                                        anyhow!(
                                            "Turbopuffer server error ({}). Please retry.",
                                            status
                                        )
                                    } else {
                                        anyhow!("Turbopuffer server error ({}): {}", status, body)
                                    }
                                }
                                _ => {
                                    if body.is_empty() {
                                        anyhow!("Turbopuffer request failed with status {}", status)
                                    } else {
                                        anyhow!(
                                            "Turbopuffer request failed with status {}: {}",
                                            status,
                                            body
                                        )
                                    }
                                }
                            }
                        }
                        e => anyhow!("Failed to send request to Turbopuffer: {}", e),
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

        for batch in chunks.chunks(MAX_VECTORS_PER_REQUEST) {
            let rows: Vec<UpsertRow> = Self::dedup_rows(
                batch
                    .iter()
                    .map(|c| UpsertRow {
                        id: c.id.clone(),
                        vector: c.vector.clone(),
                        path: c.path.clone(),
                        start_line: c.start_line,
                        end_line: c.end_line,
                        content: Self::truncate_attr(&c.content),
                        language: c.language.clone(),
                    })
                    .collect(),
            );

            let request = WriteRequest {
                upsert_rows: Some(rows),
                distance_metric: Some("cosine_distance".to_string()),
                deletes: None,
            };

            self.post_with_retry::<_, serde_json::Value>(&url, &request)?;
        }

        Ok(())
    }

    fn upsert_with_progress(&self, chunks: &[RemoteChunk], pb: &ProgressBar) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let url = self.base_url();
        let uploaded = AtomicUsize::new(0);
        let store = self.clone();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();

        batches.par_iter().try_for_each(|batch| {
            let rows: Vec<UpsertRow> = Self::dedup_rows(
                batch
                    .iter()
                    .map(|c| UpsertRow {
                        id: c.id.clone(),
                        vector: c.vector.clone(),
                        path: c.path.clone(),
                        start_line: c.start_line,
                        end_line: c.end_line,
                        content: Self::truncate_attr(&c.content),
                        language: c.language.clone(),
                    })
                    .collect(),
            );

            let request = WriteRequest {
                upsert_rows: Some(rows),
                distance_metric: Some("cosine_distance".to_string()),
                deletes: None,
            };

            store.post_with_retry::<_, serde_json::Value>(&url, &request)?;
            let uploaded_count = uploaded.fetch_add(batch.len(), Ordering::SeqCst) + batch.len();
            pb.set_position(uploaded_count as u64);
            Ok::<(), anyhow::Error>(())
        })?;

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

    #[test]
    fn truncate_attr_respects_limit() {
        let long = "a".repeat(TP_ATTR_LIMIT_BYTES + 200);
        let truncated = TurbopufferStore::truncate_attr(&long);
        assert!(truncated.len() <= TP_ATTR_LIMIT_BYTES);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn dedup_rows_removes_duplicates() {
        let rows = vec![
            UpsertRow {
                id: "a".into(),
                vector: vec![0.1],
                path: "p".into(),
                start_line: 1,
                end_line: 1,
                content: "c".into(),
                language: "rust".into(),
            },
            UpsertRow {
                id: "a".into(),
                vector: vec![0.2],
                path: "p2".into(),
                start_line: 1,
                end_line: 2,
                content: "c2".into(),
                language: "rust".into(),
            },
        ];
        let deduped = TurbopufferStore::dedup_rows(rows);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].id, "a");
    }

    #[test]
    fn max_vectors_per_request_is_set() {
        use super::MAX_VECTORS_PER_REQUEST;
        assert_eq!(MAX_VECTORS_PER_REQUEST, 1000);
    }

    fn create_test_chunk(id: usize) -> crate::remote::RemoteChunk {
        crate::remote::RemoteChunk {
            id: format!("chunk-{}", id),
            vector: vec![0.1, 0.2, 0.3],
            path: format!("file{}.rs", id),
            start_line: 1,
            end_line: 10,
            content: format!("content {}", id),
            language: "rust".to_string(),
        }
    }

    #[test]
    fn batching_splits_large_requests() {
        use super::MAX_VECTORS_PER_REQUEST;
        let chunks: Vec<crate::remote::RemoteChunk> =
            (0..2500).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 1000);
        assert_eq!(batches[1].len(), 1000);
        assert_eq!(batches[2].len(), 500);
    }

    #[test]
    fn batching_handles_exact_multiple() {
        use super::MAX_VECTORS_PER_REQUEST;
        let chunks: Vec<crate::remote::RemoteChunk> =
            (0..2000).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 1000);
        assert_eq!(batches[1].len(), 1000);
    }

    #[test]
    fn batching_handles_small_batch() {
        use super::MAX_VECTORS_PER_REQUEST;
        let chunks: Vec<crate::remote::RemoteChunk> =
            (0..500).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 500);
    }

    #[test]
    fn batching_handles_single_chunk() {
        use super::MAX_VECTORS_PER_REQUEST;
        let chunks: Vec<crate::remote::RemoteChunk> = vec![create_test_chunk(0)];

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn batching_handles_empty_chunks() {
        use super::MAX_VECTORS_PER_REQUEST;
        let chunks: Vec<crate::remote::RemoteChunk> = vec![];

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 0);
    }
}
