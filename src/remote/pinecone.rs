use anyhow::{anyhow, Result};
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::{RemoteChunk, RemoteSearchHit, RemoteVectorStore};

const DEFAULT_TOP_K: usize = 50;

#[derive(Clone)]
pub struct PineconeStore {
    api_key: String,
    endpoint: String,
    namespace: String,
    client: ureq::Agent,
}

impl PineconeStore {
    pub fn new(api_key: String, endpoint: String, namespace: String, timeout_secs: u64) -> Self {
        let client = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build();
        Self {
            api_key,
            endpoint,
            namespace,
            client,
        }
    }

    fn vectors_url(&self, suffix: &str) -> String {
        format!(
            "{}/{}",
            self.endpoint.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        )
    }

    fn should_ignore_delete_status(status: u16) -> bool {
        status == 404
    }
}

const MAX_VECTORS_PER_REQUEST: usize = 100;

impl RemoteVectorStore for PineconeStore {
    fn name(&self) -> &'static str {
        "pinecone"
    }

    fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        #[derive(Serialize)]
        struct Metadata<'a> {
            path: &'a str,
            start_line: usize,
            end_line: usize,
            content: &'a str,
            language: &'a str,
        }

        #[derive(Serialize)]
        struct Vector<'a> {
            id: &'a str,
            values: &'a [f32],
            metadata: Metadata<'a>,
        }

        #[derive(Serialize)]
        struct UpsertRequest<'a> {
            vectors: Vec<Vector<'a>>,
            namespace: &'a str,
        }

        let url = self.vectors_url("vectors/upsert");

        for batch in chunks.chunks(MAX_VECTORS_PER_REQUEST) {
            let vectors: Vec<Vector> = batch
                .iter()
                .map(|c| Vector {
                    id: &c.id,
                    values: &c.vector,
                    metadata: Metadata {
                        path: &c.path,
                        start_line: c.start_line,
                        end_line: c.end_line,
                        content: &c.content,
                        language: &c.language,
                    },
                })
                .collect();

            let req = UpsertRequest {
                vectors,
                namespace: &self.namespace,
            };

            let res = self
                .client
                .post(&url)
                .set("Api-Key", &self.api_key)
                .set("Content-Type", "application/json")
                .send_json(&req);

            match res {
                Ok(_) => {}
                Err(ureq::Error::Status(status, response)) => {
                    let body = response.into_string().unwrap_or_default();
                    if body.is_empty() {
                        return Err(anyhow!("Pinecone upsert failed with status {}", status));
                    } else {
                        return Err(anyhow!(
                            "Pinecone upsert failed with status {}: {}",
                            status,
                            body
                        ));
                    }
                }
                Err(e) => return Err(anyhow!("Pinecone upsert failed: {}", e)),
            }
        }

        Ok(())
    }

    fn upsert_with_progress(&self, chunks: &[RemoteChunk], pb: &ProgressBar) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        #[derive(Serialize)]
        struct Metadata<'a> {
            path: &'a str,
            start_line: usize,
            end_line: usize,
            content: &'a str,
            language: &'a str,
        }

        #[derive(Serialize)]
        struct Vector<'a> {
            id: &'a str,
            values: &'a [f32],
            metadata: Metadata<'a>,
        }

        #[derive(Serialize)]
        struct UpsertRequest<'a> {
            vectors: Vec<Vector<'a>>,
            namespace: &'a str,
        }

        let url = self.vectors_url("vectors/upsert");
        let uploaded = AtomicUsize::new(0);
        let store = self.clone();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();

        batches.par_iter().try_for_each(|batch| {
            let vectors: Vec<Vector> = batch
                .iter()
                .map(|c| Vector {
                    id: &c.id,
                    values: &c.vector,
                    metadata: Metadata {
                        path: &c.path,
                        start_line: c.start_line,
                        end_line: c.end_line,
                        content: &c.content,
                        language: &c.language,
                    },
                })
                .collect();

            let req = UpsertRequest {
                vectors,
                namespace: &store.namespace,
            };

            let res = store
                .client
                .post(&url)
                .set("Api-Key", &store.api_key)
                .set("Content-Type", "application/json")
                .send_json(&req);

            match res {
                Ok(_) => {
                    let uploaded_count =
                        uploaded.fetch_add(batch.len(), Ordering::SeqCst) + batch.len();
                    pb.set_position(uploaded_count as u64);
                    Ok(())
                }
                Err(ureq::Error::Status(status, response)) => {
                    let body = response.into_string().unwrap_or_default();
                    if body.is_empty() {
                        Err(anyhow!("Pinecone upsert failed with status {}", status))
                    } else {
                        Err(anyhow!(
                            "Pinecone upsert failed with status {}: {}",
                            status,
                            body
                        ))
                    }
                }
                Err(e) => Err(anyhow!("Pinecone upsert failed: {}", e)),
            }
        })?;

        Ok(())
    }

    fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<RemoteSearchHit>> {
        if vector.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct QueryRequest<'a> {
            vector: &'a [f32],
            top_k: usize,
            include_metadata: bool,
            namespace: &'a str,
        }

        #[derive(Deserialize)]
        struct Match {
            id: String,
            score: f32,
            metadata: Option<Metadata>,
        }

        #[derive(Deserialize)]
        struct Metadata {
            path: Option<String>,
            start_line: Option<usize>,
            end_line: Option<usize>,
            content: Option<String>,
            language: Option<String>,
        }

        #[derive(Deserialize)]
        struct QueryResponse {
            #[serde(default)]
            matches: Vec<Match>,
        }

        let url = self.vectors_url("query");
        let req = QueryRequest {
            vector,
            top_k: if top_k == 0 { DEFAULT_TOP_K } else { top_k },
            include_metadata: true,
            namespace: &self.namespace,
        };

        let res = self
            .client
            .post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&req);

        let resp: QueryResponse = match res {
            Ok(r) => r
                .into_json()
                .map_err(|e| anyhow!("Pinecone parse error: {}", e))?,
            Err(e) => return Err(anyhow!("Pinecone query failed: {}", e)),
        };

        let hits = resp
            .matches
            .into_iter()
            .filter_map(|m| {
                let meta = m.metadata?;
                Some(RemoteSearchHit {
                    id: m.id,
                    score: m.score,
                    path: meta.path.unwrap_or_default(),
                    start_line: meta.start_line.unwrap_or(0),
                    end_line: meta.end_line.unwrap_or(0),
                    content: meta.content.unwrap_or_default(),
                    language: meta.language.unwrap_or_else(|| "plain".to_string()),
                })
            })
            .collect();

        Ok(hits)
    }

    fn delete_namespace(&self) -> Result<()> {
        let url = self.vectors_url("vectors/delete");
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct DeleteRequest<'a> {
            delete_all: bool,
            namespace: &'a str,
        }

        let req = DeleteRequest {
            delete_all: true,
            namespace: &self.namespace,
        };

        let res = self
            .client
            .post(&url)
            .set("Api-Key", &self.api_key)
            .set("Content-Type", "application/json")
            .send_json(&req);

        match res {
            Ok(_) => Ok(()),
            Err(ureq::Error::Status(status, response))
                if Self::should_ignore_delete_status(status) =>
            {
                let _ = response.into_string();
                Ok(())
            }
            Err(ureq::Error::Status(status, response)) => {
                let body = response.into_string().unwrap_or_default();
                if body.is_empty() {
                    Err(anyhow!("Pinecone delete failed with status {}", status))
                } else {
                    Err(anyhow!(
                        "Pinecone delete failed with status {}: {}",
                        status,
                        body
                    ))
                }
            }
            Err(e) => Err(anyhow!("Pinecone delete failed: {}", e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PineconeStore, MAX_VECTORS_PER_REQUEST};
    use crate::remote::RemoteChunk;

    #[test]
    fn delete_not_found_status_is_ignored() {
        assert!(PineconeStore::should_ignore_delete_status(404));
        assert!(!PineconeStore::should_ignore_delete_status(500));
    }

    #[test]
    fn max_vectors_per_request_is_set() {
        assert_eq!(MAX_VECTORS_PER_REQUEST, 100);
    }

    fn create_test_chunk(id: usize) -> RemoteChunk {
        RemoteChunk {
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
        let chunks: Vec<RemoteChunk> = (0..250).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 100);
        assert_eq!(batches[1].len(), 100);
        assert_eq!(batches[2].len(), 50);
    }

    #[test]
    fn batching_handles_exact_multiple() {
        let chunks: Vec<RemoteChunk> = (0..200).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 100);
        assert_eq!(batches[1].len(), 100);
    }

    #[test]
    fn batching_handles_small_batch() {
        let chunks: Vec<RemoteChunk> = (0..50).map(|i| create_test_chunk(i)).collect();

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 50);
    }

    #[test]
    fn batching_handles_single_chunk() {
        let chunks: Vec<RemoteChunk> = vec![create_test_chunk(0)];

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn batching_handles_empty_chunks() {
        let chunks: Vec<RemoteChunk> = vec![];

        let batches: Vec<_> = chunks.chunks(MAX_VECTORS_PER_REQUEST).collect();
        assert_eq!(batches.len(), 0);
    }
}
