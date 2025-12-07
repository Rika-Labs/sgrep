use anyhow::{anyhow, Context, Result};
use flate2::{write::GzEncoder, Compression};
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use crate::embedding::BatchEmbedder;

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_BATCH_SIZE: usize = 128;
const MAX_RETRIES: usize = 3;
const DEFAULT_MAX_IDLE_CONNECTIONS: usize = 128;
const DEFAULT_MAX_IDLE_CONNECTIONS_PER_HOST: usize = 64;
const MAX_TEXTS_PER_REQUEST: usize = 1000;
const REQUIRED_DIMENSION: usize = 384;
const MIN_CONCURRENCY: usize = 1;
const MAX_CONCURRENCY: usize = 64;

type BatchResult = (usize, Vec<Vec<f32>>);
type SharedBatchResults = Arc<Mutex<Vec<BatchResult>>>;

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
    proxy_token_id: Option<String>,
    proxy_token_secret: Option<String>,
    dimension: usize,
    batch_size: usize,
    concurrency: usize,
    use_gzip: bool,
    pool: Arc<ThreadPool>,

    #[cfg(test)]
    mock_responder: Option<Arc<dyn Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync>>,
}

impl ModalEmbedder {
    pub fn new(
        endpoint: String,
        dimension: usize,
        proxy_token_id: Option<String>,
        proxy_token_secret: Option<String>,
    ) -> Self {
        let client = build_agent();

        let concurrency = resolve_default_concurrency();
        let pool = build_pool(concurrency);

        let use_gzip = std::env::var("SGREP_MODAL_GZIP")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        Self {
            client,
            endpoint,
            proxy_token_id,
            proxy_token_secret,
            dimension,
            batch_size: DEFAULT_BATCH_SIZE,
            concurrency,
            use_gzip,
            pool,

            #[cfg(test)]
            mock_responder: None,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        let concurrency = concurrency.clamp(MIN_CONCURRENCY, MAX_CONCURRENCY);
        self.pool = build_pool(concurrency);
        self.concurrency = concurrency;
        self
    }

    pub fn with_gzip(mut self, use_gzip: bool) -> Self {
        self.use_gzip = use_gzip;
        self
    }

    #[cfg(test)]
    pub fn with_mock_responder<F>(mut self, responder: F) -> Self
    where
        F: Fn(&[String]) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static,
    {
        self.mock_responder = Some(Arc::new(responder));
        self
    }

    fn embed_chunk(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(test)]
        if let Some(responder) = &self.mock_responder {
            return responder(texts);
        }

        let request = EmbedRequest {
            texts: texts.to_vec(),
            dimension: self.dimension,
        };

        let mut req = self
            .client
            .post(&self.endpoint)
            .set("Content-Type", "application/json");

        if let (Some(proxy_id), Some(proxy_secret)) =
            (&self.proxy_token_id, &self.proxy_token_secret)
        {
            req = req
                .set("Modal-Key", proxy_id)
                .set("Modal-Secret", proxy_secret);
        }

        let response = {
            if self.use_gzip {
                let body = serde_json::to_vec(&request).context("Serialize embed request")?;
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(&body)
                    .context("Compress embed request")?;
                let compressed = encoder.finish().context("Finish compression")?;
                req = req.set("Content-Encoding", "gzip");
                req.send_bytes(&compressed)
            } else {
                req.send_json(&request)
            }
        }
        .map_err(|e| {
            if let ureq::Error::Status(status, _) = &e {
                match *status {
                    401 => anyhow!("Modal authentication failed. Check your proxy_token_id and proxy_token_secret."),
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

    fn embed_chunk_with_retry(&self, chunk: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut last_err = None;

        for attempt in 1..=MAX_RETRIES {
            match self.embed_chunk(chunk) {
                Ok(embeddings) => return Ok(embeddings),
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        eprintln!("Modal embed retry {}/{}: {}", attempt, MAX_RETRIES, e);
                        std::thread::sleep(Duration::from_millis(500 * attempt as u64));
                    }
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("embedding failed")))
    }
}

impl BatchEmbedder for ModalEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with_progress(texts, None)
    }

    fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&crate::embedding::ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if self.dimension != REQUIRED_DIMENSION {
            return Err(anyhow!(
                "Dimension must be {} for local/remote compatibility, got {}",
                REQUIRED_DIMENSION,
                self.dimension
            ));
        }

        let total = texts.len();
        let mut all_embeddings = Vec::with_capacity(total);
        let chunk_size = self.batch_size.clamp(1, MAX_TEXTS_PER_REQUEST);

        struct Batch {
            start: usize,
            texts: Vec<String>,
        }

        let mut batches = Vec::new();
        let mut start_idx = 0usize;
        for chunk in texts.chunks(chunk_size) {
            batches.push(Batch {
                start: start_idx,
                texts: chunk.to_vec(),
            });
            start_idx += chunk.len();
        }

        let total_batches = batches.len();
        let results: SharedBatchResults = Arc::new(Mutex::new(Vec::with_capacity(total_batches)));
        let completed = AtomicUsize::new(0);
        let batches_done = AtomicUsize::new(0);

        self.pool.install(|| -> Result<()> {
            batches.par_iter().try_for_each(|batch| {
                let embeddings = self.embed_chunk_with_retry(&batch.texts)?;

                {
                    let mut guard = results.lock().unwrap();
                    guard.push((batch.start, embeddings));
                }

                let done =
                    completed.fetch_add(batch.texts.len(), Ordering::SeqCst) + batch.texts.len();
                let batch_done = batches_done.fetch_add(1, Ordering::SeqCst) + 1;

                if let Some(callback) = on_progress {
                    callback(crate::embedding::EmbedProgress {
                        completed: done,
                        total,
                        message: Some(format!("received batch {}/{}", batch_done, total_batches)),
                    });
                }

                Ok(())
            })
        })?;

        let mut ordered = results.lock().unwrap().drain(..).collect::<Vec<_>>();
        ordered.sort_by_key(|(start, _)| *start);

        for (_start, mut vecs) in ordered.into_iter() {
            all_embeddings.append(&mut vecs);
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
    use std::sync::atomic::AtomicUsize;

    fn make_embedder_for_tests() -> ModalEmbedder {
        ModalEmbedder::new(
            "https://embed.modal.run".to_string(),
            384,
            Some("wk-proxy-id".to_string()),
            Some("ws-proxy-secret".to_string()),
        )
    }

    #[test]
    fn new_embedder_has_correct_fields() {
        let embedder = make_embedder_for_tests();
        assert_eq!(embedder.endpoint, "https://embed.modal.run");
        assert_eq!(embedder.dimension, 384);
        assert_eq!(embedder.proxy_token_id, Some("wk-proxy-id".to_string()));
        assert_eq!(
            embedder.proxy_token_secret,
            Some("ws-proxy-secret".to_string())
        );
        assert_eq!(embedder.batch_size, DEFAULT_BATCH_SIZE);
        assert!(embedder.concurrency >= MIN_CONCURRENCY);
    }

    #[test]
    fn with_batch_size_sets_batch_size() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None)
            .with_batch_size(64);
        assert_eq!(embedder.batch_size, 64);
    }

    #[test]
    fn with_concurrency_sets_pool_size() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None)
            .with_concurrency(4);
        assert_eq!(embedder.concurrency, 4);
    }

    #[test]
    fn dimension_returns_correct_value() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None);
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn embed_batch_returns_empty_for_empty_input() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None);
        let result = embedder.embed_batch(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn embed_request_serialization() {
        let request = EmbedRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
            dimension: 384,
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("384"));
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

    #[test]
    fn embed_batch_rejects_wrong_dimension() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 4096, None, None);
        let result = embedder.embed_batch(&["test".to_string()]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Dimension must be 384"));
        assert!(err.contains("got 4096"));
    }

    #[test]
    fn embed_batch_rejects_dimension_zero() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 0, None, None);
        let result = embedder.embed_batch(&["test".to_string()]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Dimension must be 384"));
    }

    #[test]
    fn embed_batch_runs_concurrently_and_preserves_order() {
        let call_counter = Arc::new(AtomicUsize::new(0));
        let responder_counter = call_counter.clone();

        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None)
            .with_batch_size(2)
            .with_concurrency(4)
            .with_mock_responder(move |texts| {
                responder_counter.fetch_add(1, Ordering::SeqCst);
                Ok(texts
                    .iter()
                    .map(|t| vec![t.len() as f32, 1.0, 2.0])
                    .collect())
            });

        let inputs = vec![
            "a".to_string(),
            "bb".to_string(),
            "ccc".to_string(),
            "dddd".to_string(),
            "eeeee".to_string(),
        ];

        let result = embedder.embed_batch(&inputs).unwrap();

        assert_eq!(call_counter.load(Ordering::SeqCst), 3);
        assert_eq!(result.len(), inputs.len());
        for (out, input) in result.iter().zip(inputs.iter()) {
            assert_eq!(out[0], input.len() as f32);
        }
    }

    #[test]
    fn embed_batch_allows_more_than_service_limit_by_chunking() {
        let embedder = ModalEmbedder::new("https://embed.modal.run".to_string(), 384, None, None)
            .with_batch_size(500)
            .with_concurrency(4)
            .with_mock_responder(|texts| {
                Ok(texts
                    .iter()
                    .map(|t| vec![t.len() as f32, 0.0, 0.0])
                    .collect())
            });

        let inputs: Vec<String> = (0..1100).map(|i| format!("t{i}")).collect();
        let result = embedder.embed_batch(&inputs).unwrap();
        assert_eq!(result.len(), inputs.len());
    }
}

fn build_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .max_idle_connections(DEFAULT_MAX_IDLE_CONNECTIONS)
        .max_idle_connections_per_host(DEFAULT_MAX_IDLE_CONNECTIONS_PER_HOST)
        .build()
}

fn build_pool(concurrency: usize) -> Arc<ThreadPool> {
    let threads = concurrency.clamp(MIN_CONCURRENCY, MAX_CONCURRENCY);
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build Modal embedder threadpool"),
    )
}

fn resolve_default_concurrency() -> usize {
    let default = std::cmp::max(2, num_cpus::get().saturating_div(2)).min(8);
    let env_val = std::env::var("SGREP_MODAL_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(MIN_CONCURRENCY, MAX_CONCURRENCY));
    env_val.unwrap_or(default)
}
