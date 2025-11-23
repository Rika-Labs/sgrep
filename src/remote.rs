use std::path::Path;

use anyhow::{Context, Result, anyhow};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::embedding::Embedder;
use crate::search::{SearchConfig, SearchMatch, SearchResponse};
use crate::store::{load_index, store_name_for_root};

struct RemoteConfig {
    base_url: String,
    api_key: Option<String>,
    collection: String,
}

#[derive(Serialize)]
struct QdrantCreateCollectionVectors {
    size: usize,
    distance: String,
}

#[derive(Serialize)]
struct QdrantCreateCollectionRequest {
    vectors: QdrantCreateCollectionVectors,
}

#[derive(Serialize, Deserialize, Clone)]
struct QdrantPointPayload {
    path: String,
    start_line: u32,
    end_line: u32,
    text: String,
}

#[derive(Serialize, Clone)]
struct QdrantPoint {
    id: u64,
    vector: Vec<f32>,
    payload: QdrantPointPayload,
}

#[derive(Serialize)]
struct QdrantUpsertRequest {
    points: Vec<QdrantPoint>,
}

#[derive(Deserialize)]
struct QdrantSearchResponseBody {
    result: Vec<QdrantSearchResult>,
}

#[derive(Deserialize)]
struct QdrantSearchResult {
    score: f32,
    payload: Option<QdrantPointPayload>,
}

fn config_from_env(root: &Path) -> Result<RemoteConfig> {
    let base_url =
        std::env::var("SGREP_REMOTE_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let api_key = std::env::var("SGREP_REMOTE_API_KEY").ok();
    let collection = match std::env::var("SGREP_REMOTE_COLLECTION") {
        Ok(v) => v,
        Err(_) => store_name_for_root(root)?,
    };
    Ok(RemoteConfig {
        base_url,
        api_key,
        collection,
    })
}

fn client_for(_config: &RemoteConfig) -> Result<Client> {
    let client = Client::builder().build()?;
    Ok(client)
}

fn ensure_collection(client: &Client, config: &RemoteConfig, vector_size: usize) -> Result<()> {
    let url = format!("{}/collections/{}", config.base_url, config.collection);
    let body = QdrantCreateCollectionRequest {
        vectors: QdrantCreateCollectionVectors {
            size: vector_size,
            distance: "Cosine".to_string(),
        },
    };
    let mut request = client.put(url).json(&body);
    if let Some(key) = &config.api_key {
        request = request.header("api-key", key);
    }
    let response = request
        .send()
        .context("failed to send collection create request")?;
    if response.status().is_success() {
        return Ok(());
    }
    if response.status() == StatusCode::CONFLICT {
        return Ok(());
    }
    Err(anyhow!("failed to create or validate remote collection"))
}

pub fn remote_index(root: &Path) -> Result<usize> {
    let config = config_from_env(root)?;
    let client = client_for(&config)?;
    let chunks = load_index(root)?;
    if chunks.is_empty() {
        return Ok(0);
    }
    let vector_size = match chunks.first() {
        Some(chunk) => chunk.embedding.len(),
        None => 0,
    };
    if vector_size == 0 {
        return Ok(0);
    }
    ensure_collection(&client, &config, vector_size)?;
    let mut total = 0usize;
    let mut batch = Vec::new();
    for chunk in chunks.into_iter() {
        let payload = QdrantPointPayload {
            path: chunk.path,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            text: chunk.text,
        };
        let point = QdrantPoint {
            id: chunk.id,
            vector: chunk.embedding,
            payload,
        };
        batch.push(point);
        if batch.len() >= 128 {
            upsert_batch(&client, &config, &batch)?;
            total += batch.len();
            batch.clear();
        }
    }
    if !batch.is_empty() {
        upsert_batch(&client, &config, &batch)?;
        total += batch.len();
    }
    Ok(total)
}

fn upsert_batch(client: &Client, config: &RemoteConfig, batch: &[QdrantPoint]) -> Result<()> {
    let url = format!(
        "{}/collections/{}/points?wait=true",
        config.base_url, config.collection
    );
    let body = QdrantUpsertRequest {
        points: batch.to_vec(),
    };
    let mut request = client.post(url).json(&body);
    if let Some(key) = &config.api_key {
        request = request.header("api-key", key);
    }
    let response = request.send().context("failed to send upsert request")?;
    if !response.status().is_success() {
        return Err(anyhow!("remote upsert failed"));
    }
    Ok(())
}

pub fn remote_search(
    root: &Path,
    embedder: &Embedder,
    config_query: SearchConfig,
) -> Result<SearchResponse> {
    if config_query.query.trim().is_empty() {
        return Ok(SearchResponse {
            query: config_query.query,
            total: 0,
            matches: Vec::new(),
        });
    }
    let config = config_from_env(root)?;
    let client = client_for(&config)?;
    let embedding = embedder.embed(&config_query.query)?;
    let url = format!(
        "{}/collections/{}/points/search",
        config.base_url, config.collection
    );
    let body = serde_json::json!({
        "vector": embedding,
        "limit": config_query.max_results,
        "with_payload": true,
    });
    let mut request = client.post(url).json(&body);
    if let Some(key) = &config.api_key {
        request = request.header("api-key", key);
    }
    let response = request.send().context("failed to send search request")?;
    if !response.status().is_success() {
        return Err(anyhow!("remote search failed"));
    }
    let parsed: QdrantSearchResponseBody = response
        .json()
        .context("failed to parse remote search response")?;
    let mut matches = Vec::new();
    for item in parsed.result.into_iter() {
        let payload = match item.payload {
            Some(v) => v,
            None => continue,
        };
        let path = payload.path;
        let start_line = payload.start_line;
        let end_line = payload.end_line;
        let snippet = payload.text;
        let semantic_score = item.score;
        let keyword_score = 0.0;
        let score = semantic_score;
        let search_match = SearchMatch {
            path,
            start_line,
            end_line,
            score,
            semantic_score,
            keyword_score,
            snippet,
        };
        matches.push(search_match);
    }
    let total = matches.len();
    Ok(SearchResponse {
        query: config_query.query,
        total,
        matches,
    })
}
