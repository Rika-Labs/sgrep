use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use console::style;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::config::{Config, RemoteProviderType};
use crate::store::IndexStore;

pub mod bundle;

const UPLOAD_TEMPLATE: &str = "{prefix} Uploading vectors to {msg} ({pos}/{len}, {percent}%)";

pub mod pinecone;
pub mod turbopuffer;

/// Minimal chunk representation for remote providers.
#[derive(Clone, Debug)]
pub struct RemoteChunk {
    pub id: String,
    pub vector: Vec<f32>,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub language: String,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RemoteSearchHit {
    pub id: String,
    pub score: f32,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub language: String,
}

pub trait RemoteVectorStore: Send + Sync {
    fn name(&self) -> &'static str;
    fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()>;
    fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<RemoteSearchHit>>;
    #[allow(dead_code)]
    fn delete_namespace(&self) -> Result<()>;

    fn upsert_with_progress(&self, chunks: &[RemoteChunk], pb: &ProgressBar) -> Result<()> {
        const BATCH_SIZE: usize = 100;
        let mut uploaded = 0;

        for batch in chunks.chunks(BATCH_SIZE) {
            self.upsert(batch)?;
            uploaded += batch.len();
            pb.set_position(uploaded as u64);
        }

        Ok(())
    }
}

pub struct RemoteFactory;

impl RemoteFactory {
    pub fn build_from_config(
        cfg: &Config,
        repo_hash: &str,
    ) -> Result<Option<Arc<dyn RemoteVectorStore>>> {
        let provider = cfg
            .remote_provider
            .clone()
            .unwrap_or(RemoteProviderType::None);

        let inferred = if provider == RemoteProviderType::None {
            if cfg.pinecone.api_key.is_some() && cfg.pinecone.endpoint.is_some() {
                Some(RemoteProviderType::Pinecone)
            } else if cfg.turbopuffer.api_key.is_some() {
                Some(RemoteProviderType::Turbopuffer)
            } else {
                None
            }
        } else {
            Some(provider)
        };

        let provider = match inferred {
            Some(p) => p,
            None => return Ok(None),
        };

        match provider {
            RemoteProviderType::Turbopuffer => {
                let api_key = cfg
                    .turbopuffer
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("TURBOPUFFER_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!("Turbopuffer API key missing (set config or TURBOPUFFER_API_KEY)")
                    })?;
                let namespace = format!("{}-{}", cfg.turbopuffer.namespace_prefix, repo_hash);
                let store = turbopuffer::TurbopufferStore::new(
                    api_key,
                    namespace,
                    cfg.turbopuffer.region.clone(),
                    cfg.turbopuffer.timeout_secs,
                );
                Ok(Some(Arc::new(store)))
            }
            RemoteProviderType::Pinecone => {
                let api_key = cfg
                    .pinecone
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("PINECONE_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!("Pinecone API key missing (set config or PINECONE_API_KEY)")
                    })?;
                let endpoint = cfg.pinecone.endpoint.clone().ok_or_else(|| {
                    anyhow!("Pinecone endpoint missing (set remote.pinecone.endpoint)")
                })?;
                let namespace = cfg
                    .pinecone
                    .namespace
                    .clone()
                    .unwrap_or_else(|| repo_hash.to_string());
                let store = pinecone::PineconeStore::new(
                    api_key,
                    endpoint,
                    namespace,
                    cfg.pinecone.timeout_secs,
                );
                Ok(Some(Arc::new(store)))
            }
            RemoteProviderType::None => Ok(None),
        }
    }

    pub fn build_bundle_store(
        cfg: &Config,
        repo_hash: &str,
    ) -> Result<Option<Arc<dyn RemoteVectorStore>>> {
        let mut cfg = cfg.clone();
        if cfg.remote_provider == Some(RemoteProviderType::None) {
            cfg.remote_provider = None;
        }

        // Reuse the same inference rules as build_from_config
        let provider = cfg
            .remote_provider
            .clone()
            .unwrap_or(RemoteProviderType::None);

        let inferred = if provider == RemoteProviderType::None {
            if cfg.pinecone.api_key.is_some() && cfg.pinecone.endpoint.is_some() {
                Some(RemoteProviderType::Pinecone)
            } else if cfg.turbopuffer.api_key.is_some() {
                Some(RemoteProviderType::Turbopuffer)
            } else {
                None
            }
        } else {
            Some(provider)
        };

        let bundle_suffix = "-bundle";

        match inferred {
            Some(RemoteProviderType::Turbopuffer) => {
                let api_key = cfg
                    .turbopuffer
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("TURBOPUFFER_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!("Turbopuffer API key missing (set config or TURBOPUFFER_API_KEY)")
                    })?;
                let namespace = format!(
                    "{}-{}{}",
                    cfg.turbopuffer.namespace_prefix, repo_hash, bundle_suffix
                );
                let store = turbopuffer::TurbopufferStore::new(
                    api_key,
                    namespace,
                    cfg.turbopuffer.region.clone(),
                    cfg.turbopuffer.timeout_secs,
                );
                Ok(Some(Arc::new(store)))
            }
            Some(RemoteProviderType::Pinecone) => {
                let api_key = cfg
                    .pinecone
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("PINECONE_API_KEY").ok())
                    .ok_or_else(|| {
                        anyhow!("Pinecone API key missing (set config or PINECONE_API_KEY)")
                    })?;
                let endpoint = cfg.pinecone.endpoint.clone().ok_or_else(|| {
                    anyhow!("Pinecone endpoint missing (set remote.pinecone.endpoint)")
                })?;
                let namespace = cfg
                    .pinecone
                    .namespace
                    .clone()
                    .unwrap_or_else(|| repo_hash.to_string());
                let store = pinecone::PineconeStore::new(
                    api_key,
                    endpoint,
                    format!("{}{}", namespace, bundle_suffix),
                    cfg.pinecone.timeout_secs,
                );
                Ok(Some(Arc::new(store)))
            }
            _ => Ok(None),
        }
    }
}

pub fn push_remote_index(
    path: &Path,
    remote: &Arc<dyn RemoteVectorStore>,
    reset_namespace: bool,
) -> Result<()> {
    let store = IndexStore::new(path)?;
    let Some(index) = store.load()? else {
        return Err(anyhow!(
            "Remote push requested but no local index found for {}",
            path.display()
        ));
    };

    if reset_namespace {
        remote.delete_namespace()?;
    }

    let chunks: Vec<RemoteChunk> = index
        .chunks
        .iter()
        .zip(index.vectors.iter())
        .map(|(chunk, vec)| RemoteChunk {
            id: chunk.hash.clone(),
            vector: vec.clone(),
            path: chunk.path.display().to_string(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            content: chunk.text.clone(),
            language: chunk.language.clone(),
        })
        .collect();

    if chunks.is_empty() {
        return Ok(());
    }

    let info_prefix = style("[info]").blue().bold().to_string();
    let pb = ProgressBar::with_draw_target(Some(chunks.len() as u64), ProgressDrawTarget::stderr());
    pb.set_prefix(info_prefix.clone());
    pb.set_style(
        ProgressStyle::with_template(UPLOAD_TEMPLATE)
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_message(remote.name().to_string());

    remote.upsert_with_progress(&chunks, &pb)?;

    pb.finish_with_message(remote.name().to_string());
    Ok(())
}

pub fn push_remote_bundle(
    path: &Path,
    remote: &Arc<dyn RemoteVectorStore>,
    vector_dim: usize,
    reset_namespace: bool,
) -> Result<bundle::BundleManifest> {
    let parts = bundle::build_bundle(path, None)?;

    if reset_namespace {
        bundle::delete_bundle_namespace(remote.as_ref())?;
    }

    bundle::upload_bundle(remote.as_ref(), &parts, vector_dim)?;
    Ok(parts.manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn upload_template_is_consistent() {
        assert_eq!(
            UPLOAD_TEMPLATE,
            "{prefix} Uploading vectors to {msg} ({pos}/{len}, {percent}%)"
        );
    }

    #[test]
    fn none_provider_returns_none_when_not_inferred() {
        let cfg = Config::default();
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_none());
    }

    #[test]
    fn turbopuffer_uses_config_api_key() {
        let mut cfg = Config::default();
        cfg.remote_provider = Some(RemoteProviderType::Turbopuffer);
        cfg.turbopuffer.api_key = Some("key".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
        assert_eq!(store.unwrap().name(), "turbopuffer");
    }

    #[test]
    fn turbopuffer_infers_from_config() {
        let mut cfg = Config::default();
        cfg.turbopuffer.api_key = Some("key".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
    }

    #[test]
    fn pinecone_builds_with_config() {
        let mut cfg = Config::default();
        cfg.remote_provider = Some(RemoteProviderType::Pinecone);
        cfg.pinecone.api_key = Some("pckey".to_string());
        cfg.pinecone.endpoint = Some("https://svc.test/pinecone".to_string());
        cfg.pinecone.namespace = Some("ns".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
        assert_eq!(store.unwrap().name(), "pinecone");
    }

    #[test]
    fn pinecone_infers_with_api_key_and_endpoint() {
        let mut cfg = Config::default();
        cfg.pinecone.api_key = Some("pckey".to_string());
        cfg.pinecone.endpoint = Some("https://svc.test/pinecone".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
    }

    #[test]
    fn upsert_with_progress_batches_and_updates_position() {
        struct MockStore {
            batches: Mutex<Vec<usize>>,
        }

        impl MockStore {
            fn new() -> Self {
                Self {
                    batches: Mutex::new(Vec::new()),
                }
            }
        }

        impl RemoteVectorStore for MockStore {
            fn name(&self) -> &'static str {
                "mock"
            }

            fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()> {
                self.batches.lock().unwrap().push(chunks.len());
                Ok(())
            }

            fn query(&self, _vector: &[f32], _top_k: usize) -> Result<Vec<RemoteSearchHit>> {
                Ok(Vec::new())
            }

            fn delete_namespace(&self) -> Result<()> {
                Ok(())
            }
        }

        fn make_chunk(id: usize) -> RemoteChunk {
            RemoteChunk {
                id: format!("id-{id}"),
                vector: vec![0.0, 1.0],
                path: format!("file-{id}.rs"),
                start_line: 1,
                end_line: 2,
                content: "code".to_string(),
                language: "rust".to_string(),
            }
        }

        let store = MockStore::new();
        let pb = ProgressBar::hidden();
        let chunks: Vec<RemoteChunk> = (0..250).map(make_chunk).collect();

        RemoteVectorStore::upsert_with_progress(&store, &chunks, &pb).unwrap();

        let batches = store.batches.lock().unwrap().clone();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], 100);
        assert_eq!(batches[1], 100);
        assert_eq!(batches[2], 50);
        assert_eq!(pb.position(), chunks.len() as u64);
    }
}
