use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::config::{RemoteConfig, RemoteProviderType};

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
}

pub struct RemoteFactory;

impl RemoteFactory {
    pub fn build_from_config(
        cfg: &RemoteConfig,
        repo_hash: &str,
    ) -> Result<Option<Arc<dyn RemoteVectorStore>>> {
        let provider = match cfg.provider.clone().unwrap_or(RemoteProviderType::None) {
            RemoteProviderType::None => return Ok(None),
            other => other,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn none_provider_returns_none() {
        let cfg = RemoteConfig::default();
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_none());
    }

    #[test]
    fn turbopuffer_uses_config_api_key() {
        let mut cfg = RemoteConfig::default();
        cfg.provider = Some(RemoteProviderType::Turbopuffer);
        cfg.turbopuffer.api_key = Some("key".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
        assert_eq!(store.unwrap().name(), "turbopuffer");
    }

    #[test]
    fn turbopuffer_reads_env_api_key() {
        let mut cfg = RemoteConfig::default();
        cfg.provider = Some(RemoteProviderType::Turbopuffer);
        env::set_var("TURBOPUFFER_API_KEY", "envkey");
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        env::remove_var("TURBOPUFFER_API_KEY");
        assert!(store.is_some());
    }

    #[test]
    fn pinecone_builds_with_config() {
        let mut cfg = RemoteConfig::default();
        cfg.provider = Some(RemoteProviderType::Pinecone);
        cfg.pinecone.api_key = Some("pckey".to_string());
        cfg.pinecone.endpoint = Some("https://svc.test/pinecone".to_string());
        cfg.pinecone.namespace = Some("ns".to_string());
        let store = RemoteFactory::build_from_config(&cfg, "hash").unwrap();
        assert!(store.is_some());
        assert_eq!(store.unwrap().name(), "pinecone");
    }
}
