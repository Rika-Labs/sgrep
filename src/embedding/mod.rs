mod cache;
mod local;
mod providers;

pub use cache::{configure_offline_env, get_fastembed_cache_dir};
pub use local::{Embedder, PooledEmbedder};

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingModel {
    #[default]
    Mxbai,
    Jina,
}

pub struct ModelConfig {
    pub name: &'static str,
    pub display_name: &'static str,
    pub download_base_url: &'static str,
    pub files: &'static [(&'static str, &'static str)],
    pub native_dim: usize,
    pub output_dim: usize,
}

pub const MXBAI_CONFIG: ModelConfig = ModelConfig {
    name: "mxbai-embed-xsmall-v1",
    display_name: "Mixedbread mxbai-embed-xsmall-v1",
    download_base_url: "https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main",
    files: &[
        ("onnx/model.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
    ],
    native_dim: 384,
    output_dim: 384,
};

pub const JINA_CONFIG: ModelConfig = ModelConfig {
    name: "jina-embeddings-v2-base-code",
    display_name: "Jina Embeddings v2 Base Code",
    download_base_url: "https://huggingface.co/jinaai/jina-embeddings-v2-base-code/resolve/main",
    files: &[
        ("onnx/model_quantized.onnx", "model_quantized.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
    ],
    native_dim: 768,
    output_dim: 384,
};

impl EmbeddingModel {
    pub fn config(&self) -> &'static ModelConfig {
        match self {
            EmbeddingModel::Mxbai => &MXBAI_CONFIG,
            EmbeddingModel::Jina => &JINA_CONFIG,
        }
    }

    #[allow(dead_code)]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "mxbai" | "mixedbread" | "mxbai-embed-xsmall-v1" => Some(EmbeddingModel::Mxbai),
            "jina" | "jina-embeddings-v2-base-code" => Some(EmbeddingModel::Jina),
            _ => None,
        }
    }
}

#[cfg(not(test))]
pub const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;
#[cfg(test)]
pub const DEFAULT_VECTOR_DIM: usize = 384;

#[derive(Debug, Clone)]
pub struct EmbedProgress {
    pub completed: usize,
    #[allow(dead_code)]
    pub total: usize,
    #[allow(dead_code)]
    pub message: Option<String>,
}

pub type ProgressCallback = Box<dyn Fn(EmbedProgress) + Send + Sync>;

pub trait BatchEmbedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed texts with progress reporting.
    /// Default implementation calls embed_batch and reports completion at end.
    fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        let result = self.embed_batch(texts)?;
        if let Some(callback) = on_progress {
            callback(EmbedProgress {
                completed: texts.len(),
                total: texts.len(),
                message: Some("complete".to_string()),
            });
        }
        Ok(result)
    }

    fn dimension(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct FakeEmbedder;

    impl BatchEmbedder for FakeEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 0.0, 1.0])
                .collect())
        }

        fn dimension(&self) -> usize {
            3
        }
    }

    #[test]
    fn batch_embedder_embed_calls_batch() {
        let embedder = FakeEmbedder::default();
        let v = embedder.embed("hi").unwrap();
        assert_eq!(v, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn embed_errors_when_batch_returns_empty() {
        struct EmptyEmbedder;
        impl BatchEmbedder for EmptyEmbedder {
            fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(Vec::new())
            }
            fn dimension(&self) -> usize {
                4
            }
        }

        let embedder = EmptyEmbedder;
        let result = embedder.embed("hi");
        assert!(result.is_err());
    }
}
