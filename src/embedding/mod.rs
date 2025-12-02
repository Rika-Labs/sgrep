mod cache;
mod local;
mod providers;

pub use cache::{configure_offline_env, get_fastembed_cache_dir};
pub use local::{Embedder, PooledEmbedder};

use anyhow::Result;

pub const MODEL_NAME: &str = "jina-embeddings-v2-base-code";
pub const MODEL_DOWNLOAD_URL: &str = "https://huggingface.co/jinaai/jina-embeddings-v2-base-code/tree/main";
pub const MODEL_FILES: &[&str] = &[
    "model_quantized.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];
#[cfg(not(test))]
pub const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;
pub const DEFAULT_VECTOR_DIM: usize = 384;

pub trait BatchEmbedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
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
