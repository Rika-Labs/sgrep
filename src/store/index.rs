use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::chunker::CodeChunk;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialIndex {
    pub metadata: IndexMetadata,
    pub chunks: Vec<CodeChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryIndex {
    pub metadata: IndexMetadata,
    pub chunks: Vec<CodeChunk>,
    pub vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub version: String,
    pub repo_path: PathBuf,
    pub repo_hash: String,
    pub vector_dim: usize,
    pub indexed_at: DateTime<Utc>,
    pub total_files: usize,
    pub total_chunks: usize,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
}

fn default_embedding_model() -> String {
    crate::embedding::EmbeddingModel::default().config().name.to_string()
}

impl RepositoryIndex {
    pub fn new(metadata: IndexMetadata, chunks: Vec<CodeChunk>, vectors: Vec<Vec<f32>>) -> Self {
        debug_assert_eq!(chunks.len(), vectors.len());
        Self {
            metadata,
            chunks,
            vectors,
        }
    }
}

/// Validates that the index was built with the expected embedding model.
/// Returns an error if there's a mismatch with a helpful message.
pub fn validate_index_model(index_model: &str, current_model: &str) -> anyhow::Result<()> {
    if index_model != current_model {
        anyhow::bail!(
            "Index was built with '{}' but current model is '{}'. \
             Please re-index with: sgrep index --force",
            index_model, current_model
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_chunk() -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from("test.rs"),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: "fn test() {}".to_string(),
            hash: "abc123".to_string(),
            modified_at: Utc::now(),
        }
    }

    fn make_metadata(repo_path: PathBuf, repo_hash: String, vector_dim: usize) -> IndexMetadata {
        IndexMetadata {
            version: "0.1.0".to_string(),
            repo_path,
            repo_hash,
            vector_dim,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
            embedding_model: "jina-embeddings-v2-base-code".to_string(),
        }
    }

    #[test]
    fn repository_index_maintains_correspondence() {
        let chunk = make_chunk();
        let vector = vec![0.1, 0.2, 0.3];
        let metadata = make_metadata(PathBuf::from("/test"), "test".to_string(), 3);
        let index = RepositoryIndex::new(metadata, vec![chunk], vec![vector]);
        assert_eq!(index.chunks.len(), index.vectors.len());
    }

    #[test]
    fn validate_index_model_succeeds_when_models_match() {
        let result = validate_index_model("jina-embeddings-v2-base-code", "jina-embeddings-v2-base-code");
        assert!(result.is_ok());
    }

    #[test]
    fn validate_index_model_fails_when_models_mismatch() {
        let result = validate_index_model("mxbai-embed-xsmall-v1", "jina-embeddings-v2-base-code");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("mxbai-embed-xsmall-v1"));
        assert!(error_msg.contains("jina-embeddings-v2-base-code"));
        assert!(error_msg.contains("sgrep index --force"));
    }

    #[test]
    fn default_embedding_model_returns_mxbai() {
        assert_eq!(default_embedding_model(), "mxbai-embed-xsmall-v1");
    }
}
