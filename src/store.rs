use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::chunker::CodeChunk;

const INDEX_FILE: &str = "index.bin.zst";

#[derive(Debug, Clone)]
pub struct IndexStore {
    root: PathBuf,
    repo_hash: String,
}

impl IndexStore {
    pub fn new(repo_path: &Path) -> Result<Self> {
        let absolute = fs::canonicalize(repo_path).unwrap_or_else(|_| repo_path.to_path_buf());
        let repo_hash = hash_path(&absolute);
        let mut root = data_dir();
        root.push("indexes");
        root.push(&repo_hash);
        fs::create_dir_all(&root)
            .with_context(|| format!("Failed to create {}", root.display()))?;
        Ok(Self { root, repo_hash })
    }

    pub fn load(&self) -> Result<Option<RepositoryIndex>> {
        let path = self.root.join(INDEX_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let bytes =
            fs::read(&path).with_context(|| format!("Failed to read {}", path.display()))?;
        let buffer = zstd::stream::decode_all(bytes.as_slice())?;
        let index: RepositoryIndex = bincode::deserialize(&buffer)?;
        Ok(Some(index))
    }

    pub fn save(&self, index: &RepositoryIndex) -> Result<()> {
        let path = self.root.join(INDEX_FILE);
        let bytes = bincode::serialize(index)?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)?;
        fs::write(&path, compressed)?;
        Ok(())
    }

    pub fn repo_hash(&self) -> &str {
        &self.repo_hash
    }
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

fn data_dir() -> PathBuf {
    if let Some(dirs) = ProjectDirs::from("dev", "RikaLabs", "sgrep") {
        dirs.data_local_dir().to_path_buf()
    } else {
        dirs_next_best()
    }
}

fn dirs_next_best() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sgrep")
}

fn hash_path(path: &Path) -> String {
    let mut hasher = Hasher::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let digest = hasher.finalize();
    digest.to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn create_sample_chunk() -> CodeChunk {
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

    #[test]
    fn hash_path_is_deterministic() {
        let path = Path::new("/test/path");
        let hash1 = hash_path(path);
        let hash2 = hash_path(path);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn hash_path_differs_for_different_paths() {
        let hash1 = hash_path(Path::new("/path/one"));
        let hash2 = hash_path(Path::new("/path/two"));
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn repository_index_maintains_chunk_vector_correspondence() {
        let chunk = create_sample_chunk();
        let vector = vec![0.1, 0.2, 0.3];
        let metadata = IndexMetadata {
            version: "0.1.0".to_string(),
            repo_path: PathBuf::from("/test"),
            repo_hash: "test123".to_string(),
            vector_dim: 3,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        
        let index = RepositoryIndex::new(metadata, vec![chunk], vec![vector]);
        assert_eq!(index.chunks.len(), index.vectors.len());
        assert_eq!(index.chunks.len(), 1);
    }

    #[test]
    fn index_store_creates_deterministic_hash() {
        let temp_dir = std::env::temp_dir().join("sgrep_test_repo");
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let store1 = IndexStore::new(&temp_dir).unwrap();
        let store2 = IndexStore::new(&temp_dir).unwrap();
        
        assert_eq!(store1.repo_hash(), store2.repo_hash());
        
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn save_and_load_roundtrip() {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let store = IndexStore::new(&temp_dir).unwrap();
        let chunk = create_sample_chunk();
        let vector = vec![0.1, 0.2, 0.3];
        let metadata = IndexMetadata {
            version: "0.1.0".to_string(),
            repo_path: temp_dir.clone(),
            repo_hash: store.repo_hash().to_string(),
            vector_dim: 3,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        
        let original_index = RepositoryIndex::new(metadata, vec![chunk.clone()], vec![vector.clone()]);
        store.save(&original_index).unwrap();
        
        let loaded_index = store.load().unwrap();
        assert!(loaded_index.is_some());
        
        let loaded = loaded_index.unwrap();
        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.vectors.len(), 1);
        assert_eq!(loaded.chunks[0].text, chunk.text);
        assert_eq!(loaded.vectors[0], vector);
        
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn load_returns_none_for_nonexistent_index() {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_test_empty_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let store = IndexStore::new(&temp_dir).unwrap();
        let result = store.load().unwrap();
        assert!(result.is_none());
        
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
