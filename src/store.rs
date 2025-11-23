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
