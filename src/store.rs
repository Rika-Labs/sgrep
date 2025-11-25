use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use blake3::Hasher;
use chrono::{DateTime, Utc};
use directories::ProjectDirs;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::chunker::CodeChunk;

const INDEX_FILE: &str = "index.bin.zst";
const VECTORS_FILE: &str = "vectors.bin";
const BINARY_VECTORS_FILE: &str = "binary_vectors.bin";
const INDEX_FORMAT_VERSION: u32 = 3;
const VECTOR_HEADER_SIZE: usize = 8;
const BYTES_PER_F32: usize = 4;
const BYTES_PER_U64: usize = 8;

/// Quantize f32 vector to binary (1 bit per dimension)
/// Each dimension becomes 1 if > 0.0, else 0
fn quantize_to_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = vector.len().div_ceil(64);
    let mut binary = vec![0u64; num_words];
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            binary[i / 64] |= 1u64 << (i % 64);
        }
    }
    binary
}

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
        let vectors_path = self.root.join(VECTORS_FILE);
        let index_path = self.root.join(INDEX_FILE);

        if vectors_path.exists() && index_path.exists() {
            return self.load_mmap_format(&index_path, &vectors_path);
        }

        if index_path.exists() {
            return self.load_legacy_format(&index_path);
        }

        Ok(None)
    }

    pub fn save(&self, index: &RepositoryIndex) -> Result<()> {
        let vectors_path = self.root.join(VECTORS_FILE);
        let binary_vectors_path = self.root.join(BINARY_VECTORS_FILE);
        let index_path = self.root.join(INDEX_FILE);

        self.write_vectors_file(&vectors_path, index)?;
        self.write_binary_vectors_file(&binary_vectors_path, index)?;
        self.write_index_file(&index_path, index)?;

        Ok(())
    }

    pub fn load_mmap(&self) -> Result<Option<MmapIndex>> {
        let vectors_path = self.root.join(VECTORS_FILE);
        let binary_vectors_path = self.root.join(BINARY_VECTORS_FILE);
        let index_path = self.root.join(INDEX_FILE);

        if !vectors_path.exists() || !index_path.exists() {
            return Ok(None);
        }

        let partial = self.read_partial_index(&index_path)?;
        let mmap = self.open_vectors_mmap(&vectors_path)?;
        let (format_version, vector_dim) = parse_vectors_header(&mmap)?;

        if format_version != INDEX_FORMAT_VERSION {
            return Ok(None);
        }

        validate_vectors_size(&mmap, vector_dim, partial.chunks.len())?;

        // Try to load binary vectors (optional - gracefully degrade if missing)
        let (binary_mmap, binary_words) = if binary_vectors_path.exists() {
            match self.open_vectors_mmap(&binary_vectors_path) {
                Ok(bin_mmap) => {
                    let (bin_version, num_words) = parse_binary_header(&bin_mmap)?;
                    if bin_version == INDEX_FORMAT_VERSION {
                        (Some(bin_mmap), num_words)
                    } else {
                        (None, 0)
                    }
                }
                Err(_) => (None, 0),
            }
        } else {
            (None, 0)
        };

        Ok(Some(MmapIndex {
            metadata: partial.metadata,
            chunks: partial.chunks,
            mmap,
            binary_mmap,
            vector_dim,
            binary_words,
        }))
    }

    pub fn repo_hash(&self) -> &str {
        &self.repo_hash
    }

    fn load_legacy_format(&self, path: &Path) -> Result<Option<RepositoryIndex>> {
        let bytes = fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
        let buffer = zstd::stream::decode_all(bytes.as_slice())?;
        let index: RepositoryIndex = bincode::deserialize(&buffer)?;
        Ok(Some(index))
    }

    fn load_mmap_format(
        &self,
        index_path: &Path,
        vectors_path: &Path,
    ) -> Result<Option<RepositoryIndex>> {
        let partial = self.read_partial_index(index_path)?;
        let mmap = self.open_vectors_mmap(vectors_path)?;
        let (format_version, vector_dim) = parse_vectors_header(&mmap)?;

        if format_version != INDEX_FORMAT_VERSION {
            return self.load_legacy_format(index_path);
        }

        validate_vectors_size(&mmap, vector_dim, partial.chunks.len())?;

        let vectors = read_vectors_from_mmap(&mmap, vector_dim, partial.chunks.len());

        Ok(Some(RepositoryIndex {
            metadata: partial.metadata,
            chunks: partial.chunks,
            vectors,
        }))
    }

    fn read_partial_index(&self, path: &Path) -> Result<PartialIndex> {
        let bytes = fs::read(path).with_context(|| format!("Failed to read {}", path.display()))?;
        let buffer = zstd::stream::decode_all(bytes.as_slice())?;
        Ok(bincode::deserialize(&buffer)?)
    }

    fn open_vectors_mmap(&self, path: &Path) -> Result<Mmap> {
        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        Ok(unsafe { Mmap::map(&file)? })
    }

    fn write_vectors_file(&self, path: &Path, index: &RepositoryIndex) -> Result<()> {
        let file =
            File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&(index.metadata.vector_dim as u32).to_le_bytes())?;

        for vector in &index.vectors {
            for &val in vector {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_binary_vectors_file(&self, path: &Path, index: &RepositoryIndex) -> Result<()> {
        let file =
            File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        // Header: version (4 bytes) + num_words per vector (4 bytes)
        let num_words = index.metadata.vector_dim.div_ceil(64) as u32;
        writer.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&num_words.to_le_bytes())?;

        // Write binary vectors
        for vector in &index.vectors {
            let binary = quantize_to_binary(vector);
            for word in &binary {
                writer.write_all(&word.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn write_index_file(&self, path: &Path, index: &RepositoryIndex) -> Result<()> {
        let partial = PartialIndex {
            metadata: index.metadata.clone(),
            chunks: index.chunks.clone(),
        };
        let bytes = bincode::serialize(&partial)?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)?;
        fs::write(path, compressed)?;
        Ok(())
    }
}

fn parse_vectors_header(mmap: &Mmap) -> Result<(u32, usize)> {
    if mmap.len() < VECTOR_HEADER_SIZE {
        return Err(anyhow::anyhow!("Invalid vectors file: too small"));
    }
    let format_version = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    let vector_dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    Ok((format_version, vector_dim))
}

fn parse_binary_header(mmap: &Mmap) -> Result<(u32, usize)> {
    if mmap.len() < VECTOR_HEADER_SIZE {
        return Err(anyhow::anyhow!("Invalid binary vectors file: too small"));
    }
    let format_version = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    let num_words = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    Ok((format_version, num_words))
}

fn validate_vectors_size(mmap: &Mmap, vector_dim: usize, num_vectors: usize) -> Result<()> {
    let vector_bytes = vector_dim * BYTES_PER_F32;
    let expected_size = VECTOR_HEADER_SIZE + num_vectors * vector_bytes;
    if mmap.len() < expected_size {
        return Err(anyhow::anyhow!(
            "Invalid vectors file: expected {} bytes, got {}",
            expected_size,
            mmap.len()
        ));
    }
    Ok(())
}

fn read_vectors_from_mmap(mmap: &Mmap, vector_dim: usize, num_vectors: usize) -> Vec<Vec<f32>> {
    let vector_bytes = vector_dim * BYTES_PER_F32;
    (0..num_vectors)
        .map(|i| {
            let offset = VECTOR_HEADER_SIZE + i * vector_bytes;
            mmap[offset..offset + vector_bytes]
                .chunks_exact(BYTES_PER_F32)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartialIndex {
    metadata: IndexMetadata,
    chunks: Vec<CodeChunk>,
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

pub struct MmapIndex {
    pub metadata: IndexMetadata,
    pub chunks: Vec<CodeChunk>,
    mmap: Mmap,
    binary_mmap: Option<Mmap>,
    vector_dim: usize,
    binary_words: usize,
}

impl MmapIndex {
    #[inline]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        let vector_bytes = self.vector_dim * BYTES_PER_F32;
        let offset = VECTOR_HEADER_SIZE + idx * vector_bytes;
        let slice = &self.mmap[offset..offset + vector_bytes];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.vector_dim) }
    }

    #[inline]
    pub fn get_binary_vector(&self, idx: usize) -> Option<&[u64]> {
        let mmap = self.binary_mmap.as_ref()?;
        let binary_bytes = self.binary_words * BYTES_PER_U64;
        let offset = VECTOR_HEADER_SIZE + idx * binary_bytes;
        let slice = &mmap[offset..offset + binary_bytes];
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u64, self.binary_words) })
    }

    #[inline]
    pub fn has_binary_vectors(&self) -> bool {
        self.binary_mmap.is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn to_repository_index(&self) -> RepositoryIndex {
        let vectors: Vec<Vec<f32>> = (0..self.len())
            .map(|i| self.get_vector(i).to_vec())
            .collect();
        RepositoryIndex {
            metadata: self.metadata.clone(),
            chunks: self.chunks.clone(),
            vectors,
        }
    }
}

fn data_dir() -> PathBuf {
    if let Ok(home) = env::var("SGREP_HOME") {
        return PathBuf::from(home);
    }
    ProjectDirs::from("dev", "RikaLabs", "sgrep")
        .map(|d| d.data_local_dir().to_path_buf())
        .unwrap_or_else(fallback_data_dir)
}

fn fallback_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sgrep")
}

fn hash_path(path: &Path) -> String {
    let mut hasher = Hasher::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use uuid::Uuid;

    fn temp_dir_with_name(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sgrep_test_{}_{}", name, Uuid::new_v4()))
    }

    fn set_test_home() -> PathBuf {
        let temp_dir = temp_dir_with_name("home");
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::env::set_var("SGREP_HOME", &temp_dir);
        temp_dir
    }

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
        }
    }

    #[test]
    fn hash_is_deterministic() {
        let path = Path::new("/test/path");
        assert_eq!(hash_path(path), hash_path(path));
    }

    #[test]
    fn hash_differs_for_different_paths() {
        assert_ne!(hash_path(Path::new("/a")), hash_path(Path::new("/b")));
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
    #[serial]
    fn store_creates_deterministic_hash() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("repo");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store1 = IndexStore::new(&temp_dir).unwrap();
        let store2 = IndexStore::new(&temp_dir).unwrap();
        assert_eq!(store1.repo_hash(), store2.repo_hash());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn save_and_load_roundtrip() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("roundtrip");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let chunk = make_chunk();
        let vector = vec![0.1, 0.2, 0.3];
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), 3);
        let original = RepositoryIndex::new(metadata, vec![chunk.clone()], vec![vector.clone()]);

        store.save(&original).unwrap();
        let loaded = store.load().unwrap().unwrap();

        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.vectors.len(), 1);
        assert_eq!(loaded.chunks[0].text, chunk.text);
        assert_eq!(loaded.vectors[0], vector);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn load_returns_none_for_missing_index() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("empty");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        assert!(store.load().unwrap().is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn mmap_index_provides_zero_copy_access() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("mmap");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let chunk = make_chunk();
        let vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), 3);
        let original = RepositoryIndex::new(metadata, vec![chunk.clone(), chunk], vectors.clone());

        store.save(&original).unwrap();
        let mmap_index = store.load_mmap().unwrap().unwrap();

        assert_eq!(mmap_index.len(), 2);
        assert_eq!(mmap_index.get_vector(0), &vectors[0][..]);
        assert_eq!(mmap_index.get_vector(1), &vectors[1][..]);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn mmap_index_converts_to_repository_index() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("convert");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let chunk = make_chunk();
        let vector = vec![1.5, 2.5, 3.5];
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), 3);
        let original = RepositoryIndex::new(metadata, vec![chunk], vec![vector.clone()]);

        store.save(&original).unwrap();
        let mmap_index = store.load_mmap().unwrap().unwrap();
        let converted = mmap_index.to_repository_index();

        assert_eq!(converted.vectors[0], vector);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn mmap_returns_none_for_missing_index() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("mmap_missing");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        assert!(store.load_mmap().unwrap().is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn data_dir_uses_sgrep_home_when_set() {
        let prev = std::env::var("SGREP_HOME").ok();
        std::env::set_var("SGREP_HOME", "/custom/path");
        assert_eq!(data_dir(), PathBuf::from("/custom/path"));
        match prev {
            Some(v) => std::env::set_var("SGREP_HOME", v),
            None => std::env::remove_var("SGREP_HOME"),
        }
    }

    #[test]
    fn fallback_uses_home_directory() {
        let dir = fallback_data_dir();
        assert!(dir.to_string_lossy().contains(".sgrep"));
    }

    #[test]
    #[serial]
    fn save_creates_both_files() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("files");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let chunk = make_chunk();
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), 3);
        let index = RepositoryIndex::new(metadata, vec![chunk], vec![vec![0.1, 0.2, 0.3]]);

        store.save(&index).unwrap();

        assert!(store.root.join(INDEX_FILE).exists());
        assert!(store.root.join(VECTORS_FILE).exists());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn mmap_is_empty_works() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("empty_check");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), 3);
        let index = RepositoryIndex::new(metadata, vec![], vec![]);

        store.save(&index).unwrap();
        let mmap_index = store.load_mmap().unwrap().unwrap();

        assert!(mmap_index.is_empty());
        assert_eq!(mmap_index.len(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
