mod hierarchy;
mod index;
mod mmap;
pub mod utils;

pub use hierarchy::HierarchicalIndex;
pub use index::{IndexMetadata, PartialIndex, RepositoryIndex};
pub use mmap::MmapIndex;
pub use utils::quantize_to_binary;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use memmap2::Mmap;

use crate::graph::CodeGraph;
use mmap::{
    parse_binary_header, parse_vectors_header, read_vectors_from_mmap, validate_vectors_size,
    BYTES_PER_F32, VECTOR_HEADER_SIZE,
};
use utils::{data_dir, get_main_repo_path, hash_path, is_worktree};

const INDEX_FILE: &str = "index.bin.zst";
const HIERARCHY_FILE: &str = "hierarchy.bin.zst";
const FILE_VECTORS_FILE: &str = "file_vectors.bin";
const DIR_VECTORS_FILE: &str = "dir_vectors.bin";
const VECTORS_FILE: &str = "vectors.bin";
const BINARY_VECTORS_FILE: &str = "binary_vectors.bin";
const GRAPH_FILE: &str = "graph.bin.zst";
const INDEX_FORMAT_VERSION: u32 = 6;

#[derive(Debug, Clone)]
pub struct IndexStore {
    root: PathBuf,
    repo_hash: String,
}

#[allow(dead_code)]
impl IndexStore {
    pub fn new(repo_path: &Path) -> Result<Self> {
        let absolute = fs::canonicalize(repo_path).unwrap_or_else(|_| repo_path.to_path_buf());
        let repo_hash = hash_path(&absolute);
        let mut root = data_dir();
        root.push("indexes");
        root.push(&repo_hash);

        let index_exists = root.join(INDEX_FILE).exists();

        if !index_exists && is_worktree(&absolute) {
            if let Some(main_repo) = get_main_repo_path(&absolute) {
                let main_hash = hash_path(&main_repo);
                let mut main_root = data_dir();
                main_root.push("indexes");
                main_root.push(&main_hash);

                if main_root.join(INDEX_FILE).exists() {
                    fs::create_dir_all(&root)
                        .with_context(|| format!("Failed to create {}", root.display()))?;
                    copy_index_files(&main_root, &root)?;
                }
            }
        }

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

    pub fn save_graph(&self, graph: &CodeGraph) -> Result<()> {
        let graph_path = self.root.join(GRAPH_FILE);
        let bytes = bincode::serialize(graph)?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)?;
        fs::write(&graph_path, compressed)
            .with_context(|| format!("Failed to write graph to {}", graph_path.display()))?;
        Ok(())
    }

    pub fn load_graph(&self) -> Result<Option<CodeGraph>> {
        let graph_path = self.root.join(GRAPH_FILE);
        if !graph_path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&graph_path)
            .with_context(|| format!("Failed to read graph from {}", graph_path.display()))?;
        let decompressed = zstd::stream::decode_all(bytes.as_slice())?;
        let graph: CodeGraph = bincode::deserialize(&decompressed)?;
        Ok(Some(graph))
    }

    pub fn has_graph(&self) -> bool {
        self.root.join(GRAPH_FILE).exists()
    }

    pub fn save_hierarchy(&self, hier: &HierarchicalIndex) -> Result<()> {
        let hier_path = self.root.join(HIERARCHY_FILE);
        let file_vectors_path = self.root.join(FILE_VECTORS_FILE);
        let dir_vectors_path = self.root.join(DIR_VECTORS_FILE);

        let bytes = bincode::serialize(hier)?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)?;
        fs::write(&hier_path, compressed)
            .with_context(|| format!("Failed to write hierarchy to {}", hier_path.display()))?;

        self.write_hierarchy_vectors(&file_vectors_path, &hier.file_vectors)?;
        self.write_hierarchy_vectors(&dir_vectors_path, &hier.dir_vectors)?;

        Ok(())
    }

    pub fn load_hierarchy(&self) -> Result<Option<HierarchicalIndex>> {
        let hier_path = self.root.join(HIERARCHY_FILE);
        let file_vectors_path = self.root.join(FILE_VECTORS_FILE);
        let dir_vectors_path = self.root.join(DIR_VECTORS_FILE);

        if !hier_path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(&hier_path)
            .with_context(|| format!("Failed to read hierarchy from {}", hier_path.display()))?;
        let decompressed = zstd::stream::decode_all(bytes.as_slice())?;
        let mut hier: HierarchicalIndex = bincode::deserialize(&decompressed)?;

        if file_vectors_path.exists() {
            hier.file_vectors =
                self.read_hierarchy_vectors(&file_vectors_path, hier.files.len())?;
        }

        if dir_vectors_path.exists() {
            hier.dir_vectors =
                self.read_hierarchy_vectors(&dir_vectors_path, hier.directories.len())?;
        }

        Ok(Some(hier))
    }

    pub fn has_hierarchy(&self) -> bool {
        self.root.join(HIERARCHY_FILE).exists()
    }

    fn write_hierarchy_vectors(&self, path: &Path, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let file =
            File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
        let mut writer = BufWriter::new(file);

        let vector_dim = vectors.first().map(|v| v.len()).unwrap_or(0) as u32;
        writer.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&vector_dim.to_le_bytes())?;

        for vector in vectors {
            for &val in vector {
                writer.write_all(&val.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn read_hierarchy_vectors(&self, path: &Path, count: usize) -> Result<Vec<Vec<f32>>> {
        let bytes = fs::read(path)?;
        if bytes.len() < VECTOR_HEADER_SIZE {
            return Ok(vec![]);
        }

        let _format_version = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let vector_dim = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

        let mut vectors = Vec::with_capacity(count);
        let vector_bytes = vector_dim * BYTES_PER_F32;

        for i in 0..count {
            let offset = VECTOR_HEADER_SIZE + i * vector_bytes;
            if offset + vector_bytes > bytes.len() {
                break;
            }
            let vec: Vec<f32> = bytes[offset..offset + vector_bytes]
                .chunks_exact(BYTES_PER_F32)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            vectors.push(vec);
        }

        Ok(vectors)
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

        let num_words = index.metadata.vector_dim.div_ceil(64) as u32;
        writer.write_all(&INDEX_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&num_words.to_le_bytes())?;

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

fn copy_index_files(src: &Path, dst: &Path) -> Result<()> {
    let files = [
        INDEX_FILE,
        VECTORS_FILE,
        BINARY_VECTORS_FILE,
        HIERARCHY_FILE,
        FILE_VECTORS_FILE,
        DIR_VECTORS_FILE,
        GRAPH_FILE,
    ];

    for file in &files {
        let src_path = src.join(file);
        let dst_path = dst.join(file);
        if src_path.exists() {
            fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use chrono::Utc;
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

    #[test]
    #[serial]
    fn save_and_load_graph_roundtrip() {
        use crate::graph::{CodeGraph, Symbol, SymbolKind};

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("graph");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let mut graph = CodeGraph::new();
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "test_fn".to_string(),
            qualified_name: "test_fn".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 5,
            language: "rust".to_string(),
            signature: "fn test_fn()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        store.save_graph(&graph).unwrap();
        assert!(store.has_graph());

        let loaded = store.load_graph().unwrap().unwrap();
        assert_eq!(loaded.symbols.len(), 1);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn load_graph_returns_none_when_missing() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("no_graph");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        assert!(!store.has_graph());
        assert!(store.load_graph().unwrap().is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn save_and_load_hierarchy_roundtrip() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hier");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let hier = HierarchicalIndex::new();

        store.save_hierarchy(&hier).unwrap();
        assert!(store.has_hierarchy());

        let loaded = store.load_hierarchy().unwrap().unwrap();
        assert_eq!(loaded.files.len(), 0);

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn load_hierarchy_returns_none_when_missing() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("no_hier");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        assert!(!store.has_hierarchy());
        assert!(store.load_hierarchy().unwrap().is_none());

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
