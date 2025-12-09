mod hierarchy;
mod index;
mod mmap;
pub mod utils;

pub use hierarchy::HierarchicalIndex;
pub use index::{validate_index_model, IndexMetadata, PartialIndex, RepositoryIndex};
pub use mmap::{HnswHeader, MmapIndex};
pub use utils::quantize_to_binary;

use std::fs::{self, File};
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, System};

#[derive(Debug, Clone, Serialize)]
pub struct IndexStats {
    pub repo_path: PathBuf,
    pub indexed_at: DateTime<Utc>,
    pub vector_dim: usize,
    pub total_files: usize,
    pub total_chunks: usize,
    pub graph_symbols: usize,
    pub graph_edges: usize,
    pub mmap_available: bool,
    pub binary_vectors_available: bool,
}

use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::graph::CodeGraph;
use crate::search::config::{
    HNSW_CONNECTIVITY, HNSW_EXPANSION_ADD, HNSW_EXPANSION_SEARCH, HNSW_THRESHOLD,
};
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
const HNSW_FILE: &str = "hnsw.usearch";
const HNSW_HEADER_FILE: &str = "hnsw_header.bin";
const INDEX_FORMAT_VERSION: u32 = 6;
const BUILDING_FILE: &str = "index.building";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BuildMarker {
    pub(crate) pid: u32,
    pub(crate) started_at: DateTime<Utc>,
    pub(crate) version: String,
}

impl BuildMarker {
    fn new_current() -> Self {
        Self {
            pid: std::process::id(),
            started_at: Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexHealth {
    Ready,
    Missing,
    Partial,
}

#[derive(Debug, Clone)]
pub enum BuildState {
    InProgress(BuildMarker),
    Interrupted(Option<BuildMarker>),
    Ready,
    Missing,
}

#[derive(Debug, Clone)]
pub struct IndexStore {
    root: PathBuf,
    repo_hash: String,
    repo_path: PathBuf,
}

pub struct BuildGuard {
    path: PathBuf,
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
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
        Ok(Self {
            root,
            repo_hash,
            repo_path: absolute,
        })
    }

    pub fn is_building(&self) -> bool {
        matches!(self.build_state(), BuildState::InProgress(_))
    }

    pub fn build_state(&self) -> BuildState {
        let marker_path = self.root.join(BUILDING_FILE);
        let marker = self.read_build_marker(&marker_path);
        if let Some(marker) = marker {
            if Self::is_pid_alive(marker.pid) {
                return BuildState::InProgress(marker);
            }

            return BuildState::Interrupted(Some(marker));
        }

        let health = self.index_health();
        match health {
            IndexHealth::Ready => BuildState::Ready,
            IndexHealth::Missing => BuildState::Missing,
            IndexHealth::Partial => BuildState::Interrupted(None),
        }
    }

    pub fn start_build_guard(&self) -> Result<BuildGuard> {
        if let BuildState::InProgress(marker) = self.build_state() {
            return Err(anyhow!(
                "Index build already in progress for {} (pid {}, started {})",
                self.repo_path.display(),
                marker.pid,
                marker.started_at
            ));
        }

        let path = self.root.join(BUILDING_FILE);
        let _ = fs::remove_file(&path);

        let marker = BuildMarker::new_current();
        let payload = serde_json::to_vec(&marker)
            .with_context(|| format!("Failed to serialize build marker for {}", path.display()))?;

        fs::write(&path, payload)
            .with_context(|| format!("Failed to create {}", path.display()))?;

        Ok(BuildGuard { path })
    }

    fn read_build_marker(&self, path: &Path) -> Option<BuildMarker> {
        let content = fs::read_to_string(path).ok()?;
        if let Ok(marker) = serde_json::from_str::<BuildMarker>(&content) {
            return Some(marker);
        }

        if content.trim().is_empty() {
            return None;
        }

        // Legacy markers were plain text (e.g., "building"); treat as stale
        Some(BuildMarker {
            pid: 0,
            started_at: Utc::now(),
            version: "legacy".to_string(),
        })
    }

    fn index_health(&self) -> IndexHealth {
        let has_any_artifact = [
            INDEX_FILE,
            VECTORS_FILE,
            HIERARCHY_FILE,
            DIR_VECTORS_FILE,
            FILE_VECTORS_FILE,
            GRAPH_FILE,
            BINARY_VECTORS_FILE,
            HNSW_FILE,
            HNSW_HEADER_FILE,
        ]
        .iter()
        .any(|file| self.root.join(file).exists());

        match self.load() {
            Ok(Some(_)) => IndexHealth::Ready,
            Ok(None) => {
                if has_any_artifact {
                    IndexHealth::Partial
                } else {
                    IndexHealth::Missing
                }
            }
            Err(_) => IndexHealth::Partial,
        }
    }

    fn is_pid_alive(pid: u32) -> bool {
        let system = System::new_all();
        system.process(Pid::from_u32(pid)).is_some()
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
        self.save_hnsw(index)?;

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

        let hnsw = self.load_hnsw(vector_dim, partial.chunks.len());

        Ok(Some(MmapIndex {
            metadata: partial.metadata,
            chunks: partial.chunks,
            mmap,
            binary_mmap,
            vector_dim,
            binary_words,
            hnsw,
        }))
    }

    pub fn repo_hash(&self) -> &str {
        &self.repo_hash
    }

    pub fn get_stats(&self) -> Result<Option<IndexStats>> {
        let mmap = self.load_mmap()?;
        let Some(mmap) = mmap else {
            return Ok(None);
        };

        let graph = self.load_graph()?;
        let (graph_symbols, graph_edges) = graph
            .map(|g| (g.symbols.len(), g.edges.len()))
            .unwrap_or((0, 0));

        Ok(Some(IndexStats {
            repo_path: mmap.metadata.repo_path,
            indexed_at: mmap.metadata.indexed_at,
            vector_dim: mmap.vector_dim,
            total_files: mmap.metadata.total_files,
            total_chunks: mmap.metadata.total_chunks,
            graph_symbols,
            graph_edges,
            mmap_available: true,
            binary_vectors_available: mmap.binary_mmap.is_some(),
        }))
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

    pub fn save_hnsw(&self, index: &RepositoryIndex) -> Result<()> {
        let num_vectors = index.vectors.len();
        if num_vectors < HNSW_THRESHOLD {
            return Ok(());
        }

        let options = IndexOptions {
            dimensions: index.metadata.vector_dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: HNSW_CONNECTIVITY,
            expansion_add: HNSW_EXPANSION_ADD,
            expansion_search: HNSW_EXPANSION_SEARCH,
            multi: false,
        };

        let hnsw = Index::new(&options)
            .map_err(|e| anyhow::anyhow!("Failed to create HNSW index: {}", e))?;
        hnsw.reserve(num_vectors)
            .map_err(|e| anyhow::anyhow!("Failed to reserve HNSW capacity: {}", e))?;

        for (i, vector) in index.vectors.iter().enumerate() {
            hnsw.add(i as u64, vector)
                .map_err(|e| anyhow::anyhow!("Failed to add vector {} to HNSW: {}", i, e))?;
        }

        let hnsw_path = self.root.join(HNSW_FILE);
        hnsw.save(&hnsw_path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("Failed to save HNSW index: {}", e))?;

        let header = HnswHeader {
            format_version: INDEX_FORMAT_VERSION,
            vector_dim: index.metadata.vector_dim as u32,
            connectivity: HNSW_CONNECTIVITY as u32,
            expansion_add: HNSW_EXPANSION_ADD as u32,
            num_vectors: num_vectors as u32,
        };
        let header_path = self.root.join(HNSW_HEADER_FILE);
        fs::write(&header_path, header.to_bytes())
            .with_context(|| format!("Failed to write HNSW header to {}", header_path.display()))?;

        Ok(())
    }

    fn load_hnsw(&self, expected_dim: usize, expected_count: usize) -> Option<Index> {
        let hnsw_path = self.root.join(HNSW_FILE);
        let header_path = self.root.join(HNSW_HEADER_FILE);

        if !hnsw_path.exists() || !header_path.exists() {
            return None;
        }

        let header_bytes = fs::read(&header_path).ok()?;
        let header = HnswHeader::from_bytes(&header_bytes).ok()?;

        if header.format_version != INDEX_FORMAT_VERSION
            || header.vector_dim as usize != expected_dim
            || header.connectivity as usize != HNSW_CONNECTIVITY
            || header.expansion_add as usize != HNSW_EXPANSION_ADD
            || header.num_vectors as usize != expected_count
        {
            return None;
        }

        let options = IndexOptions {
            dimensions: expected_dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: HNSW_CONNECTIVITY,
            expansion_add: HNSW_EXPANSION_ADD,
            expansion_search: HNSW_EXPANSION_SEARCH,
            multi: false,
        };

        let hnsw = Index::new(&options).ok()?;
        hnsw.load(&hnsw_path.to_string_lossy()).ok()?;

        Some(hnsw)
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
        HNSW_FILE,
        HNSW_HEADER_FILE,
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
    use serde_json;
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
            embedding_model: "jina-embeddings-v2-base-code".to_string(),
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
    fn build_state_reports_stale_marker_as_interrupted() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("stale_marker");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let marker_path = store.root.join(BUILDING_FILE);
        let marker = BuildMarker {
            pid: u32::MAX,
            started_at: Utc::now(),
            version: "test".to_string(),
        };
        fs::write(&marker_path, serde_json::to_vec(&marker).unwrap()).unwrap();

        assert!(matches!(
            store.build_state(),
            BuildState::Interrupted(Some(_))
        ));

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn start_build_guard_removes_stale_marker_and_allows_new_build() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("stale_guard");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let marker_path = store.root.join(BUILDING_FILE);
        fs::write(&marker_path, b"building").unwrap();

        let guard = store.start_build_guard().unwrap();
        assert!(matches!(store.build_state(), BuildState::InProgress(_)));

        drop(guard);
        assert!(!marker_path.exists());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn build_state_detects_partial_without_marker() {
        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("partial");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        // Write a single artifact to simulate an interrupted build
        fs::write(store.root.join(VECTORS_FILE), b"partial").unwrap();

        assert!(matches!(store.build_state(), BuildState::Interrupted(None)));

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

    fn make_chunks_with_vectors(
        count: usize,
        vector_dim: usize,
    ) -> (Vec<CodeChunk>, Vec<Vec<f32>>) {
        let chunks: Vec<CodeChunk> = (0..count)
            .map(|i| CodeChunk {
                id: Uuid::new_v4(),
                path: PathBuf::from(format!("test_{}.rs", i)),
                language: "rust".to_string(),
                start_line: 1,
                end_line: 10,
                text: format!("fn test_{}() {{}}", i),
                hash: format!("hash_{}", i),
                modified_at: Utc::now(),
            })
            .collect();

        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..vector_dim)
                    .map(|j| ((i + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        (chunks, vectors)
    }

    #[test]
    #[serial]
    fn save_and_load_hnsw_roundtrip() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_roundtrip");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD + 100; // Above threshold

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // HNSW file should exist
        let hnsw_path = store.root.join("hnsw.usearch");
        let header_path = store.root.join("hnsw_header.bin");
        assert!(hnsw_path.exists(), "HNSW file should be created");
        assert!(header_path.exists(), "HNSW header file should be created");

        // Load via mmap and verify HNSW is loaded
        let loaded = store.load_mmap().unwrap().unwrap();
        assert!(loaded.has_hnsw(), "MmapIndex should have loaded HNSW");

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn small_index_does_not_persist_hnsw() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_small");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD - 100; // Below threshold

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // HNSW files should NOT exist for small indexes
        let hnsw_path = store.root.join("hnsw.usearch");
        let header_path = store.root.join("hnsw_header.bin");
        assert!(
            !hnsw_path.exists(),
            "HNSW file should not be created for small indexes"
        );
        assert!(
            !header_path.exists(),
            "HNSW header should not be created for small indexes"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn hnsw_fallback_on_version_mismatch() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_version");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD + 100;

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // Corrupt the header with a wrong version
        let header_path = store.root.join("hnsw_header.bin");
        let mut header_bytes = std::fs::read(&header_path).unwrap();
        // Set format_version to 999 (first 4 bytes)
        header_bytes[0..4].copy_from_slice(&999u32.to_le_bytes());
        std::fs::write(&header_path, header_bytes).unwrap();

        // Load should succeed but HNSW should be None (fallback)
        let loaded = store.load_mmap().unwrap().unwrap();
        assert!(
            !loaded.has_hnsw(),
            "HNSW should not load with version mismatch"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn hnsw_fallback_on_dimension_mismatch() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_dim");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD + 100;

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // Corrupt the header with a wrong dimension
        let header_path = store.root.join("hnsw_header.bin");
        let mut header_bytes = std::fs::read(&header_path).unwrap();
        // Set vector_dim to 768 (bytes 4-7)
        header_bytes[4..8].copy_from_slice(&768u32.to_le_bytes());
        std::fs::write(&header_path, header_bytes).unwrap();

        // Load should succeed but HNSW should be None (fallback)
        let loaded = store.load_mmap().unwrap().unwrap();
        assert!(
            !loaded.has_hnsw(),
            "HNSW should not load with dimension mismatch"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn hnsw_fallback_on_count_mismatch() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_count");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD + 100;

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // Corrupt the header with a wrong count
        let header_path = store.root.join("hnsw_header.bin");
        let mut header_bytes = std::fs::read(&header_path).unwrap();
        // Set num_vectors to a different value (bytes 16-19)
        header_bytes[16..20].copy_from_slice(&(num_vectors as u32 + 50).to_le_bytes());
        std::fs::write(&header_path, header_bytes).unwrap();

        // Load should succeed but HNSW should be None (fallback)
        let loaded = store.load_mmap().unwrap().unwrap();
        assert!(
            !loaded.has_hnsw(),
            "HNSW should not load with count mismatch"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn search_works_without_hnsw_file() {
        use crate::search::config::HNSW_THRESHOLD;

        let _home = set_test_home();
        let temp_dir = temp_dir_with_name("hnsw_missing");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let store = IndexStore::new(&temp_dir).unwrap();
        let vector_dim = 384;
        let num_vectors = HNSW_THRESHOLD + 100;

        let (chunks, vectors) = make_chunks_with_vectors(num_vectors, vector_dim);
        let metadata = make_metadata(temp_dir.clone(), store.repo_hash().to_string(), vector_dim);
        let original = RepositoryIndex::new(metadata, chunks, vectors);

        store.save(&original).unwrap();

        // Delete HNSW files to simulate old index
        let hnsw_path = store.root.join("hnsw.usearch");
        let header_path = store.root.join("hnsw_header.bin");
        std::fs::remove_file(&hnsw_path).ok();
        std::fs::remove_file(&header_path).ok();

        // Load should succeed but without HNSW
        let loaded = store.load_mmap().unwrap().unwrap();
        assert!(
            !loaded.has_hnsw(),
            "MmapIndex should not have HNSW when files are missing"
        );
        assert_eq!(
            loaded.chunks.len(),
            num_vectors,
            "Chunks should still be loaded"
        );

        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
