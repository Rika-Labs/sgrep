use std::io::{self, Write};

use anyhow::{anyhow, Result};
use base64::Engine;
use zstd::stream::{read::Decoder, Encoder};

use crate::graph::CodeGraph;
use crate::remote::{RemoteChunk, RemoteVectorStore};

const GRAPH_BLOB_ID: &str = "__sgrep_graph__";
const MARKER_VALUE: f32 = 1_000_000.0;

fn compress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = Encoder::new(Vec::new(), 3)?;
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = Decoder::new(data)?;
    let mut out = Vec::new();
    io::copy(&mut decoder, &mut out)?;
    Ok(out)
}

fn encode_base64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn decode_base64(data: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| anyhow!("base64 decode: {e}"))
}

fn marker_vector(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim.max(1)];
    v[0] = MARKER_VALUE;
    v
}

pub fn push_graph_blob(
    remote: &dyn RemoteVectorStore,
    graph: &CodeGraph,
    vector_dim: usize,
) -> Result<()> {
    let serialized = bincode::serialize(graph)?;
    let compressed = compress_bytes(&serialized)?;
    let encoded = encode_base64(&compressed);

    let chunk = RemoteChunk {
        id: GRAPH_BLOB_ID.to_string(),
        vector: marker_vector(vector_dim),
        path: "graph".to_string(),
        start_line: 0,
        end_line: 0,
        content: encoded,
        language: "graph".to_string(),
        symbols: Vec::new(),
    };

    remote.upsert(&[chunk])
}

pub fn fetch_graph_blob(
    remote: &dyn RemoteVectorStore,
    vector_dim: usize,
) -> Result<Option<CodeGraph>> {
    let hits = remote.query(&marker_vector(vector_dim), 1)?;

    let hit = match hits.iter().find(|h| h.id == GRAPH_BLOB_ID) {
        Some(h) => h,
        None => return Ok(None),
    };

    let compressed = decode_base64(&hit.content)?;
    let decompressed = decompress_bytes(&compressed)?;
    let graph: CodeGraph =
        bincode::deserialize(&decompressed).map_err(|e| anyhow!("graph deserialize: {e}"))?;

    Ok(Some(graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeKind, Symbol, SymbolKind};
    use crate::remote::RemoteSearchHit;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use uuid::Uuid;

    struct MockStore {
        chunks: Mutex<Vec<RemoteChunk>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                chunks: Mutex::new(Vec::new()),
            }
        }
    }

    impl RemoteVectorStore for MockStore {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn upsert(&self, chunks: &[RemoteChunk]) -> Result<()> {
            self.chunks.lock().unwrap().extend(chunks.iter().cloned());
            Ok(())
        }

        fn query(&self, _vector: &[f32], _top_k: usize) -> Result<Vec<RemoteSearchHit>> {
            let chunks = self.chunks.lock().unwrap();
            Ok(chunks
                .iter()
                .map(|c| RemoteSearchHit {
                    id: c.id.clone(),
                    score: 1.0,
                    path: c.path.clone(),
                    start_line: c.start_line,
                    end_line: c.end_line,
                    content: c.content.clone(),
                    language: c.language.clone(),
                    symbols: c.symbols.clone(),
                })
                .collect())
        }

        fn delete_namespace(&self) -> Result<()> {
            self.chunks.lock().unwrap().clear();
            Ok(())
        }
    }

    fn sample_graph() -> CodeGraph {
        let mut graph = CodeGraph::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        graph.add_symbol(Symbol {
            id: id1,
            name: "func_a".to_string(),
            qualified_name: "mod::func_a".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn func_a()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_symbol(Symbol {
            id: id2,
            name: "func_b".to_string(),
            qualified_name: "mod::func_b".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("src/lib.rs"),
            start_line: 15,
            end_line: 25,
            language: "rust".to_string(),
            signature: "fn func_b()".to_string(),
            parent_id: None,
            chunk_id: None,
        });

        graph.add_edge(Edge {
            source_id: id1,
            target_id: id2,
            kind: EdgeKind::Calls,
            metadata: None,
        });

        graph
    }

    #[test]
    fn marker_vector_has_correct_dimension() {
        let v = marker_vector(384);
        assert_eq!(v.len(), 384);
        assert_eq!(v[0], MARKER_VALUE);
        assert!(v[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn marker_vector_handles_zero_dim() {
        let v = marker_vector(0);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], MARKER_VALUE);
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let data = b"hello world this is a test of compression";
        let compressed = compress_bytes(data).unwrap();
        let decompressed = decompress_bytes(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"binary data \x00\x01\x02";
        let encoded = encode_base64(data);
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn push_and_fetch_graph_roundtrip() {
        let store = MockStore::new();
        let graph = sample_graph();
        let dim = 384;

        push_graph_blob(&store, &graph, dim).unwrap();

        let fetched = fetch_graph_blob(&store, dim).unwrap();
        assert!(fetched.is_some());

        let fetched = fetched.unwrap();
        assert_eq!(fetched.symbols.len(), graph.symbols.len());
        assert_eq!(fetched.edges.len(), graph.edges.len());
    }

    #[test]
    fn fetch_returns_none_when_empty() {
        let store = MockStore::new();
        let fetched = fetch_graph_blob(&store, 384).unwrap();
        assert!(fetched.is_none());
    }

    #[test]
    fn push_uses_correct_id() {
        let store = MockStore::new();
        let graph = CodeGraph::new();

        push_graph_blob(&store, &graph, 384).unwrap();

        let chunks = store.chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].id, GRAPH_BLOB_ID);
    }

    #[test]
    fn push_uses_marker_vector() {
        let store = MockStore::new();
        let graph = CodeGraph::new();
        let dim = 128;

        push_graph_blob(&store, &graph, dim).unwrap();

        let chunks = store.chunks.lock().unwrap();
        assert_eq!(chunks[0].vector.len(), dim);
        assert_eq!(chunks[0].vector[0], MARKER_VALUE);
    }

    #[test]
    fn graph_with_symbols_and_edges_survives_roundtrip() {
        let store = MockStore::new();
        let original = sample_graph();
        let dim = 384;

        push_graph_blob(&store, &original, dim).unwrap();
        let restored = fetch_graph_blob(&store, dim).unwrap().unwrap();

        assert_eq!(restored.symbols.len(), 2);
        assert_eq!(restored.edges.len(), 1);

        let names: Vec<_> = restored.find_by_name("func_a");
        assert_eq!(names.len(), 1);
        assert_eq!(names[0].name, "func_a");

        let names: Vec<_> = restored.find_by_name("func_b");
        assert_eq!(names.len(), 1);
    }
}
