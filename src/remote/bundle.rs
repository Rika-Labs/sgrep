use std::io::{self, Write};
use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use blake3::Hasher;
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zstd::stream::{read::Decoder, Encoder};

use crate::graph::CodeGraph;
use crate::remote::{RemoteChunk, RemoteVectorStore};
use crate::store::{HierarchicalIndex, IndexStore, RepositoryIndex};

const MANIFEST_ID: &str = "__bundle_manifest";
const MANIFEST_SHARD_ID_PREFIX: &str = "__bundle_manifest_shard_";
const PART_PREFIX: &str = "__bundle_part_";
const DEFAULT_PART_SIZE: usize = 24 * 1024; // before base64 expansion
const MAX_METADATA_BYTES: usize = 39_000; // Pinecone metadata limit ~40KB
const MARKER_VALUE: f32 = 1.0; // base marker magnitude
const BUNDLE_BATCH_BYTES: usize = 1_500_000; // keep well under Pinecone 2MB request cap
const TP_ATTR_LIMIT_BYTES: usize = 4_000; // Turbopuffer filterable attr hard limit
const TP_ATTR_SAFE_B64_BYTES: usize = 3_400; // safety buffer under truncation (3800)
const MAX_PARTS_PER_SHARD: usize = 9_000; // stay under top_k 10k
const MAX_SHARDS: usize = 256; // safety bound
const TOO_LARGE_PREFIX: &str = "bundle too large";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub repo_hash: String,
    pub vector_dim: usize,
    pub part_size: usize,
    pub parts: usize,
    pub shard_counts: Vec<usize>,
    pub total_bytes: usize,
    pub hash: String,
}

#[derive(Serialize, Deserialize)]
struct BundlePayload {
    index: RepositoryIndex,
    graph: Option<CodeGraph>,
    hierarchy: Option<HierarchicalIndex>,
}

fn compress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = Encoder::new(Vec::new(), 3)?;
    encoder.write_all(data)?;
    let compressed = encoder.finish()?;
    Ok(compressed)
}

fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = Decoder::new(data)?;
    let mut out = Vec::new();
    io::copy(&mut decoder, &mut out)?;
    Ok(out)
}

fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

fn encode_base64(data: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn decode_base64(data: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| anyhow!("Failed to decode bundle part: {e}"))
}

pub struct BundleParts {
    pub manifest: BundleManifest,
    pub manifest_bytes: Vec<u8>,
    pub parts: Vec<Vec<u8>>,       // flat parts
    pub shards: Vec<Range<usize>>, // ranges into parts
}

pub fn is_bundle_too_large(err: &anyhow::Error) -> bool {
    format!("{}", err).contains(TOO_LARGE_PREFIX)
}

fn manifest_vector(dim: usize) -> Vec<f32> {
    if dim == 0 {
        return vec![MARKER_VALUE * 1_000_000.0];
    }
    let mut v = vec![0.0; dim];
    v[0] = MARKER_VALUE * 1_000_000.0; // dominant magnitude
    v
}

fn marker_vector_shard(dim: usize, shard: usize) -> Vec<f32> {
    if dim == 0 {
        return vec![MARKER_VALUE * (shard as f32 + 1.0)];
    }
    let mut v = vec![0.0; dim];
    if dim > 1 {
        v[1] = shard as f32 + 1.0; // orthogonal to manifest first lane
    } else {
        v[0] = shard as f32 + 2.0; // fallback when dim == 1
    }
    v
}

pub fn build_bundle(path: &Path, part_size: Option<usize>) -> Result<BundleParts> {
    let store = IndexStore::new(path)?;
    let index = store
        .load()?
        .with_context(|| format!("No local index found for {}", path.display()))?;
    let graph = store.load_graph()?;
    let hierarchy = store.load_hierarchy()?;

    let payload = BundlePayload {
        index: index.clone(),
        graph,
        hierarchy,
    };

    let serialized = bincode::serialize(&payload)?;
    let compressed = compress_bytes(&serialized)?;
    let hash = hash_bytes(&compressed);

    let desired = part_size.unwrap_or(DEFAULT_PART_SIZE).max(8 * 1024);
    let pinecone_chunk = (MAX_METADATA_BYTES * 3) / 4;
    let turbopuffer_chunk = (TP_ATTR_SAFE_B64_BYTES * 3) / 4;
    let max_chunk = pinecone_chunk.min(turbopuffer_chunk);
    let min_chunk = 1024;

    let effective = desired.clamp(min_chunk, max_chunk);
    let parts: Vec<Vec<u8>> = compressed.chunks(effective).map(|c| c.to_vec()).collect();

    let mut shards: Vec<Range<usize>> = Vec::new();
    let mut start = 0;
    while start < parts.len() {
        let end = (start + MAX_PARTS_PER_SHARD).min(parts.len());
        shards.push(start..end);
        start = end;
    }

    if shards.len() > MAX_SHARDS {
        return Err(anyhow!(
            "{}: {} parts require {} shards (max {} shards). Try increasing part_size or reducing index size.",
            TOO_LARGE_PREFIX,
            parts.len(),
            shards.len(),
            MAX_SHARDS
        ));
    }

    let manifest = BundleManifest {
        repo_hash: index.metadata.repo_hash.clone(),
        vector_dim: index.metadata.vector_dim,
        part_size: effective,
        parts: parts.len(),
        shard_counts: shards.iter().map(|r| r.len()).collect(),
        total_bytes: compressed.len(),
        hash,
    };

    let manifest_bytes = bincode::serialize(&manifest)?;

    Ok(BundleParts {
        manifest,
        manifest_bytes,
        parts,
        shards,
    })
}

pub fn encode_parts_for_upload(parts: &BundleParts, vector_dim: usize) -> Vec<RemoteChunk> {
    let manifest_vec = manifest_vector(vector_dim);
    let mut chunks: Vec<RemoteChunk> = Vec::with_capacity(parts.parts.len() + 1);

    chunks.push(RemoteChunk {
        id: MANIFEST_ID.to_string(),
        vector: manifest_vec.clone(),
        path: "bundle".to_string(),
        start_line: parts.parts.len(),
        end_line: 0,
        content: encode_base64(&parts.manifest_bytes),
        language: "bundle".to_string(),
    });

    for (shard_idx, _range) in parts.shards.iter().enumerate() {
        let shard_vec = marker_vector_shard(vector_dim, shard_idx + 1);
        chunks.push(RemoteChunk {
            id: format!("{MANIFEST_SHARD_ID_PREFIX}{shard_idx}"),
            vector: shard_vec.clone(),
            path: "bundle".to_string(),
            start_line: parts.parts.len(),
            end_line: 0,
            content: encode_base64(&parts.manifest_bytes),
            language: "bundle".to_string(),
        });
    }

    for (shard_idx, range) in parts.shards.iter().enumerate() {
        let shard_vec = marker_vector_shard(vector_dim, shard_idx + 1);
        for (local, idx) in (range.start..range.end).enumerate() {
            let part = &parts.parts[idx];
            chunks.push(RemoteChunk {
                id: format!("{PART_PREFIX}{}_{local}", shard_idx),
                vector: shard_vec.clone(),
                path: "bundle".to_string(),
                start_line: local,
                end_line: range.len(),
                content: encode_base64(part),
                language: "bundle".to_string(),
            });
        }
    }

    chunks
}

pub fn upload_bundle(
    remote: &dyn RemoteVectorStore,
    parts: &BundleParts,
    vector_dim: usize,
) -> Result<()> {
    let chunks = encode_parts_for_upload(parts, vector_dim);
    let pb = ProgressBar::hidden();

    let mut batch: Vec<RemoteChunk> = Vec::new();
    let mut batch_bytes: usize = 0;

    for chunk in chunks.iter() {
        let chunk_bytes = chunk.content.len() + 512;
        if !batch.is_empty() && batch_bytes + chunk_bytes > BUNDLE_BATCH_BYTES {
            remote.upsert(&batch)?;
            pb.inc(batch.len() as u64);
            batch.clear();
            batch_bytes = 0;
        }
        batch.push(chunk.clone());
        batch_bytes += chunk_bytes;
    }

    if !batch.is_empty() {
        remote.upsert(&batch)?;
        pb.inc(batch.len() as u64);
    }

    pb.finish_and_clear();
    Ok(())
}

fn find_manifest(
    hits: &[crate::remote::RemoteSearchHit],
) -> Option<&crate::remote::RemoteSearchHit> {
    hits.iter().find(|h| h.id == MANIFEST_ID)
}

pub fn download_bundle(
    remote: &dyn RemoteVectorStore,
    vector_dim: usize,
) -> Result<(
    RepositoryIndex,
    Option<CodeGraph>,
    Option<HierarchicalIndex>,
)> {
    let manifest_vec = manifest_vector(vector_dim);
    let mut hits = remote.query(&manifest_vec, 1)?;

    let manifest_hit = if let Some(hit) = find_manifest(&hits) {
        hit
    } else {
        // fallback: shard 0 query
        let shard0 = marker_vector_shard(vector_dim, 1);
        hits = remote.query(&shard0, 4)?;
        if let Some(hit) = find_manifest(&hits) {
            hit
        } else if let Some(hit) = hits
            .iter()
            .find(|h| h.id.starts_with(MANIFEST_SHARD_ID_PREFIX))
        {
            hit
        } else {
            // last resort: query all shard markers up to MAX_SHARDS
            let mut found: Option<crate::remote::RemoteSearchHit> = None;
            for shard_idx in 1..=MAX_SHARDS {
                let vec = marker_vector_shard(vector_dim, shard_idx);
                let shard_hits = remote.query(&vec, 4)?;
                if let Some(hit) = shard_hits.iter().find(|h| {
                    h.id == MANIFEST_ID
                        || h.id == format!("{MANIFEST_SHARD_ID_PREFIX}{}", shard_idx)
                }) {
                    found = Some(hit.clone());
                    break;
                }
            }
            let hit = found.ok_or_else(|| {
                anyhow!("Remote bundle manifest not found; re-run sgrep index --remote")
            })?;
            hits.push(hit);
            hits.last().unwrap()
        }
    };

    let manifest_bytes = decode_base64(&manifest_hit.content)?;
    let manifest: BundleManifest = bincode::deserialize(&manifest_bytes)
        .map_err(|e| anyhow!("Failed to decode remote bundle manifest: {e}"))?;

    if manifest.shard_counts.len() > MAX_SHARDS {
        return Err(anyhow!(
            "{}: manifest has {} shards (max {})",
            TOO_LARGE_PREFIX,
            manifest.shard_counts.len(),
            MAX_SHARDS
        ));
    }

    let mut all_parts: Vec<Vec<u8>> = Vec::with_capacity(manifest.parts);

    for (shard_idx, count) in manifest.shard_counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        let shard_vec = marker_vector_shard(vector_dim, shard_idx + 1);
        let hits = remote.query(&shard_vec, count + 4)?;
        let mut shard_parts = decode_parts(&hits, shard_idx, *count)?;
        all_parts.append(&mut shard_parts);
    }

    if all_parts.len() != manifest.parts {
        return Err(anyhow!(
            "Remote bundle incomplete: expected {} parts, got {}",
            manifest.parts,
            all_parts.len()
        ));
    }

    let combined = all_parts.concat();

    if combined.len() != manifest.total_bytes {
        return Err(anyhow!(
            "Remote bundle size mismatch: expected {} bytes, got {}",
            manifest.total_bytes,
            combined.len()
        ));
    }

    let actual_hash = hash_bytes(&combined);
    if actual_hash != manifest.hash {
        return Err(anyhow!(
            "Remote bundle hash mismatch (expected {}, got {})",
            manifest.hash,
            actual_hash
        ));
    }

    let decompressed = decompress_bytes(&combined)?;
    let payload: BundlePayload = bincode::deserialize(&decompressed)
        .map_err(|e| anyhow!("Failed to decode remote bundle payload: {e}"))?;

    Ok((payload.index, payload.graph, payload.hierarchy))
}

fn decode_parts(
    hits: &[crate::remote::RemoteSearchHit],
    shard_idx: usize,
    expected: usize,
) -> Result<Vec<Vec<u8>>> {
    let prefix = format!("{PART_PREFIX}{shard_idx}_");
    let mut parts: Vec<(usize, Vec<u8>)> = hits
        .iter()
        .filter_map(|h| h.id.strip_prefix(&prefix).map(|idx| (idx, h)))
        .filter_map(|(idx_str, hit)| idx_str.parse::<usize>().ok().map(|i| (i, hit)))
        .map(|(i, hit)| Ok((i, decode_base64(&hit.content)?)))
        .collect::<Result<Vec<_>>>()?;

    if parts.len() != expected {
        return Err(anyhow!(
            "Remote shard {} incomplete: expected {} parts, found {}",
            shard_idx,
            expected,
            parts.len()
        ));
    }

    parts.par_sort_by_key(|(i, _)| *i);

    let mut ordered = Vec::with_capacity(expected);
    for (idx, part) in parts {
        if idx >= expected {
            return Err(anyhow!("Remote bundle part index out of range"));
        }
        ordered.push(part);
    }

    Ok(ordered)
}

pub fn delete_bundle_namespace(remote: &dyn RemoteVectorStore) -> Result<()> {
    remote.delete_namespace()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use crate::store::{IndexMetadata, RepositoryIndex};
    use chrono::Utc;
    use serial_test::serial;
    use std::env;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn mock_index(path: &Path, dim: usize) -> RepositoryIndex {
        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: path.join("file.rs"),
            language: "rust".into(),
            start_line: 1,
            end_line: 5,
            text: "fn main() {}".into(),
            hash: "hash".into(),
            modified_at: Utc::now(),
        };
        let meta = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").into(),
            repo_path: path.to_path_buf(),
            repo_hash: "hash".into(),
            vector_dim: dim,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        RepositoryIndex::new(meta, vec![chunk], vec![vec![0.1; dim]])
    }

    #[test]
    fn marker_vector_is_non_zero() {
        let v = manifest_vector(3);
        assert!(v[0] > 0.0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    #[serial]
    fn part_size_respects_provider_limits_and_shards() {
        let dir = tempdir().unwrap();
        env::set_var("SGREP_HOME", dir.path());
        let idx = mock_index(dir.path(), 4);
        let store = IndexStore::new(dir.path()).unwrap();
        store.save(&idx).unwrap();
        let parts = build_bundle(dir.path(), None).unwrap();
        let max_raw = (MAX_METADATA_BYTES.min(TP_ATTR_SAFE_B64_BYTES) * 3) / 4;
        assert!(parts.parts.iter().all(|p| p.len() <= max_raw));
        assert!(parts.shards.len() <= MAX_SHARDS);
        assert!(parts.shards.iter().all(|r| r.len() <= MAX_PARTS_PER_SHARD));
        assert_eq!(
            parts.shards.iter().map(|r| r.len()).sum::<usize>(),
            parts.parts.len()
        );
        env::remove_var("SGREP_HOME");
    }
}
