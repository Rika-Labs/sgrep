use anyhow::Result;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub const HNSW_CONNECTIVITY: usize = 16;
pub const HNSW_EXPANSION_ADD: usize = 128;
pub const HNSW_EXPANSION_SEARCH: usize = 64;
pub const HNSW_OVERSAMPLE_FACTOR: usize = 4;

pub fn build_hnsw_index(dimensions: usize, capacity: usize) -> Result<Index> {
    let options = IndexOptions {
        dimensions,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: HNSW_CONNECTIVITY,
        expansion_add: HNSW_EXPANSION_ADD,
        expansion_search: HNSW_EXPANSION_SEARCH,
        multi: false,
    };

    let hnsw = Index::new(&options).map_err(|e| anyhow::anyhow!("HNSW creation failed: {}", e))?;
    hnsw.reserve(capacity)
        .map_err(|e| anyhow::anyhow!("HNSW reserve failed: {}", e))?;
    Ok(hnsw)
}

pub fn search_hnsw_candidates(
    hnsw: &Index,
    query_vec: &[f32],
    limit: usize,
    max_candidates: usize,
) -> Result<Vec<usize>> {
    let oversample = (limit * HNSW_OVERSAMPLE_FACTOR).min(max_candidates);
    let results = hnsw
        .search(query_vec, oversample)
        .map_err(|e| anyhow::anyhow!("HNSW search failed: {}", e))?;
    Ok(results.keys.iter().map(|&k| k as usize).collect())
}
