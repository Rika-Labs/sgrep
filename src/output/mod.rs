use std::time::Duration;

use serde::Serialize;

use crate::{search, store};

#[derive(Serialize)]
pub struct JsonResponse {
    pub query: String,
    pub limit: usize,
    pub duration_ms: u128,
    pub results: Vec<JsonMatch>,
    pub index: JsonIndexMeta,
}

#[derive(Serialize)]
pub struct JsonMatch {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub score: f32,
    pub semantic_score: f32,
    pub snippet: String,
}

#[derive(Serialize)]
pub struct JsonIndexMeta {
    pub repo_path: String,
    pub repo_hash: String,
    pub vector_dim: usize,
    pub indexed_at: String,
    pub total_files: usize,
    pub total_chunks: usize,
}

impl JsonResponse {
    pub fn from_results(
        query: &str,
        limit: usize,
        results: Vec<search::SearchResult>,
        index: &store::RepositoryIndex,
        duration: Duration,
    ) -> Self {
        let matches = results
            .into_iter()
            .map(|r| JsonMatch {
                path: r.chunk.path.to_string_lossy().to_string(),
                start_line: r.chunk.start_line,
                end_line: r.chunk.end_line,
                language: r.chunk.language.clone(),
                score: r.score,
                semantic_score: r.semantic_score,
                snippet: r.render_snippet(),
            })
            .collect();

        let index_meta = &index.metadata;
        Self {
            query: query.to_string(),
            limit,
            duration_ms: duration.as_millis(),
            results: matches,
            index: JsonIndexMeta {
                repo_path: index_meta.repo_path.to_string_lossy().to_string(),
                repo_hash: index_meta.repo_hash.clone(),
                vector_dim: index_meta.vector_dim,
                indexed_at: index_meta.indexed_at.to_rfc3339(),
                total_files: index_meta.total_files,
                total_chunks: index_meta.total_chunks,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::CodeChunk;
    use crate::store::{IndexMetadata, RepositoryIndex};
    use chrono::Utc;
    use std::path::Path;
    use uuid::Uuid;

    fn sample_index(root: &Path) -> RepositoryIndex {
        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: root.join("lib.rs"),
            language: "rust".into(),
            start_line: 1,
            end_line: 1,
            text: "pub fn hi() {}".into(),
            hash: "hash".into(),
            modified_at: Utc::now(),
        };
        let meta = IndexMetadata {
            version: env!("CARGO_PKG_VERSION").into(),
            repo_path: root.to_path_buf(),
            repo_hash: "hash".into(),
            vector_dim: 3,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        };
        RepositoryIndex::new(meta, vec![chunk], vec![vec![1.0, 2.0, 3.0]])
    }

    #[test]
    fn json_response_from_results_populates_fields() {
        let chunk = CodeChunk {
            id: Uuid::new_v4(),
            path: Path::new("lib.rs").to_path_buf(),
            language: "rust".into(),
            start_line: 1,
            end_line: 2,
            text: "pub fn hi() {}".into(),
            hash: "h".into(),
            modified_at: Utc::now(),
        };
        let index = sample_index(Path::new("/tmp/test"));
        let result = search::SearchResult {
            chunk,
            score: 0.5,
            semantic_score: 0.4,
            bm25_score: 0.0,
            show_full_context: false,
        };
        let json =
            JsonResponse::from_results("hi", 5, vec![result], &index, Duration::from_millis(10));
        assert_eq!(json.query, "hi");
        assert_eq!(json.results.len(), 1);
        assert_eq!(json.index.total_chunks, 1);
    }
}
