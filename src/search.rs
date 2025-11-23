use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;

use crate::chunker::CodeChunk;
use crate::embedding::Embedder;
use crate::fts;
use crate::store::RepositoryIndex;

pub struct SearchOptions {
    pub limit: usize,
    pub include_context: bool,
    pub glob: Vec<String>,
    pub filters: Vec<String>,
}

pub struct SearchEngine {
    embedder: Arc<Embedder>,
}

impl SearchEngine {
    pub fn new(embedder: Arc<Embedder>) -> Self {
        Self { embedder }
    }

    pub fn search(
        &self,
        index: &RepositoryIndex,
        query: &str,
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let keywords = fts::extract_keywords(query);
        let query_vec = self.embedder.embed(query)?;
        let limit = options.limit.max(1);

        let globset = fts::build_globset(&options.glob);
        let mut matches: Vec<SearchResult> = index
            .chunks
            .iter()
            .zip(&index.vectors)
            .filter(|(chunk, _)| fts::glob_matches(globset.as_ref(), &chunk.path))
            .filter(|(chunk, _)| fts::matches_filters(&options.filters, chunk))
            .map(|(chunk, vector)| {
                let semantic = cosine_similarity(&query_vec, vector);
                let keyword = fts::keyword_score(&keywords, &chunk.text);
                let recency = recency_boost(chunk);
                let final_score = 0.6 * semantic + 0.3 * keyword + 0.1 * recency;
                SearchResult {
                    chunk: chunk.clone(),
                    score: final_score,
                    semantic_score: semantic,
                    keyword_score: keyword,
                    show_full_context: options.include_context,
                }
            })
            .collect();

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        Ok(matches)
    }
}

#[derive(Clone)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f32,
    pub semantic_score: f32,
    pub keyword_score: f32,
    pub show_full_context: bool,
}

impl SearchResult {
    pub fn render_snippet(&self) -> String {
        if self.show_full_context {
            self.chunk.text.clone()
        } else {
            self.chunk
                .text
                .lines()
                .take(12)
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dot = lhs.iter().zip(rhs).map(|(l, r)| l * r).sum::<f32>();
    let lhs_norm = lhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f32>().sqrt();
    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        0.0
    } else {
        dot / (lhs_norm * rhs_norm)
    }
}

fn recency_boost(chunk: &CodeChunk) -> f32 {
    let age_hours = (Utc::now() - chunk.modified_at).num_hours().max(0) as f32;
    1.0 / (1.0 + age_hours / 48.0)
}
