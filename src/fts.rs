use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::schema::document::TantivyDocument;
use tantivy::schema::{STORED, Schema, TEXT, Value};
use tantivy::{Index, doc};

use crate::store::{ChunkRecord, resolve_paths};

fn fts_dir_for_root(root: &Path) -> Result<PathBuf> {
    let paths = resolve_paths(root)?;
    Ok(paths.fts_path)
}

fn schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_i64_field("id", STORED);
    builder.add_text_field("path", STORED);
    builder.add_text_field("body", TEXT | STORED);
    builder.build()
}

pub fn index_chunks(root: &Path, chunks: &[ChunkRecord]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let dir_path = fts_dir_for_root(root)?;
    if dir_path.exists() {
        std::fs::remove_dir_all(&dir_path)
            .with_context(|| format!("failed to clear {}", dir_path.display()))?;
    }
    std::fs::create_dir_all(&dir_path)
        .with_context(|| format!("failed to create {}", dir_path.display()))?;
    let schema = schema();
    let directory = MmapDirectory::open(&dir_path)
        .with_context(|| format!("failed to open {}", dir_path.display()))?;
    let settings = tantivy::IndexSettings::default();
    let index =
        Index::create(directory, schema.clone(), settings).context("failed to create index")?;
    let mut writer = index
        .writer(32 * 1024 * 1024)
        .context("failed to create index writer")?;
    let id_field = schema.get_field("id").unwrap();
    let path_field = schema.get_field("path").unwrap();
    let body_field = schema.get_field("body").unwrap();
    for chunk in chunks.iter() {
        let document = doc!(
            id_field => chunk.id as i64,
            path_field => chunk.path.clone(),
            body_field => chunk.text.clone(),
        );
        writer
            .add_document(document)
            .context("failed to add document")?;
    }
    writer.commit().context("failed to commit index")?;
    Ok(())
}

pub fn keyword_scores(root: &Path, query: &str, limit: usize) -> Result<HashMap<u64, f32>> {
    if query.trim().is_empty() {
        return Ok(HashMap::new());
    }
    let dir_path = fts_dir_for_root(root)?;
    if !dir_path.exists() {
        return Ok(HashMap::new());
    }
    let schema = schema();
    let directory = MmapDirectory::open(&dir_path)
        .with_context(|| format!("failed to open {}", dir_path.display()))?;
    let index = Index::open(directory).context("failed to open index")?;
    let reader = index.reader().context("failed to create reader")?;
    let searcher = reader.searcher();
    let body_field = schema.get_field("body").unwrap();
    let id_field = schema.get_field("id").unwrap();
    let parser = tantivy::query::QueryParser::for_index(&index, vec![body_field]);
    let query = parser.parse_query(query).context("failed to parse query")?;
    let top_docs = searcher
        .search(&query, &TopDocs::with_limit(limit))
        .context("failed to run search")?;
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for (score, address) in top_docs.into_iter() {
        let doc = searcher
            .doc::<TantivyDocument>(address)
            .context("failed to load document")?;
        let mut id_value: Option<u64> = None;
        for field_value in doc.get_all(id_field) {
            if let Some(v) = field_value.as_i64() {
                if v >= 0 {
                    id_value = Some(v as u64);
                }
            }
        }
        let id = match id_value {
            Some(v) => v,
            None => continue,
        };
        let existing = scores.get(&id).copied().unwrap_or(0.0);
        if score > existing {
            scores.insert(id, score as f32);
        }
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_gives_no_scores() {
        let root = Path::new(".");
        let scores = keyword_scores(root, "", 10).unwrap();
        assert!(scores.is_empty());
    }
}
