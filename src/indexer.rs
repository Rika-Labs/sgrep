use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use blake3::Hasher as Blake3;
use ignore::WalkBuilder;

use crate::chunker::{FileType, chunk_file};
use crate::embedding::Embedder;
use crate::fts;
use crate::store::{
    ChunkRecord, FileMetadata, load_index, load_metadata, save_index, save_metadata,
};
const MAX_CHUNKS_PER_FILE: usize = 256;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

pub struct IndexOptions {
    pub root: PathBuf,
    pub force_reindex: bool,
    pub dry_run: bool,
    pub include_markdown: bool,
}

#[derive(Debug, Default)]
pub struct IndexStats {
    pub chunks: usize,
    pub files: usize,
    pub skipped_existing: bool,
    pub reused_files: usize,
}

pub fn index_repository(embedder: &Embedder, options: IndexOptions) -> Result<IndexStats> {
    let path = if options.root.as_os_str().is_empty() {
        std::env::current_dir().context("failed to read current directory")?
    } else {
        options.root.clone()
    };
    let existing_chunks = load_index(&path).unwrap_or_default();
    let existing_meta = load_metadata(&path).unwrap_or_default();
    let mut chunks: Vec<ChunkRecord> = Vec::new();
    let mut stats = IndexStats::default();
    let mut seen_chunk_ids: HashSet<u64> = HashSet::new();
    let mut meta_map: HashMap<String, FileMetadata> = HashMap::new();
    for m in existing_meta.into_iter() {
        meta_map.insert(m.path.clone(), m);
    }
    let mut reused = 0usize;
    let mut hash_by_path: HashMap<String, String> = HashMap::new();
    let walker = WalkBuilder::new(&path)
        .hidden(false)
        .ignore(true)
        .git_ignore(true)
        .git_exclude(true)
        .build();
    for entry in walker {
        let entry = entry?;
        let file_path = entry.path();
        if !should_index_path(file_path) {
            continue;
        }
        if !options.include_markdown {
            if let Some(ext) = file_path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("md") || ext.eq_ignore_ascii_case("mdx") {
                    continue;
                }
            }
        }
        if skip_large_file(file_path) {
            continue;
        }
        let content_hash = hash_file(file_path)?;
        if !options.force_reindex {
            if let Some(meta) = meta_map.get(file_path.to_string_lossy().as_ref()) {
                if meta.content_hash == content_hash {
                    for chunk in existing_chunks
                        .iter()
                        .filter(|c| c.path == file_path.to_string_lossy())
                    {
                        chunks.push(chunk.clone());
                        seen_chunk_ids.insert(chunk.id);
                    }
                    stats.files += 1;
                    reused += 1;
                    hash_by_path.insert(file_path.to_string_lossy().to_string(), content_hash);
                    continue;
                }
            }
        }
        index_file(
            file_path,
            &mut chunks,
            embedder,
            &mut seen_chunk_ids,
            &mut stats,
        )?;
        hash_by_path.insert(file_path.to_string_lossy().to_string(), content_hash);
    }
    stats.reused_files = reused;
    if options.dry_run {
        return Ok(stats);
    }
    save_index(&path, &chunks)?;
    let metas = build_metadata(&chunks, &hash_by_path);
    save_metadata(&path, &metas)?;
    fts::index_chunks(&path, &chunks)?;
    stats.chunks = chunks.len();
    Ok(stats)
}

fn should_index_path(path: &Path) -> bool {
    let ext = match path.extension().and_then(|s| s.to_str()) {
        Some(v) => v.to_ascii_lowercase(),
        None => String::new(),
    };
    if ext.is_empty() {
        return false;
    }
    let allowed = [
        "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "json", "yaml", "yml", "toml", "md",
    ];
    for value in allowed.iter() {
        if *value == ext {
            return true;
        }
    }
    false
}

fn skip_large_file(path: &Path) -> bool {
    match path.metadata() {
        Ok(meta) => meta.len() > MAX_FILE_BYTES,
        Err(_) => false,
    }
}

fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = Blake3::new();
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}

fn build_metadata(chunks: &[ChunkRecord], hashes: &HashMap<String, String>) -> Vec<FileMetadata> {
    let mut map: HashMap<String, FileMetadata> = HashMap::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    for chunk in chunks {
        let entry = map.entry(chunk.path.clone()).or_insert(FileMetadata {
            path: chunk.path.clone(),
            content_hash: hashes.get(&chunk.path).cloned().unwrap_or_default(),
            indexed_at: now,
            chunk_ids: Vec::new(),
        });
        entry.chunk_ids.push(chunk.id);
    }
    map.into_values().collect()
}

fn index_file(
    path: &Path,
    out: &mut Vec<ChunkRecord>,
    embedder: &Embedder,
    seen_chunk_ids: &mut HashSet<u64>,
    stats: &mut IndexStats,
) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let chunked = chunk_file(path, &contents)?;
    let mut emitted = 0usize;
    for chunk in chunked.chunks.into_iter() {
        if emitted >= MAX_CHUNKS_PER_FILE {
            break;
        }
        let slice = &contents[chunk.start_byte..chunk.end_byte];
        emit_chunk(
            path,
            chunk.start_line,
            chunk.end_line,
            slice,
            chunked.file_type,
            out,
            embedder,
            seen_chunk_ids,
        )?;
        emitted += 1;
    }
    stats.files += 1;
    Ok(())
}

fn emit_chunk(
    path: &Path,
    start_line: u32,
    end_line: u32,
    text: &str,
    file_type: FileType,
    out: &mut Vec<ChunkRecord>,
    embedder: &Embedder,
    seen_chunk_ids: &mut HashSet<u64>,
) -> Result<()> {
    if text.trim().is_empty() {
        return Ok(());
    }
    let embedding = embedder.embed(text)?;
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    start_line.hash(&mut hasher);
    end_line.hash(&mut hasher);
    text.hash(&mut hasher);
    let id = hasher.finish();
    if !seen_chunk_ids.insert(id) {
        return Ok(());
    }
    let record = ChunkRecord {
        id,
        path: path.to_string_lossy().to_string(),
        start_line,
        end_line,
        text: text.to_string(),
        embedding,
        file_type: match file_type {
            FileType::Code => "code".to_string(),
            FileType::Documentation => "doc".to_string(),
            FileType::Config => "config".to_string(),
            FileType::Data => "data".to_string(),
            FileType::Other => "other".to_string(),
        },
    };
    out.push(record);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn should_index_known_extension() {
        let path = PathBuf::from("./example.rs");
        assert!(should_index_path(&path));
    }

    #[test]
    fn should_not_index_unknown_extension() {
        let path = PathBuf::from("image.png");
        assert!(!should_index_path(&path));
    }
}
