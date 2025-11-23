use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;

use crate::embedding::Embedder;
use crate::fts;
use crate::store::{ChunkRecord, index_exists, save_index};

const CHUNK_MAX_LINES: usize = 40;
const CHUNK_OVERLAP_LINES: usize = 8;
const MAX_CHUNKS_PER_FILE: usize = 256;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

pub struct IndexOptions {
    pub root: PathBuf,
    pub force_reindex: bool,
    pub dry_run: bool,
}

#[derive(Debug, Default)]
pub struct IndexStats {
    pub chunks: usize,
    pub files: usize,
    pub skipped_existing: bool,
}

pub fn index_repository(embedder: &Embedder, options: IndexOptions) -> Result<IndexStats> {
    let path = if options.root.as_os_str().is_empty() {
        std::env::current_dir().context("failed to read current directory")?
    } else {
        options.root.clone()
    };
    if !options.force_reindex && index_exists(&path)? {
        return Ok(IndexStats {
            skipped_existing: true,
            ..Default::default()
        });
    }
    let mut chunks: Vec<ChunkRecord> = Vec::new();
    let mut stats = IndexStats::default();
    let mut seen_chunk_ids: HashSet<u64> = HashSet::new();
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
        if skip_large_file(file_path) {
            continue;
        }
        index_file(file_path, &mut chunks, embedder, &mut seen_chunk_ids, &mut stats)?;
    }
    if options.dry_run {
        return Ok(stats);
    }
    save_index(&path, &chunks)?;
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

fn index_file(
    path: &Path,
    out: &mut Vec<ChunkRecord>,
    embedder: &Embedder,
    seen_chunk_ids: &mut HashSet<u64>,
    stats: &mut IndexStats,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut buffer: Vec<String> = Vec::new();
    let mut start_line = 1u32;
    let mut current_line = 1u32;
    let mut chunks_emitted = 0usize;
    for line in reader.lines() {
        let line = line?;
        if buffer.is_empty() {
            start_line = current_line;
        }
        buffer.push(line);
        if buffer.len() >= CHUNK_MAX_LINES {
            emit_chunk(
                path,
                start_line,
                current_line,
                &mut buffer,
                out,
                embedder,
                seen_chunk_ids,
            )?;
            chunks_emitted += 1;
            if CHUNK_OVERLAP_LINES > 0 {
                let overlap = buffer
                    .split_off(buffer.len().saturating_sub(CHUNK_OVERLAP_LINES));
                buffer = overlap;
                start_line = current_line.saturating_sub(buffer.len() as u32 - 1);
            } else {
                buffer.clear();
            }
            if chunks_emitted >= MAX_CHUNKS_PER_FILE {
                stats.files += 1;
                return Ok(());
            }
        }
        current_line += 1;
    }
    if buffer.is_empty() {
        return Ok(());
    }
    emit_chunk(
        path,
        start_line,
        current_line.saturating_sub(1),
        &mut buffer,
        out,
        embedder,
        seen_chunk_ids,
    )?;
    stats.files += 1;
    Ok(())
}

fn emit_chunk(
    path: &Path,
    start_line: u32,
    end_line: u32,
    buffer: &mut Vec<String>,
    out: &mut Vec<ChunkRecord>,
    embedder: &Embedder,
    seen_chunk_ids: &mut HashSet<u64>,
) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    let text = buffer.join("\n");
    if text.trim().is_empty() {
        buffer.clear();
        return Ok(());
    }
    buffer.clear();
    let embedding = embedder.embed(&text)?;
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
        text,
        embedding,
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
