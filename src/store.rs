use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: u64,
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoreSummary {
    pub id: String,
    pub index_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct StorePaths {
    pub base_dir: PathBuf,
    pub index_path: PathBuf,
    pub fts_path: PathBuf,
}

const DATA_DIR_NAME: &str = ".sgrep";
const STORES_DIR_NAME: &str = "stores";
const INDEX_FILE_NAME: &str = "index.jsonl";
const FTS_DIR_NAME: &str = "fts";

fn data_root() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("SGREP_DATA_DIR") {
        return Ok(PathBuf::from(custom));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home).join(DATA_DIR_NAME));
    }
    let cwd = std::env::current_dir().context("failed to read current directory")?;
    Ok(cwd.join(DATA_DIR_NAME))
}

pub fn stores_dir() -> Result<PathBuf> {
    Ok(data_root()?.join(STORES_DIR_NAME))
}

fn store_id(root: &Path) -> String {
    let canonical = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let name = canonical
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("repo");
    let mut hasher = DefaultHasher::new();
    canonical.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();
    format!("{}-{:016x}", name, hash)
}

pub fn store_name_for_root(root: &Path) -> Result<String> {
    Ok(store_id(root))
}

pub fn resolve_paths(root: &Path) -> Result<StorePaths> {
    let base = stores_dir()?;
    let id = store_id(root);
    let base_dir = base.join(&id);
    let index_path = base_dir.join(INDEX_FILE_NAME);
    let fts_path = base_dir.join(FTS_DIR_NAME);
    Ok(StorePaths {
        base_dir,
        index_path,
        fts_path,
    })
}

pub fn index_exists(root: &Path) -> Result<bool> {
    let paths = resolve_paths(root)?;
    Ok(paths.index_path.exists())
}

pub fn save_index(root: &Path, chunks: &[ChunkRecord]) -> Result<()> {
    let paths = resolve_paths(root)?;
    let index_path = paths.index_path;
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp_path = index_path.with_extension("jsonl.tmp");
    let file = File::create(&tmp_path)
        .with_context(|| format!("failed to create {}", tmp_path.display()))?;
    let mut writer = BufWriter::new(file);
    for chunk in chunks {
        let line = serde_json::to_string(chunk)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    std::fs::rename(&tmp_path, &index_path).with_context(|| {
        format!(
            "failed to move {} to {}",
            tmp_path.display(),
            index_path.display()
        )
    })?;
    Ok(())
}

pub fn load_index(root: &Path) -> Result<Vec<ChunkRecord>> {
    let paths = resolve_paths(root)?;
    let index_path = paths.index_path;
    if !index_path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(&index_path)
        .with_context(|| format!("failed to open {}", index_path.display()))?;
    let reader = BufReader::new(file);
    let mut chunks = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: ChunkRecord = serde_json::from_str(&line)?;
        chunks.push(chunk);
    }
    Ok(chunks)
}

pub fn list_stores() -> Result<Vec<StoreSummary>> {
    let dir = stores_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut stores = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let id = match path.file_name().and_then(|s| s.to_str()) {
            Some(v) => v.to_string(),
            None => continue,
        };
        let index_path = path.join(INDEX_FILE_NAME);
        stores.push(StoreSummary { id, index_path });
    }
    Ok(stores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn store_id_is_stable_for_same_path() {
        let root = PathBuf::from("/tmp/example");
        let a = store_id(&root);
        let b = store_id(&root);
        assert_eq!(a, b);
    }

    #[test]
    #[serial]
    fn resolve_paths_builds_expected_layout() {
        let dir = tempfile::tempdir().unwrap();
        let data_root = dir.path().join("data");
        let original_data = std::env::var("SGREP_DATA_DIR").ok();
        unsafe { std::env::set_var("SGREP_DATA_DIR", &data_root); }

        let root = dir.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let paths = resolve_paths(&root).unwrap();
        assert!(paths.index_path.ends_with("index.jsonl"));
        assert!(paths.fts_path.ends_with("fts"));

        if let Some(data) = original_data {
            unsafe { std::env::set_var("SGREP_DATA_DIR", data); }
        } else {
            unsafe { std::env::remove_var("SGREP_DATA_DIR"); }
        }
    }
}
