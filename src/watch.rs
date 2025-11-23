use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use notify::{Event, RecursiveMode, Watcher};

use crate::embedding::Embedder;
use crate::indexer::{IndexOptions, index_repository};
use crate::store::stores_dir;

const WATCH_FILE_NAME: &str = "watch.jsonl";
const DEFAULT_DEBOUNCE_MS: u64 = 750;

fn watch_state_path() -> Result<PathBuf> {
    let stores = stores_dir()?;
    let base = stores
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(stores.clone());
    Ok(base.join(WATCH_FILE_NAME))
}

fn canonicalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub fn list() -> Result<Vec<PathBuf>> {
    let path = watch_state_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(&path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut paths = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        paths.push(PathBuf::from(line));
    }
    Ok(paths)
}

fn persist(paths: &[PathBuf]) -> Result<()> {
    let state_path = watch_state_path()?;
    if let Some(parent) = state_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let tmp = state_path.with_extension("tmp");
    let file = File::create(&tmp).with_context(|| format!("failed to create {}", tmp.display()))?;
    let mut writer = BufWriter::new(file);
    for path in paths {
        writer.write_all(path.to_string_lossy().as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    std::fs::rename(&tmp, &state_path).with_context(|| {
        format!(
            "failed to move {} to {}",
            tmp.display(),
            state_path.display()
        )
    })?;
    Ok(())
}

pub fn add(path: &Path) -> Result<bool> {
    let mut paths = list()?;
    let new_path = canonicalize(path);
    if paths.iter().any(|p| p == &new_path) {
        return Ok(false);
    }
    paths.push(new_path);
    persist(&paths)?;
    Ok(true)
}

pub fn remove(path: &Path) -> Result<bool> {
    let target = canonicalize(path);
    let mut paths = list()?;
    let before = paths.len();
    paths.retain(|p| p != &target);
    if paths.len() == before {
        return Ok(false);
    }
    persist(&paths)?;
    Ok(true)
}

pub fn clear() -> Result<()> {
    persist(&[])?;
    Ok(())
}

fn initial_paths(user_path: Option<PathBuf>) -> Result<Vec<PathBuf>> {
    if let Some(p) = user_path {
        return Ok(vec![canonicalize(&p)]);
    }
    let stored = list()?;
    if !stored.is_empty() {
        return Ok(stored.into_iter().map(|p| canonicalize(&p)).collect());
    }
    let current = std::env::current_dir().context("failed to read current directory")?;
    Ok(vec![canonicalize(&current)])
}

fn ensure_indexes(embedder: &Embedder, paths: &[PathBuf]) -> Result<()> {
    for root in paths {
        let opts = IndexOptions {
            root: root.clone(),
            force_reindex: true,
            dry_run: false,
            include_markdown: true,
        };
        let _ = index_repository(embedder, opts)?;
    }
    Ok(())
}

pub fn run_watch(
    embedder: &Embedder,
    user_path: Option<PathBuf>,
    debounce_ms: Option<u64>,
) -> Result<()> {
    let paths = initial_paths(user_path)?;
    persist(&paths)?;
    ensure_indexes(embedder, &paths)?;

    let debounce = Duration::from_millis(debounce_ms.unwrap_or(DEFAULT_DEBOUNCE_MS));
    let (tx, rx) = channel::<notify::Result<Event>>();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    for path in &paths {
        watcher.watch(path, RecursiveMode::Recursive)?;
    }

    println!(
        "Watching {} path(s) for changes. Press Ctrl+C to stop.",
        paths.len()
    );

    let mut pending = false;
    let mut last = Instant::now();
    let mut seen_roots: HashSet<PathBuf> = HashSet::new();

    for path in &paths {
        seen_roots.insert(path.clone());
    }

    loop {
        match rx.recv_timeout(debounce) {
            Ok(Ok(event)) => {
                use notify::event::EventKind;
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_) => {
                        pending = true;
                        last = Instant::now();
                    }
                    _ => {}
                }
            }
            Ok(Err(err)) => {
                eprintln!("watch error: {}", err);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if pending && last.elapsed() >= debounce {
                    for root in seen_roots.iter() {
                        let opts = IndexOptions {
                            root: root.clone(),
                            force_reindex: true,
                            dry_run: false,
                            include_markdown: true,
                        };
                        if let Err(err) = index_repository(embedder, opts) {
                            eprintln!("reindex failed for {}: {}", root.display(), err);
                        } else {
                            println!("re-indexed {}", root.display());
                        }
                    }
                    pending = false;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("watch channel disconnected"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    #[test]
    #[serial]
    fn persist_add_remove_clear_round_trip() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("proj");
        std::fs::create_dir_all(&root).unwrap();
        let data_root = dir.path().join("data");
        unsafe {
            std::env::set_var("SGREP_DATA_DIR", &data_root);
        }
        assert!(list().unwrap().is_empty());

        assert!(add(&root).unwrap());
        let listed = list().unwrap();
        assert_eq!(listed.len(), 1);

        assert!(!add(&root).unwrap());

        assert!(remove(&root).unwrap());
        assert!(list().unwrap().is_empty());

        add(&root).unwrap();
        clear().unwrap();
        assert!(list().unwrap().is_empty());

        unsafe {
            std::env::remove_var("SGREP_DATA_DIR");
        }
    }
}
