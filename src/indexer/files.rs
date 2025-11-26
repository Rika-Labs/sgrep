use std::path::{Path, PathBuf};
use std::sync::Arc;

use globset::{Glob, GlobSetBuilder};
use ignore::{WalkBuilder, WalkState};

const DEFAULT_IGNORE: &str = include_str!("../../default-ignore.txt");
pub const MAX_FILE_BYTES: u64 = 5 * 1024 * 1024;

const BINARY_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "svg", "ico", "webp", "avif", "psd", "tiff", "tif", "jxl",
    "heic", "heif", "jp2", "jpx", "pic", "icns", "cur", "raw", "arw", "cr2", "nef", "dng", "mp4",
    "mov", "avi", "mkv", "webm", "wmv", "flv", "m4v", "mp3", "wav", "flac", "ogg", "aac", "m4a",
    "wma", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "zip", "gz", "bz2", "7z", "rar",
    "tar", "xz", "zst", "exe", "dll", "so", "a", "dylib", "bin", "msi", "class", "wasm", "pyc",
    "pyo", "woff", "woff2", "ttf", "otf", "eot", "glb", "gltf", "obj", "fbx", "stl", "blend",
    "sqlite", "sqlite3", "db", "mdb",
];

pub fn collect_files(root: &Path) -> Vec<PathBuf> {
    use std::sync::Mutex;

    let default_excludes = build_default_excludes();
    let files = Arc::new(Mutex::new(Vec::new()));
    let root_arc = Arc::new(root.to_path_buf());
    let excludes_arc = Arc::new(default_excludes);

    WalkBuilder::new(root)
        .hidden(true)
        .ignore(true)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .parents(true)
        .follow_links(false)
        .add_custom_ignore_filename(".sgrepignore")
        .threads(crate::threading::ThreadConfig::get().walker_threads)
        .build_parallel()
        .run(|| {
            let files = Arc::clone(&files);
            let root = Arc::clone(&root_arc);
            let excludes = Arc::clone(&excludes_arc);

            Box::new(move |entry| {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => return WalkState::Continue,
                };

                if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                    return WalkState::Continue;
                }

                if is_probably_binary(entry.path()) {
                    return WalkState::Continue;
                }

                if let Ok(meta) = entry.metadata() {
                    if meta.len() > MAX_FILE_BYTES {
                        return WalkState::Continue;
                    }
                }

                let path = entry.path();
                let relative_path = path.strip_prefix(root.as_path()).unwrap_or(path);

                if excludes.is_match(relative_path) {
                    return WalkState::Continue;
                }

                files.lock().unwrap().push(entry.into_path());
                WalkState::Continue
            })
        });

    Arc::try_unwrap(files)
        .expect("All references should be dropped")
        .into_inner()
        .unwrap()
}

pub fn is_probably_binary(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("min") || BINARY_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

pub fn build_default_excludes() -> globset::GlobSet {
    let mut builder = GlobSetBuilder::new();

    for line in DEFAULT_IGNORE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Ok(glob) = Glob::new(line) {
            builder.add(glob);
        }

        let pattern_without_slash = line.trim_end_matches('/');
        if pattern_without_slash != line {
            if let Ok(glob) = Glob::new(&format!("**/{}", pattern_without_slash)) {
                builder.add(glob);
            }
            if let Ok(glob) = Glob::new(&format!("{}/**", pattern_without_slash)) {
                builder.add(glob);
            }
        }
    }

    builder
        .build()
        .unwrap_or_else(|_| GlobSetBuilder::new().build().unwrap())
}

pub fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub fn normalize_to_relative(path: &Path, root: &Path) -> PathBuf {
    if let Ok(stripped) = path.strip_prefix(root) {
        return stripped.to_path_buf();
    }

    if !path.is_absolute() {
        return path.to_path_buf();
    }

    if let Ok(canonicalized) = path.canonicalize() {
        if let Ok(stripped) = canonicalized.strip_prefix(root) {
            return stripped.to_path_buf();
        }
        return canonicalized;
    }

    if let (Some(parent), Some(file_name)) = (path.parent(), path.file_name()) {
        if let Ok(parent_canonical) = parent.canonicalize() {
            let recomposed = parent_canonical.join(file_name);
            if let Ok(stripped) = recomposed.strip_prefix(root) {
                return stripped.to_path_buf();
            }
            return recomposed;
        }
    }

    path.to_path_buf()
}

pub fn detect_language_for_graph(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "js" | "jsx" => Some("javascript"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" => Some("cpp"),
        "cs" => Some("csharp"),
        "rb" => Some("ruby"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::fs;
    use uuid::Uuid;

    fn create_test_repo() -> PathBuf {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_files_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();
        fs::write(
            temp_dir.join("test.rs"),
            "fn hello() { println!(\"Hello\"); }",
        )
        .unwrap();
        temp_dir
    }

    #[test]
    #[serial]
    fn canonical_handles_nonexistent_paths() {
        let nonexistent = PathBuf::from("/this/path/does/not/exist");
        let result = canonical(&nonexistent);
        assert_eq!(result, nonexistent);
    }

    #[test]
    #[serial]
    fn is_probably_binary_catches_minified_and_common_binary_exts() {
        assert!(is_probably_binary(&PathBuf::from("app.min")));
        assert!(is_probably_binary(&PathBuf::from("image.png")));
        assert!(!is_probably_binary(&PathBuf::from("main.rs")));
    }

    #[test]
    #[serial]
    fn collect_files_skips_binary_and_large_assets() {
        let temp_dir = std::env::temp_dir().join(format!("sgrep_collect_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let keep = temp_dir.join("keep.rs");
        fs::write(&keep, "fn keep() {}").unwrap();

        let binary = temp_dir.join("bundle.min.js");
        fs::write(&binary, "function minified(){}").unwrap();

        let big = temp_dir.join("big.txt");
        let large_contents = vec![b'a'; (MAX_FILE_BYTES + 1) as usize];
        fs::write(&big, large_contents).unwrap();

        let files = collect_files(&temp_dir);

        assert!(files.iter().any(|p| p.ends_with("keep.rs")));
        assert!(!files.iter().any(|p| p.ends_with("bundle.min.js")));
        assert!(!files.iter().any(|p| p.ends_with("big.txt")));

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    #[serial]
    fn normalize_to_relative_handles_deleted_absolute_path() {
        let repo = create_test_repo();
        let missing = repo.join("ghost.rs");
        let _ = std::fs::remove_file(&missing);

        let rel = normalize_to_relative(&missing, &repo);
        assert_eq!(rel, PathBuf::from("ghost.rs"));

        let nested_missing = repo.join("src").join("ghost_nested.rs");
        fs::create_dir_all(repo.join("src")).ok();
        let _ = std::fs::remove_file(&nested_missing);
        let rel_nested = normalize_to_relative(&nested_missing, &repo);
        assert_eq!(rel_nested, PathBuf::from("src/ghost_nested.rs"));

        fs::remove_dir_all(&repo).ok();
    }

    #[test]
    #[serial]
    fn collect_files_respects_gitignore() {
        let test_repo = create_test_repo();

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(&test_repo)
            .output()
            .ok();

        fs::write(test_repo.join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(test_repo.join("ignored.txt"), "this should be ignored").unwrap();
        fs::write(test_repo.join("visible.txt"), "this should be visible").unwrap();

        let files = collect_files(&test_repo);
        let file_names: Vec<String> = files
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();

        assert!(
            !file_names.contains(&"ignored.txt".to_string()),
            "ignored.txt should not be collected"
        );
        assert!(file_names.contains(&"visible.txt".to_string()));

        fs::remove_dir_all(&test_repo).ok();
    }
}
