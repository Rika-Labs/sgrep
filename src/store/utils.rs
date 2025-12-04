use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use blake3::Hasher;
use directories::ProjectDirs;

pub fn find_worktree_root(path: &Path) -> Option<PathBuf> {
    let mut current = path.to_path_buf();

    if current.is_file() {
        current = current.parent()?.to_path_buf();
    }

    loop {
        let git_path = current.join(".git");

        if git_path.exists() {
            return Some(current);
        }

        if !current.pop() {
            break;
        }
    }

    None
}

pub fn is_worktree(path: &Path) -> bool {
    if let Some(root) = find_worktree_root(path) {
        let git_path = root.join(".git");
        git_path.is_file()
    } else {
        false
    }
}

pub fn get_main_repo_path(worktree_path: &Path) -> Option<PathBuf> {
    let root = find_worktree_root(worktree_path)?;
    let git_file = root.join(".git");

    if !git_file.is_file() {
        return None;
    }

    let content = fs::read_to_string(&git_file).ok()?;
    let gitdir_line = content.lines().next()?;

    let gitdir = gitdir_line.strip_prefix("gitdir: ")?.trim();
    let gitdir_path = PathBuf::from(gitdir);

    let mut path = if gitdir_path.is_absolute() {
        gitdir_path
    } else {
        git_file.parent()?.join(gitdir).canonicalize().ok()?
    };

    for _ in 0..3 {
        path = path.parent()?.to_path_buf();
    }

    if path.join(".git").is_dir() {
        Some(fs::canonicalize(&path).unwrap_or(path))
    } else {
        None
    }
}

pub fn quantize_to_binary(vector: &[f32]) -> Vec<u64> {
    let num_words = vector.len().div_ceil(64);
    let mut binary = vec![0u64; num_words];
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            binary[i / 64] |= 1u64 << (i % 64);
        }
    }
    binary
}

pub fn data_dir() -> PathBuf {
    if let Ok(home) = env::var("SGREP_HOME") {
        return PathBuf::from(home);
    }
    ProjectDirs::from("dev", "RikaLabs", "sgrep")
        .map(|d| d.data_local_dir().to_path_buf())
        .unwrap_or_else(fallback_data_dir)
}

pub fn fallback_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".sgrep")
}

pub fn hash_path(path: &Path) -> String {
    let mut hasher = Hasher::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use uuid::Uuid;

    fn temp_dir_with_name(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("sgrep_test_{}_{}", name, Uuid::new_v4()))
    }

    #[test]
    fn hash_is_deterministic() {
        let path = Path::new("/test/path");
        assert_eq!(hash_path(path), hash_path(path));
    }

    #[test]
    fn hash_differs_for_different_paths() {
        assert_ne!(hash_path(Path::new("/a")), hash_path(Path::new("/b")));
    }

    #[test]
    fn data_dir_uses_sgrep_home_when_set() {
        let prev = std::env::var("SGREP_HOME").ok();
        std::env::set_var("SGREP_HOME", "/custom/path");
        assert_eq!(data_dir(), PathBuf::from("/custom/path"));
        match prev {
            Some(v) => std::env::set_var("SGREP_HOME", v),
            None => std::env::remove_var("SGREP_HOME"),
        }
    }

    #[test]
    fn fallback_uses_home_directory() {
        let dir = fallback_data_dir();
        assert!(dir.to_string_lossy().contains(".sgrep"));
    }

    #[test]
    fn find_worktree_root_finds_normal_repo() {
        let temp_dir = temp_dir_with_name("git_root");
        fs::create_dir_all(&temp_dir).unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(&temp_dir)
            .output()
            .ok();

        let nested = temp_dir.join("src").join("lib");
        fs::create_dir_all(&nested).unwrap();

        let result = find_worktree_root(&nested);
        assert!(result.is_some());
        let root = result.unwrap();
        assert!(root.join(".git").exists());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn find_worktree_root_returns_none_for_non_git() {
        let temp_dir = temp_dir_with_name("not_git");
        fs::create_dir_all(&temp_dir).unwrap();

        let result = find_worktree_root(&temp_dir);
        assert!(result.is_none());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn is_worktree_returns_false_for_normal_repo() {
        let temp_dir = temp_dir_with_name("normal_repo");
        fs::create_dir_all(&temp_dir).unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(&temp_dir)
            .output()
            .ok();

        assert!(!is_worktree(&temp_dir));

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn get_main_repo_path_returns_none_for_normal_repo() {
        let temp_dir = temp_dir_with_name("normal_repo2");
        fs::create_dir_all(&temp_dir).unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(&temp_dir)
            .output()
            .ok();

        assert!(get_main_repo_path(&temp_dir).is_none());

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn worktree_functions_handle_actual_worktree() {
        let main_repo = temp_dir_with_name("main_repo");
        let worktree_dir = temp_dir_with_name("worktree");
        fs::create_dir_all(&main_repo).unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(&main_repo)
            .output()
            .ok();

        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(&main_repo)
            .output()
            .ok();

        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&main_repo)
            .output()
            .ok();

        fs::write(main_repo.join("test.txt"), "content").unwrap();

        Command::new("git")
            .args(["add", "."])
            .current_dir(&main_repo)
            .output()
            .ok();

        Command::new("git")
            .args(["commit", "-m", "Initial"])
            .current_dir(&main_repo)
            .output()
            .ok();

        let worktree_result = Command::new("git")
            .args([
                "worktree",
                "add",
                worktree_dir.to_str().unwrap(),
                "-b",
                "test-branch",
            ])
            .current_dir(&main_repo)
            .output();

        if worktree_result.is_err() {
            fs::remove_dir_all(&main_repo).ok();
            return;
        }

        let git_file = worktree_dir.join(".git");
        if !git_file.is_file() {
            fs::remove_dir_all(&main_repo).ok();
            fs::remove_dir_all(&worktree_dir).ok();
            return;
        }

        assert!(is_worktree(&worktree_dir));

        let main_path = get_main_repo_path(&worktree_dir);
        assert!(main_path.is_some());
        let main_canonical = fs::canonicalize(&main_repo).unwrap();
        assert_eq!(main_path.unwrap(), main_canonical);

        Command::new("git")
            .args(["worktree", "remove", worktree_dir.to_str().unwrap()])
            .current_dir(&main_repo)
            .output()
            .ok();

        fs::remove_dir_all(&main_repo).ok();
        fs::remove_dir_all(&worktree_dir).ok();
    }
}
