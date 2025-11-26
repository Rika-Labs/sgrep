use std::env;
use std::path::{Path, PathBuf};

use blake3::Hasher;
use directories::ProjectDirs;

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
}
