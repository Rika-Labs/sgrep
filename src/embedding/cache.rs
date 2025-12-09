use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tracing::warn;

pub fn configure_offline_env(offline: bool) -> Result<()> {
    if offline {
        env::set_var("HF_HUB_OFFLINE", "1");
        env::set_var("FASTEMBED_DISABLE_TELEMETRY", "1");
    }

    let cache_dir = get_fastembed_cache_dir();
    fs::create_dir_all(&cache_dir).with_context(|| {
        format!(
            "Failed to prepare cache directory at {}",
            cache_dir.display()
        )
    })?;

    if offline && !cache_has_model(&cache_dir) {
        let config = super::EmbeddingModel::default().config();
        let files_list = config
            .files
            .iter()
            .map(|(_, local)| format!("  - {}", local))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(anyhow!(
            "Offline mode enabled but no cached model found.\n\n\
            Model cache directory: {}\n\n\
            Required files in {}/:\n{}\n\n\
            Run 'sgrep config --show-model-dir' for the exact path.\n\
            Download from: {}",
            cache_dir.display(),
            config.name,
            files_list,
            config.download_base_url
        ));
    }

    Ok(())
}

pub fn cache_has_model(cache_dir: &Path) -> bool {
    if !cache_dir.exists() {
        return false;
    }
    let mut stack = vec![cache_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if ext.eq_ignore_ascii_case("onnx") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub fn setup_fastembed_cache_dir() -> Option<RestoreDirGuard> {
    let cache_dir = get_fastembed_cache_dir();
    if let Err(e) = fs::create_dir_all(&cache_dir) {
        warn!(
            "Failed to create fastembed cache directory at {}: {}",
            cache_dir.display(),
            e
        );
        return None;
    }

    let original_dir = match env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            warn!("Failed to get current directory: {}", e);
            return None;
        }
    };

    if let Err(e) = env::set_current_dir(&cache_dir) {
        warn!(
            "Failed to change working directory to cache dir {}: {}",
            cache_dir.display(),
            e
        );
        return None;
    }

    Some(RestoreDirGuard { original_dir })
}

pub struct RestoreDirGuard {
    original_dir: PathBuf,
}

impl Drop for RestoreDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original_dir);
    }
}

pub fn get_fastembed_cache_dir() -> PathBuf {
    if let Some(cache) = env::var_os("FASTEMBED_CACHE_DIR") {
        return PathBuf::from(cache);
    }

    let mut cache = if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home).join(".sgrep")
    } else {
        PathBuf::from(".sgrep")
    };
    cache.push("cache");
    cache.push("fastembed");
    cache
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use uuid::Uuid;

    #[test]
    #[serial]
    fn configure_offline_env_errors_without_cached_model() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_{}", Uuid::new_v4()));
        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);
        let result = configure_offline_env(true);
        assert!(result.is_err());
        env::remove_var("FASTEMBED_CACHE_DIR");
        env::remove_var("HF_HUB_OFFLINE");
    }

    #[test]
    #[serial]
    fn configure_offline_env_errors_when_cache_dir_unwritable() {
        let temp_file =
            std::env::temp_dir().join(format!("sgrep_cache_unwritable_{}", Uuid::new_v4()));
        std::fs::write(&temp_file, b"not a directory").unwrap();
        env::set_var("FASTEMBED_CACHE_DIR", &temp_file);

        let result = configure_offline_env(false);
        assert!(result.is_err());

        env::remove_var("FASTEMBED_CACHE_DIR");
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    #[serial]
    fn configure_offline_env_succeeds_with_cached_model() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_ok_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_cache).unwrap();
        let dummy_model = temp_cache.join("dummy.onnx");
        std::fs::write(&dummy_model, b"onnx").unwrap();

        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);
        let result = configure_offline_env(true);
        assert!(result.is_ok());
        assert_eq!(env::var("HF_HUB_OFFLINE").unwrap_or_default(), "1");

        env::remove_var("FASTEMBED_CACHE_DIR");
        env::remove_var("HF_HUB_OFFLINE");
    }

    #[test]
    #[serial]
    fn configure_offline_env_noop_when_not_offline() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_noop_{}", Uuid::new_v4()));
        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);
        let result = configure_offline_env(false);
        assert!(result.is_ok());
        assert!(env::var("HF_HUB_OFFLINE").is_err());
        env::remove_var("FASTEMBED_CACHE_DIR");
    }

    #[test]
    fn cache_has_model_detects_onnx_file() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_model_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_cache).unwrap();
        let model_path = temp_cache.join("model.onnx");
        std::fs::write(&model_path, b"onnx").unwrap();
        assert!(cache_has_model(&temp_cache));
    }

    #[test]
    fn cache_has_model_returns_false_for_missing_dir() {
        let temp_cache =
            std::env::temp_dir().join(format!("sgrep_cache_missing_{}", Uuid::new_v4()));
        if temp_cache.exists() {
            std::fs::remove_dir_all(&temp_cache).unwrap();
        }
        assert!(!cache_has_model(&temp_cache));
    }

    #[test]
    #[serial]
    fn get_fastembed_cache_dir_respects_env() {
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_env_{}", Uuid::new_v4()));
        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);
        let dir = get_fastembed_cache_dir();
        assert_eq!(dir, temp_cache);
        env::remove_var("FASTEMBED_CACHE_DIR");
    }

    #[test]
    #[serial]
    fn get_fastembed_cache_dir_uses_home_sgrep() {
        let env_backup = env::var("FASTEMBED_CACHE_DIR").ok();
        env::remove_var("FASTEMBED_CACHE_DIR");
        let dir = get_fastembed_cache_dir();
        assert!(dir.to_string_lossy().contains(".sgrep"));
        assert!(dir.to_string_lossy().contains("cache"));
        assert!(dir.to_string_lossy().contains("fastembed"));
        if let Some(v) = env_backup {
            env::set_var("FASTEMBED_CACHE_DIR", v);
        }
    }

    #[test]
    #[serial]
    fn setup_fastembed_cache_dir_restores_original_workdir() {
        let original = env::current_dir().unwrap();
        let temp_cache = std::env::temp_dir().join(format!("sgrep_cache_guard_{}", Uuid::new_v4()));
        env::set_var("FASTEMBED_CACHE_DIR", &temp_cache);
        let guard = setup_fastembed_cache_dir().expect("should create and change dir");
        drop(guard);
        let now = env::current_dir().unwrap();
        assert_eq!(now, original);
        env::remove_var("FASTEMBED_CACHE_DIR");
    }

    #[test]
    #[serial]
    fn setup_fastembed_cache_dir_returns_none_on_create_failure() {
        let temp_file =
            std::env::temp_dir().join(format!("sgrep_cache_guard_file_{}", Uuid::new_v4()));
        std::fs::write(&temp_file, b"not a dir").unwrap();
        env::set_var("FASTEMBED_CACHE_DIR", &temp_file);
        let guard = setup_fastembed_cache_dir();
        assert!(guard.is_none());
        env::remove_var("FASTEMBED_CACHE_DIR");
        std::fs::remove_file(&temp_file).ok();
    }
}
