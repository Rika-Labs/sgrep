use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(not(test))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(test))]
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
#[cfg(not(test))]
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
#[cfg(not(test))]
use moka::sync::Cache;
#[cfg(not(test))]
use once_cell::sync::Lazy;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};
use tracing::warn;

#[cfg(not(test))]
const DEFAULT_MAX_CACHE: u64 = 50_000;
pub const DEFAULT_VECTOR_DIM: usize = 384;

/// Minimal interface so the indexer can be tested with stub embedders.
pub trait BatchEmbedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

#[cfg_attr(not(test), derive(Clone))]
#[cfg_attr(test, derive(Clone, Default))]
pub struct Embedder {
    #[cfg(not(test))]
    cache: Cache<String, Arc<Vec<f32>>>,
    #[cfg(not(test))]
    model: Arc<std::sync::Mutex<TextEmbedding>>,
}

/// Configures environment and performs preflight checks for offline usage.
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
        return Err(anyhow!(
            "Offline mode enabled but no cached model was found under {}. \
             Download the model once with network enabled or place the BGE-small-en-v1.5-q files in that directory.",
            cache_dir.display()
        ));
    }

    Ok(())
}

fn cache_has_model(cache_dir: &Path) -> bool {
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

#[cfg(not(test))]
static INIT_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(not(test))]
impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHE)
    }
}

#[cfg(not(test))]
impl Embedder {
    pub fn new(max_cache: u64) -> Self {
        Self::with_options(max_cache, true)
    }

    fn with_options(max_cache: u64, show_download_progress: bool) -> Self {
        // Serialize model initialization to avoid fastembed lock contention when tests run in parallel
        let _init_guard = INIT_LOCK.lock().unwrap();

        let _cache_guard = setup_fastembed_cache_dir();
        let execution_providers = select_execution_providers();
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15Q)
                .with_execution_providers(execution_providers.clone())
                .with_show_download_progress(show_download_progress),
        )
        .expect("Failed to initialize embedding model");
        drop(_cache_guard);

        Self {
            cache: Cache::builder().max_capacity(max_cache).build(),
            model: Arc::new(std::sync::Mutex::new(model)),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(vec) = self.cache.get(text) {
            return Ok(vec.as_ref().clone());
        }
        let mut model = self.model.lock().unwrap();
        let embeddings = model.embed(vec![text], None)?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;
        self.cache
            .insert(text.to_string(), Arc::new(vector.clone()));
        Ok(vector)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut uncached = Vec::new();
        let mut uncached_indices = Vec::new();
        let mut results = vec![Vec::new(); texts.len()];

        for (i, text) in texts.iter().enumerate() {
            if let Some(vec) = self.cache.get(text) {
                results[i] = vec.as_ref().clone();
            } else {
                uncached.push(text.as_str());
                uncached_indices.push(i);
            }
        }

        if !uncached.is_empty() {
            let mut model = self.model.lock().unwrap();
            let embeddings = model.embed(uncached.clone(), None)?;

            for (embedding, &idx) in embeddings.iter().zip(&uncached_indices) {
                results[idx] = embedding.clone();
                self.cache
                    .insert(texts[idx].clone(), Arc::new(embedding.clone()));
            }
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

#[cfg(not(test))]
impl BatchEmbedder for Embedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Embedder::embed_batch(self, texts)
    }

    fn dimension(&self) -> usize {
        Embedder::dimension(self)
    }
}

#[cfg(test)]
impl BatchEmbedder for Embedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32; DEFAULT_VECTOR_DIM])
            .collect())
    }

    fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

#[cfg(not(test))]
pub struct PooledEmbedder {
    pool: Vec<Embedder>,
    counter: AtomicUsize,
    _cache: Cache<String, Arc<Vec<f32>>>,
}

#[cfg(not(test))]
impl Clone for PooledEmbedder {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            counter: AtomicUsize::new(self.counter.load(Ordering::Relaxed)),
            _cache: self._cache.clone(),
        }
    }
}

#[cfg(not(test))]
impl PooledEmbedder {
    pub fn new(pool_size: usize, max_cache: u64) -> Self {
        let execution_providers = select_execution_providers();
        let show_progress = pool_size == 1;

        let mut pool = Vec::with_capacity(pool_size);
        let cache = Cache::builder().max_capacity(max_cache).build();

        for i in 0..pool_size {
            let _init_guard = INIT_LOCK.lock().unwrap();
            let _cache_guard = setup_fastembed_cache_dir();

            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::BGESmallENV15Q)
                    .with_execution_providers(execution_providers.clone())
                    .with_show_download_progress(show_progress && i == 0),
            )
            .expect("Failed to initialize embedding model");
            drop(_cache_guard);

            let embedder = Embedder {
                cache: cache.clone(),
                model: Arc::new(Mutex::new(model)),
            };
            pool.push(embedder);
        }

        Self {
            pool,
            counter: AtomicUsize::new(0),
            _cache: cache,
        }
    }

    fn get_embedder(&self) -> &Embedder {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.pool.len();
        &self.pool[idx]
    }
}

#[cfg(not(test))]
impl BatchEmbedder for PooledEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.get_embedder().embed(text)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.get_embedder().embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

#[cfg(not(test))]
impl Default for PooledEmbedder {
    fn default() -> Self {
        let pool_size = env::var("SGREP_EMBEDDER_POOL_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get().clamp(1, 8))
                    .unwrap_or(4)
            });
        Self::new(pool_size, DEFAULT_MAX_CACHE)
    }
}

#[cfg(test)]
#[derive(Clone)]
pub struct PooledEmbedder;

#[cfg(test)]
impl BatchEmbedder for PooledEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(vec![text.len() as f32; DEFAULT_VECTOR_DIM])
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32; DEFAULT_VECTOR_DIM])
            .collect())
    }

    fn dimension(&self) -> usize {
        DEFAULT_VECTOR_DIM
    }
}

#[cfg(test)]
impl Default for PooledEmbedder {
    fn default() -> Self {
        Self
    }
}

fn setup_fastembed_cache_dir() -> Option<RestoreDirGuard> {
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

struct RestoreDirGuard {
    original_dir: PathBuf,
}

impl Drop for RestoreDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original_dir);
    }
}

fn get_fastembed_cache_dir() -> PathBuf {
    if let Some(cache) = env::var_os("FASTEMBED_CACHE_DIR") {
        return PathBuf::from(cache);
    }

    if let Some(dirs) = directories::ProjectDirs::from("dev", "RikaLabs", "sgrep") {
        let mut cache = dirs.cache_dir().to_path_buf();
        cache.push("fastembed");
        return cache;
    }

    let mut cache = dirs_next_best_cache();
    cache.push("fastembed");
    cache
}

fn dirs_next_best_cache() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        let mut cache = PathBuf::from(home);
        cache.push(".sgrep");
        cache.push("cache");
        return cache;
    }

    PathBuf::from(".sgrep").join("cache")
}

fn optimized_cpu_provider() -> ExecutionProviderDispatch {
    CPUExecutionProvider::default().with_arena_allocator(true).into()
}

pub(crate) fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    configure_onnx_threading();

    let device = env::var("SGREP_DEVICE").unwrap_or_default().to_lowercase();

    match device.as_str() {
        "cpu" => vec![optimized_cpu_provider()],
        "coreml" => vec![CoreMLExecutionProvider::default().into(), optimized_cpu_provider()],
        "cuda" => vec![CUDAExecutionProvider::default().into(), optimized_cpu_provider()],
        _ => auto_detect_providers(),
    }
}

fn auto_detect_providers() -> Vec<ExecutionProviderDispatch> {
    if is_apple_silicon() {
        return vec![CoreMLExecutionProvider::default().into(), optimized_cpu_provider()];
    }

    if has_nvidia_gpu() {
        return vec![CUDAExecutionProvider::default().into(), optimized_cpu_provider()];
    }

    vec![optimized_cpu_provider()]
}

fn configure_onnx_threading() {
    let parallelism = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    if env::var_os("ORT_NUM_THREADS").is_none() {
        env::set_var("ORT_NUM_THREADS", parallelism.to_string());
    }

    if env::var_os("ORT_INTER_OP_NUM_THREADS").is_none() {
        let inter_threads = (parallelism / 2).max(1);
        env::set_var("ORT_INTER_OP_NUM_THREADS", inter_threads.to_string());
    }
}

fn is_apple_silicon() -> bool {
    #[cfg(test)]
    if let Ok(val) = std::env::var("SGREP_TEST_APPLE") {
        return match val.as_str() {
            "1" => true,
            "0" => false,
            _ => cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"),
        };
    }

    cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
}

fn has_nvidia_gpu() -> bool {
    #[cfg(test)]
    if std::env::var("SGREP_TEST_NVIDIA")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }

    Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;
    use uuid::Uuid;

    #[derive(Clone, Default)]
    struct FakeEmbedder;

    impl BatchEmbedder for FakeEmbedder {
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32, 0.0, 1.0])
                .collect())
        }

        fn dimension(&self) -> usize {
            3
        }
    }

    #[test]
    fn batch_embedder_embed_calls_batch() {
        let embedder = FakeEmbedder::default();
        let v = embedder.embed("hi").unwrap();
        assert_eq!(v, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    #[serial]
    fn select_cpu_when_forced() {
        env::set_var("SGREP_DEVICE", "cpu");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CPUExecutionProvider"));
        assert_eq!(eps.len(), 1);
    }

    #[test]
    #[serial]
    fn select_coreml_when_forced() {
        env::set_var("SGREP_DEVICE", "coreml");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CoreMLExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider")); // fallback stays
    }

    #[test]
    #[serial]
    fn select_cuda_when_forced() {
        env::set_var("SGREP_DEVICE", "cuda");
        let eps = select_execution_providers();
        env::remove_var("SGREP_DEVICE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CUDAExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

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
        // Simulate a cached ONNX model file
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
    fn embed_errors_when_batch_returns_empty() {
        struct EmptyEmbedder;
        impl BatchEmbedder for EmptyEmbedder {
            fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(Vec::new())
            }
            fn dimension(&self) -> usize {
                4
            }
        }

        let embedder = EmptyEmbedder;
        let result = embedder.embed("hi");
        assert!(result.is_err());
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
    fn dirs_next_best_cache_without_home() {
        let home_backup = env::var("HOME").ok();
        env::remove_var("HOME");
        let dir = dirs_next_best_cache();
        assert!(dir.ends_with(".sgrep/cache"));
        env::remove_var("HOME");
        if let Some(home) = home_backup {
            env::set_var("HOME", home);
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

    #[test]
    #[serial]
    fn select_execution_providers_prefers_test_gpu_override() {
        env::remove_var("SGREP_DEVICE");
        env::set_var("SGREP_TEST_APPLE", "0");
        env::set_var("SGREP_TEST_NVIDIA", "1");
        let eps = select_execution_providers();
        env::remove_var("SGREP_TEST_NVIDIA");
        env::remove_var("SGREP_TEST_APPLE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CUDAExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

    #[test]
    #[serial]
    fn select_execution_providers_prefers_test_apple_override() {
        env::remove_var("SGREP_DEVICE");
        env::set_var("SGREP_TEST_APPLE", "1");
        let eps = select_execution_providers();
        env::remove_var("SGREP_TEST_APPLE");
        let joined = format!("{:?}", eps);
        assert!(joined.contains("CoreMLExecutionProvider"));
        assert!(joined.contains("CPUExecutionProvider"));
    }

    #[test]
    fn pooled_embedder_uses_default_impl() {
        let embedder = PooledEmbedder::default();
        let vec = embedder.embed("hello").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
    }
}
