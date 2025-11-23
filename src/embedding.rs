use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use moka::sync::Cache;
use once_cell::sync::Lazy;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};
use tracing::{info, warn};

const DEFAULT_MAX_CACHE: u64 = 50_000;
pub const DEFAULT_VECTOR_DIM: usize = 384;

/// Minimal interface so the indexer can be tested with stub embedders.
pub trait BatchEmbedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        results.into_iter().next().ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

#[derive(Clone)]
pub struct Embedder {
    cache: Cache<String, Arc<Vec<f32>>>,
    model: Arc<std::sync::Mutex<TextEmbedding>>,
}

static INIT_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CACHE)
    }
}

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

        info!(
            "Initialized embedder with providers: {}",
            execution_providers
                .iter()
                .map(|ep| format!("{ep:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
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

impl BatchEmbedder for Embedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Embedder::embed_batch(self, texts)
    }

    fn dimension(&self) -> usize {
        Embedder::dimension(self)
    }
}

pub struct PooledEmbedder {
    pool: Vec<Embedder>,
    counter: AtomicUsize,
    _cache: Cache<String, Arc<Vec<f32>>>,
}

impl PooledEmbedder {
    pub fn new(pool_size: usize, max_cache: u64) -> Self {
        let execution_providers = select_execution_providers();
        let show_progress = pool_size == 1;

        info!(
            "Initializing embedder pool with {} instances",
            pool_size
        );

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

        info!("Embedder pool initialized with {} instances", pool_size);

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

impl Default for PooledEmbedder {
    fn default() -> Self {
        let pool_size = env::var("SGREP_EMBEDDER_POOL_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get().min(8).max(1))
                    .unwrap_or(4)
            });
        Self::new(pool_size, DEFAULT_MAX_CACHE)
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

pub(crate) fn select_execution_providers() -> Vec<ExecutionProviderDispatch> {
    if env::var("SGREP_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false)
    {
        return vec![CPUExecutionProvider::default().into()];
    }

    if env::var("SGREP_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("coreml"))
        .unwrap_or(false)
    {
        return vec![
            CoreMLExecutionProvider::default().into(),
            CPUExecutionProvider::default().into(),
        ];
    }

    if env::var("SGREP_DEVICE")
        .map(|v| v.eq_ignore_ascii_case("cuda"))
        .unwrap_or(false)
    {
        return vec![
            CUDAExecutionProvider::default().into(),
            CPUExecutionProvider::default().into(),
        ];
    }

    // Auto-detect
    if is_apple_silicon() {
        return vec![
            CoreMLExecutionProvider::default().into(),
            CPUExecutionProvider::default().into(),
        ];
    }

    if has_nvidia_gpu() {
        return vec![
            CUDAExecutionProvider::default().into(),
            CPUExecutionProvider::default().into(),
        ];
    }

    // Default CPU, and tune threads if unset
    if env::var_os("ORT_NUM_THREADS").is_none() {
        if let Ok(parallelism) = std::thread::available_parallelism() {
            env::set_var("ORT_NUM_THREADS", parallelism.get().to_string());
        }
    }

    vec![CPUExecutionProvider::default().into()]
}

fn is_apple_silicon() -> bool {
    cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")
}

fn has_nvidia_gpu() -> bool {
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

    #[test]
    fn embeddings_have_correct_dimension() {
        let embedder = Embedder::default();
        let vec = embedder.embed("fn login() {}").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
    }

    #[test]
    fn identical_inputs_use_cache() {
        let embedder = Embedder::default();
        let v1 = embedder.embed("auth logic").unwrap();
        let v2 = embedder.embed("auth logic").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn similar_code_has_high_similarity() {
        let embedder = Embedder::default();
        let v1 = embedder.embed("function authenticate user").unwrap();
        let v2 = embedder.embed("function to auth users").unwrap();
        let dot: f32 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|v| v * v).sum::<f32>().sqrt();
        let similarity = dot / (norm1 * norm2);
        assert!(
            similarity > 0.5,
            "Similar code should have similarity > 0.5, got {}",
            similarity
        );
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
}
