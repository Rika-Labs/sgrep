use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use blake3::Hasher;
use crossbeam::queue::ArrayQueue;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use moka::sync::Cache;
use once_cell::sync::Lazy;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};
use rayon::prelude::*;
use tracing::{debug, info, warn};

const DEFAULT_MAX_CACHE: u64 = 50_000;
pub const DEFAULT_VECTOR_DIM: usize = 384;
const PERSISTENT_CACHE_SIZE_MB: u64 = 1000; // 1GB default

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
    persistent_cache: Option<Arc<PersistentCache>>,
    model: Arc<std::sync::Mutex<TextEmbedding>>,
}

static INIT_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Persistent cache for embeddings using cacache
struct PersistentCache {
    cache_dir: PathBuf,
}

impl PersistentCache {
    fn new() -> Result<Self> {
        let cache_dir = get_persistent_cache_dir();
        fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    fn get(&self, key: &[u8]) -> Option<Vec<f32>> {
        let key_hex = hex::encode(key);
        match cacache::read_sync(&self.cache_dir, &key_hex) {
            Ok(data) => bincode::deserialize(&data).ok(),
            Err(_) => None,
        }
    }

    fn set(&self, key: &[u8], value: &[f32]) -> Result<()> {
        let key_hex = hex::encode(key);
        let data = bincode::serialize(value)?;
        cacache::write_sync(&self.cache_dir, &key_hex, data)?;
        Ok(())
    }

    fn cache_size(&self) -> u64 {
        cacache::index::ls(&self.cache_dir)
            .map(|entry| entry.size)
            .sum::<u64>()
    }

    fn should_evict(&self) -> bool {
        let max_bytes = PERSISTENT_CACHE_SIZE_MB * 1024 * 1024;
        self.cache_size() > max_bytes
    }
}

fn get_persistent_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("SGREP_CACHE_DIR") {
        return PathBuf::from(dir).join("embeddings");
    }

    if let Some(dirs) = directories::ProjectDirs::from("dev", "RikaLabs", "sgrep") {
        return dirs.cache_dir().join("embeddings");
    }

    PathBuf::from(".sgrep").join("cache").join("embeddings")
}

fn hash_text(text: &str) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(text.as_bytes());
    *hasher.finalize().as_bytes()
}

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

        // Enable persistent cache unless disabled
        let persistent_cache = if env::var("SGREP_DISABLE_PERSISTENT_CACHE").is_ok() {
            None
        } else {
            match PersistentCache::new() {
                Ok(cache) => {
                    debug!("Persistent embedding cache enabled at {:?}", cache.cache_dir);
                    Some(Arc::new(cache))
                }
                Err(e) => {
                    warn!("Failed to initialize persistent cache: {}, falling back to memory-only", e);
                    None
                }
            }
        };

        Self {
            cache: Cache::builder().max_capacity(max_cache).build(),
            persistent_cache,
            model: Arc::new(std::sync::Mutex::new(model)),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Check memory cache first
        if let Some(vec) = self.cache.get(text) {
            return Ok(vec.as_ref().clone());
        }

        // Check persistent cache
        if let Some(ref pcache) = self.persistent_cache {
            let hash = hash_text(text);
            if let Some(vec) = pcache.get(&hash) {
                self.cache.insert(text.to_string(), Arc::new(vec.clone()));
                return Ok(vec);
            }
        }

        // Generate embedding
        let mut model = self.model.lock().unwrap();
        let embeddings = model.embed(vec![text], None)?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;

        // Store in caches
        self.cache.insert(text.to_string(), Arc::new(vector.clone()));
        if let Some(ref pcache) = self.persistent_cache {
            let hash = hash_text(text);
            let _ = pcache.set(&hash, &vector); // Ignore errors on write
        }

        Ok(vector)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut uncached = Vec::new();
        let mut uncached_indices = Vec::new();
        let mut results = vec![Vec::new(); texts.len()];

        // Check memory cache
        for (i, text) in texts.iter().enumerate() {
            if let Some(vec) = self.cache.get(text) {
                results[i] = vec.as_ref().clone();
            } else {
                uncached.push(text.clone());
                uncached_indices.push(i);
            }
        }

        if uncached.is_empty() {
            return Ok(results);
        }

        // Check persistent cache for uncached items
        let mut still_uncached = Vec::new();
        let mut still_uncached_indices = Vec::new();

        if let Some(ref pcache) = self.persistent_cache {
            for (text, &idx) in uncached.iter().zip(&uncached_indices) {
                let hash = hash_text(text);
                if let Some(vec) = pcache.get(&hash) {
                    results[idx] = vec.clone();
                    self.cache.insert(text.clone(), Arc::new(vec));
                } else {
                    still_uncached.push(text.as_str());
                    still_uncached_indices.push(idx);
                }
            }
        } else {
            still_uncached = uncached.iter().map(|s| s.as_str()).collect();
            still_uncached_indices = uncached_indices.clone();
        }

        // Generate embeddings for items not in any cache
        if !still_uncached.is_empty() {
            let mut model = self.model.lock().unwrap();
            let embeddings = model.embed(still_uncached.clone(), None)?;

            for (embedding, &idx) in embeddings.iter().zip(&still_uncached_indices) {
                results[idx] = embedding.clone();
                let text = &texts[idx];
                self.cache.insert(text.clone(), Arc::new(embedding.clone()));

                // Store in persistent cache
                if let Some(ref pcache) = self.persistent_cache {
                    let hash = hash_text(text);
                    let _ = pcache.set(&hash, embedding);
                }
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

/// Work-stealing pooled embedder for better load balancing
pub struct PooledEmbedder {
    workers: Vec<Arc<Embedder>>,
    work_queue: Arc<ArrayQueue<usize>>,
    _cache: Cache<String, Arc<Vec<f32>>>,
}

impl PooledEmbedder {
    pub fn new(pool_size: usize, max_cache: u64) -> Self {
        let execution_providers = select_execution_providers();
        let show_progress = pool_size == 1;
        let cache = Cache::builder().max_capacity(max_cache).build();

        info!("Initializing embedder pool with {} workers (pre-warming in parallel)", pool_size);

        // Pre-warm all model instances in parallel using rayon
        let workers: Vec<Arc<Embedder>> = (0..pool_size)
            .into_par_iter()
            .map(|i| {
                let _init_guard = INIT_LOCK.lock().unwrap();
                let _cache_guard = setup_fastembed_cache_dir();

                let model = TextEmbedding::try_new(
                    InitOptions::new(EmbeddingModel::BGESmallENV15Q)
                        .with_execution_providers(execution_providers.clone())
                        .with_show_download_progress(show_progress && i == 0),
                )
                .expect("Failed to initialize embedding model");
                drop(_cache_guard);

                // Enable persistent cache for all workers (shared across pool)
                let persistent_cache = if env::var("SGREP_DISABLE_PERSISTENT_CACHE").is_ok() {
                    None
                } else {
                    PersistentCache::new().ok().map(Arc::new)
                };

                Arc::new(Embedder {
                    cache: cache.clone(),
                    persistent_cache,
                    model: Arc::new(Mutex::new(model)),
                })
            })
            .collect();

        info!("Embedder pool initialized successfully");

        // Initialize work queue with worker indices
        let work_queue = Arc::new(ArrayQueue::new(pool_size));
        for i in 0..pool_size {
            work_queue.push(i).unwrap();
        }

        Self {
            workers,
            work_queue,
            _cache: cache,
        }
    }

    fn get_embedder(&self) -> Arc<Embedder> {
        // Try to steal a worker from the queue
        if let Some(idx) = self.work_queue.pop() {
            let worker = self.workers[idx].clone();
            // Return worker to queue when done (handled by caller)
            worker
        } else {
            // Fallback to first worker if queue is empty (shouldn't happen often)
            self.workers[0].clone()
        }
    }

    fn return_worker(&self, worker_idx: usize) {
        let _ = self.work_queue.push(worker_idx);
    }
}

impl BatchEmbedder for PooledEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embedder = self.get_embedder();
        let result = embedder.embed(text);

        // Find worker index to return it
        for (idx, worker) in self.workers.iter().enumerate() {
            if Arc::ptr_eq(&embedder, worker) {
                self.return_worker(idx);
                break;
            }
        }

        result
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embedder = self.get_embedder();
        let result = embedder.embed_batch(texts);

        // Find worker index to return it
        for (idx, worker) in self.workers.iter().enumerate() {
            if Arc::ptr_eq(&embedder, worker) {
                self.return_worker(idx);
                break;
            }
        }

        result
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
        env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
        let embedder = Embedder::default();
        let vec = embedder.embed("fn login() {}").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
        env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
    }

    #[test]
    fn identical_inputs_use_cache() {
        env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
        let embedder = Embedder::default();
        let v1 = embedder.embed("auth logic").unwrap();
        let v2 = embedder.embed("auth logic").unwrap();
        assert_eq!(v1, v2);
        env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
    }

    #[test]
    fn similar_code_has_high_similarity() {
        env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
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
        env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
    }

    #[test]
    fn persistent_cache_works() {
        let embedder = Embedder::default();
        let text = "fn persistent_test() {}";

        // First embed (cache miss)
        let v1 = embedder.embed(text).unwrap();

        // Clear memory cache
        embedder.cache.invalidate_all();

        // Second embed (should hit persistent cache)
        let v2 = embedder.embed(text).unwrap();

        assert_eq!(v1, v2, "Persistent cache should return same embedding");
    }

    #[test]
    fn pooled_embedder_works() {
        env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
        env::set_var("SGREP_EMBEDDER_POOL_SIZE", "2");
        let embedder = PooledEmbedder::default();
        let vec = embedder.embed("fn pooled_test() {}").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
        env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
        env::remove_var("SGREP_EMBEDDER_POOL_SIZE");
    }

    #[test]
    fn batch_embedding_works() {
        env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
        let embedder = Embedder::default();
        let texts = vec![
            "fn test1() {}".to_string(),
            "fn test2() {}".to_string(),
        ];
        let results = embedder.embed_batch(&texts).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), DEFAULT_VECTOR_DIM);
        assert_eq!(results[1].len(), DEFAULT_VECTOR_DIM);
        env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
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
    fn hash_text_is_deterministic() {
        let text = "fn test() {}";
        let h1 = hash_text(text);
        let h2 = hash_text(text);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_text_differs_for_different_input() {
        let h1 = hash_text("fn test1() {}");
        let h2 = hash_text("fn test2() {}");
        assert_ne!(h1, h2);
    }
}
