#[cfg(not(test))]
use std::env;
#[cfg(not(test))]
use std::fs;
#[cfg(not(test))]
use std::io::Write;
#[cfg(not(test))]
use std::path::PathBuf;
#[cfg(not(test))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(test))]
use std::sync::{Arc, Mutex};
#[cfg(not(test))]
use std::time::Duration;

use anyhow::Result;
#[cfg(not(test))]
use anyhow::{anyhow, Context};
#[cfg(not(test))]
use fastembed::{InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
#[cfg(not(test))]
use moka::sync::Cache;
#[cfg(not(test))]
use once_cell::sync::Lazy;
#[cfg(not(test))]
use ort::execution_providers::ExecutionProviderDispatch;
#[cfg(not(test))]
use ureq::{Agent, AgentBuilder, Proxy};

#[cfg(not(test))]
use super::cache::{get_fastembed_cache_dir, setup_fastembed_cache_dir};
#[cfg(not(test))]
use super::providers::select_execution_providers;
#[cfg(not(test))]
use super::EmbeddingModel;
use super::BatchEmbedder;
#[cfg(test)]
use super::DEFAULT_VECTOR_DIM;

#[cfg(not(test))]
const DEFAULT_MAX_CACHE: u64 = 100_000;

#[cfg(not(test))]
static INIT_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(not(test))]
fn create_http_agent() -> Agent {
    let proxy_url = env::var("https_proxy")
        .or_else(|_| env::var("HTTPS_PROXY"))
        .or_else(|_| env::var("http_proxy"))
        .or_else(|_| env::var("HTTP_PROXY"))
        .ok();

    let mut builder = AgentBuilder::new();

    if let Some(url) = proxy_url {
        if let Ok(proxy) = Proxy::new(&url) {
            builder = builder.proxy(proxy);
        }
    }

    builder.build()
}

#[cfg_attr(not(test), derive(Clone))]
#[cfg_attr(test, derive(Clone, Default))]
pub struct Embedder {
    #[cfg(not(test))]
    model: EmbeddingModel,
    #[cfg(not(test))]
    cache: Cache<String, Arc<Vec<f32>>>,
    #[cfg(not(test))]
    inner: Arc<std::sync::Mutex<TextEmbedding>>,
}

#[cfg(not(test))]
impl Default for Embedder {
    fn default() -> Self {
        Self::with_default_model(DEFAULT_MAX_CACHE)
    }
}

#[cfg(not(test))]
impl Embedder {
    pub fn new(model: EmbeddingModel, max_cache: u64) -> Self {
        Self::with_options(model, max_cache, true)
    }

    pub fn with_default_model(max_cache: u64) -> Self {
        Self::new(EmbeddingModel::default(), max_cache)
    }

    fn with_options(model: EmbeddingModel, max_cache: u64, show_download_progress: bool) -> Self {
        let _init_guard = INIT_LOCK.lock().unwrap();

        let _cache_guard = setup_fastembed_cache_dir();
        let execution_providers = select_execution_providers();

        let init_timeout = Duration::from_secs(
            env::var("SGREP_INIT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(super::DEFAULT_INIT_TIMEOUT_SECS),
        );

        let text_embedding =
            init_model_with_timeout(model, execution_providers, show_download_progress, init_timeout)
                .expect(
                    "Failed to initialize embedding model (try increasing SGREP_INIT_TIMEOUT_SECS)",
                );
        drop(_cache_guard);

        Self {
            model,
            cache: Cache::builder().max_capacity(max_cache).build(),
            inner: Arc::new(std::sync::Mutex::new(text_embedding)),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(vec) = self.cache.get(text) {
            return Ok(vec.as_ref().clone());
        }
        let mut inner = self.inner.lock().unwrap();
        let embeddings = inner.embed(vec![text], None)?;
        let mut vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?;

        let config = self.model.config();
        if config.native_dim != config.output_dim {
            vector.truncate(config.output_dim);
        }

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
            let mut inner = self.inner.lock().unwrap();
            let embeddings = inner.embed(uncached.clone(), None)?;

            let config = self.model.config();
            for (embedding, &idx) in embeddings.iter().zip(&uncached_indices) {
                let mut truncated = embedding.clone();
                if config.native_dim != config.output_dim {
                    truncated.truncate(config.output_dim);
                }
                results[idx] = truncated.clone();
                self.cache.insert(texts[idx].clone(), Arc::new(truncated));
            }
        }

        Ok(results)
    }

    pub fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&super::ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let total = texts.len();
        let mut results = Vec::with_capacity(total);

        for (i, text) in texts.iter().enumerate() {
            let vec = self.embed(text)?;
            results.push(vec);

            if let Some(callback) = on_progress {
                callback(super::EmbedProgress {
                    completed: i + 1,
                    total,
                    message: None,
                });
            }
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.model.config().output_dim
    }
}

#[cfg(not(test))]
impl BatchEmbedder for Embedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Embedder::embed_batch(self, texts)
    }

    fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&super::ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        Embedder::embed_batch_with_progress(self, texts, on_progress)
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

    fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&super::ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let mut results = Vec::with_capacity(total);

        for (i, text) in texts.iter().enumerate() {
            results.push(vec![text.len() as f32; DEFAULT_VECTOR_DIM]);

            if let Some(callback) = on_progress {
                callback(super::EmbedProgress {
                    completed: i + 1,
                    total,
                    message: Some(format!("{}/{}", i + 1, total)),
                });
            }
        }

        Ok(results)
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
    pub fn new(model: EmbeddingModel, pool_size: usize, max_cache: u64) -> Self {
        let execution_providers = select_execution_providers();
        let show_progress = pool_size == 1;

        let init_timeout = Duration::from_secs(
            env::var("SGREP_INIT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(super::DEFAULT_INIT_TIMEOUT_SECS),
        );

        let mut pool = Vec::with_capacity(pool_size);
        let cache = Cache::builder().max_capacity(max_cache).build();

        for i in 0..pool_size {
            let _init_guard = INIT_LOCK.lock().unwrap();
            let _cache_guard = setup_fastembed_cache_dir();

            let text_embedding = init_model_with_timeout(
                model,
                execution_providers.clone(),
                show_progress && i == 0,
                init_timeout,
            )
            .expect(
                "Failed to initialize embedding model (try increasing SGREP_INIT_TIMEOUT_SECS)",
            );
            drop(_cache_guard);

            let embedder = Embedder {
                model,
                cache: cache.clone(),
                inner: Arc::new(Mutex::new(text_embedding)),
            };
            pool.push(embedder);
        }

        Self {
            pool,
            counter: AtomicUsize::new(0),
            _cache: cache,
        }
    }

    pub fn with_default_model(pool_size: usize, max_cache: u64) -> Self {
        Self::new(EmbeddingModel::default(), pool_size, max_cache)
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

    fn embed_batch_with_progress(
        &self,
        texts: &[String],
        on_progress: Option<&super::ProgressCallback>,
    ) -> Result<Vec<Vec<f32>>> {
        self.get_embedder()
            .embed_batch_with_progress(texts, on_progress)
    }

    fn dimension(&self) -> usize {
        self.get_embedder().dimension()
    }
}

#[cfg(not(test))]
impl Default for PooledEmbedder {
    fn default() -> Self {
        Self::with_default_model(1, DEFAULT_MAX_CACHE)
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

#[cfg(not(test))]
fn get_model_cache_dir(model: EmbeddingModel) -> PathBuf {
    get_fastembed_cache_dir().join(model.config().name)
}

#[cfg(not(test))]
fn download_file(url: &str, path: &std::path::Path, show_progress: bool) -> Result<()> {
    use std::io::Read;

    if path.exists() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    if show_progress {
        eprintln!(
            "Downloading {}...",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    let agent = create_http_agent();
    let response = agent.get(url).call()?;

    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;

    let mut file = fs::File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}

#[cfg(not(test))]
fn load_model(model: EmbeddingModel, show_download_progress: bool) -> Result<UserDefinedEmbeddingModel> {
    let config = model.config();
    let cache_dir = get_model_cache_dir(model);

    for (remote_path, local_name) in config.files {
        let url = format!("{}/{}", config.download_base_url, remote_path);
        let local_path = cache_dir.join(local_name);
        download_file(&url, &local_path, show_download_progress).with_context(|| {
            let files_list: Vec<&str> = config.files.iter().map(|(_, local)| *local).collect();
            format!(
                "Failed to download {}\n\n\
                If HuggingFace is blocked in your region:\n\
                1. Use proxy: export HTTPS_PROXY=http://proxy:port\n\
                2. Manual download: Place files in {}\n\
                   Required: {}\n\n\
                Run 'sgrep config --show-model-dir' to see the exact path.\n\
                Download from: {}",
                url,
                cache_dir.display(),
                files_list.join(", "),
                config.download_base_url
            )
        })?;
    }

    let onnx_local_name = config.files.iter()
        .find(|(remote, _)| remote.ends_with(".onnx"))
        .map(|(_, local)| *local)
        .expect("ModelConfig must have an ONNX file");
    let onnx_path = cache_dir.join(onnx_local_name);
    let tokenizer_path = cache_dir.join("tokenizer.json");
    let config_path = cache_dir.join("config.json");
    let special_tokens_path = cache_dir.join("special_tokens_map.json");
    let tokenizer_config_path = cache_dir.join("tokenizer_config.json");

    let onnx_file =
        fs::read(&onnx_path).with_context(|| format!("Failed to read {}", onnx_path.display()))?;
    let tokenizer_file = fs::read(&tokenizer_path)
        .with_context(|| format!("Failed to read {}", tokenizer_path.display()))?;
    let config_file = fs::read(&config_path)
        .with_context(|| format!("Failed to read {}", config_path.display()))?;
    let special_tokens_map_file = fs::read(&special_tokens_path)
        .with_context(|| format!("Failed to read {}", special_tokens_path.display()))?;
    let tokenizer_config_file = fs::read(&tokenizer_config_path)
        .with_context(|| format!("Failed to read {}", tokenizer_config_path.display()))?;

    Ok(UserDefinedEmbeddingModel::new(
        onnx_file,
        TokenizerFiles {
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        },
    ))
}

#[cfg(not(test))]
fn init_model_with_timeout(
    model: EmbeddingModel,
    execution_providers: Vec<ExecutionProviderDispatch>,
    show_download_progress: bool,
    timeout: Duration,
) -> Result<TextEmbedding> {
    use std::sync::mpsc;
    use std::thread;

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let result = (|| {
            let model_data = load_model(model, show_download_progress)?;
            TextEmbedding::try_new_from_user_defined(
                model_data,
                InitOptionsUserDefined::default().with_execution_providers(execution_providers),
            )
            .map_err(|e| anyhow!("{}", e))
        })();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(model)) => Ok(model),
        Ok(Err(e)) => Err(anyhow!("Model initialization failed: {}", e)),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!(
            "Model initialization timed out after {:?}.\n\n\
                Possible causes:\n\
                - First-time model download is slow or blocked\n\
                - HuggingFace may be unreachable in your region\n\n\
                Solutions:\n\
                - Increase timeout: SGREP_INIT_TIMEOUT_SECS=600\n\
                - Use proxy: HTTPS_PROXY=http://your-proxy:port\n\
                - Manual download: sgrep config --show-model-dir",
            timeout
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(anyhow!("Model initialization thread crashed unexpectedly"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_embedder_uses_default_impl() {
        let embedder = PooledEmbedder::default();
        let vec = embedder.embed("hello").unwrap();
        assert_eq!(vec.len(), DEFAULT_VECTOR_DIM);
    }

    #[test]
    fn embed_batch_with_progress_reports_one_at_a_time() {
        let embedder = Embedder::default();
        let texts: Vec<String> = (0..5).map(|i| format!("text {}", i)).collect();

        let progress_values = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_clone = progress_values.clone();

        let callback: super::super::ProgressCallback = Box::new(move |p| {
            progress_clone.lock().unwrap().push(p.completed);
        });

        let result = embedder.embed_batch_with_progress(&texts, Some(&callback));
        assert!(result.is_ok());

        let values = progress_values.lock().unwrap();
        assert_eq!(*values, vec![1, 2, 3, 4, 5]);
    }
}
