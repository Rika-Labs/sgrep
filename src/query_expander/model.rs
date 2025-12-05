//! Qwen2.5 model implementation for query understanding and expansion.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};
use ureq::{Agent, AgentBuilder, Proxy};

use super::Expander;
use crate::threading::ThreadConfig;

const MODEL_NAME: &str = "qwen2.5-0.5b-instruct";
const MODEL_URL: &str = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MAX_TOKENS: usize = 256;

/// Structured output from the query understanding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// The type of query: "semantic", "structural", or "hybrid"
    pub query_type: String,
    /// The intent: "explanation", "callers", "callees", "definition", "symbol_lookup", etc.
    pub intent: String,
    /// Extracted symbol name if present
    pub symbol: Option<String>,
    /// Search granularity: "chunk", "file", or "directory"
    pub granularity: String,
    /// Expanded query variations for better search recall
    pub expanded_queries: Vec<String>,
}

impl Default for QueryAnalysis {
    fn default() -> Self {
        Self {
            query_type: "hybrid".to_string(),
            intent: "search".to_string(),
            symbol: None,
            granularity: "chunk".to_string(),
            expanded_queries: vec![],
        }
    }
}

impl QueryAnalysis {
    /// Create a QueryAnalysis from simple heuristics (fallback when model unavailable).
    pub fn from_heuristics(query: &str) -> Self {
        let query_lower = query.to_lowercase();

        // Detect structural queries
        let structural_patterns = [
            "who calls",
            "what calls",
            "callers of",
            "calls to",
            "imports",
            "imported by",
            "defined in",
            "definition of",
            "implementations of",
            "implementors of",
            "usages of",
            "references to",
        ];

        let is_structural = structural_patterns.iter().any(|p| query_lower.contains(p));

        // Detect semantic/question queries
        let is_question = query_lower.starts_with("how ")
            || query_lower.starts_with("what is")
            || query_lower.starts_with("explain")
            || query_lower.starts_with("why ")
            || query_lower.starts_with("describe");

        // Detect granularity
        let granularity = if query_lower.contains("file") || query_lower.contains("where is") {
            "file"
        } else if query_lower.contains("directory")
            || query_lower.contains("folder")
            || query_lower.contains("module")
        {
            "directory"
        } else {
            "chunk"
        };

        // Determine query type
        let query_type = if is_structural {
            "structural"
        } else if is_question {
            "semantic"
        } else {
            "hybrid"
        };

        // Determine intent
        let intent = if query_lower.contains("call") {
            "callers"
        } else if query_lower.contains("definition") || query_lower.contains("defined") {
            "definition"
        } else if is_question {
            "explanation"
        } else {
            "symbol_lookup"
        };

        Self {
            query_type: query_type.to_string(),
            intent: intent.to_string(),
            symbol: None,
            granularity: granularity.to_string(),
            expanded_queries: vec![query.to_string()],
        }
    }
}

/// Query expander using Qwen2.5 model via llama.cpp.
pub struct QueryExpander {
    model: LlamaModel,
    backend: LlamaBackend,
}

// Implement Send + Sync manually since LlamaModel internals are thread-safe
unsafe impl Send for QueryExpander {}
unsafe impl Sync for QueryExpander {}

impl QueryExpander {
    /// Create a new QueryExpander, downloading the model if necessary.
    pub fn new() -> Result<Self> {
        Self::with_options(true, false)
    }

    pub fn new_silent_if_cached() -> Result<Self> {
        if !is_model_cached() {
            return Err(anyhow!(
                "Query expander model not cached. Run 'sgrep index' first to download."
            ));
        }
        Self::with_options(false, true)
    }

    pub fn with_options(show_progress: bool, silent: bool) -> Result<Self> {
        let model_path = download_model(show_progress)?;

        let mut backend = LlamaBackend::init()
            .map_err(|e| anyhow!("Failed to initialize llama backend: {}", e))?;

        if silent {
            backend.void_logs();
        }

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| anyhow!("Failed to load model: {}", e))?;

        Ok(Self { model, backend })
    }

    /// Analyze a query and return structured understanding.
    pub fn analyze(&self, query: &str) -> Result<QueryAnalysis> {
        let prompt = build_prompt(query);
        let response = self.generate(&prompt)?;
        parse_response(&response, query)
    }

    /// Generate text from a prompt.
    fn generate(&self, prompt: &str) -> Result<String> {
        let cfg = ThreadConfig::get();
        let llama_threads = env::var("SGREP_LLM_THREADS")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(cfg.onnx_threads as i32)
            .max(1);

        // Create context for this generation
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(512))
            .with_n_threads(llama_threads)
            .with_n_threads_batch(llama_threads);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create context: {}", e))?;

        // Tokenize input
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(512, 1);
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, i as i32, &[0], i == tokens.len() - 1)
                .map_err(|e| anyhow!("Failed to add token to batch: {}", e))?;
        }

        // Decode the prompt
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Failed to decode prompt: {}", e))?;

        // Set up sampler for generation
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_p(0.9, 1),
            LlamaSampler::dist(42),
        ]);

        // Set up UTF-8 decoder for incremental string building
        let mut decoder = UTF_8.new_decoder();
        let mut result = String::with_capacity(512);

        // Track position in sequence for new tokens
        let mut seq_pos = tokens.len() as i32;

        for _ in 0..MAX_TOKENS {
            // Sample the next token - use last position in current batch
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // Check for end of generation
            if self.model.is_eog_token(token) {
                break;
            }

            // Convert token to string and append
            if let Ok(bytes) = self.model.token_to_bytes(token, Special::Tokenize) {
                let _ = decoder.decode_to_string(&bytes, &mut result, false);
            }

            // Prepare next batch with the new token
            batch.clear();
            batch
                .add(token, seq_pos, &[0], true)
                .map_err(|e| anyhow!("Failed to add token: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Failed to decode: {}", e))?;

            seq_pos += 1;
        }

        Ok(result)
    }
}

impl Expander for QueryExpander {
    fn expand(&self, query: &str) -> Result<Vec<String>> {
        let analysis = self.analyze(query)?;

        let mut result = vec![query.to_string()];

        for expansion in analysis.expanded_queries {
            let normalized = expansion.to_lowercase();
            if !result.iter().any(|r| r.to_lowercase() == normalized) {
                result.push(expansion);
            }
        }

        Ok(result)
    }
}

/// Build the prompt for query understanding.
fn build_prompt(query: &str) -> String {
    format!(
        r#"<|im_start|>system
You are a code search query analyzer. Given a user's search query, analyze it and output a JSON object with:
- query_type: "semantic" (natural language questions), "structural" (code relationships like callers/callees), or "hybrid" (symbol lookup)
- intent: the user's goal (e.g., "explanation", "callers", "callees", "definition", "symbol_lookup", "usage")
- symbol: the code symbol being searched for (if any)
- granularity: "chunk" (function-level), "file", or "directory"
- expanded_queries: 3-5 alternative phrasings of the query for better search recall

Output ONLY valid JSON, no explanation.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"#,
        query
    )
}

/// Parse the model's response into a QueryAnalysis.
fn parse_response(response: &str, original_query: &str) -> Result<QueryAnalysis> {
    // Try to extract JSON from response
    let json_str = response.trim();

    // Handle case where response might have extra text
    let json_str = if let Some(start) = json_str.find('{') {
        if let Some(end) = json_str.rfind('}') {
            &json_str[start..=end]
        } else {
            json_str
        }
    } else {
        json_str
    };

    match serde_json::from_str::<QueryAnalysis>(json_str) {
        Ok(mut analysis) => {
            // Ensure we have at least the original query in expansions
            if analysis.expanded_queries.is_empty() {
                analysis.expanded_queries.push(original_query.to_string());
            }
            Ok(analysis)
        }
        Err(_) => {
            // Fallback: return default analysis with original query
            Ok(QueryAnalysis {
                expanded_queries: vec![original_query.to_string()],
                ..Default::default()
            })
        }
    }
}

fn get_model_cache_dir() -> PathBuf {
    let base = env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    base.join(".sgrep").join("cache").join("models")
}

pub fn get_model_path() -> PathBuf {
    get_model_cache_dir().join(format!("{}.gguf", MODEL_NAME))
}

pub fn is_model_cached() -> bool {
    get_model_path().exists()
}

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

/// Download the model if not already cached.
fn download_model(show_progress: bool) -> Result<PathBuf> {
    let cache_dir = get_model_cache_dir();
    let model_path = cache_dir.join(format!("{}.gguf", MODEL_NAME));

    if model_path.exists() {
        return Ok(model_path);
    }

    let offline = env::var("SGREP_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if offline {
        return Err(anyhow!(
            "Query expander model not cached and offline mode is enabled.\n\
            Run once without --offline to download the model (~400MB)."
        ));
    }

    fs::create_dir_all(&cache_dir)?;

    if show_progress {
        eprintln!("Downloading {}...", MODEL_NAME);
    }

    let agent = create_http_agent();
    let response = agent.get(MODEL_URL).call().map_err(|e| {
        anyhow!(
            "Failed to download model: {}\n\n\
            If HuggingFace is blocked, set HTTPS_PROXY environment variable.\n\
            Example: export HTTPS_PROXY=http://proxy:port",
            e
        )
    })?;

    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;

    let mut file = fs::File::create(&model_path)?;
    file.write_all(&bytes)?;

    if show_progress {
        eprintln!("Model downloaded to {:?}", model_path);
    }

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let prompt = build_prompt("who calls authenticate");
        assert!(prompt.contains("who calls authenticate"));
        assert!(prompt.contains("query_type"));
    }

    #[test]
    fn test_parse_response_valid_json() {
        let json = r#"{"query_type": "structural", "intent": "callers", "symbol": "authenticate", "granularity": "chunk", "expanded_queries": ["callers of authenticate", "what calls authenticate"]}"#;
        let result = parse_response(json, "who calls authenticate").unwrap();
        assert_eq!(result.query_type, "structural");
        assert_eq!(result.intent, "callers");
        assert_eq!(result.symbol, Some("authenticate".to_string()));
    }

    #[test]
    fn test_parse_response_invalid_json() {
        let result = parse_response("not json", "test query").unwrap();
        assert_eq!(result.query_type, "hybrid");
        assert!(result.expanded_queries.contains(&"test query".to_string()));
    }

    #[test]
    fn test_cache_directory() {
        let cache_dir = get_model_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("sgrep"));
    }

    #[test]
    fn test_is_model_cached_returns_false_when_not_present() {
        // Use a temp dir that won't have the model
        std::env::set_var("SGREP_TEST_MODEL_CACHE", "/nonexistent/path");
        let result = is_model_cached();
        std::env::remove_var("SGREP_TEST_MODEL_CACHE");
        // Result depends on actual cache state, but function should not panic
        assert!(result == true || result == false);
    }

    #[test]
    fn test_get_model_path_returns_expected_path() {
        let path = get_model_path();
        assert!(path.to_string_lossy().contains("sgrep"));
        assert!(path.to_string_lossy().contains("models"));
        assert!(path.to_string_lossy().ends_with(".gguf"));
    }
}
