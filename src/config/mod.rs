use std::path::PathBuf;
use std::{env, fs};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderType {
    #[default]
    Local,
    Modal,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub provider: EmbeddingProviderType,
}

/// Configuration for Modal.dev offload
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModalConfig {
    /// API token for Modal endpoint authentication
    pub api_token: Option<String>,
    /// GPU tier: "budget" (T4), "balanced" (A10G), "high" (L40S)
    #[serde(default = "default_gpu_tier")]
    pub gpu_tier: String,
    /// Embedding dimension (384-4096)
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    /// Batch size for embedding requests
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Cached endpoint URL (auto-populated after first deploy)
    pub endpoint: Option<String>,
}

fn default_gpu_tier() -> String {
    "high".to_string()
}

fn default_dimension() -> usize {
    4096
}

fn default_batch_size() -> usize {
    32
}

/// Configuration for Turbopuffer remote storage
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TurbopufferConfig {
    /// Turbopuffer API key
    pub api_key: Option<String>,
    /// Region (default: gcp-us-central1)
    #[serde(default = "default_region")]
    pub region: String,
    /// Custom namespace prefix (default: "sgrep")
    #[serde(default = "default_namespace_prefix")]
    pub namespace_prefix: String,
}

fn default_region() -> String {
    "gcp-us-central1".to_string()
}

fn default_namespace_prefix() -> String {
    "sgrep".to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub modal: ModalConfig,
    #[serde(default)]
    pub turbopuffer: TurbopufferConfig,
}

#[allow(dead_code)]
impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;

        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;

        Ok(config)
    }

    pub fn config_path() -> PathBuf {
        if let Ok(path) = env::var("SGREP_CONFIG") {
            return PathBuf::from(path);
        }

        if let Ok(home) = env::var("SGREP_HOME") {
            return PathBuf::from(home).join("config.toml");
        }

        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(".sgrep").join("config.toml");
        }

        PathBuf::from(".sgrep").join("config.toml")
    }

    pub fn create_default_config() -> Result<PathBuf> {
        let config_path = Self::config_path();

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }

        let default_config = r#"[embedding]
provider = "local"
"#;

        fs::write(&config_path, default_config)
            .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;

        Ok(config_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;

    #[test]
    fn default_config_uses_local_provider() {
        let config = Config::default();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Local);
    }

    #[test]
    fn parse_local_config() {
        let toml = r#"
[embedding]
provider = "local"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Local);
    }

    #[test]
    fn parse_empty_config_defaults_to_local() {
        let toml = "";
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Local);
    }

    #[test]
    #[serial]
    fn load_missing_config_returns_default() {
        let temp = std::env::temp_dir().join(format!("sgrep_config_test_{}", uuid::Uuid::new_v4()));
        env::set_var("SGREP_CONFIG", temp.join("nonexistent.toml"));
        let config = Config::load().unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Local);
        env::remove_var("SGREP_CONFIG");
    }

    #[test]
    #[serial]
    fn config_path_respects_env() {
        let custom_path = "/custom/path/config.toml";
        env::set_var("SGREP_CONFIG", custom_path);
        assert_eq!(Config::config_path(), PathBuf::from(custom_path));
        env::remove_var("SGREP_CONFIG");
    }

    #[test]
    #[serial]
    fn config_path_uses_sgrep_home() {
        env::remove_var("SGREP_CONFIG");
        let home_path = "/custom/sgrep/home";
        env::set_var("SGREP_HOME", home_path);
        assert_eq!(
            Config::config_path(),
            PathBuf::from(home_path).join("config.toml")
        );
        env::remove_var("SGREP_HOME");
    }

    #[test]
    #[serial]
    fn load_valid_config_file() {
        let temp = std::env::temp_dir().join(format!("sgrep_cfg_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp).unwrap();
        let config_file = temp.join("config.toml");
        std::fs::write(&config_file, "[embedding]\nprovider = \"local\"\n").unwrap();
        env::set_var("SGREP_CONFIG", &config_file);

        let config = Config::load().unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Local);

        env::remove_var("SGREP_CONFIG");
        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    #[serial]
    fn create_default_config_creates_file() {
        let temp = std::env::temp_dir().join(format!("sgrep_create_{}", uuid::Uuid::new_v4()));
        let config_file = temp.join("config.toml");
        env::set_var("SGREP_CONFIG", &config_file);

        let path = Config::create_default_config().unwrap();
        assert!(path.exists());
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("local"));

        env::remove_var("SGREP_CONFIG");
        std::fs::remove_dir_all(&temp).ok();
    }

    // Modal config tests
    #[test]
    fn parse_modal_config() {
        let toml = r#"
[embedding]
provider = "modal"

[modal]
api_token = "test-token"
gpu_tier = "balanced"
dimension = 1024
batch_size = 64
endpoint = "https://example.modal.run"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Modal);
        assert_eq!(config.modal.api_token, Some("test-token".to_string()));
        assert_eq!(config.modal.gpu_tier, "balanced");
        assert_eq!(config.modal.dimension, 1024);
        assert_eq!(config.modal.batch_size, 64);
        assert_eq!(
            config.modal.endpoint,
            Some("https://example.modal.run".to_string())
        );
    }

    #[test]
    fn modal_config_defaults() {
        let toml = r#"
[modal]
api_token = "test-token"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.modal.gpu_tier, "high");
        assert_eq!(config.modal.dimension, 4096);
        assert_eq!(config.modal.batch_size, 32);
        assert_eq!(config.modal.endpoint, None);
    }

    #[test]
    fn empty_config_has_modal_defaults() {
        let config = Config::default();
        assert_eq!(config.modal.gpu_tier, "");
        assert_eq!(config.modal.dimension, 0);
    }

    // Turbopuffer config tests
    #[test]
    fn parse_turbopuffer_config() {
        let toml = r#"
[turbopuffer]
api_key = "tpuf_test_key"
region = "aws-us-east-1"
namespace_prefix = "myproject"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.turbopuffer.api_key, Some("tpuf_test_key".to_string()));
        assert_eq!(config.turbopuffer.region, "aws-us-east-1");
        assert_eq!(config.turbopuffer.namespace_prefix, "myproject");
    }

    #[test]
    fn turbopuffer_config_defaults() {
        let toml = r#"
[turbopuffer]
api_key = "tpuf_test_key"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.turbopuffer.region, "gcp-us-central1");
        assert_eq!(config.turbopuffer.namespace_prefix, "sgrep");
    }

    // Combined config test
    #[test]
    fn parse_full_config() {
        let toml = r#"
[embedding]
provider = "modal"

[modal]
api_token = "modal-token"
gpu_tier = "high"
dimension = 4096

[turbopuffer]
api_key = "tpuf-key"
region = "gcp-us-central1"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Modal);
        assert_eq!(config.modal.api_token, Some("modal-token".to_string()));
        assert_eq!(config.turbopuffer.api_key, Some("tpuf-key".to_string()));
    }
}
