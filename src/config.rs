use std::path::PathBuf;
use std::{env, fs};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderType {
    #[default]
    Local,
    Voyage,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub provider: EmbeddingProviderType,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

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

        if config.embedding.provider == EmbeddingProviderType::Voyage
            && config.embedding.api_key.is_none()
        {
            anyhow::bail!(
                "Voyage provider selected but no API key provided.\n\
                 Add 'api_key = \"pa-...\"' to [embedding] section in {}",
                config_path.display()
            );
        }

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
# api_key = "pa-..."
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
        assert!(config.embedding.api_key.is_none());
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
    fn parse_voyage_config() {
        let toml = r#"
[embedding]
provider = "voyage"
api_key = "pa-test-key"
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.embedding.provider, EmbeddingProviderType::Voyage);
        assert_eq!(config.embedding.api_key, Some("pa-test-key".to_string()));
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
    fn load_voyage_without_api_key_errors() {
        let temp = std::env::temp_dir().join(format!("sgrep_config_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp).unwrap();
        let config_path = temp.join("config.toml");
        std::fs::write(
            &config_path,
            r#"
[embedding]
provider = "voyage"
"#,
        )
        .unwrap();

        env::set_var("SGREP_CONFIG", &config_path);
        let result = Config::load();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key"));
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
}
