use std::path::PathBuf;
use std::{env, fs};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderType {
    #[default]
    Local,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub provider: EmbeddingProviderType,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    #[serde(default)]
    pub embedding: EmbeddingConfig,
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
        assert_eq!(Config::config_path(), PathBuf::from(home_path).join("config.toml"));
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
}
