//! Modal service auto-deployer.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use super::MODAL_SERVICE_PY;

const HEALTH_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Serialize, Deserialize)]
struct EndpointCache {
    embed_url: String,
    health_url: String,
}

pub struct ModalDeployer {
    token_id: Option<String>,
    token_secret: Option<String>,
    cache_path: PathBuf,
}

impl ModalDeployer {
    pub fn new(token_id: Option<String>, token_secret: Option<String>) -> Self {
        let cache_path = Self::default_cache_path();
        Self {
            token_id,
            token_secret,
            cache_path,
        }
    }

    fn default_cache_path() -> PathBuf {
        if let Ok(home) = std::env::var("SGREP_HOME") {
            return PathBuf::from(home).join("modal_cache.json");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".sgrep").join("modal_cache.json");
        }
        PathBuf::from(".sgrep").join("modal_cache.json")
    }

    pub fn check_modal_cli(&self) -> Result<()> {
        if Command::new("modal").arg("--version").output().is_err() {
            eprintln!("[info] Modal CLI not found. Installing...");
            let pip = if Command::new("pip3").arg("--version").output().is_ok() {
                "pip3"
            } else {
                "pip"
            };

            let status = Command::new(pip)
                .args(["install", "modal"])
                .status()
                .context("Failed to run pip install modal")?;

            if !status.success() {
                return Err(anyhow!(
                    "Failed to install Modal CLI. Install manually with: pip install modal"
                ));
            }
            eprintln!("[info] Modal CLI installed.");
        }

        if self.token_id.is_some() && self.token_secret.is_some() {
            return Ok(());
        }

        let is_authenticated = Command::new("modal")
            .args(["token", "show"])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if !is_authenticated {
            return Err(anyhow!(
                "Modal CLI not authenticated.\n\
                 Either:\n\
                 1. Add token_id and token_secret to your config (~/.sgrep/config.toml):\n\
                    [modal]\n\
                    token_id = \"your-token-id\"\n\
                    token_secret = \"your-token-secret\"\n\n\
                 2. Or run: modal token new\n\n\
                 Get tokens from: https://modal.com/settings"
            ));
        }

        Ok(())
    }

    fn modal_command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new("modal");
        cmd.args(args);
        if let Some(token_id) = &self.token_id {
            cmd.env("MODAL_TOKEN_ID", token_id);
        }
        if let Some(token_secret) = &self.token_secret {
            cmd.env("MODAL_TOKEN_SECRET", token_secret);
        }
        cmd
    }

    pub fn check_health(&self, health_url: &str) -> Result<bool> {
        let client = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(HEALTH_TIMEOUT_SECS))
            .build();

        match client.get(health_url).call() {
            Ok(resp) if resp.status() == 200 => Ok(true),
            Ok(_) => Ok(false),
            Err(_) => Ok(false),
        }
    }

    fn load_cache(&self) -> Option<EndpointCache> {
        let contents = fs::read_to_string(&self.cache_path).ok()?;
        serde_json::from_str(&contents).ok()
    }

    fn save_cache(&self, cache: &EndpointCache) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(cache)?;
        fs::write(&self.cache_path, contents)?;
        Ok(())
    }

    fn deploy(&self) -> Result<EndpointCache> {
        self.check_modal_cli()?;

        let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
        let service_path = temp_dir.path().join("service.py");
        fs::write(&service_path, MODAL_SERVICE_PY).context("Failed to write service.py")?;

        eprintln!("[info] Deploying Modal service to modal.com...");

        let service_path_str = service_path
            .to_str()
            .ok_or_else(|| anyhow!("Service path contains invalid UTF-8"))?;

        let output = self
            .modal_command(&["deploy", service_path_str])
            .output()
            .context("Failed to run modal deploy")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Modal deploy failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_deploy_output(&stdout)
    }

    fn parse_deploy_output(&self, output: &str) -> Result<EndpointCache> {
        let mut embed_url = None;
        let mut health_url = None;

        for line in output.lines() {
            if line.contains("modal.run") {
                if let Some(url) = line.split_whitespace().find(|s| s.starts_with("https://")) {
                    let url = url.trim();
                    if line.contains("embed") && !line.contains("health") {
                        embed_url = Some(url.to_string());
                    } else if line.contains("health") {
                        health_url = Some(url.to_string());
                    }
                }
            }
        }

        let embed_url = embed_url.ok_or_else(|| anyhow!("Could not find embed endpoint URL"))?;
        let health_url = health_url.ok_or_else(|| anyhow!("Could not find health endpoint URL"))?;

        Ok(EndpointCache {
            embed_url,
            health_url,
        })
    }

    pub fn ensure_deployed(&self) -> Result<(String, bool)> {
        if let Some(cache) = self.load_cache() {
            eprintln!("[info] Checking cached Modal endpoint health...");
            if self.check_health(&cache.health_url).unwrap_or(false) {
                return Ok((cache.embed_url, true));
            }
            eprintln!("[info] Cached endpoint unhealthy, redeploying...");
        }

        let cache = self.deploy()?;
        self.save_cache(&cache)?;
        Ok((cache.embed_url, false))
    }

    #[allow(dead_code)]
    pub fn get_embed_endpoint(&self) -> Result<String> {
        let (embed_url, _) = self.ensure_deployed()?;
        Ok(embed_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cache_path_is_valid() {
        let path = ModalDeployer::default_cache_path();
        assert!(path.to_string_lossy().contains("modal_cache.json"));
    }

    #[test]
    fn new_deployer_has_correct_fields() {
        let deployer = ModalDeployer::new(
            Some("token-id".to_string()),
            Some("token-secret".to_string()),
        );
        assert_eq!(deployer.token_id, Some("token-id".to_string()));
        assert_eq!(deployer.token_secret, Some("token-secret".to_string()));
    }

    #[test]
    fn parse_deploy_output_extracts_urls() {
        let deployer = ModalDeployer::new(None, None);
        let output = r#"
Creating objects...
Created fastapi_endpoint embed at https://user--sgrep-offload-embed.modal.run
Created fastapi_endpoint health at https://user--sgrep-offload-health.modal.run
"#;
        let cache = deployer.parse_deploy_output(output).unwrap();
        assert_eq!(
            cache.embed_url,
            "https://user--sgrep-offload-embed.modal.run"
        );
        assert_eq!(
            cache.health_url,
            "https://user--sgrep-offload-health.modal.run"
        );
    }

    #[test]
    fn check_health_returns_false_for_invalid_url() {
        let deployer = ModalDeployer::new(None, None);
        let result = deployer.check_health("http://localhost:99999/health");
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn endpoint_cache_serialization() {
        let cache = EndpointCache {
            embed_url: "https://embed.modal.run".to_string(),
            health_url: "https://health.modal.run".to_string(),
        };

        let json = serde_json::to_string(&cache).unwrap();
        let parsed: EndpointCache = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.embed_url, cache.embed_url);
    }
}
