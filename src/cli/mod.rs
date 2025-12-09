use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueHint};

/// Parse bool from string, accepting "1", "true", "0", "false" (case-insensitive)
fn parse_bool_flexible(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => Err(format!(
            "invalid value '{}': expected 1, true, 0, or false",
            s
        )),
    }
}

/// Parse optional bool from string for flags that can be omitted, used as flag, or given a value
fn parse_optional_bool_flexible(s: &str) -> Result<bool, String> {
    parse_bool_flexible(s)
}

#[derive(Parser, Debug, Clone)]
#[command(name = "sgrep", version, about = "Lightning-fast semantic code search")]
pub struct Cli {
    /// Preferred device for embeddings (cpu|cuda|coreml). Also reads SGREP_DEVICE.
    #[arg(global = true, long, env = "SGREP_DEVICE")]
    pub device: Option<String>,

    /// Offline mode: skip all network fetches and fail fast if models/runtime are missing. Also reads SGREP_OFFLINE.
    #[arg(global = true, long, env = "SGREP_OFFLINE", default_value_t = false, value_parser = parse_bool_flexible)]
    pub offline: bool,

    /// Maximum threads for parallel operations (0 = auto). Also reads SGREP_MAX_THREADS.
    #[arg(global = true, long = "threads", env = "SGREP_MAX_THREADS")]
    pub max_threads: Option<usize>,

    /// CPU preset: auto (75%), low (25%), medium (50%), high (100%), background (25%). Also reads SGREP_CPU_PRESET.
    #[arg(global = true, long = "cpu-preset", env = "SGREP_CPU_PRESET")]
    pub cpu_preset: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Search for code using natural language
    Search {
        /// Query such as "where do we handle authentication?"
        query: String,
        /// Repository path (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        path: PathBuf,
        /// Maximum results to return
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
        /// Show extra context around matches
        #[arg(short, long)]
        context: bool,
        /// Restrict to globs (repeatable)
        #[arg(long)]
        glob: Vec<String>,
        /// Simple metadata filters like lang=rust
        #[arg(long)]
        filters: Vec<String>,
        /// Emit structured JSON output for agents
        #[arg(long)]
        json: bool,
        /// Show debug info (scores, timing)
        #[arg(long)]
        debug: bool,
        /// Offload embeddings to Modal.dev (auto-deploys if needed)
        #[arg(
            long,
            env = "SGREP_OFFLOAD",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        offload: Option<bool>,
        /// Store/query index from remote vector DB (turbopuffer/pinecone)
        #[arg(
            long,
            env = "SGREP_REMOTE",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        remote: Option<bool>,
    },
    /// Index a repository for semantic search
    Index {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Force full re-index
        #[arg(short, long)]
        force: bool,
        /// Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.
        #[arg(
            long,
            env = "SGREP_BATCH_SIZE",
            value_parser = clap::value_parser!(usize),
            help = "Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE."
        )]
        batch_size: Option<usize>,
        /// Emit per-phase timings and throughput stats
        #[arg(long, default_value_t = false)]
        profile: bool,
        /// Show index statistics without rebuilding
        #[arg(long, default_value_t = false)]
        stats: bool,
        /// Output stats as JSON (only with --stats)
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Offload embeddings to Modal.dev (auto-deploys if needed)
        #[arg(
            long,
            env = "SGREP_OFFLOAD",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        offload: Option<bool>,
        /// Store index in remote vector DB (turbopuffer/pinecone)
        #[arg(
            long,
            env = "SGREP_REMOTE",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        remote: Option<bool>,
        /// Run index in detached/background mode
        #[arg(short = 'd', long, default_value_t = false)]
        detach: bool,
    },
    /// Watch a repository and keep the index fresh
    Watch {
        /// Repository path
        #[arg(value_hint = ValueHint::DirPath)]
        path: Option<PathBuf>,
        /// Debounce window in milliseconds
        #[arg(long, default_value_t = 500)]
        debounce_ms: u64,
        /// Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE.
        #[arg(
            long,
            env = "SGREP_BATCH_SIZE",
            value_parser = clap::value_parser!(usize),
            help = "Override embedding batch size (16-2048). Also reads SGREP_BATCH_SIZE."
        )]
        batch_size: Option<usize>,
        /// Offload embeddings to Modal.dev (auto-deploys if needed)
        #[arg(
            long,
            env = "SGREP_OFFLOAD",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        offload: Option<bool>,
        /// Store index in remote vector DB (turbopuffer/pinecone)
        #[arg(
            long,
            env = "SGREP_REMOTE",
            num_args = 0..=1,
            default_missing_value = "true",
            value_parser = parse_optional_bool_flexible
        )]
        remote: Option<bool>,
        /// Run watch in detached/background mode
        #[arg(short = 'd', long, default_value_t = false)]
        detach: bool,
    },
    /// Show or create configuration
    Config {
        /// Create a default config file if none exists
        #[arg(long)]
        init: bool,
        /// Show the model cache directory path
        #[arg(long)]
        show_model_dir: bool,
        /// Verify model files are present
        #[arg(long)]
        verify_model: bool,
    },
}

pub fn resolve_repo_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(p) => Ok(p),
        None => std::env::current_dir().context("Failed to determine current directory"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_offline_flag() {
        let cli = Cli::parse_from(["sgrep", "--offline", "index", "."]);
        assert!(cli.offline);
        matches!(cli.command, Commands::Index { .. });
    }

    #[test]
    fn resolve_repo_path_defaults_to_cwd() {
        let cwd = std::env::current_dir().unwrap();
        let resolved = resolve_repo_path(None).unwrap();
        assert_eq!(resolved, cwd);
    }

    #[test]
    fn cli_parses_config_show_model_dir() {
        let cli = Cli::parse_from(["sgrep", "config", "--show-model-dir"]);
        match cli.command {
            Commands::Config {
                show_model_dir,
                verify_model,
                init,
            } => {
                assert!(show_model_dir);
                assert!(!verify_model);
                assert!(!init);
            }
            _ => panic!("Expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_verify_model() {
        let cli = Cli::parse_from(["sgrep", "config", "--verify-model"]);
        match cli.command {
            Commands::Config {
                show_model_dir,
                verify_model,
                init,
            } => {
                assert!(!show_model_dir);
                assert!(verify_model);
                assert!(!init);
            }
            _ => panic!("Expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_init() {
        let cli = Cli::parse_from(["sgrep", "config", "--init"]);
        match cli.command {
            Commands::Config {
                show_model_dir,
                verify_model,
                init,
            } => {
                assert!(!show_model_dir);
                assert!(!verify_model);
                assert!(init);
            }
            _ => panic!("Expected Config command"),
        }
    }

    #[test]
    fn cli_parses_index_stats_flag() {
        let cli = Cli::parse_from(["sgrep", "index", "--stats"]);
        match cli.command {
            Commands::Index { stats, .. } => assert!(stats),
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_parses_index_stats_with_json() {
        let cli = Cli::parse_from(["sgrep", "index", "--stats", "--json"]);
        match cli.command {
            Commands::Index { stats, json, .. } => {
                assert!(stats);
                assert!(json);
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_parses_index_offload_flag() {
        let cli = Cli::parse_from(["sgrep", "index", "--offload"]);
        match cli.command {
            Commands::Index { offload, .. } => {
                assert_eq!(
                    offload,
                    Some(true),
                    "Expected --offload flag to be Some(true)"
                );
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_index_offload_default_is_false() {
        let cli = Cli::parse_from(["sgrep", "index", "."]);
        match cli.command {
            Commands::Index { offload, .. } => {
                assert!(offload.is_none(), "Expected offload to be None by default");
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_parses_watch_offload_flag() {
        let cli = Cli::parse_from(["sgrep", "watch", "--offload"]);
        match cli.command {
            Commands::Watch { offload, .. } => {
                assert_eq!(
                    offload,
                    Some(true),
                    "Expected --offload flag to be Some(true)"
                );
            }
            _ => panic!("Expected Watch command"),
        }
    }

    #[test]
    fn cli_parses_search_offload_flag() {
        let cli = Cli::parse_from(["sgrep", "search", "query", "--offload"]);
        match cli.command {
            Commands::Search { offload, .. } => {
                assert_eq!(
                    offload,
                    Some(true),
                    "Expected --offload flag to be Some(true)"
                );
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn cli_parses_index_remote_flag() {
        let cli = Cli::parse_from(["sgrep", "index", "--remote"]);
        match cli.command {
            Commands::Index { remote, .. } => {
                assert_eq!(
                    remote,
                    Some(true),
                    "Expected --remote flag to be Some(true)"
                );
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_parses_index_remote_false() {
        let cli = Cli::parse_from(["sgrep", "index", "--remote=false"]);
        match cli.command {
            Commands::Index { remote, .. } => {
                assert_eq!(
                    remote,
                    Some(false),
                    "Expected --remote=false to be Some(false)"
                );
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_index_remote_default_is_none() {
        let cli = Cli::parse_from(["sgrep", "index", "."]);
        match cli.command {
            Commands::Index { remote, .. } => {
                assert!(remote.is_none(), "Expected remote to be None by default");
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn cli_parses_watch_remote_flag() {
        let cli = Cli::parse_from(["sgrep", "watch", "--remote"]);
        match cli.command {
            Commands::Watch { remote, .. } => {
                assert_eq!(
                    remote,
                    Some(true),
                    "Expected --remote flag to be Some(true)"
                );
            }
            _ => panic!("Expected Watch command"),
        }
    }

    #[test]
    fn cli_parses_search_remote_flag() {
        let cli = Cli::parse_from(["sgrep", "search", "query", "--remote"]);
        match cli.command {
            Commands::Search { remote, .. } => {
                assert_eq!(
                    remote,
                    Some(true),
                    "Expected --remote flag to be Some(true)"
                );
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn cli_parses_index_offload_and_remote_flags() {
        let cli = Cli::parse_from(["sgrep", "index", "--offload", "--remote"]);
        match cli.command {
            Commands::Index {
                offload, remote, ..
            } => {
                assert_eq!(
                    offload,
                    Some(true),
                    "Expected --offload flag to be Some(true)"
                );
                assert_eq!(
                    remote,
                    Some(true),
                    "Expected --remote flag to be Some(true)"
                );
            }
            _ => panic!("Expected Index command"),
        }
    }
}
