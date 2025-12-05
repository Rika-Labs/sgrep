use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueHint};

#[derive(Parser, Debug, Clone)]
#[command(name = "sgrep", version, about = "Lightning-fast semantic code search")]
pub struct Cli {
    /// Preferred device for embeddings (cpu|cuda|coreml). Also reads SGREP_DEVICE.
    #[arg(global = true, long, env = "SGREP_DEVICE")]
    pub device: Option<String>,

    /// Offline mode: skip all network fetches and fail fast if models/runtime are missing. Also reads SGREP_OFFLINE.
    #[arg(global = true, long, env = "SGREP_OFFLINE", default_value_t = false)]
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
        /// Disable cross-encoder reranking (enabled by default)
        #[arg(long)]
        no_rerank: bool,
        /// Oversample factor for reranking (default: 3, meaning fetch 3x candidates before rerank)
        #[arg(long, default_value_t = 3)]
        rerank_oversample: usize,
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
    fn cli_parses_no_rerank_flag() {
        let cli = Cli::parse_from(["sgrep", "search", "query", "--no-rerank"]);
        match cli.command {
            Commands::Search { no_rerank, .. } => {
                assert!(no_rerank, "Expected --no-rerank flag to be true");
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn cli_search_rerank_enabled_by_default() {
        let cli = Cli::parse_from(["sgrep", "search", "query"]);
        match cli.command {
            Commands::Search { no_rerank, .. } => {
                assert!(!no_rerank, "Expected no_rerank to be false by default");
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn cli_parses_rerank_oversample() {
        let cli = Cli::parse_from(["sgrep", "search", "query", "--rerank-oversample", "5"]);
        match cli.command {
            Commands::Search {
                rerank_oversample, ..
            } => {
                assert_eq!(rerank_oversample, 5);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn cli_rerank_oversample_default() {
        let cli = Cli::parse_from(["sgrep", "search", "query"]);
        match cli.command {
            Commands::Search {
                rerank_oversample, ..
            } => {
                assert_eq!(rerank_oversample, 3, "Expected default oversample to be 3");
            }
            _ => panic!("Expected Search command"),
        }
    }
}
