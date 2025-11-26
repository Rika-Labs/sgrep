use std::env;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Result};

use crate::embedding::BatchEmbedder;

pub fn determine_batch_size(override_val: Option<usize>) -> usize {
    if let Some(v) = override_val {
        return v.clamp(16, 2048);
    }

    if let Ok(value) = env::var("SGREP_BATCH_SIZE") {
        if let Ok(parsed) = value.parse::<usize>() {
            return parsed.clamp(16, 2048);
        }
    }

    match env::var("SGREP_DEVICE")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "cuda" | "coreml" => 128,
        _ => 64,
    }
}

pub fn adjust_batch_size_for_progress(base: usize, total_chunks: usize) -> usize {
    if total_chunks == 0 {
        return base;
    }

    let estimated_batches = total_chunks.div_ceil(base);
    if estimated_batches >= 4 {
        return base;
    }

    let desired_batches = total_chunks.min(4);
    let progress_friendly = total_chunks.div_ceil(desired_batches);

    progress_friendly.max(1).min(base)
}

pub fn determine_token_budget() -> usize {
    env::var("SGREP_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(512, 20_000))
        .unwrap_or(6_000)
}

pub fn determine_embed_timeout() -> Duration {
    env::var("SGREP_EMBED_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or_else(|| Duration::from_secs(120))
}

pub fn embed_batch_with_timeout(
    embedder: Arc<dyn BatchEmbedder>,
    texts: Vec<String>,
    timeout: Duration,
) -> Result<Vec<Vec<f32>>> {
    let text_len = texts.len();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = embedder.embed_batch(&texts);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(res) => res,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!(
            "embedding batch timed out after {:?} ({} items)",
            timeout,
            text_len
        )),
        Err(err) => Err(anyhow!("embedding worker failed: {}", err)),
    }
}

pub fn estimate_tokens(text: &str) -> usize {
    let mut token_count = 0usize;
    let mut in_word = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            if !in_word {
                token_count += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if is_operator_or_punctuation(ch) {
                token_count += 1;
            }
        }
    }

    token_count.max(1)
}

pub fn is_operator_or_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '<'
            | '>'
            | ';'
            | ':'
            | ','
            | '.'
            | '='
            | '+'
            | '-'
            | '*'
            | '/'
            | '%'
            | '!'
            | '&'
            | '|'
            | '^'
            | '~'
            | '?'
            | '@'
            | '#'
            | '$'
            | '\\'
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::time::Instant;

    #[derive(Clone)]
    struct SlowEmbedder {
        delay: Duration,
    }

    impl BatchEmbedder for SlowEmbedder {
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            std::thread::sleep(self.delay);
            Ok(vec![vec![1.0, 0.0, 0.0, 0.0]])
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[test]
    #[serial]
    fn determine_batch_size_respects_env() {
        env::set_var("SGREP_BATCH_SIZE", "1024");
        assert_eq!(determine_batch_size(None), 1024);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[test]
    #[serial]
    fn determine_batch_size_prefers_override() {
        env::set_var("SGREP_BATCH_SIZE", "64");
        assert_eq!(determine_batch_size(Some(512)), 512);
        env::remove_var("SGREP_BATCH_SIZE");
    }

    #[test]
    #[serial]
    fn adjust_batch_size_reduces_single_batch_case() {
        let adjusted = adjust_batch_size_for_progress(256, 95);
        assert!(adjusted < 256);
        let batches = (95 + adjusted - 1) / adjusted;
        assert!(batches >= 2);
    }

    #[test]
    #[serial]
    fn adjust_batch_size_keeps_large_jobs_intact() {
        let adjusted = adjust_batch_size_for_progress(256, 10_000);
        assert_eq!(adjusted, 256);
    }

    #[test]
    #[serial]
    fn token_budget_clamps_min_and_max() {
        env::set_var("SGREP_TOKEN_BUDGET", "128");
        assert_eq!(determine_token_budget(), 512);

        env::set_var("SGREP_TOKEN_BUDGET", "50000");
        assert_eq!(determine_token_budget(), 20_000);

        env::remove_var("SGREP_TOKEN_BUDGET");
    }

    #[test]
    #[serial]
    fn determine_batch_size_respects_device_env() {
        env::set_var("SGREP_DEVICE", "coreml");
        assert_eq!(determine_batch_size(None), 128);
        env::set_var("SGREP_DEVICE", "cuda");
        assert_eq!(determine_batch_size(None), 128);
        env::remove_var("SGREP_DEVICE");
    }

    #[test]
    #[serial]
    fn embed_batch_with_timeout_times_out() {
        let embedder = Arc::new(SlowEmbedder {
            delay: Duration::from_millis(200),
        });
        let start = Instant::now();
        let result =
            embed_batch_with_timeout(embedder, vec!["slow".into()], Duration::from_millis(50));
        assert!(result.is_err());
        assert!(start.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    #[serial]
    fn embed_batch_with_timeout_succeeds() {
        let embedder = Arc::new(SlowEmbedder {
            delay: Duration::from_millis(5),
        });
        let result =
            embed_batch_with_timeout(embedder, vec!["ok".into()], Duration::from_millis(100));
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn determine_embed_timeout_reads_env() {
        env::set_var("SGREP_EMBED_TIMEOUT_SECS", "2");
        let timeout = determine_embed_timeout();
        assert_eq!(timeout, Duration::from_secs(2));
        env::remove_var("SGREP_EMBED_TIMEOUT_SECS");
    }

    #[test]
    fn estimate_tokens_counts_words_and_operators() {
        assert_eq!(estimate_tokens("fn hello() {}"), 6);
        assert_eq!(estimate_tokens("let x = 5 + 3;"), 7);
        assert_eq!(estimate_tokens(""), 1);
        assert_eq!(estimate_tokens("   "), 1);
    }

    #[test]
    fn estimate_tokens_handles_complex_code() {
        let code = "fn calculate(x: i32, y: i32) -> i32 { x + y }";
        let tokens = estimate_tokens(code);
        assert!(tokens > 10);
        assert!(tokens < 30);
    }

    #[test]
    fn estimate_tokens_counts_identifiers_once() {
        assert_eq!(estimate_tokens("hello_world"), 1);
        assert_eq!(estimate_tokens("hello world"), 2);
        assert_eq!(estimate_tokens("hello123world"), 1);
    }

    #[test]
    fn is_operator_or_punctuation_covers_common_operators() {
        assert!(is_operator_or_punctuation('('));
        assert!(is_operator_or_punctuation(')'));
        assert!(is_operator_or_punctuation('{'));
        assert!(is_operator_or_punctuation('}'));
        assert!(is_operator_or_punctuation(';'));
        assert!(is_operator_or_punctuation('='));
        assert!(is_operator_or_punctuation('+'));
        assert!(!is_operator_or_punctuation('a'));
        assert!(!is_operator_or_punctuation(' '));
        assert!(!is_operator_or_punctuation('\n'));
    }

    #[test]
    fn adjust_batch_size_returns_base_for_zero_chunks() {
        let adjusted = adjust_batch_size_for_progress(256, 0);
        assert_eq!(adjusted, 256);
    }

    #[test]
    fn adjust_batch_size_for_very_small_jobs() {
        let adjusted = adjust_batch_size_for_progress(64, 1);
        assert!(adjusted <= 64);
        assert!(adjusted >= 1);

        let adjusted2 = adjust_batch_size_for_progress(64, 2);
        assert!(adjusted2 <= 64);
        assert!(adjusted2 >= 1);

        let adjusted3 = adjust_batch_size_for_progress(64, 3);
        assert!(adjusted3 <= 64);
        assert!(adjusted3 >= 1);
    }
}
