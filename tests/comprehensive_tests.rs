// Comprehensive integration tests for sgrep
//
// These tests aim to achieve 90%+ code coverage by testing:
// - Edge cases and error conditions
// - All public APIs
// - Integration between modules
// - Performance-critical paths

use chrono::Utc;
use sgrep::chunker::{chunk_file, CodeChunk};
use sgrep::embedding::{BatchEmbedder, Embedder, PooledEmbedder};
use sgrep::fts::{build_globset, extract_keywords, glob_matches, keyword_score, matches_filters};
use sgrep::indexer::{IndexRequest, Indexer};
use sgrep::search::{SearchRequest, Searcher};
use sgrep::store::{IndexMetadata, IndexStore, RepositoryIndex};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use uuid::Uuid;

// Helper functions

fn create_test_repo() -> TempDir {
    let temp = tempfile::tempdir().unwrap();
    let base = temp.path();

    // Create diverse file types
    fs::create_dir_all(base.join("src")).unwrap();
    fs::create_dir_all(base.join("tests")).unwrap();
    fs::create_dir_all(base.join("config")).unwrap();

    // Rust files
    fs::write(
        base.join("src/main.rs"),
        r#"fn main() {
    println!("Hello, world!");
}

fn authenticate_user(username: &str, password: &str) -> bool {
    // Authentication logic here
    true
}
"#,
    )
    .unwrap();

    fs::write(
        base.join("src/lib.rs"),
        r#"pub mod auth;
pub mod database;

pub fn process_request(req: Request) -> Response {
    // Request processing logic
    Response::Ok
}
"#,
    )
    .unwrap();

    // Python file
    fs::write(
        base.join("tests/test_auth.py"),
        r#"def test_authentication():
    """Test user authentication flow"""
    assert authenticate_user("admin", "password")

def test_database_connection():
    """Test database connectivity"""
    conn = database.connect()
    assert conn.is_connected()
"#,
    )
    .unwrap();

    // JavaScript file
    fs::write(
        base.join("src/api.js"),
        r#"function handleApiRequest(req, res) {
    const data = processRequest(req);
    res.json(data);
}

module.exports = { handleApiRequest };
"#,
    )
    .unwrap();

    // Config files (simple files - fast-path chunking)
    fs::write(
        base.join("config/settings.json"),
        r#"{
  "database": {
    "host": "localhost",
    "port": 5432
  },
  "auth": {
    "enabled": true
  }
}
"#,
    )
    .unwrap();

    fs::write(
        base.join("config/app.yaml"),
        r#"version: 1.0
features:
  - authentication
  - database
  - api
"#,
    )
    .unwrap();

    fs::write(
        base.join("README.md"),
        "# Test Project\n\nThis is a test project for sgrep.\n\n## Features\n- Authentication\n- Database\n",
    )
    .unwrap();

    temp
}

fn sample_chunk(text: &str, language: &str, path: &str) -> CodeChunk {
    CodeChunk {
        id: Uuid::new_v4(),
        path: PathBuf::from(path),
        language: language.to_string(),
        start_line: 1,
        end_line: 10,
        text: text.to_string(),
        hash: blake3::hash(text.as_bytes()).to_string(),
        modified_at: Utc::now(),
    }
}

// Chunker tests

#[test]
fn test_chunker_handles_rust_files() {
    let temp = create_test_repo();
    let rust_file = temp.path().join("src/main.rs");

    let chunks = chunk_file(&rust_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from Rust file");
    assert!(chunks.iter().any(|c| c.language == "rust"));
}

#[test]
fn test_chunker_handles_python_files() {
    let temp = create_test_repo();
    let python_file = temp.path().join("tests/test_auth.py");

    let chunks = chunk_file(&python_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from Python file");
    assert!(chunks.iter().any(|c| c.language == "python"));
}

#[test]
fn test_chunker_handles_javascript_files() {
    let temp = create_test_repo();
    let js_file = temp.path().join("src/api.js");

    let chunks = chunk_file(&js_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from JS file");
    assert!(chunks.iter().any(|c| c.language == "javascript"));
}

#[test]
fn test_chunker_fast_path_for_json() {
    let temp = create_test_repo();
    let json_file = temp.path().join("config/settings.json");

    let chunks = chunk_file(&json_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from JSON file");
    assert!(chunks.iter().any(|c| c.language == "json"));
}

#[test]
fn test_chunker_fast_path_for_yaml() {
    let temp = create_test_repo();
    let yaml_file = temp.path().join("config/app.yaml");

    let chunks = chunk_file(&yaml_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from YAML file");
    assert!(chunks.iter().any(|c| c.language == "yaml"));
}

#[test]
fn test_chunker_fast_path_for_markdown() {
    let temp = create_test_repo();
    let md_file = temp.path().join("README.md");

    let chunks = chunk_file(&md_file, temp.path()).unwrap();
    assert!(!chunks.is_empty(), "Should extract chunks from Markdown file");
    assert!(chunks.iter().any(|c| c.language == "markdown"));
}

#[test]
fn test_chunker_handles_empty_files() {
    let temp = tempfile::tempdir().unwrap();
    let empty_file = temp.path().join("empty.rs");
    fs::write(&empty_file, "").unwrap();

    let chunks = chunk_file(&empty_file, temp.path()).unwrap();
    assert!(chunks.is_empty(), "Empty file should produce no chunks");
}

#[test]
fn test_chunker_handles_nonexistent_files() {
    let temp = tempfile::tempdir().unwrap();
    let missing_file = temp.path().join("missing.rs");

    let result = chunk_file(&missing_file, temp.path());
    assert!(result.is_err(), "Should error on nonexistent file");
}

// FTS tests

#[test]
fn test_extract_keywords_removes_stopwords() {
    let keywords = extract_keywords("the quick brown fox");
    assert!(!keywords.contains(&"the".to_string()));
    assert!(keywords.contains(&"quick".to_string()));
    assert!(keywords.contains(&"brown".to_string()));
}

#[test]
fn test_extract_keywords_filters_short_tokens() {
    let keywords = extract_keywords("a be cat dog");
    assert!(!keywords.contains(&"a".to_string()));
    assert!(!keywords.contains(&"be".to_string()));
    assert!(keywords.contains(&"cat".to_string()));
    assert!(keywords.contains(&"dog".to_string()));
}

#[test]
fn test_keyword_score_boosts_filename_matches() {
    let keywords = vec!["auth".to_string()];
    let score = keyword_score(&keywords, "some code", Path::new("auth.rs"));
    assert!(score > 2.0, "Filename match should have high score");
}

#[test]
fn test_keyword_score_boosts_path_matches() {
    let keywords = vec!["database".to_string()];
    let score = keyword_score(&keywords, "some code", Path::new("src/database/conn.rs"));
    assert!(score > 1.0, "Path match should have medium score");
}

#[test]
fn test_keyword_score_text_matches() {
    let keywords = vec!["authentication".to_string()];
    let score = keyword_score(&keywords, "authentication logic here", Path::new("lib.rs"));
    assert!(score > 0.0, "Text match should have low score");
}

#[test]
fn test_keyword_score_empty_keywords() {
    let keywords = vec![];
    let score = keyword_score(&keywords, "some text", Path::new("file.rs"));
    assert_eq!(score, 0.0, "Empty keywords should return 0 score");
}

#[test]
fn test_matches_filters_language() {
    let chunk = sample_chunk("fn test() {}", "rust", "test.rs");
    assert!(matches_filters(&["lang=rust".to_string()], &chunk));
    assert!(matches_filters(&["language=RUST".to_string()], &chunk)); // Case insensitive
    assert!(!matches_filters(&["lang=python".to_string()], &chunk));
}

#[test]
fn test_matches_filters_path() {
    let chunk = sample_chunk("code", "rust", "src/auth/login.rs");
    assert!(matches_filters(&["path=auth".to_string()], &chunk));
    assert!(matches_filters(&["file=login".to_string()], &chunk));
    assert!(!matches_filters(&["path=database".to_string()], &chunk));
}

#[test]
fn test_matches_filters_multiple() {
    let chunk = sample_chunk("code", "rust", "src/lib.rs");
    assert!(matches_filters(
        &["lang=rust".to_string(), "path=src".to_string()],
        &chunk
    ));
    assert!(!matches_filters(
        &["lang=rust".to_string(), "path=tests".to_string()],
        &chunk
    ));
}

#[test]
fn test_matches_filters_unknown_filter() {
    let chunk = sample_chunk("code", "rust", "test.rs");
    // Unknown filters are ignored (return true)
    assert!(matches_filters(&["unknown=value".to_string()], &chunk));
}

#[test]
fn test_build_globset_single_pattern() {
    let globset = build_globset(&["*.rs".to_string()]);
    assert!(globset.is_some());
}

#[test]
fn test_build_globset_multiple_patterns() {
    let globset = build_globset(&["*.rs".to_string(), "*.py".to_string()]);
    assert!(globset.is_some());
}

#[test]
fn test_build_globset_empty() {
    let globset = build_globset(&[]);
    assert!(globset.is_none());
}

#[test]
fn test_build_globset_invalid_pattern() {
    let globset = build_globset(&["[invalid".to_string()]);
    assert!(globset.is_none(), "Invalid patterns should return None");
}

#[test]
fn test_glob_matches_with_pattern() {
    let globset = build_globset(&["src/**/*.rs".to_string()]);
    assert!(glob_matches(
        globset.as_ref(),
        Path::new("src/main.rs")
    ));
    assert!(glob_matches(
        globset.as_ref(),
        Path::new("src/auth/login.rs")
    ));
    assert!(!glob_matches(globset.as_ref(), Path::new("tests/test.py")));
}

#[test]
fn test_glob_matches_no_pattern() {
    assert!(glob_matches(None, Path::new("any/file.rs")));
}

// Store tests

#[test]
fn test_store_save_and_load() {
    let temp = tempfile::tempdir().unwrap();
    let store = IndexStore::new(temp.path()).unwrap();

    let chunk = sample_chunk("fn test() {}", "rust", "test.rs");
    let index = RepositoryIndex {
        metadata: IndexMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            repo_path: temp.path().to_path_buf(),
            repo_hash: "test_hash".to_string(),
            vector_dim: 384,
            indexed_at: Utc::now(),
            total_files: 1,
            total_chunks: 1,
        },
        chunks: vec![chunk],
        vectors: vec![vec![0.1; 384]],
    };

    store.save(&index).unwrap();
    let loaded = store.load().unwrap();

    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert_eq!(loaded.chunks.len(), 1);
    assert_eq!(loaded.vectors.len(), 1);
}

#[test]
fn test_store_load_missing_index() {
    let temp = tempfile::tempdir().unwrap();
    let store = IndexStore::new(temp.path()).unwrap();

    let loaded = store.load().unwrap();
    assert!(loaded.is_none());
}

#[test]
fn test_store_repo_hash() {
    let temp = tempfile::tempdir().unwrap();
    let store = IndexStore::new(temp.path()).unwrap();

    let hash = store.repo_hash();
    assert!(!hash.is_empty());
    assert_eq!(hash.len(), 64); // BLAKE3 hex string length
}

// Embedding tests (comprehensive)

#[test]
fn test_embedder_caches_results() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let embedder = Embedder::default();

    let text = "fn cached_test() {}";
    let v1 = embedder.embed(text).unwrap();
    let v2 = embedder.embed(text).unwrap();

    assert_eq!(v1, v2);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_embedder_batch_mixed_cached() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let embedder = Embedder::default();

    // Pre-cache one text
    let _ = embedder.embed("fn cached() {}").unwrap();

    // Batch with mix of cached and uncached
    let texts = vec![
        "fn cached() {}".to_string(),
        "fn uncached() {}".to_string(),
    ];
    let results = embedder.embed_batch(&texts).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].len(), 384);
    assert_eq!(results[1].len(), 384);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_embedder_batch_all_cached() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let embedder = Embedder::default();

    let texts = vec!["fn test1() {}".to_string(), "fn test2() {}".to_string()];

    // First call - populate cache
    let _ = embedder.embed_batch(&texts).unwrap();

    // Second call - all from cache
    let results = embedder.embed_batch(&texts).unwrap();
    assert_eq!(results.len(), 2);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_pooled_embedder_distributes_work() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    env::set_var("SGREP_EMBEDDER_POOL_SIZE", "2");
    let embedder = PooledEmbedder::default();

    // Multiple embeds should use different workers
    let v1 = embedder.embed("fn test1() {}").unwrap();
    let v2 = embedder.embed("fn test2() {}").unwrap();

    assert_eq!(v1.len(), 384);
    assert_eq!(v2.len(), 384);
    assert_ne!(v1, v2); // Different inputs should have different embeddings

    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
    env::remove_var("SGREP_EMBEDDER_POOL_SIZE");
}

#[test]
fn test_pooled_embedder_batch() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    env::set_var("SGREP_EMBEDDER_POOL_SIZE", "2");
    let embedder = PooledEmbedder::default();

    let texts = vec![
        "fn test1() {}".to_string(),
        "fn test2() {}".to_string(),
        "fn test3() {}".to_string(),
    ];

    let results = embedder.embed_batch(&texts).unwrap();
    assert_eq!(results.len(), 3);
    for vec in results {
        assert_eq!(vec.len(), 384);
    }

    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
    env::remove_var("SGREP_EMBEDDER_POOL_SIZE");
}

// Search tests (additional coverage)

#[test]
fn test_search_with_no_results() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = tempfile::tempdir().unwrap();
    fs::write(temp.path().join("test.rs"), "fn test() {}").unwrap();

    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: None,
        profile: false,
    };
    indexer.build_full(&request).unwrap();

    let searcher = Searcher::new(temp.path()).unwrap();
    let search_req = SearchRequest {
        query: "nonexistent_function_xyz".to_string(),
        limit: 10,
        globs: vec![],
        filters: vec![],
    };

    let results = searcher.search(&search_req).unwrap();
    assert_eq!(results.matches.len(), 0);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_search_with_filters() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = create_test_repo();

    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: None,
        profile: false,
    };
    indexer.build_full(&request).unwrap();

    let searcher = Searcher::new(temp.path()).unwrap();
    let search_req = SearchRequest {
        query: "function".to_string(),
        limit: 10,
        globs: vec![],
        filters: vec!["lang=rust".to_string()],
    };

    let results = searcher.search(&search_req).unwrap();
    // Should only return Rust results
    for m in results.matches {
        assert_eq!(m.chunk.language, "rust");
    }
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_search_with_globs() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = create_test_repo();

    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: None,
        profile: false,
    };
    indexer.build_full(&request).unwrap();

    let searcher = Searcher::new(temp.path()).unwrap();
    let search_req = SearchRequest {
        query: "test".to_string(),
        limit: 10,
        globs: vec!["tests/**/*.py".to_string()],
        filters: vec![],
    };

    let results = searcher.search(&search_req).unwrap();
    // Should only return Python test files
    for m in results.matches {
        assert!(m.chunk.path.to_string_lossy().contains("tests"));
    }
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

// Indexer tests (additional coverage)

#[test]
fn test_indexer_force_rebuild() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = create_test_repo();

    let indexer = Indexer::new(temp.path());

    // First build
    let request1 = IndexRequest {
        force: false,
        batch_size: None,
        profile: false,
    };
    let result1 = indexer.build_full(&request1).unwrap();
    let first_count = result1.total_chunks;

    // Force rebuild
    let request2 = IndexRequest {
        force: true,
        batch_size: None,
        profile: false,
    };
    let result2 = indexer.build_full(&request2).unwrap();

    assert_eq!(result2.total_chunks, first_count);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_indexer_custom_batch_size() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = create_test_repo();

    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: Some(128),
        profile: false,
    };

    let result = indexer.build_full(&request).unwrap();
    assert!(result.total_chunks > 0);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_indexer_profile_timings() {
    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let temp = create_test_repo();

    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: None,
        profile: true,
    };

    let result = indexer.build_full(&request).unwrap();
    assert!(result.timings.is_some());

    let timings = result.timings.unwrap();
    assert!(timings.walk.as_millis() > 0);
    assert!(timings.chunk.as_millis() >= 0);
    assert!(timings.embed.as_millis() >= 0);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}

#[test]
fn test_indexer_respects_gitignore() {
    let temp = tempfile::tempdir().unwrap();

    // Create a file that should be ignored
    fs::write(temp.path().join(".gitignore"), "ignored.rs\n").unwrap();
    fs::write(temp.path().join("included.rs"), "fn test() {}").unwrap();
    fs::write(temp.path().join("ignored.rs"), "fn ignored() {}").unwrap();

    env::set_var("SGREP_DISABLE_PERSISTENT_CACHE", "1");
    let indexer = Indexer::new(temp.path());
    let request = IndexRequest {
        force: false,
        batch_size: None,
        profile: false,
    };

    let result = indexer.build_full(&request).unwrap();
    // Should only index included.rs
    assert_eq!(result.total_files, 1);
    env::remove_var("SGREP_DISABLE_PERSISTENT_CACHE");
}
