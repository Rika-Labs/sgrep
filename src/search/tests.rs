use super::*;
use crate::chunker::CodeChunk;
use crate::embedding::BatchEmbedder;
use crate::search::dedup::DEFAULT_SEMANTIC_DEDUP_THRESHOLD;
use crate::search::scoring::select_top_k;
use crate::store::{HierarchicalIndex, IndexMetadata, RepositoryIndex};
use anyhow::Result;
use chrono::Utc;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone, Default)]
struct MockEmbedder;

impl BatchEmbedder for MockEmbedder {
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32, 1.0, 0.0, 0.0])
            .collect())
    }

    fn dimension(&self) -> usize {
        4
    }
}

fn make_chunk(text: &str, language: &str, path: &str) -> CodeChunk {
    let hash = format!("{}_{}", path, text.len());
    CodeChunk {
        id: Uuid::new_v4(),
        path: PathBuf::from(path),
        language: language.to_string(),
        start_line: 1,
        end_line: 10,
        text: text.to_string(),
        hash,
        modified_at: Utc::now(),
    }
}

fn make_index(chunks: Vec<CodeChunk>, vectors: Vec<Vec<f32>>) -> RepositoryIndex {
    let metadata = IndexMetadata {
        version: "0.1.0".to_string(),
        repo_path: PathBuf::from("/test"),
        repo_hash: "test123".to_string(),
        vector_dim: vectors.first().map(|v| v.len()).unwrap_or(0),
        indexed_at: Utc::now(),
        total_files: 1,
        total_chunks: chunks.len(),
        embedding_model: "jina-embeddings-v2-base-code".to_string(),
    };
    RepositoryIndex::new(metadata, chunks, vectors)
}

#[test]
fn search_respects_limit() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![
        make_chunk("fn foo() {}", "rust", "a.rs"),
        make_chunk("fn bar() {}", "rust", "b.rs"),
        make_chunk("fn baz() {}", "rust", "c.rs"),
    ];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions {
                limit: 2,
                dedup: DedupOptions {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 2);
}

#[test]
fn search_filters_by_glob() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![
        make_chunk("fn test1() {}", "rust", "src/auth.rs"),
        make_chunk("fn test2() {}", "rust", "tests/auth.rs"),
    ];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                glob: vec!["src/**/*.rs".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].chunk.path.to_string_lossy().contains("src"));
}

#[test]
fn search_filters_by_language() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![
        make_chunk("fn test() {}", "rust", "test.rs"),
        make_chunk("def test():", "python", "test.py"),
    ];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                filters: vec!["lang=rust".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.language, "rust");
}

#[test]
fn select_top_k_returns_highest_scores() {
    let chunk = make_chunk("test", "rust", "test.rs");
    let mut matches = vec![
        SearchResult {
            chunk: chunk.clone(),
            score: 0.3,
            semantic_score: 0.3,
            bm25_score: 0.0,
            show_full_context: false,
        },
        SearchResult {
            chunk: chunk.clone(),
            score: 0.9,
            semantic_score: 0.9,
            bm25_score: 0.0,
            show_full_context: false,
        },
        SearchResult {
            chunk: chunk.clone(),
            score: 0.6,
            semantic_score: 0.6,
            bm25_score: 0.0,
            show_full_context: false,
        },
        SearchResult {
            chunk: chunk.clone(),
            score: 0.1,
            semantic_score: 0.1,
            bm25_score: 0.0,
            show_full_context: false,
        },
    ];

    select_top_k(&mut matches, 2);

    assert_eq!(matches.len(), 2);
    assert!((matches[0].score - 0.9).abs() < 1e-6);
    assert!((matches[1].score - 0.6).abs() < 1e-6);
}

#[test]
fn hnsw_search_produces_results_for_large_index() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks: Vec<CodeChunk> = (0..600)
        .map(|i| {
            make_chunk(
                &format!("fn func{}() {{}}", i),
                "rust",
                &format!("file{}.rs", i),
            )
        })
        .collect();
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions {
                dedup: DedupOptions {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 10);
}

#[test]
fn binary_quantization_search_produces_results() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks: Vec<CodeChunk> = (0..1100)
        .map(|i| {
            make_chunk(
                &format!("fn func{}() {{}}", i),
                "rust",
                &format!("file{}.rs", i),
            )
        })
        .collect();
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "function",
            SearchOptions {
                dedup: DedupOptions {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 10);
}

#[test]
fn search_files_returns_file_level_results() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let mut hier = HierarchicalIndex::new();
    hier.add_file(
        PathBuf::from("src/auth.rs"),
        vec![0, 1],
        vec![1.0, 0.0, 0.0, 0.0],
    );
    hier.add_file(
        PathBuf::from("src/db.rs"),
        vec![2],
        vec![0.0, 1.0, 0.0, 0.0],
    );
    hier.add_file(
        PathBuf::from("src/api.rs"),
        vec![3, 4, 5],
        vec![0.5, 0.5, 0.0, 0.0],
    );

    let results = engine.search_files(&hier, "auth", 2).unwrap();

    assert!(!results.is_empty());
    assert!(results.len() <= 2);
    for result in &results {
        assert!(!result.path.as_os_str().is_empty());
    }
}

#[test]
fn search_directories_returns_dir_level_results() {
    use crate::indexer::compute_directory_embeddings;

    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let mut hier = HierarchicalIndex::new();
    hier.add_file(
        PathBuf::from("src/auth/login.rs"),
        vec![0],
        vec![1.0, 0.0, 0.0, 0.0],
    );
    hier.add_file(
        PathBuf::from("src/auth/logout.rs"),
        vec![1],
        vec![0.8, 0.2, 0.0, 0.0],
    );
    hier.add_file(
        PathBuf::from("src/db/connection.rs"),
        vec![2],
        vec![0.0, 1.0, 0.0, 0.0],
    );

    compute_directory_embeddings(&mut hier);

    let results = engine
        .search_directories(&hier, "authentication", 2)
        .unwrap();

    assert!(!results.is_empty());
}

#[test]
fn search_empty_index_returns_empty() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());
    let index = make_index(vec![], vec![]);

    let results = engine
        .search(&index, "query", SearchOptions::default())
        .unwrap();

    assert!(results.is_empty());
}

#[test]
fn search_files_empty_hierarchy_returns_empty() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());
    let hier = HierarchicalIndex::new();

    let results = engine.search_files(&hier, "query", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_directories_empty_hierarchy_returns_empty() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());
    let hier = HierarchicalIndex::new();

    let results = engine.search_directories(&hier, "query", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn filter_by_language_matches_chunks() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![
        make_chunk("fn test() {}", "rust", "test.rs"),
        make_chunk("def test():", "python", "test.py"),
        make_chunk("function test() {}", "javascript", "test.js"),
    ];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                filters: vec!["lang=python".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.language, "python");
}

#[test]
fn multiple_glob_patterns_filter_correctly() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![
        make_chunk("fn test1() {}", "rust", "src/auth.rs"),
        make_chunk("fn test2() {}", "rust", "src/db.rs"),
        make_chunk("fn test3() {}", "rust", "tests/auth.rs"),
        make_chunk("fn test4() {}", "rust", "lib/utils.rs"),
    ];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                glob: vec!["src/**/*.rs".to_string()],
                dedup: DedupOptions {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    for r in &results {
        assert!(r.chunk.path.starts_with("src/"));
    }
}

#[test]
fn find_symbol_returns_empty_without_graph() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder);

    let results = engine.find_symbol("test");
    assert!(results.is_empty());
}

#[test]
fn find_callers_returns_empty_without_graph() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder);

    let results = engine.find_callers("test");
    assert!(results.is_empty());
}

#[test]
fn find_callees_returns_empty_without_graph() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder);

    let results = engine.find_callees("test");
    assert!(results.is_empty());
}

#[test]
fn graph_stats_returns_none_without_graph() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder);

    assert!(engine.graph_stats().is_none());
}

#[test]
fn find_symbol_with_graph() {
    use crate::graph::{CodeGraph, Symbol, SymbolKind};

    let embedder = Arc::new(MockEmbedder);
    let mut engine = SearchEngine::new(embedder);

    let mut graph = CodeGraph::new();
    graph.add_symbol(Symbol {
        id: Uuid::new_v4(),
        name: "authenticate".to_string(),
        qualified_name: "auth::authenticate".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/auth.rs"),
        start_line: 1,
        end_line: 10,
        language: "rust".to_string(),
        signature: "fn authenticate()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    engine.set_graph(graph);

    let results = engine.find_symbol("authenticate");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "authenticate");
}

#[test]
fn find_callers_with_graph() {
    use crate::graph::{CodeGraph, Edge, EdgeKind, Symbol, SymbolKind};

    let embedder = Arc::new(MockEmbedder);
    let mut engine = SearchEngine::new(embedder);

    let mut graph = CodeGraph::new();
    let callee_id = Uuid::new_v4();
    let caller_id = Uuid::new_v4();

    graph.add_symbol(Symbol {
        id: callee_id,
        name: "authenticate".to_string(),
        qualified_name: "auth::authenticate".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/auth.rs"),
        start_line: 1,
        end_line: 10,
        language: "rust".to_string(),
        signature: "fn authenticate()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    graph.add_symbol(Symbol {
        id: caller_id,
        name: "login".to_string(),
        qualified_name: "login::login".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/login.rs"),
        start_line: 1,
        end_line: 20,
        language: "rust".to_string(),
        signature: "fn login()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    graph.add_edge(Edge {
        source_id: caller_id,
        target_id: callee_id,
        kind: EdgeKind::Calls,
        metadata: None,
    });

    engine.set_graph(graph);

    let callers = engine.find_callers("authenticate");
    assert_eq!(callers.len(), 1);
    assert_eq!(callers[0].name, "login");
}

#[test]
fn find_callees_with_graph() {
    use crate::graph::{CodeGraph, Edge, EdgeKind, Symbol, SymbolKind};

    let embedder = Arc::new(MockEmbedder);
    let mut engine = SearchEngine::new(embedder);

    let mut graph = CodeGraph::new();
    let caller_id = Uuid::new_v4();
    let callee_id = Uuid::new_v4();

    graph.add_symbol(Symbol {
        id: caller_id,
        name: "login".to_string(),
        qualified_name: "login::login".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/login.rs"),
        start_line: 1,
        end_line: 20,
        language: "rust".to_string(),
        signature: "fn login()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    graph.add_symbol(Symbol {
        id: callee_id,
        name: "authenticate".to_string(),
        qualified_name: "auth::authenticate".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/auth.rs"),
        start_line: 1,
        end_line: 10,
        language: "rust".to_string(),
        signature: "fn authenticate()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    graph.add_edge(Edge {
        source_id: caller_id,
        target_id: callee_id,
        kind: EdgeKind::Calls,
        metadata: None,
    });

    engine.set_graph(graph);

    let callees = engine.find_callees("login");
    assert_eq!(callees.len(), 1);
    assert_eq!(callees[0].name, "authenticate");
}

#[test]
fn graph_stats_with_graph() {
    use crate::graph::{CodeGraph, Symbol, SymbolKind};

    let embedder = Arc::new(MockEmbedder);
    let mut engine = SearchEngine::new(embedder);

    let mut graph = CodeGraph::new();
    graph.add_symbol(Symbol {
        id: Uuid::new_v4(),
        name: "test".to_string(),
        qualified_name: "test::test".to_string(),
        kind: SymbolKind::Function,
        file_path: PathBuf::from("src/test.rs"),
        start_line: 1,
        end_line: 10,
        language: "rust".to_string(),
        signature: "fn test()".to_string(),
        parent_id: None,
        chunk_id: None,
    });

    engine.set_graph(graph);

    let stats = engine.graph_stats();
    assert!(stats.is_some());
    assert_eq!(stats.unwrap().total_symbols, 1);
}

#[test]
fn cosine_similarity_identical_vectors() {
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let similarity = cosine_similarity(&v1, &v1);
    assert!((similarity - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_orthogonal_vectors() {
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0, 0.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert!(similarity.abs() < 1e-6);
}

#[test]
fn cosine_similarity_opposite_vectors() {
    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![-1.0, 0.0, 0.0, 0.0];
    let similarity = cosine_similarity(&v1, &v2);
    assert!((similarity + 1.0).abs() < 1e-6);
}

#[test]
fn search_with_include_context() {
    let embedder = Arc::new(MockEmbedder);
    let engine = SearchEngine::new(embedder.clone());

    let chunks = vec![make_chunk("fn test() {}", "rust", "test.rs")];
    let vectors: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| embedder.embed(&c.text).unwrap())
        .collect();

    let results = engine
        .search(
            &make_index(chunks, vectors),
            "test",
            SearchOptions {
                include_context: true,
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].show_full_context);
}

#[test]
fn select_top_k_handles_less_than_k() {
    let mut matches: Vec<SearchResult> = vec![];
    select_top_k(&mut matches, 5);
    assert!(matches.is_empty());
}

// Test for search consolidation (Phase 5) - verify SearchContext reduces duplication
mod search_consolidation_tests {
    use super::*;

    #[test]
    fn search_context_calculates_fetch_limit() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());
        let options = SearchOptions {
            limit: 5,
            ..Default::default()
        };
        let (_, fetch_limit) = engine.prepare_search("test", &options).unwrap();
        assert_eq!(fetch_limit, 5);
    }

    #[test]
    fn merge_results_keeps_higher_scores() {
        let chunk = make_chunk("test", "rust", "test.rs");
        let mut existing = vec![SearchResult {
            chunk: chunk.clone(),
            score: 0.5,
            semantic_score: 0.5,
            bm25_score: 0.0,
            show_full_context: false,
        }];
        let new_result = SearchResult {
            chunk: chunk.clone(),
            score: 0.8,
            semantic_score: 0.8,
            bm25_score: 0.0,
            show_full_context: false,
        };

        SearchEngine::merge_result(&mut existing, new_result);

        assert_eq!(existing.len(), 1);
        assert!((existing[0].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn merge_results_adds_new_chunks() {
        let chunk1 = make_chunk("test1", "rust", "test1.rs");
        let chunk2 = make_chunk("test2", "rust", "test2.rs");
        let mut existing = vec![SearchResult {
            chunk: chunk1,
            score: 0.5,
            semantic_score: 0.5,
            bm25_score: 0.0,
            show_full_context: false,
        }];
        let new_result = SearchResult {
            chunk: chunk2,
            score: 0.6,
            semantic_score: 0.6,
            bm25_score: 0.0,
            show_full_context: false,
        };

        SearchEngine::merge_result(&mut existing, new_result);

        assert_eq!(existing.len(), 2);
    }
}

// Test to verify embedding is hoisted outside loops (Phase 2.1 performance fix)
mod embedding_call_count_tests {
    use super::*;
    use crate::graph::{Symbol, SymbolKind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEmbedder {
        call_count: AtomicUsize,
    }

    impl CountingEmbedder {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }

        fn get_call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl BatchEmbedder for CountingEmbedder {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1.0, 0.5, 0.0, -0.5])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|_| vec![1.0, 0.5, 0.0, -0.5]).collect())
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    fn make_symbol(name: &str, path: &str, start: usize, end: usize) -> Symbol {
        Symbol {
            id: Uuid::new_v4(),
            name: name.to_string(),
            qualified_name: name.to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from(path),
            start_line: start,
            end_line: end,
            language: "rust".to_string(),
            signature: format!("fn {}()", name),
            parent_id: None,
            chunk_id: None,
        }
    }

    #[test]
    fn graph_results_to_search_results_embeds_query_once() {
        let embedder = Arc::new(CountingEmbedder::new());
        let engine = SearchEngine::new(embedder.clone());

        // Create multiple chunks and symbols
        let chunks = vec![
            make_chunk("fn foo() {}", "rust", "src/foo.rs"),
            make_chunk("fn bar() {}", "rust", "src/bar.rs"),
            make_chunk("fn baz() {}", "rust", "src/baz.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks.iter().map(|_| vec![1.0, 0.5, 0.0, -0.5]).collect();
        let index = make_index(chunks, vectors);

        // Create symbols that match the chunks
        let symbols = vec![
            make_symbol("foo", "src/foo.rs", 1, 10),
            make_symbol("bar", "src/bar.rs", 1, 10),
            make_symbol("baz", "src/baz.rs", 1, 10),
        ];

        let options = SearchOptions {
            limit: 10,
            ..Default::default()
        };

        let initial_count = embedder.get_call_count();
        let _ = engine.graph_results_to_search_results(&symbols, &index, "test query", &options);
        let calls_made = embedder.get_call_count() - initial_count;
        assert_eq!(
            calls_made, 1,
            "Expected embed() to be called once, but was called {} times",
            calls_made
        );
    }
}

// Tests for BM25F caching (GitHub issue #19)
mod bm25_cache_integration_tests {
    use super::*;

    fn counts(engine: &SearchEngine) -> (usize, usize, usize) {
        engine.bm25_cache_counts()
    }

    #[test]
    fn cache_hit_on_multiple_searches_same_index() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk("fn foo() {}", "rust", "a.rs"),
            make_chunk("fn bar() {}", "rust", "b.rs"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();
        let index = make_index(chunks, vectors);

        // First search - cache miss
        let _ = engine
            .search(&index, "function", SearchOptions::default())
            .unwrap();

        // Check cache has entry
        let (entries, hits, misses) = counts(&engine);
        assert_eq!(entries, 1);
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Second search - should be cache hit
        let _ = engine
            .search(&index, "different query", SearchOptions::default())
            .unwrap();

        let (_, hits_after, _) = counts(&engine);
        assert_eq!(hits_after, 1);
    }

    #[test]
    fn cache_invalidated_on_set_graph() {
        use crate::graph::{CodeGraph, Symbol, SymbolKind};

        let embedder = Arc::new(MockEmbedder);
        let mut engine = SearchEngine::new(embedder.clone());

        let chunks = vec![make_chunk("fn test() {}", "rust", "test.rs")];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();
        let index = make_index(chunks, vectors);

        // First search - cache miss, stores entry
        let _ = engine
            .search(&index, "test", SearchOptions::default())
            .unwrap();
        let (entries, _, _) = counts(&engine);
        assert_eq!(entries, 1);

        // Set graph - should invalidate cache
        let mut graph = CodeGraph::new();
        graph.add_symbol(Symbol {
            id: Uuid::new_v4(),
            name: "test".to_string(),
            qualified_name: "test::test".to_string(),
            kind: SymbolKind::Function,
            file_path: PathBuf::from("test.rs"),
            start_line: 1,
            end_line: 10,
            language: "rust".to_string(),
            signature: "fn test()".to_string(),
            parent_id: None,
            chunk_id: None,
        });
        engine.set_graph(graph);

        // Cache should be cleared
        let (entries_cleared, _, _) = counts(&engine);
        assert_eq!(entries_cleared, 0);

        // Next search should be cache miss
        let _ = engine
            .search(&index, "test", SearchOptions::default())
            .unwrap();
        let (_, _, misses_after) = counts(&engine);
        assert_eq!(misses_after, 2);
    }

    #[test]
    fn cache_miss_on_different_repo() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks1 = vec![make_chunk("fn foo() {}", "rust", "a.rs")];
        let vectors1: Vec<Vec<f32>> = chunks1
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let mut index1 = make_index(chunks1, vectors1.clone());
        index1.metadata.repo_hash = "repo1".to_string();

        let chunks2 = vec![make_chunk("fn bar() {}", "rust", "b.rs")];
        let vectors2: Vec<Vec<f32>> = chunks2
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let mut index2 = make_index(chunks2, vectors2);
        index2.metadata.repo_hash = "repo2".to_string();

        // Search on first index
        let _ = engine
            .search(&index1, "test", SearchOptions::default())
            .unwrap();
        let (_, _, misses_first) = counts(&engine);
        assert_eq!(misses_first, 1);

        // Search on second index - different repo, should be cache miss
        let _ = engine
            .search(&index2, "test", SearchOptions::default())
            .unwrap();
        let (_, _, misses_second) = counts(&engine);
        assert_eq!(misses_second, 2);

        // Cache should still have 1 entry (replaced)
        let (entries_after, _, _) = counts(&engine);
        assert_eq!(entries_after, 1);
    }

    #[test]
    fn cache_miss_on_different_chunk_count() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks1 = vec![make_chunk("fn foo() {}", "rust", "a.rs")];
        let vectors1: Vec<Vec<f32>> = chunks1
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();
        let index1 = make_index(chunks1, vectors1);

        let chunks2 = vec![
            make_chunk("fn foo() {}", "rust", "a.rs"),
            make_chunk("fn bar() {}", "rust", "b.rs"),
        ];
        let vectors2: Vec<Vec<f32>> = chunks2
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let mut index2 = make_index(chunks2, vectors2);
        // Same repo_hash but different chunk count
        index2.metadata.repo_hash = index1.metadata.repo_hash.clone();

        // Search on first index
        let _ = engine
            .search(&index1, "test", SearchOptions::default())
            .unwrap();
        let (_, _, misses_first) = counts(&engine);
        assert_eq!(misses_first, 1);

        // Search on second index - different chunk count, should be cache miss
        let _ = engine
            .search(&index2, "test", SearchOptions::default())
            .unwrap();
        let (_, _, misses_second) = counts(&engine);
        assert_eq!(misses_second, 2);
    }
}

mod near_duplicate_suppression_tests {
    use super::*;

    fn make_chunk_with_hash(text: &str, path: &str, hash: &str) -> CodeChunk {
        CodeChunk {
            id: Uuid::new_v4(),
            path: PathBuf::from(path),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            text: text.to_string(),
            hash: hash.to_string(),
            modified_at: Utc::now(),
        }
    }

    #[allow(dead_code)]
    fn make_result_with_score(
        chunk: CodeChunk,
        score: f32,
        vector: Vec<f32>,
    ) -> (SearchResult, Vec<f32>) {
        (
            SearchResult {
                chunk,
                score,
                semantic_score: score,
                bm25_score: 0.0,
                show_full_context: false,
            },
            vector,
        )
    }

    #[test]
    fn suppress_identical_hash_chunks() {
        let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
        let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "hash123");
        let chunk3 = make_chunk_with_hash("fn bar() {}", "c.rs", "hash456");

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.8,
                semantic_score: 0.8,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk3,
                score: 0.7,
                semantic_score: 0.7,
                bm25_score: 0.0,
                show_full_context: false,
            },
        ];

        suppress_near_duplicates(&mut results, &[], &DedupOptions::default());

        assert_eq!(results.len(), 2);
        assert!((results[0].score - 0.9).abs() < 1e-6);
        assert_eq!(results[0].chunk.hash, "hash123");
        assert_eq!(results[1].chunk.hash, "hash456");
    }

    #[test]
    fn suppress_semantically_similar_chunks() {
        let chunk1 =
            make_chunk_with_hash("fn authenticate() { check_password(); }", "auth.rs", "h1");
        let chunk2 = make_chunk_with_hash(
            "fn authenticate() { check_password(); verify(); }",
            "auth2.rs",
            "h2",
        );
        let chunk3 = make_chunk_with_hash("fn connect_to_database() {}", "db.rs", "h3");

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.99, 0.14, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.85,
                semantic_score: 0.85,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk3,
                score: 0.7,
                semantic_score: 0.7,
                bm25_score: 0.0,
                show_full_context: false,
            },
        ];

        let options = DedupOptions {
            semantic_threshold: 0.95,
            ..Default::default()
        };

        suppress_near_duplicates(&mut results, &vectors, &options);

        assert_eq!(results.len(), 2);
        assert!(results[0].chunk.path.to_string_lossy().contains("auth.rs"));
        assert!(results[1].chunk.path.to_string_lossy().contains("db.rs"));
    }

    #[test]
    fn preserve_similar_but_distinct_chunks() {
        let chunk1 = make_chunk_with_hash("fn process_user() {}", "a.rs", "h1");
        let chunk2 = make_chunk_with_hash("fn process_order() {}", "b.rs", "h2");

        let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.8, 0.6, 0.0, 0.0]];

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.85,
                semantic_score: 0.85,
                bm25_score: 0.0,
                show_full_context: false,
            },
        ];

        let options = DedupOptions {
            semantic_threshold: 0.95,
            ..Default::default()
        };

        suppress_near_duplicates(&mut results, &vectors, &options);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn ordering_is_reproducible() {
        let chunk1 = make_chunk_with_hash("fn a() {}", "a.rs", "h1");
        let chunk2 = make_chunk_with_hash("fn b() {}", "b.rs", "h2");
        let chunk3 = make_chunk_with_hash("fn c() {}", "c.rs", "h3");

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let create_results = || {
            vec![
                SearchResult {
                    chunk: chunk1.clone(),
                    score: 0.9,
                    semantic_score: 0.9,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk2.clone(),
                    score: 0.85,
                    semantic_score: 0.85,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
                SearchResult {
                    chunk: chunk3.clone(),
                    score: 0.8,
                    semantic_score: 0.8,
                    bm25_score: 0.0,
                    show_full_context: false,
                },
            ]
        };

        let mut results1 = create_results();
        let mut results2 = create_results();

        suppress_near_duplicates(&mut results1, &vectors, &DedupOptions::default());
        suppress_near_duplicates(&mut results2, &vectors, &DedupOptions::default());

        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.chunk.id, r2.chunk.id);
            assert!((r1.score - r2.score).abs() < 1e-6);
        }
    }

    #[test]
    fn dedup_options_default_values() {
        let options = DedupOptions::default();
        assert!((options.semantic_threshold - DEFAULT_SEMANTIC_DEDUP_THRESHOLD).abs() < 1e-6);
        assert!(options.enabled);
    }

    #[test]
    fn suppression_disabled_when_not_enabled() {
        let chunk1 = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
        let chunk2 = make_chunk_with_hash("fn foo() {}", "b.rs", "hash123"); // Same hash

        let mut results = vec![
            SearchResult {
                chunk: chunk1,
                score: 0.9,
                semantic_score: 0.9,
                bm25_score: 0.0,
                show_full_context: false,
            },
            SearchResult {
                chunk: chunk2,
                score: 0.8,
                semantic_score: 0.8,
                bm25_score: 0.0,
                show_full_context: false,
            },
        ];

        let options = DedupOptions {
            enabled: false,
            ..Default::default()
        };

        suppress_near_duplicates(&mut results, &[], &options);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn empty_results_handled_gracefully() {
        let mut results: Vec<SearchResult> = vec![];
        suppress_near_duplicates(&mut results, &[], &DedupOptions::default());
        assert!(results.is_empty());
    }

    #[test]
    fn single_result_unchanged() {
        let chunk = make_chunk_with_hash("fn foo() {}", "a.rs", "hash123");
        let mut results = vec![SearchResult {
            chunk,
            score: 0.9,
            semantic_score: 0.9,
            bm25_score: 0.0,
            show_full_context: false,
        }];

        let original_len = results.len();
        suppress_near_duplicates(
            &mut results,
            &[vec![1.0, 0.0, 0.0, 0.0]],
            &DedupOptions::default(),
        );
        assert_eq!(results.len(), original_len);
    }

    #[test]
    fn search_options_includes_dedup_setting() {
        let options = SearchOptions::default();
        assert!(options.dedup.enabled, "Dedup should be enabled by default");
    }

    #[test]
    fn search_applies_dedup_by_default() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk_with_hash("fn duplicate() { same_code(); }", "a.rs", "same_hash"),
            make_chunk_with_hash("fn duplicate() { same_code(); }", "b.rs", "same_hash"),
            make_chunk_with_hash("fn unique() { different_code(); }", "c.rs", "diff_hash"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let results = engine
            .search(
                &make_index(chunks, vectors),
                "duplicate",
                SearchOptions::default(),
            )
            .unwrap();

        let same_hash_count = results
            .iter()
            .filter(|r| r.chunk.hash == "same_hash")
            .count();
        assert_eq!(same_hash_count, 1);
    }

    #[test]
    fn dedup_can_be_disabled_in_search_options() {
        let embedder = Arc::new(MockEmbedder);
        let engine = SearchEngine::new(embedder.clone());

        let chunks = vec![
            make_chunk_with_hash("fn duplicate() { same_code(); }", "a.rs", "same_hash"),
            make_chunk_with_hash("fn duplicate() { same_code(); }", "b.rs", "same_hash"),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| embedder.embed(&c.text).unwrap())
            .collect();

        let options = SearchOptions {
            dedup: DedupOptions {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let results = engine
            .search(&make_index(chunks, vectors), "duplicate", options)
            .unwrap();

        assert_eq!(results.len(), 2);
    }
}
