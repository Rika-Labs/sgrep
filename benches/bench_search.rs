use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

fn benchmark_search_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_latency");

    // Search should be fast, normal measurement time
    group.measurement_time(Duration::from_secs(10));

    let queries = vec![
        ("simple", "authentication"),
        ("complex", "how does the middleware handle rate limiting?"),
        ("code_specific", "retry logic with exponential backoff"),
    ];

    for (name, query) in queries {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &query,
            |b, &q| {
                b.iter(|| {
                    // TODO: Integrate with actual sgrep search
                    // let searcher = sgrep::search::Searcher::load_index(&repo_path);
                    // let results = searcher.search(q, 10);
                    // black_box(results);

                    black_box(q);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity");

    // Test SIMD vs scalar cosine similarity
    group.bench_function("cosine_similarity_1000_vectors", |b| {
        // TODO: Generate test vectors
        // let query_vec = vec![0.1; 384]; // BGE-small dimension
        // let index_vectors = vec![vec![0.1; 384]; 1000];

        b.iter(|| {
            // TODO: Benchmark cosine similarity
            // for vec in &index_vectors {
            //     let score = sgrep::search::cosine_similarity(&query_vec, vec);
            //     black_box(score);
            // }
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_search_latency, benchmark_vector_operations);
criterion_main!(benches);
