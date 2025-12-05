use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs;
use std::path::PathBuf;

use sgrep::chunker;

fn setup_test_files(count: usize) -> (PathBuf, Vec<PathBuf>) {
    let dir = std::env::temp_dir().join(format!("sgrep_bench_{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&dir).unwrap();

    let files: Vec<PathBuf> = (0..count)
        .map(|i| {
            let ext = match i % 4 {
                0 => "rs",
                1 => "py",
                2 => "js",
                _ => "go",
            };
            let content = generate_code(ext, i, 50);
            let path = dir.join(format!("file_{}.{}", i, ext));
            fs::write(&path, content).unwrap();
            path
        })
        .collect();

    (dir, files)
}

fn generate_code(ext: &str, id: usize, lines: usize) -> String {
    let mut content = String::new();

    match ext {
        "rs" => {
            content.push_str(&format!("//! Module {}\n\n", id));
            content.push_str("use std::collections::HashMap;\n\n");
            for i in 0..lines {
                content.push_str(&format!(
                    "fn function_{}_{id}(x: i32) -> i32 {{\n    x + {}\n}}\n\n",
                    i, i
                ));
            }
        }
        "py" => {
            content.push_str(&format!("\"\"\"Module {}\"\"\"\n\n", id));
            content.push_str("from typing import Dict, List\n\n");
            for i in 0..lines {
                content.push_str(&format!(
                    "def function_{}_{id}(x: int) -> int:\n    return x + {}\n\n",
                    i, i
                ));
            }
        }
        "js" => {
            content.push_str(&format!("// Module {}\n\n", id));
            content.push_str("import {{ something }} from 'somewhere';\n\n");
            for i in 0..lines {
                content.push_str(&format!(
                    "function function_{}_{id}(x) {{\n    return x + {};\n}}\n\n",
                    i, i
                ));
            }
        }
        "go" => {
            content.push_str(&format!("// Module {}\n", id));
            content.push_str(&format!("package module{}\n\n", id));
            content.push_str("import \"fmt\"\n\n");
            for i in 0..lines {
                content.push_str(&format!(
                    "func Function_{}_{id}(x int) int {{\n    return x + {}\n}}\n\n",
                    i, i
                ));
            }
        }
        _ => {
            for i in 0..lines {
                content.push_str(&format!("Line {} of file {}\n", i, id));
            }
        }
    }

    content
}

fn cleanup_test_files(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

fn bench_chunk_single_file(c: &mut Criterion) {
    let (dir, files) = setup_test_files(1);
    let file = &files[0];

    c.bench_with_input(BenchmarkId::new("chunk_single_file", "rust"), file, |b, path| {
        b.iter(|| chunker::chunk_file(path, &dir).unwrap())
    });

    cleanup_test_files(&dir);
}

fn bench_chunk_files_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_sequential");

    for count in [10, 50, 100].iter() {
        let (dir, files) = setup_test_files(*count);

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &files, |b, files| {
            b.iter(|| {
                for path in files {
                    let _ = chunker::chunk_file(path, &dir);
                }
            })
        });

        cleanup_test_files(&dir);
    }

    group.finish();
}

fn bench_chunk_files_parallel(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("chunk_parallel");

    for count in [10, 50, 100].iter() {
        let (dir, files) = setup_test_files(*count);

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &files, |b, files| {
            b.iter(|| {
                let _: Vec<_> = files
                    .par_iter()
                    .map(|path| chunker::chunk_file(path, &dir))
                    .collect();
            })
        });

        cleanup_test_files(&dir);
    }

    group.finish();
}

fn bench_parser_creation(c: &mut Criterion) {
    use tree_sitter::Parser;

    c.bench_function("parser_creation", |b| {
        b.iter(|| {
            let mut parser = Parser::new();
            let _ = parser.set_language(&tree_sitter_rust::LANGUAGE.into());
            parser
        })
    });
}

criterion_group!(
    benches,
    bench_chunk_single_file,
    bench_chunk_files_sequential,
    bench_chunk_files_parallel,
    bench_parser_creation,
);
criterion_main!(benches);
