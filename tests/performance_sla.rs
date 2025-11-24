/*!
Performance SLA Tests

These tests ensure that sgrep meets guaranteed performance targets.
They are NOT unit tests - they verify end-to-end throughput.

Run with: cargo test --release --test performance_sla -- --ignored
*/

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

const BENCHES_DIR: &str = "benches/repos";

/// Helper to check if a test repo exists
fn repo_exists(name: &str) -> bool {
    PathBuf::from(BENCHES_DIR).join(name).exists()
}

/// Time a full index operation
fn time_index(repo_path: &str) -> Duration {
    let start = Instant::now();

    let output = Command::new(env!("CARGO_BIN_EXE_sgrep"))
        .args(&["index", "--force", repo_path])
        .output()
        .expect("Failed to execute sgrep");

    let elapsed = start.elapsed();

    if !output.status.success() {
        panic!(
            "Indexing failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    elapsed
}

#[test]
#[ignore] // Run with --ignored flag (slow test)
fn sla_small_repo_under_5s() {
    let repo_name = "actix-web";

    if !repo_exists(repo_name) {
        eprintln!("⚠️  Skipping: {} not found", repo_name);
        eprintln!("   Run: cd benches/repos && ./download_repos.sh");
        return;
    }

    let repo_path = format!("{}/{}", BENCHES_DIR, repo_name);
    let duration = time_index(&repo_path);

    println!("✅ {}: indexed in {:.2}s", repo_name, duration.as_secs_f64());

    assert!(
        duration.as_secs() < 5,
        "❌ SLA VIOLATION: Small repo (<1K files) took {}s (target: <5s)",
        duration.as_secs()
    );
}

#[test]
#[ignore]
fn sla_medium_repo_under_20s() {
    let repo_name = "vscode";

    if !repo_exists(repo_name) {
        eprintln!("⚠️  Skipping: {} not found", repo_name);
        eprintln!("   Run: cd benches/repos && ./download_repos.sh");
        return;
    }

    let repo_path = format!("{}/{}", BENCHES_DIR, repo_name);
    let duration = time_index(&repo_path);

    println!("✅ {}: indexed in {:.2}s", repo_name, duration.as_secs_f64());

    assert!(
        duration.as_secs() < 20,
        "❌ SLA VIOLATION: Medium repo (1K-10K files) took {}s (target: <20s)",
        duration.as_secs()
    );
}

#[test]
#[ignore]
fn sla_10k_synthetic_under_15s() {
    let repo_name = "synthetic_10000";

    if !repo_exists(repo_name) {
        eprintln!("⚠️  Skipping: {} not found", repo_name);
        eprintln!("   Run: cd benches/repos && ./generate_synthetic.py 10000");
        return;
    }

    let repo_path = format!("{}/{}", BENCHES_DIR, repo_name);
    let duration = time_index(&repo_path);

    let files_per_sec = 10_000.0 / duration.as_secs_f64();
    println!(
        "✅ {}: indexed in {:.2}s ({:.0} files/s)",
        repo_name,
        duration.as_secs_f64(),
        files_per_sec
    );

    assert!(
        duration.as_secs() < 15,
        "❌ SLA VIOLATION: 10K files took {}s (target: <15s)",
        duration.as_secs()
    );
}

#[test]
#[ignore]
fn sla_50k_files_under_60s() {
    let repo_name = "synthetic_50000";

    if !repo_exists(repo_name) {
        eprintln!("⚠️  Skipping: {} not found", repo_name);
        eprintln!("   Run: cd benches/repos && ./generate_synthetic.py 50000");
        return;
    }

    let repo_path = format!("{}/{}", BENCHES_DIR, repo_name);

    println!("🎯 PRIMARY SLA TEST: Indexing 50K files...");
    let duration = time_index(&repo_path);

    let files_per_sec = 50_000.0 / duration.as_secs_f64();
    println!(
        "✅ {}: indexed in {:.2}s ({:.0} files/s)",
        repo_name,
        duration.as_secs_f64(),
        files_per_sec
    );

    assert!(
        duration.as_secs() < 60,
        "❌ SLA VIOLATION: 50K files took {}s (target: <60s, {:.0} files/s required)",
        duration.as_secs(),
        833.0
    );

    // Also assert minimum throughput
    assert!(
        files_per_sec >= 833.0,
        "❌ THROUGHPUT VIOLATION: {:.0} files/s (target: 833+ files/s)",
        files_per_sec
    );
}

#[test]
#[ignore]
fn sla_search_latency_p95() {
    let repo_name = "synthetic_50000";

    if !repo_exists(repo_name) {
        eprintln!("⚠️  Skipping: {} not found", repo_name);
        return;
    }

    let repo_path = format!("{}/{}", BENCHES_DIR, repo_name);

    // Ensure index exists
    let _ = time_index(&repo_path);

    // Run multiple searches and measure P95
    let queries = vec![
        "authentication middleware",
        "database connection pooling",
        "error handling retry logic",
        "API rate limiting",
        "user validation",
    ];

    let mut latencies: Vec<Duration> = Vec::new();

    for query in &queries {
        for _ in 0..10 {
            // 10 runs per query
            let start = Instant::now();

            let output = Command::new(env!("CARGO_BIN_EXE_sgrep"))
                .args(&["search", query, "--path", &repo_path, "-n", "10"])
                .output()
                .expect("Failed to execute search");

            let elapsed = start.elapsed();
            latencies.push(elapsed);

            assert!(
                output.status.success(),
                "Search failed for query: {}",
                query
            );
        }
    }

    // Calculate P95
    latencies.sort();
    let p95_index = (latencies.len() as f64 * 0.95) as usize;
    let p95_latency = latencies[p95_index];

    println!(
        "✅ Search P95 latency: {:.2}ms (target: <200ms)",
        p95_latency.as_millis()
    );

    assert!(
        p95_latency.as_millis() < 200,
        "❌ SEARCH SLA VIOLATION: P95 latency {}ms (target: <200ms)",
        p95_latency.as_millis()
    );
}

#[test]
fn sla_suite_instructions() {
    println!("
╔════════════════════════════════════════════════════════════════╗
║          SGREP PERFORMANCE SLA TEST SUITE                      ║
╚════════════════════════════════════════════════════════════════╝

These tests verify guaranteed performance targets for sgrep.

SETUP:
  1. Download/generate test repositories:
     cd benches/repos && ./download_repos.sh

  2. Ensure sgrep is built in release mode:
     cargo build --release

RUN TESTS:
  # Run all SLA tests
  cargo test --release --test performance_sla -- --ignored

  # Run just the 50K file target
  cargo test --release --test performance_sla sla_50k_files_under_60s -- --ignored --nocapture

TARGETS:
  ✅ Small repo (<1K files):     <5s
  ✅ Medium repo (1K-10K files):  <20s
  ✅ 10K synthetic files:        <15s
  🎯 50K synthetic files:        <60s  (PRIMARY TARGET)
  ✅ Search P95 latency:         <200ms

NOTES:
  - Tests run with --release for accurate performance
  - Tests are marked #[ignore] because they're slow
  - Use --nocapture to see timing output
  - Failed assertions indicate SLA violations
    ");
}
