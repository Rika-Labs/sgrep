#!/bin/bash
# Download real repositories for benchmarking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Downloading benchmark repositories..."

# Small repo: actix-web (~1.2K files)
if [ ! -d "actix-web" ]; then
    echo "Cloning actix-web (small repo: ~1.2K files)..."
    git clone --depth 1 https://github.com/actix/actix-web.git
    echo "✅ actix-web cloned"
else
    echo "✓ actix-web already exists"
fi

# Medium repo: vscode (~9.8K files)
if [ ! -d "vscode" ]; then
    echo "Cloning vscode (medium repo: ~9.8K files)..."
    git clone --depth 1 https://github.com/microsoft/vscode.git
    echo "✅ vscode cloned"
else
    echo "✓ vscode already exists"
fi

# Large repo: We'll use a synthetic one since cloning Chromium is impractical
echo ""
echo "🔧 Generating synthetic repositories..."

# Generate synthetic repos if Python is available
if command -v python3 &> /dev/null; then
    chmod +x generate_synthetic.py

    # 10K files
    if [ ! -d "synthetic_10000" ]; then
        echo "Generating synthetic_10000 (10K files)..."
        ./generate_synthetic.py 10000
    else
        echo "✓ synthetic_10000 already exists"
    fi

    # 50K files (primary target)
    if [ ! -d "synthetic_50000" ]; then
        echo "Generating synthetic_50000 (50K files - PRIMARY TARGET)..."
        ./generate_synthetic.py 50000
    else
        echo "✓ synthetic_50000 already exists"
    fi
else
    echo "⚠️  Python3 not found, skipping synthetic repo generation"
    echo "   Install Python3 or generate manually with: ./generate_synthetic.py 50000"
fi

echo ""
echo "✅ Benchmark repositories ready!"
echo ""
echo "Repository summary:"
echo "  - actix-web: ~1.2K files (real Rust project)"
echo "  - vscode: ~9.8K files (real TypeScript project)"
echo "  - synthetic_10000: 10K files (mixed languages)"
echo "  - synthetic_50000: 50K files (mixed languages) ⭐ PRIMARY TARGET"
echo ""
echo "Run benchmarks with: cargo bench --bench bench_indexing"
