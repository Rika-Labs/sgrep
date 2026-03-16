#!/bin/bash
set -euo pipefail

cargo build --release --quiet
./target/release/sgrep index --force --offline . >/dev/null
python3 benchmarks/search_quality/eval.py --sgrep ./target/release/sgrep --repo .
