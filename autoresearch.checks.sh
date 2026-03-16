#!/bin/bash
set -euo pipefail

cargo fmt -- --check >/dev/null
cargo clippy -- -D warnings >/dev/null
cargo test >/dev/null
