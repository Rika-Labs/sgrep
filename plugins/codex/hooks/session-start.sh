#!/bin/bash
set -euo pipefail

# Codex SessionStart hook — runs sgrep index at session start.
# Does NOT start sgrep watch (Codex has no SessionEnd to clean up).
# Use plugins/codex/sgrep-watch.sh for full watch lifecycle.

if ! command -v sgrep &> /dev/null; then
  exit 0
fi

sgrep index > /dev/null 2>&1 || true
