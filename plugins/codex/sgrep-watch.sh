#!/bin/bash
set -euo pipefail

# Wrapper that runs Codex with sgrep watch lifecycle.
# Codex has no SessionEnd event, so this uses trap EXIT to kill
# the sgrep watch process when Codex exits.
#
# Usage: ./plugins/codex/sgrep-watch.sh [codex args...]

WATCH_PID=""

cleanup() {
  if [ -n "$WATCH_PID" ] && kill -0 "$WATCH_PID" 2>/dev/null; then
    kill "$WATCH_PID" 2>/dev/null
    wait "$WATCH_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if ! command -v sgrep &> /dev/null; then
  echo "warning: sgrep not found, starting codex without watch" >&2
  exec codex "$@"
fi

sgrep index > /dev/null 2>&1 || true

WATCH_OUTPUT=$(sgrep watch 2>/dev/null)
WATCH_PID=$(echo "$WATCH_OUTPUT" | grep -oE '[0-9]+')

if [ -z "$WATCH_PID" ]; then
  echo "warning: failed to start sgrep watch, starting codex without it" >&2
  exec codex "$@"
fi

codex "$@"
