#!/bin/bash

set -euo pipefail

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4 || echo "")

if [ -z "$SESSION_ID" ]; then
  echo '{"error": "Failed to extract session_id from hook input"}' >&2
  exit 1
fi

if ! command -v sgrep &> /dev/null; then
  echo '{"error": "sgrep is not installed. Please install sgrep first: curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install.sh | sh"}' >&2
  exit 1
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PID_DIR="$HOME/.sgrep/watch-pids"
mkdir -p "$PID_DIR"

PID_FILE="$PID_DIR/$SESSION_ID"

if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE")
  if ps -p "$OLD_PID" > /dev/null 2>&1; then
    echo "{\"message\": \"sgrep watch already running for session $SESSION_ID (PID: $OLD_PID)\", \"pid\": $OLD_PID}"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

sgrep index -d "$PROJECT_DIR" > /dev/null 2>&1 || true
WATCH_OUTPUT=$(sgrep watch -d "$PROJECT_DIR")
WATCH_PID=$(echo "$WATCH_OUTPUT" | grep -oE '[0-9]+')

if [ -z "$WATCH_PID" ]; then
  echo "{\"error\": \"Failed to start sgrep watch\"}" >&2
  exit 1
fi

echo "$WATCH_PID" > "$PID_FILE"

echo "{\"message\": \"sgrep watch started for session $SESSION_ID (PID: $WATCH_PID, project: $PROJECT_DIR)\", \"pid\": $WATCH_PID, \"project_dir\": \"$PROJECT_DIR\"}"

