#!/bin/bash

set -euo pipefail

HOOK_INPUT=$(cat)
SESSION_ID=$(echo "$HOOK_INPUT" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4 || echo "")

if [ -z "$SESSION_ID" ]; then
  echo '{"error": "Failed to extract session_id from hook input"}' >&2
  exit 1
fi

PID_DIR="$HOME/.sgrep/watch-pids"
PID_FILE="$PID_DIR/$SESSION_ID"

if [ ! -f "$PID_FILE" ]; then
  echo "{\"message\": \"No watch process found for session $SESSION_ID\"}"
  exit 0
fi

WATCH_PID=$(cat "$PID_FILE")

if ! ps -p "$WATCH_PID" > /dev/null 2>&1; then
  echo "{\"message\": \"Watch process $WATCH_PID not running (may have been terminated externally)\"}"
  rm -f "$PID_FILE"
  exit 0
fi

kill -TERM "$WATCH_PID" 2>/dev/null || true

for i in {1..5}; do
  sleep 0.5
  if ! ps -p "$WATCH_PID" > /dev/null 2>&1; then
    echo "{\"message\": \"sgrep watch stopped gracefully for session $SESSION_ID (PID: $WATCH_PID)\"}"
    rm -f "$PID_FILE"
    exit 0
  fi
done

kill -KILL "$WATCH_PID" 2>/dev/null || true
sleep 0.2

if ! ps -p "$WATCH_PID" > /dev/null 2>&1; then
  echo "{\"message\": \"sgrep watch force-stopped for session $SESSION_ID (PID: $WATCH_PID)\"}"
else
  echo "{\"warning\": \"Failed to stop watch process $WATCH_PID\"}" >&2
fi

rm -f "$PID_FILE"

