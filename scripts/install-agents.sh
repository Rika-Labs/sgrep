#!/bin/bash
set -euo pipefail

# install-agents.sh
#
# Installs the sgrep skill + watch lifecycle for one or more coding agents.
# Works both from a local clone and via curl:
#
#   curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s claude
#   curl -fsSL https://raw.githubusercontent.com/rika-labs/sgrep/main/scripts/install-agents.sh | sh -s all

REPO="rika-labs/sgrep"
BRANCH="main"
RAW="https://raw.githubusercontent.com/$REPO/$BRANCH"

# Detect if running from a local clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
REPO_ROOT=""
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../.factory/skills/sgrep/SKILL.md" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

usage() {
  echo "Usage: $0 <agent>"
  echo ""
  echo "Agents:"
  echo "  claude    Claude Code — skill + SessionStart/Stop hooks"
  echo "  codex     Codex CLI   — skill + session-start hook"
  echo "  pi        Pi          — skill (no frontmatter) + extension"
  echo "  opencode  OpenCode    — prints plugin config instructions"
  echo "  all       Install all agents"
  echo ""
  echo "Remote install:"
  echo "  curl -fsSL $RAW/scripts/install-agents.sh | sh -s claude"
  exit 1
}

download() {
  local url="$1" dest="$2"
  if command -v curl &> /dev/null; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget &> /dev/null; then
    wget -q "$url" -O "$dest"
  else
    echo "error: curl or wget required" >&2
    exit 1
  fi
}

# Copy from local clone or download from GitHub
fetch_file() {
  local repo_path="$1" dest="$2"
  if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/$repo_path" ]; then
    cp "$REPO_ROOT/$repo_path" "$dest"
  else
    download "$RAW/$repo_path" "$dest"
  fi
}

fetch_skill() {
  fetch_file ".factory/skills/sgrep/SKILL.md" "$1"
}

fetch_skill_no_frontmatter() {
  local dest="$1"
  local tmp
  tmp=$(mktemp)
  fetch_skill "$tmp"
  awk 'BEGIN{skip=0} /^---$/{skip++; next} skip<2{next} {print}' "$tmp" > "$dest"
  rm -f "$tmp"
}

install_claude() {
  echo "=== Claude Code ==="

  mkdir -p "$HOME/.claude/skills/sgrep"
  fetch_skill "$HOME/.claude/skills/sgrep/SKILL.md"
  echo "  skill: ~/.claude/skills/sgrep/SKILL.md"

  local settings="$HOME/.claude/settings.json"
  if [ -f "$settings" ]; then
    local has_hooks
    has_hooks=$(python3 -c "
import json
d = json.load(open('$settings'))
hooks = d.get('hooks', {})
has_start = any(
    any('sgrep' in h.get('command','') for h in entry.get('hooks',[]))
    for entry in hooks.get('SessionStart', [])
)
print('yes' if has_start else 'no')
" 2>/dev/null || echo "no")

    if [ "$has_hooks" = "yes" ]; then
      echo "  hooks: already configured in $settings"
    else
      python3 -c "
import json
f = '$settings'
d = json.load(open(f))
hooks = d.setdefault('hooks', {})
hooks.setdefault('SessionStart', []).append({
    'matcher': '',
    'hooks': [{'type': 'command', 'command': 'sgrep index > /dev/null 2>&1 || true; sgrep watch --detach > /dev/null 2>&1 || true', 'timeout': 60}]
})
hooks.setdefault('Stop', []).append({
    'matcher': '',
    'hooks': [{'type': 'command', 'command': 'pkill -f \"sgrep watch\" 2>/dev/null || true', 'timeout': 10}]
})
json.dump(d, open(f, 'w'), indent=2)
print('  hooks: configured SessionStart + Stop in $settings')
"
    fi
  else
    python3 -c "
import json
d = {
    'hooks': {
        'SessionStart': [{
            'matcher': '',
            'hooks': [{'type': 'command', 'command': 'sgrep index > /dev/null 2>&1 || true; sgrep watch --detach > /dev/null 2>&1 || true', 'timeout': 60}]
        }],
        'Stop': [{
            'matcher': '',
            'hooks': [{'type': 'command', 'command': 'pkill -f \"sgrep watch\" 2>/dev/null || true', 'timeout': 10}]
        }]
    }
}
json.dump(d, open('$settings', 'w'), indent=2)
print('  hooks: created $settings with SessionStart + Stop')
"
  fi
  echo ""
}

install_codex() {
  echo "=== Codex CLI ==="

  mkdir -p "$HOME/.agents/skills/sgrep"
  fetch_skill "$HOME/.agents/skills/sgrep/SKILL.md"
  echo "  skill: ~/.agents/skills/sgrep/SKILL.md"

  mkdir -p "$HOME/.codex/hooks"
  fetch_file "plugins/codex/hooks/session-start.sh" "$HOME/.codex/hooks/session-start.sh"
  chmod +x "$HOME/.codex/hooks/session-start.sh"
  echo "  hook:  ~/.codex/hooks/session-start.sh"
  echo ""
}

install_pi() {
  echo "=== Pi ==="

  local PI_PKG="https://github.com/$REPO?subdir=plugins/pi"
  local settings="$HOME/.pi/agent/settings.json"

  if [ -f "$settings" ]; then
    local has_pkg
    has_pkg=$(python3 -c "
import json
d = json.load(open('$settings'))
pkgs = d.get('packages', [])
print('yes' if any('sgrep' in p for p in pkgs) else 'no')
" 2>/dev/null || echo "no")

    if [ "$has_pkg" = "yes" ]; then
      echo "  package: already in $settings"
    else
      python3 -c "
import json
f = '$settings'
d = json.load(open(f))
d.setdefault('packages', []).append('$PI_PKG')
json.dump(d, open(f, 'w'), indent=2)
print('  package: added to $settings')
"
    fi
  else
    echo "  Pi settings not found at $settings"
    echo "  Add manually in Pi: /install $PI_PKG"
  fi
  echo "  Restart Pi to load the plugin."
  echo ""
}

install_opencode() {
  echo "=== OpenCode ==="
  echo "  Add to your OpenCode config:"
  echo ""
  echo '    { "plugins": ["sgrep-opencode"] }'
  echo ""
}

if [ $# -eq 0 ]; then
  usage
fi

for agent in "$@"; do
  case "$agent" in
    claude)   install_claude ;;
    codex)    install_codex ;;
    pi)       install_pi ;;
    opencode) install_opencode ;;
    all)
      install_claude
      install_codex
      install_pi
      install_opencode
      ;;
    *)
      echo "error: unknown agent '$agent'" >&2
      usage
      ;;
  esac
done

echo "Done."
