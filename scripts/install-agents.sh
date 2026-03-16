#!/bin/bash
set -euo pipefail

# install-agents.sh
#
# Installs the sgrep skill + watch lifecycle for one or more coding agents.
#
# Usage: ./scripts/install-agents.sh <agent>
#   claude   — skill + SessionStart/Stop hooks in ~/.claude/settings.json
#   codex    — skill + session-start hook + wrapper script
#   pi       — skill (no YAML frontmatter) + extension
#   opencode — prints plugin registration instructions
#   all      — all of the above

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CANONICAL="$REPO_ROOT/.factory/skills/sgrep/SKILL.md"

if [ ! -f "$CANONICAL" ]; then
  echo "error: canonical skill not found at $CANONICAL" >&2
  exit 1
fi

usage() {
  echo "Usage: $0 <agent>"
  echo ""
  echo "Agents:"
  echo "  claude    Claude Code — skill + SessionStart/Stop hooks"
  echo "  codex     Codex CLI   — skill + session-start hook + wrapper script"
  echo "  pi        Pi          — skill (no frontmatter) + extension"
  echo "  opencode  OpenCode    — prints plugin registration instructions"
  echo "  all       Install all agents"
  exit 1
}

strip_frontmatter() {
  awk 'BEGIN{skip=0} /^---$/{skip++; next} skip<2{next} {print}' "$1"
}

install_claude() {
  echo "=== Claude Code ==="

  mkdir -p "$HOME/.claude/skills/sgrep"
  cp "$CANONICAL" "$HOME/.claude/skills/sgrep/SKILL.md"
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
  cp "$CANONICAL" "$HOME/.agents/skills/sgrep/SKILL.md"
  echo "  skill: ~/.agents/skills/sgrep/SKILL.md"

  mkdir -p "$HOME/.codex/hooks"
  cp "$REPO_ROOT/plugins/codex/hooks/session-start.sh" "$HOME/.codex/hooks/session-start.sh"
  chmod +x "$HOME/.codex/hooks/session-start.sh"
  echo "  hook:  ~/.codex/hooks/session-start.sh"

  echo "  watch: use the wrapper for full lifecycle:"
  echo "         $REPO_ROOT/plugins/codex/sgrep-watch.sh [codex args...]"
  echo ""
}

install_pi() {
  echo "=== Pi ==="

  mkdir -p "$HOME/.pi/agent/skills/sgrep"
  strip_frontmatter "$CANONICAL" > "$HOME/.pi/agent/skills/sgrep/README.md"
  echo "  skill: ~/.pi/agent/skills/sgrep/README.md (no frontmatter)"

  mkdir -p "$HOME/.pi/extensions"
  cp "$REPO_ROOT/plugins/pi/extensions/sgrep-watch.ts" "$HOME/.pi/extensions/sgrep-watch.ts"
  echo "  ext:   ~/.pi/extensions/sgrep-watch.ts"
  echo ""
}

install_opencode() {
  echo "=== OpenCode ==="
  echo "  OpenCode uses a TypeScript plugin. Add to your config:"
  echo ""
  echo '    { "plugins": ["sgrep-opencode"] }'
  echo ""
  echo "  Or for local dev:"
  echo ""
  echo "    { \"plugins\": [\"file://$REPO_ROOT/plugins/opencode\"] }"
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
