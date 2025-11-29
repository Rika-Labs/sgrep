#!/bin/bash
set -e

# sgrep Factory Skill Installer
# Installs the sgrep skill to ~/.factory/skills/sgrep/

SKILL_DIR="$HOME/.factory/skills/sgrep"
SKILL_URL="https://raw.githubusercontent.com/rika-labs/sgrep/main/.factory/skills/sgrep/SKILL.md"

echo "Installing sgrep Factory skill..."

# Create skill directory
mkdir -p "$SKILL_DIR"

# Download skill file
if command -v curl &> /dev/null; then
    curl -fsSL "$SKILL_URL" -o "$SKILL_DIR/SKILL.md"
elif command -v wget &> /dev/null; then
    wget -q "$SKILL_URL" -O "$SKILL_DIR/SKILL.md"
else
    echo "Error: curl or wget required"
    exit 1
fi

echo "Installed sgrep skill to $SKILL_DIR"
echo "Restart Factory to use the skill."
