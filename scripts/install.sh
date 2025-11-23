#!/usr/bin/env bash
set -euo pipefail

REPO=${SGREP_REPO:-"rika-labs/sgrep"}
INSTALL_DIR=${INSTALL_DIR:-"/usr/local/bin"}

detect_os_arch() {
  local os arch
  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *) echo "Unsupported OS $(uname -s)" >&2; exit 1 ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac

  printf '%s-%s' "$os" "$arch"
}

download_and_install() {
  local platform
  platform=$(detect_os_arch)
  local asset="sgrep-${platform}.tar.gz"
  local url="https://github.com/${REPO}/releases/latest/download/${asset}"
  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT

  echo "Downloading $asset ..."
  curl -fsSL "$url" -o "$tmp/$asset"

  echo "Extracting..."
  tar -C "$tmp" -xzf "$tmp/$asset"

  if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
  fi

  echo "Installing to $INSTALL_DIR (may require sudo)..."
  install -m 0755 "$tmp/sgrep" "$INSTALL_DIR/sgrep"

  echo "sgrep installed at $(command -v sgrep)"
  sgrep --version || true
}

download_and_install
