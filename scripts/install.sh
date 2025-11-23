#!/usr/bin/env bash
set -euo pipefail

REPO=${SGREP_REPO:-"rika-labs/sgrep"}
INSTALL_DIR=${INSTALL_DIR:-"/usr/local/bin"}
TMP_DIR=""

cleanup() {
  if [ -n "${TMP_DIR:-}" ] && [ -d "${TMP_DIR:-}" ]; then
    rm -rf "${TMP_DIR:-}"
  fi
}

trap cleanup EXIT

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
  TMP_DIR=$(mktemp -d)

  echo "Downloading $asset ..."
  curl -fsSL "$url" -o "$TMP_DIR/$asset"

  echo "Extracting..."
  tar -C "$TMP_DIR" -xzf "$TMP_DIR/$asset"

  if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
  fi

  echo "Installing to $INSTALL_DIR (may require sudo)..."
  install -m 0755 "$TMP_DIR/sgrep" "$INSTALL_DIR/sgrep"

  echo "sgrep installed at $(command -v sgrep)"
  sgrep --version || true
}

download_and_install
