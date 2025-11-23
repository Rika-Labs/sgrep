#!/usr/bin/env sh
set -eu

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
  os_name="$(uname -s)"
  case "$os_name" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *) echo "Unsupported OS $os_name" >&2; exit 1 ;;
  esac

  arch_name="$(uname -m)"
  case "$arch_name" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) echo "Unsupported architecture $arch_name" >&2; exit 1 ;;
  esac

  printf '%s-%s' "$os" "$arch"
}

download_and_install() {
  platform="$(detect_os_arch)"
  asset="sgrep-${platform}.tar.gz"
  url="https://github.com/${REPO}/releases/latest/download/${asset}"
  TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t sgrep)"

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
  sgrep --version || :
}

download_and_install
