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
    MINGW*|MSYS*|CYGWIN*)
      echo "Error: Native Windows is not currently supported." >&2
      echo "" >&2
      echo "sgrep requires native Unix toolchains for its C++ dependencies (llama.cpp)." >&2
      echo "" >&2
      echo "Recommended alternatives:" >&2
      echo "  1. Use WSL (Windows Subsystem for Linux):" >&2
      echo "     wsl --install" >&2
      echo "     # Then run this script inside WSL" >&2
      echo "" >&2
      echo "  2. Use Docker:" >&2
      echo "     docker run -it --rm -v \$(pwd):/workspace ubuntu" >&2
      echo "     # Then run this script inside the container" >&2
      echo "" >&2
      echo "For more information, see: https://github.com/rika-labs/sgrep#windows-support" >&2
      exit 1
      ;;
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

  echo "Installing to $INSTALL_DIR..."
  if [ -w "$INSTALL_DIR" ]; then
    install -m 0755 "$TMP_DIR/sgrep" "$INSTALL_DIR/sgrep"
  else
    echo "Elevated permissions required. You may be prompted for your password."
    sudo install -m 0755 "$TMP_DIR/sgrep" "$INSTALL_DIR/sgrep"
  fi

  echo "sgrep installed at $(command -v sgrep)"
  sgrep --version || :
}

download_and_install
