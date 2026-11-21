#!/usr/bin/env bash
# Build an Ubuntu / Debian `.deb` package for the standalone `shardai` CLI binary.
#
# Usage:
#   ./scripts/build_deb.sh
#   ./scripts/build_deb.sh --binary dist/cli/shardai --version 0.1.0 --arch amd64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BINARY_PATH="${REPO_ROOT}/dist/cli/shardai"
PKG_VERSION=""
PKG_ARCH=""
OUTPUT_DIR="${REPO_ROOT}/dist/cli"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      BINARY_PATH="$2"
      shift 2
      ;;
    --version)
      PKG_VERSION="$2"
      shift 2
      ;;
    --arch)
      PKG_ARCH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/build_deb.sh [--binary PATH] [--version VER] [--arch amd64|arm64] [--output-dir DIR]
EOF
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      exit 1
      ;;
  esac
done

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "Error: dpkg-deb is required to build a .deb package (sudo apt-get install -y dpkg)." >&2
  exit 1
fi

if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "Error: Binary not found at ${BINARY_PATH}. Run './scripts/build_core.sh --cli-only' first." >&2
  exit 1
fi

if [[ -z "${PKG_VERSION}" ]]; then
  if [[ -f "${REPO_ROOT}/core-engine/pyproject.toml" ]]; then
    PKG_VERSION="$(grep -E '^version\s*=' "${REPO_ROOT}/core-engine/pyproject.toml" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')"
  fi
  PKG_VERSION="${PKG_VERSION:-0.1.0}"
fi

if [[ -z "${PKG_ARCH}" ]]; then
  case "$(uname -m)" in
    x86_64|amd64)  PKG_ARCH="amd64" ;;
    aarch64|arm64) PKG_ARCH="arm64" ;;
    *)             PKG_ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)" ;;
  esac
fi

mkdir -p "${OUTPUT_DIR}"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

PKG_ROOT="${STAGE_DIR}/shardai_${PKG_VERSION}_${PKG_ARCH}"
mkdir -p \
  "${PKG_ROOT}/DEBIAN" \
  "${PKG_ROOT}/usr/bin" \
  "${PKG_ROOT}/usr/lib/systemd/user" \
  "${PKG_ROOT}/lib/systemd/system" \
  "${PKG_ROOT}/usr/share/bash-completion/completions" \
  "${PKG_ROOT}/usr/share/zsh/vendor-completions"

install -m 0755 "${BINARY_PATH}" "${PKG_ROOT}/usr/bin/shardai"
install -m 0644 "${REPO_ROOT}/packaging/linux/shardai-user.service" "${PKG_ROOT}/usr/lib/systemd/user/shardai.service"
install -m 0644 "${REPO_ROOT}/packaging/linux/shardai.service" "${PKG_ROOT}/lib/systemd/system/shardai.service"
install -m 0644 "${REPO_ROOT}/packaging/linux/completions/shardai.bash" "${PKG_ROOT}/usr/share/bash-completion/completions/shardai"
install -m 0644 "${REPO_ROOT}/packaging/linux/completions/_shardai" "${PKG_ROOT}/usr/share/zsh/vendor-completions/_shardai"

cat > "${PKG_ROOT}/DEBIAN/control" <<EOF
Package: shardai
Version: ${PKG_VERSION}
Section: utils
Priority: optional
Architecture: ${PKG_ARCH}
Maintainer: ShardAi Maintainers <https://shardai.online>
Homepage: https://shardai.online
Description: ShardAi P2P Distributed LLM CLI and Local OpenAI-Compatible Gateway
 ShardAi lets consumer Linux and Ubuntu workstations pool idle RAM and VRAM
 over LAN or Internet swarms and serve a local OpenAI-compatible API gateway
 at http://127.0.0.1:8000/v1.
EOF

cat > "${PKG_ROOT}/DEBIAN/postinst" <<'EOF'
#!/usr/bin/env bash
set -e
if ! getent group shardai >/dev/null 2>&1; then
  groupadd --system shardai || true
fi
if ! getent passwd shardai >/dev/null 2>&1; then
  useradd --system --gid shardai --home-dir /var/lib/shardai --create-home --shell /usr/sbin/nologin shardai || true
fi
mkdir -p /var/lib/shardai
chown -R shardai:shardai /var/lib/shardai || true
if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
  systemctl daemon-reload >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "${PKG_ROOT}/DEBIAN/postinst"

cat > "${PKG_ROOT}/DEBIAN/prerm" <<'EOF'
#!/usr/bin/env bash
set -e
if [[ "$1" == "remove" || "$1" == "purge" ]]; then
  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    systemctl stop shardai.service >/dev/null 2>&1 || true
    systemctl disable shardai.service >/dev/null 2>&1 || true
  fi
fi
exit 0
EOF
chmod 0755 "${PKG_ROOT}/DEBIAN/prerm"

DEB_FILE="${OUTPUT_DIR}/shardai_${PKG_VERSION}_${PKG_ARCH}.deb"
dpkg-deb --build --root-owner-group "${PKG_ROOT}" "${DEB_FILE}"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "${OUTPUT_DIR}"
    sha256sum "$(basename "${DEB_FILE}")" > "$(basename "${DEB_FILE}").sha256"
  )
fi

echo "==> Debian package built: ${DEB_FILE}"
