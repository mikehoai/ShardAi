#!/usr/bin/env bash
# Build `core-engine/main.py` into a standalone PyInstaller Onefile binary
# for Tauri's externalBin target-triple convention (macOS / Linux) and standalone CLI bundles.
#
# Usage:
#   ./scripts/build_core.sh
#   ./scripts/build_core.sh --cli-only
#   ./scripts/build_core.sh --cli-only x86_64-unknown-linux-gnu
#   ./scripts/build_core.sh aarch64-apple-darwin
#   TARGET_ARCH=universal2 ./scripts/build_core.sh universal-apple-darwin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORE_DIR="${REPO_ROOT}/core-engine"
BINARIES_DIR="${REPO_ROOT}/src-tauri/binaries"
CLI_DIST_DIR="${REPO_ROOT}/dist/cli"
PYINSTALLER_DIST_DIR="${CORE_DIR}/dist"
PYINSTALLER_WORK_DIR="${CORE_DIR}/build/pyinstaller"

CLI_ONLY="false"
POSITIONAL_TRIPLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli-only)
      CLI_ONLY="true"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/build_core.sh [--cli-only] [TARGET_TRIPLE]

Options:
  --cli-only      Build standalone CLI artifacts only (skip src-tauri/binaries copy)
  TARGET_TRIPLE   Optional Rust/Tauri target triple (auto-detected if omitted)
EOF
      exit 0
      ;;
    *)
      if [[ -z "${POSITIONAL_TRIPLE}" ]]; then
        POSITIONAL_TRIPLE="$1"
      else
        echo "Error: Unexpected argument '$1'" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

detect_target_triple() {
  local os_name arch_name
  os_name="$(uname -s)"
  arch_name="$(uname -m)"

  case "${os_name}" in
    Darwin)
      case "${arch_name}" in
        arm64|aarch64) echo "aarch64-apple-darwin" ;;
        x86_64)        echo "x86_64-apple-darwin" ;;
        *)             echo "${arch_name}-apple-darwin" ;;
      esac
      ;;
    Linux)
      case "${arch_name}" in
        x86_64|amd64)  echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *)             echo "${arch_name}-unknown-linux-gnu" ;;
      esac
      ;;
    MINGW*|MSYS*|CYGWIN*)
      echo "x86_64-pc-windows-msvc"
      ;;
    *)
      echo "Unsupported OS: ${os_name}" >&2
      exit 1
      ;;
  esac
}

compute_sha256() {
  local target_file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${target_file}" > "${target_file}.sha256"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${target_file}" > "${target_file}.sha256"
  fi
}

TARGET_TRIPLE="${POSITIONAL_TRIPLE:-${TARGET_TRIPLE:-$(detect_target_triple)}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXE="${PYTHON_BIN}"
elif [[ -x "${CORE_DIR}/.venv/bin/python" ]]; then
  PYTHON_EXE="${CORE_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python3)"
else
  PYTHON_EXE="$(command -v python)"
fi

echo "==> Target Triple : ${TARGET_TRIPLE}"
echo "==> CLI Only      : ${CLI_ONLY}"
echo "==> Python        : ${PYTHON_EXE}"

if ! "${PYTHON_EXE}" -m PyInstaller --version >/dev/null 2>&1; then
  echo "Error: PyInstaller is not installed in ${PYTHON_EXE}. Install it with 'pip install pyinstaller'." >&2
  exit 1
fi

mkdir -p "${CLI_DIST_DIR}" "${PYINSTALLER_WORK_DIR}"
if [[ "${CLI_ONLY}" != "true" ]]; then
  mkdir -p "${BINARIES_DIR}"
fi

PYINSTALLER_EXTRA_ARGS=()
case "${TARGET_TRIPLE}" in
  aarch64-apple-darwin)
    PYINSTALLER_EXTRA_ARGS+=(--target-arch arm64)
    ;;
  x86_64-apple-darwin)
    PYINSTALLER_EXTRA_ARGS+=(--target-arch x86_64)
    ;;
  universal-apple-darwin)
    PYINSTALLER_EXTRA_ARGS+=(--target-arch "${TARGET_ARCH:-universal2}")
    ;;
esac

(
  cd "${CORE_DIR}"
  "${PYTHON_EXE}" -m PyInstaller \
    --noconfirm \
    --clean \
    --onefile \
    --name core-engine \
    --distpath "${PYINSTALLER_DIST_DIR}" \
    --workpath "${PYINSTALLER_WORK_DIR}" \
    --specpath "${PYINSTALLER_WORK_DIR}" \
    --paths "${CORE_DIR}" \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import typer \
    --hidden-import click \
    --hidden-import zeroconf \
    --hidden-import zeroconf._utils.ipaddress \
    --hidden-import zeroconf._handlers.answers \
    --collect-submodules zeroconf \
    --collect-submodules shard_engine \
    --collect-submodules models \
    --collect-submodules p2p \
    "${PYINSTALLER_EXTRA_ARGS[@]}" \
    "${CORE_DIR}/main.py"
)

EXT=""
if [[ "${TARGET_TRIPLE}" == *"-windows-"* ]]; then
  EXT=".exe"
fi

BUILT_BINARY="${PYINSTALLER_DIST_DIR}/core-engine${EXT}"
if [[ ! -f "${BUILT_BINARY}" ]]; then
  echo "Error: Expected PyInstaller output at ${BUILT_BINARY}" >&2
  exit 1
fi

CLI_TARGET_TRIPLE="${CLI_DIST_DIR}/shardai-${TARGET_TRIPLE}${EXT}"
CLI_TARGET_SHORT="${CLI_DIST_DIR}/shardai${EXT}"

cp -f "${BUILT_BINARY}" "${CLI_TARGET_TRIPLE}"
cp -f "${BUILT_BINARY}" "${CLI_TARGET_SHORT}"
chmod +x "${CLI_TARGET_TRIPLE}" "${CLI_TARGET_SHORT}"

if [[ "${CLI_ONLY}" != "true" ]]; then
  SIDECAR_TARGET="${BINARIES_DIR}/core-engine-${TARGET_TRIPLE}${EXT}"
  cp -f "${BUILT_BINARY}" "${SIDECAR_TARGET}"
  chmod +x "${SIDECAR_TARGET}"
  echo "==> Tauri Sidecar binary ready   : ${SIDECAR_TARGET}"
fi

if [[ -z "${EXT}" ]] && command -v tar >/dev/null 2>&1; then
  (
    cd "${CLI_DIST_DIR}"
    tar -czf "shardai-${TARGET_TRIPLE}.tar.gz" "shardai"
    compute_sha256 "shardai-${TARGET_TRIPLE}.tar.gz"
    compute_sha256 "shardai-${TARGET_TRIPLE}"
  )
  echo "==> Release archive ready        : ${CLI_DIST_DIR}/shardai-${TARGET_TRIPLE}.tar.gz"
fi

echo "==> Standalone CLI binary ready  : ${CLI_TARGET_SHORT}"
echo "==> Target-triple CLI binary     : ${CLI_TARGET_TRIPLE}"
echo "==> Usage: shardai --help | shardai create --model llama-3.1-8b --ram 4"
