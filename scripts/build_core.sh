#!/usr/bin/env bash
# Build `core-engine/main.py` into a standalone PyInstaller Onefile binary
# and rename it to Tauri's externalBin target-triple convention (macOS / Linux).
#
# Usage:
#   ./scripts/build_core.sh
#   ./scripts/build_core.sh aarch64-apple-darwin
#   ./scripts/build_core.sh x86_64-apple-darwin
#   TARGET_ARCH=universal2 ./scripts/build_core.sh universal-apple-darwin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORE_DIR="${REPO_ROOT}/core-engine"
BINARIES_DIR="${REPO_ROOT}/src-tauri/binaries"
CLI_DIST_DIR="${REPO_ROOT}/dist/cli"
PYINSTALLER_DIST_DIR="${CORE_DIR}/dist"
PYINSTALLER_WORK_DIR="${CORE_DIR}/build/pyinstaller"

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
        x86_64)        echo "x86_64-unknown-linux-gnu" ;;
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

TARGET_TRIPLE="${1:-${TARGET_TRIPLE:-$(detect_target_triple)}}"

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
echo "==> Python        : ${PYTHON_EXE}"

if ! "${PYTHON_EXE}" -m PyInstaller --version >/dev/null 2>&1; then
  echo "Error: PyInstaller is not installed in ${PYTHON_EXE}. Install it with 'pip install pyinstaller'." >&2
  exit 1
fi

mkdir -p "${BINARIES_DIR}" "${CLI_DIST_DIR}" "${PYINSTALLER_WORK_DIR}"

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

SIDECAR_TARGET="${BINARIES_DIR}/core-engine-${TARGET_TRIPLE}${EXT}"
CLI_TARGET="${CLI_DIST_DIR}/shardai-core-${TARGET_TRIPLE}${EXT}"

cp -f "${BUILT_BINARY}" "${SIDECAR_TARGET}"
cp -f "${BUILT_BINARY}" "${CLI_TARGET}"
chmod +x "${SIDECAR_TARGET}" "${CLI_TARGET}"

echo "==> Tauri Sidecar binary ready : ${SIDECAR_TARGET}"
echo "==> Standalone CLI binary ready: ${CLI_TARGET}"
