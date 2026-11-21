#!/usr/bin/env bash
# Install the `shardai` CLI command onto the user's PATH (macOS / Linux / Ubuntu).
#
# Usage:
#   ./scripts/install_cli.sh
#   ./scripts/install_cli.sh --binary
#   SHARDAI_INSTALL_DIR="$HOME/.local/bin" ./scripts/install_cli.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORE_DIR="${REPO_ROOT}/core-engine"
INSTALL_DIR="${SHARDAI_INSTALL_DIR:-${HOME}/.local/bin}"
USE_BINARY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      USE_BINARY="true"
      shift
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/install_cli.sh [--binary] [--install-dir DIR]

Options:
  --binary           Install the standalone PyInstaller binary from dist/cli/shardai
  --install-dir DIR  Destination bin directory (default: $HOME/.local/bin)
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${INSTALL_DIR}"
TARGET_BIN="${INSTALL_DIR}/shardai"

if [[ "${USE_BINARY}" == "true" ]]; then
  SOURCE_BIN="${REPO_ROOT}/dist/cli/shardai"
  if [[ ! -f "${SOURCE_BIN}" ]]; then
    echo "Error: Standalone CLI binary not found at ${SOURCE_BIN}. Run './scripts/build_core.sh --cli-only' first." >&2
    exit 1
  fi
  cp -f "${SOURCE_BIN}" "${TARGET_BIN}"
  chmod +x "${TARGET_BIN}"
  echo "==> Installed standalone CLI binary: ${TARGET_BIN}"
else
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_EXE="${PYTHON_BIN}"
  elif [[ -x "${CORE_DIR}/.venv/bin/python" ]]; then
    PYTHON_EXE="${CORE_DIR}/.venv/bin/python"
  else
    BASE_PYTHON=""
    if command -v python3 >/dev/null 2>&1; then
      BASE_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
      BASE_PYTHON="$(command -v python)"
    else
      echo "Error: Python 3.11+ is required. On Ubuntu/Debian run: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" >&2
      exit 1
    fi

    # Create an isolated venv to comply with PEP 668 on Ubuntu 24.04+ / Debian 12+
    echo "==> Creating isolated virtualenv at ${CORE_DIR}/.venv (PEP 668 safe)..."
    if ! "${BASE_PYTHON}" -m venv "${CORE_DIR}/.venv" >/dev/null 2>&1; then
      echo "Error: Failed to create Python virtualenv. On Ubuntu/Debian run: sudo apt-get install -y python3-venv" >&2
      exit 1
    fi
    PYTHON_EXE="${CORE_DIR}/.venv/bin/python"
  fi

  if ! "${PYTHON_EXE}" -c "import typer, fastapi, httpx, pydantic" >/dev/null 2>&1; then
    echo "==> Installing Python dependencies from ${CORE_DIR}/requirements.txt..."
    "${PYTHON_EXE}" -m pip install --upgrade pip
    "${PYTHON_EXE}" -m pip install -r "${CORE_DIR}/requirements.txt"
  fi

  echo "==> Installing shardai CLI package from ${CORE_DIR} using ${PYTHON_EXE}..."
  "${PYTHON_EXE}" -m pip install --no-build-isolation --no-deps -e "${CORE_DIR}"
  RESOLVED_PYTHON="$("${PYTHON_EXE}" -c 'import sys; print(sys.executable)')"

  cat > "${TARGET_BIN}" <<EOF
#!/usr/bin/env bash
exec "${RESOLVED_PYTHON}" -m shard_engine "\$@"
EOF
  chmod +x "${TARGET_BIN}"
  echo "==> Installed CLI wrapper: ${TARGET_BIN}"
fi

if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
  echo ""
  echo "Note: Add '${INSTALL_DIR}' to your shell PATH:"
  echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi

echo ""
echo "ShardAi CLI quickstart:"
echo "  shardai --help"
echo "  shardai doctor"
echo "  shardai create --model llama-3.1-8b --room my-room --ram 4"
echo "  shardai join my-room"
echo "  shardai status"
echo "  shardai chat \"Hello from ShardAi!\""
