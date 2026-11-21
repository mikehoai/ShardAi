#!/usr/bin/env bash
# ShardAi Linux / Ubuntu Headless CLI Installer & Lifecycle Setup
#
# Supports source-based virtualenv installation (PEP 668 safe for Ubuntu 24.04+),
# standalone PyInstaller binary installation, and GitHub Release tar.gz installation.
#
# Usage:
#   ./scripts/install_linux.sh
#   ./scripts/install_linux.sh --check
#   ./scripts/install_linux.sh --source --with-systemd --shell-completion
#   ./scripts/install_linux.sh --binary ./dist/cli/shardai
#   ./scripts/install_linux.sh --release latest
#   ./scripts/install_linux.sh --uninstall [--purge]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORE_DIR="${REPO_ROOT}/core-engine"
DEFAULT_RELEASE_REPO="mikehoai/ShardAi"

MODE="source"
BINARY_PATH=""
RELEASE_VERSION="latest"
SYSTEM_INSTALL="false"
INSTALL_DIR="${SHARDAI_INSTALL_DIR:-}"
VENV_DIR="${SHARDAI_VENV_DIR:-}"
PYTHON_EXE="${PYTHON_BIN:-}"
WITH_SYSTEMD="false"
INSTALL_COMPLETION="false"
NO_PATH="false"
CHECK_ONLY="false"
UNINSTALL="false"
PURGE="false"

usage() {
  cat <<'EOF'
Usage: ./scripts/install_linux.sh [OPTIONS]

Installation modes:
  --source                  Install from local core-engine/ inside an isolated virtualenv (default)
  --binary [PATH]           Install a prebuilt standalone ELF binary (default: dist/cli/shardai)
  --release [VERSION]       Download & verify release tarball (default: latest)

Options:
  --install-dir DIR         Target bin directory (default: ~/.local/bin or /usr/local/bin with --system)
  --system                  Install system-wide (/usr/local/bin and /opt/shardai/venv; requires root)
  --venv-dir DIR            Custom Python virtualenv directory
  --python BIN              Specific Python 3.11+ executable
  --with-systemd            Install and enable shardai.service systemd unit
  --shell-completion        Install Bash and Zsh completion scripts
  --no-path                 Do not update shell rc/profile files
  --check                   Verify Ubuntu/Linux environment prerequisites without modifying disk
  --uninstall               Remove installed shardai binary, wrappers, and completions
  --purge                   Also remove ~/.shardai cached data during --uninstall
  -h, --help                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      MODE="source"
      shift
      ;;
    --binary)
      MODE="binary"
      if [[ $# -ge 2 && "$2" != -* ]]; then
        BINARY_PATH="$2"
        shift 2
      else
        shift
      fi
      ;;
    --release)
      MODE="release"
      if [[ $# -ge 2 && "$2" != -* ]]; then
        RELEASE_VERSION="$2"
        shift 2
      else
        shift
      fi
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --system)
      SYSTEM_INSTALL="true"
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --with-systemd)
      WITH_SYSTEMD="true"
      shift
      ;;
    --shell-completion)
      INSTALL_COMPLETION="true"
      shift
      ;;
    --no-path)
      NO_PATH="true"
      shift
      ;;
    --check)
      CHECK_ONLY="true"
      shift
      ;;
    --uninstall)
      UNINSTALL="true"
      shift
      ;;
    --purge)
      PURGE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${SYSTEM_INSTALL}" == "true" ]]; then
  INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
  VENV_DIR="${VENV_DIR:-/opt/shardai/venv}"
else
  INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local/bin}"
  VENV_DIR="${VENV_DIR:-${HOME}/.local/share/shardai/venv}"
fi

TARGET_BIN="${INSTALL_DIR}/shardai"

detect_distro_info() {
  DISTRO_ID="linux"
  DISTRO_PRETTY="Linux"
  DISTRO_FAMILY="linux"
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    DISTRO_ID="${ID:-linux}"
    DISTRO_PRETTY="${PRETTY_NAME:-${NAME:-Linux}}"
    if [[ "${DISTRO_ID}" == "ubuntu" || "${DISTRO_ID}" == "debian" || "${ID_LIKE:-}" == *debian* || "${ID_LIKE:-}" == *ubuntu* ]]; then
      DISTRO_FAMILY="debian"
    fi
  fi
}

detect_triple() {
  local arch_name
  arch_name="$(uname -m)"
  case "${arch_name}" in
    x86_64|amd64)  echo "x86_64-unknown-linux-gnu" ;;
    aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
    *)             echo "${arch_name}-unknown-linux-gnu" ;;
  esac
}

find_python311() {
  if [[ -n "${PYTHON_EXE}" ]]; then
    echo "${PYTHON_EXE}"
    return 0
  fi
  local cand
  for cand in python3.12 python3.11 python3 python; do
    if command -v "${cand}" >/dev/null 2>&1; then
      if "${cand}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then
        command -v "${cand}"
        return 0
      fi
    fi
  done
  return 1
}

print_apt_hint() {
  local pkgs="$1"
  if [[ "${DISTRO_FAMILY}" == "debian" ]]; then
    echo "Ubuntu/Debian prerequisite hint:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y ${pkgs}" >&2
  fi
}

check_environment() {
  detect_distro_info
  local triple
  triple="$(detect_triple)"
  echo "==> Detected OS     : ${DISTRO_PRETTY} (${DISTRO_ID})"
  echo "==> Target Triple   : ${triple}"
  echo "==> Install Mode    : ${MODE}"
  echo "==> Target Binary   : ${TARGET_BIN}"

  local missing=0
  if [[ "${MODE}" == "source" ]]; then
    local py_found=""
    if py_found="$(find_python311)"; then
      echo "==> Python 3.11+    : ${py_found} ($("${py_found}" --version 2>&1))"
      if ! "${py_found}" -m venv --help >/dev/null 2>&1; then
        echo "Error: Python venv module is missing." >&2
        print_apt_hint "python3-venv python3-pip"
        missing=1
      fi
    else
      echo "Error: Python 3.11 or newer was not found." >&2
      print_apt_hint "python3 python3-venv python3-pip"
      missing=1
    fi
  fi

  if [[ "${MODE}" == "release" ]]; then
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
      echo "Error: curl or wget is required for --release installation." >&2
      print_apt_hint "curl ca-certificates tar"
      missing=1
    fi
    if ! command -v tar >/dev/null 2>&1; then
      echo "Error: tar is required for --release extraction." >&2
      print_apt_hint "tar"
      missing=1
    fi
  fi

  if [[ "${SYSTEM_INSTALL}" == "true" && "${EUID:-$(id -u)}" -ne 0 && "${CHECK_ONLY}" != "true" ]]; then
    echo "Error: --system installation requires root privileges (re-run with sudo)." >&2
    missing=1
  fi

  return "${missing}"
}

append_path_block() {
  local rc_file="$1"
  local bin_dir="$2"
  local start_marker="# >>> shardai CLI >>>"
  local end_marker="# <<< shardai CLI <<<"

  touch "${rc_file}"
  if grep -Fq "${start_marker}" "${rc_file}"; then
    return 0
  fi

  cat >> "${rc_file}" <<EOF

${start_marker}
export PATH="${bin_dir}:\$PATH"
${end_marker}
EOF
  echo "==> Added ${bin_dir} to PATH in ${rc_file}"
}

remove_path_block() {
  local rc_file="$1"
  local start_marker="# >>> shardai CLI >>>"
  local end_marker="# <<< shardai CLI <<<"
  if [[ ! -f "${rc_file}" ]] || ! grep -Fq "${start_marker}" "${rc_file}"; then
    return 0
  fi
  local tmp_file
  tmp_file="$(mktemp)"
  awk -v s="${start_marker}" -v e="${end_marker}" '
    $0 == s { skip = 1; next }
    $0 == e { skip = 0; next }
    !skip { print }
  ' "${rc_file}" > "${tmp_file}"
  mv "${tmp_file}" "${rc_file}"
}

install_completions() {
  local bash_src="${REPO_ROOT}/packaging/linux/completions/shardai.bash"
  local zsh_src="${REPO_ROOT}/packaging/linux/completions/_shardai"
  local bash_dest zsh_dest

  if [[ "${SYSTEM_INSTALL}" == "true" ]]; then
    bash_dest="/etc/bash_completion.d/shardai"
    zsh_dest="/usr/local/share/zsh/site-functions/_shardai"
  else
    bash_dest="${HOME}/.local/share/bash-completion/completions/shardai"
    zsh_dest="${HOME}/.zsh/completions/_shardai"
  fi

  mkdir -p "$(dirname "${bash_dest}")" "$(dirname "${zsh_dest}")"
  if [[ -f "${bash_src}" ]]; then
    cp -f "${bash_src}" "${bash_dest}"
  else
    "${TARGET_BIN}" completion bash > "${bash_dest}"
  fi
  if [[ -f "${zsh_src}" ]]; then
    cp -f "${zsh_src}" "${zsh_dest}"
  else
    "${TARGET_BIN}" completion zsh > "${zsh_dest}"
  fi
  echo "==> Installed Bash completion: ${bash_dest}"
  echo "==> Installed Zsh completion : ${zsh_dest}"
}

run_uninstall() {
  echo "==> Uninstalling ShardAi CLI..."
  if command -v systemctl >/dev/null 2>&1; then
    if [[ "${SYSTEM_INSTALL}" == "true" ]]; then
      systemctl disable --now shardai.service >/dev/null 2>&1 || true
      rm -f /etc/systemd/system/shardai.service
      systemctl daemon-reload >/dev/null 2>&1 || true
    else
      systemctl --user disable --now shardai.service >/dev/null 2>&1 || true
      rm -f "${HOME}/.config/systemd/user/shardai.service"
      systemctl --user daemon-reload >/dev/null 2>&1 || true
    fi
  fi

  rm -f "${TARGET_BIN}"
  rm -rf "${VENV_DIR}"
  rm -f "${HOME}/.local/share/bash-completion/completions/shardai" "${HOME}/.zsh/completions/_shardai"
  if [[ "${SYSTEM_INSTALL}" == "true" ]]; then
    rm -f /etc/bash_completion.d/shardai /usr/local/share/zsh/site-functions/_shardai
  fi

  if [[ "${NO_PATH}" != "true" && "${SYSTEM_INSTALL}" != "true" ]]; then
    for rc in "${HOME}/.bashrc" "${HOME}/.zshrc" "${HOME}/.profile"; do
      remove_path_block "${rc}"
    done
  fi

  if [[ "${PURGE}" == "true" ]]; then
    rm -rf "${SHARDAI_HOME_DIR:-${HOME}/.shardai}"
    echo "==> Purged ShardAi data directory."
  fi

  echo "==> Uninstall completed."
}

if [[ "${UNINSTALL}" == "true" ]]; then
  run_uninstall
  exit 0
fi

if ! check_environment; then
  exit 1
fi

if [[ "${CHECK_ONLY}" == "true" ]]; then
  echo "==> Environment check passed."
  exit 0
fi

mkdir -p "${INSTALL_DIR}"

case "${MODE}" in
  source)
    if [[ ! -d "${CORE_DIR}" ]]; then
      echo "Error: core-engine directory not found at ${CORE_DIR}." >&2
      exit 1
    fi
    PY_SYS="$(find_python311)"
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
      echo "==> Creating isolated virtualenv at ${VENV_DIR}..."
      mkdir -p "$(dirname "${VENV_DIR}")"
      "${PY_SYS}" -m venv "${VENV_DIR}"
    fi
    VENV_PY="${VENV_DIR}/bin/python"
    echo "==> Installing Python dependencies inside ${VENV_DIR}..."
    "${VENV_PY}" -m pip install --upgrade pip
    "${VENV_PY}" -m pip install -r "${CORE_DIR}/requirements.txt"
    "${VENV_PY}" -m pip install --no-build-isolation --no-deps -e "${CORE_DIR}"

    cat > "${TARGET_BIN}" <<EOF
#!/usr/bin/env bash
exec "${VENV_PY}" -m shard_engine "\$@"
EOF
    chmod +x "${TARGET_BIN}"
    echo "==> Installed source CLI wrapper: ${TARGET_BIN}"
    ;;

  binary)
    SRC_BIN="${BINARY_PATH:-${REPO_ROOT}/dist/cli/shardai}"
    if [[ ! -f "${SRC_BIN}" ]]; then
      echo "Error: Standalone binary not found at ${SRC_BIN}." >&2
      exit 1
    fi
    cp -f "${SRC_BIN}" "${TARGET_BIN}"
    chmod +x "${TARGET_BIN}"
    echo "==> Installed standalone CLI binary: ${TARGET_BIN}"
    ;;

  release)
    TRIPLE="$(detect_triple)"
    ARCHIVE_NAME="shardai-${TRIPLE}.tar.gz"
    BASE_URL="https://github.com/${DEFAULT_RELEASE_REPO}/releases"
    if [[ "${RELEASE_VERSION}" == "latest" ]]; then
      DL_URL="${BASE_URL}/latest/download/${ARCHIVE_NAME}"
      SHA_URL="${BASE_URL}/latest/download/${ARCHIVE_NAME}.sha256"
    else
      DL_URL="${BASE_URL}/download/${RELEASE_VERSION}/${ARCHIVE_NAME}"
      SHA_URL="${BASE_URL}/download/${RELEASE_VERSION}/${ARCHIVE_NAME}.sha256"
    fi

    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "${TMP_DIR}"' EXIT
    echo "==> Downloading ${DL_URL}..."
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "${DL_URL}" -o "${TMP_DIR}/${ARCHIVE_NAME}"
      curl -fsSL "${SHA_URL}" -o "${TMP_DIR}/${ARCHIVE_NAME}.sha256" || true
    else
      wget -qO "${TMP_DIR}/${ARCHIVE_NAME}" "${DL_URL}"
      wget -qO "${TMP_DIR}/${ARCHIVE_NAME}.sha256" "${SHA_URL}" || true
    fi

    if [[ -s "${TMP_DIR}/${ARCHIVE_NAME}.sha256" ]]; then
      echo "==> Verifying SHA256 checksum..."
      if command -v sha256sum >/dev/null 2>&1; then
        (cd "${TMP_DIR}" && sha256sum -c "${ARCHIVE_NAME}.sha256")
      elif command -v shasum >/dev/null 2>&1; then
        (cd "${TMP_DIR}" && shasum -a 256 -c "${ARCHIVE_NAME}.sha256")
      fi
    fi

    tar -xzf "${TMP_DIR}/${ARCHIVE_NAME}" -C "${TMP_DIR}"
    if [[ ! -f "${TMP_DIR}/shardai" ]]; then
      echo "Error: Extracted archive did not contain 'shardai' executable." >&2
      exit 1
    fi
    cp -f "${TMP_DIR}/shardai" "${TARGET_BIN}"
    chmod +x "${TARGET_BIN}"
    echo "==> Installed release CLI binary: ${TARGET_BIN}"
    ;;
esac

if [[ "${NO_PATH}" != "true" && "${SYSTEM_INSTALL}" != "true" && ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
  for rc in "${HOME}/.bashrc" "${HOME}/.profile"; do
    append_path_block "${rc}" "${INSTALL_DIR}"
  done
  if [[ -f "${HOME}/.zshrc" ]]; then
    append_path_block "${HOME}/.zshrc" "${INSTALL_DIR}"
  fi
fi

if [[ "${INSTALL_COMPLETION}" == "true" ]]; then
  install_completions
fi

if [[ "${WITH_SYSTEMD}" == "true" ]]; then
  SYSTEMD_SCOPE="user"
  if [[ "${SYSTEM_INSTALL}" == "true" ]]; then
    SYSTEMD_SCOPE="system"
  fi
  echo "==> Installing systemd unit (scope=${SYSTEMD_SCOPE})..."
  "${TARGET_BIN}" service install --scope "${SYSTEMD_SCOPE}"
fi

echo "==> Verifying installation..."
"${TARGET_BIN}" --version

echo ""
echo "ShardAi Linux/Ubuntu CLI is ready!"
echo "Quickstart:"
echo "  shardai doctor"
echo "  shardai create --model llama-3.1-8b --ram 4"
echo "  shardai join my-room"
echo "  shardai serve --background"
echo "  shardai status"
echo "  shardai chat \"Hello from Ubuntu!\""
