"""Linux & Ubuntu environment detection, daemon PID/service helpers, and diagnostics."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Literal

from models.downloader import get_shardai_home

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
ALL_CLI_SUBCOMMANDS: tuple[str, ...] = (
    "create",
    "join",
    "serve",
    "status",
    "stop",
    "rooms",
    "models",
    "pull",
    "peers",
    "chat",
    "network",
    "doctor",
    "service",
    "completion",
    "version",
)
SERVICE_ACTIONS: tuple[str, ...] = (
    "unit",
    "install",
    "uninstall",
    "status",
    "start",
    "stop",
    "restart",
    "logs",
)
_UNBOUNDED_CGROUP_SENTINEL: int = 1 << 60


@dataclass(frozen=True, slots=True)
class LinuxDistroInfo:
    """Parsed host distribution metadata from `/etc/os-release` and WSL probes."""

    id: str
    name: str
    pretty_name: str
    version_id: str
    id_like: str
    is_ubuntu: bool
    is_debian_family: bool
    is_wsl: bool


def validate_loopback_address(host: str) -> str:
    """Validate that `host` is a loopback address (`127.0.0.1`, `localhost`, or `::1`)."""
    cleaned = host.strip().lower()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned not in ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Control plane host must be loopback-only (127.0.0.1, localhost, or ::1); got '{host}'."
        )
    return cleaned


def parse_os_release(content: str, *, is_wsl: bool = False) -> LinuxDistroInfo:
    """Parse `/etc/os-release` key-value lines into a `LinuxDistroInfo` record."""
    fields: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        val = raw_value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        fields[key] = val

    distro_id = fields.get("ID", "").strip().lower()
    id_like = fields.get("ID_LIKE", "").strip().lower()
    like_tokens = set(id_like.split())
    name = fields.get("NAME", "").strip() or (distro_id.capitalize() if distro_id else "Linux")
    pretty_name = fields.get("PRETTY_NAME", "").strip() or name
    version_id = fields.get("VERSION_ID", "").strip()

    is_ubuntu = distro_id == "ubuntu" or "ubuntu" in like_tokens
    is_debian_family = distro_id in (
        "debian",
        "ubuntu",
        "linuxmint",
        "pop",
        "raspbian",
        "elementary",
        "zorin",
    ) or bool(like_tokens & {"debian", "ubuntu"})

    return LinuxDistroInfo(
        id=distro_id or "linux",
        name=name,
        pretty_name=pretty_name,
        version_id=version_id,
        id_like=id_like,
        is_ubuntu=is_ubuntu,
        is_debian_family=is_debian_family,
        is_wsl=is_wsl,
    )


def detect_linux_distro(
    os_release_path: Path = Path("/etc/os-release"),
    proc_version_path: Path = Path("/proc/version"),
) -> LinuxDistroInfo:
    """Inspect `/etc/os-release` and `/proc/version` to identify Ubuntu/Debian/WSL hosts."""
    is_wsl = False
    try:
        if proc_version_path.is_file():
            proc_text = proc_version_path.read_text(encoding="utf-8", errors="ignore").lower()
            is_wsl = "microsoft" in proc_text or "wsl" in proc_text
    except OSError:
        is_wsl = False

    if not is_wsl and proc_version_path == Path("/proc/version"):
        is_wsl = bool(os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"))

    try:
        if os_release_path.is_file():
            content = os_release_path.read_text(encoding="utf-8", errors="ignore")
            return parse_os_release(content, is_wsl=is_wsl)
    except OSError:
        pass

    sys_name = platform.system() or "Unknown"
    return LinuxDistroInfo(
        id=sys_name.lower(),
        name=sys_name,
        pretty_name=platform.platform(),
        version_id=platform.release(),
        id_like="",
        is_ubuntu=False,
        is_debian_family=False,
        is_wsl=is_wsl,
    )


def detect_linux_target_triple(machine: str | None = None) -> str:
    """Map CPU architecture string to the Linux GNU target triple."""
    arch = (machine if machine is not None else platform.machine()).strip().lower()
    mapping: dict[str, str] = {
        "x86_64": "x86_64-unknown-linux-gnu",
        "amd64": "x86_64-unknown-linux-gnu",
        "aarch64": "aarch64-unknown-linux-gnu",
        "arm64": "aarch64-unknown-linux-gnu",
    }
    if arch in mapping:
        return mapping[arch]
    safe_arch = arch or "x86_64"
    return f"{safe_arch}-unknown-linux-gnu"


def _parse_positive_cgroup_int(file_path: Path) -> int | None:
    """Read a positive integer from a cgroup file, returning `None` for `'max'` or sentinels."""
    try:
        if not file_path.is_file():
            return None
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    if not raw or raw.lower() == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0 or value >= _UNBOUNDED_CGROUP_SENTINEL:
        return None
    return value


def read_cgroup_memory_limit(
    cgroup_v2_root: Path = Path("/sys/fs/cgroup"),
    cgroup_v1_root: Path = Path("/sys/fs/cgroup/memory"),
) -> tuple[int | None, int | None]:
    """Return finite `(limit_bytes, current_bytes)` from Linux cgroup v2 or v1, if constrained."""
    v2_limit = _parse_positive_cgroup_int(cgroup_v2_root / "memory.max")
    if v2_limit is not None:
        v2_current = _parse_positive_cgroup_int(cgroup_v2_root / "memory.current")
        return v2_limit, v2_current

    v1_limit = _parse_positive_cgroup_int(cgroup_v1_root / "memory.limit_in_bytes")
    if v1_limit is not None:
        v1_current = _parse_positive_cgroup_int(cgroup_v1_root / "memory.usage_in_bytes")
        return v1_limit, v1_current

    return None, None


def _ensure_private_dir(path: Path) -> Path:
    """Create `path` if missing and tighten permissions to `0o700` on POSIX."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


def get_runtime_dir() -> Path:
    """Resolve writable per-user directory for ShardAi daemon PID/runtime files."""
    override = os.getenv("SHARDAI_RUNTIME_DIR", "").strip()
    if override:
        return _ensure_private_dir(Path(override).expanduser().resolve())

    if platform.system() == "Linux":
        xdg_runtime = os.getenv("XDG_RUNTIME_DIR", "").strip()
        if xdg_runtime:
            candidate = Path(xdg_runtime).expanduser()
            if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
                try:
                    return _ensure_private_dir((candidate / "shardai").resolve())
                except OSError:
                    pass

    return _ensure_private_dir((get_shardai_home() / "run").resolve())


def get_log_dir() -> Path:
    """Resolve writable directory for ShardAi background daemon logs."""
    override = os.getenv("SHARDAI_LOG_DIR", "").strip()
    if override:
        return _ensure_private_dir(Path(override).expanduser().resolve())

    if platform.system() == "Linux":
        xdg_state = os.getenv("XDG_STATE_HOME", "").strip()
        if xdg_state:
            try:
                return _ensure_private_dir(
                    (Path(xdg_state).expanduser() / "shardai" / "logs").resolve()
                )
            except OSError:
                pass

    return _ensure_private_dir((get_shardai_home() / "logs").resolve())


def get_daemon_pid_file(port: int) -> Path:
    """Return the PID file path for the local ShardAi daemon on `port`."""
    return get_runtime_dir() / f"shardai-{port}.pid"


def get_daemon_log_file(port: int) -> Path:
    """Return the log file path for the local ShardAi daemon on `port`."""
    return get_log_dir() / f"shardai-{port}.log"


def is_process_alive(pid: int) -> bool:
    """Return True when `pid` corresponds to a live OS process."""
    if pid <= 0:
        return False

    if os.name == "nt":
        try:
            process_query_limited_info = 0x1000
            still_active = 259
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(process_query_limited_info, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong(0)
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return int(exit_code.value) == still_active
                return True
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def read_daemon_pid(port: int) -> int | None:
    """Read a valid positive integer PID from the daemon PID file, or return `None`."""
    pid_file = get_daemon_pid_file(port)
    if not pid_file.is_file():
        return None
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw)
        return pid if pid > 0 else None
    except (OSError, ValueError):
        return None


def write_daemon_pid(port: int, pid: int) -> Path:
    """Atomically record the daemon PID to disk with restricted user-only permissions."""
    if pid <= 0:
        raise ValueError(f"Daemon PID must be positive; got {pid}.")
    pid_file = get_daemon_pid_file(port)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pid_file.with_suffix(".pid.tmp")
    tmp_path.write_text(f"{pid}\n", encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
    os.replace(tmp_path, pid_file)
    return pid_file


def clear_daemon_pid(port: int) -> None:
    """Remove the daemon PID file for `port` if it exists."""
    try:
        get_daemon_pid_file(port).unlink(missing_ok=True)
    except OSError:
        pass


def terminate_daemon_process(port: int, timeout_seconds: float = 5.0) -> tuple[bool, str]:
    """Terminate the background ShardAi daemon recorded for `port` and clean its PID file."""
    pid = read_daemon_pid(port)
    if pid is None:
        return False, f"No background daemon PID file found for port {port}."

    if not is_process_alive(pid):
        clear_daemon_pid(port)
        return False, f"Removed stale PID file for inactive daemon (PID {pid})."

    term_sig = getattr(signal, "SIGTERM", signal.SIGABRT)
    try:
        os.kill(pid, term_sig)
    except ProcessLookupError:
        clear_daemon_pid(port)
        return True, f"ShardAi daemon (PID {pid}) already exited."
    except PermissionError as exc:
        return False, f"Insufficient permissions to stop daemon PID {pid}: {exc}"
    except OSError as exc:
        return False, f"Failed to send termination signal to daemon PID {pid}: {exc}"

    deadline = time.monotonic() + max(0.2, timeout_seconds)
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            clear_daemon_pid(port)
            return True, f"Stopped ShardAi background daemon (PID {pid})."
        time.sleep(0.05)

    return False, f"Timed out after {timeout_seconds:.1f}s waiting for daemon PID {pid} to exit."


def detect_accelerator() -> dict[str, object]:
    """Detect available GPU/CPU compute acceleration safely without throwing."""
    nvidia_smi_found = shutil.which("nvidia-smi") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    backend = "cpu"
    device_name: str | None = None
    device_count = 0

    if torch_installed:
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
                backend = "rocm" if is_rocm else "cuda"
                device_count = int(torch.cuda.device_count())
                if device_count > 0:
                    device_name = str(torch.cuda.get_device_name(0))
            elif (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                backend = "mps"
                device_count = 1
                device_name = "Apple Silicon MPS"
        except Exception:
            pass

    return {
        "backend": backend,
        "device_name": device_name,
        "device_count": device_count,
        "nvidia_smi_found": nvidia_smi_found,
        "torch_installed": torch_installed,
    }


def render_systemd_unit(
    exec_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    scope: Literal["user", "system"] = "user",
    tracker_url: str | None = None,
    offline: bool = False,
) -> str:
    """Render a hardened loopback-only `systemd` `.service` unit for `shardai serve`."""
    safe_host = validate_loopback_address(host)
    if not (1024 <= port <= 65535):
        raise ValueError(f"Port must be between 1024 and 65535; got {port}.")
    if scope not in ("user", "system"):
        raise ValueError(f"Scope must be 'user' or 'system'; got '{scope}'.")

    cleaned_exec = exec_path.strip() or "/usr/bin/shardai"
    quoted_exec = shlex.quote(cleaned_exec) if " " in cleaned_exec else cleaned_exec
    exec_parts = [quoted_exec, "serve", "--host", safe_host, "--port", str(port)]
    if offline:
        exec_parts.append("--offline")

    lines: list[str] = [
        "[Unit]",
        "Description=ShardAi Local OpenAI Gateway and P2P LLM Node Daemon",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={' '.join(exec_parts)}",
        "Restart=on-failure",
        "RestartSec=5",
        "KillSignal=SIGTERM",
        "TimeoutStopSec=15",
        "Environment=PYTHONUNBUFFERED=1",
    ]
    if tracker_url and tracker_url.strip():
        lines.append(f"Environment=SHARDAI_TRACKER_URL={tracker_url.strip()}")
    if offline:
        lines.append("Environment=SHARDAI_OFFLINE_MODE=true")

    if scope == "system":
        lines.extend(
            [
                "User=shardai",
                "Group=shardai",
                "WorkingDirectory=/var/lib/shardai",
                "Environment=SHARDAI_HOME_DIR=/var/lib/shardai",
                "StateDirectory=shardai",
                "RuntimeDirectory=shardai",
                "LogsDirectory=shardai",
                "NoNewPrivileges=true",
                "PrivateTmp=true",
                "ProtectSystem=strict",
                "ProtectHome=true",
                "ReadWritePaths=/var/lib/shardai",
            ]
        )

    lines.extend(
        [
            "",
            "[Install]",
            f"WantedBy={'multi-user.target' if scope == 'system' else 'default.target'}",
        ]
    )
    return "\n".join(lines) + "\n"


def get_systemd_unit_path(scope: Literal["user", "system"] = "user") -> Path:
    """Return the target `.service` unit path for user or system `systemd` scope."""
    if scope == "system":
        return Path("/etc/systemd/system/shardai.service")
    return Path.home() / ".config" / "systemd" / "user" / "shardai.service"


def render_shell_completion(shell: Literal["bash", "zsh"]) -> str:
    """Render Bash or Zsh shell completion script for all `shardai` subcommands and flags."""
    commands = " ".join(ALL_CLI_SUBCOMMANDS)
    service_actions = " ".join(SERVICE_ACTIONS)

    if shell == "bash":
        return (
            "# Bash completion for shardai CLI\n"
            "_shardai_completions() {\n"
            "  local cur prev cmd\n"
            "  cur=\"${COMP_WORDS[COMP_CWORD]}\"\n"
            "  prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n"
            "  cmd=\"\"\n"
            "  local i=1\n"
            "  while [[ $i -lt $COMP_CWORD ]]; do\n"
            "    local word=\"${COMP_WORDS[$i]}\"\n"
            "    if [[ \"$word\" != -* ]]; then\n"
            "      cmd=\"$word\"\n"
            "      break\n"
            "    fi\n"
            "    ((i++))\n"
            "  done\n"
            "\n"
            f"  local subcommands=\"{commands}\"\n"
            "  if [[ -z \"$cmd\" ]]; then\n"
            "    if [[ \"$cur\" == -* ]]; then\n"
            "      COMPREPLY=( $(compgen -W \"--help --version --mode --host --port --offline --online\" -- \"$cur\") )\n"
            "    else\n"
            "      COMPREPLY=( $(compgen -W \"$subcommands\" -- \"$cur\") )\n"
            "    fi\n"
            "    return 0\n"
            "  fi\n"
            "\n"
            "  case \"$cmd\" in\n"
            "    create)\n"
            "      COMPREPLY=( $(compgen -W \"--model --room --room-key --unlisted --no-unlisted --ram --network --source-type --filename --p2p-port --ssd-offload --no-ssd-offload --offline --online --node-id --host --port --foreground --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    join)\n"
            "      COMPREPLY=( $(compgen -W \"--room --model --room-key --ram --network --source-type --filename --p2p-port --ssd-offload --no-ssd-offload --offline --online --node-id --host --port --foreground --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    serve)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --offline --online --background --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    status)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --api-url --show-key --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    stop)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --api-url --shutdown-daemon --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    rooms|models|network)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --api-url --offline --online --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    pull)\n"
            "      COMPREPLY=( $(compgen -W \"--model --source-type --filename --ram --host --port --api-url --offline --online --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    peers)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --api-url --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    chat)\n"
            "      COMPREPLY=( $(compgen -W \"--prompt --model --system --max-tokens --temperature --stream --no-stream --api-key --host --port --api-url --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    doctor)\n"
            "      COMPREPLY=( $(compgen -W \"--host --port --api-url --json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    service)\n"
            "      if [[ \"$cur\" == -* ]]; then\n"
            "        COMPREPLY=( $(compgen -W \"--scope --host --port --tracker-url --offline --online --enable --no-enable --json --help\" -- \"$cur\") )\n"
            "      else\n"
            f"        COMPREPLY=( $(compgen -W \"{service_actions}\" -- \"$cur\") )\n"
            "      fi\n"
            "      ;;\n"
            "    completion)\n"
            "      COMPREPLY=( $(compgen -W \"bash zsh --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "    version)\n"
            "      COMPREPLY=( $(compgen -W \"--json --help\" -- \"$cur\") )\n"
            "      ;;\n"
            "  esac\n"
            "}\n"
            "complete -F _shardai_completions shardai\n"
        )

    if shell == "zsh":
        return (
            "#compdef shardai\n"
            "# Zsh completion for shardai CLI\n"
            "_shardai() {\n"
            "  local -a subcommands\n"
            "  subcommands=(\n"
            "    'create:Create a new ShardAi room and share RAM/VRAM'\n"
            "    'join:Join an existing ShardAi room'\n"
            "    'serve:Start the local ShardAi FastAPI daemon'\n"
            "    'status:Show active node status and masked API key'\n"
            "    'stop:Stop the active node and release memory'\n"
            "    'rooms:List available Internet and LAN rooms'\n"
            "    'models:List curated catalog models'\n"
            "    'pull:Download and cache model weights'\n"
            "    'peers:List direct P2P routing peers'\n"
            "    'chat:Chat with the active ShardAi node'\n"
            "    'network:Check Signaling Tracker and host memory'\n"
            "    'doctor:Inspect Linux/Ubuntu and runtime readiness'\n"
            "    'service:Generate or manage systemd daemon unit'\n"
            "    'completion:Output bash or zsh completion script'\n"
            "    'version:Print installed ShardAi CLI version'\n"
            "  )\n"
            "\n"
            "  if (( CURRENT == 2 )); then\n"
            "    _describe -t commands 'shardai commands' subcommands\n"
            "    return\n"
            "  fi\n"
            "\n"
            "  case \"${words[2]}\" in\n"
            "    service)\n"
            f"      _values 'action' {service_actions}\n"
            "      ;;\n"
            "    completion)\n"
            "      _values 'shell' bash zsh\n"
            "      ;;\n"
            "    *)\n"
            "      _arguments \\\n"
            "        '--help[Show command help]' \\\n"
            "        '--json[Output machine-readable JSON]' \\\n"
            "        '--host[Loopback daemon host]:host:(127.0.0.1 localhost ::1)' \\\n"
            "        '--port[Loopback daemon port]:port:'\n"
            "      ;;\n"
            "  esac\n"
            "}\n"
            "_shardai \"$@\"\n"
        )

    raise ValueError(f"Unsupported shell '{shell}'; choose 'bash' or 'zsh'.")


def _is_dir_writable(path: Path) -> bool:
    """Return True if `path` exists or can be created and is writable by the current user."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".shardai-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def collect_environment_diagnostics(
    host: str,
    port: int,
    daemon_reachable: bool,
) -> dict[str, object]:
    """Collect structured Linux/Ubuntu and cross-platform host diagnostics for `shardai doctor`."""
    from shard_engine.memory_manager import get_system_memory_info

    distro = detect_linux_distro()
    arch = platform.machine() or "unknown"
    target_triple = detect_linux_target_triple(arch)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)

    total_gb, avail_gb, max_share_gb = get_system_memory_info()
    cgroup_limit, cgroup_current = (
        read_cgroup_memory_limit() if platform.system() == "Linux" else (None, None)
    )
    accelerator = detect_accelerator()

    deps = {
        "llama_cpp": importlib.util.find_spec("llama_cpp") is not None,
        "zeroconf": importlib.util.find_spec("zeroconf") is not None,
        "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
        "torch": bool(accelerator.get("torch_installed")),
    }

    systemctl_available = shutil.which("systemctl") is not None
    systemd_pid1 = False
    if platform.system() == "Linux":
        try:
            systemd_pid1 = Path("/proc/1/comm").read_text(encoding="utf-8").strip() == "systemd"
        except OSError:
            systemd_pid1 = Path("/run/systemd/system").is_dir()

    shardai_home = get_shardai_home()
    runtime_dir = get_runtime_dir()
    log_dir = get_log_dir()
    home_writable = _is_dir_writable(shardai_home)
    daemon_pid = read_daemon_pid(port)
    daemon_pid_alive = is_process_alive(daemon_pid) if daemon_pid is not None else False

    checks: list[dict[str, str]] = [
        {
            "name": "python_runtime",
            "status": "pass" if py_ok else "fail",
            "detail": f"Python {py_ver} ({'supported >= 3.11' if py_ok else 'requires >= 3.11'})",
        },
        {
            "name": "host_memory",
            "status": "pass" if avail_gb >= 2.0 else "warn",
            "detail": (
                f"{avail_gb:.2f} GB free / {total_gb:.2f} GB total "
                f"(max shareable: {max_share_gb} GB)"
            ),
        },
        {
            "name": "home_storage",
            "status": "pass" if home_writable else "fail",
            "detail": f"{shardai_home} ({'writable' if home_writable else 'not writable'})",
        },
        {
            "name": "loopback_daemon",
            "status": "pass" if daemon_reachable else "warn",
            "detail": (
                f"Reachable at http://{host}:{port}"
                if daemon_reachable
                else f"Not running at http://{host}:{port} (start with 'shardai serve')"
            ),
        },
        {
            "name": "gguf_engine",
            "status": "pass" if deps["llama_cpp"] else "warn",
            "detail": (
                "llama-cpp-python installed"
                if deps["llama_cpp"]
                else "llama-cpp-python not installed (install optional [gguf] extra for native GGUF)"
            ),
        },
        {
            "name": "lan_discovery",
            "status": "pass" if deps["zeroconf"] else "warn",
            "detail": (
                "zeroconf mDNS available"
                if deps["zeroconf"]
                else "zeroconf missing; UDP multicast fallback only"
            ),
        },
    ]

    overall_ok = all(item["status"] != "fail" for item in checks)
    return {
        "ok": overall_ok,
        "os": platform.system(),
        "platform": platform.platform(),
        "architecture": arch,
        "target_triple": target_triple,
        "distro": asdict(distro),
        "python": {
            "version": py_ver,
            "executable": sys.executable,
            "ok": py_ok,
        },
        "memory": {
            "total_ram_gb": total_gb,
            "available_ram_gb": avail_gb,
            "max_shareable_ram_gb": max_share_gb,
            "cgroup_limit_bytes": cgroup_limit,
            "cgroup_current_bytes": cgroup_current,
        },
        "accelerator": accelerator,
        "dependencies": deps,
        "systemd": {
            "systemctl_available": systemctl_available,
            "systemd_pid1": systemd_pid1,
            "user_unit_exists": get_systemd_unit_path("user").is_file(),
            "system_unit_exists": get_systemd_unit_path("system").is_file(),
        },
        "paths": {
            "shardai_home": str(shardai_home),
            "runtime_dir": str(runtime_dir),
            "log_dir": str(log_dir),
            "home_writable": home_writable,
        },
        "daemon": {
            "host": host,
            "port": port,
            "reachable": daemon_reachable,
            "pid": daemon_pid,
            "pid_alive": daemon_pid_alive,
        },
        "checks": checks,
    }


def run_systemctl(
    args: list[str],
    *,
    scope: Literal["user", "system"],
) -> subprocess.CompletedProcess[str]:
    """Run `systemctl` for user or system unit management."""
    cmd = ["systemctl"]
    if scope == "user":
        cmd.append("--user")
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)
