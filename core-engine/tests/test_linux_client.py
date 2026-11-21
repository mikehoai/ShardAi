"""Comprehensive tests for Linux/Ubuntu headless client support and CLI parity (Issue #32)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from models.downloader import resolve_default_cache_dir
from p2p.mdns_discovery import _list_local_ipv4_addresses
from shard_engine.cli import cli_app
from shard_engine.linux_env import (
    ALL_CLI_SUBCOMMANDS,
    clear_daemon_pid,
    detect_linux_distro,
    detect_linux_target_triple,
    get_daemon_pid_file,
    parse_os_release,
    read_cgroup_memory_limit,
    read_daemon_pid,
    render_shell_completion,
    render_systemd_unit,
    terminate_daemon_process,
    write_daemon_pid,
)
from shard_engine.memory_manager import _parse_linux_meminfo, get_system_memory_info

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_os_release_ubuntu_debian_and_alpine(tmp_path: Path) -> None:
    """Distro parsing identifies Ubuntu, Debian, generic Linux, and WSL2 hosts."""
    ubuntu_release = (
        'PRETTY_NAME="Ubuntu 24.04.1 LTS"\n'
        'NAME="Ubuntu"\n'
        'VERSION_ID="24.04"\n'
        'ID=ubuntu\n'
        'ID_LIKE=debian\n'
    )
    os_file = tmp_path / "os-release"
    proc_file = tmp_path / "version"
    os_file.write_text(ubuntu_release, encoding="utf-8")
    proc_file.write_text("Linux version 6.8.0-45-generic (gcc)", encoding="utf-8")

    ubuntu_info = detect_linux_distro(os_release_path=os_file, proc_version_path=proc_file)
    assert ubuntu_info.id == "ubuntu"
    assert ubuntu_info.version_id == "24.04"
    assert ubuntu_info.is_ubuntu is True
    assert ubuntu_info.is_debian_family is True
    assert ubuntu_info.is_wsl is False

    # WSL2 detection via /proc/version
    proc_file.write_text("Linux version 5.15.153.1-microsoft-standard-WSL2", encoding="utf-8")
    wsl_info = detect_linux_distro(os_release_path=os_file, proc_version_path=proc_file)
    assert wsl_info.is_ubuntu is True
    assert wsl_info.is_wsl is True

    # Debian 12
    debian_info = parse_os_release('ID=debian\nNAME="Debian GNU/Linux"\nVERSION_ID="12"\n')
    assert debian_info.is_ubuntu is False
    assert debian_info.is_debian_family is True

    # Alpine Linux
    alpine_info = parse_os_release('ID=alpine\nNAME="Alpine Linux"\nVERSION_ID="3.20"\n')
    assert alpine_info.is_ubuntu is False
    assert alpine_info.is_debian_family is False


def test_detect_linux_target_triple_maps_architectures() -> None:
    """Architecture strings map cleanly to GNU Linux target triples."""
    assert detect_linux_target_triple("x86_64") == "x86_64-unknown-linux-gnu"
    assert detect_linux_target_triple("amd64") == "x86_64-unknown-linux-gnu"
    assert detect_linux_target_triple("aarch64") == "aarch64-unknown-linux-gnu"
    assert detect_linux_target_triple("arm64") == "aarch64-unknown-linux-gnu"
    assert detect_linux_target_triple("riscv64") == "riscv64-unknown-linux-gnu"


def test_linux_meminfo_fallback_and_cgroup_clamping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux RAM parsing supports MemAvailable fallback and clamps to cgroup v1/v2 limits."""
    # 1. MemAvailable absent -> MemFree + Buffers + Cached
    legacy_meminfo = (
        "MemTotal:        8388608 kB\n"
        "MemFree:         2097152 kB\n"
        "Buffers:          524288 kB\n"
        "Cached:          1572864 kB\n"
    )
    total_b, avail_b = _parse_linux_meminfo(legacy_meminfo)
    assert total_b == 8 * 1024**3
    assert avail_b == 4 * 1024**3

    # 2. cgroup v2 finite limit overrides unbounded host RAM
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text(str(6 * 1024**3), encoding="utf-8")
    (cg2 / "memory.current").write_text(str(1 * 1024**3), encoding="utf-8")
    cg1 = tmp_path / "cg1"
    cg1.mkdir()
    limit_b, curr_b = read_cgroup_memory_limit(cgroup_v2_root=cg2, cgroup_v1_root=cg1)
    assert limit_b == 6 * 1024**3
    assert curr_b == 1 * 1024**3

    # 3. cgroup v2 "max" falls through to cgroup v1, ignoring sentinels >= 1 << 60
    (cg2 / "memory.max").write_text("max\n", encoding="utf-8")
    (cg1 / "memory.limit_in_bytes").write_text(str(1 << 62), encoding="utf-8")
    assert read_cgroup_memory_limit(cgroup_v2_root=cg2, cgroup_v1_root=cg1) == (None, None)

    (cg1 / "memory.limit_in_bytes").write_text(str(4 * 1024**3), encoding="utf-8")
    (cg1 / "memory.usage_in_bytes").write_text(str(512 * 1024**2), encoding="utf-8")
    assert read_cgroup_memory_limit(cgroup_v2_root=cg2, cgroup_v1_root=cg1) == (
        4 * 1024**3,
        512 * 1024**2,
    )

    # 4. get_system_memory_info clamps host /proc/meminfo using cgroup limits on Linux
    proc_meminfo = (
        f"MemTotal:        {32 * 1024 * 1024} kB\n"
        f"MemAvailable:    {24 * 1024 * 1024} kB\n"
    )
    monkeypatch.setattr("shard_engine.memory_manager.platform.system", lambda: "Linux")

    real_exists = Path.exists
    real_read_text = Path.read_text

    def fake_exists(self: Path) -> bool:
        if str(self).replace("\\", "/") == "/proc/meminfo":
            return True
        return real_exists(self)

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self).replace("\\", "/") == "/proc/meminfo":
            return proc_meminfo
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(
        "shard_engine.linux_env.read_cgroup_memory_limit",
        lambda: (4 * 1024**3, 1 * 1024**3),
    )

    total_gb, avail_gb, max_share = get_system_memory_info()
    assert total_gb == 4.0
    assert avail_gb == 3.0
    assert max_share == 2


def test_resolve_default_cache_dir_for_env_frozen_and_readonly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelDownloader cache resolution handles SHARDAI_CACHE_DIR, frozen binaries, and read-only dirs."""
    home_dir = tmp_path / "shardai_home"

    # 1. Explicit SHARDAI_CACHE_DIR wins
    custom_cache = tmp_path / "custom_cache"
    monkeypatch.setenv("SHARDAI_CACHE_DIR", str(custom_cache))
    assert resolve_default_cache_dir(home_dir) == custom_cache.resolve()
    monkeypatch.delenv("SHARDAI_CACHE_DIR", raising=False)

    # 2. Frozen PyInstaller executable avoids ephemeral _MEI dirs
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert resolve_default_cache_dir(home_dir) == (home_dir / "cache_models").resolve()
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    # 3. Non-writable source tree falls back to XDG_CACHE_HOME on Linux, else <home>/cache_models
    monkeypatch.setattr("models.downloader.os.access", lambda _path, _mode: False)
    monkeypatch.setattr("models.downloader.platform.system", lambda: "Linux")
    xdg_cache = tmp_path / "xdg_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    assert resolve_default_cache_dir(home_dir) == (xdg_cache / "shardai" / "cache_models").resolve()

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert resolve_default_cache_dir(home_dir) == (home_dir / "cache_models").resolve()


def test_cli_help_includes_all_15_subcommands() -> None:
    """Root --help includes both original 12 and new Linux/headless subcommands."""
    res = runner.invoke(cli_app, ["--help"])
    assert res.exit_code == 0
    for cmd in ALL_CLI_SUBCOMMANDS:
        assert cmd in res.stdout, f"Missing subcommand '{cmd}' in root help."


def test_shardai_doctor_reports_linux_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shardai doctor outputs structured Linux diagnostics in text and --json modes."""
    monkeypatch.setenv("SHARDAI_HOME_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("SHARDAI_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("SHARDAI_LOG_DIR", str(tmp_path / "logs"))

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))
    monkeypatch.setattr(
        "shard_engine.linux_env.detect_linux_distro",
        lambda: parse_os_release('ID=ubuntu\nNAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04 LTS"\nVERSION_ID="24.04"\n', is_wsl=True),
    )

    res_json = runner.invoke(cli_app, ["doctor", "--json"])
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.stdout)
    assert data["ok"] is True
    assert data["distro"]["is_ubuntu"] is True
    assert data["distro"]["is_wsl"] is True
    assert data["daemon"]["reachable"] is True

    res_text = runner.invoke(cli_app, ["doctor"])
    assert res_text.exit_code == 0
    assert "Ubuntu 24.04 LTS" in res_text.stdout
    assert "[WSL]" in res_text.stdout
    assert "loopback_daemon" in res_text.stdout


def test_shardai_service_unit_and_security_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai service unit enforces loopback hosts, renders hardened units, and handles non-Linux OS."""
    user_unit = render_systemd_unit(exec_path="/usr/bin/shardai", host="127.0.0.1", port=8000, scope="user")
    assert "ExecStart=/usr/bin/shardai serve --host 127.0.0.1 --port 8000" in user_unit
    assert "Restart=on-failure" in user_unit
    assert "KillSignal=SIGTERM" in user_unit
    assert "WantedBy=default.target" in user_unit

    sys_unit = render_systemd_unit(
        exec_path="/usr/bin/shardai",
        host="localhost",
        port=8080,
        scope="system",
        tracker_url="https://network.shardai.online",
        offline=True,
    )
    assert "NoNewPrivileges=true" in sys_unit
    assert "ProtectSystem=strict" in sys_unit
    assert "SHARDAI_OFFLINE_MODE=true" in sys_unit
    assert "SHARDAI_TRACKER_URL=https://network.shardai.online" in sys_unit
    assert "WantedBy=multi-user.target" in sys_unit
    assert "\r" not in sys_unit

    with pytest.raises(ValueError):
        render_systemd_unit(exec_path="/usr/bin/shardai", host="0.0.0.0")

    res_unit = runner.invoke(cli_app, ["service", "unit", "--scope", "system", "--json"])
    assert res_unit.exit_code == 0, res_unit.output
    parsed = json.loads(res_unit.stdout)
    assert parsed["scope"] == "system"
    assert "NoNewPrivileges=true" in parsed["content"]

    res_bad_host = runner.invoke(cli_app, ["service", "unit", "--host", "0.0.0.0"])
    assert res_bad_host.exit_code != 0

    monkeypatch.setattr("shard_engine.cli.platform.system", lambda: "Windows")
    res_non_linux = runner.invoke(cli_app, ["service", "status"])
    assert res_non_linux.exit_code == 1
    assert "only available on Linux" in res_non_linux.output


def test_shardai_completion_bash_and_zsh() -> None:
    """shardai completion emits valid Bash and Zsh completion scripts for all 15 commands."""
    for shell_name in ("bash", "zsh"):
        script = render_shell_completion(shell_name)
        assert "\r" not in script
        for cmd in ALL_CLI_SUBCOMMANDS:
            assert cmd in script

        res = runner.invoke(cli_app, ["completion", shell_name])
        assert res.exit_code == 0
        assert "shardai" in res.stdout

    res_bad = runner.invoke(cli_app, ["completion", "fish"])
    assert res_bad.exit_code == 1


def test_serve_background_stop_shutdown_daemon_and_local_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serve --background, PID tracking, X-ShardAi-Local-Token, and stop --shutdown-daemon work end-to-end."""
    monkeypatch.setenv("SHARDAI_HOME_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("SHARDAI_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("SHARDAI_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SHARDAI_LOCAL_TOKEN", "ipc-secret-local-token")

    health_checks = 0
    captured_headers: list[str | None] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal health_checks
        captured_headers.append(request.headers.get("X-ShardAi-Local-Token"))
        if request.url.path == "/health":
            health_checks += 1
            # Fail once while daemon is not running yet, then become healthy after Popen
            if health_checks <= 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/stop-node" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "state": "STOPPED",
                    "allocated_ram_mb": 0,
                    "allocated_vram_mb": 0,
                    "message": "Node stopped.",
                },
            )
        return httpx.Response(404)

    class _FakeProc:
        pid = 42424

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            pass

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))
    monkeypatch.setattr("shard_engine.cli.subprocess.Popen", lambda *a, **kw: _FakeProc())

    res_serve = runner.invoke(cli_app, ["serve", "--background", "--port", "8123", "--json"])
    assert res_serve.exit_code == 0, res_serve.output
    serve_data = json.loads(res_serve.stdout)
    assert serve_data["status"] == "running"
    assert serve_data["pid"] == 42424
    assert read_daemon_pid(8123) == 42424
    assert get_daemon_pid_file(8123).is_file()

    # Stop with --shutdown-daemon sends X-ShardAi-Local-Token and terminates PID cleanly
    alive_states = iter([True, False])
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr("shard_engine.linux_env.is_process_alive", lambda _pid: next(alive_states, False))
    monkeypatch.setattr("shard_engine.linux_env.os.kill", lambda pid, sig: kill_calls.append((pid, int(sig))))

    res_stop = runner.invoke(cli_app, ["stop", "--port", "8123", "--shutdown-daemon", "--json"])
    assert res_stop.exit_code == 0, res_stop.output
    stop_data = json.loads(res_stop.stdout)
    assert stop_data["daemon_stopped"] is True
    assert kill_calls and kill_calls[0][0] == 42424
    assert read_daemon_pid(8123) is None
    assert "ipc-secret-local-token" in captured_headers

    # Sanity check direct helper cleanup on stale PID
    write_daemon_pid(8124, 99999)
    monkeypatch.setattr("shard_engine.linux_env.is_process_alive", lambda _pid: False)
    ok, _msg = terminate_daemon_process(8124)
    assert ok is False
    clear_daemon_pid(8124)
    assert read_daemon_pid(8124) is None


def test_lan_ipv4_discovery_fallback_for_ubuntu_etc_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """_list_local_ipv4_addresses skips 127.0.1.1 and probes fallback RFC1918 subnets."""
    monkeypatch.setattr(
        "p2p.mdns_discovery.socket.getaddrinfo",
        lambda *_a, **_kw: [(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("127.0.1.1", 0))],
    )

    class _FakeProbeSocket:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self._ip = "127.0.0.1"

        def connect(self, addr: tuple[str, int]) -> None:
            if addr[0] == "10.255.255.255":
                raise OSError("Network unreachable")
            if addr[0] == "192.168.255.255":
                self._ip = "192.168.1.42"
            else:
                raise OSError("Network unreachable")

        def getsockname(self) -> tuple[str, int]:
            return (self._ip, 0)

        def close(self) -> None:
            pass

    monkeypatch.setattr("p2p.mdns_discovery.socket.socket", _FakeProbeSocket)
    ips = _list_local_ipv4_addresses()
    assert ips == ["192.168.1.42"]


def test_linux_packaging_lf_line_endings_security_and_ci() -> None:
    """Verify Linux scripts, units, completions, Dockerfile, CI, and English/public compliance."""
    linux_lf_files = (
        REPO_ROOT / ".gitattributes",
        REPO_ROOT / "scripts" / "install_linux.sh",
        REPO_ROOT / "scripts" / "install_cli.sh",
        REPO_ROOT / "scripts" / "build_core.sh",
        REPO_ROOT / "scripts" / "build_deb.sh",
        REPO_ROOT / "packaging" / "linux" / "shardai.service",
        REPO_ROOT / "packaging" / "linux" / "shardai-user.service",
        REPO_ROOT / "packaging" / "linux" / "completions" / "shardai.bash",
        REPO_ROOT / "packaging" / "linux" / "completions" / "_shardai",
        REPO_ROOT / "packaging" / "linux" / "Dockerfile",
    )
    for file_path in linux_lf_files:
        assert file_path.is_file(), f"Missing Linux artifact: {file_path}"
        raw_bytes = file_path.read_bytes()
        assert b"\r" not in raw_bytes, f"CRLF detected in Linux file {file_path}"

    assert (REPO_ROOT / "docs" / "LINUX.md").is_file()

    # Loopback-only security inside systemd units and Dockerfile
    for sec_path in (
        REPO_ROOT / "packaging" / "linux" / "shardai.service",
        REPO_ROOT / "packaging" / "linux" / "shardai-user.service",
        REPO_ROOT / "packaging" / "linux" / "Dockerfile",
    ):
        text = sec_path.read_text(encoding="utf-8")
        assert "0.0.0.0" not in text, f"Forbidden wildcard 0.0.0.0 in {sec_path}"
        assert "127.0.0.1" in text

    # GitHub Release CI includes ubuntu-latest Linux CLI & .deb build
    release_yml = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "ubuntu-latest" in release_yml
    assert "build_core.sh --cli-only" in release_yml
    assert "build_deb.sh" in release_yml

    # English-only & no private cloud terms across new/modified docs and scripts
    forbidden_cloud_terms = (
        "cloudflare",
        "hono",
        "wrangler",
        "shardai-cloud",
        "admin_secret_key",
    )
    for doc_or_code in (
        *linux_lf_files,
        REPO_ROOT / "docs" / "LINUX.md",
        REPO_ROOT / "docs" / "BUILD.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "core-engine" / "shard_engine" / "linux_env.py",
    ):
        content = doc_or_code.read_text(encoding="utf-8")
        lowered = content.lower()
        for term in forbidden_cloud_terms:
            assert term not in lowered, f"Forbidden term '{term}' in {doc_or_code}"
        # Zero Vietnamese diacritics (all non-ASCII letters checked carefully)
        non_ascii = [ch for ch in content if ord(ch) > 127 and ch.isalpha()]
        assert not non_ascii, f"Non-ASCII letters found in {doc_or_code}: {non_ascii[:5]}"
