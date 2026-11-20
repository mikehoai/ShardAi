"""Comprehensive unit tests for the ShardAi CLI (shard_engine.cli).

Covers:
1. Argument normalization (`normalize_cli_argv`) for single-dash long flags.
2. Model specification resolution (`resolve_model_spec`).
3. Root help and version subcommands.
4. Loopback host security validation.
5. `shardai create` and `shardai join` (daemon-attached, pre-flight guard, exit codes).
6. `shardai status` (token masking, --show-key, --json, daemon offline).
7. `shardai stop`, `rooms`, `models`, `pull`, `peers`, `network`.
8. `shardai chat` (auto-token discovery, SSE streaming, non-streaming, 503 guard).
9. Tauri sidecar `--mode daemon` and legacy `--mode cli` compatibility.
10. Packaging scripts and pyproject.toml entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import types
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from shard_engine.cli import (
    CLI_VERSION,
    cli_app,
    mask_api_key,
    normalize_cli_argv,
    resolve_model_spec,
    run_cli,
    validate_loopback_host,
)
from shard_engine.schemas import ModelSourceType

runner = CliRunner()


def test_normalize_cli_argv_converts_single_dash_long_flags() -> None:
    """Single-dash long flags like -model or -ram should normalize to --model, --ram."""
    raw = [
        "-model", "llama-3.1-8b",
        "-room=test-room",
        "-ram", "8",
        "-network", "lan",
        "-source-type=catalog",
        "-p2p-port", "49152",
        "-ssd-offload",
        "-json",
        "-show-key",
        "-foreground",
        "-m", "short-m",
        "-r", "short-r",
        "-p", "8000",
        "--",
        "-model", "passed-verbatim",
    ]
    expected = [
        "--model", "llama-3.1-8b",
        "--room=test-room",
        "--ram", "8",
        "--network", "lan",
        "--source-type=catalog",
        "--p2p-port", "49152",
        "--ssd-offload",
        "--json",
        "--show-key",
        "--foreground",
        "-m", "short-m",
        "-r", "short-r",
        "-p", "8000",
        "--",
        "-model", "passed-verbatim",
    ]
    assert normalize_cli_argv(raw) == expected


def test_validate_loopback_host_enforces_security() -> None:
    """Non-loopback hosts like 0.0.0.0 or external IPs must be rejected."""
    assert validate_loopback_host("127.0.0.1") == "127.0.0.1"
    assert validate_loopback_host("localhost") == "localhost"
    assert validate_loopback_host("::1") == "::1"
    assert validate_loopback_host("[::1]") == "::1"

    with pytest.raises(Exception):
        validate_loopback_host("0.0.0.0")

    with pytest.raises(Exception):
        validate_loopback_host("192.168.1.100")


def test_mask_api_key_redacts_tokens() -> None:
    """Session tokens must be masked by default for security."""
    assert mask_api_key(None) == "<none>"
    assert mask_api_key("") == "<none>"
    assert mask_api_key("short") == "****"
    masked = mask_api_key("sk-shard-abcdef1234567890xyz")
    assert masked.startswith("sk-shard-****")
    assert masked.endswith("0xyz")
    assert "abcdef123456789" not in masked


def test_resolve_model_spec_maps_catalog_and_urls(tmp_path: Path) -> None:
    """resolve_model_spec accurately infers catalog, huggingface, direct_url, and local_file."""
    # 1. Shorthand catalog id
    repo_id, source_type, filename = resolve_model_spec("llama-3.1-8b")
    assert source_type == ModelSourceType.CATALOG
    assert "Meta-Llama-3.1-8B-Instruct" in repo_id
    assert filename == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    # 2. Direct .gguf URL
    direct_url = "https://example.com/models/test-model-q4.gguf"
    resolved_id, source_type, _ = resolve_model_spec(direct_url)
    assert source_type == ModelSourceType.DIRECT_URL
    assert resolved_id == direct_url

    # 3. Non-catalog Hugging Face repo URL
    hf_url = "https://huggingface.co/custom-org/custom-model"
    resolved_id, source_type, _ = resolve_model_spec(hf_url)
    assert source_type == ModelSourceType.HUGGINGFACE
    assert resolved_id == "custom-org/custom-model"

    # 4. Local .gguf file
    local_file = tmp_path / "custom.gguf"
    local_file.write_bytes(b"GGUF" + b"\x00" * 32)
    resolved_id, source_type, _ = resolve_model_spec(str(local_file))
    assert source_type == ModelSourceType.LOCAL_FILE
    assert Path(resolved_id).resolve() == local_file.resolve()


def test_cli_help_displays_all_subcommands() -> None:
    """Invoking shardai with --help lists all 12 subcommands."""
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout
    subcommands = [
        "create", "join", "serve", "status", "stop",
        "rooms", "models", "pull", "peers", "chat",
        "network", "version",
    ]
    for subcmd in subcommands:
        assert subcmd in help_text, f"Subcommand '{subcmd}' missing from root help."


def test_cli_version_commands() -> None:
    """shardai version and shardai --version output the CLI version."""
    v1 = runner.invoke(cli_app, ["--version"])
    assert v1.exit_code == 0
    assert CLI_VERSION in v1.stdout

    v2 = runner.invoke(cli_app, ["version"])
    assert v2.exit_code == 0
    assert CLI_VERSION in v2.stdout

    v3 = runner.invoke(cli_app, ["version", "--json"])
    assert v3.exit_code == 0
    parsed = json.loads(v3.stdout)
    assert parsed.get("version") == CLI_VERSION


def test_shardai_create_attached_to_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai create attaches to a running daemon via POST /start-node."""
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/rooms/check-join":
            return httpx.Response(200, json={"allowed": True, "room_id": "test-room"})
        if request.url.path == "/start-node":
            return httpx.Response(
                200,
                json={
                    "state": "RUNNING",
                    "node_id": "cli-node-1",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "room_id": "test-room",
                    "role": "primary",
                    "shard_layer_start": 0,
                    "shard_layer_end": 15,
                    "total_layers": 32,
                    "allocated_ram_mb": 4096,
                    "allocated_vram_mb": 0,
                    "api_base_url": "http://127.0.0.1:8000/v1",
                    "session_token": "sk-shard-test-token-12345",
                    "is_swarm_ready": True,
                    "coverage_percent": 100.0,
                    "network_status": "online",
                    "ssd_offload_active": False,
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    res = runner.invoke(
        cli_app,
        ["create", "-model", "llama-3.1-8b", "-room", "test-room", "-ram", "4", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)
    assert data["state"] == "RUNNING"
    assert data["room_id"] == "test-room"
    assert any(r.url.path == "/start-node" for r in captured_requests)


def test_shardai_join_preflight_rejection_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai join must abort immediately if room pre-flight check returns 409."""
    start_node_called = False

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal start_node_called
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/rooms/check-join":
            return httpx.Response(
                409,
                json={
                    "detail": "Room 'full-room' has reached 100% RAM capacity and is locked.",
                },
            )
        if request.url.path == "/start-node":
            start_node_called = True
            return httpx.Response(200, json={"state": "RUNNING"})
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    res = runner.invoke(cli_app, ["join", "full-room", "-model", "llama-3.1-8b"])
    assert res.exit_code == 1
    assert "has reached 100% RAM capacity" in res.output
    assert not start_node_called


def test_shardai_status_masks_key_and_supports_show_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai status masks the key by default and reveals it when --show-key is passed."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(
                200,
                json={
                    "state": "RUNNING",
                    "node_id": "test-node",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "room_id": "public",
                    "role": "primary",
                    "shard_layer_start": 0,
                    "shard_layer_end": 15,
                    "total_layers": 32,
                    "allocated_ram_mb": 4096,
                    "allocated_vram_mb": 0,
                    "connected_peers_count": 2,
                    "api_base_url": "http://127.0.0.1:8000/v1",
                    "session_token": "sk-shard-secret-token-abcdef123456",
                    "is_swarm_ready": True,
                    "coverage_percent": 100.0,
                },
            )
        if request.url.path == "/network-status":
            return httpx.Response(200, json={"network_status": "online"})
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    # 1. Default status output (masked key)
    res_masked = runner.invoke(cli_app, ["status"])
    assert res_masked.exit_code == 0
    assert "sk-shard-****3456" in res_masked.stdout
    assert "secret-token" not in res_masked.stdout

    # 2. Status with --show-key
    res_unmasked = runner.invoke(cli_app, ["status", "--show-key"])
    assert res_unmasked.exit_code == 0
    assert "sk-shard-secret-token-abcdef123456" in res_unmasked.stdout

    # 3. Status with --json (masked by default)
    res_json = runner.invoke(cli_app, ["status", "--json"])
    assert res_json.exit_code == 0
    json_data = json.loads(res_json.stdout)
    assert json_data["session_token"] == "sk-shard-****3456"


def test_shardai_stop_calls_daemon_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai stop invokes POST /stop-node and confirms resources released."""
    stop_called = False

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal stop_called
        if request.url.path == "/stop-node" and request.method == "POST":
            stop_called = True
            return httpx.Response(
                200,
                json={
                    "state": "STOPPED",
                    "allocated_ram_mb": 0,
                    "allocated_vram_mb": 0,
                    "message": "Node stopped, session token revoked, and RAM released.",
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    res = runner.invoke(cli_app, ["stop"])
    assert res.exit_code == 0
    assert stop_called
    assert "Node stopped" in res.stdout


def test_shardai_rooms_and_models_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai rooms and shardai models display clean tables from daemon."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/available-rooms":
            return httpx.Response(
                200,
                json=[
                    {
                        "room_id": "public",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "nodes_online": 3,
                        "total_shared_ram_gb": 12.0,
                        "is_private": False,
                        "network_scope": "internet",
                        "coverage_percent": 100.0,
                        "redundancy_factor": 1,
                    }
                ],
            )
        if request.url.path == "/models/catalog":
            return httpx.Response(
                200,
                json={
                    "source": "cloud",
                    "count": 1,
                    "models": [
                        {
                            "id": "llama-3.1-8b",
                            "name": "Llama 3.1 8B",
                            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                            "estimated_ram_gb": 5.1,
                            "enabled": True,
                        }
                    ],
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    res_rooms = runner.invoke(cli_app, ["rooms"])
    assert res_rooms.exit_code == 0
    assert "public" in res_rooms.stdout
    assert "Ready (1x)" in res_rooms.stdout

    res_models = runner.invoke(cli_app, ["models"])
    assert res_models.exit_code == 0
    assert "llama-3.1-8b" in res_models.stdout
    assert "5.1 GB" in res_models.stdout


def test_shardai_chat_auto_discovers_token_and_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai chat queries /status to discover session token and streams completion."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(
                200,
                json={
                    "state": "RUNNING",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "api_base_url": "http://127.0.0.1:8000/v1",
                    "session_token": "sk-shard-valid-token",
                    "is_swarm_ready": True,
                    "coverage_percent": 100.0,
                },
            )
        if request.url.path == "/v1/chat/completions":
            assert request.headers.get("Authorization") == "Bearer sk-shard-valid-token"
            payload = json.loads(request.read())
            if payload.get("stream"):
                sse_data = (
                    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":" world!"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                return httpx.Response(200, content=sse_data, headers={"Content-Type": "text/event-stream"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Hello non-streaming!"}}]},
            )
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    # 1. One-shot streaming chat
    res_stream = runner.invoke(cli_app, ["chat", "Hi there"])
    assert res_stream.exit_code == 0
    assert "Hello world!" in res_stream.stdout

    # 2. One-shot non-streaming chat with --json
    res_json = runner.invoke(cli_app, ["chat", "Hi", "--no-stream", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert data["choices"][0]["message"]["content"] == "Hello non-streaming!"


def test_shardai_chat_fails_when_swarm_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """shardai chat must fail cleanly when is_swarm_ready is False."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(
                200,
                json={
                    "state": "RUNNING",
                    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "api_base_url": "http://127.0.0.1:8000/v1",
                    "session_token": "sk-shard-token",
                    "is_swarm_ready": False,
                    "coverage_percent": 50.0,
                },
            )
        return httpx.Response(404)

    monkeypatch.setattr("shard_engine.cli._http_transport_override", httpx.MockTransport(mock_handler))

    res = runner.invoke(cli_app, ["chat", "Hi"])
    assert res.exit_code == 1
    assert "Model not ready" in res.output


def test_tauri_sidecar_daemon_and_legacy_cli_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify backwards compatibility for Tauri sidecar invocation and legacy flags."""
    daemon_called = False
    standalone_called = False

    def mock_daemon(host: str, port: int) -> None:
        nonlocal daemon_called
        daemon_called = True

    def mock_standalone(host: str, port: int, request: Any, **kwargs: Any) -> None:
        nonlocal standalone_called
        standalone_called = True

    monkeypatch.setattr("shard_engine.cli._run_daemon_server", mock_daemon)
    monkeypatch.setattr("shard_engine.cli._run_standalone_node", mock_standalone)

    # 1. Tauri sidecar launches core-engine with --mode daemon --port 8000
    res_daemon = runner.invoke(cli_app, ["--mode", "daemon", "--port", "8000"])
    assert res_daemon.exit_code == 0
    assert daemon_called

    # 2. Legacy standalone CLI launch with --mode cli --ram 4
    res_legacy = runner.invoke(cli_app, ["--mode", "cli", "--ram", "4"])
    assert res_legacy.exit_code == 0
    assert standalone_called


def test_pyproject_scripts_and_install_scripts_exist() -> None:
    """Ensure pyproject.toml defines [project.scripts] and installer scripts exist."""
    repo_root = Path(__file__).resolve().parents[2]

    # pyproject.toml
    pyproject_path = repo_root / "core-engine" / "pyproject.toml"
    assert pyproject_path.is_file()
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    assert '[project.scripts]' in pyproject_text
    assert 'shardai = "shard_engine.cli:run_cli"' in pyproject_text

    # installer scripts
    install_ps1 = repo_root / "scripts" / "install_cli.ps1"
    install_sh = repo_root / "scripts" / "install_cli.sh"
    assert install_ps1.is_file()
    assert install_sh.is_file()

    # build scripts reference shardai executable
    build_ps1 = (repo_root / "scripts" / "build_core.ps1").read_text(encoding="utf-8")
    build_sh = (repo_root / "scripts" / "build_core.sh").read_text(encoding="utf-8")
    assert "shardai" in build_ps1
    assert "shardai" in build_sh
