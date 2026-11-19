"""Unit tests for Issue #1: Prevent room join and model download when a room is full or busy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

from main import app, runtime
from shard_engine.schemas import NetworkStatus, NodeState
from shard_engine.tracker_client import (
    SignalingTrackerClient,
    extract_tracker_error_detail,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_runtime_and_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset runtime state and restore original tracker client after each test."""
    original_tracker = runtime.tracker_client
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    yield
    runtime.tracker_client = original_tracker
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


def _make_mock_tracker(
    handler: Any,
    max_retries: int = 1,
) -> SignalingTrackerClient:
    return SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=max_retries,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )


def _enable_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")


def test_check_room_join_rejects_full_room_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_tracker(monkeypatch)
    rejection_msg = 'Room "deepseek-r1-vn-01" is at 100% RAM/Layer capacity and locked.'

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "/peers/deepseek-r1-vn-01/" in request.url.path
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_FULL", "message": rejection_msg},
                "message": rejection_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "deepseek-r1-vn-01",
            "model_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == rejection_msg


def test_check_room_join_rejects_busy_queue_409(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_tracker(monkeypatch)
    rejection_msg = 'Room "qwen-coder-7b-dev" already has another client joining and syncing layers.'

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_QUEUE_BUSY", "message": rejection_msg},
                "message": rejection_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "qwen-coder-7b-dev",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == rejection_msg


def test_check_room_join_rejects_invalid_private_key_403(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_tracker(monkeypatch)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=403,
            json={
                "success": False,
                "error": {"code": "FORBIDDEN", "message": "Invalid Room Key"},
                "message": "Invalid Room Key",
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "private-vault",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "network_mode": "internet",
            "room_key": "wrong-key",
        },
    )
    assert res.status_code == 403
    assert res.json()["detail"] == "Invalid Room Key"


def test_check_room_join_allows_offline_fallback_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Tracker timed out")

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "public",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["allowed"] is True
    assert body["room_id"] == "public"


def test_download_stream_blocked_on_409_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)
    rejection_msg = 'Room "deepseek-r1-vn-01" is full (100% RAM).'

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_FULL", "message": rejection_msg},
                "message": rejection_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)
    download_spy = AsyncMock()
    monkeypatch.setattr(runtime.model_downloader, "ensure_model_downloaded", download_spy)

    res = client.get(
        "/models/download-stream",
        params={
            "model_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "room_id": "deepseek-r1-vn-01",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == rejection_msg
    download_spy.assert_not_awaited()


def test_start_node_rejects_409_full_room_without_download_or_offline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)
    rejection_msg = 'Room "deepseek-r1-vn-01" is at 100% RAM/Layer capacity.'

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_FULL", "message": rejection_msg},
                "message": rejection_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)
    download_spy = AsyncMock()
    monkeypatch.setattr(runtime.model_downloader, "ensure_model_downloaded", download_spy)

    res = client.post(
        "/start-node",
        json={
            "node_id": "test-node-full",
            "model_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "network_mode": "internet",
            "room_id": "deepseek-r1-vn-01",
            "ram_gb": 4,
            "shared_ram_gb": 4,
            "max_ram_mb": 4096,
            "max_vram_mb": 0,
            "p2p_listen_port": 49310,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == rejection_msg
    download_spy.assert_not_awaited()
    assert runtime.state == NodeState.STOPPED
    assert runtime.session_token is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0


def test_start_node_rejects_409_busy_queue_without_download_or_offline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)
    rejection_msg = 'Room "qwen-coder-7b-dev" already has a client joining.'

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_QUEUE_BUSY", "message": rejection_msg},
                "message": rejection_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)
    download_spy = AsyncMock()
    monkeypatch.setattr(runtime.model_downloader, "ensure_model_downloaded", download_spy)

    res = client.post(
        "/start-node",
        json={
            "node_id": "test-node-busy",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "network_mode": "internet",
            "room_id": "qwen-coder-7b-dev",
            "ram_gb": 4,
            "shared_ram_gb": 4,
            "max_ram_mb": 4096,
            "max_vram_mb": 0,
            "p2p_listen_port": 49311,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == rejection_msg
    download_spy.assert_not_awaited()
    assert runtime.state == NodeState.STOPPED
    assert runtime.session_token is None
    assert runtime.memory_manager.allocated_ram_mb == 0


def test_start_node_preserves_offline_solo_fallback_on_tracker_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _enable_tracker(monkeypatch)
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("No internet")

    runtime.tracker_client = _make_mock_tracker(_handler)
    fake_model = tmp_path / "dummy.gguf"
    fake_model.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fake_model),
    )

    res = client.post(
        "/start-node",
        json={
            "node_id": "test-node-offline-fallback",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "network_mode": "internet",
            "room_id": "public",
            "ram_gb": 4,
            "shared_ram_gb": 4,
            "max_ram_mb": 4096,
            "max_vram_mb": 0,
            "p2p_listen_port": 49312,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["state"] == "RUNNING"
    assert body["network_status"] == NetworkStatus.OFFLINE_SOLO.value

    stop_res = client.post("/stop-node")
    assert stop_res.status_code == 200
    assert stop_res.json()["allocated_ram_mb"] == 0


def test_check_room_join_normalizes_vietnamese_full_room_notice_to_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)
    vn_msg = (
        'Room "deepseek-r1-vn-01" \u0111\u00E3 \u0111\u1EA1t 100% RAM/Layer '
        'v\u00E0 \u0111ang kh\u00F3a phi\u00EAn ch\u1EA1y model.'
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_FULL", "message": vn_msg},
                "message": vn_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "deepseek-r1-vn-01",
            "model_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == (
        'Room "deepseek-r1-vn-01" has reached 100% RAM/Layer capacity and is locked '
        "for active inference. Please select another room or create a new room."
    )


def test_check_room_join_normalizes_vietnamese_busy_queue_notice_to_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_tracker(monkeypatch)
    vn_msg = (
        'Room "qwen-coder-7b-dev" (65% RAM) hi\u1EC7n \u0111\u00E3 c\u00F3 '
        'l\u01B0\u1EE3t ch\u1EDD k\u1EBFt n\u1ED1i \u0111ang \u0111\u1ED3ng b\u1ED9 layer.'
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=409,
            json={
                "success": False,
                "error": {"code": "ROOM_QUEUE_BUSY", "message": vn_msg},
                "message": vn_msg,
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "qwen-coder-7b-dev",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == (
        'Room "qwen-coder-7b-dev" (65% RAM) currently has another client joining '
        "and syncing layers. Please wait or create a new room."
    )


def test_extract_tracker_error_detail_handles_vietnamese_forbidden_and_fallback() -> None:
    forbidden_res = httpx.Response(
        status_code=403,
        json={
            "success": False,
            "error": {
                "code": "FORBIDDEN",
                "message": "M\u00E3 ph\u00F2ng ri\u00EAng kh\u00F4ng h\u1EE3p l\u1EC7.",
            },
        },
    )
    assert (
        extract_tracker_error_detail(forbidden_res, "Fallback forbidden")
        == "Invalid room key for this private room."
    )

    unknown_vn_res = httpx.Response(
        status_code=409,
        json={
            "success": False,
            "error": {
                "code": "CUSTOM_ERROR",
                "message": "Kh\u00F4ng th\u1EC3 tham gia ph\u00F2ng l\u00FAc n\u00E0y.",
            },
        },
    )
    assert (
        extract_tracker_error_detail(unknown_vn_res, "Unable to join room.")
        == "Unable to join room."
    )
