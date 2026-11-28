"""Regression tests for Issue #68: model-scoped room coverage and shared RAM fallback."""

from __future__ import annotations

from collections.abc import Iterator
import time
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

import main as main_module
from main import _beacon_to_peer, app, runtime
from shard_engine.schemas import (
    NetworkMode,
    NetworkStatus,
    NodeState,
    PeerNodeInfo,
    PeerRole,
    RoomLobbyItem,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient, _normalize_peer

client = TestClient(app)


def _reset_runtime_fields() -> None:
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.model_id = None
    runtime.room_id = None
    runtime.room_key_hash = None
    runtime.network_mode = None
    runtime.peers = []
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.coverage_percent = 0.0
    runtime.is_swarm_ready = False
    runtime.room_total_shared_ram_gb = 0.0
    runtime._last_heartbeat_at = 0.0
    runtime._active_start_request = None
    runtime._active_max_layers_can_hold = 0
    runtime.p2p_listen_port = None
    runtime.n_layers = 0


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Iterator[None]:
    """Reset runtime state and restore the original tracker client around each test."""
    original_tracker = runtime.tracker_client
    _reset_runtime_fields()
    yield
    runtime.tracker_client = original_tracker
    _reset_runtime_fields()


def _configure_running_node(
    *,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str | None,
    room_id: str = "public",
    shared_ram_gb: int = 8,
    shard_layer_start: int = 0,
    shard_layer_end: int = 15,
    total_layers: int = 32,
) -> None:
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setattr(
        main_module,
        "_scan_all_lan_beacons",
        AsyncMock(return_value=[]),
    )

    runtime.state = NodeState.RUNNING
    runtime.network_mode = NetworkMode.INTERNET
    runtime.network_status = NetworkStatus.ONLINE
    runtime.node_id = "local-node-68"
    runtime.model_id = model_id
    runtime.room_id = room_id
    runtime.session_token = "sk-shard-test-68"
    runtime.shard_layer_start = shard_layer_start
    runtime.shard_layer_end = shard_layer_end
    runtime.total_layers = total_layers
    runtime._last_heartbeat_at = time.monotonic()
    runtime._active_start_request = StartNodeRequest(
        node_id="local-node-68",
        model_id=model_id or "meta-llama/Llama-3.1-8B-Instruct",
        network_mode=NetworkMode.INTERNET,
        room_id=room_id,
        ram_gb=shared_ram_gb,
        shared_ram_gb=shared_ram_gb,
        max_ram_mb=shared_ram_gb * 1024,
        shard_layer_start=shard_layer_start,
        shard_layer_end=shard_layer_end,
        total_layers=total_layers,
    )


@pytest.mark.asyncio
async def test_refresh_room_coverage_ignores_earlier_different_model_in_public_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model A (40%) listed first in 'public' must not corrupt Model B's 100% readiness or RAM."""
    model_a = "Qwen/Qwen2.5-7B-Instruct"
    model_b = "meta-llama/Llama-3.1-8B-Instruct"
    _configure_running_node(monkeypatch=monkeypatch, model_id=model_b, shared_ram_gb=8)

    rooms_payload = [
        RoomLobbyItem(
            room_id="public",
            model=model_a,
            nodes_online=1,
            total_shared_ram_gb=6.0,
            coverage_percent=40.0,
        ),
        RoomLobbyItem(
            room_id="public",
            model=model_b,
            nodes_online=2,
            total_shared_ram_gb=16.0,
            coverage_percent=100.0,
        ),
    ]
    peers_payload = [
        PeerNodeInfo(
            peer_id="peer-b-2",
            ip_address="203.0.113.12",
            port=49153,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
            available_ram_gb=8.0,
            room_id="public",
        )
    ]

    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_available_rooms",
        AsyncMock(return_value=rooms_payload),
    )
    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_peers",
        AsyncMock(return_value=peers_payload),
    )

    await runtime.refresh_room_coverage()

    assert runtime.coverage_percent == 100.0
    assert runtime.is_swarm_ready is True
    assert runtime.room_total_shared_ram_gb == 16.0


@pytest.mark.asyncio
async def test_refresh_room_coverage_does_not_mark_incomplete_model_ready_from_other_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model A (100%) listed first in 'public' must not falsely mark Model B (40%) ready."""
    model_a = "meta-llama/Llama-3.1-8B-Instruct"
    model_b = "Qwen/Qwen2.5-72B-Instruct"
    _configure_running_node(
        monkeypatch=monkeypatch,
        model_id=model_b,
        shared_ram_gb=4,
        shard_layer_start=0,
        shard_layer_end=11,
        total_layers=80,
    )

    rooms_payload = [
        RoomLobbyItem(
            room_id="public",
            model=model_a,
            nodes_online=3,
            total_shared_ram_gb=24.0,
            coverage_percent=100.0,
        ),
        RoomLobbyItem(
            room_id="public",
            model=model_b,
            nodes_online=2,
            total_shared_ram_gb=10.0,
            coverage_percent=40.0,
        ),
    ]
    peers_payload = [
        PeerNodeInfo(
            peer_id="peer-qwen-2",
            ip_address="203.0.113.15",
            port=49154,
            shard_layer_start=12,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
            available_ram_gb=6.0,
            room_id="public",
        )
    ]

    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_available_rooms",
        AsyncMock(return_value=rooms_payload),
    )
    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_peers",
        AsyncMock(return_value=peers_payload),
    )

    await runtime.refresh_room_coverage()

    assert runtime.coverage_percent == 40.0
    assert runtime.is_swarm_ready is False
    assert runtime.room_total_shared_ram_gb == 10.0

    # /v1/chat/completions must reject requests while this model's swarm is <100% ready
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-shard-test-68"},
        json={
            "model": model_b,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_refresh_room_coverage_falls_back_locally_when_only_other_model_in_public_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If /rooms only lists a different model for 'public', use local peer coverage and shared RAM."""
    _configure_running_node(
        monkeypatch=monkeypatch,
        model_id="Qwen/Qwen2.5-7B-Instruct",
        shared_ram_gb=4,
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
    )

    rooms_payload = [
        RoomLobbyItem(
            room_id="public",
            model="meta-llama/Llama-3.1-8B-Instruct",
            nodes_online=2,
            total_shared_ram_gb=32.0,
            coverage_percent=100.0,
        )
    ]
    # CPU-only peer with 0 VRAM and 6.5 GB shared RAM covering layers 16..23 (total 24/32 = 75%)
    peers_payload = [
        PeerNodeInfo(
            peer_id="cpu-peer-01",
            ip_address="203.0.113.20",
            port=49155,
            shard_layer_start=16,
            shard_layer_end=23,
            role=PeerRole.PRIMARY,
            available_vram_mb=0,
            available_ram_gb=6.5,
            room_id="public",
        )
    ]

    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_available_rooms",
        AsyncMock(return_value=rooms_payload),
    )
    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_peers",
        AsyncMock(return_value=peers_payload),
    )

    await runtime.refresh_room_coverage()

    assert runtime.coverage_percent == 75.0
    assert runtime.is_swarm_ready is False
    assert runtime.room_total_shared_ram_gb == 10.5


@pytest.mark.asyncio
async def test_refresh_room_coverage_matches_room_id_when_model_id_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When runtime.model_id is None, matching falls back to room_id only."""
    _configure_running_node(monkeypatch=monkeypatch, model_id=None, shared_ram_gb=4)

    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_available_rooms",
        AsyncMock(
            return_value=[
                RoomLobbyItem(
                    room_id="public",
                    model="any/model",
                    nodes_online=2,
                    total_shared_ram_gb=14.0,
                    coverage_percent=100.0,
                )
            ]
        ),
    )

    await runtime.refresh_room_coverage()

    assert runtime.coverage_percent == 100.0
    assert runtime.is_swarm_ready is True
    assert runtime.room_total_shared_ram_gb == 14.0


@pytest.mark.asyncio
async def test_refresh_room_coverage_local_fallback_sums_cpu_only_peer_shared_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local fallback sums CPU-only peers' available_ram_gb even when available_vram_mb == 0."""
    _configure_running_node(
        monkeypatch=monkeypatch,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        room_id="lan-cpu-room",
        shared_ram_gb=4,
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
    )

    async def _failing_rooms() -> list[RoomLobbyItem]:
        raise httpx.ConnectError("tracker unreachable")

    monkeypatch.setattr(runtime.tracker_client, "fetch_available_rooms", _failing_rooms)
    monkeypatch.setattr(
        main_module,
        "_scan_all_lan_beacons",
        AsyncMock(
            return_value=[
                {
                    "node_id": "cpu-lan-peer-1",
                    "room_id": "lan-cpu-room",
                    "ip_address": "192.168.1.50",
                    "p2p_port": 49160,
                    "shard_layer_start": 16,
                    "shard_layer_end": 23,
                    "role": "primary",
                    "ram_gb": 6.5,
                },
                {
                    "node_id": "cpu-lan-peer-2",
                    "room_id": "lan-cpu-room",
                    "ip_address": "192.168.1.51",
                    "p2p_port": 49161,
                    "shard_layer_start": 24,
                    "shard_layer_end": 31,
                    "role": "primary",
                    "ram_gb": 8,
                },
            ]
        ),
    )

    await runtime.refresh_room_coverage()

    assert runtime.network_status == NetworkStatus.LAN_ONLY
    assert runtime.coverage_percent == 100.0
    assert runtime.is_swarm_ready is True
    assert [p.available_ram_gb for p in runtime.peers] == [6.5, 8.0]
    assert runtime.room_total_shared_ram_gb == 18.5


@pytest.mark.parametrize(
    ("raw_payload", "expected_ram_gb"),
    [
        ({"available_ram_gb": 12.5}, 12.5),
        ({"shared_ram_gb": 8}, 8.0),
        ({"ram_gb": "6.25"}, 6.25),
        ({"available_ram_gb": 0.0, "shared_ram_gb": 4.0}, 4.0),
        ({"available_ram_gb": None, "ram_gb": None}, 0.0),
        ({"available_ram_gb": "not-a-number"}, 0.0),
        ({"available_ram_gb": float("nan")}, 0.0),
        ({"available_ram_gb": float("inf")}, 0.0),
        ({"available_ram_gb": -5.0}, 0.0),
        ({"available_ram_gb": True}, 0.0),
        ({}, 0.0),
    ],
)
def test_normalize_peer_available_ram_gb(
    raw_payload: dict[str, Any],
    expected_ram_gb: float,
) -> None:
    """_normalize_peer extracts and sanitizes available_ram_gb safely."""
    base: dict[str, Any] = {
        "node_id": "peer-ram-test",
        "ip": "203.0.113.99",
        "port": 49152,
        "assigned_start_layer": 0,
        "assigned_end_layer": 15,
    }
    base.update(raw_payload)
    peer = _normalize_peer(base, default_room_id="public")
    assert peer.available_ram_gb == expected_ram_gb


@pytest.mark.parametrize(
    ("ram_gb_value", "expected_ram_gb"),
    [
        (8, 8.0),
        ("12.5", 12.5),
        (None, 0.0),
        ("invalid", 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (-3.0, 0.0),
        (True, 0.0),
    ],
)
def test_beacon_to_peer_available_ram_gb(
    ram_gb_value: Any,
    expected_ram_gb: float,
) -> None:
    """_beacon_to_peer populates available_ram_gb from beacon ram_gb safely."""
    beacon: dict[str, Any] = {
        "node_id": "lan-peer-ram",
        "ip_address": "192.168.1.40",
        "p2p_port": 49152,
        "shard_layer_start": 0,
        "shard_layer_end": 15,
        "ram_gb": ram_gb_value,
    }
    peer = _beacon_to_peer(beacon, room_id="public", total_layers=32)
    assert peer.available_ram_gb == expected_ram_gb


@pytest.mark.asyncio
async def test_fetch_peers_preserves_available_ram_gb_from_tracker() -> None:
    """SignalingTrackerClient.fetch_peers preserves peer available_ram_gb."""
    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "peer-cpu-1",
                        "ip": "198.51.100.11",
                        "port": 49152,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 15,
                        "available_vram_mb": 0,
                        "available_ram_gb": 10.5,
                    },
                    {
                        "node_id": "peer-cpu-2",
                        "ip": "198.51.100.12",
                        "port": 49153,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                        "available_vram_mb": 0,
                        "shared_ram_gb": 6.0,
                    },
                ]
            },
        )

    tracker = SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        transport=httpx.MockTransport(_handler),
    )
    peers = await tracker.fetch_peers(model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert [p.available_ram_gb for p in peers] == [10.5, 6.0]
