"""Unit & async concurrency tests for Issue #5:
1. Periodic Signaling Tracker heartbeat re-announce (`POST /announce`) before 60s TTL expiry.
2. Serialized `/start-node` & `/stop-node` lifecycle transitions via `get_lifecycle_lock()`.
3. Online-mode P2P TCP listener binding on `0.0.0.0` (while FastAPI stays on `127.0.0.1`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import AsyncMock

from fastapi import HTTPException
import httpx
import pytest

import main as main_module
from main import (
    _heartbeat_interval_seconds,
    runtime,
    start_node,
    stop_node,
)
from models.shard_runner import ShardModelRunner
from shard_engine.schemas import (
    NetworkMode,
    NetworkStatus,
    NodeState,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient, hash_room_key
from tests._mock_runner import patch_start_node_runner


@pytest.fixture(autouse=True)
async def reset_lifecycle_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Guarantee a clean stopped runtime before and after every async test."""
    original_tracker = runtime.tracker_client
    monkeypatch.delenv("SHARDAI_P2P_BIND_HOST", raising=False)
    monkeypatch.delenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0.0")
    await stop_node()
    runtime.last_tracker_reachable = None
    yield
    await stop_node()
    runtime.tracker_client = original_tracker
    runtime.last_tracker_reachable = None


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


def _stub_fast_local_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stub disk downloads and LAN discovery so tests run fast and deterministically."""
    fake_model = tmp_path / "mock-model.gguf"
    fake_model.write_bytes(b"GGUF" + b"\x00" * 4096)

    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fake_model),
    )
    monkeypatch.setattr(
        main_module,
        "_scan_all_lan_beacons",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        main_module.lan_mdns_service,
        "start_advertising",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        main_module.lan_mdns_service,
        "stop_advertising",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        main_module.zeroconf_discovery,
        "start_advertising",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        main_module.zeroconf_discovery,
        "stop_advertising",
        AsyncMock(return_value=None),
    )
    patch_start_node_runner(monkeypatch)
    return fake_model


def test_heartbeat_interval_clamped_below_60s_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default heartbeat interval is 25s and is always clamped strictly below the 60s TTL."""
    monkeypatch.delenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    assert _heartbeat_interval_seconds() == 25.0

    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "90")
    assert _heartbeat_interval_seconds() == 50.0

    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "invalid-number")
    assert _heartbeat_interval_seconds() == 25.0


@pytest.mark.asyncio
async def test_online_node_heartbeat_reannounces_before_60s_ttl_and_keeps_swarm_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Online nodes periodically re-announce via POST /announce with room_key_hash."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "25.0")

    announce_payloads: list[dict[str, Any]] = []
    expected_key_hash = hash_room_key("top-secret-room-key")

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/announce":
            body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
            announce_payloads.append(body)
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "online-heartbeat-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 8.0,
                            "is_private": True,
                            "network_scope": "internet",
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        if request.method == "GET" and request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=200, json={"peers": []})
        if request.method == "POST" and request.url.path == "/nodes/leave":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    runtime.tracker_client = _make_mock_tracker(_handler)

    start_resp = await start_node(
        StartNodeRequest(
            node_id="heartbeat-node-01",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            network_mode=NetworkMode.INTERNET,
            room_id="online-heartbeat-room",
            room_key="top-secret-room-key",
            ram_gb=8,
            shared_ram_gb=8,
            max_ram_mb=8192,
            p2p_listen_port=49410,
            shard_layer_start=0,
            shard_layer_end=31,
            total_layers=32,
        )
    )
    assert start_resp.state == NodeState.RUNNING
    assert start_resp.network_status == NetworkStatus.ONLINE
    assert start_resp.is_swarm_ready is True
    assert start_resp.coverage_percent == 100.0
    # Plaintext room_key must never be retained in active start request snapshot
    assert runtime._active_start_request is not None
    assert runtime._active_start_request.room_key is None
    assert runtime.room_key_hash == expected_key_hash

    startup_announce_count = len(announce_payloads)
    assert startup_announce_count >= 1

    # Immediate poll before heartbeat interval elapses must NOT send duplicate POST /announce
    await runtime.refresh_room_coverage()
    assert len(announce_payloads) == startup_announce_count

    # Advance monotonic delta past the 25s interval (well before 60s tracker TTL)
    runtime._last_heartbeat_at = time.monotonic() - 30.0
    await runtime.refresh_room_coverage()

    assert len(announce_payloads) == startup_announce_count + 1
    hb = announce_payloads[-1]
    assert hb["node_id"] == "heartbeat-node-01"
    assert hb["room_id"] == "online-heartbeat-room"
    assert hb["is_private"] is True
    assert hb["room_key_hash"] == expected_key_hash
    assert hb["p2p_port"] == start_resp.p2p_listen_port
    assert hb["shard_layer_start"] == start_resp.shard_layer_start
    assert hb["shard_layer_end"] == start_resp.shard_layer_end
    assert runtime.is_swarm_ready is True
    assert runtime.coverage_percent == 100.0

    # Next immediate poll after successful heartbeat must not re-announce again
    await runtime.refresh_room_coverage()
    assert len(announce_payloads) == startup_announce_count + 1


@pytest.mark.asyncio
async def test_heartbeat_retries_on_next_tick_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If a heartbeat POST /announce fails transiently, the next poll tick retries immediately."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "25.0")

    announce_calls = 0
    fail_next_announce = False

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal announce_calls, fail_next_announce
        if request.method == "POST" and request.url.path == "/announce":
            announce_calls += 1
            if fail_next_announce:
                fail_next_announce = False
                return httpx.Response(status_code=503, json={"detail": "transient error"})
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(status_code=200, json={"rooms": []})
        if request.method == "GET" and request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=200, json={"peers": []})
        if request.method == "POST" and request.url.path == "/nodes/leave":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler, max_retries=1)

    await start_node(
        StartNodeRequest(
            node_id="heartbeat-retry-node",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            network_mode=NetworkMode.INTERNET,
            room_id="retry-room",
            ram_gb=8,
            shared_ram_gb=8,
            max_ram_mb=8192,
            p2p_listen_port=49420,
            shard_layer_start=0,
            shard_layer_end=31,
            total_layers=32,
        )
    )
    baseline_calls = announce_calls

    # Force heartbeat due and fail the next /announce
    fail_next_announce = True
    stale_ts = time.monotonic() - 30.0
    runtime._last_heartbeat_at = stale_ts
    await runtime.refresh_room_coverage()
    assert announce_calls == baseline_calls + 1
    assert runtime._last_heartbeat_at == stale_ts  # Not advanced on failure

    # Very next poll tick retries without waiting another 25 seconds and succeeds
    await runtime.refresh_room_coverage()
    assert announce_calls == baseline_calls + 2
    assert runtime._last_heartbeat_at > stale_ts


@pytest.mark.asyncio
async def test_heartbeat_skipped_in_lan_and_offline_solo_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Nodes running in OFFLINE_SOLO or LAN_ONLY mode must never send cloud heartbeats."""
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")
    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "0.05")

    announce_calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal announce_calls
        if request.method == "POST" and request.url.path == "/announce":
            announce_calls += 1
        return httpx.Response(status_code=200, json={"rooms": [], "peers": []})

    runtime.tracker_client = _make_mock_tracker(_handler)

    await start_node(
        StartNodeRequest(
            node_id="offline-solo-node",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            network_mode=NetworkMode.LAN,
            room_id="lan-room",
            ram_gb=8,
            shared_ram_gb=8,
            max_ram_mb=8192,
            p2p_listen_port=49430,
            shard_layer_start=0,
            shard_layer_end=31,
            total_layers=32,
        )
    )
    assert runtime.network_status == NetworkStatus.OFFLINE_SOLO

    runtime._last_heartbeat_at = 0.0
    assert await runtime.maybe_send_tracker_heartbeat(force=True) is False
    await runtime.refresh_room_coverage()

    # Even if network_status is LAN_ONLY and tracker env is enabled, no cloud heartbeat fires
    _enable_tracker(monkeypatch)
    runtime.network_status = NetworkStatus.LAN_ONLY
    assert await runtime.maybe_send_tracker_heartbeat(force=True) is False
    assert announce_calls == 0


@pytest.mark.asyncio
async def test_concurrent_stop_during_start_serializes_and_ends_cleanly_stopped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A /stop-node arriving while /start-node is STARTING must wait and leave node STOPPED."""
    fake_model = _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")

    entered_download = asyncio.Event()
    release_download = asyncio.Event()

    async def _slow_download(*_args: Any, **_kwargs: Any) -> Path:
        entered_download.set()
        await release_download.wait()
        return fake_model

    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        _slow_download,
    )

    start_task = asyncio.create_task(
        start_node(
            StartNodeRequest(
                node_id="race-start-stop-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="race-room",
                ram_gb=4,
                shared_ram_gb=4,
                max_ram_mb=4096,
                p2p_listen_port=49440,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    )
    await asyncio.wait_for(entered_download.wait(), timeout=3.0)
    assert runtime.state == NodeState.STARTING

    # Issue concurrent /stop-node while start_node is still inside STARTING
    stop_task = asyncio.create_task(stop_node())
    await asyncio.sleep(0.05)
    assert not stop_task.done(), "stop_node must wait for _lifecycle_lock while STARTING"

    release_download.set()
    start_res, stop_res = await asyncio.gather(start_task, stop_task)

    assert start_res.state == NodeState.RUNNING
    assert stop_res.state == NodeState.STOPPED
    assert runtime.state == NodeState.STOPPED
    assert runtime.p2p_server is None
    assert runtime.p2p_bind_host is None
    assert runtime.model_runner is None
    assert runtime.session_token is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0
    assert runtime.memory_manager.is_loaded is False
    assert runtime._active_start_request is None


@pytest.mark.asyncio
async def test_concurrent_double_start_serializes_without_leaking_p2p_server_or_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two concurrent /start-node calls must serialize and release the first node's resources."""
    fake_model = _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    download_calls = 0
    created_servers: list[asyncio.AbstractServer] = []
    released_runners: list[ShardModelRunner] = []

    orig_start_listener = runtime.start_p2p_listener
    orig_release = ShardModelRunner.release

    async def _tracking_start_listener(
        port: int, network_mode: NetworkMode = NetworkMode.INTERNET
    ) -> int:
        bound = await orig_start_listener(port, network_mode=network_mode)
        assert runtime.p2p_server is not None
        created_servers.append(runtime.p2p_server)
        return bound

    def _tracking_release(self: ShardModelRunner) -> None:
        released_runners.append(self)
        orig_release(self)

    async def _gated_download(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal download_calls
        download_calls += 1
        if download_calls == 1:
            first_entered.set()
            await release_first.wait()
        return fake_model

    monkeypatch.setattr(runtime, "start_p2p_listener", _tracking_start_listener)
    monkeypatch.setattr(ShardModelRunner, "release", _tracking_release)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        _gated_download,
    )

    first_task = asyncio.create_task(
        start_node(
            StartNodeRequest(
                node_id="concurrent-node-A",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="room-a",
                ram_gb=4,
                shared_ram_gb=4,
                max_ram_mb=4096,
                p2p_listen_port=49450,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    )
    await asyncio.wait_for(first_entered.wait(), timeout=3.0)
    assert runtime.state == NodeState.STARTING
    assert len(created_servers) == 1

    second_task = asyncio.create_task(
        start_node(
            StartNodeRequest(
                node_id="concurrent-node-B",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="room-b",
                ram_gb=6,
                shared_ram_gb=6,
                max_ram_mb=6144,
                p2p_listen_port=49460,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    )
    await asyncio.sleep(0.05)
    assert not second_task.done(), "Second start_node must wait for _lifecycle_lock"
    assert download_calls == 1, "Second start_node must not enter startup concurrently"

    release_first.set()
    first_res, second_res = await asyncio.gather(first_task, second_task)

    assert first_res.node_id == "concurrent-node-A"
    assert second_res.node_id == "concurrent-node-B"
    assert runtime.state == NodeState.RUNNING
    assert runtime.node_id == "concurrent-node-B"
    assert runtime.room_id == "room-b"
    assert len(created_servers) == 2
    assert not created_servers[0].is_serving(), "First P2P TCP server must be closed cleanly"
    assert created_servers[1].is_serving(), "Second P2P TCP server must be actively serving"
    assert runtime.p2p_server is created_servers[1]
    assert len(released_runners) == 1, "First ShardModelRunner must be released before second load"
    assert runtime.memory_manager.allocated_ram_mb > 0


@pytest.mark.asyncio
async def test_failed_start_node_releases_lifecycle_lock_and_resources_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failed start_node (409 rejection or download exception) releases lock and resources cleanly."""
    _enable_tracker(monkeypatch)
    fake_model = _stub_fast_local_env(monkeypatch, tmp_path)

    # Case 1: 409 Conflict from Signaling Tracker -> STOPPED, p2p_server closed, lock released
    def _reject_409_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/announce":
            return httpx.Response(
                status_code=409,
                json={"error": {"code": "ROOM_FULL", "message": "Room is full."}},
            )
        return httpx.Response(status_code=200, json={"rooms": [], "peers": []})

    runtime.tracker_client = _make_mock_tracker(_reject_409_handler)
    with pytest.raises(HTTPException) as exc_409:
        await start_node(
            StartNodeRequest(
                node_id="reject-409-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="full-room",
                ram_gb=4,
                shared_ram_gb=4,
                max_ram_mb=4096,
                p2p_listen_port=49475,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    assert exc_409.value.status_code == 409
    assert runtime.state == NodeState.STOPPED
    assert runtime.p2p_server is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0
    assert runtime._active_start_request is None
    assert not runtime.get_lifecycle_lock().locked()

    # Case 2: Internal download error -> ERROR, p2p_server closed, lock released, retry succeeds
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(side_effect=RuntimeError("simulated disk write failure")),
    )
    with pytest.raises(HTTPException) as exc_500:
        await start_node(
            StartNodeRequest(
                node_id="fail-500-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="err-room",
                ram_gb=4,
                shared_ram_gb=4,
                max_ram_mb=4096,
                p2p_listen_port=49476,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    assert exc_500.value.status_code == 500
    assert runtime.state == NodeState.ERROR
    assert runtime.p2p_server is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime._active_start_request is None
    assert not runtime.get_lifecycle_lock().locked()

    # Subsequent start_node after ERROR must acquire lock without hanging and succeed cleanly
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fake_model),
    )
    retry_res = await asyncio.wait_for(
        start_node(
            StartNodeRequest(
                node_id="recover-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="recover-room",
                ram_gb=4,
                shared_ram_gb=4,
                max_ram_mb=4096,
                p2p_listen_port=49477,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        ),
        timeout=5.0,
    )
    assert retry_res.state == NodeState.RUNNING
    assert retry_res.node_id == "recover-node"


@pytest.mark.asyncio
async def test_refresh_room_coverage_ignores_stale_tick_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale background refresh_room_coverage ticks must not mutate a stopped runtime."""
    _enable_tracker(monkeypatch)
    runtime.state = NodeState.RUNNING
    runtime.network_status = NetworkStatus.ONLINE
    runtime.node_id = "stale-tick-node"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "stale-room"
    runtime._last_heartbeat_at = time.monotonic()

    async def _stopping_fetch_rooms() -> list[Any]:
        runtime.state = NodeState.STOPPED
        return []

    monkeypatch.setattr(
        runtime.tracker_client,
        "fetch_available_rooms",
        _stopping_fetch_rooms,
    )

    await runtime.refresh_room_coverage()
    assert runtime.state == NodeState.STOPPED
    assert runtime.coverage_percent == 0.0
    assert runtime.is_swarm_ready is False


@pytest.mark.asyncio
async def test_start_p2p_listener_binds_all_interfaces_in_internet_and_lan_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2P TCP listener binds 0.0.0.0 by default in both INTERNET and LAN modes, with env override."""
    monkeypatch.delenv("SHARDAI_P2P_BIND_HOST", raising=False)

    port_internet = await runtime.start_p2p_listener(0, network_mode=NetworkMode.INTERNET)
    assert port_internet > 0
    assert runtime.p2p_bind_host == "0.0.0.0"
    assert runtime.p2p_server is not None and runtime.p2p_server.sockets
    assert runtime.p2p_server.sockets[0].getsockname()[0] == "0.0.0.0"

    port_lan = await runtime.start_p2p_listener(0, network_mode=NetworkMode.LAN)
    assert port_lan > 0
    assert runtime.p2p_bind_host == "0.0.0.0"
    assert runtime.p2p_server is not None and runtime.p2p_server.sockets
    assert runtime.p2p_server.sockets[0].getsockname()[0] == "0.0.0.0"

    monkeypatch.setenv("SHARDAI_P2P_BIND_HOST", "127.0.0.1")
    port_override = await runtime.start_p2p_listener(0, network_mode=NetworkMode.INTERNET)
    assert port_override > 0
    assert runtime.p2p_bind_host == "127.0.0.1"
    assert runtime.p2p_server is not None and runtime.p2p_server.sockets
    assert runtime.p2p_server.sockets[0].getsockname()[0] == "127.0.0.1"

    await runtime.stop_p2p_listener()
    assert runtime.p2p_server is None
    assert runtime.p2p_bind_host is None


@pytest.mark.asyncio
async def test_online_start_node_binds_p2p_to_0_0_0_0_while_api_stays_loopback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Online mode start_node binds P2P TCP listener to 0.0.0.0 while HTTP API stays on 127.0.0.1."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.delenv("SHARDAI_P2P_BIND_HOST", raising=False)

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/announce":
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(status_code=200, json={"rooms": []})
        if request.method == "GET" and request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=200, json={"peers": []})
        if request.method == "POST" and request.url.path == "/nodes/leave":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)

    res = await start_node(
        StartNodeRequest(
            node_id="online-bind-node",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            network_mode=NetworkMode.INTERNET,
            room_id="online-bind-room",
            ram_gb=8,
            shared_ram_gb=8,
            max_ram_mb=8192,
            p2p_listen_port=49470,
            shard_layer_start=0,
            shard_layer_end=31,
            total_layers=32,
        )
    )
    assert res.state == NodeState.RUNNING
    assert res.network_status == NetworkStatus.ONLINE
    assert runtime.p2p_bind_host == "0.0.0.0"
    assert runtime.p2p_server is not None and runtime.p2p_server.sockets
    assert runtime.p2p_server.sockets[0].getsockname()[0] == "0.0.0.0"
    assert res.api_base_url is not None
    assert res.api_base_url.startswith("http://127.0.0.1:")


def _make_recording_tracker(
    announce_calls: list[dict[str, Any]],
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]],
    *,
    announce_status: int = 200,
    announce_json: dict[str, Any] | None = None,
    announce_exc: Exception | None = None,
    leave_status: int = 200,
) -> SignalingTrackerClient:
    """Create a MockTransport SignalingTrackerClient recording /announce and /nodes/leave."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/announce":
            body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
            announce_calls.append(body)
            if announce_exc is not None:
                raise announce_exc
            if announce_status != 200:
                return httpx.Response(
                    status_code=announce_status,
                    json=announce_json
                    or {"error": {"code": "ROOM_FULL", "message": "Room is full."}},
                )
            return httpx.Response(
                status_code=200,
                json=announce_json
                or {
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "POST" and request.url.path == "/nodes/leave":
            leave_body: dict[str, Any] = json.loads(request.content.decode("utf-8"))
            leave_calls.append((leave_body, dict(request.headers)))
            return httpx.Response(
                status_code=leave_status,
                json={"ok": leave_status < 400},
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(status_code=200, json={"rooms": []})
        if request.method == "GET" and request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=200, json={"peers": []})
        return httpx.Response(status_code=404)

    return _make_mock_tracker(_handler, max_retries=1)


@pytest.mark.asyncio
async def test_post_announce_download_exception_unregisters_zombie_node_with_private_key_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #27: Post-announce download failure must immediately call POST /nodes/leave."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(side_effect=RuntimeError("download checksum mismatch")),
    )

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_calls, leave_calls)

    secret_room_key = "my-private-room-secret"
    expected_key_hash = hash_room_key(secret_room_key)

    with pytest.raises(HTTPException) as exc_info:
        await start_node(
            StartNodeRequest(
                node_id="zombie-download-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="private-zombie-room",
                room_key=secret_room_key,
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49480,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )

    assert exc_info.value.status_code == 500
    assert "download checksum mismatch" in str(exc_info.value.detail)
    assert len(announce_calls) == 1
    assert len(leave_calls) == 1

    leave_payload, leave_headers = leave_calls[0]
    assert leave_payload["node_id"] == "zombie-download-node"
    assert leave_payload["room_id"] == "private-zombie-room"
    assert leave_payload["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert leave_payload["room_key_hash"] == expected_key_hash
    assert leave_headers.get("x-room-key-hash") == expected_key_hash
    assert "room_key" not in leave_payload
    assert secret_room_key not in caplog.text

    assert runtime.state == NodeState.ERROR
    assert runtime.node_id is None
    assert runtime.room_id is None
    assert runtime.room_key_hash is None
    assert runtime.p2p_server is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0

    # Subsequent /stop-node after failed start must NOT issue a duplicate POST /nodes/leave
    stop_res = await stop_node()
    assert stop_res.state == NodeState.STOPPED
    assert len(leave_calls) == 1


@pytest.mark.asyncio
async def test_post_announce_http_exception_unregisters_zombie_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #27: Post-announce HTTPException must still unregister the announced node."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(
            side_effect=HTTPException(status_code=502, detail="mirror failed")
        ),
    )

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_calls, leave_calls)

    with pytest.raises(HTTPException) as exc_info:
        await start_node(
            StartNodeRequest(
                node_id="zombie-http-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="http-fail-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49481,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "mirror failed"
    assert len(announce_calls) == 1
    assert len(leave_calls) == 1
    assert leave_calls[0][0]["node_id"] == "zombie-http-node"
    assert runtime.state == NodeState.ERROR
    assert runtime.node_id is None
    assert runtime.p2p_server is None
    assert runtime.memory_manager.allocated_ram_mb == 0


@pytest.mark.asyncio
async def test_post_announce_shard_runner_oom_unregisters_normalized_node_id_and_releases_ram(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #27: ShardModelRunner OOM unregisters exact normalized node_id and frees RAM."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)

    def _oom_runner(*args: Any, **kwargs: Any) -> ShardModelRunner:
        raise MemoryError("simulated OOM during model init")

    monkeypatch.setattr(main_module, "ShardModelRunner", _oom_runner)

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_calls, leave_calls)

    with pytest.raises(HTTPException) as exc_info:
        await start_node(
            StartNodeRequest(
                node_id="shardai-client-01",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="oom-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49482,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )

    assert exc_info.value.status_code == 500
    assert "simulated OOM during model init" in str(exc_info.value.detail)
    assert len(announce_calls) >= 1
    announced_id = announce_calls[0]["node_id"]
    assert announced_id != "shardai-client-01"
    assert announced_id.startswith("shardai-client-01-")
    assert all(call["node_id"] == announced_id for call in announce_calls)

    assert len(leave_calls) == 1
    assert leave_calls[0][0]["node_id"] == announced_id
    assert leave_calls[0][0]["room_id"] == "oom-room"
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0
    assert runtime.model_runner is None
    assert runtime.p2p_server is None
    assert runtime.state == NodeState.ERROR


@pytest.mark.asyncio
async def test_post_announce_mdns_failure_unregisters_zombie_node_and_releases_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #27: Post-runner mDNS failure unregisters node and releases ShardModelRunner."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)

    release_calls = 0
    original_release = ShardModelRunner.release

    def _spy_release(self: ShardModelRunner) -> None:
        nonlocal release_calls
        release_calls += 1
        original_release(self)

    monkeypatch.setattr(ShardModelRunner, "release", _spy_release)
    monkeypatch.setattr(
        main_module.lan_mdns_service,
        "start_advertising",
        AsyncMock(side_effect=OSError("UDP multicast error")),
    )

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_calls, leave_calls)

    with pytest.raises(HTTPException) as exc_info:
        await start_node(
            StartNodeRequest(
                node_id="zombie-mdns-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="mdns-fail-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49483,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )

    assert exc_info.value.status_code == 500
    assert len(announce_calls) >= 1
    assert len(leave_calls) == 1
    assert leave_calls[0][0]["node_id"] == "zombie-mdns-node"
    assert release_calls == 1
    assert runtime.model_runner is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.p2p_server is None
    assert runtime.state == NodeState.ERROR


@pytest.mark.asyncio
async def test_rollback_unregister_failure_preserves_original_error_and_local_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #27: Rollback /nodes/leave failure must not mask original error or leak resources."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(side_effect=RuntimeError("corrupted model weights")),
    )

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(
        announce_calls,
        leave_calls,
        leave_status=503,
    )

    with pytest.raises(HTTPException) as exc_info:
        await start_node(
            StartNodeRequest(
                node_id="rollback-leave-fail-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="leave-fail-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49484,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )

    assert exc_info.value.status_code == 500
    assert "corrupted model weights" in str(exc_info.value.detail)
    assert len(announce_calls) == 1
    assert len(leave_calls) == 1
    assert "Failed to unregister node 'rollback-leave-fail-node'" in caplog.text
    assert runtime.p2p_server is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.node_id is None
    assert runtime.state == NodeState.ERROR
    assert not runtime.get_lifecycle_lock().locked()


@pytest.mark.asyncio
async def test_pre_announce_rejection_and_offline_fallback_do_not_call_unregister(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #27: Pre-announce rejections, fallback modes, and LAN failures never call /nodes/leave."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)

    # Subcase A: 409 Conflict during POST /announce
    announce_a: list[dict[str, Any]] = []
    leave_a: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(
        announce_a, leave_a, announce_status=409
    )
    with pytest.raises(HTTPException) as exc_a:
        await start_node(
            StartNodeRequest(
                node_id="pre-announce-409",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="full-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49485,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    assert exc_a.value.status_code == 409
    assert len(leave_a) == 0
    assert runtime.state == NodeState.STOPPED

    # Subcase B: Tracker timeout -> OFFLINE_SOLO fallback, followed by download failure
    announce_b: list[dict[str, Any]] = []
    leave_b: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(
        announce_b,
        leave_b,
        announce_exc=httpx.ConnectTimeout("Tracker timed out"),
    )
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(side_effect=RuntimeError("offline disk error")),
    )
    with pytest.raises(HTTPException) as exc_b:
        await start_node(
            StartNodeRequest(
                node_id="fallback-offline-fail",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="solo-fallback-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49486,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    assert exc_b.value.status_code == 500
    assert len(leave_b) == 0
    assert runtime.state == NodeState.ERROR

    # Subcase C: LAN mode download failure -> never calls /nodes/leave
    announce_c: list[dict[str, Any]] = []
    leave_c: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_c, leave_c)
    with pytest.raises(HTTPException) as exc_c:
        await start_node(
            StartNodeRequest(
                node_id="lan-fail-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.LAN,
                room_id="lan-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49487,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    assert exc_c.value.status_code == 500
    assert len(announce_c) == 0
    assert len(leave_c) == 0
    assert runtime.state == NodeState.ERROR


@pytest.mark.asyncio
async def test_post_announce_task_cancellation_unregisters_zombie_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #27: Cancelling startup post-announce must unregister node and transition to STOPPED."""
    _enable_tracker(monkeypatch)
    _stub_fast_local_env(monkeypatch, tmp_path)

    download_entered = asyncio.Event()
    never_set = asyncio.Event()

    async def _blocking_download(*_args: Any, **_kwargs: Any) -> Path:
        download_entered.set()
        await never_set.wait()
        return tmp_path / "unreachable.gguf"

    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        _blocking_download,
    )

    announce_calls: list[dict[str, Any]] = []
    leave_calls: list[tuple[dict[str, Any], dict[str, str]]] = []
    runtime.tracker_client = _make_recording_tracker(announce_calls, leave_calls)

    start_task = asyncio.create_task(
        start_node(
            StartNodeRequest(
                node_id="cancel-zombie-node",
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                network_mode=NetworkMode.INTERNET,
                room_id="cancel-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49488,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
    )
    await asyncio.wait_for(download_entered.wait(), timeout=5.0)
    start_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert len(announce_calls) == 1
    assert len(leave_calls) == 1
    assert leave_calls[0][0]["node_id"] == "cancel-zombie-node"
    assert leave_calls[0][0]["room_id"] == "cancel-room"
    assert runtime.state == NodeState.STOPPED
    assert runtime.p2p_server is None
    assert runtime.node_id is None
    assert not runtime.get_lifecycle_lock().locked()


@pytest.mark.asyncio
async def test_build_heartbeat_request_clamps_low_allocated_ram_to_min_two_gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback heartbeat builder clamps `ram_gb`/`shared_ram_gb` to `StartNodeRequest` bounds."""
    _enable_tracker(monkeypatch)
    runtime._active_start_request = None
    runtime.state = NodeState.RUNNING
    runtime.network_status = NetworkStatus.ONLINE
    runtime.node_id = "hb-low-ram-node"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "hb-low-ram-room"
    runtime.p2p_listen_port = 49415
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32
    runtime._last_heartbeat_at = 0.0

    for allocated_mb in (0, 256, 512, 1024):
        runtime.memory_manager._allocated_ram_mb = allocated_mb
        hb_req = runtime._build_heartbeat_request()
        assert hb_req is not None
        assert hb_req.ram_gb == 2
        assert hb_req.shared_ram_gb == 2
        assert hb_req.max_ram_mb == 2048

    runtime.memory_manager._allocated_ram_mb = 2560
    hb_mid = runtime._build_heartbeat_request()
    assert hb_mid is not None
    assert hb_mid.ram_gb == 3
    assert hb_mid.shared_ram_gb == 3
    assert hb_mid.max_ram_mb == 3072

    runtime.memory_manager._allocated_ram_mb = 300_000
    hb_high = runtime._build_heartbeat_request()
    assert hb_high is not None
    assert hb_high.ram_gb == 256
    assert hb_high.shared_ram_gb == 256
    assert hb_high.max_ram_mb == 256 * 1024

    runtime.memory_manager._allocated_ram_mb = 512
    posted_payloads: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/announce":
            posted_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        return httpx.Response(status_code=200, json={"peers": []})

    runtime.tracker_client = _make_mock_tracker(_handler)
    assert await runtime.maybe_send_tracker_heartbeat(force=True) is True
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["ram_gb"] == 2
