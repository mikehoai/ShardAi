"""Regression tests for Issue #64: Deduplicate cloud-announced nodes in /available-rooms."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import struct
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

import main as main_module
from main import app, runtime
from models.downloader import GGUF_MAGIC, ModelDownloader
import p2p.discovery as discovery_module
from p2p.discovery import ZeroconfLanDiscovery, _beacon_to_txt
from p2p.mdns_discovery import (
    PROBE_MAGIC,
    LanMdnsService,
    LanRoomBeacon,
    _MdnsResponderProtocol,
    lan_mdns_service,
)
from shard_engine.schemas import (
    ModelSourceType,
    NetworkMode,
    NetworkStatus,
    NodeState,
)
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import patch_start_node_runner

client = TestClient(app)


def _write_synthetic_gguf(target_path: Path, total_size: int = 1_100_000) -> Path:
    """Create a minimal valid GGUF v3 header file for local/catalog model tests."""
    key_bytes = b"llama.block_count"
    header = bytearray()
    header.extend(GGUF_MAGIC)
    header.extend(struct.pack("<IQQ", 3, 0, 1))
    header.extend(struct.pack("<Q", len(key_bytes)))
    header.extend(key_bytes)
    header.extend(struct.pack("<II", 4, 32))
    if total_size > len(header):
        header.extend(b"\x00" * (total_size - len(header)))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(bytes(header))
    return target_path


def _reset_runtime() -> None:
    runtime.session_token = None
    runtime.memory_manager.release_all()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.model_id = None
    runtime.model_path = None
    runtime.mmproj_path = None
    runtime.modality = None
    runtime.room_id = None
    runtime.room_key_hash = None
    runtime.peers = []
    runtime.network_mode = None
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.coverage_percent = 0.0
    runtime.is_swarm_ready = False
    runtime._last_heartbeat_at = 0.0
    runtime._active_start_request = None
    runtime._active_max_layers_can_hold = 0
    runtime.p2p_listen_port = None
    runtime.n_layers = 0


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Iterator[None]:
    """Restore runtime, tracker client, downloader, and mDNS beacon around each test."""
    orig_tracker = runtime.tracker_client
    orig_downloader = runtime.model_downloader
    orig_beacon = lan_mdns_service._active_beacon
    _reset_runtime()
    lan_mdns_service._active_beacon = None
    yield
    runtime.tracker_client = orig_tracker
    runtime.model_downloader = orig_downloader
    lan_mdns_service._active_beacon = orig_beacon
    _reset_runtime()


def _make_mock_tracker(handler: Any) -> SignalingTrackerClient:
    return SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=1,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_public_internet_room_creator_not_duplicated_in_available_rooms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Public Internet room creator sees 1 node / 2.0 GB, not 2 clients / 4.0 GB."""
    patch_start_node_runner(monkeypatch)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")

    models_dir = tmp_path / "models"
    model_path = _write_synthetic_gguf(models_dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    downloader = ModelDownloader(
        cache_dir=tmp_path / "cache",
        models_dir=models_dir,
        catalog_cache_path=tmp_path / "models_cache.json",
    )
    monkeypatch.setattr(
        downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=model_path),
    )
    runtime.model_downloader = downloader

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/announce":
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "public-internet-room",
                            "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                            "modality": "text",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 2.0,
                            "is_private": False,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        if request.url.path == "/unregister":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)

    async def _fake_start_advertising(beacon: LanRoomBeacon) -> None:
        lan_mdns_service._active_beacon = beacon

    async def _fake_stop_advertising() -> None:
        lan_mdns_service._active_beacon = None

    monkeypatch.setattr(lan_mdns_service, "start_advertising", _fake_start_advertising)
    monkeypatch.setattr(lan_mdns_service, "stop_advertising", _fake_stop_advertising)
    monkeypatch.setattr("p2p.mdns_discovery._list_local_ipv4_addresses", lambda: [])
    monkeypatch.setattr(
        main_module.zeroconf_discovery,
        "discover_peers",
        AsyncMock(return_value=[]),
    )

    try:
        start_res = client.post(
            "/start-node",
            json={
                "node_id": "creator-online-node",
                "model_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "source_type": ModelSourceType.CATALOG.value,
                "network_mode": NetworkMode.INTERNET.value,
                "room_id": "public-internet-room",
                "ram_gb": 2,
                "p2p_listen_port": 59201,
            },
        )
        assert start_res.status_code == 200, start_res.text
        assert lan_mdns_service._active_beacon is not None
        assert lan_mdns_service._active_beacon.announced_to_cloud is True

        rooms_res = client.get("/available-rooms")
        assert rooms_res.status_code == 200
        rooms = {r["room_id"]: r for r in rooms_res.json()}
        room = rooms["public-internet-room"]
        assert room["nodes_online"] == 1
        assert room["total_shared_ram_gb"] == 2.0
        assert room["network_scope"] == "internet"
    finally:
        if runtime.state != NodeState.STOPPED:
            client.post(
                "/stop-node",
                headers={"Authorization": f"Bearer {runtime.session_token or ''}"},
            )


@pytest.mark.asyncio
async def test_lan_peer_view_deduplicates_cloud_announced_beacon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LAN peer seeing a cloud room + UDP beacon with announced_to_cloud=True keeps 1 node."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "shared-internet-room",
                            "model": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                            "modality": None,
                            "nodes_online": 1,
                            "total_shared_ram_gb": 2.0,
                            "is_private": False,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)

    async def _fake_scan_lan_beacons(
        udp_timeout_seconds: float = 0.25,
        zeroconf_wait_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        _ = (udp_timeout_seconds, zeroconf_wait_seconds)
        return [
            {
                "node_id": "remote-creator-node",
                "room_id": "shared-internet-room",
                "model": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                "modality": "vision",
                "ram_gb": 2.0,
                "is_private": False,
                "p2p_port": 49152,
                "shard_layer_start": 0,
                "shard_layer_end": 31,
                "total_layers": 32,
                "announced_to_cloud": True,
            }
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}
    room = rooms["shared-internet-room"]
    assert room["nodes_online"] == 1
    assert room["total_shared_ram_gb"] == 2.0
    assert room["network_scope"] == "internet"
    assert room["modality"] == "vision"


@pytest.mark.asyncio
async def test_true_hybrid_room_merges_only_lan_only_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid merge counts the cloud node once and adds only `announced_to_cloud=False` LAN peers."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "true-hybrid-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "modality": "text",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 2.0,
                            "is_private": False,
                            "coverage_percent": 50.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)

    async def _fake_scan_lan_beacons(
        udp_timeout_seconds: float = 0.25,
        zeroconf_wait_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        _ = (udp_timeout_seconds, zeroconf_wait_seconds)
        return [
            {
                "node_id": "cloud-announced-peer",
                "room_id": "true-hybrid-room",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "ram_gb": 2.0,
                "is_private": False,
                "p2p_port": 49152,
                "shard_layer_start": 0,
                "shard_layer_end": 15,
                "total_layers": 32,
                "announced_to_cloud": True,
            },
            {
                "node_id": "lan-only-peer",
                "room_id": "true-hybrid-room",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "ram_gb": 4.0,
                "is_private": False,
                "p2p_port": 49153,
                "shard_layer_start": 16,
                "shard_layer_end": 31,
                "total_layers": 32,
                "announced_to_cloud": False,
            },
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}
    room = rooms["true-hybrid-room"]
    assert room["nodes_online"] == 2
    assert room["total_shared_ram_gb"] == 6.0
    assert room["network_scope"] == "hybrid"


@pytest.mark.asyncio
async def test_legacy_self_beacon_without_flag_deduplicated_when_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy beacon lacking `announced_to_cloud` is deduplicated for matching online `runtime.node_id`."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    runtime.state = NodeState.RUNNING
    runtime.network_status = NetworkStatus.ONLINE
    runtime.node_id = "legacy-self-node"

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "legacy-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "modality": "text",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 2.0,
                            "is_private": False,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)

    async def _fake_scan_lan_beacons(
        udp_timeout_seconds: float = 0.25,
        zeroconf_wait_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        _ = (udp_timeout_seconds, zeroconf_wait_seconds)
        return [
            {
                "node_id": "legacy-self-node",
                "room_id": "legacy-room",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "ram_gb": 2.0,
                "is_private": False,
                "p2p_port": 49152,
                "shard_layer_start": 0,
                "shard_layer_end": 31,
                "total_layers": 32,
            }
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}
    room = rooms["legacy-room"]
    assert room["nodes_online"] == 1
    assert room["total_shared_ram_gb"] == 2.0
    assert room["network_scope"] == "internet"


@pytest.mark.asyncio
async def test_offline_tracker_surfaces_lan_beacon_as_lan_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Cloud Tracker is unreachable, LAN beacon surfaces with `network_scope='lan'`."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    async def _failing_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("tracker unreachable")

    runtime.tracker_client = _make_mock_tracker(_failing_handler)

    async def _fake_scan_lan_beacons(
        udp_timeout_seconds: float = 0.25,
        zeroconf_wait_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        _ = (udp_timeout_seconds, zeroconf_wait_seconds)
        return [
            {
                "node_id": "offline-lan-node",
                "room_id": "offline-lan-room",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "ram_gb": 2.0,
                "is_private": False,
                "p2p_port": 49152,
                "shard_layer_start": 0,
                "shard_layer_end": 31,
                "total_layers": 32,
                "announced_to_cloud": False,
            }
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}
    room = rooms["offline-lan-room"]
    assert room["nodes_online"] == 1
    assert room["total_shared_ram_gb"] == 2.0
    assert room["network_scope"] == "lan"


@pytest.mark.asyncio
async def test_udp_json_and_zeroconf_txt_round_trip_announced_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UDP JSON responder, LanMdnsService local beacon, and Zeroconf TXT preserve `announced_to_cloud`."""
    sent_packets: list[tuple[bytes, tuple[str, int]]] = []

    class _FakeTransport:
        def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
            sent_packets.append((data, addr))

    online_beacon = LanRoomBeacon(
        node_id="node-cloud-true",
        room_id="room-a",
        model="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=2.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49170,
        announced_to_cloud=True,
    )
    offline_beacon = LanRoomBeacon(
        node_id="node-cloud-false",
        room_id="room-b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=2.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49171,
        announced_to_cloud=False,
    )

    proto_true = _MdnsResponderProtocol(online_beacon)
    proto_true.connection_made(_FakeTransport())  # type: ignore[arg-type]
    proto_true.datagram_received(PROBE_MAGIC, ("192.168.1.50", 54321))

    proto_false = _MdnsResponderProtocol(offline_beacon)
    proto_false.connection_made(_FakeTransport())  # type: ignore[arg-type]
    proto_false.datagram_received(PROBE_MAGIC, ("192.168.1.51", 54322))

    assert json.loads(sent_packets[0][0].decode("utf-8"))["announced_to_cloud"] is True
    assert json.loads(sent_packets[1][0].decode("utf-8"))["announced_to_cloud"] is False

    svc = LanMdnsService()
    svc._active_beacon = online_beacon
    monkeypatch.setattr("p2p.mdns_discovery._list_local_ipv4_addresses", lambda: [])
    results = await svc.scan_lan_rooms(timeout_seconds=0.01)
    by_node = {r["node_id"]: r for r in results}
    assert by_node["node-cloud-true"]["announced_to_cloud"] is True

    txt_true = _beacon_to_txt(online_beacon)
    txt_false = _beacon_to_txt(offline_beacon)
    assert txt_true["announced_to_cloud"] == "1"
    assert txt_false["announced_to_cloud"] == "0"

    class _FakeAsyncServiceInfo:
        def __init__(self, service_type: str, name: str) -> None:
            self.service_type = service_type
            self.name = name
            self.port = 49170
            raw_txt = txt_true if "true" in name else txt_false
            self.properties = {
                k.encode("utf-8"): v.encode("utf-8") for k, v in raw_txt.items()
            }

        async def async_request(self, _zc: Any, _timeout_ms: int) -> bool:
            return True

        def parsed_addresses(self, _ip_version: Any = None) -> list[str]:
            return ["192.168.1.88"]

    monkeypatch.setattr(discovery_module, "_AsyncServiceInfo", _FakeAsyncServiceInfo)

    disc = ZeroconfLanDiscovery()
    await disc._resolve_service(object(), "_shardai._tcp.local.", "node-cloud-true._shardai._tcp.local.")  # type: ignore[arg-type]
    await disc._resolve_service(object(), "_shardai._tcp.local.", "node-cloud-false._shardai._tcp.local.")  # type: ignore[arg-type]

    assert disc._peers["node-cloud-true._shardai._tcp.local."]["announced_to_cloud"] is True
    assert disc._peers["node-cloud-false._shardai._tcp.local."]["announced_to_cloud"] is False


@pytest.mark.asyncio
async def test_refresh_room_coverage_updates_beacon_on_cloud_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When transitioning from offline to ONLINE, active beacon is re-advertised with announced_to_cloud=True."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "recovered-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "modality": "text",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 2.0,
                            "is_private": False,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        if request.url.path == "/announce":
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 31,
                    "role": "primary",
                    "peers": [],
                },
            )
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)
    runtime.state = NodeState.RUNNING
    runtime.network_mode = NetworkMode.INTERNET
    runtime.network_status = NetworkStatus.OFFLINE_SOLO
    runtime.node_id = "recovering-node"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "recovered-room"
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32

    initial_beacon = LanRoomBeacon(
        node_id="recovering-node",
        room_id="recovered-room",
        model="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=2.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49152,
        announced_to_cloud=False,
    )
    lan_mdns_service._active_beacon = initial_beacon

    readvertised: list[LanRoomBeacon] = []

    async def _fake_start_advertising(beacon: LanRoomBeacon) -> None:
        readvertised.append(beacon)
        lan_mdns_service._active_beacon = beacon

    monkeypatch.setattr(lan_mdns_service, "start_advertising", _fake_start_advertising)
    monkeypatch.setattr(
        main_module,
        "_scan_all_lan_beacons",
        AsyncMock(return_value=[]),
    )

    await runtime.refresh_room_coverage()

    assert runtime.network_status == NetworkStatus.ONLINE
    assert len(readvertised) == 1
    assert readvertised[0].announced_to_cloud is True
    assert lan_mdns_service._active_beacon is not None
    assert lan_mdns_service._active_beacon.announced_to_cloud is True
