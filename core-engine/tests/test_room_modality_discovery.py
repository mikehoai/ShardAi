"""Unit and integration tests for Issue #47: Room modality preservation and advertisement."""

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
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import patch_start_node_runner

client = TestClient(app)


def _write_synthetic_gguf(target_path: Path, total_size: int = 128) -> Path:
    """Create a minimal valid GGUF v3 header file for local model tests."""
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
    """Restore runtime, tracker client, and downloader around each test."""
    orig_tracker = runtime.tracker_client
    orig_downloader = runtime.model_downloader
    _reset_runtime()
    yield
    runtime.tracker_client = orig_tracker
    runtime.model_downloader = orig_downloader
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
async def test_get_available_rooms_hybrid_merge_preserves_and_falls_back_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid room merge preserves `existing.modality` and falls back to `lan_modality` when None."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "hybrid-remote-vision",
                            "model": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                            "modality": "vision",
                            "nodes_online": 1,
                            "total_shared_ram_gb": 8.0,
                            "is_private": False,
                            "coverage_percent": 50.0,
                            "redundancy_factor": 1,
                        },
                        {
                            "room_id": "hybrid-lan-vision",
                            "model": "custom/vision-model",
                            "modality": None,
                            "nodes_online": 1,
                            "total_shared_ram_gb": 8.0,
                            "is_private": False,
                            "coverage_percent": 50.0,
                            "redundancy_factor": 1,
                        },
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
                "node_id": "lan-peer-1",
                "room_id": "hybrid-remote-vision",
                "model": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                "ram_gb": 8.0,
                "is_private": False,
                "p2p_port": 49152,
                "shard_layer_start": 16,
                "shard_layer_end": 31,
                "total_layers": 32,
            },
            {
                "node_id": "lan-peer-2",
                "room_id": "hybrid-lan-vision",
                "model": "custom/vision-model",
                "modality": "vision",
                "ram_gb": 8.0,
                "is_private": False,
                "p2p_port": 49153,
                "shard_layer_start": 16,
                "shard_layer_end": 31,
                "total_layers": 32,
            },
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}

    assert rooms["hybrid-remote-vision"]["network_scope"] == "hybrid"
    assert rooms["hybrid-remote-vision"]["modality"] == "vision"
    assert rooms["hybrid-remote-vision"]["nodes_online"] == 2

    assert rooms["hybrid-lan-vision"]["network_scope"] == "hybrid"
    assert rooms["hybrid-lan-vision"]["modality"] == "vision"
    assert rooms["hybrid-lan-vision"]["nodes_online"] == 2


@pytest.mark.asyncio
async def test_get_available_rooms_lan_only_surfaces_vision_and_sanitizes_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LAN-only rooms surface `modality='vision'` and safely ignore unrecognized modality values."""
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "false")

    async def _fake_scan_lan_beacons(
        udp_timeout_seconds: float = 0.25,
        zeroconf_wait_seconds: float = 0.0,
    ) -> list[dict[str, Any]]:
        _ = (udp_timeout_seconds, zeroconf_wait_seconds)
        return [
            {
                "node_id": "lan-vision-0",
                "room_id": "lan-vision-room",
                "model": "local/my-vision-model.gguf",
                "modality": "vision",
                "ram_gb": 8.0,
                "is_private": False,
                "p2p_port": 49160,
                "shard_layer_start": 0,
                "shard_layer_end": 15,
                "total_layers": 32,
            },
            {
                "node_id": "lan-vision-1",
                "room_id": "lan-vision-room",
                "model": "local/my-vision-model.gguf",
                "modality": "text",
                "ram_gb": 8.0,
                "is_private": False,
                "p2p_port": 49161,
                "shard_layer_start": 16,
                "shard_layer_end": 31,
                "total_layers": 32,
            },
            {
                "node_id": "lan-corrupt-0",
                "room_id": "lan-corrupt-room",
                "model": "custom/text-model",
                "modality": "invalid_modality",
                "ram_gb": 4.0,
                "is_private": False,
                "p2p_port": 49162,
                "shard_layer_start": 0,
                "shard_layer_end": 31,
                "total_layers": 32,
            },
        ]

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan_beacons)

    res = client.get("/available-rooms")
    assert res.status_code == 200
    rooms = {r["room_id"]: r for r in res.json()}

    assert rooms["lan-vision-room"]["network_scope"] == "lan"
    assert rooms["lan-vision-room"]["modality"] == "vision"
    assert rooms["lan-vision-room"]["nodes_online"] == 2

    assert rooms["lan-corrupt-room"]["network_scope"] == "lan"
    assert rooms["lan-corrupt-room"]["modality"] is None


@pytest.mark.asyncio
async def test_lan_beacon_udp_json_round_trip_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UDP `LanRoomBeacon` responder and `LanMdnsService.scan_lan_rooms` round-trip `modality`."""
    sent_packets: list[tuple[bytes, tuple[str, int]]] = []

    class _FakeTransport:
        def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
            sent_packets.append((data, addr))

    vision_beacon = LanRoomBeacon(
        node_id="node-udp-vision",
        room_id="udp-vision-room",
        model="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        ram_gb=8.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49170,
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        role="primary",
        modality="vision",
    )
    protocol = _MdnsResponderProtocol(vision_beacon)
    protocol.connection_made(_FakeTransport())  # type: ignore[arg-type]
    protocol.datagram_received(PROBE_MAGIC, ("192.168.1.50", 54321))

    assert len(sent_packets) == 1
    decoded_udp = json.loads(sent_packets[0][0].decode("utf-8"))
    assert decoded_udp["type"] == "SHARDAI_MDNS_ROOM_V1"
    assert decoded_udp["modality"] == "vision"

    # Legacy beacon without modality omits the field cleanly
    legacy_beacon = LanRoomBeacon(
        node_id="node-udp-legacy",
        room_id="udp-legacy-room",
        model="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=8.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49171,
    )
    legacy_protocol = _MdnsResponderProtocol(legacy_beacon)
    legacy_protocol.connection_made(_FakeTransport())  # type: ignore[arg-type]
    legacy_protocol.datagram_received(PROBE_MAGIC, ("192.168.1.51", 54322))
    decoded_legacy = json.loads(sent_packets[1][0].decode("utf-8"))
    assert "modality" not in decoded_legacy

    # Local active beacon in LanMdnsService.scan_lan_rooms preserves modality
    svc = LanMdnsService()
    svc._active_beacon = vision_beacon
    monkeypatch.setattr("p2p.mdns_discovery._list_local_ipv4_addresses", lambda: [])
    results = await svc.scan_lan_rooms(timeout_seconds=0.01)
    assert len(results) >= 1
    by_node = {r["node_id"]: r for r in results}
    assert by_node["node-udp-vision"]["modality"] == "vision"


@pytest.mark.asyncio
async def test_zeroconf_txt_round_trip_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_beacon_to_txt` and `ZeroconfLanDiscovery._resolve_service` round-trip `modality`."""
    vision_beacon = LanRoomBeacon(
        node_id="zc-vision-node",
        room_id="zc-vision-room",
        model="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        ram_gb=8.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49180,
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        role="primary",
        modality="vision",
    )
    txt = _beacon_to_txt(vision_beacon)
    assert txt["modality"] == "vision"

    legacy_beacon = LanRoomBeacon(
        node_id="zc-legacy-node",
        room_id="zc-legacy-room",
        model="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=8.0,
        is_private=False,
        is_unlisted=False,
        p2p_port=49181,
    )
    assert "modality" not in _beacon_to_txt(legacy_beacon)

    class _FakeAsyncServiceInfo:
        def __init__(self, service_type: str, name: str) -> None:
            self.service_type = service_type
            self.name = name
            self.port = 49180
            raw_txt = txt if "vision" in name else _beacon_to_txt(legacy_beacon)
            self.properties = {
                k.encode("utf-8"): v.encode("utf-8") for k, v in raw_txt.items()
            }

        async def async_request(self, _zc: Any, _timeout_ms: int) -> bool:
            return True

        def parsed_addresses(self, _ip_version: Any = None) -> list[str]:
            return ["192.168.1.77"]

    monkeypatch.setattr(discovery_module, "_AsyncServiceInfo", _FakeAsyncServiceInfo)

    disc = ZeroconfLanDiscovery()
    await disc._resolve_service(object(), "_shardai._tcp.local.", "zc-vision-node._shardai._tcp.local.")  # type: ignore[arg-type]
    await disc._resolve_service(object(), "_shardai._tcp.local.", "zc-legacy-node._shardai._tcp.local.")  # type: ignore[arg-type]

    assert disc._peers["zc-vision-node._shardai._tcp.local."]["modality"] == "vision"
    assert disc._peers["zc-legacy-node._shardai._tcp.local."]["modality"] is None


@pytest.mark.asyncio
async def test_tracker_client_announce_transmits_optional_modality() -> None:
    """`POST /announce` includes `modality` when provided and omits it when absent."""
    captured_payloads: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/announce"
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            status_code=200,
            json={
                "assigned_start_layer": 0,
                "assigned_end_layer": 31,
                "role": "primary",
                "peers": [],
            },
        )

    tracker = _make_mock_tracker(_handler)
    req = StartNodeRequest(
        node_id="announce-modality-node",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        room_id="vision-room",
        ram_gb=8,
        p2p_listen_port=49190,
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
    )

    await tracker.register_node_with_assignment(req, modality="vision")
    await tracker.register_node_with_assignment(req, modality=None)

    assert len(captured_payloads) == 2
    assert captured_payloads[0]["modality"] == "vision"
    assert "modality" not in captured_payloads[1]


@pytest.mark.asyncio
async def test_start_node_lan_custom_local_vision_surfaces_modality_in_available_rooms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Starting a LAN node with a local `.gguf` + sibling `mmproj` advertises `modality='vision'`."""
    patch_start_node_runner(monkeypatch)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0.0")

    models_dir = tmp_path / "models"
    model_file = _write_synthetic_gguf(models_dir / "custom-vl-7b.gguf")
    _write_synthetic_gguf(models_dir / "mmproj-custom-vl-f16.gguf")

    downloader = ModelDownloader(
        cache_dir=tmp_path / "cache",
        models_dir=models_dir,
        catalog_cache_path=tmp_path / "models_cache.json",
    )
    runtime.model_downloader = downloader

    monkeypatch.setattr(lan_mdns_service, "start_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(lan_mdns_service, "stop_advertising", AsyncMock(return_value=None))

    try:
        res = client.post(
            "/start-node",
            json={
                "node_id": "custom-local-vl-node",
                "model_id": str(model_file),
                "source_type": ModelSourceType.LOCAL_FILE.value,
                "network_mode": NetworkMode.LAN.value,
                "room_id": "custom-lan-vision-room",
                "ram_gb": 8,
                "p2p_listen_port": 59195,
            },
        )
        assert res.status_code == 200, res.text
        assert runtime.modality == "vision"

        advertised_beacon = lan_mdns_service.start_advertising.call_args[0][0]  # type: ignore[attr-defined]
        assert advertised_beacon.modality == "vision"

        # Simulate LAN room discovery finding that advertised beacon
        lan_mdns_service._active_beacon = advertised_beacon
        monkeypatch.setattr("p2p.mdns_discovery._list_local_ipv4_addresses", lambda: [])
        monkeypatch.setattr(
            main_module.zeroconf_discovery,
            "discover_peers",
            AsyncMock(return_value=[]),
        )

        rooms_res = client.get("/available-rooms")
        assert rooms_res.status_code == 200
        rooms = {r["room_id"]: r for r in rooms_res.json()}
        assert rooms["custom-lan-vision-room"]["network_scope"] == "lan"
        assert rooms["custom-lan-vision-room"]["modality"] == "vision"
    finally:
        lan_mdns_service._active_beacon = None
        if runtime.state != NodeState.STOPPED:
            client.post(
                "/stop-node",
                headers={"Authorization": f"Bearer {runtime.session_token or ''}"},
            )


@pytest.mark.asyncio
async def test_start_node_online_vision_catalog_transmits_modality_in_announce_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Online startup with a vision catalog model sends `modality='vision'` in `POST /announce`."""
    patch_start_node_runner(monkeypatch)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")

    models_dir = tmp_path / "models"
    catalog_cache = tmp_path / "models_cache.json"
    catalog_cache.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "qwen2.5-vl-7b",
                        "name": "Qwen2.5-VL 7B Instruct",
                        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                        "filename": "qwen2.5-vl-7b-instruct-q4_k_m.gguf",
                        "mmproj_filename": "mmproj-model-f16.gguf",
                        "modality": "vision",
                        "estimated_ram_gb": 7.5,
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    model_path = _write_synthetic_gguf(
        models_dir / "qwen2.5-vl-7b-instruct-q4_k_m.gguf",
        total_size=1_100_000,
    )
    mmproj_path = _write_synthetic_gguf(
        models_dir / "mmproj-model-f16.gguf",
        total_size=1_100_000,
    )

    downloader = ModelDownloader(
        cache_dir=tmp_path / "cache",
        models_dir=models_dir,
        catalog_cache_path=catalog_cache,
    )
    downloader._resolved_mmproj_paths["Qwen/Qwen2.5-VL-7B-Instruct-GGUF"] = mmproj_path
    monkeypatch.setattr(
        downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=model_path),
    )
    runtime.model_downloader = downloader

    announce_payloads: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/announce":
            announce_payloads.append(json.loads(request.content.decode("utf-8")))
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
            return httpx.Response(status_code=200, json={"rooms": []})
        if request.url.path == "/unregister":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)
    monkeypatch.setattr(lan_mdns_service, "start_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(lan_mdns_service, "stop_advertising", AsyncMock(return_value=None))

    try:
        res = client.post(
            "/start-node",
            json={
                "node_id": "online-vision-node",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
                "source_type": ModelSourceType.CATALOG.value,
                "network_mode": NetworkMode.INTERNET.value,
                "room_id": "online-vision-room",
                "ram_gb": 8,
                "p2p_listen_port": 59196,
            },
        )
        assert res.status_code == 200, res.text
        assert len(announce_payloads) >= 1
        assert all(p.get("modality") == "vision" for p in announce_payloads)

        advertised_beacon = lan_mdns_service.start_advertising.call_args[0][0]  # type: ignore[attr-defined]
        assert advertised_beacon.modality == "vision"

        # Heartbeat re-announce preserves modality="vision"
        sent = await runtime.maybe_send_tracker_heartbeat(force=True)
        assert sent is True
        assert announce_payloads[-1].get("modality") == "vision"
    finally:
        if runtime.state != NodeState.STOPPED:
            client.post(
                "/stop-node",
                headers={"Authorization": f"Bearer {runtime.session_token or ''}"},
            )


@pytest.mark.asyncio
async def test_start_node_text_model_advertises_text_modality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Standard text models advertise `modality='text'` to `POST /announce` and `LanRoomBeacon`."""
    patch_start_node_runner(monkeypatch)
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")

    models_dir = tmp_path / "models"
    model_path = _write_synthetic_gguf(
        models_dir / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        total_size=1_100_000,
    )
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

    announce_payloads: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/announce":
            announce_payloads.append(json.loads(request.content.decode("utf-8")))
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
            return httpx.Response(status_code=200, json={"rooms": []})
        if request.url.path == "/unregister":
            return httpx.Response(status_code=200, json={"ok": True})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_handler)
    monkeypatch.setattr(lan_mdns_service, "start_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(lan_mdns_service, "stop_advertising", AsyncMock(return_value=None))

    try:
        res = client.post(
            "/start-node",
            json={
                "node_id": "online-text-node",
                "model_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "source_type": ModelSourceType.CATALOG.value,
                "network_mode": NetworkMode.INTERNET.value,
                "room_id": "online-text-room",
                "ram_gb": 8,
                "p2p_listen_port": 59197,
            },
        )
        assert res.status_code == 200, res.text
        assert len(announce_payloads) >= 1
        assert all(p.get("modality") == "text" for p in announce_payloads)
        advertised_beacon = lan_mdns_service.start_advertising.call_args[0][0]  # type: ignore[attr-defined]
        assert advertised_beacon.modality == "text"
    finally:
        if runtime.state != NodeState.STOPPED:
            client.post(
                "/stop-node",
                headers={"Authorization": f"Bearer {runtime.session_token or ''}"},
            )
