"""Unit and integration tests for Issue #10: Signaling client peer discovery and unregister."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

from main import app, runtime
from shard_engine.schemas import NetworkMode, NetworkStatus, NodeState, StartNodeRequest
from shard_engine.tracker_client import (
    SignalingTrackerClient,
    TrackerUnreachableError,
    hash_room_key,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Iterator[None]:
    """Reset runtime state and restore the original tracker client around each test."""
    original_tracker = runtime.tracker_client
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.model_id = None
    runtime.room_id = None
    runtime.room_key_hash = None
    runtime.peers = []
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    yield
    runtime.tracker_client = original_tracker
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.model_id = None
    runtime.room_id = None
    runtime.room_key_hash = None
    runtime.peers = []
    runtime.network_status = None


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


@pytest.mark.asyncio
async def test_fetch_peers_sends_model_query_param_and_handles_slash_model_id() -> None:
    """Slash-containing `owner/repo` model IDs send percent-encoded path + `model` query param."""
    captured_requests: list[httpx.Request] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "data": {
                    "peers": [
                        {
                            "node_id": "peer-remote-01",
                            "ip": "198.51.100.10",
                            "port": 49152,
                            "assigned_start_layer": 16,
                            "assigned_end_layer": 31,
                            "role": "primary",
                        }
                    ]
                },
            },
        )

    tracker = _make_mock_tracker(_handler)
    peers = await tracker.fetch_peers(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        room_id="public",
    )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.method == "GET"
    assert req.url.raw_path.split(b"?", 1)[0] == (
        b"/peers/public/meta-llama%2FLlama-3.1-8B-Instruct"
    )
    assert req.url.params["room_id"] == "public"
    assert req.url.params["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert len(peers) == 1
    assert peers[0].peer_id == "peer-remote-01"
    assert peers[0].ip_address == "198.51.100.10"


@pytest.mark.asyncio
async def test_fetch_peers_falls_back_to_query_endpoint_on_404_for_slash_model() -> None:
    """HTTP 404 on `/peers/{room}/{model}` triggers fallback to `GET /peers` with same params/headers."""
    captured_requests: list[httpx.Request] = []
    secret = "lab-secret-key"
    expected_hash = hash_room_key(secret)
    assert expected_hash is not None

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        path_only = request.url.path
        if path_only.startswith("/peers/"):
            return httpx.Response(status_code=404, json={"error": "Not Found"})
        assert path_only == "/peers"
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "peer-same-model",
                        "ip": "203.0.113.25",
                        "port": 49200,
                        "shard_layer_start": 0,
                        "shard_layer_end": 15,
                        "role": "primary",
                    }
                ]
            },
        )

    tracker = _make_mock_tracker(_handler)
    peers = await tracker.fetch_peers(
        model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        room_id="private-lab",
        room_key=secret,
        node_id="node-alpha",
    )

    assert len(captured_requests) == 2
    first_req, fallback_req = captured_requests
    assert first_req.url.path.startswith("/peers/private-lab/")
    assert fallback_req.url.path == "/peers"
    for req in (first_req, fallback_req):
        assert req.url.params["room_id"] == "private-lab"
        assert req.url.params["model"] == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
        assert req.url.params["node_id"] == "node-alpha"
        assert req.url.params["room_key_hash"] == expected_hash
        assert req.headers.get("X-Room-Key-Hash") == expected_hash
        assert secret not in str(req.url)

    assert len(peers) == 1
    assert peers[0].peer_id == "peer-same-model"


@pytest.mark.asyncio
async def test_fetch_peers_and_check_room_join_propagate_403_and_409_without_404_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403/409 propagate immediately with 1 request; 404 fallback preserves 403/409 errors too."""
    _enable_tracker(monkeypatch)
    seen_paths: list[str] = []

    async def _direct_403_handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(
            status_code=403,
            json={"error": {"code": "FORBIDDEN", "message": "Invalid Room Key"}},
        )

    tracker_403 = _make_mock_tracker(_direct_403_handler, max_retries=3)
    with pytest.raises(httpx.HTTPStatusError) as exc_403:
        await tracker_403.fetch_peers(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            room_id="vault-1",
            room_key="bad-key",
        )
    assert exc_403.value.response.status_code == 403
    assert len(seen_paths) == 1

    seen_paths.clear()

    async def _direct_409_handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(
            status_code=409,
            json={"error": {"code": "ROOM_FULL", "message": "Room is full."}},
        )

    tracker_409 = _make_mock_tracker(_direct_409_handler, max_retries=3)
    with pytest.raises(httpx.HTTPStatusError) as exc_409:
        await tracker_409.check_room_join(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            room_id="full-room",
        )
    assert exc_409.value.response.status_code == 409
    assert len(seen_paths) == 1

    fallback_paths: list[str] = []

    async def _fallback_409_handler(request: httpx.Request) -> httpx.Response:
        fallback_paths.append(request.url.path)
        if request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=404, json={"error": "Not Found"})
        return httpx.Response(
            status_code=409,
            json={"error": {"code": "ROOM_FULL", "message": "Room full on fallback."}},
        )

    runtime.tracker_client = _make_mock_tracker(_fallback_409_handler)
    res = client.post(
        "/rooms/check-join",
        json={
            "room_id": "full-slash-room",
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "network_mode": "internet",
        },
    )
    assert res.status_code == 409
    assert res.json()["detail"] == "Room full on fallback."
    assert len(fallback_paths) == 2
    assert fallback_paths[1] == "/peers"


@pytest.mark.asyncio
async def test_fetch_peers_sends_x_room_key_hash_header_and_never_plaintext() -> None:
    """Private rooms send `X-Room-Key-Hash` and `room_key_hash`; public/blank keys omit both."""
    requests_seen: list[httpx.Request] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return httpx.Response(status_code=200, json={"peers": []})

    tracker = _make_mock_tracker(_handler)

    secret = "top-secret-passphrase"
    expected_hash = hash_room_key(secret)
    assert expected_hash is not None

    await tracker.fetch_peers(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        room_id="private-room",
        room_key=secret,
    )
    await tracker.fetch_peers(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        room_id="private-room",
        room_key_hash=expected_hash,
    )
    await tracker.fetch_peers(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        room_id="public",
        room_key="   ",
    )

    assert len(requests_seen) == 3
    for priv_req in requests_seen[:2]:
        assert priv_req.headers.get("X-Room-Key-Hash") == expected_hash
        assert priv_req.url.params.get("room_key_hash") == expected_hash
        assert secret not in str(priv_req.url)
        assert "room_key" not in priv_req.url.params

    pub_req = requests_seen[2]
    assert "X-Room-Key-Hash" not in pub_req.headers
    assert "room_key_hash" not in pub_req.url.params


@pytest.mark.asyncio
async def test_fetch_peers_same_nat_self_filtering_by_node_id() -> None:
    """Same-NAT peers sharing one public IP are preserved while self `node_id` is excluded."""
    captured_params: dict[str, str] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "node-a",
                        "ip": "203.0.113.50",
                        "port": 49152,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 15,
                    },
                    {
                        "node_id": "node-b",
                        "ip": "203.0.113.50",
                        "port": 49153,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                    },
                ]
            },
        )

    tracker = _make_mock_tracker(_handler)
    peers = await tracker.fetch_peers(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        room_id="nat-office",
        node_id="node-a",
    )

    assert captured_params.get("node_id") == "node-a"
    assert [peer.peer_id for peer in peers] == ["node-b"]
    assert peers[0].ip_address == "203.0.113.50"
    assert peers[0].port == 49153


@pytest.mark.asyncio
async def test_register_node_with_assignment_filters_self_and_forwards_node_id_to_fallback_fetch_peers() -> None:
    """Embedded `/announce` peers exclude self, and omitted peers trigger `fetch_peers` with `node_id` + `404` fallback."""
    secret = "team-vault-key"
    expected_hash = hash_room_key(secret)
    assert expected_hash is not None

    # Case A: /announce embeds peers (including self + same-NAT peer)
    announce_requests_a: list[httpx.Request] = []

    async def _embedded_handler(request: httpx.Request) -> httpx.Response:
        announce_requests_a.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "assigned_start_layer": 0,
                "assigned_end_layer": 15,
                "role": "primary",
                "peers": [
                    {
                        "node_id": "node-self",
                        "ip": "203.0.113.50",
                        "port": 49152,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 15,
                    },
                    {
                        "node_id": "node-peer",
                        "ip": "203.0.113.50",
                        "port": 49153,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                    },
                ],
            },
        )

    tracker_a = _make_mock_tracker(_embedded_handler)
    cfg = StartNodeRequest(
        node_id="node-self",
        model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        network_mode=NetworkMode.INTERNET,
        room_id="team-nat",
        room_key=secret,
        ram_gb=4,
        p2p_listen_port=49152,
    )
    assignment_a = await tracker_a.register_node_with_assignment(cfg)
    assert len(announce_requests_a) == 1
    assert announce_requests_a[0].headers.get("X-Room-Key-Hash") == expected_hash
    assert [p.peer_id for p in assignment_a.peers] == ["node-peer"]

    # Case B: /announce omits "peers" -> follow-up fetch_peers with node_id + 404 fallback
    requests_b: list[httpx.Request] = []

    async def _followup_handler(request: httpx.Request) -> httpx.Response:
        requests_b.append(request)
        if request.method == "POST" and request.url.path == "/announce":
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 15,
                    "role": "primary",
                },
            )
        if request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=404, json={"error": "Not Found"})
        assert request.url.path == "/peers"
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "node-self",
                        "ip": "203.0.113.50",
                        "port": 49152,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 15,
                    },
                    {
                        "node_id": "node-other",
                        "ip": "203.0.113.50",
                        "port": 49154,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                    },
                ]
            },
        )

    tracker_b = _make_mock_tracker(_followup_handler)
    assignment_b = await tracker_b.register_node_with_assignment(cfg)
    assert len(requests_b) == 3
    fallback_peer_req = requests_b[2]
    assert fallback_peer_req.url.path == "/peers"
    assert fallback_peer_req.url.params.get("node_id") == "node-self"
    assert fallback_peer_req.url.params.get("model") == "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    assert fallback_peer_req.url.params.get("room_key_hash") == expected_hash
    assert fallback_peer_req.headers.get("X-Room-Key-Hash") == expected_hash
    assert [p.peer_id for p in assignment_b.peers] == ["node-other"]


@pytest.mark.asyncio
async def test_refresh_room_coverage_passes_node_id_and_discovers_same_nat_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`refresh_room_coverage()` forwards `node_id`, `model`, and `X-Room-Key-Hash` and keeps same-NAT peers."""
    _enable_tracker(monkeypatch)
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")
    expected_hash = hash_room_key("room-secret")
    assert expected_hash is not None
    peer_requests: list[httpx.Request] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "nat-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "nodes_online": 2,
                            "total_shared_ram_gb": 12.0,
                            "is_private": True,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        peer_requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "node-a",
                        "ip": "203.0.113.50",
                        "port": 49152,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 15,
                    },
                    {
                        "node_id": "node-b",
                        "ip": "203.0.113.50",
                        "port": 49153,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                    },
                ]
            },
        )

    runtime.tracker_client = _make_mock_tracker(_handler)
    runtime.state = NodeState.RUNNING
    runtime.network_status = NetworkStatus.ONLINE
    runtime.node_id = "node-a"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "nat-room"
    runtime.room_key_hash = expected_hash
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32

    await runtime.refresh_room_coverage()

    assert len(peer_requests) == 1
    req = peer_requests[0]
    assert req.url.params.get("node_id") == "node-a"
    assert req.url.params.get("model") == "meta-llama/Llama-3.1-8B-Instruct"
    assert req.url.params.get("room_key_hash") == expected_hash
    assert req.headers.get("X-Room-Key-Hash") == expected_hash
    assert [p.peer_id for p in runtime.peers] == ["node-b"]


@pytest.mark.asyncio
async def test_unregister_node_sends_model_room_key_hash_header_and_retries_503() -> None:
    """`unregister_node()` sends `model`, `room_key_hash`, and `X-Room-Key-Hash` and retries 503."""
    attempts = 0
    captured_requests: list[httpx.Request] = []
    secret = "private-leave-secret"
    expected_hash = hash_room_key(secret)
    assert expected_hash is not None

    async def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        captured_requests.append(request)
        if attempts == 1:
            return httpx.Response(status_code=503, json={"error": "Transient error"})
        return httpx.Response(status_code=200, json={"success": True})

    tracker = _make_mock_tracker(_handler, max_retries=3)
    await tracker.unregister_node(
        node_id="node-leave-01",
        room_id="vault-room",
        model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        room_key=secret,
    )

    assert attempts == 2
    req = captured_requests[-1]
    assert req.method == "POST"
    assert req.url.path == "/nodes/leave"
    assert req.headers.get("X-Room-Key-Hash") == expected_hash
    body = json.loads(req.content.decode("utf-8"))
    assert body == {
        "node_id": "node-leave-01",
        "room_id": "vault-room",
        "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "room_key_hash": expected_hash,
    }
    assert "room_key" not in body

    # Blank/None optional fields are cleanly omitted
    public_requests: list[httpx.Request] = []

    async def _public_handler(request: httpx.Request) -> httpx.Response:
        public_requests.append(request)
        return httpx.Response(status_code=200, json={"success": True})

    public_tracker = _make_mock_tracker(_public_handler)
    await public_tracker.unregister_node(
        node_id="node-leave-02",
        room_id="  ",
        model_id="",
        room_key="   ",
    )
    pub_req = public_requests[0]
    assert "X-Room-Key-Hash" not in pub_req.headers
    assert json.loads(pub_req.content.decode("utf-8")) == {"node_id": "node-leave-02"}


def test_start_and_stop_node_end_to_end_private_slash_model_discovery_and_leave(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end `/start-node` -> `/stop-node` sends slash model, `node_id`, and `room_key_hash` on discovery and leave."""
    _enable_tracker(monkeypatch)
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")

    secret = "e2e-private-key"
    expected_hash = hash_room_key(secret)
    assert expected_hash is not None

    seen_requests: list[tuple[str, str, dict[str, str], dict[str, str], dict[str, Any] | None]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] | None = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        seen_requests.append((
            request.method,
            request.url.path,
            dict(request.url.params),
            dict(request.headers),
            body,
        ))

        if request.method == "POST" and request.url.path == "/announce":
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 15,
                    "role": "primary",
                },
            )
        if request.method == "GET" and request.url.path.startswith("/peers/"):
            return httpx.Response(status_code=404, json={"error": "Not Found"})
        if request.method == "GET" and request.url.path == "/peers":
            return httpx.Response(
                status_code=200,
                json={
                    "peers": [
                        {
                            "node_id": "e2e-node-alpha",
                            "ip": "203.0.113.50",
                            "port": 49345,
                            "assigned_start_layer": 0,
                            "assigned_end_layer": 15,
                        },
                        {
                            "node_id": "e2e-node-beta",
                            "ip": "203.0.113.50",
                            "port": 49346,
                            "assigned_start_layer": 16,
                            "assigned_end_layer": 31,
                        },
                    ]
                },
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "e2e-private-room",
                            "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                            "nodes_online": 2,
                            "total_shared_ram_gb": 10.0,
                            "is_private": True,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        if request.method == "POST" and request.url.path == "/nodes/leave":
            return httpx.Response(status_code=200, json={"success": True})
        return httpx.Response(status_code=404, json={"error": "Unhandled route"})

    runtime.tracker_client = _make_mock_tracker(_handler)
    fake_model = tmp_path / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    fake_model.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fake_model),
    )

    start_res = client.post(
        "/start-node",
        json={
            "node_id": "e2e-node-alpha",
            "model_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "network_mode": "internet",
            "room_id": "e2e-private-room",
            "room_key": secret,
            "ram_gb": 4,
            "shared_ram_gb": 4,
            "max_ram_mb": 4096,
            "max_vram_mb": 0,
            "p2p_listen_port": 49345,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
            "total_layers": 32,
        },
    )
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["state"] == "RUNNING"
    assert start_data["network_status"] == NetworkStatus.ONLINE.value
    assert [p["peer_id"] for p in start_data["peers"]] == ["e2e-node-beta"]

    stop_res = client.post("/stop-node")
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["state"] == "STOPPED"
    assert stop_data["allocated_ram_mb"] == 0
    assert stop_data["allocated_vram_mb"] == 0
    assert stop_data["session_token"] is None
    assert runtime.room_key_hash is None

    leave_calls = [entry for entry in seen_requests if entry[1] == "/nodes/leave"]
    assert len(leave_calls) == 1
    _method, _path, _params, leave_headers, leave_body = leave_calls[0]
    assert leave_headers.get("x-room-key-hash") == expected_hash
    assert leave_body == {
        "node_id": "e2e-node-alpha",
        "room_id": "e2e-private-room",
        "model": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "room_key_hash": expected_hash,
    }


@pytest.mark.asyncio
async def test_fetch_peers_retries_transient_503_and_raises_unreachable_on_timeout() -> None:
    """Transient 503 recovers on retry; repeated ConnectTimeout raises `TrackerUnreachableError`."""
    attempts = 0

    async def _flaky_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(status_code=503, json={"error": "Busy"})
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "node_id": "peer-recovered",
                        "ip": "198.51.100.77",
                        "port": 49160,
                        "assigned_start_layer": 0,
                        "assigned_end_layer": 31,
                    }
                ]
            },
        )

    flaky_tracker = _make_mock_tracker(_flaky_handler, max_retries=3)
    peers = await flaky_tracker.fetch_peers(model_id="Qwen/Qwen2.5-7B-Instruct")
    assert attempts == 2
    assert [p.peer_id for p in peers] == ["peer-recovered"]

    async def _timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Tracker unreachable")

    timeout_tracker = _make_mock_tracker(_timeout_handler, max_retries=2)
    with pytest.raises(TrackerUnreachableError):
        await timeout_tracker.fetch_peers(model_id="Qwen/Qwen2.5-7B-Instruct")
