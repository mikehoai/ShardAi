"""Integration & regression tests for Internet NAT traversal, public IP advertising, and relay fallback."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import httpx
import pytest
import websockets.asyncio.server as ws_server
from websockets.http11 import Response as WsHttpResponse

import main as main_module
from main import LocalNodeRuntime, _resolve_advertise_ip, runtime
from p2p.relay_client import RelayClient, validate_relay_url
from p2p.relay_envelope import RelayPacketKind, decode_envelope, encode_envelope
from p2p.serializer import InferenceErrorFrame, serialize_activation_tensor
from p2p.transport import (
    ACK_BYTE,
    FRAME_LEN_STRUCT,
    NACK_BYTE,
    PeerUnreachableError,
    current_relay_source,
)
from shard_engine.schemas import (
    NetworkMode,
    NodeState,
    PeerNodeInfo,
    PeerRole,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import make_mock_runner


def _reserve_unused_local_port() -> int:
    """Return a closed loopback TCP port so direct TCP connection fails with PeerUnreachableError."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _InProcessRelayServer:
    """Minimal WebSocket relay server matching the Signaling Tracker relay contract."""

    def __init__(self, token_to_node_id: dict[str, str]) -> None:
        self.token_to_node_id = dict(token_to_node_id)
        self.connections: dict[str, ws_server.ServerConnection] = {}
        self.forwarded_count: int = 0
        self.server: ws_server.Server | None = None
        self.port: int = 0

    async def __aenter__(self) -> _InProcessRelayServer:
        async def _process_request(
            _conn: ws_server.ServerConnection, request: Any
        ) -> WsHttpResponse | None:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return WsHttpResponse(401, "Unauthorized", ws_server.Headers(), b"missing bearer")
            token = auth[len("Bearer ") :].strip()
            if token not in self.token_to_node_id:
                return WsHttpResponse(401, "Unauthorized", ws_server.Headers(), b"invalid token")
            return None

        async def _handler(ws: ws_server.ServerConnection) -> None:
            auth = ws.request.headers.get("Authorization", "") if ws.request else ""
            token = auth[len("Bearer ") :].strip()
            authed_node_id = self.token_to_node_id[token]
            self.connections[authed_node_id] = ws
            try:
                async for raw_msg in ws:
                    if not isinstance(raw_msg, (bytes, bytearray)):
                        continue
                    env = decode_envelope(bytes(raw_msg))
                    if env.kind == RelayPacketKind.PING:
                        pong = encode_envelope(
                            kind=RelayPacketKind.PONG,
                            msg_id=env.msg_id,
                            src="relay-server",
                            dst=authed_node_id,
                        )
                        await ws.send(pong)
                        continue

                    target_ws = self.connections.get(env.dst)
                    if target_ws is None:
                        no_route = encode_envelope(
                            kind=RelayPacketKind.NO_ROUTE,
                            msg_id=env.msg_id,
                            src=env.dst,
                            dst=authed_node_id,
                        )
                        await ws.send(no_route)
                        continue

                    # Rewrite src to authenticated sender node_id
                    rewritten = encode_envelope(
                        kind=env.kind,
                        msg_id=env.msg_id,
                        src=authed_node_id,
                        dst=env.dst,
                        payload=env.payload,
                        chunk_index=env.chunk_index,
                        chunk_count=env.chunk_count,
                    )
                    self.forwarded_count += 1
                    await target_ws.send(rewritten)
            finally:
                if self.connections.get(authed_node_id) is ws:
                    self.connections.pop(authed_node_id, None)

        self.server = await ws_server.serve(
            _handler,
            host="127.0.0.1",
            port=0,
            process_request=_process_request,
            max_size=1_100_000,
        )
        assert self.server.sockets
        self.port = int(self.server.sockets[0].getsockname()[1])
        return self

    async def __aexit__(self, *_args: Any) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()


def _configure_node(
    node: LocalNodeRuntime,
    *,
    node_id: str,
    shard_start: int,
    shard_end: int,
    total_layers: int,
    p2p_port: int,
    peers: list[PeerNodeInfo],
    public_ip: str | None = None,
) -> None:
    node.state = NodeState.RUNNING
    node.node_id = node_id
    node.model_id = "test/nat-sharded-16l"
    node.room_id = "internet-room"
    node.network_mode = NetworkMode.INTERNET
    node.p2p_listen_port = p2p_port
    node.public_ip = public_ip
    node.shard_layer_start = shard_start
    node.shard_layer_end = shard_end
    node.total_layers = total_layers
    node.n_layers = shard_end - shard_start + 1
    node.is_swarm_ready = True
    node.coverage_percent = 100.0
    node.peers = list(peers)
    node.model_runner = make_mock_runner(
        model_id=node.model_id,
        shard_layer_start=shard_start,
        shard_layer_end=shard_end,
        total_layers=total_layers,
        n_layers=node.n_layers,
    )


async def _cleanup_node(node: LocalNodeRuntime) -> None:
    await node.relay_client.stop()
    await node.cleanup_proxy_state()
    await node.stop_p2p_listener()
    if node.model_runner is not None:
        node.model_runner.release()
        node.model_runner = None
    node.state = NodeState.STOPPED


@pytest.mark.asyncio
async def test_relay_url_validation_and_ping_pong_and_no_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relay URL validator enforces wss:// unless local test mode enabled; server enforces auth, NO_ROUTE, and PONG."""
    monkeypatch.delenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", raising=False)
    with pytest.raises(ValueError):
        validate_relay_url("ws://127.0.0.1:9000/relay/room", allowed_host="127.0.0.1")

    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")
    with pytest.raises(ValueError):
        validate_relay_url("ws://example.com/relay/room")
    with pytest.raises(ValueError):
        validate_relay_url("wss://evil.example/relay/room", allowed_host="network.shardai.online")
    assert validate_relay_url("ws://127.0.0.1:9000/relay/room", allowed_host="127.0.0.1")

    async with _InProcessRelayServer({"tok-a": "node-a"}) as relay:
        relay_url = f"ws://127.0.0.1:{relay.port}/relay/internet-room"

        # Unauthorized token must never connect
        bad_client = RelayClient()
        await bad_client.start(
            url=relay_url,
            token="wrong-token",
            node_id="node-a",
            on_frame=lambda _b, _s: asyncio.sleep(0),
            allowed_host="127.0.0.1",
        )
        assert bad_client.is_connected is False
        await bad_client.stop()

        # Authorized client connects, receives NO_ROUTE for offline dst, and receives PONG on PING
        client = RelayClient()
        await client.start(
            url=relay_url,
            token="tok-a",
            node_id="node-a",
            on_frame=lambda _b, _s: asyncio.sleep(0),
            allowed_host="127.0.0.1",
        )
        assert client.is_connected is True

        with pytest.raises(RuntimeError, match="NO_ROUTE"):
            await client.send_frame("offline-node", b"SHRD\x01\x00hello", ack_timeout=2.0)

        assert client._ws is not None
        ping_id = uuid.uuid4().bytes
        await client._ws.send(
            encode_envelope(
                kind=RelayPacketKind.PING,
                msg_id=ping_id,
                src="spoofed-src",
                dst="relay",
            )
        )
        # Ensure connection stays healthy after PONG
        await asyncio.sleep(0.05)
        assert client.is_connected is True
        await client.stop()


@pytest.mark.asyncio
async def test_distributed_pipeline_and_non_head_proxy_end_to_end_via_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Head and Tail with unreachable TCP endpoints run both pipeline inference and #63 proxy via relay."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")
    closed_head_port = _reserve_unused_local_port()
    closed_tail_port = _reserve_unused_local_port()

    head_node = LocalNodeRuntime()
    tail_node = LocalNodeRuntime()
    try:
        async with _InProcessRelayServer(
            {"tok-head": "head-node", "tok-tail": "tail-node"}
        ) as relay:
            relay_url = f"ws://127.0.0.1:{relay.port}/relay/internet-room"
            head_peer_info = PeerNodeInfo(
                peer_id="head-node",
                ip_address="127.0.0.1",
                port=closed_head_port,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
                room_id="internet-room",
            )
            tail_peer_info = PeerNodeInfo(
                peer_id="tail-node",
                ip_address="127.0.0.1",
                port=closed_tail_port,
                shard_layer_start=8,
                shard_layer_end=15,
                role=PeerRole.PRIMARY,
                room_id="internet-room",
            )

            _configure_node(
                head_node,
                node_id="head-node",
                shard_start=0,
                shard_end=7,
                total_layers=16,
                p2p_port=closed_head_port,
                peers=[tail_peer_info],
                public_ip="198.51.100.10",
            )
            _configure_node(
                tail_node,
                node_id="tail-node",
                shard_start=8,
                shard_end=15,
                total_layers=16,
                p2p_port=closed_tail_port,
                peers=[head_peer_info],
                public_ip="198.51.100.20",
            )

            await head_node.relay_client.start(
                url=relay_url,
                token="tok-head",
                node_id="head-node",
                on_frame=head_node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )
            await tail_node.relay_client.start(
                url=relay_url,
                token="tok-tail",
                node_id="tail-node",
                on_frame=tail_node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )

            # 1. Distributed pipeline inference initiated on Head Node
            head_tokens: list[tuple[str, int, int, str, bool]] = []
            async for item in head_node.stream_p2p_pipeline_tokens(
                prompt="Hello NAT traversal",
                max_tokens=3,
            ):
                head_tokens.append(item)

            assert len(head_tokens) == 3
            assert head_tokens[-1][4] is True
            assert head_node._get_cached_peer_route("tail-node") == "relay"
            assert tail_node._get_cached_peer_route("head-node") == "relay"

            # 2. Non-Head chat proxy (#63 flow) initiated on Tail Node
            proxy_tokens: list[tuple[str, int, int, str, bool]] = []
            async for item in tail_node.stream_p2p_pipeline_tokens(
                prompt="Proxied over relay",
                max_tokens=3,
            ):
                proxy_tokens.append(item)

            assert len(proxy_tokens) == 3
            assert proxy_tokens[-1][4] is True
            assert relay.forwarded_count > 0
    finally:
        await _cleanup_node(head_node)
        await _cleanup_node(tail_node)


@pytest.mark.asyncio
async def test_nack_from_direct_peer_is_not_retried_via_relay_and_unreachable_requires_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct peer NACK must raise without relay retry; PeerUnreachableError triggers fallback only if relay is connected."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    async def _nack_server_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
            (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
            await reader.readexactly(frame_len)
            writer.write(NACK_BYTE)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    tcp_server = await asyncio.start_server(_nack_server_handler, "127.0.0.1", 0)
    assert tcp_server.sockets
    nack_port = int(tcp_server.sockets[0].getsockname()[1])

    node = LocalNodeRuntime()
    relay_received: list[bytes] = []
    try:
        async with _InProcessRelayServer({"tok-a": "node-a", "tok-b": "node-b"}) as relay:
            relay_url = f"ws://127.0.0.1:{relay.port}/relay/room"
            sink = RelayClient()

            async def _on_sink_frame(frame_bytes: bytes, _src: str) -> None:
                relay_received.append(frame_bytes)

            await sink.start(
                url=relay_url,
                token="tok-b",
                node_id="node-b",
                on_frame=_on_sink_frame,
                allowed_host="127.0.0.1",
            )
            await node.relay_client.start(
                url=relay_url,
                token="tok-a",
                node_id="node-a",
                on_frame=node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )

            nack_peer = PeerNodeInfo(
                peer_id="node-b",
                ip_address="127.0.0.1",
                port=nack_port,
                shard_layer_start=8,
                shard_layer_end=15,
            )
            with pytest.raises(RuntimeError, match="NACK"):
                await node.send_frame_to_peer(nack_peer, b"SHRD\x01\x00frame", timeout_seconds=2.0)
            assert relay_received == []

            # Closed TCP port -> PeerUnreachableError -> falls back to relay when connected
            closed_port = _reserve_unused_local_port()
            unreachable_peer = PeerNodeInfo(
                peer_id="node-b",
                ip_address="127.0.0.1",
                port=closed_port,
                shard_layer_start=8,
                shard_layer_end=15,
            )
            rtt_ms = await node.send_frame_to_peer(
                unreachable_peer, b"SHRD\x01\x00via-relay", timeout_seconds=2.0
            )
            assert rtt_ms >= 0.0
            assert relay_received == [b"SHRD\x01\x00via-relay"]

            # Without relay connected, PeerUnreachableError propagates
            await node.relay_client.stop()
            node._peer_routes.clear()
            with pytest.raises(PeerUnreachableError):
                await node.send_frame_to_peer(
                    unreachable_peer, b"SHRD\x01\x00no-relay", timeout_seconds=1.0
                )
            await sink.stop()
    finally:
        tcp_server.close()
        await tcp_server.wait_closed()
        await _cleanup_node(node)


def test_resolve_advertise_ip_global_vs_private_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_advertise_ip` returns public_ip for global targets and LAN/loopback for private targets."""
    monkeypatch.delenv("SHARDAI_ADVERTISE_IP", raising=False)
    orig_public_ip = runtime.public_ip
    try:
        runtime.public_ip = "203.0.113.55"
        assert _resolve_advertise_ip("8.8.8.8", 49152) == "203.0.113.55"
        assert _resolve_advertise_ip("93.184.216.34", 49152) == "203.0.113.55"
        assert _resolve_advertise_ip("127.0.0.1", 49152) == "127.0.0.1"
        private_res = _resolve_advertise_ip("192.168.1.50", 49152)
        assert private_res != "203.0.113.55"

        monkeypatch.setenv("SHARDAI_ADVERTISE_IP", "198.51.100.99")
        assert _resolve_advertise_ip("192.168.1.50", 49152) == "198.51.100.99"
    finally:
        runtime.public_ip = orig_public_ip


@pytest.mark.asyncio
async def test_tracker_client_parses_observed_ip_and_validates_relay_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SignalingTrackerClient parses observed_ip & matching relay creds and drops mismatched relay hosts."""
    monkeypatch.delenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", raising=False)

    def _make_transport(payload: dict[str, Any]) -> httpx.MockTransport:
        def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, json=payload)

        return httpx.MockTransport(_handler)

    req = StartNodeRequest(
        node_id="node-tracker-test",
        model_id="test/model",
        network_mode=NetworkMode.INTERNET,
        room_id="public",
        ram_gb=4,
    )

    # Matching host + valid observed_ip
    client_ok = SignalingTrackerClient(
        base_url="https://network.shardai.online",
        transport=_make_transport(
            {
                "assigned_start_layer": 0,
                "assigned_end_layer": 15,
                "role": "primary",
                "peers": [],
                "observed_ip": "203.0.113.88",
                "relay": {
                    "url": "wss://network.shardai.online/relay/public",
                    "token": "secret-relay-token",
                    "expires_at": 1800000000,
                },
            }
        ),
    )
    res_ok = await client_ok.register_node_with_assignment(req)
    assert res_ok.observed_ip == "203.0.113.88"
    assert res_ok.relay_url == "wss://network.shardai.online/relay/public"
    assert res_ok.relay_token == "secret-relay-token"

    # Mismatched relay host + invalid observed_ip must be dropped silently
    client_mismatch = SignalingTrackerClient(
        base_url="https://network.shardai.online",
        transport=_make_transport(
            {
                "assigned_start_layer": 0,
                "assigned_end_layer": 15,
                "role": "primary",
                "peers": [],
                "observed_ip": "not-an-ip",
                "relay": {
                    "url": "wss://attacker.example/relay/public",
                    "token": "secret-relay-token",
                    "expires_at": 1800000000,
                },
            }
        ),
    )
    res_mismatch = await client_mismatch.register_node_with_assignment(req)
    assert res_mismatch.observed_ip is None
    assert res_mismatch.relay_url is None
    assert res_mismatch.relay_token is None


@pytest.mark.asyncio
async def test_relay_frame_with_unknown_source_is_rejected_and_nacked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames delivered via relay whose authenticated source is not in runtime.peers must be rejected with NACK."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    node = LocalNodeRuntime()
    spoof_sender = RelayClient()
    try:
        async with _InProcessRelayServer(
            {"tok-target": "target-node", "tok-spoof": "unknown-peer"}
        ) as relay:
            relay_url = f"ws://127.0.0.1:{relay.port}/relay/room"
            known_peer = PeerNodeInfo(
                peer_id="legit-head",
                ip_address="198.51.100.10",
                port=49152,
                shard_layer_start=0,
                shard_layer_end=7,
            )
            _configure_node(
                node,
                node_id="target-node",
                shard_start=8,
                shard_end=15,
                total_layers=16,
                p2p_port=49153,
                peers=[known_peer],
                public_ip="198.51.100.20",
            )
            await node.relay_client.start(
                url=relay_url,
                token="tok-target",
                node_id="target-node",
                on_frame=node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )
            await spoof_sender.start(
                url=relay_url,
                token="tok-spoof",
                node_id="unknown-peer",
                on_frame=lambda _b, _s: asyncio.sleep(0),
                allowed_host="127.0.0.1",
            )

            # Even if origin_ip matches known_peer, current_relay_source ("unknown-peer") is not in node.peers
            runner = node.model_runner
            assert runner is not None
            raw_tensor, shape = runner.embed_tokens_to_raw_tensor([1, 2, 3])
            frame_bytes = serialize_activation_tensor(
                tensor_data=raw_tensor,
                shape=shape,
                request_id=str(uuid.uuid4()),
                origin_ip="198.51.100.10",
                origin_port=49152,
                current_layer=7,
                total_layers=16,
                step_index=0,
                max_tokens=2,
            )

            with pytest.raises(RuntimeError, match="NACK"):
                await spoof_sender.send_frame("target-node", frame_bytes, ack_timeout=3.0)
    finally:
        await spoof_sender.stop()
        await _cleanup_node(node)
