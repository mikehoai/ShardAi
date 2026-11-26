"""Regression tests for Issue #65: Public room peer connection over NAT, IPv6, and WebSocket relay."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import pytest
import websockets.asyncio.server as ws_server
from websockets.datastructures import Headers
from websockets.http11 import Response as WsHttpResponse

from main import LocalNodeRuntime, _resolve_advertise_ip, runtime
from p2p.relay_client import RelayClient
from p2p.relay_envelope import RelayPacketKind, decode_envelope, encode_envelope
from p2p.serializer import (
    deserialize_activation_tensor,
    deserialize_inference_request,
    serialize_activation_tensor,
    serialize_inference_request,
)
from p2p.transport import (
    DEFAULT_P2P_PORT,
    PeerUnreachableError,
    _parse_host_port,
    send_activations,
)
from shard_engine.schemas import (
    NetworkMode,
    NetworkStatus,
    NodeState,
    PeerNodeInfo,
    PeerRole,
)
from tests._mock_runner import make_mock_runner


class _ControllableRelayServer:
    """In-process WebSocket relay server supporting 403 auth refresh and delayed peer arrival."""

    def __init__(
        self,
        token_to_node_id: dict[str, str],
        *,
        rejected_tokens: set[str] | None = None,
    ) -> None:
        self.token_to_node_id = dict(token_to_node_id)
        self.rejected_tokens: set[str] = set(rejected_tokens or ())
        self.connections: dict[str, ws_server.ServerConnection] = {}
        self.forwarded_pairs: list[tuple[str, str]] = []
        self.data_forwarded_pairs: list[tuple[str, str]] = []
        self.no_route_msg_ids: list[bytes] = []
        self.auth_reject_count: int = 0
        self.server: ws_server.Server | None = None
        self.port: int = 0

    async def __aenter__(self) -> _ControllableRelayServer:
        async def _process_request(
            _conn: ws_server.ServerConnection, request: Any
        ) -> WsHttpResponse | None:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                self.auth_reject_count += 1
                return WsHttpResponse(401, "Unauthorized", Headers(), b"missing bearer")
            token = auth[len("Bearer ") :].strip()
            if token in self.rejected_tokens:
                self.auth_reject_count += 1
                return WsHttpResponse(403, "Forbidden", Headers(), b"expired token")
            if token not in self.token_to_node_id:
                self.auth_reject_count += 1
                return WsHttpResponse(401, "Unauthorized", Headers(), b"invalid token")
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
                        self.no_route_msg_ids.append(env.msg_id)
                        no_route = encode_envelope(
                            kind=RelayPacketKind.NO_ROUTE,
                            msg_id=env.msg_id,
                            src=env.dst,
                            dst=authed_node_id,
                        )
                        await ws.send(no_route)
                        continue

                    rewritten = encode_envelope(
                        kind=env.kind,
                        msg_id=env.msg_id,
                        src=authed_node_id,
                        dst=env.dst,
                        payload=env.payload,
                        chunk_index=env.chunk_index,
                        chunk_count=env.chunk_count,
                    )
                    self.forwarded_pairs.append((authed_node_id, env.dst))
                    if env.kind == RelayPacketKind.DATA:
                        self.data_forwarded_pairs.append((authed_node_id, env.dst))
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


def _configure_relay_node(
    node: LocalNodeRuntime,
    *,
    node_id: str,
    shard_start: int,
    shard_end: int,
    total_layers: int,
    p2p_port: int,
    peers: list[PeerNodeInfo],
    public_ip: str | None,
) -> None:
    node.state = NodeState.RUNNING
    node.network_status = NetworkStatus.ONLINE
    node.node_id = node_id
    node.model_id = "test/public-room-sharded-18l"
    node.room_id = "public-room-65"
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


def test_parse_host_port_ipv6_ipv4_hostname_and_malformed() -> None:
    """`_parse_host_port` parses IPv4, bare/bracketed IPv6, and hostnames, rejecting malformed strings."""
    assert _parse_host_port("192.0.2.10") == ("192.0.2.10", DEFAULT_P2P_PORT)
    assert _parse_host_port("192.0.2.10:49153") == ("192.0.2.10", 49153)
    assert _parse_host_port("2402:800::1", 49200) == ("2402:800::1", 49200)
    assert _parse_host_port("fe80::1%en0", 49201) == ("fe80::1%en0", 49201)
    assert _parse_host_port("[2402:800::1]:49155") == ("2402:800::1", 49155)
    assert _parse_host_port("[::1]", 49202) == ("::1", 49202)
    assert _parse_host_port("[fe80::1%en0]:49156") == ("fe80::1%en0", 49156)
    assert _parse_host_port("peer-node.local") == ("peer-node.local", DEFAULT_P2P_PORT)
    assert _parse_host_port("peer-node.local:49160") == ("peer-node.local", 49160)

    for bad_endpoint in (
        "",
        "   ",
        "bad:multi:colon",
        "2001:db8::zzzz",
        "[2402:800::1",
        "[]",
        "[192.0.2.1]:49152",
        "[::1]:0",
        "[::1]:65536",
        "[::1]extra",
        "192.0.2.1:0",
        "192.0.2.1:70000",
        "192.0.2.1:notaport",
        ":49152",
    ):
        with pytest.raises(ValueError):
            _parse_host_port(bad_endpoint)


@pytest.mark.asyncio
async def test_send_activations_wraps_malformed_endpoint_as_peer_unreachable() -> None:
    """`send_activations` wraps malformed endpoint ValueError into `PeerUnreachableError` for relay fallback."""
    for bad_ip in ("", "bad:multi:colon", "[::1]:99999", "192.0.2.1:notaport"):
        with pytest.raises(PeerUnreachableError):
            await send_activations(bad_ip, b"SHRD\x01\x00test", timeout_seconds=0.5)


def test_ipv6_resolve_advertise_ip_and_binary_serializer_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv6 public_ip and IPv6 peers never crash `_resolve_advertise_ip` or binary frame serializers."""
    monkeypatch.delenv("SHARDAI_ADVERTISE_IP", raising=False)
    orig_public_ip = runtime.public_ip
    try:
        runtime.public_ip = "2402:800:61c0::1"
        resolved_v6 = _resolve_advertise_ip("2402:800:61c0::2", 49152)
        assert resolved_v6 == "127.0.0.1"

        # IPv4 public_ip still applies for global IPv6 target
        assert _resolve_advertise_ip(
            "2402:800:61c0::2", 49152, public_ip="203.0.113.44"
        ) == "203.0.113.44"

        # Invalid port / host must gracefully fall back to 127.0.0.1
        assert _resolve_advertise_ip("not-a-host-!@#", 999999, public_ip="not-an-ip") == "127.0.0.1"
    finally:
        runtime.public_ip = orig_public_ip

    req_id = str(uuid.uuid4())
    act_bytes = serialize_activation_tensor(
        tensor_data=b"\x00\x00\x80\x3f" * 4,
        shape=(1, 4),
        request_id=req_id,
        origin_ip="2402:800:61c0::1",
        origin_port=49152,
        current_layer=5,
        total_layers=12,
        step_index=0,
        max_tokens=4,
    )
    act_frame = deserialize_activation_tensor(act_bytes)
    assert act_frame.origin_ip == "127.0.0.1"

    inf_bytes = serialize_inference_request(
        request_id=req_id,
        requester_ip="fe80::1%en0",
        requester_port=49153,
        prompt="hello ipv6",
        max_tokens=8,
    )
    inf_frame = deserialize_inference_request(inf_bytes)
    assert inf_frame.requester_ip == "127.0.0.1"

    inf_v4_bytes = serialize_inference_request(
        request_id=req_id,
        requester_ip="198.51.100.77",
        requester_port=49153,
        prompt="hello ipv4",
        max_tokens=8,
    )
    assert deserialize_inference_request(inf_v4_bytes).requester_ip == "198.51.100.77"

    with pytest.raises(OSError):
        serialize_activation_tensor(
            tensor_data=b"\x00\x00\x80\x3f" * 4,
            shape=(1, 4),
            request_id=req_id,
            origin_ip="invalid-non-ip",
            origin_port=49152,
            current_layer=0,
            total_layers=4,
            step_index=0,
        )
    with pytest.raises(OSError):
        serialize_inference_request(
            request_id=req_id,
            requester_ip="invalid-non-ip",
            requester_port=49152,
            prompt="bad",
        )


@pytest.mark.asyncio
async def test_two_node_public_room_ipv6_nat_peers_over_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2-node Public room with IPv6/NAT peers works both Head->Tail and non-Head proxy->Head->Tail->Head->non-Head."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    head_node = LocalNodeRuntime()
    tail_node = LocalNodeRuntime()
    try:
        async with _ControllableRelayServer(
            {"tok-head": "head-v6", "tok-tail": "tail-v6"}
        ) as relay:
            relay_url = f"ws://127.0.0.1:{relay.port}/relay/public-room-65"
            head_peer = PeerNodeInfo(
                peer_id="head-v6",
                ip_address="2402:800:61c0::10",
                port=49152,
                shard_layer_start=0,
                shard_layer_end=8,
                role=PeerRole.PRIMARY,
                room_id="public-room-65",
            )
            tail_peer = PeerNodeInfo(
                peer_id="tail-v6",
                ip_address="2402:800:61c0::20",
                port=49153,
                shard_layer_start=9,
                shard_layer_end=17,
                role=PeerRole.PRIMARY,
                room_id="public-room-65",
            )

            _configure_relay_node(
                head_node,
                node_id="head-v6",
                shard_start=0,
                shard_end=8,
                total_layers=18,
                p2p_port=49152,
                peers=[tail_peer],
                public_ip="2402:800:61c0::10",
            )
            _configure_relay_node(
                tail_node,
                node_id="tail-v6",
                shard_start=9,
                shard_end=17,
                total_layers=18,
                p2p_port=49153,
                peers=[head_peer],
                public_ip="2402:800:61c0::20",
            )

            await head_node.relay_client.start(
                url=relay_url,
                token="tok-head",
                node_id="head-v6",
                on_frame=head_node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )
            await tail_node.relay_client.start(
                url=relay_url,
                token="tok-tail",
                node_id="tail-v6",
                on_frame=tail_node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )
            assert await head_node.relay_client.wait_until_connected(timeout=3.0)
            assert await tail_node.relay_client.wait_until_connected(timeout=3.0)

            head_tokens = [
                item
                async for item in head_node.stream_p2p_pipeline_tokens(
                    prompt="Public IPv6 Head->Tail", max_tokens=3
                )
            ]
            assert len(head_tokens) == 3
            assert head_tokens[-1][4] is True

            proxy_tokens = [
                item
                async for item in tail_node.stream_p2p_pipeline_tokens(
                    prompt="Public IPv6 non-Head proxy", max_tokens=3
                )
            ]
            assert len(proxy_tokens) == 3
            assert proxy_tokens[-1][4] is True
    finally:
        await _cleanup_node(head_node)
        await _cleanup_node(tail_node)


@pytest.mark.asyncio
async def test_three_node_head_middle_tail_over_relay_returns_tokens_to_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3-node Head->Middle->Tail over relay must return DecodedTokenFrame from Tail directly to Head."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    head_node = LocalNodeRuntime()
    middle_node = LocalNodeRuntime()
    tail_node = LocalNodeRuntime()
    try:
        async with _ControllableRelayServer(
            {
                "tok-head": "head-3",
                "tok-mid": "middle-3",
                "tok-tail": "tail-3",
            }
        ) as relay:
            relay_url = f"ws://127.0.0.1:{relay.port}/relay/public-room-65"
            head_peer = PeerNodeInfo(
                peer_id="head-3",
                ip_address="198.51.100.1",
                port=49152,
                shard_layer_start=0,
                shard_layer_end=5,
                role=PeerRole.PRIMARY,
                room_id="public-room-65",
            )
            mid_peer = PeerNodeInfo(
                peer_id="middle-3",
                ip_address="198.51.100.2",
                port=49153,
                shard_layer_start=6,
                shard_layer_end=11,
                role=PeerRole.PRIMARY,
                room_id="public-room-65",
            )
            tail_peer = PeerNodeInfo(
                peer_id="tail-3",
                ip_address="198.51.100.3",
                port=49154,
                shard_layer_start=12,
                shard_layer_end=17,
                role=PeerRole.PRIMARY,
                room_id="public-room-65",
            )

            # Force NAT mismatch: peers' advertised public_ip is IPv6 while tracker IP is unreachable IPv4
            _configure_relay_node(
                head_node,
                node_id="head-3",
                shard_start=0,
                shard_end=5,
                total_layers=18,
                p2p_port=49152,
                peers=[mid_peer, tail_peer],
                public_ip="2402:800::1",
            )
            _configure_relay_node(
                middle_node,
                node_id="middle-3",
                shard_start=6,
                shard_end=11,
                total_layers=18,
                p2p_port=49153,
                peers=[head_peer, tail_peer],
                public_ip="2402:800::2",
            )
            _configure_relay_node(
                tail_node,
                node_id="tail-3",
                shard_start=12,
                shard_end=17,
                total_layers=18,
                p2p_port=49154,
                peers=[head_peer, mid_peer],
                public_ip="2402:800::3",
            )

            for node, tok, nid in (
                (head_node, "tok-head", "head-3"),
                (middle_node, "tok-mid", "middle-3"),
                (tail_node, "tok-tail", "tail-3"),
            ):
                await node.relay_client.start(
                    url=relay_url,
                    token=tok,
                    node_id=nid,
                    on_frame=node._handle_relay_frame,
                    allowed_host="127.0.0.1",
                )
                assert await node.relay_client.wait_until_connected(timeout=3.0)

            tokens = [
                item
                async for item in head_node.stream_p2p_pipeline_tokens(
                    prompt="Three node pipeline", max_tokens=2
                )
            ]
            assert len(tokens) == 2
            assert tokens[-1][4] is True
            # Verify Tail routed DecodedTokenFrame DATA back to Head, never back to Middle
            assert ("tail-3", "head-3") in relay.data_forwarded_pairs
            assert ("tail-3", "middle-3") not in relay.data_forwarded_pairs
    finally:
        await _cleanup_node(head_node)
        await _cleanup_node(middle_node)
        await _cleanup_node(tail_node)


@pytest.mark.asyncio
async def test_relay_client_wait_until_connected_no_route_retry_and_403_auth_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `wait_until_connected`, `NO_ROUTE` retry with fresh `msg_id`, and `on_auth_failure` on HTTP 403."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    async with _ControllableRelayServer(
        {"tok-a": "node-a", "tok-b": "node-b"},
        rejected_tokens={"expired-tok-a"},
    ) as relay:
        relay_url = f"ws://127.0.0.1:{relay.port}/relay/public-room-65"
        client_a = RelayClient()
        client_b = RelayClient()
        auth_failures: int = 0
        received_on_b: list[bytes] = []

        async def _refresh_auth() -> None:
            nonlocal auth_failures
            auth_failures += 1
            relay.rejected_tokens.discard("expired-tok-a")
            client_a.update_credentials(relay_url, "tok-a")

        async def _on_b_frame(data: bytes, _src: str) -> None:
            received_on_b.append(data)

        try:
            # Start with 403-rejected token; on_auth_failure refreshes token and connects immediately
            await client_a.start(
                url=relay_url,
                token="expired-tok-a",
                node_id="node-a",
                on_frame=lambda _b, _s: asyncio.sleep(0),
                allowed_host="127.0.0.1",
                on_auth_failure=_refresh_auth,
            )
            connected = await client_a.wait_until_connected(timeout=3.0)
            assert connected is True
            assert auth_failures >= 1
            assert relay.auth_reject_count >= 1

            # Send to node-b while node-b is not yet connected -> NO_ROUTE on attempt 1,
            # then node-b connects during retry backoff and attempt 2 succeeds with a fresh msg_id.
            async def _connect_b_shortly() -> None:
                await asyncio.sleep(0.15)
                await client_b.start(
                    url=relay_url,
                    token="tok-b",
                    node_id="node-b",
                    on_frame=_on_b_frame,
                    allowed_host="127.0.0.1",
                )
                await client_b.wait_until_connected(timeout=3.0)

            connector_task = asyncio.create_task(_connect_b_shortly())
            t0 = time.monotonic()
            rtt = await client_a.send_frame(
                "node-b", b"SHRD\x01\x00retry-ok", ack_timeout=3.0
            )
            await connector_task
            assert rtt >= 0.0
            assert (time.monotonic() - t0) >= 0.30
            assert len(relay.no_route_msg_ids) == 1
            assert received_on_b == [b"SHRD\x01\x00retry-ok"]
        finally:
            await client_a.stop()
            await client_b.stop()


@pytest.mark.asyncio
async def test_send_frame_to_peer_lazy_heartbeat_and_wait_until_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`send_frame_to_peer` triggers lazy heartbeat when relay not started and awaits connection."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    async with _ControllableRelayServer(
        {"tok-src": "src-node", "tok-dst": "dst-node"}
    ) as relay:
        relay_url = f"ws://127.0.0.1:{relay.port}/relay/public-room-65"
        src_node = LocalNodeRuntime()
        dst_client = RelayClient()
        received: list[bytes] = []
        hb_calls = 0

        async def _on_dst_frame(frame: bytes, _src: str) -> None:
            received.append(frame)

        async def _fake_heartbeat(force: bool = False) -> tuple[bool, list[PeerNodeInfo] | None]:
            nonlocal hb_calls
            assert force is True
            hb_calls += 1
            await src_node.relay_client.start(
                url=relay_url,
                token="tok-src",
                node_id="src-node",
                on_frame=src_node._handle_relay_frame,
                allowed_host="127.0.0.1",
            )
            return True, None

        try:
            await dst_client.start(
                url=relay_url,
                token="tok-dst",
                node_id="dst-node",
                on_frame=_on_dst_frame,
                allowed_host="127.0.0.1",
            )
            await dst_client.wait_until_connected(timeout=3.0)

            peer = PeerNodeInfo(
                peer_id="dst-node",
                ip_address="bad:multi:colon",
                port=49152,
                shard_layer_start=8,
                shard_layer_end=15,
            )
            _configure_relay_node(
                src_node,
                node_id="src-node",
                shard_start=0,
                shard_end=7,
                total_layers=16,
                p2p_port=49151,
                peers=[peer],
                public_ip="2402:800::1",
            )
            monkeypatch.setattr(src_node, "_send_tracker_heartbeat", _fake_heartbeat)

            rtt = await src_node.send_frame_to_peer(peer, b"SHRD\x01\x00lazy-hb", timeout_seconds=3.0)
            assert rtt >= 0.0
            assert hb_calls == 1
            assert src_node._get_cached_peer_route("dst-node") == "relay"
            assert received == [b"SHRD\x01\x00lazy-hb"]
        finally:
            await dst_client.stop()
            await _cleanup_node(src_node)
