"""End-to-end tests for production Iroh QUIC P2P transport, NAT auth, and fallback routing."""

from __future__ import annotations

import asyncio
import os
import socket
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import httpx
import pytest

from main import LocalNodeRuntime
from p2p.quic_transport import IROH_AVAILABLE, IrohP2PTransport
from p2p.serializer import serialize_activation_tensor
from p2p.transport import PeerUnreachableError
from shard_engine.schemas import (
    NetworkMode,
    NodeState,
    PeerNodeInfo,
    PeerRole,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import make_mock_runner

pytestmark = pytest.mark.skipif(not IROH_AVAILABLE, reason="iroh package is not installed")


def _reserve_unused_local_port() -> int:
    """Return an unbound loopback TCP port so direct TCP always fails with PeerUnreachableError."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_node(
    node: LocalNodeRuntime,
    *,
    node_id: str,
    shard_start: int,
    shard_end: int,
    total_layers: int,
    p2p_port: int,
    peers: list[PeerNodeInfo],
    public_ip: str,
) -> None:
    node.state = NodeState.RUNNING
    node.node_id = node_id
    node.model_id = "test/quic-sharded-16l"
    node.room_id = "quic-room"
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
    await node.quic_transport.stop()
    await node.relay_client.stop()
    await node.cleanup_proxy_state()
    await node.stop_p2p_listener()
    if node.model_runner is not None:
        node.model_runner.release()
        node.model_runner = None
    node.state = NodeState.STOPPED


@pytest.mark.asyncio
async def test_iroh_quic_transport_roundtrip_large_payload_and_nack() -> None:
    """Two IrohP2PTransport instances exchange >1 MiB frames over pooled QUIC streams and propagate NACKs."""
    received: list[tuple[bytes, str]] = []
    reject_next = False

    async def _on_frame(frame: bytes, remote_id: str) -> None:
        if reject_next:
            raise ValueError("Invalid tensor magic header")
        received.append((frame, remote_id))

    server = IrohP2PTransport()
    client = IrohP2PTransport()
    try:
        server_id, server_ticket = await server.start(
            _on_frame, relay_mode="disabled", wait_online_seconds=0.0
        )
        client_id, _client_ticket = await client.start(
            lambda _b, _s: asyncio.sleep(0),
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )
        assert server.is_running and client.is_running
        assert server_id and server_ticket.startswith("endpoint")

        # Small control/tensor frame
        rtt1 = await client.send_frame(server_ticket, b"SHRD\x01hello-quic", ack_timeout=10.0)
        assert rtt1 >= 0.0

        # Large >1 MiB activation payload on pooled QUIC connection
        large_payload = b"SHRD" + (b"\xAB" * (1_500_000))
        rtt2 = await client.send_frame(server_ticket, large_payload, ack_timeout=15.0)
        assert rtt2 >= 0.0
        assert len(received) == 2
        assert received[0] == (b"SHRD\x01hello-quic", client_id)
        assert received[1] == (large_payload, client_id)
        assert client.get_direct_peers_count() >= 1
        assert client.get_peer_path_kind(server_id) == "direct_quic"

        # Handler exception turns into NACK with reason
        reject_next = True
        with pytest.raises(RuntimeError, match="NACK.*Invalid tensor magic header"):
            await client.send_frame(server_ticket, b"SHRD\x01bad", ack_timeout=10.0)
    finally:
        await client.stop()
        await server.stop()


@pytest.mark.asyncio
async def test_distributed_pipeline_and_proxy_end_to_end_over_quic_only() -> None:
    """Head and Tail with closed TCP ports and no WS relay complete pipeline + proxy strictly over QUIC."""
    closed_head_port = _reserve_unused_local_port()
    closed_tail_port = _reserve_unused_local_port()

    head_node = LocalNodeRuntime()
    tail_node = LocalNodeRuntime()
    try:
        head_ep_id, head_ticket = await head_node.quic_transport.start(
            head_node._handle_quic_frame,
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )
        tail_ep_id, tail_ticket = await tail_node.quic_transport.start(
            tail_node._handle_quic_frame,
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )

        # Deliberately give Tail a mismatched origin/public IP vs CGNAT to verify identity-based QUIC auth
        head_peer_info = PeerNodeInfo(
            peer_id="head-quic-node",
            ip_address="198.51.100.10",
            port=closed_head_port,
            shard_layer_start=0,
            shard_layer_end=7,
            role=PeerRole.PRIMARY,
            room_id="quic-room",
            p2p_endpoint_id=head_ep_id,
            p2p_ticket=head_ticket,
        )
        tail_peer_info = PeerNodeInfo(
            peer_id="tail-quic-node",
            ip_address="203.0.113.20",
            port=closed_tail_port,
            shard_layer_start=8,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
            room_id="quic-room",
            p2p_endpoint_id=tail_ep_id,
            p2p_ticket=tail_ticket,
        )

        _configure_node(
            head_node,
            node_id="head-quic-node",
            shard_start=0,
            shard_end=7,
            total_layers=16,
            p2p_port=closed_head_port,
            peers=[tail_peer_info],
            public_ip="198.51.100.99",  # intentionally differs from CGNAT / dual-stack IP
        )
        _configure_node(
            tail_node,
            node_id="tail-quic-node",
            shard_start=8,
            shard_end=15,
            total_layers=16,
            p2p_port=closed_tail_port,
            peers=[head_peer_info],
            public_ip="203.0.113.99",
        )

        assert head_node.relay_client.is_connected is False
        assert tail_node.relay_client.is_connected is False

        # 1. Pipeline inference initiated on Head Node
        head_tokens: list[tuple[str, int, int, str, bool]] = []
        async for item in head_node.stream_p2p_pipeline_tokens(
            prompt="Hello Iroh QUIC",
            max_tokens=3,
        ):
            head_tokens.append(item)

        assert len(head_tokens) == 3
        assert head_tokens[-1][4] is True
        assert head_node._get_cached_peer_route("tail-quic-node") == "quic"
        assert tail_node._get_cached_peer_route("head-quic-node") == "quic"

        # 2. Non-Head proxy inference initiated on Tail Node
        proxy_tokens: list[tuple[str, int, int, str, bool]] = []
        async for item in tail_node.stream_p2p_pipeline_tokens(
            prompt="Proxied over Iroh QUIC",
            max_tokens=3,
        ):
            proxy_tokens.append(item)

        assert len(proxy_tokens) == 3
        assert proxy_tokens[-1][4] is True

        status_snap = head_node.to_status_response()
        assert status_snap.p2p_quic_enabled is True
        assert status_snap.p2p_endpoint_id == head_ep_id
        assert status_snap.p2p_direct_peers_count >= 1
    finally:
        await _cleanup_node(head_node)
        await _cleanup_node(tail_node)


@pytest.mark.asyncio
async def test_quic_unauthorized_endpoint_rejected_and_unreachable_falls_back() -> None:
    """Unknown EndpointIds are rejected with NACK; unreachable QUIC peers fall back to TCP."""
    target_node = LocalNodeRuntime()
    spoof_transport = IrohP2PTransport()
    sender_node = LocalNodeRuntime()
    try:
        target_ep_id, target_ticket = await target_node.quic_transport.start(
            target_node._handle_quic_frame,
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )
        await spoof_transport.start(
            lambda _b, _s: asyncio.sleep(0),
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )
        await sender_node.quic_transport.start(
            sender_node._handle_quic_frame,
            relay_mode="disabled",
            wait_online_seconds=0.0,
        )

        known_peer = PeerNodeInfo(
            peer_id="legit-head",
            ip_address="198.51.100.10",
            port=49152,
            shard_layer_start=0,
            shard_layer_end=7,
            p2p_endpoint_id="different-endpoint-id-not-spoof",
        )
        _configure_node(
            target_node,
            node_id="target-node",
            shard_start=8,
            shard_end=15,
            total_layers=16,
            p2p_port=49153,
            peers=[known_peer],
            public_ip="203.0.113.20",
        )

        runner = target_node.model_runner
        assert runner is not None
        raw_tensor, shape = runner.embed_tokens_to_raw_tensor([1, 2, 3])
        spoof_frame = serialize_activation_tensor(
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
            await spoof_transport.send_frame(target_ticket, spoof_frame, ack_timeout=5.0)

        # Fallback: bad ticket triggers PeerUnreachableError in QUIC -> falls back to direct TCP listener
        tcp_received: list[bytes] = []

        async def _tcp_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            from p2p.transport import ACK_BYTE, FRAME_LEN_STRUCT

            len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
            (flen,) = FRAME_LEN_STRUCT.unpack(len_bytes)
            tcp_received.append(await reader.readexactly(flen))
            writer.write(ACK_BYTE)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        tcp_srv = await asyncio.start_server(_tcp_handler, "127.0.0.1", 0)
        assert tcp_srv.sockets
        tcp_port = int(tcp_srv.sockets[0].getsockname()[1])
        try:
            sender_node.network_mode = NetworkMode.INTERNET
            fallback_peer = PeerNodeInfo(
                peer_id="tcp-fallback-peer",
                ip_address="127.0.0.1",
                port=tcp_port,
                shard_layer_start=8,
                shard_layer_end=15,
                p2p_endpoint_id=target_ep_id,
                p2p_ticket="not-a-valid-iroh-ticket",
            )
            rtt = await sender_node.send_frame_to_peer(
                fallback_peer, b"SHRD\x01\x00tcp-fallback", timeout_seconds=2.0
            )
            assert rtt >= 0.0
            assert tcp_received == [b"SHRD\x01\x00tcp-fallback"]
            assert sender_node._get_cached_peer_route("tcp-fallback-peer") == "direct"
        finally:
            tcp_srv.close()
            await tcp_srv.wait_closed()
    finally:
        await spoof_transport.stop()
        await _cleanup_node(sender_node)
        await _cleanup_node(target_node)


@pytest.mark.asyncio
async def test_tracker_client_sends_and_parses_quic_endpoint_fields() -> None:
    """SignalingTrackerClient transmits p2p_endpoint_id/p2p_ticket and parses them on PeerNodeInfo."""
    captured_payload: dict[str, Any] = {}
    ep_id_self = "00d152a1e827075ac4b1e0ad441622a9c27f26606bdecb56fec441e7c5e4e6cc"
    ticket_self = "endpointaaancuvb5atqowwewhqk2raweku4e7zgmbv55s2w73cedz6f4ttmyaqbaafaa"
    ep_id_peer = "11d152a1e827075ac4b1e0ad441622a9c27f26606bdecb56fec441e7c5e4e6dd"
    ticket_peer = "endpointbbbncuvb5atqowwewhqk2raweku4e7zgmbv55s2w73cedz6f4ttmyaqbaafbb"

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            status_code=200,
            json={
                "assigned_start_layer": 0,
                "assigned_end_layer": 15,
                "role": "primary",
                "peers": [
                    {
                        "node_id": "peer-quic-2",
                        "ip": "203.0.113.50",
                        "port": 49152,
                        "assigned_start_layer": 16,
                        "assigned_end_layer": 31,
                        "role": "primary",
                        "p2p_endpoint_id": ep_id_peer,
                        "p2p_ticket": ticket_peer,
                    }
                ],
            },
        )

    client = SignalingTrackerClient(
        base_url="https://network.shardai.online",
        transport=httpx.MockTransport(_handler),
    )
    req = StartNodeRequest(
        node_id="node-quic-1",
        model_id="test/model",
        network_mode=NetworkMode.INTERNET,
        room_id="public",
        ram_gb=4,
    )
    res = await client.register_node_with_assignment(
        req,
        p2p_endpoint_id=ep_id_self,
        p2p_ticket=ticket_self,
    )
    assert captured_payload["p2p_endpoint_id"] == ep_id_self
    assert captured_payload["p2p_ticket"] == ticket_self
    assert len(res.peers) == 1
    assert res.peers[0].p2p_endpoint_id == ep_id_peer
    assert res.peers[0].p2p_ticket == ticket_peer
