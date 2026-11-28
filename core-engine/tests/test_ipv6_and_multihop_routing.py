"""Regression tests for Issue #82: IPv6 binary frame endpoints and >=3-node multi-hop origin routing."""

from __future__ import annotations

from array import array
import asyncio
import os
import socket
import struct
from typing import Any
import uuid
import zlib

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import pytest

import main as main_module
from main import LocalNodeRuntime, _resolve_advertise_ip, runtime
from p2p.serializer import (
    ACTIVATION_HEADER_STRUCT,
    ACTIVATION_ROUTING_EXT_FLAG,
    LEGACY_ACTIVATION_HEADER_STRUCT,
    MAGIC_HEADER,
    MAX_ENDPOINT_IP_BYTES,
    MAX_PEER_ID_BYTES,
    ActivationFrame,
    CompressionCodec,
    DecodedTokenFrame,
    InferenceRequestFrame,
    PacketType,
    deserialize_activation_tensor,
    deserialize_decoded_token,
    deserialize_inference_request,
    serialize_activation_tensor,
    serialize_inference_request,
)
from p2p.transport import (
    ACK_BYTE,
    FRAME_LEN_STRUCT,
    NACK_BYTE,
    current_quic_peer_id,
    current_relay_source,
    current_tcp_peer_ip,
    receive_activations,
    send_activations,
)
from shard_engine.schemas import NetworkMode, NodeState, PeerNodeInfo, PeerRole
from tests._mock_runner import make_mock_runner


def _configure_pipeline_node(
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
    """Configure a `LocalNodeRuntime` instance with a deterministic mock shard runner."""
    node.state = NodeState.RUNNING
    node.node_id = node_id
    node.model_id = "test/issue-82-sharded-24l"
    node.room_id = "room-issue-82"
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
    """Cleanly stop P2P listeners and release mock runners."""
    await node.cleanup_proxy_state()
    await node.stop_p2p_listener()
    if node.model_runner is not None:
        node.model_runner.release()
        node.model_runner = None
    node.state = NodeState.STOPPED


# ==============================================================================
# 1. Binary Serializer IPv6 + origin_peer_id + Legacy Backward Compatibility
# ==============================================================================


def test_activation_and_inference_request_ipv6_origin_peer_id_and_legacy_compat() -> None:
    """IPv6 endpoints and peer IDs round-trip losslessly while legacy v1/v2 frames decode cleanly."""
    raw_tensor = array("f", [0.125, -0.25, 0.5, 1.0] * 8).tobytes()
    req_id = str(uuid.uuid4())

    # 1. ActivationFrame with canonical + non-canonical + bracketed IPv6 and origin_peer_id
    for input_ip, expected_ip in (
        ("2001:db8::abcd", "2001:db8::abcd"),
        ("2001:db8:0:0::abcd", "2001:db8::abcd"),
        ("[2001:db8::abcd]", "2001:db8::abcd"),
        ("::1", "::1"),
    ):
        packed = serialize_activation_tensor(
            tensor_data=raw_tensor,
            shape=(2, 16),
            request_id=req_id,
            origin_ip=input_ip,
            origin_port=49152,
            current_layer=7,
            total_layers=24,
            step_index=1,
            max_tokens=16,
            temperature=0.55,
            origin_peer_id="head-node-2",
        )
        decoded = deserialize_activation_tensor(packed)
        assert isinstance(decoded, ActivationFrame)
        assert decoded.origin_ip == expected_ip
        assert decoded.origin_ip != "127.0.0.1"
        assert decoded.origin_peer_id == "head-node-2"
        assert decoded.origin_port == 49152
        assert decoded.temperature == pytest.approx(0.55)
        assert decoded.raw_tensor_bytes == raw_tensor

    # 2. IPv4 ActivationFrame with origin_peer_id preserves both IPv4 and origin_peer_id
    packed_v4_peer = serialize_activation_tensor(
        tensor_data=raw_tensor,
        shape=(2, 16),
        request_id=req_id,
        origin_ip="192.168.1.10",
        origin_port=49152,
        current_layer=7,
        total_layers=24,
        step_index=0,
        origin_peer_id="head-b",
    )
    decoded_v4_peer = deserialize_activation_tensor(packed_v4_peer)
    assert decoded_v4_peer.origin_ip == "192.168.1.10"
    assert decoded_v4_peer.origin_peer_id == "head-b"

    # 3. Legacy v1 (52-byte header) and v2 (60-byte header) frames without extension decode with origin_peer_id=None
    v1_crc = zlib.crc32(raw_tensor) & 0xFFFFFFFF
    v1_header = LEGACY_ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(CompressionCodec.NONE),
        uuid.UUID(req_id).bytes,
        socket.inet_aton("198.51.100.11"),
        49200,
        7,
        24,
        0,
        8,
        2,
        16,
        len(raw_tensor),
        v1_crc,
    )
    v1_decoded = deserialize_activation_tensor(v1_header + raw_tensor)
    assert v1_decoded.origin_ip == "198.51.100.11"
    assert v1_decoded.origin_peer_id is None
    assert v1_decoded.raw_tensor_bytes == raw_tensor

    v2_header = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(CompressionCodec.NONE),
        uuid.UUID(req_id).bytes,
        socket.inet_aton("198.51.100.12"),
        49201,
        7,
        24,
        2,
        8,
        2,
        16,
        len(raw_tensor),
        v1_crc,
        0.8,
    )
    v2_decoded = deserialize_activation_tensor(v2_header + raw_tensor)
    assert v2_decoded.origin_ip == "198.51.100.12"
    assert v2_decoded.origin_peer_id is None
    assert v2_decoded.temperature == pytest.approx(0.8)

    # 4. InferenceRequestFrame with IPv6 requester_ip and requester_peer_id
    inf_packed = serialize_inference_request(
        request_id=req_id,
        requester_ip="[2001:db8:0:0::abcd]",
        requester_port=49300,
        prompt="Hello over IPv6",
        messages=[{"role": "user", "content": "Hello over IPv6"}],
        max_tokens=32,
        temperature=0.6,
        requester_peer_id="tail-requester-2",
    )
    inf_decoded = deserialize_inference_request(inf_packed)
    assert isinstance(inf_decoded, InferenceRequestFrame)
    assert inf_decoded.requester_ip == "2001:db8::abcd"
    assert inf_decoded.requester_peer_id == "tail-requester-2"
    assert inf_decoded.requester_port == 49300


def test_serializer_enforces_strict_routing_extension_bounds() -> None:
    """Oversized peer IDs (>128 bytes) and malformed wire routing extensions are rejected."""
    raw_tensor = array("f", [1.0] * 8).tobytes()
    req_id = str(uuid.uuid4())

    oversized_peer_id = "p" * (MAX_PEER_ID_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        serialize_activation_tensor(
            tensor_data=raw_tensor,
            shape=(1, 8),
            request_id=req_id,
            origin_ip="2001:db8::1",
            origin_port=49152,
            current_layer=0,
            total_layers=4,
            step_index=0,
            origin_peer_id=oversized_peer_id,
        )
    with pytest.raises(ValueError, match="exceeds"):
        serialize_inference_request(
            request_id=req_id,
            requester_ip="2001:db8::1",
            requester_port=49152,
            prompt="hi",
            requester_peer_id=oversized_peer_id,
        )

    # Crafted wire extension claiming origin_ip length > MAX_ENDPOINT_IP_BYTES (64)
    bad_ip_bytes = b"a" * (MAX_ENDPOINT_IP_BYTES + 5)
    bad_ext = struct.pack("!BB", len(bad_ip_bytes), 0) + bad_ip_bytes + raw_tensor
    bad_crc = zlib.crc32(bad_ext) & 0xFFFFFFFF
    bad_header = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(CompressionCodec.NONE) | ACTIVATION_ROUTING_EXT_FLAG,
        uuid.UUID(req_id).bytes,
        b"\x7f\x00\x00\x01",
        49152,
        0,
        4,
        0,
        8,
        1,
        8,
        len(raw_tensor),
        bad_crc,
        0.7,
    )
    with pytest.raises(ValueError, match="exceeds"):
        deserialize_activation_tensor(bad_header + bad_ext)


# ==============================================================================
# 2. _resolve_advertise_ip IPv6 AF_INET6 Probing & Non-Collapse
# ==============================================================================


def test_resolve_advertise_ip_uses_af_inet6_and_never_collapses_to_ipv4_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_resolve_advertise_ip` probes IPv6 targets with `AF_INET6` and never collapses to `127.0.0.1`."""
    monkeypatch.delenv("SHARDAI_ADVERTISE_IP", raising=False)
    orig_public_ip = runtime.public_ip
    observed_families: list[int] = []
    real_socket_cls = socket.socket

    class _SpySocket:
        def __init__(self, family: int = socket.AF_INET, sock_type: int = socket.SOCK_DGRAM) -> None:
            observed_families.append(family)
            self._inner = real_socket_cls(family, sock_type)

        def connect(self, address: tuple[Any, ...]) -> None:
            if observed_families[-1] == socket.AF_INET6 and address[0] == "2001:db8::1":
                return
            self._inner.connect(address)

        def getsockname(self) -> Any:
            if observed_families[-1] == socket.AF_INET6:
                try:
                    name = self._inner.getsockname()
                    if str(name[0]) not in ("::", "0.0.0.0"):
                        return name
                except OSError:
                    pass
                return ("2001:db8::99%en0", 51234, 0, 0)
            return self._inner.getsockname()

        def close(self) -> None:
            self._inner.close()

    try:
        runtime.public_ip = None
        monkeypatch.setattr(main_module.socket, "socket", _SpySocket)

        resolved_doc_v6 = _resolve_advertise_ip("2001:db8::1", 49152)
        assert observed_families[-1] == socket.AF_INET6
        assert resolved_doc_v6 == "2001:db8::99"
        assert resolved_doc_v6 != "127.0.0.1"

        resolved_bracket_v6 = _resolve_advertise_ip("[2001:db8::1]", 49152)
        assert observed_families[-1] == socket.AF_INET6
        assert resolved_bracket_v6 == "2001:db8::99"

        monkeypatch.setattr(main_module.socket, "socket", real_socket_cls)
        resolved_loopback_v6 = _resolve_advertise_ip("::1", 49152)
        assert resolved_loopback_v6 == "::1"
        assert resolved_loopback_v6 != "127.0.0.1"

        # Even if AF_INET6 connect raises OSError (no IPv6 default route), fallback is "::1", never "127.0.0.1"
        class _UnreachableV6Socket:
            def __init__(self, family: int, _sock_type: int) -> None:
                observed_families.append(family)

            def connect(self, _addr: Any) -> None:
                raise OSError("No route to IPv6 host")

            def close(self) -> None:
                pass

        monkeypatch.setattr(main_module.socket, "socket", _UnreachableV6Socket)
        fallback_v6 = _resolve_advertise_ip("2001:db8::1", 49152)
        assert observed_families[-1] == socket.AF_INET6
        assert fallback_v6 == "::1"
    finally:
        runtime.public_ip = orig_public_ip


# ==============================================================================
# 3. Direct IPv6 Authorization & External TCP Spoofing Rejection
# ==============================================================================


@pytest.mark.asyncio
async def test_ipv6_peer_authorization_canonicalization_and_spoof_rejection() -> None:
    """Expanded/compressed IPv6 and IPv4-mapped IPv6 match peers, while spoofed origins/senders fail."""
    node = LocalNodeRuntime()
    node.state = NodeState.RUNNING
    node.shard_layer_start = 16
    node.shard_layer_end = 31
    node.total_layers = 32
    node.peers = [
        PeerNodeInfo(
            peer_id="head-v6",
            ip_address="2001:db8::1",
            port=49152,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="mid-v4",
            ip_address="192.0.2.1",
            port=49153,
            shard_layer_start=8,
            shard_layer_end=15,
            role=PeerRole.BACKUP,
        ),
    ]

    # Non-canonical IPv6 "2001:db8:0:0::1" matches "2001:db8::1"
    assert node._is_authorized_activation_origin(
        "2001:db8:0:0::1", 49152, origin_peer_id="head-v6"
    )
    assert node._is_authorized_activation_origin("2001:db8:0:0::1", 49152)
    assert node._is_authorized_peer_endpoint(
        "2001:db8:0:0::1", 49152, requester_peer_id="head-v6"
    )

    # IPv4-mapped IPv6 "::ffff:192.0.2.1" matches "192.0.2.1"
    assert node._is_authorized_peer_endpoint(
        "::ffff:192.0.2.1", 49153, requester_peer_id="mid-v4"
    )
    assert node._peer_for_endpoint("::ffff:192.0.2.1", 49153) is not None

    # Spoofed non-peer or non-Head origin_peer_id is rejected
    assert not node._is_authorized_activation_origin(
        "2001:db8::1", 49152, origin_peer_id="evil-external-node"
    )
    assert not node._is_authorized_activation_origin(
        "192.0.2.1", 49153, origin_peer_id="mid-v4"
    )

    # Unauthenticated external TCP sender from unknown IP cannot spoof origin_peer_id="head-v6"
    tok = current_tcp_peer_ip.set("2001:db8::dead:beef")
    try:
        assert not node._is_authorized_activation_origin(
            "2001:db8::1", 49152, origin_peer_id="head-v6"
        )
        assert not node._is_authorized_peer_endpoint(
            "2001:db8::1", 49152, requester_peer_id="head-v6"
        )
    finally:
        current_tcp_peer_ip.reset(tok)


@pytest.mark.asyncio
async def test_live_ipv6_loopback_two_node_tcp_pipeline() -> None:
    """Live 2-node TCP pipeline bound on `::1` streams activations and returns tokens over IPv6."""
    # Skip cleanly only if the host kernel has IPv6 loopback disabled
    try:
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.bind(("::1", 0))
        probe.close()
    except OSError:
        pytest.skip("IPv6 loopback ::1 is not available on this host")

    head = LocalNodeRuntime()
    tail = LocalNodeRuntime()
    try:
        head_srv = await receive_activations(
            host="::1",
            port=0,
            on_activation_frame=head.handle_incoming_activation,
            on_token_frame=head.handle_incoming_decoded_token,
        )
        tail_srv = await receive_activations(
            host="::1",
            port=0,
            on_activation_frame=tail.handle_incoming_activation,
            on_token_frame=tail.handle_incoming_decoded_token,
        )
        head.p2p_server = head_srv
        tail.p2p_server = tail_srv
        head_port = int(head_srv.sockets[0].getsockname()[1])
        tail_port = int(tail_srv.sockets[0].getsockname()[1])

        _configure_pipeline_node(
            head,
            node_id="head-v6-live",
            shard_start=0,
            shard_end=11,
            total_layers=24,
            p2p_port=head_port,
            peers=[
                PeerNodeInfo(
                    peer_id="tail-v6-live",
                    ip_address="::1",
                    port=tail_port,
                    shard_layer_start=12,
                    shard_layer_end=23,
                    role=PeerRole.PRIMARY,
                )
            ],
        )
        _configure_pipeline_node(
            tail,
            node_id="tail-v6-live",
            shard_start=12,
            shard_end=23,
            total_layers=24,
            p2p_port=tail_port,
            peers=[
                PeerNodeInfo(
                    peer_id="head-v6-live",
                    ip_address="::1",
                    port=head_port,
                    shard_layer_start=0,
                    shard_layer_end=11,
                    role=PeerRole.PRIMARY,
                )
            ],
        )

        tokens = [
            item
            async for item in head.stream_p2p_pipeline_tokens(
                prompt="IPv6 direct TCP pipeline",
                max_tokens=2,
            )
        ]
        assert len(tokens) == 2
        assert tokens[-1][4] is True
    finally:
        await _cleanup_node(head)
        await _cleanup_node(tail)


# ==============================================================================
# 4. 3-Node Multi-Hop Tail -> Head-B Deterministic Token Return Routing
# ==============================================================================


@pytest.mark.asyncio
async def test_three_node_secondary_head_lan_middle_remote_tail_routes_tokens_to_origin_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In `Head-B -> Middle (LAN) -> Tail (remote)`, `Tail` sends `DecodedTokenFrame` to `Head-B`
    (`203.0.113.20`), NEVER `Head-A` (`head_peers[0]`) and NEVER `Middle` (`current_quic_peer_id` / `current_relay_source`).
    """
    monkeypatch.delenv("SHARDAI_ADVERTISE_IP", raising=False)

    head_a = PeerNodeInfo(
        peer_id="head-a",
        ip_address="203.0.113.10",
        port=49151,
        shard_layer_start=0,
        shard_layer_end=7,
        role=PeerRole.PRIMARY,
    )
    head_b_public = PeerNodeInfo(
        peer_id="head-b",
        ip_address="203.0.113.20",
        port=49152,
        shard_layer_start=0,
        shard_layer_end=7,
        role=PeerRole.BACKUP,
    )
    head_b_lan = PeerNodeInfo(
        peer_id="head-b",
        ip_address="192.168.1.10",
        port=49152,
        shard_layer_start=0,
        shard_layer_end=7,
        role=PeerRole.BACKUP,
    )
    middle_lan = PeerNodeInfo(
        peer_id="middle-lan",
        ip_address="192.168.1.20",
        port=49153,
        shard_layer_start=8,
        shard_layer_end=15,
        role=PeerRole.PRIMARY,
    )
    middle_public = PeerNodeInfo(
        peer_id="middle-lan",
        ip_address="198.51.100.30",
        port=49153,
        shard_layer_start=8,
        shard_layer_end=15,
        role=PeerRole.PRIMARY,
    )
    tail_public = PeerNodeInfo(
        peer_id="tail-remote",
        ip_address="198.51.100.40",
        port=49154,
        shard_layer_start=16,
        shard_layer_end=23,
        role=PeerRole.PRIMARY,
    )

    head_b_node = LocalNodeRuntime()
    middle_node = LocalNodeRuntime()
    tail_node = LocalNodeRuntime()

    _configure_pipeline_node(
        head_b_node,
        node_id="head-b",
        shard_start=0,
        shard_end=7,
        total_layers=24,
        p2p_port=49152,
        peers=[middle_lan, tail_public],
        public_ip="203.0.113.20",
    )
    _configure_pipeline_node(
        middle_node,
        node_id="middle-lan",
        shard_start=8,
        shard_end=15,
        total_layers=24,
        p2p_port=49153,
        peers=[head_a, head_b_lan, tail_public],
        public_ip="198.51.100.30",
    )
    # Tail lists Head-A FIRST (`head_peers[0]`) and Head-B SECOND at public IP 203.0.113.20
    _configure_pipeline_node(
        tail_node,
        node_id="tail-remote",
        shard_start=16,
        shard_end=23,
        total_layers=24,
        p2p_port=49154,
        peers=[head_a, head_b_public, middle_public],
        public_ip="198.51.100.40",
    )
    assert tail_node.find_head_node_peers()[0].peer_id == "head-a"

    routed_token_targets: list[tuple[str, str, int, DecodedTokenFrame]] = []
    forwarded_middle_frames: list[ActivationFrame] = []

    async def _route_head_b_send(
        peer: PeerNodeInfo,
        frame: bytes,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        assert peer.peer_id == "middle-lan"
        act = deserialize_activation_tensor(frame)
        # Force LAN advertise IP ("192.168.1.10") as seen between Head-B and same-LAN Middle
        repacked = serialize_activation_tensor(
            tensor_data=act.raw_tensor_bytes,
            shape=act.shape,
            request_id=act.request_id,
            origin_ip="192.168.1.10",
            origin_port=act.origin_port,
            current_layer=act.current_layer,
            total_layers=act.total_layers,
            step_index=act.step_index,
            max_tokens=act.max_tokens,
            temperature=act.temperature,
            origin_peer_id=act.origin_peer_id,
        )
        tok = current_tcp_peer_ip.set("192.168.1.10")
        try:
            await middle_node.handle_incoming_activation(
                deserialize_activation_tensor(repacked)
            )
        finally:
            current_tcp_peer_ip.reset(tok)
        return 1.0

    async def _route_middle_send(
        peer: PeerNodeInfo,
        frame: bytes,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        assert peer.peer_id == "tail-remote"
        act = deserialize_activation_tensor(frame)
        forwarded_middle_frames.append(act)
        # Deliver to Tail with BOTH current_quic_peer_id and current_relay_source bound to Middle
        # to prove Tail NEVER misroutes DecodedTokenFrame back to Middle!
        q_tok = current_quic_peer_id.set("middle-lan")
        r_tok = current_relay_source.set("middle-lan")
        try:
            await tail_node.handle_incoming_activation(act)
        finally:
            current_relay_source.reset(r_tok)
            current_quic_peer_id.reset(q_tok)
        return 1.0

    async def _route_tail_send(
        peer: PeerNodeInfo,
        frame: bytes,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        tok_frame = deserialize_decoded_token(frame)
        routed_token_targets.append((peer.peer_id, peer.ip_address, peer.port, tok_frame))
        if peer.peer_id == "head-b":
            await head_b_node.handle_incoming_decoded_token(tok_frame)
        return 1.0

    monkeypatch.setattr(head_b_node, "send_frame_to_peer", _route_head_b_send)
    monkeypatch.setattr(middle_node, "send_frame_to_peer", _route_middle_send)
    monkeypatch.setattr(tail_node, "send_frame_to_peer", _route_tail_send)

    try:
        tokens = [
            item
            async for item in head_b_node.stream_p2p_pipeline_tokens(
                prompt="Secondary Head 3-node multi-hop test",
                max_tokens=3,
            )
        ]
        assert len(tokens) == 3
        assert tokens[-1][4] is True

        # Middle preserved Head-B's LAN origin_ip AND origin_peer_id="head-b"
        assert len(forwarded_middle_frames) == 3
        for fwd in forwarded_middle_frames:
            assert fwd.origin_ip == "192.168.1.10"
            assert fwd.origin_peer_id == "head-b"

        # Tail routed every DecodedTokenFrame strictly to Head-B ("head-b", "203.0.113.20", 49152),
        # NEVER to Head-A ("head-a") and NEVER to Middle ("middle-lan")!
        assert len(routed_token_targets) == 3
        for target_id, target_ip, target_port, _tok in routed_token_targets:
            assert target_id == "head-b"
            assert target_ip == "203.0.113.20"
            assert target_port == 49152

        # Also verify `_send_to_endpoint(..., target_peer_id=None)` on Tail NEVER falls back to Middle
        direct_tcp_calls: list[tuple[str, int]] = []

        async def _spy_send_activations(
            peer_ip: str,
            byte_stream: bytes,
            peer_port: int = 49152,
            timeout_seconds: float = 15.0,
            ack_timeout_seconds: float | None = None,
        ) -> float:
            direct_tcp_calls.append((peer_ip, peer_port))
            return 1.0

        monkeypatch.setattr(main_module, "send_activations", _spy_send_activations)
        q_tok = current_quic_peer_id.set("middle-lan")
        try:
            await tail_node._send_to_endpoint(
                ip="192.168.1.10",
                port=49152,
                frame=b"SHRD\x02\x00test",
                target_peer_id=None,
            )
        finally:
            current_quic_peer_id.reset(q_tok)
        assert len(routed_token_targets) == 3  # send_frame_to_peer was NOT called with Middle!
        assert direct_tcp_calls == [("192.168.1.10", 49152)]
    finally:
        await _cleanup_node(head_b_node)
        await _cleanup_node(middle_node)
        await _cleanup_node(tail_node)
