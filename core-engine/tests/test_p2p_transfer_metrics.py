"""Unit and regression tests for P2P transfer RTT normalization and metrics (Issue #79)."""

from __future__ import annotations

import math
from typing import Any

import pytest

import main as main_module
from main import LocalNodeRuntime, _normalize_rtt_ms
from shard_engine.schemas import NetworkMode, PeerNodeInfo, PeerRole


def _make_peer(peer_id: str = "peer-1", ticket: str | None = None) -> PeerNodeInfo:
    return PeerNodeInfo(
        peer_id=peer_id,
        ip_address="127.0.0.1",
        port=49550,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
        p2p_ticket=ticket,
    )


@pytest.mark.parametrize(
    "raw_rtt",
    [
        None,
        True,
        False,
        "bad",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_record_p2p_transfer_metrics_ignores_invalid_and_non_finite_rtt(
    raw_rtt: Any,
) -> None:
    """Invalid, boolean, None, and non-finite RTTs must be ignored without altering state."""
    node = LocalNodeRuntime()
    assert _normalize_rtt_ms(raw_rtt) is None

    # Initial state remains None when invalid RTTs arrive
    node._record_p2p_transfer_metrics(1024, raw_rtt)
    assert node.last_p2p_rtt_ms is None
    assert node.p2p_throughput_mbps is None

    # Existing valid EWMA metrics remain untouched when subsequent RTTs are invalid
    node.last_p2p_rtt_ms = 12.5
    node.p2p_throughput_mbps = 4.25
    node._record_p2p_transfer_metrics(1024, raw_rtt)
    assert node.last_p2p_rtt_ms == 12.5
    assert node.p2p_throughput_mbps == 4.25


def test_record_p2p_transfer_metrics_clamps_negative_finite_rtt() -> None:
    """Negative finite RTT values are clamped to 0.1 ms for safe throughput calculation."""
    node = LocalNodeRuntime()
    node._record_p2p_transfer_metrics(1250, -5.0)
    # 1250 bytes * 8 bits = 10_000 bits over 0.0001 s = 100 Mbps
    assert node.last_p2p_rtt_ms == 0.1
    assert node.p2p_throughput_mbps == 100.0


def test_record_p2p_transfer_metrics_valid_positive_rtt_and_ewma_updates() -> None:
    """Valid positive RTTs initialize metrics and apply 0.7/0.3 EWMA smoothing."""
    node = LocalNodeRuntime()

    # First sample: 1000 bytes in 10.0 ms -> 0.8 Mbps
    node._record_p2p_transfer_metrics(1000, 10.0)
    assert node.last_p2p_rtt_ms == 10.0
    assert node.p2p_throughput_mbps == 0.8

    # Second sample: 2000 bytes in 20.0 ms -> 0.8 Mbps, RTT EWMA = 0.7*10 + 0.3*20 = 13.0
    node._record_p2p_transfer_metrics(2000, 20.0)
    assert node.last_p2p_rtt_ms == 13.0
    assert node.p2p_throughput_mbps == 0.8

    # Third sample: 4000 bytes in 10.0 ms -> 3.2 Mbps
    # RTT EWMA = round(0.7*13.0 + 0.3*10.0, 1) = 12.1
    # Throughput EWMA = round(0.7*0.8 + 0.3*3.2, 2) = 1.52
    node._record_p2p_transfer_metrics(4000, 10.0)
    assert node.last_p2p_rtt_ms == 12.1
    assert node.p2p_throughput_mbps == 1.52


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_return", [None, float("nan"), float("inf"), True, "bad"])
async def test_send_frame_to_peer_handles_non_numeric_or_non_finite_send_activations(
    monkeypatch: pytest.MonkeyPatch,
    mock_return: Any,
) -> None:
    """`send_frame_to_peer` falls back to 0.0 and skips metric corruption on bad RTT returns."""
    node = LocalNodeRuntime()
    peer = _make_peer()

    async def _fake_send_activations(**_kwargs: Any) -> Any:
        return mock_return

    monkeypatch.setattr(main_module, "send_activations", _fake_send_activations)

    rtt = await node.send_frame_to_peer(peer, b"SHRD\x01\x00test", timeout_seconds=1.0)
    assert isinstance(rtt, float)
    assert math.isfinite(rtt)
    assert rtt == 0.0
    assert node.last_p2p_rtt_ms is None
    assert node.p2p_throughput_mbps is None


@pytest.mark.asyncio
async def test_send_frame_to_peer_returns_finite_float_for_quic_and_relay_and_valid_tcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QUIC, WS relay, and direct TCP paths all normalize RTT and record valid metrics."""
    node = LocalNodeRuntime()
    node.network_mode = NetworkMode.INTERNET

    # 1. Direct TCP valid float updates metrics and returns exact float
    tcp_peer = _make_peer("tcp-peer")

    async def _valid_send_activations(**_kwargs: Any) -> float:
        return 12.5

    monkeypatch.setattr(main_module, "send_activations", _valid_send_activations)
    tcp_rtt = await node.send_frame_to_peer(tcp_peer, b"x" * 1250, timeout_seconds=1.0)
    assert tcp_rtt == 12.5
    assert node.last_p2p_rtt_ms == 12.5
    assert node.p2p_throughput_mbps == 0.8

    # 2. Cached QUIC transport returning None falls back to 0.0 without corrupting metrics
    quic_peer = _make_peer("quic-peer", ticket="iroh-ticket-123")
    node._set_cached_peer_route(quic_peer.peer_id, "quic")
    monkeypatch.setattr(node.quic_transport, "_running", True, raising=False)
    monkeypatch.setattr(type(node.quic_transport), "is_running", property(lambda _self: True))

    async def _quic_none(*_args: Any, **_kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(node.quic_transport, "send_frame", _quic_none)
    quic_rtt = await node.send_frame_to_peer(quic_peer, b"x" * 100, timeout_seconds=1.0)
    assert quic_rtt == 0.0
    assert node.last_p2p_rtt_ms == 12.5

    # 3. Cached relay transport returning NaN falls back to 0.0 without corrupting metrics
    relay_peer = _make_peer("relay-peer")
    node._set_cached_peer_route(relay_peer.peer_id, "relay")
    monkeypatch.setattr(type(node.relay_client), "is_connected", property(lambda _self: True))

    async def _relay_nan(*_args: Any, **_kwargs: Any) -> Any:
        return float("nan")

    monkeypatch.setattr(node.relay_client, "send_frame", _relay_nan)
    relay_rtt = await node.send_frame_to_peer(relay_peer, b"x" * 100, timeout_seconds=1.0)
    assert relay_rtt == 0.0
    assert node.last_p2p_rtt_ms == 12.5
