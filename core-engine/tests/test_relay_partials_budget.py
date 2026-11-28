"""Unit and lifecycle tests for RelayClient global partial byte budget and idle sweeper."""

from __future__ import annotations

import asyncio
import uuid

import pytest
import websockets.asyncio.server as ws_server

import p2p.relay_client as relay_client_mod
from p2p.relay_client import (
    MAX_TOTAL_PARTIAL_BYTES,
    RelayClient,
)
from p2p.relay_envelope import (
    RelayEnvelope,
    RelayPacketKind,
    decode_envelope,
    encode_envelope,
)
from p2p.transport import MAX_WIRE_FRAME_BYTES


def _make_data_env(
    *,
    src: str = "peer-a",
    dst: str = "node-local",
    msg_id: bytes | None = None,
    chunk_index: int = 0,
    chunk_count: int = 2,
    payload: bytes = b"x" * 100,
) -> RelayEnvelope:
    return RelayEnvelope(
        kind=RelayPacketKind.DATA,
        msg_id=msg_id if msg_id is not None else uuid.uuid4().bytes,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        src=src,
        dst=dst,
        payload=payload,
    )


def _actual_partial_bytes(client: RelayClient) -> int:
    return sum(
        buf.total_bytes
        for src_map in client._partials.values()
        for buf in src_map.values()
    )


def test_partial_budget_constant_fits_max_wire_frame() -> None:
    """Global partial cap must always accommodate at least one full maximum wire frame."""
    assert MAX_TOTAL_PARTIAL_BYTES >= MAX_WIRE_FRAME_BYTES


def test_partial_sweep_interval_respects_ttl_and_min_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweeper interval defaults to half the partial TTL and clamps to the configured floor."""
    monkeypatch.setattr(relay_client_mod, "PARTIAL_MESSAGE_TTL_SECONDS", 90.0)
    monkeypatch.setattr(relay_client_mod, "PARTIAL_SWEEP_MIN_INTERVAL_SECONDS", 1.0)
    assert RelayClient._partial_sweep_interval() == 45.0

    monkeypatch.setattr(relay_client_mod, "PARTIAL_MESSAGE_TTL_SECONDS", 0.2)
    assert RelayClient._partial_sweep_interval() == 1.0


def test_partial_counter_consistency_on_add_overwrite_completion_and_errors() -> None:
    """Aggregate byte counter stays exact across chunk add, overwrite, completion, and errors."""
    client = RelayClient()
    msg_ok = uuid.uuid4().bytes

    frame, err = client._reassemble_data_chunk(
        _make_data_env(msg_id=msg_ok, chunk_index=0, chunk_count=3, payload=b"a" * 120)
    )
    assert (frame, err) == (None, None)
    assert client.partial_total_bytes == 120 == _actual_partial_bytes(client)

    # Duplicate chunk with smaller payload adjusts total downward accurately.
    frame, err = client._reassemble_data_chunk(
        _make_data_env(msg_id=msg_ok, chunk_index=0, chunk_count=3, payload=b"b" * 80)
    )
    assert (frame, err) == (None, None)
    assert client.partial_total_bytes == 80 == _actual_partial_bytes(client)

    # Duplicate chunk with larger payload adjusts total upward accurately.
    frame, err = client._reassemble_data_chunk(
        _make_data_env(msg_id=msg_ok, chunk_index=0, chunk_count=3, payload=b"c" * 150)
    )
    assert (frame, err) == (None, None)
    assert client.partial_total_bytes == 150 == _actual_partial_bytes(client)

    # Add second chunk and an unrelated partial that will fail chunk_count consistency.
    msg_bad_count = uuid.uuid4().bytes
    client._reassemble_data_chunk(
        _make_data_env(
            src="peer-b", msg_id=msg_bad_count, chunk_index=0, chunk_count=2, payload=b"z" * 90
        )
    )
    assert client.partial_total_bytes == 240 == _actual_partial_bytes(client)

    frame, err = client._reassemble_data_chunk(
        _make_data_env(
            src="peer-b", msg_id=msg_bad_count, chunk_index=1, chunk_count=4, payload=b"z" * 90
        )
    )
    assert frame is None
    assert err == "Inconsistent chunk_count across chunks"
    assert "peer-b" not in client._partials
    assert client.partial_total_bytes == 150 == _actual_partial_bytes(client)

    # Complete msg_ok and confirm counter returns to zero.
    client._reassemble_data_chunk(
        _make_data_env(msg_id=msg_ok, chunk_index=1, chunk_count=3, payload=b"d" * 50)
    )
    assert client.partial_total_bytes == 200 == _actual_partial_bytes(client)

    frame, err = client._reassemble_data_chunk(
        _make_data_env(msg_id=msg_ok, chunk_index=2, chunk_count=3, payload=b"e" * 25)
    )
    assert err is None
    assert frame == (b"c" * 150) + (b"d" * 50) + (b"e" * 25)
    assert client.partial_total_bytes == 0
    assert client._partials == {}


def test_global_cap_evicts_oldest_other_partials_and_preserves_newer_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceeding `MAX_TOTAL_PARTIAL_BYTES` evicts oldest non-target partials in timestamp order."""
    monkeypatch.setattr(relay_client_mod, "MAX_TOTAL_PARTIAL_BYTES", 1000)
    monkeypatch.setattr(relay_client_mod, "PARTIAL_MESSAGE_TTL_SECONDS", 90.0)

    fake_now = 1000.0
    monkeypatch.setattr(relay_client_mod.time, "monotonic", lambda: fake_now)

    client = RelayClient()
    msg_a = uuid.uuid4().bytes
    msg_b = uuid.uuid4().bytes
    msg_c = uuid.uuid4().bytes

    # t=1000: peer-a stores 400 bytes (oldest)
    assert client._reassemble_data_chunk(
        _make_data_env(src="peer-a", msg_id=msg_a, chunk_index=0, chunk_count=2, payload=b"a" * 400)
    ) == (None, None)

    # t=1005: peer-b stores 400 bytes (middle)
    fake_now = 1005.0
    assert client._reassemble_data_chunk(
        _make_data_env(src="peer-b", msg_id=msg_b, chunk_index=0, chunk_count=2, payload=b"b" * 400)
    ) == (None, None)

    # t=1010: peer-c stores 150 bytes (newest); total = 950 <= 1000
    fake_now = 1010.0
    assert client._reassemble_data_chunk(
        _make_data_env(src="peer-c", msg_id=msg_c, chunk_index=0, chunk_count=3, payload=b"c" * 150)
    ) == (None, None)
    assert client.partial_total_bytes == 950

    # t=1012: peer-c adds 300 bytes -> 1250 > 1000. Evicting oldest (peer-a, 400B) leaves 850 <= 1000.
    fake_now = 1012.0
    assert client._reassemble_data_chunk(
        _make_data_env(src="peer-c", msg_id=msg_c, chunk_index=1, chunk_count=3, payload=b"d" * 300)
    ) == (None, None)
    assert "peer-a" not in client._partials
    assert "peer-b" in client._partials
    assert "peer-c" in client._partials
    assert client.partial_total_bytes == 850 == _actual_partial_bytes(client)

    # t=1014: peer-c completes with 300 bytes -> 1150 > 1000. Evicts peer-b (400B), then completes!
    fake_now = 1014.0
    frame, err = client._reassemble_data_chunk(
        _make_data_env(src="peer-c", msg_id=msg_c, chunk_index=2, chunk_count=3, payload=b"e" * 300)
    )
    assert err is None
    assert frame == (b"c" * 150) + (b"d" * 300) + (b"e" * 300)
    assert client._partials == {}
    assert client.partial_total_bytes == 0


def test_ensure_partial_capacity_prunes_expired_before_evicting_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity check reclaims expired buffers first before evicting unexpired peers."""
    monkeypatch.setattr(relay_client_mod, "MAX_TOTAL_PARTIAL_BYTES", 1000)
    monkeypatch.setattr(relay_client_mod, "PARTIAL_MESSAGE_TTL_SECONDS", 10.0)

    fake_now = 100.0
    monkeypatch.setattr(relay_client_mod.time, "monotonic", lambda: fake_now)

    client = RelayClient()
    msg_expired = uuid.uuid4().bytes
    msg_active = uuid.uuid4().bytes
    msg_new = uuid.uuid4().bytes

    # Active buffer created FIRST at t=100, wait: make msg_expired created at t=100, active at t=105.
    client._reassemble_data_chunk(
        _make_data_env(
            src="peer-expired", msg_id=msg_expired, chunk_index=0, chunk_count=2, payload=b"x" * 600
        )
    )
    fake_now = 105.0
    client._reassemble_data_chunk(
        _make_data_env(
            src="peer-active", msg_id=msg_active, chunk_index=0, chunk_count=2, payload=b"y" * 350
        )
    )
    assert client.partial_total_bytes == 950

    # At t=111, peer-expired is expired (age 11 > 10), peer-active is still valid (age 6 <= 10).
    ok = client._ensure_partial_capacity(
        additional_bytes=500,
        protected=("peer-new", msg_new),
        now=111.0,
    )
    assert ok is True
    assert "peer-expired" not in client._partials
    assert "peer-active" in client._partials
    assert client.partial_total_bytes == 350 == _actual_partial_bytes(client)


@pytest.mark.asyncio
async def test_global_cap_sends_nack_when_eviction_cannot_fit_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When evicting all other partials still cannot satisfy the budget, emit NACK and drop buffer."""
    monkeypatch.setattr(relay_client_mod, "MAX_TOTAL_PARTIAL_BYTES", 500)

    client = RelayClient()
    sent_controls: list[tuple[RelayPacketKind, bytes, str, bytes]] = []

    async def _capture_control(
        *,
        kind: RelayPacketKind,
        msg_id: bytes,
        dst: str,
        payload: bytes = b"",
    ) -> None:
        sent_controls.append((kind, msg_id, dst, payload))

    monkeypatch.setattr(client, "_send_control_envelope", _capture_control)

    other_id = uuid.uuid4().bytes
    await client._handle_incoming_envelope(
        _make_data_env(src="peer-other", msg_id=other_id, chunk_index=0, chunk_count=2, payload=b"o" * 150)
    )
    target_id = uuid.uuid4().bytes
    await client._handle_incoming_envelope(
        _make_data_env(src="peer-big", msg_id=target_id, chunk_index=0, chunk_count=3, payload=b"a" * 300)
    )
    assert client.partial_total_bytes == 450
    assert sent_controls == []

    # Second chunk for peer-big needs 300 + 300 = 600 > 500 even after evicting peer-other.
    await client._handle_incoming_envelope(
        _make_data_env(src="peer-big", msg_id=target_id, chunk_index=1, chunk_count=3, payload=b"b" * 300)
    )

    assert len(sent_controls) == 1
    kind, nack_msg_id, dst, reason = sent_controls[0]
    assert kind == RelayPacketKind.NACK
    assert nack_msg_id == target_id
    assert dst == "peer-big"
    assert b"budget exhausted" in reason
    assert client._partials == {}
    assert client.partial_total_bytes == 0


@pytest.mark.asyncio
async def test_idle_sweeper_reclaims_expired_partials_and_cancels_cleanly_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle background sweeper frees expired partials without incoming DATA and stops cleanly."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")
    monkeypatch.setattr(relay_client_mod, "PARTIAL_MESSAGE_TTL_SECONDS", 0.06)
    monkeypatch.setattr(relay_client_mod, "PARTIAL_SWEEP_MIN_INTERVAL_SECONDS", 0.02)

    server_side_conns: list[ws_server.ServerConnection] = []
    connected_evt = asyncio.Event()

    async def _ws_handler(conn: ws_server.ServerConnection) -> None:
        server_side_conns.append(conn)
        connected_evt.set()
        try:
            async for raw in conn:
                # Decode any control messages if sent; keep link open and idle.
                if isinstance(raw, (bytes, bytearray)):
                    decode_envelope(bytes(raw))
        except Exception:
            pass

    async with ws_server.serve(_ws_handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        relay_url = f"ws://127.0.0.1:{port}/relay/test-room"

        received_frames: list[tuple[bytes, str]] = []

        async def _on_frame(frame: bytes, src: str) -> None:
            received_frames.append((frame, src))

        client = RelayClient()
        await client.start(
            url=relay_url,
            token="test-token",
            node_id="node-receiver",
            on_frame=_on_frame,
            allowed_host="127.0.0.1",
        )
        try:
            await asyncio.wait_for(connected_evt.wait(), timeout=2.0)
            assert client.is_connected is True
            sweeper_task = client._sweeper_task
            assert sweeper_task is not None and not sweeper_task.done()

            # Inject a single incomplete chunk (1 of 2) and then keep the link completely idle.
            abandoned_id = uuid.uuid4().bytes
            await server_side_conns[0].send(
                encode_envelope(
                    kind=RelayPacketKind.DATA,
                    msg_id=abandoned_id,
                    src="peer-abandoned",
                    dst="node-receiver",
                    payload=b"p" * 4096,
                    chunk_index=0,
                    chunk_count=2,
                )
            )

            # Wait briefly for chunk 0 to be buffered.
            deadline = asyncio.get_running_loop().time() + 1.0
            while client.partial_total_bytes == 0 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.005)
            assert client.partial_total_bytes == 4096
            assert "peer-abandoned" in client._partials

            # Without sending any further frames, wait for TTL + sweeper tick to reclaim RAM.
            sweep_deadline = asyncio.get_running_loop().time() + 1.0
            while client._partials and asyncio.get_running_loop().time() < sweep_deadline:
                await asyncio.sleep(0.01)

            assert client.is_connected is True
            assert client._partials == {}
            assert client.partial_total_bytes == 0
            assert received_frames == []
        finally:
            sweeper_ref = client._sweeper_task
            await client.stop()

        assert sweeper_ref is not None and sweeper_ref.done()
        assert client._sweeper_task is None
        assert client.is_started is False
