"""Unit and integration tests for Issue #7:
1. Post-validation/post-handler P2P socket ACK/NACK propagation on Middle-Node failure.
2. Step-synchronized token queue ingestion/consumption preventing stale token desynchronization
   after Primary timeout and Backup failover.
"""

from __future__ import annotations

import asyncio
import os
import struct
import time
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import pytest

import main as main_module
from main import LocalNodeRuntime
from models.shard_runner import ShardModelRunner
from tests._mock_runner import make_mock_runner
from p2p.serializer import (
    MAGIC_HEADER,
    ActivationFrame,
    DecodedTokenFrame,
    deserialize_activation_tensor,
    serialize_activation_tensor,
    serialize_decoded_token,
)
from p2p.transport import (
    ACK_BYTE,
    FRAME_LEN_STRUCT,
    NACK_BYTE,
    receive_activations,
    send_activations,
)
from shard_engine.schemas import (
    NetworkStatus,
    NodeState,
    PeerNodeInfo,
    PeerRole,
    RoomLobbyItem,
    StartNodeRequest,
)
from shard_engine.tracker_client import AnnounceAssignment


def _get_bound_port(server: asyncio.AbstractServer) -> int:
    assert server.sockets
    return int(server.sockets[0].getsockname()[1])


async def _send_raw_and_read_reply(port: int, payload: bytes) -> bytes:
    """Send a length-prefixed raw frame and return the single-byte ACK/NACK reply."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(FRAME_LEN_STRUCT.pack(len(payload)) + payload)
        await writer.drain()
        return await asyncio.wait_for(reader.readexactly(1), timeout=2.0)
    finally:
        writer.close()
        await writer.wait_closed()


def _make_valid_activation_frame(
    request_id: str | None = None,
    origin_port: int = 49999,
    current_layer: int = 15,
    total_layers: int = 32,
    step_index: int = 0,
    max_tokens: int = 4,
) -> bytes:
    runner = ShardModelRunner(shard_layer_start=0, shard_layer_end=15, total_layers=total_layers)
    raw_bytes, shape = runner.embed_tokens_to_raw_tensor([11, 22, 33])
    return serialize_activation_tensor(
        tensor_data=raw_bytes,
        shape=shape,
        request_id=request_id or str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=origin_port,
        current_layer=current_layer,
        total_layers=total_layers,
        step_index=step_index,
        max_tokens=max_tokens,
        force_compression=True,
    )


@pytest.mark.asyncio
async def test_p2p_transport_sends_nack_on_invalid_magic_corrupted_crc_and_bad_type() -> None:
    """Malformed frames must receive NACK_BYTE and never invoke the activation callback."""
    handler_calls: int = 0

    async def _on_activation(_frame: ActivationFrame) -> None:
        nonlocal handler_calls
        handler_calls += 1

    server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_on_activation,
    )
    port = _get_bound_port(server)
    try:
        # 1. Invalid magic bytes
        bad_magic = b"NOPE" + b"\x01\x00" + b"\x00" * 32
        assert await _send_raw_and_read_reply(port, bad_magic) == NACK_BYTE
        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", bad_magic, peer_port=port, timeout_seconds=2.0)

        # 2. Valid header + corrupted payload byte (CRC32 mismatch)
        valid_frame = _make_valid_activation_frame()
        corrupted = bytearray(valid_frame)
        corrupted[-1] ^= 0xFF
        corrupted_bytes = bytes(corrupted)
        assert await _send_raw_and_read_reply(port, corrupted_bytes) == NACK_BYTE
        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations(
                "127.0.0.1", corrupted_bytes, peer_port=port, timeout_seconds=2.0
            )

        # 3. Unsupported packet type
        bad_pkt_type = MAGIC_HEADER + bytes([0xFE, 0x00]) + b"\x00" * 32
        assert await _send_raw_and_read_reply(port, bad_pkt_type) == NACK_BYTE
        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", bad_pkt_type, peer_port=port, timeout_seconds=2.0)

        assert handler_calls == 0
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_p2p_transport_acks_only_after_handler_completes_and_nacks_on_exception() -> None:
    """ACK_BYTE must only be sent after handler finishes; handler exceptions must return NACK_BYTE."""
    handler_completed: bool = False
    should_fail: bool = False

    async def _on_activation(_frame: ActivationFrame) -> None:
        nonlocal handler_completed
        await asyncio.sleep(0.05)
        if should_fail:
            raise RuntimeError("Simulated middle-node forwarding failure")
        handler_completed = True

    server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_on_activation,
    )
    port = _get_bound_port(server)
    try:
        frame = _make_valid_activation_frame()

        # Success path: send_activations must not return before handler_completed is True
        await send_activations("127.0.0.1", frame, peer_port=port, timeout_seconds=2.0)
        assert handler_completed is True

        # Failure path: handler raises -> peer immediately returns NACK_BYTE
        should_fail = True
        assert await _send_raw_and_read_reply(port, frame) == NACK_BYTE
        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", frame, peer_port=port, timeout_seconds=2.0)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_p2p_transport_converts_abrupt_eof_to_runtime_error() -> None:
    """Abrupt peer socket close without ACK/NACK must raise RuntimeError (caught by failover)."""

    async def _drop_connection(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
        (frame_len,) = struct.unpack("!I", len_bytes)
        await reader.readexactly(frame_len)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_drop_connection, host="127.0.0.1", port=0)
    port = _get_bound_port(server)
    try:
        frame = _make_valid_activation_frame()
        with pytest.raises(RuntimeError, match="closed connection before sending ACK"):
            await send_activations("127.0.0.1", frame, peer_port=port, timeout_seconds=2.0)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_p2p_transport_nacks_decoded_token_when_no_token_handler_registered() -> None:
    """Sending DECODED_TOKEN to a listener without `on_token_frame` must return NACK_BYTE."""
    server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=lambda _f: asyncio.sleep(0),
        on_token_frame=None,
    )
    port = _get_bound_port(server)
    try:
        token_pkt = serialize_decoded_token(
            request_id=str(uuid.uuid4()),
            step_index=0,
            token_id=42,
            token_text="hello",
            is_eos=False,
        )
        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", token_pkt, peer_port=port, timeout_seconds=2.0)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_middle_node_non_running_and_missing_downstream_peer_return_nack() -> None:
    """Stopped nodes and real-model Middle Nodes missing a downstream peer must return NACK."""
    # Case 1: Node is STOPPED
    stopped_node = LocalNodeRuntime()
    stopped_node.state = NodeState.STOPPED
    stopped_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=stopped_node.handle_incoming_activation,
    )
    stopped_port = _get_bound_port(stopped_server)

    # Case 2: Real-model Middle Node with no downstream peer for layer 24
    class _FakeRealMiddleRunner(ShardModelRunner):
        @property
        def is_real_model(self) -> bool:
            return True

    orphan_middle = LocalNodeRuntime()
    orphan_middle.state = NodeState.RUNNING
    orphan_middle.shard_layer_start = 16
    orphan_middle.shard_layer_end = 23
    orphan_middle.total_layers = 32
    orphan_middle.peers = []
    orphan_middle.model_runner = _FakeRealMiddleRunner(
        shard_layer_start=16,
        shard_layer_end=23,
        total_layers=32,
    )
    orphan_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=orphan_middle.handle_incoming_activation,
    )
    orphan_port = _get_bound_port(orphan_server)

    try:
        frame = _make_valid_activation_frame(current_layer=15, total_layers=32)

        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", frame, peer_port=stopped_port, timeout_seconds=2.0)

        with pytest.raises(RuntimeError, match="NACK"):
            await send_activations("127.0.0.1", frame, peer_port=orphan_port, timeout_seconds=2.0)
    finally:
        stopped_server.close()
        orphan_server.close()
        await stopped_server.wait_closed()
        await orphan_server.wait_closed()


@pytest.mark.asyncio
async def test_head_node_fails_over_immediately_on_middle_node_forwarding_failure() -> None:
    """When Primary Middle Node fails to forward downstream, Head Node receives NACK and fails over immediately."""

    async def _crashed_tail_drop(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    crashed_tail_server = await asyncio.start_server(_crashed_tail_drop, host="127.0.0.1", port=0)
    dead_downstream_port = _get_bound_port(crashed_tail_server)

    # 1. Broken Primary Middle Node (layers 16..23 -> crashed downstream peer)
    broken_middle = LocalNodeRuntime()
    broken_middle.state = NodeState.RUNNING
    broken_middle.shard_layer_start = 16
    broken_middle.shard_layer_end = 23
    broken_middle.total_layers = 32
    broken_middle.model_runner = make_mock_runner(
        shard_layer_start=16, shard_layer_end=23, total_layers=32
    )
    broken_middle.peers = [
        PeerNodeInfo(
            peer_id="dead-tail-peer",
            ip_address="127.0.0.1",
            port=dead_downstream_port,
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        )
    ]
    broken_middle_port = await broken_middle.start_p2p_listener(port=0)

    # 2. Healthy Backup Tail Node (layers 16..31)
    backup_tail = LocalNodeRuntime()
    backup_tail.state = NodeState.RUNNING
    backup_tail.shard_layer_start = 16
    backup_tail.shard_layer_end = 31
    backup_tail.total_layers = 32
    backup_tail.model_runner = make_mock_runner(
        shard_layer_start=16, shard_layer_end=31, total_layers=32
    )
    backup_tail_port = await backup_tail.start_p2p_listener(port=0)

    # 3. Head Node (layers 0..15) with a 5.0s token timeout to prove failover is immediate (< 2.0s)
    head_node = LocalNodeRuntime()
    head_node.state = NodeState.RUNNING
    head_node.shard_layer_start = 0
    head_node.shard_layer_end = 15
    head_node.total_layers = 32
    head_node.failover_token_timeout_seconds = 5.0
    head_node.model_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=15, total_layers=32
    )
    head_node.peers = [
        PeerNodeInfo(
            peer_id="broken-primary-middle",
            ip_address="127.0.0.1",
            port=broken_middle_port,
            shard_layer_start=16,
            shard_layer_end=23,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="healthy-backup-tail",
            ip_address="127.0.0.1",
            port=backup_tail_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]
    head_port = await head_node.start_p2p_listener(port=0)
    assert head_node.p2p_listen_port == head_port
    head_peer = PeerNodeInfo(
        peer_id="head-node",
        ip_address="127.0.0.1",
        port=head_port,
        shard_layer_start=0,
        shard_layer_end=15,
        role=PeerRole.PRIMARY,
    )
    broken_middle.peers.append(head_peer)
    backup_tail.peers.append(head_peer)

    try:
        start_time = time.monotonic()
        streamed_tokens: list[tuple[int, int, str]] = []
        async for _req_id, step_idx, tok_id, tok_text, _is_eos in head_node.stream_p2p_pipeline_tokens(
            prompt="Test immediate NACK failover",
            max_tokens=2,
        ):
            streamed_tokens.append((step_idx, tok_id, tok_text))
        elapsed = time.monotonic() - start_time

        assert len(streamed_tokens) == 2
        assert [step for step, _, _ in streamed_tokens] == [0, 1]
        # Must fail over immediately via NACK_BYTE, NOT wait for the 5.0s token timeout
        assert elapsed < 2.5, f"Expected immediate NACK failover (<2.5s), took {elapsed:.2f}s"
        assert head_node.pending_token_queues == {}
        assert head_node.active_request_steps == {}
    finally:
        await head_node.stop_p2p_listener()
        await broken_middle.stop_p2p_listener()
        await backup_tail.stop_p2p_listener()
        crashed_tail_server.close()
        await crashed_tail_server.wait_closed()


@pytest.mark.asyncio
async def test_late_token_from_timed_out_primary_does_not_desynchronize_after_backup_failover() -> None:
    """A late DecodedTokenFrame(step=0) arriving from a timed-out Primary peer during step=1
    must be discarded and must not shift or corrupt subsequent tokens from Backup peer.
    """
    head_node = LocalNodeRuntime()
    head_node.state = NodeState.RUNNING
    head_node.shard_layer_start = 0
    head_node.shard_layer_end = 15
    head_node.total_layers = 32
    head_node.failover_token_timeout_seconds = 0.20
    head_node.model_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=15, total_layers=32
    )
    head_port = await head_node.start_p2p_listener(port=0)
    assert head_node.p2p_listen_port == head_port

    late_primary_tasks: list[asyncio.Task[Any]] = []
    step1_started = asyncio.Event()
    late_primary_sent = asyncio.Event()

    async def _primary_slow_server(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Simulates a Primary peer that ACKs receipt for step 0, stalls past Head's timeout,
        and then delivers its stale step-0 token AFTER Head has moved on to step 1.
        """
        try:
            len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
            (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
            raw_frame = await reader.readexactly(frame_len)
            from p2p.serializer import deserialize_activation_tensor

            frame = deserialize_activation_tensor(raw_frame)

            if frame.step_index == 0:
                # Immediately ACK so Head enters _wait_for_step_token and times out at 0.20s
                writer.write(ACK_BYTE)
                await writer.drain()

                async def _send_stale_token_later() -> None:
                    # Wait deterministically until Head has moved on to step 1 and invoked Backup
                    await asyncio.wait_for(step1_started.wait(), timeout=2.0)
                    stale_packet = serialize_decoded_token(
                        request_id=frame.request_id,
                        step_index=0,
                        token_id=9999,
                        token_text="STALE_PRIMARY_STEP0",
                        is_eos=False,
                    )
                    await send_activations(
                        peer_ip=frame.origin_ip,
                        byte_stream=stale_packet,
                        peer_port=frame.origin_port,
                        timeout_seconds=2.0,
                    )
                    late_primary_sent.set()

                late_primary_tasks.append(asyncio.create_task(_send_stale_token_later()))
            else:
                # Fail fast with NACK on subsequent steps so Backup handles step 1 & step 2 cleanly
                writer.write(NACK_BYTE)
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _backup_on_activation(frame: ActivationFrame) -> None:
        if frame.step_index == 1:
            # Ensure the late Primary step-0 token arrives BEFORE Backup sends step-1 token
            step1_started.set()
            await asyncio.wait_for(late_primary_sent.wait(), timeout=2.0)
        token_packet = serialize_decoded_token(
            request_id=frame.request_id,
            step_index=frame.step_index,
            token_id=100 + frame.step_index,
            token_text=f"BACKUP_STEP_{frame.step_index}",
            is_eos=False,
        )
        await send_activations(
            peer_ip=frame.origin_ip,
            byte_stream=token_packet,
            peer_port=frame.origin_port,
            timeout_seconds=2.0,
        )

    primary_server = await asyncio.start_server(_primary_slow_server, host="127.0.0.1", port=0)
    primary_port = _get_bound_port(primary_server)

    backup_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_backup_on_activation,
    )
    backup_port = _get_bound_port(backup_server)

    head_node.peers = [
        PeerNodeInfo(
            peer_id="slow-primary-peer",
            ip_address="127.0.0.1",
            port=primary_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="fast-backup-peer",
            ip_address="127.0.0.1",
            port=backup_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]

    try:
        results: list[tuple[int, int, str]] = []
        async for _req_id, step_idx, tok_id, tok_text, _is_eos in head_node.stream_p2p_pipeline_tokens(
            prompt="Test stale token rejection",
            max_tokens=3,
        ):
            results.append((step_idx, tok_id, tok_text))

        assert late_primary_sent.is_set(), "Late primary token must have been transmitted during test"
        assert results == [
            (0, 100, "BACKUP_STEP_0"),
            (1, 101, "BACKUP_STEP_1"),
            (2, 102, "BACKUP_STEP_2"),
        ]
        assert head_node.pending_token_queues == {}
        assert head_node.active_request_steps == {}
    finally:
        for task in late_primary_tasks:
            if not task.done():
                task.cancel()
        if late_primary_tasks:
            await asyncio.gather(*late_primary_tasks, return_exceptions=True)
        primary_server.close()
        backup_server.close()
        await primary_server.wait_closed()
        await backup_server.wait_closed()
        await head_node.stop_p2p_listener()


@pytest.mark.asyncio
async def test_wait_for_step_token_skips_stale_queued_frames_and_enforces_deadline() -> None:
    """`_wait_for_step_token` and `handle_incoming_decoded_token` must skip mismatched steps
    without resetting the timeout budget and keep queues bounded.
    """
    node = LocalNodeRuntime()
    req_id = str(uuid.uuid4())
    other_req_id = str(uuid.uuid4())
    queue: asyncio.Queue[DecodedTokenFrame] = asyncio.Queue(maxsize=64)
    node.pending_token_queues[req_id] = queue
    node.active_request_steps[req_id] = 2

    # 1. Ingress filter drops out-of-step frames immediately
    await node.handle_incoming_decoded_token(
        DecodedTokenFrame(
            request_id=req_id,
            step_index=1,
            token_id=11,
            token_text="STALE_INGRESS",
            is_eos=False,
        )
    )
    assert queue.empty()

    # 2. Even if stale/mismatched frames are already queued, `_wait_for_step_token` skips them
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=req_id, step_index=0, token_id=1, token_text="OLD_0", is_eos=False
        )
    )
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=other_req_id, step_index=2, token_id=2, token_text="WRONG_REQ", is_eos=False
        )
    )
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=req_id, step_index=2, token_id=222, token_text="VALID_2", is_eos=False
        )
    )

    matched = await node._wait_for_step_token(
        queue,
        request_id=req_id,
        expected_step=2,
        timeout_seconds=1.0,
    )
    assert matched.step_index == 2
    assert matched.token_id == 222
    assert matched.token_text == "VALID_2"

    # 3. Continuous stale frames must NOT reset the deadline
    async def _spam_stale() -> None:
        for _ in range(5):
            await asyncio.sleep(0.03)
            queue.put_nowait(
                DecodedTokenFrame(
                    request_id=req_id,
                    step_index=1,
                    token_id=99,
                    token_text="SPAM_STALE",
                    is_eos=False,
                )
            )

    spammer = asyncio.create_task(_spam_stale())
    try:
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            await node._wait_for_step_token(
                queue,
                request_id=req_id,
                expected_step=2,
                timeout_seconds=0.12,
            )
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
    finally:
        spammer.cancel()
        await asyncio.gather(spammer, return_exceptions=True)

    # 4. Bounded queue overflow protection: sending > 64 valid frames never blocks
    for i in range(80):
        await node.handle_incoming_decoded_token(
            DecodedTokenFrame(
                request_id=req_id,
                step_index=2,
                token_id=i,
                token_text=f"tok_{i}",
                is_eos=False,
            )
        )
    assert queue.qsize() == 64


@pytest.mark.asyncio
async def test_drain_discards_late_step0_token_arriving_before_backup_retry() -> None:
    """`_drain_token_queue_for_step(..., discard_all=True)` clears late primary tokens before
    failing over to backup on the same step, while `discard_all=False` keeps valid frames.
    """
    node = LocalNodeRuntime()
    req_id = str(uuid.uuid4())
    other_req_id = str(uuid.uuid4())
    queue: asyncio.Queue[DecodedTokenFrame] = asyncio.Queue(maxsize=64)

    # Mixed queue: stale step 0, mismatched request, and valid step 1
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=req_id, step_index=0, token_id=10, token_text="LATE_0", is_eos=False
        )
    )
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=other_req_id, step_index=1, token_id=20, token_text="OTHER_REQ", is_eos=False
        )
    )
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=req_id, step_index=1, token_id=30, token_text="VALID_1", is_eos=False
        )
    )

    discarded = node._drain_token_queue_for_step(
        queue,
        request_id=req_id,
        expected_step=1,
        discard_all=False,
    )
    assert discarded == 2
    assert queue.qsize() == 1
    retained = queue.get_nowait()
    assert retained.request_id == req_id
    assert retained.step_index == 1
    assert retained.token_id == 30

    # Failover retry (`discard_all=True`) must flush even a same-step late primary frame
    queue.put_nowait(
        DecodedTokenFrame(
            request_id=req_id, step_index=1, token_id=99, token_text="LATE_SAME_STEP", is_eos=False
        )
    )
    discarded_failover = node._drain_token_queue_for_step(
        queue,
        request_id=req_id,
        expected_step=1,
        discard_all=True,
    )
    assert discarded_failover == 1
    assert queue.empty()


@pytest.mark.asyncio
async def test_pipeline_failover_evicts_dead_primary_and_promotes_backup_without_repeat_timeouts() -> None:
    """Issue #28: After Primary fails at step 2, Backup is promoted to PRIMARY and Primary is
    evicted so steps 3..7 route directly to Backup with zero repeated timeout penalties.
    """
    head_node = LocalNodeRuntime()
    head_node.state = NodeState.RUNNING
    head_node.shard_layer_start = 0
    head_node.shard_layer_end = 15
    head_node.total_layers = 32
    head_node.failover_token_timeout_seconds = 0.25
    head_node.model_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=15, total_layers=32
    )
    await head_node.start_p2p_listener(port=0)

    primary_attempts: int = 0
    backup_attempts: int = 0

    async def _primary_stalls_at_step2(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal primary_attempts
        try:
            len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
            (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
            raw_frame = await reader.readexactly(frame_len)
            frame = deserialize_activation_tensor(raw_frame)
            primary_attempts += 1

            writer.write(ACK_BYTE)
            await writer.drain()

            if frame.step_index < 2:
                token_packet = serialize_decoded_token(
                    request_id=frame.request_id,
                    step_index=frame.step_index,
                    token_id=10 + frame.step_index,
                    token_text=f"PRIMARY_STEP_{frame.step_index}",
                    is_eos=False,
                )
                await send_activations(
                    peer_ip=frame.origin_ip,
                    byte_stream=token_packet,
                    peer_port=frame.origin_port,
                    timeout_seconds=2.0,
                )
            # For step_index >= 2: stall (never deliver decoded token)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _backup_on_activation(frame: ActivationFrame) -> None:
        nonlocal backup_attempts
        backup_attempts += 1
        token_packet = serialize_decoded_token(
            request_id=frame.request_id,
            step_index=frame.step_index,
            token_id=200 + frame.step_index,
            token_text=f"BACKUP_STEP_{frame.step_index}",
            is_eos=False,
        )
        await send_activations(
            peer_ip=frame.origin_ip,
            byte_stream=token_packet,
            peer_port=frame.origin_port,
            timeout_seconds=2.0,
        )

    primary_server = await asyncio.start_server(
        _primary_stalls_at_step2, host="127.0.0.1", port=0
    )
    primary_port = _get_bound_port(primary_server)

    backup_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_backup_on_activation,
    )
    backup_port = _get_bound_port(backup_server)

    head_node.peers = [
        PeerNodeInfo(
            peer_id="dead-primary-tail",
            ip_address="127.0.0.1",
            port=primary_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="backup-tail",
            ip_address="127.0.0.1",
            port=backup_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]

    try:
        start_time = time.monotonic()
        tokens: list[tuple[int, int, str]] = []
        async for _req_id, step_idx, tok_id, tok_text, _is_eos in head_node.stream_p2p_pipeline_tokens(
            prompt="Test persistent failover promotion",
            max_tokens=8,
        ):
            tokens.append((step_idx, tok_id, tok_text))
        elapsed = time.monotonic() - start_time

        assert tokens == [
            (0, 10, "PRIMARY_STEP_0"),
            (1, 11, "PRIMARY_STEP_1"),
            (2, 202, "BACKUP_STEP_2"),
            (3, 203, "BACKUP_STEP_3"),
            (4, 204, "BACKUP_STEP_4"),
            (5, 205, "BACKUP_STEP_5"),
            (6, 206, "BACKUP_STEP_6"),
            (7, 207, "BACKUP_STEP_7"),
        ]
        # Steps 0, 1 succeeded + ONE failed attempt on step 2 -> ZERO attempts on steps 3..7
        assert primary_attempts == 3
        assert backup_attempts == 6
        assert elapsed < 1.5, f"Expected single failover penalty (<1.5s), took {elapsed:.2f}s"

        assert [p.peer_id for p in head_node.peers] == ["backup-tail"]
        assert head_node.peers[0].role == PeerRole.PRIMARY
        next_hop = head_node.find_next_hop_peer(15)
        assert next_hop is not None
        assert next_hop.peer_id == "backup-tail"
        assert next_hop.role == PeerRole.PRIMARY
        assert head_node.is_swarm_ready is True
    finally:
        primary_server.close()
        backup_server.close()
        await primary_server.wait_closed()
        await backup_server.wait_closed()
        await head_node.stop_p2p_listener()


@pytest.mark.asyncio
async def test_send_with_failover_promotes_backup_and_evicts_failed_primary_on_middle_node() -> None:
    """Issue #28: Middle-node `send_with_failover` must evict the failed primary and promote
    the healthy backup so subsequent hops route directly without retrying the dead peer.
    """
    middle = LocalNodeRuntime()
    middle.state = NodeState.RUNNING
    middle.shard_layer_start = 16
    middle.shard_layer_end = 23
    middle.total_layers = 32

    dead_attempts: int = 0
    backup_attempts: int = 0

    async def _dead_nack_server(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        nonlocal dead_attempts
        try:
            len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
            (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
            await reader.readexactly(frame_len)
            dead_attempts += 1
            writer.write(NACK_BYTE)
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _healthy_backup(_frame: ActivationFrame) -> None:
        nonlocal backup_attempts
        backup_attempts += 1

    dead_server = await asyncio.start_server(_dead_nack_server, host="127.0.0.1", port=0)
    dead_port = _get_bound_port(dead_server)

    backup_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_healthy_backup,
    )
    backup_port = _get_bound_port(backup_server)

    middle.peers = [
        PeerNodeInfo(
            peer_id="dead-primary-tail",
            ip_address="127.0.0.1",
            port=dead_port,
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="healthy-backup-tail",
            ip_address="127.0.0.1",
            port=backup_port,
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]

    try:
        frame = _make_valid_activation_frame(current_layer=23, total_layers=32)

        first = await middle.send_with_failover(
            completed_layer=23,
            frame_bytes=frame,
            timeout_seconds=1.0,
            ack_timeout_seconds=1.0,
        )
        assert first.peer_id == "healthy-backup-tail"
        assert first.role == PeerRole.PRIMARY
        assert dead_attempts == 1
        assert backup_attempts == 1
        assert [p.peer_id for p in middle.peers] == ["healthy-backup-tail"]
        assert middle.peers[0].role == PeerRole.PRIMARY

        second = await middle.send_with_failover(
            completed_layer=23,
            frame_bytes=frame,
            timeout_seconds=1.0,
            ack_timeout_seconds=1.0,
        )
        assert second.peer_id == "healthy-backup-tail"
        assert second.role == PeerRole.PRIMARY
        assert dead_attempts == 1
        assert backup_attempts == 2
    finally:
        dead_server.close()
        backup_server.close()
        await dead_server.wait_closed()
        await backup_server.wait_closed()


@pytest.mark.asyncio
async def test_cascading_failover_evicts_multiple_failed_candidates_and_promotes_working_backup() -> None:
    """Issue #28: Cascading failover across `[P1, B1, B2]` must evict both `P1` and `B1`,
    promote `B2` to PRIMARY, and preserve unrelated layer peers in `runtime.peers`.
    """
    runtime = LocalNodeRuntime()
    runtime.state = NodeState.RUNNING
    runtime.shard_layer_start = 16
    runtime.shard_layer_end = 23
    runtime.total_layers = 32

    p1_attempts: int = 0
    b1_attempts: int = 0
    b2_attempts: int = 0

    async def _make_nack_handler(counter_name: str) -> asyncio.AbstractServer:
        async def _handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            nonlocal p1_attempts, b1_attempts
            try:
                len_bytes = await reader.readexactly(FRAME_LEN_STRUCT.size)
                (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
                await reader.readexactly(frame_len)
                if counter_name == "p1":
                    p1_attempts += 1
                else:
                    b1_attempts += 1
                writer.write(NACK_BYTE)
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        return await asyncio.start_server(_handler, host="127.0.0.1", port=0)

    async def _b2_ok(_frame: ActivationFrame) -> None:
        nonlocal b2_attempts
        b2_attempts += 1

    p1_server = await _make_nack_handler("p1")
    b1_server = await _make_nack_handler("b1")
    b2_server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_b2_ok,
    )

    unrelated_head = PeerNodeInfo(
        peer_id="unrelated-head",
        ip_address="127.0.0.1",
        port=49100,
        shard_layer_start=0,
        shard_layer_end=15,
        role=PeerRole.PRIMARY,
    )
    runtime.peers = [
        unrelated_head,
        PeerNodeInfo(
            peer_id="p1",
            ip_address="127.0.0.1",
            port=_get_bound_port(p1_server),
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="b1",
            ip_address="127.0.0.1",
            port=_get_bound_port(b1_server),
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
        PeerNodeInfo(
            peer_id="b2",
            ip_address="127.0.0.1",
            port=_get_bound_port(b2_server),
            shard_layer_start=24,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]

    try:
        frame = _make_valid_activation_frame(current_layer=23, total_layers=32)
        promoted = await runtime.send_with_failover(
            completed_layer=23,
            frame_bytes=frame,
            timeout_seconds=1.0,
            ack_timeout_seconds=1.0,
        )
        assert promoted.peer_id == "b2"
        assert promoted.role == PeerRole.PRIMARY
        assert [p.peer_id for p in runtime.peers] == ["unrelated-head", "b2"]
        assert runtime.peers[0].role == PeerRole.PRIMARY
        assert runtime.peers[1].role == PeerRole.PRIMARY

        second = await runtime.send_with_failover(
            completed_layer=23,
            frame_bytes=frame,
            timeout_seconds=1.0,
            ack_timeout_seconds=1.0,
        )
        assert second.peer_id == "b2"
        assert p1_attempts == 1
        assert b1_attempts == 1
        assert b2_attempts == 2
    finally:
        p1_server.close()
        b1_server.close()
        b2_server.close()
        await p1_server.wait_closed()
        await b1_server.wait_closed()
        await b2_server.wait_closed()


@pytest.mark.asyncio
async def test_all_candidates_failed_preserves_peers_without_destructive_eviction() -> None:
    """Issue #28: When every primary and backup candidate fails, raise RuntimeError without
    destructively wiping `runtime.peers` or quarantining peers.
    """
    runtime = LocalNodeRuntime()
    runtime.state = NodeState.RUNNING
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.model_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=15, total_layers=32
    )

    async def _always_nack(
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
            try:
                await writer.wait_closed()
            except Exception:
                pass

    p1_server = await asyncio.start_server(_always_nack, host="127.0.0.1", port=0)
    b1_server = await asyncio.start_server(_always_nack, host="127.0.0.1", port=0)

    runtime.peers = [
        PeerNodeInfo(
            peer_id="p1",
            ip_address="127.0.0.1",
            port=_get_bound_port(p1_server),
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="b1",
            ip_address="127.0.0.1",
            port=_get_bound_port(b1_server),
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.BACKUP,
        ),
    ]

    try:
        frame = _make_valid_activation_frame(current_layer=15, total_layers=32)
        with pytest.raises(RuntimeError, match="All primary and backup peers failed"):
            await runtime.send_with_failover(
                completed_layer=15,
                frame_bytes=frame,
                timeout_seconds=1.0,
                ack_timeout_seconds=1.0,
            )

        assert [p.peer_id for p in runtime.peers] == ["p1", "b1"]
        assert runtime._evicted_peers == {}
        assert runtime._promoted_peer_ids == {}

        with pytest.raises(RuntimeError, match="All primary and backup peers failed"):
            async for _ in runtime.stream_p2p_pipeline_tokens(
                prompt="All peers down",
                max_tokens=2,
            ):
                pass

        assert [p.peer_id for p in runtime.peers] == ["p1", "b1"]
        assert runtime._evicted_peers == {}
        assert runtime._promoted_peer_ids == {}
    finally:
        p1_server.close()
        b1_server.close()
        await p1_server.wait_closed()
        await b1_server.wait_closed()


@pytest.mark.asyncio
async def test_failover_quarantine_blocks_stale_refresh_and_allows_reentry_on_new_port_or_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #28: Background refreshes must respect dead-peer quarantine and backup promotion
    during cooldown, while allowing re-entry on a new `(ip, port)`, after cooldown expiry, or
    via `POST /peers/register`.
    """
    runtime = LocalNodeRuntime()
    runtime.state = NodeState.RUNNING
    runtime.node_id = "head-node"
    runtime.model_id = "test/model"
    runtime.room_id = "public"
    runtime.network_status = NetworkStatus.ONLINE
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.p2p_listen_port = 49152
    runtime._active_start_request = StartNodeRequest(
        node_id="head-node",
        model_id="test/model",
        room_id="public",
        ram_gb=4,
        shared_ram_gb=4,
        p2p_listen_port=49152,
    )
    runtime._last_heartbeat_at = time.monotonic()

    p1_old = PeerNodeInfo(
        peer_id="p1",
        ip_address="127.0.0.1",
        port=49201,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
    )
    b1_backup = PeerNodeInfo(
        peer_id="b1",
        ip_address="127.0.0.1",
        port=49202,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.BACKUP,
    )
    runtime.peers = [p1_old, b1_backup]

    promoted = runtime._promote_failover_peer(promoted_peer=b1_backup, failed_peers=[p1_old])
    assert promoted.role == PeerRole.PRIMARY
    assert [p.peer_id for p in runtime.peers] == ["b1"]

    tracker_peers_payload: list[PeerNodeInfo] = [p1_old, b1_backup]

    async def _fake_rooms() -> list[RoomLobbyItem]:
        return [
            RoomLobbyItem(
                room_id="public",
                model="test/model",
                nodes_online=2,
                total_shared_ram_gb=8.0,
                coverage_percent=100.0,
            )
        ]

    async def _fake_fetch_peers(**_kwargs: Any) -> list[PeerNodeInfo]:
        return list(tracker_peers_payload)

    async def _fake_register_node(*_args: Any, **_kwargs: Any) -> AnnounceAssignment:
        return AnnounceAssignment(
            assigned_start_layer=0,
            assigned_end_layer=15,
            role=PeerRole.PRIMARY,
            peers=list(tracker_peers_payload),
        )

    async def _no_lan_beacons(**_kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(main_module, "_is_tracker_enabled", lambda: True)
    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _no_lan_beacons)
    monkeypatch.setattr(runtime.tracker_client, "fetch_available_rooms", _fake_rooms)
    monkeypatch.setattr(runtime.tracker_client, "fetch_peers", _fake_fetch_peers)
    monkeypatch.setattr(
        runtime.tracker_client, "register_node_with_assignment", _fake_register_node
    )

    # 1. Stale heartbeat and tracker refresh with p1 at same (ip, port) and b1 as BACKUP stay filtered
    assert await runtime.maybe_send_tracker_heartbeat(force=True) is True
    assert [p.peer_id for p in runtime.peers] == ["b1"]
    assert runtime.peers[0].role == PeerRole.PRIMARY

    await runtime.refresh_room_coverage()
    assert [p.peer_id for p in runtime.peers] == ["b1"]
    assert runtime.peers[0].role == PeerRole.PRIMARY

    # 2. If p1 restarts on a NEW port, quarantine is lifted automatically
    p1_restarted = p1_old.model_copy(update={"port": 49299})
    tracker_peers_payload = [p1_restarted, b1_backup]
    await runtime.refresh_room_coverage()
    assert {p.peer_id for p in runtime.peers} == {"p1", "b1"}
    assert "p1" not in runtime._evicted_peers
    b1_in_table = next(p for p in runtime.peers if p.peer_id == "b1")
    assert b1_in_table.role == PeerRole.PRIMARY

    # 3. Cooldown expiry prunes both quarantine and promotion overrides
    runtime.failed_peer_cooldown_seconds = 0.01
    runtime._promote_failover_peer(promoted_peer=b1_backup, failed_peers=[p1_old])
    assert "p1" in runtime._evicted_peers
    assert "b1" in runtime._promoted_peer_ids
    await asyncio.sleep(0.025)

    after_expiry = runtime._apply_peer_failover_overrides([p1_old, b1_backup])
    assert [p.peer_id for p in after_expiry] == ["p1", "b1"]
    assert after_expiry[1].role == PeerRole.BACKUP
    assert runtime._evicted_peers == {}
    assert runtime._promoted_peer_ids == {}

    # 4. Explicit /peers/register removes quarantine and _stop_node_unlocked clears failover state
    orig_peers = list(main_module.runtime.peers)
    orig_evicted = dict(main_module.runtime._evicted_peers)
    orig_promoted = dict(main_module.runtime._promoted_peer_ids)
    try:
        main_module.runtime._evicted_peers["p1"] = (time.monotonic() + 60.0, "127.0.0.1", 49201)
        registered = await main_module.register_peer(p1_old)
        assert "p1" not in main_module.runtime._evicted_peers
        assert any(p.peer_id == "p1" for p in registered)

        main_module.runtime.state = NodeState.RUNNING
        main_module.runtime._evicted_peers["p1"] = (time.monotonic() + 60.0, "127.0.0.1", 49201)
        main_module.runtime._promoted_peer_ids["b1"] = time.monotonic() + 60.0
        await main_module._stop_node_unlocked()
        assert main_module.runtime._evicted_peers == {}
        assert main_module.runtime._promoted_peer_ids == {}
    finally:
        main_module.runtime.peers = orig_peers
        main_module.runtime._evicted_peers = orig_evicted
        main_module.runtime._promoted_peer_ids = orig_promoted
        main_module.runtime.state = NodeState.STOPPED
