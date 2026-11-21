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

from main import LocalNodeRuntime
from models.shard_runner import ShardModelRunner
from p2p.serializer import (
    MAGIC_HEADER,
    ActivationFrame,
    DecodedTokenFrame,
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
from shard_engine.schemas import NodeState, PeerNodeInfo, PeerRole


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
    broken_middle.model_runner = ShardModelRunner(
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
    backup_tail.model_runner = ShardModelRunner(
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
    head_node.model_runner = ShardModelRunner(
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
    head_node.model_runner = ShardModelRunner(
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
