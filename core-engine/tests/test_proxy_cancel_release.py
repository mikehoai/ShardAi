"""Regression tests for Issue #81: cancelled non-Head proxy streams NACK frames and release Head turn."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
import os
import threading
import time
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import pytest

import main as main_module
from main import LocalNodeRuntime, UnknownInferenceRequestError
from p2p.quic_transport import IrohP2PTransport
from p2p.relay_client import RelayClient
from p2p.relay_envelope import RelayEnvelope, RelayPacketKind
from p2p.serializer import (
    DecodedTokenFrame,
    InferenceHeartbeatFrame,
    InferenceRequestFrame,
    serialize_decoded_token,
    serialize_inference_heartbeat,
)
from p2p.transport import FRAME_LEN_STRUCT, NACK_BYTE, send_activations
from shard_engine.schemas import NetworkMode, NodeState, PeerNodeInfo, PeerRole
from tests._mock_runner import make_mock_runner


async def _cleanup_node(node: LocalNodeRuntime) -> None:
    """Tear down background tasks, P2P listeners, and mock runners cleanly."""
    await node.cleanup_proxy_state()
    await node.stop_p2p_listener()
    if node.model_runner is not None:
        node.model_runner.release()
        node.model_runner = None
    node.memory_manager.release_all()
    node.state = NodeState.STOPPED


def _configure_sharded_node(
    node: LocalNodeRuntime,
    *,
    node_id: str,
    shard_start: int,
    shard_end: int,
    total_layers: int,
) -> None:
    """Configure a `LocalNodeRuntime` with a mock pipeline runner."""
    node.state = NodeState.RUNNING
    node.node_id = node_id
    node.model_id = "test/sharded-llama-32l"
    node.room_id = "room-81"
    node.shard_layer_start = shard_start
    node.shard_layer_end = shard_end
    node.total_layers = total_layers
    node.n_layers = shard_end - shard_start + 1
    node.is_swarm_ready = True
    node.coverage_percent = 100.0
    node.model_runner = make_mock_runner(
        model_id=node.model_id,
        shard_layer_start=shard_start,
        shard_layer_end=shard_end,
        total_layers=total_layers,
        n_layers=node.n_layers,
    )


@pytest.mark.asyncio
async def test_unknown_request_token_and_heartbeat_raise_nack_error_and_keep_debug_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown/inactive `request_id` on token or heartbeat raises `UnknownInferenceRequestError`."""
    node = LocalNodeRuntime()
    unknown_id = str(uuid.uuid4())

    with caplog.at_level(logging.DEBUG, logger="shardai.main"):
        with pytest.raises(UnknownInferenceRequestError, match=unknown_id):
            await node.handle_incoming_decoded_token(
                DecodedTokenFrame(
                    request_id=unknown_id,
                    step_index=0,
                    token_id=42,
                    token_text="orphan",
                    is_eos=False,
                )
            )

        with pytest.raises(UnknownInferenceRequestError, match=unknown_id):
            await node.handle_incoming_inference_heartbeat(
                InferenceHeartbeatFrame(
                    request_id=unknown_id,
                    is_computing=True,
                    progress_index=1,
                )
            )

    assert any(unknown_id in rec.message for rec in caplog.records)

    # Active proxy queue still accepts both frame types without raising
    active_proxy_id = str(uuid.uuid4())
    proxy_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=8)
    node.pending_proxy_queues[active_proxy_id] = proxy_q

    hb = InferenceHeartbeatFrame(
        request_id=active_proxy_id, is_computing=True, progress_index=2
    )
    tok = DecodedTokenFrame(
        request_id=active_proxy_id,
        step_index=0,
        token_id=7,
        token_text="hi",
        is_eos=False,
    )
    await node.handle_incoming_inference_heartbeat(hb)
    await node.handle_incoming_decoded_token(tok)
    assert await proxy_q.get() == hb
    assert await proxy_q.get() == tok


@pytest.mark.asyncio
async def test_tcp_quic_and_relay_transports_map_unknown_request_frames_to_nack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TCP, QUIC, and Relay frame handlers all reply with NACK when `request_id` is unknown."""
    monkeypatch.setenv("SHARDAI_P2P_BIND_HOST", "127.0.0.1")
    node = LocalNodeRuntime()
    try:
        port = await node.start_p2p_listener(0, network_mode=NetworkMode.LAN)
        unknown_id = str(uuid.uuid4())
        tok_bytes = serialize_decoded_token(
            request_id=unknown_id,
            step_index=0,
            token_id=9,
            token_text="x",
            is_eos=False,
        )
        hb_bytes = serialize_inference_heartbeat(
            request_id=unknown_id,
            is_computing=True,
            progress_index=0,
        )

        # 1. TCP transport raises RuntimeError("... rejected frame with NACK")
        for payload in (tok_bytes, hb_bytes):
            with pytest.raises(RuntimeError, match="NACK"):
                await send_activations(
                    peer_ip="127.0.0.1",
                    byte_stream=payload,
                    peer_port=port,
                    timeout_seconds=2.0,
                )

        # 2. QUIC stream handler writes NACK_BYTE + reason when `on_frame` raises
        node.peers = [
            PeerNodeInfo(
                peer_id="head-quic",
                ip_address="127.0.0.1",
                port=port,
                shard_layer_start=0,
                shard_layer_end=15,
                role=PeerRole.PRIMARY,
                p2p_endpoint_id="ep-head-quic",
            )
        ]
        quic_transport = IrohP2PTransport()
        quic_transport._on_frame = node._handle_quic_frame

        class _FakeQuicRecv:
            def __init__(self, wire: bytes) -> None:
                self._buf = bytearray(wire)

            async def read_exact(self, n: int) -> bytes:
                out = bytes(self._buf[:n])
                del self._buf[:n]
                return out

        class _FakeQuicSend:
            def __init__(self) -> None:
                self.written = bytearray()
                self.finished = False

            async def write_all(self, data: bytes) -> None:
                self.written.extend(data)

            async def finish(self) -> None:
                self.finished = True

        class _FakeBiStream:
            def __init__(self, wire: bytes) -> None:
                self._recv = _FakeQuicRecv(wire)
                self._send = _FakeQuicSend()

            def recv(self) -> _FakeQuicRecv:
                return self._recv

            def send(self) -> _FakeQuicSend:
                return self._send

        for payload in (tok_bytes, hb_bytes):
            wire = FRAME_LEN_STRUCT.pack(len(payload)) + payload
            bi_stream = _FakeBiStream(wire)
            await quic_transport._handle_bi_stream(bi_stream, "ep-head-quic")
            assert bytes(bi_stream.send().written).startswith(NACK_BYTE)
            assert unknown_id.encode("utf-8") in bytes(bi_stream.send().written)
            assert bi_stream.send().finished is True

        # 3. Relay client `_dispatch_and_ack` sends RelayPacketKind.NACK envelope
        sent_envelopes: list[tuple[RelayPacketKind, bytes, str, bytes]] = []
        relay = RelayClient()
        relay._on_frame = node._handle_relay_frame

        async def _capture_control(
            *,
            kind: RelayPacketKind,
            msg_id: bytes,
            dst: str,
            payload: bytes = b"",
        ) -> None:
            sent_envelopes.append((kind, msg_id, dst, payload))

        monkeypatch.setattr(relay, "_send_control_envelope", _capture_control)

        for idx, payload in enumerate((tok_bytes, hb_bytes)):
            msg_id = uuid.uuid4().bytes
            await relay._dispatch_and_ack(payload, "head-relay", msg_id)
            assert len(sent_envelopes) == idx + 1
            kind, recorded_id, dst, reason_bytes = sent_envelopes[-1]
            assert kind == RelayPacketKind.NACK
            assert recorded_id == msg_id
            assert dst == "head-relay"
            assert unknown_id in reason_bytes.decode("utf-8")
            _ = RelayEnvelope  # verify import cleanliness
    finally:
        await _cleanup_node(node)


@pytest.mark.asyncio
async def test_head_proxied_task_stops_early_on_token_nack_and_releases_turn_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When token delivery NACKs, `_run_proxied_inference_task` stops immediately and frees the turn lock."""
    head = LocalNodeRuntime()
    try:
        _configure_sharded_node(
            head,
            node_id="head-nack",
            shard_start=0,
            shard_end=31,
            total_layers=32,
        )
        assert head.model_runner is not None
        # Keep `is_real_model=True` while forcing the sharded pipeline code path
        monkeypatch.setattr(
            type(head.model_runner), "has_native_llama", property(lambda _self: False)
        )

        generated_steps: list[int] = []
        orig_decode = head.model_runner.decode_final_layer_to_token

        def _counting_decode(
            hidden_state_bytes: bytes,
            step_index: int,
            max_tokens: int = 16,
            request_id: str = "default",
            temperature: float = 0.7,
        ) -> tuple[int, str, bool]:
            generated_steps.append(step_index)
            return orig_decode(
                hidden_state_bytes,
                step_index=step_index,
                max_tokens=max_tokens,
                request_id=request_id,
                temperature=temperature,
            )

        head.model_runner.decode_final_layer_to_token = _counting_decode  # type: ignore[method-assign]

        token_send_attempts = 0

        async def _mock_send_to_endpoint(
            ip: str,
            port: int,
            frame: bytes,
            timeout_seconds: float = 15.0,
            ack_timeout_seconds: float | None = None,
            target_peer_id: str | None = None,
        ) -> float:
            nonlocal token_send_attempts
            _ = ip, port, frame, timeout_seconds, ack_timeout_seconds, target_peer_id
            token_send_attempts += 1
            if token_send_attempts >= 2:
                raise RuntimeError("Peer 127.0.0.1:49999 rejected frame with NACK")
            return 0.5

        head._send_to_endpoint = _mock_send_to_endpoint  # type: ignore[method-assign]
        head.proxy_heartbeat_interval_seconds = 5.0

        req_frame = InferenceRequestFrame(
            request_id=str(uuid.uuid4()),
            requester_ip="127.0.0.1",
            requester_port=49999,
            prompt="Write a very long essay",
            messages=None,
            max_tokens=64,
            temperature=0.7,
            compress_activations=True,
        )

        await asyncio.wait_for(
            head._run_proxied_inference_task(req_frame, requester_peer_id="peer-tail"),
            timeout=2.0,
        )

        # Must stop right at step 1 (2 tokens generated, NOT all 64) and release the turn lock
        assert generated_steps == [0, 1]
        assert not head.get_pipeline_turn_lock().locked()
        assert head.pending_token_queues == {}
        assert head.active_request_steps == {}
    finally:
        await _cleanup_node(head)


@pytest.mark.asyncio
async def test_aborted_non_head_proxy_stream_releases_head_and_swarm_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aborting `_stream_via_head_node_proxy` after 1 token causes NACK on step 1 and frees Head for next request."""
    monkeypatch.setenv("SHARDAI_P2P_BIND_HOST", "127.0.0.1")
    head = LocalNodeRuntime()
    tail = LocalNodeRuntime()
    try:
        _configure_sharded_node(
            head, node_id="head-e2e", shard_start=0, shard_end=15, total_layers=32
        )
        _configure_sharded_node(
            tail, node_id="tail-e2e", shard_start=16, shard_end=31, total_layers=32
        )
        head.proxy_heartbeat_interval_seconds = 0.05

        tail_decoded_steps: list[int] = []
        assert tail.model_runner is not None
        orig_tail_decode = tail.model_runner.decode_final_layer_to_token

        def _slow_tail_decode(
            hidden_state_bytes: bytes,
            step_index: int,
            max_tokens: int = 16,
            request_id: str = "default",
            temperature: float = 0.7,
        ) -> tuple[int, str, bool]:
            time.sleep(0.04)
            tail_decoded_steps.append(step_index)
            return orig_tail_decode(
                hidden_state_bytes,
                step_index=step_index,
                max_tokens=max_tokens,
                request_id=request_id,
                temperature=temperature,
            )

        tail.model_runner.decode_final_layer_to_token = _slow_tail_decode  # type: ignore[method-assign]

        head_port = await head.start_p2p_listener(0, network_mode=NetworkMode.LAN)
        tail_port = await tail.start_p2p_listener(0, network_mode=NetworkMode.LAN)

        head_peer = PeerNodeInfo(
            peer_id="head-e2e",
            ip_address="127.0.0.1",
            port=head_port,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
        )
        tail_peer = PeerNodeInfo(
            peer_id="tail-e2e",
            ip_address="127.0.0.1",
            port=tail_port,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        )
        head.peers = [tail_peer]
        tail.peers = [head_peer]

        # Consume 1 token on the non-Head node and immediately close/abort the stream
        async with aclosing(
            tail.stream_p2p_pipeline_tokens(prompt="Abort me early", max_tokens=40)
        ) as stream:
            first_item = await asyncio.wait_for(anext(stream), timeout=3.0)
            assert first_item[1] == 0

        assert tail.pending_proxy_queues == {}

        # Head proxied background task must detect NACK and finish promptly instead of running 40 steps
        deadline = time.monotonic() + 1.5
        while head._proxied_inference_tasks and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

        assert len(head._proxied_inference_tasks) == 0
        assert not head.get_pipeline_turn_lock().locked()
        assert len(tail_decoded_steps) < 6

        # A follow-up request on the swarm succeeds immediately without waiting for 40 steps
        followup_tokens: list[str] = []
        async with aclosing(
            tail.stream_p2p_pipeline_tokens(prompt="Second request", max_tokens=2)
        ) as stream2:
            async for _req_id, _step, _tok_id, text, _eos in stream2:
                followup_tokens.append(text)
        assert len(followup_tokens) == 2
    finally:
        await _cleanup_node(tail)
        await _cleanup_node(head)


@pytest.mark.asyncio
async def test_heartbeat_nack_while_queued_cancels_proxied_task_before_acquiring_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a queued proxied request aborts, heartbeat NACK cancels it without acquiring the turn lock."""
    head = LocalNodeRuntime()
    try:
        _configure_sharded_node(
            head, node_id="head-queued", shard_start=0, shard_end=31, total_layers=32
        )
        assert head.model_runner is not None
        monkeypatch.setattr(
            type(head.model_runner), "has_native_llama", property(lambda _self: False)
        )
        head.proxy_heartbeat_interval_seconds = 0.05

        ran_forward = False
        orig_forward = head.model_runner.forward_pass

        def _spy_forward(*args: Any, **kwargs: Any) -> tuple[bytes, tuple[int, int]]:
            nonlocal ran_forward
            ran_forward = True
            return orig_forward(*args, **kwargs)

        head.model_runner.forward_pass = _spy_forward  # type: ignore[method-assign]

        async def _nack_heartbeat(*args: Any, **kwargs: Any) -> float:
            _ = args, kwargs
            raise RuntimeError("Peer rejected frame with NACK")

        head._send_to_endpoint = _nack_heartbeat  # type: ignore[method-assign]

        turn_lock = head.get_pipeline_turn_lock()
        await turn_lock.acquire()
        try:
            req_frame = InferenceRequestFrame(
                request_id=str(uuid.uuid4()),
                requester_ip="127.0.0.1",
                requester_port=49998,
                prompt="Queued then cancelled",
                messages=None,
                max_tokens=16,
                temperature=0.7,
                compress_activations=True,
            )
            await asyncio.wait_for(
                head._run_proxied_inference_task(req_frame, requester_peer_id="peer-q"),
                timeout=1.0,
            )
            assert ran_forward is False
        finally:
            turn_lock.release()
    finally:
        await _cleanup_node(head)


@pytest.mark.asyncio
async def test_stream_p2p_pipeline_tokens_runs_tokenization_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both `tokenize` and `tokenize_chat` in `stream_p2p_pipeline_tokens` run via `asyncio.to_thread`."""
    head = LocalNodeRuntime()
    try:
        _configure_sharded_node(
            head, node_id="head-tok", shard_start=0, shard_end=31, total_layers=32
        )
        assert head.model_runner is not None
        monkeypatch.setattr(
            type(head.model_runner), "has_native_llama", property(lambda _self: False)
        )

        to_thread_Targets: list[str] = []
        tokenize_threads: list[int] = []
        orig_to_thread = asyncio.to_thread
        main_tid = threading.get_ident()

        async def _spy_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
            to_thread_Targets.append(getattr(func, "__name__", repr(func)))
            return await orig_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(main_module.asyncio, "to_thread", _spy_to_thread)

        def _fake_tokenize(prompt: str) -> list[int]:
            _ = prompt
            tokenize_threads.append(threading.get_ident())
            return [101, 102]

        def _fake_tokenize_chat(messages: list[dict[str, Any]]) -> list[int]:
            _ = messages
            tokenize_threads.append(threading.get_ident())
            return [201, 202, 203]

        head.model_runner.tokenize = _fake_tokenize  # type: ignore[method-assign]
        head.model_runner.tokenize_chat = _fake_tokenize_chat  # type: ignore[method-assign]

        # 1. Prompt tokenization path (`messages=None`)
        async with aclosing(
            head.stream_p2p_pipeline_tokens(prompt="Hello thread", max_tokens=1)
        ) as s1:
            items1 = [item async for item in s1]
        assert len(items1) == 1
        assert "_fake_tokenize" in to_thread_Targets
        assert tokenize_threads[0] != main_tid

        # 2. Chat-template tokenization path (`messages` provided and `is_real_model` True)
        async with aclosing(
            head.stream_p2p_pipeline_tokens(
                prompt="fallback",
                messages=[{"role": "user", "content": "Hello chat thread"}],
                max_tokens=1,
            )
        ) as s2:
            items2 = [item async for item in s2]
        assert len(items2) == 1
        assert "_fake_tokenize_chat" in to_thread_Targets
        assert tokenize_threads[1] != main_tid
    finally:
        await _cleanup_node(head)
