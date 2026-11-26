"""Regression tests for Issue #63: proxy inference requests from non-Head peers to Head Node via P2P."""

from __future__ import annotations

import asyncio
import json
import os
import struct
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

from fastapi import HTTPException
import httpx
import pytest

from main import LocalNodeRuntime, app, runtime
from p2p.serializer import (
    INFERENCE_ERROR_HEADER_STRUCT,
    INFERENCE_REQUEST_HEADER_STRUCT,
    MAGIC_HEADER,
    MAX_INFERENCE_ERROR_BYTES,
    MAX_INFERENCE_REQUEST_PAYLOAD_BYTES,
    MAX_PROXIED_TOKENS,
    InferenceErrorFrame,
    InferenceRequestFrame,
    PacketType,
    deserialize_inference_error,
    deserialize_inference_request,
    serialize_inference_error,
    serialize_inference_request,
)
from p2p.transport import ACK_BYTE, FRAME_LEN_STRUCT, NACK_BYTE, receive_activations
from shard_engine.schemas import NetworkMode, NodeState, PeerNodeInfo, PeerRole
from tests._mock_runner import make_mock_runner


@pytest.fixture(autouse=True)
async def reset_global_runtime_async(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Ensure clean global runtime state before and after each test."""
    monkeypatch.setenv("SHARDAI_P2P_BIND_HOST", "127.0.0.1")
    await _reset_runtime(runtime)
    yield
    await _reset_runtime(runtime)


async def _reset_runtime(target: LocalNodeRuntime) -> None:
    """Reset runtime fields, P2P listener, proxy tasks, and runners."""
    await target.cleanup_proxy_state()
    await target.stop_p2p_listener()
    target.session_token = None
    target.is_swarm_ready = False
    target.coverage_percent = 0.0
    target.peers = []
    target.pending_token_queues.clear()
    target.active_request_steps.clear()
    target._evicted_peers.clear()
    target._promoted_peer_ids.clear()
    if target.model_runner is not None:
        target.model_runner.release()
        target.model_runner = None
    target.memory_manager.release_all()
    target.state = NodeState.STOPPED
    target.node_id = None
    target.model_id = None
    target.shard_layer_start = 0
    target.shard_layer_end = 15
    target.total_layers = 32
    target.p2p_listen_port = None


def _parse_sse_events(raw_body: str) -> tuple[list[dict[str, Any]], bool]:
    """Parse SSE `data:` lines into JSON dicts and report whether `data: [DONE]` appeared."""
    events: list[dict[str, Any]] = []
    saw_done = False
    for line in raw_body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        events.append(json.loads(payload))
    return events, saw_done


def _configure_node(
    node: LocalNodeRuntime,
    *,
    node_id: str,
    shard_start: int,
    shard_end: int,
    total_layers: int,
    peers: list[PeerNodeInfo] | None = None,
    session_token: str | None = None,
    is_swarm_ready: bool = True,
) -> None:
    """Configure a `LocalNodeRuntime` with a mock pipeline runner."""
    node.state = NodeState.RUNNING
    node.node_id = node_id
    node.model_id = "test/sharded-llama-28l"
    node.room_id = "room-63"
    node.session_token = session_token
    node.shard_layer_start = shard_start
    node.shard_layer_end = shard_end
    node.total_layers = total_layers
    node.n_layers = shard_end - shard_start + 1
    node.is_swarm_ready = is_swarm_ready
    node.coverage_percent = 100.0 if is_swarm_ready else 0.0
    node.peers = list(peers or [])
    node.model_runner = make_mock_runner(
        model_id=node.model_id,
        shard_layer_start=shard_start,
        shard_layer_end=shard_end,
        total_layers=total_layers,
        n_layers=node.n_layers,
    )


async def _send_raw_wire_frame(host: str, port: int, frame_bytes: bytes) -> bytes:
    """Send a raw length-prefixed binary frame and return the 1-byte ACK/NACK."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(FRAME_LEN_STRUCT.pack(len(frame_bytes)) + frame_bytes)
        await writer.drain()
        return await asyncio.wait_for(reader.readexactly(1), timeout=3.0)
    finally:
        writer.close()
        await writer.wait_closed()


# ==============================================================================
# 1. Binary Serializer Unit Tests
# ==============================================================================


def test_inference_request_frame_roundtrip_and_flags() -> None:
    """`InferenceRequestFrame` serializes and deserializes cleanly with messages and flags."""
    req_id = str(uuid.uuid4())
    messages = [{"role": "user", "content": "Hello Head Node!"}]
    packed = serialize_inference_request(
        request_id=req_id,
        requester_ip="127.0.0.1",
        requester_port=49201,
        prompt="[user] Hello Head Node!",
        messages=messages,
        max_tokens=128,
        temperature=0.65,
        compress_activations=False,
    )
    assert len(packed) > INFERENCE_REQUEST_HEADER_STRUCT.size
    frame = deserialize_inference_request(packed)
    assert isinstance(frame, InferenceRequestFrame)
    assert frame.request_id == req_id
    assert frame.requester_ip == "127.0.0.1"
    assert frame.requester_port == 49201
    assert frame.prompt == "[user] Hello Head Node!"
    assert frame.messages == messages
    assert frame.max_tokens == 128
    assert frame.temperature == pytest.approx(0.65)
    assert frame.compress_activations is False


def test_inference_request_crc_corruption_and_bounds_validation() -> None:
    """Corrupted CRC32, oversized payloads, invalid temperatures, and invalid max_tokens raise ValueError."""
    req_id = str(uuid.uuid4())
    packed = bytearray(
        serialize_inference_request(
            request_id=req_id,
            requester_ip="127.0.0.1",
            requester_port=49201,
            prompt="hello",
            messages=None,
            max_tokens=16,
            temperature=0.7,
        )
    )
    # Flip a byte inside the JSON payload
    packed[-1] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32"):
        deserialize_inference_request(bytes(packed))

    # Oversized payload rejected on serialize
    huge_prompt = "a" * (MAX_INFERENCE_REQUEST_PAYLOAD_BYTES + 100)
    with pytest.raises(ValueError, match="exceeds"):
        serialize_inference_request(
            request_id=req_id,
            requester_ip="127.0.0.1",
            requester_port=49201,
            prompt=huge_prompt,
        )

    # Oversized payload rejected on deserialize
    oversized_header = INFERENCE_REQUEST_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.INFERENCE_REQUEST),
        1,
        uuid.UUID(req_id).bytes,
        b"\x7f\x00\x00\x01",
        49201,
        16,
        0.7,
        0,
    )
    with pytest.raises(ValueError, match="exceeds"):
        deserialize_inference_request(
            oversized_header + (b"x" * (MAX_INFERENCE_REQUEST_PAYLOAD_BYTES + 1))
        )

    # Invalid temperature on serialize and deserialize
    for bad_temp in (-0.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="temperature"):
            serialize_inference_request(
                request_id=req_id,
                requester_ip="127.0.0.1",
                requester_port=49201,
                prompt="hi",
                temperature=bad_temp,
            )
        body = b'{"prompt":"hi","messages":null}'
        import zlib

        bad_header = INFERENCE_REQUEST_HEADER_STRUCT.pack(
            MAGIC_HEADER,
            int(PacketType.INFERENCE_REQUEST),
            1,
            uuid.UUID(req_id).bytes,
            b"\x7f\x00\x00\x01",
            49201,
            16,
            bad_temp,
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        with pytest.raises(ValueError, match="temperature"):
            deserialize_inference_request(bad_header + body)

    # Invalid max_tokens on serialize and deserialize
    for bad_tokens in (0, MAX_PROXIED_TOKENS + 1):
        with pytest.raises(ValueError, match="max_tokens"):
            serialize_inference_request(
                request_id=req_id,
                requester_ip="127.0.0.1",
                requester_port=49201,
                prompt="hi",
                max_tokens=bad_tokens,
            )
        body = b'{"prompt":"hi","messages":null}'
        import zlib

        bad_tok_header = INFERENCE_REQUEST_HEADER_STRUCT.pack(
            MAGIC_HEADER,
            int(PacketType.INFERENCE_REQUEST),
            1,
            uuid.UUID(req_id).bytes,
            b"\x7f\x00\x00\x01",
            49201,
            bad_tokens,
            0.7,
            zlib.crc32(body) & 0xFFFFFFFF,
        )
        with pytest.raises(ValueError, match="max_tokens"):
            deserialize_inference_request(bad_tok_header + body)

    # Non-dict JSON structure rejected
    import zlib

    bad_json_body = b'["not","a","dict"]'
    bad_json_header = INFERENCE_REQUEST_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.INFERENCE_REQUEST),
        1,
        uuid.UUID(req_id).bytes,
        b"\x7f\x00\x00\x01",
        49201,
        8,
        0.7,
        zlib.crc32(bad_json_body) & 0xFFFFFFFF,
    )
    with pytest.raises(ValueError, match="JSON object"):
        deserialize_inference_request(bad_json_header + bad_json_body)


def test_inference_error_frame_roundtrip_and_bounds() -> None:
    """`InferenceErrorFrame` round-trips cleanly and enforces size bounds."""
    req_id = str(uuid.uuid4())
    packed = serialize_inference_error(req_id, 422, "Unsupported GGUF sharding")
    assert len(packed) > INFERENCE_ERROR_HEADER_STRUCT.size
    frame = deserialize_inference_error(packed)
    assert isinstance(frame, InferenceErrorFrame)
    assert frame.request_id == req_id
    assert frame.status_code == 422
    assert frame.detail == "Unsupported GGUF sharding"

    # Oversized error detail is truncated safely on serialize
    long_detail = "E" * (MAX_INFERENCE_ERROR_BYTES + 500)
    capped_bytes = serialize_inference_error(req_id, 503, long_detail)
    capped_frame = deserialize_inference_error(capped_bytes)
    assert len(capped_frame.detail.encode("utf-8")) <= MAX_INFERENCE_ERROR_BYTES

    # Oversized raw error frame rejected on deserialize
    bad_err_header = struct.pack(
        "!4sBB16sH",
        MAGIC_HEADER,
        int(PacketType.INFERENCE_ERROR),
        0,
        uuid.UUID(req_id).bytes,
        500,
    )
    with pytest.raises(ValueError, match="exceeds"):
        deserialize_inference_error(
            bad_err_header + (b"x" * (MAX_INFERENCE_ERROR_BYTES + 1))
        )


# ==============================================================================
# 2. Security & Loop-Prevention Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_head_node_rejects_unauthorized_peer_with_nack() -> None:
    """Head Node must reject `INFERENCE_REQUEST` from unregistered endpoints with NACK."""
    head = LocalNodeRuntime()
    _configure_node(
        head,
        node_id="head-sec",
        shard_start=0,
        shard_end=7,
        total_layers=28,
        peers=[
            PeerNodeInfo(
                peer_id="legit-tail",
                ip_address="127.0.0.1",
                port=49888,
                shard_layer_start=8,
                shard_layer_end=27,
                role=PeerRole.PRIMARY,
            )
        ],
    )
    port = await head.start_p2p_listener(0, network_mode=NetworkMode.LAN)
    try:
        spoofed_req = serialize_inference_request(
            request_id=str(uuid.uuid4()),
            requester_ip="127.0.0.1",
            requester_port=49999,
            prompt="Spoofed",
            max_tokens=4,
        )
        ack = await _send_raw_wire_frame("127.0.0.1", port, spoofed_req)
        assert ack == NACK_BYTE
        assert len(head._proxied_inference_tasks) == 0
    finally:
        await _reset_runtime(head)


@pytest.mark.asyncio
async def test_non_head_node_rejects_incoming_inference_request_with_nack() -> None:
    """Non-Head nodes must NACK incoming `INFERENCE_REQUEST` frames to prevent forwarding loops."""
    tail = LocalNodeRuntime()
    _configure_node(
        tail,
        node_id="tail-no-loop",
        shard_start=8,
        shard_end=27,
        total_layers=28,
        peers=[
            PeerNodeInfo(
                peer_id="head-peer",
                ip_address="127.0.0.1",
                port=49880,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
            )
        ],
    )
    port = await tail.start_p2p_listener(0, network_mode=NetworkMode.LAN)
    try:
        loop_req = serialize_inference_request(
            request_id=str(uuid.uuid4()),
            requester_ip="127.0.0.1",
            requester_port=49880,
            prompt="Loop check",
            max_tokens=4,
        )
        ack = await _send_raw_wire_frame("127.0.0.1", port, loop_req)
        assert ack == NACK_BYTE
    finally:
        await _reset_runtime(tail)


@pytest.mark.asyncio
async def test_head_node_enforces_concurrent_proxy_task_limit() -> None:
    """Head Node rejects incoming proxied requests once 16 tasks are active."""
    head = LocalNodeRuntime()
    _configure_node(
        head,
        node_id="head-cap",
        shard_start=0,
        shard_end=7,
        total_layers=28,
        peers=[
            PeerNodeInfo(
                peer_id="tail-peer",
                ip_address="127.0.0.1",
                port=49881,
                shard_layer_start=8,
                shard_layer_end=27,
                role=PeerRole.PRIMARY,
            )
        ],
    )
    hold_event = asyncio.Event()

    async def _hold() -> None:
        await hold_event.wait()

    tasks = {asyncio.create_task(_hold()) for _ in range(16)}
    head._proxied_inference_tasks.update(tasks)
    try:
        frame = InferenceRequestFrame(
            request_id=str(uuid.uuid4()),
            requester_ip="127.0.0.1",
            requester_port=49881,
            prompt="hello",
            messages=None,
            max_tokens=4,
            temperature=0.7,
            compress_activations=True,
        )
        with pytest.raises(RuntimeError, match="maximum concurrent"):
            await head.handle_incoming_inference_request(frame)
    finally:
        hold_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await _reset_runtime(head)


# ==============================================================================
# 3. Live 2-Node TCP Integration & HTTP Endpoint Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_live_two_node_tcp_chat_proxy_streaming_non_streaming_and_generate() -> None:
    """Joining Tail (8..27) proxies chat/generate over TCP to Head (0..7) -> Tail -> Head -> Tail."""
    head = LocalNodeRuntime()
    try:
        _configure_node(
            head,
            node_id="head-node-0-7",
            shard_start=0,
            shard_end=7,
            total_layers=28,
        )
        _configure_node(
            runtime,
            node_id="tail-node-8-27",
            shard_start=8,
            shard_end=27,
            total_layers=28,
            session_token="sk-shard-issue63-live",
        )

        head_port = await head.start_p2p_listener(0, network_mode=NetworkMode.LAN)
        tail_port = await runtime.start_p2p_listener(0, network_mode=NetworkMode.LAN)

        head.peers = [
            PeerNodeInfo(
                peer_id="tail-node-8-27",
                ip_address="127.0.0.1",
                port=tail_port,
                shard_layer_start=8,
                shard_layer_end=27,
                role=PeerRole.PRIMARY,
            )
        ]
        runtime.peers = [
            PeerNodeInfo(
                peer_id="head-node-0-7",
                ip_address="127.0.0.1",
                port=head_port,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
            )
        ]

        headers = {"Authorization": "Bearer sk-shard-issue63-live"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8000"
        ) as async_client:
            # 1. Streaming /v1/chat/completions called directly on Joining Tail node
            stream_res = await async_client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test/sharded-llama-28l",
                    "messages": [{"role": "user", "content": "Hello from Tail"}],
                    "stream": True,
                    "max_tokens": 3,
                },
            )
            assert stream_res.status_code == 200
            sse_events, saw_done = _parse_sse_events(stream_res.text)
            assert saw_done is True
            streamed_text = "".join(
                evt["choices"][0]["delta"].get("content", "")
                for evt in sse_events
                if evt.get("choices")
            )
            assert streamed_text == "tok0tok1tok2"

            # 2. Non-streaming /v1/chat/completions called directly on Joining Tail node
            non_stream_res = await async_client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test/sharded-llama-28l",
                    "messages": [{"role": "user", "content": "Non-stream from Tail"}],
                    "stream": False,
                    "max_tokens": 2,
                },
            )
            assert non_stream_res.status_code == 200
            body = non_stream_res.json()
            assert body["choices"][0]["message"]["content"] == "tok0tok1"
            assert body["usage"]["completion_tokens"] == 2

            # 3. /generate SSE called directly on Joining Tail node
            gen_res = await async_client.post(
                "/generate",
                headers=headers,
                json={"prompt": "Generate via Head", "max_tokens": 2},
            )
            assert gen_res.status_code == 200
            gen_events, gen_done = _parse_sse_events(gen_res.text)
            assert gen_done is True
            assert [evt["token"] for evt in gen_events] == ["tok0", "tok1"]
            assert gen_events[-1]["is_eos"] is True

        assert runtime.pending_proxy_queues == {}
    finally:
        await _reset_runtime(head)


# ==============================================================================
# 4. Error Propagation, Missing Head Fallback, and Vision Guard Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_inference_error_frame_propagates_http_and_runtime_errors() -> None:
    """`InferenceErrorFrame` from Head Node propagates 4xx/503 HTTPException and 500 RuntimeError."""
    head = LocalNodeRuntime()
    try:
        # Configure Head (0..7 of 28) with a GGUF-like non-sharding runner to trigger 422 preflight error
        _configure_node(
            head,
            node_id="head-err",
            shard_start=0,
            shard_end=7,
            total_layers=28,
        )
        # Head does NOT have a downstream peer for layer 8 -> validate_pipeline_preflight raises 503
        head.peers = []

        _configure_node(
            runtime,
            node_id="tail-err",
            shard_start=8,
            shard_end=27,
            total_layers=28,
            session_token="sk-shard-issue63-err",
        )
        head_port = await head.start_p2p_listener(0, network_mode=NetworkMode.LAN)
        tail_port = await runtime.start_p2p_listener(0, network_mode=NetworkMode.LAN)

        # Register Tail in Head with shard_layer_start=0..7 so Tail is an authorized peer
        # endpoint while find_next_hop_peer(7) returns None (triggering HTTP 503 preflight).
        head.peers = [
            PeerNodeInfo(
                peer_id="tail-err",
                ip_address="127.0.0.1",
                port=tail_port,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
            )
        ]
        runtime.peers = [
            PeerNodeInfo(
                peer_id="head-err",
                ip_address="127.0.0.1",
                port=head_port,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
            )
        ]

        # 503 from Head preflight propagates as HTTPException(503) inside proxy stream
        with pytest.raises(HTTPException) as exc_info:
            async for _ in runtime.stream_p2p_pipeline_tokens(
                prompt="Should fail on Head",
                max_tokens=2,
            ):
                pass
        assert exc_info.value.status_code == 503
        assert "no downstream peer" in str(exc_info.value.detail)
        assert runtime.pending_proxy_queues == {}

        # Also verify direct 500 InferenceErrorFrame maps to RuntimeError
        async def _inject_500(
            req_frame: InferenceRequestFrame,
        ) -> None:
            await runtime.handle_incoming_inference_error(
                InferenceErrorFrame(
                    request_id=req_frame.request_id,
                    status_code=500,
                    detail="Simulated Head internal failure",
                )
            )

        err_server = await receive_activations(
            host="127.0.0.1",
            port=0,
            on_activation_frame=head.handle_incoming_activation,
            on_inference_request=_inject_500,
        )
        try:
            err_port = int(err_server.sockets[0].getsockname()[1])
            runtime.peers = [
                PeerNodeInfo(
                    peer_id="head-500",
                    ip_address="127.0.0.1",
                    port=err_port,
                    shard_layer_start=0,
                    shard_layer_end=7,
                    role=PeerRole.PRIMARY,
                )
            ]
            with pytest.raises(RuntimeError, match="Simulated Head internal failure"):
                async for _ in runtime.stream_p2p_pipeline_tokens(
                    prompt="Trigger 500",
                    max_tokens=2,
                ):
                    pass
            assert runtime.pending_proxy_queues == {}
        finally:
            err_server.close()
            await err_server.wait_closed()
    finally:
        await _reset_runtime(head)


@pytest.mark.asyncio
async def test_non_head_without_head_peer_returns_400_with_head_node_detail() -> None:
    """Without any registered Head Node peer, non-Head nodes return HTTP 400 mentioning 'Head Node'."""
    _configure_node(
        runtime,
        node_id="tail-orphan",
        shard_start=8,
        shard_end=27,
        total_layers=28,
        peers=[],
        session_token="sk-shard-issue63-nohead",
    )
    headers = {"Authorization": "Bearer sk-shard-issue63-nohead"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as async_client:
        for is_stream in (True, False):
            res = await async_client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test/sharded-llama-28l",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": is_stream,
                    "max_tokens": 4,
                },
            )
            assert res.status_code == 400
            assert "Head Node" in res.json()["detail"]

        gen_res = await async_client.post(
            "/generate",
            headers=headers,
            json={"prompt": "Hello", "max_tokens": 4},
        )
        assert gen_res.status_code == 400
        assert "Head Node" in gen_res.json()["detail"]


@pytest.mark.asyncio
async def test_vision_attachment_on_non_head_node_returns_400_vision_sharded_error() -> None:
    """Vision attachments on a non-Head node return HTTP 400 `vision_sharded_mode_not_supported`."""
    _configure_node(
        runtime,
        node_id="tail-vision",
        shard_start=8,
        shard_end=27,
        total_layers=28,
        peers=[
            PeerNodeInfo(
                peer_id="head-vision",
                ip_address="127.0.0.1",
                port=49899,
                shard_layer_start=0,
                shard_layer_end=7,
                role=PeerRole.PRIMARY,
            )
        ],
        session_token="sk-shard-issue63-vision",
    )
    headers = {"Authorization": "Bearer sk-shard-issue63-vision"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as async_client:
        res = await async_client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "test/sharded-llama-28l",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    }
                ],
                "stream": False,
                "max_tokens": 4,
            },
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "vision_sharded_mode_not_supported"


def test_find_head_node_peers_sorting_order() -> None:
    """`find_head_node_peers` orders PRIMARY before BACKUP and largest `shard_layer_end` first."""
    node = LocalNodeRuntime()
    node.peers = [
        PeerNodeInfo(
            peer_id="non-head",
            ip_address="127.0.0.1",
            port=49100,
            shard_layer_start=8,
            shard_layer_end=27,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="backup-wide",
            ip_address="127.0.0.1",
            port=49101,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.BACKUP,
        ),
        PeerNodeInfo(
            peer_id="primary-narrow",
            ip_address="127.0.0.1",
            port=49102,
            shard_layer_start=0,
            shard_layer_end=7,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="primary-wide",
            ip_address="127.0.0.1",
            port=49103,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
        ),
    ]
    ordered_ids = [p.peer_id for p in node.find_head_node_peers()]
    assert ordered_ids == ["primary-wide", "primary-narrow", "backup-wide"]
