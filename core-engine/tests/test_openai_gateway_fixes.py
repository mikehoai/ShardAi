"""Regression tests for Issue #6: OpenAI Gateway SSE error handling, /generate auth & readiness, and remote Tail temperature propagation."""

from __future__ import annotations

from array import array
import asyncio
import json
import math
import os
from pathlib import Path
import socket
from typing import Any
import uuid
import zlib

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

from fastapi.testclient import TestClient
import pytest

import main as main_module
from main import LocalNodeRuntime, app, runtime
from models.shard_runner import ShardModelRunner
from p2p.serializer import (
    ACTIVATION_HEADER_STRUCT,
    DEFAULT_TEMPERATURE,
    LEGACY_ACTIVATION_HEADER_STRUCT,
    MAGIC_HEADER,
    CompressionCodec,
    DecodedTokenFrame,
    PacketType,
    deserialize_activation_tensor,
    serialize_activation_tensor,
)
from shard_engine.schemas import NodeState, PeerNodeInfo, PeerRole

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_global_runtime() -> Any:
    """Guarantee every test starts and ends with clean runtime state and zero allocated RAM."""
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.peers = []
    runtime.pending_token_queues.clear()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    yield
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.peers = []
    runtime.pending_token_queues.clear()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None


def _configure_running_node(
    *,
    shard_start: int = 0,
    shard_end: int = 31,
    total_layers: int = 32,
    is_swarm_ready: bool = True,
    coverage_percent: float = 100.0,
    peers: list[PeerNodeInfo] | None = None,
) -> dict[str, str]:
    """Configure global runtime into a deterministic RUNNING state and return Bearer auth headers."""
    token = "sk-shard-issue6-test-token"
    runtime.state = NodeState.RUNNING
    runtime.node_id = "node-issue6-test"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "issue6-room"
    runtime.session_token = token
    runtime.shard_layer_start = shard_start
    runtime.shard_layer_end = shard_end
    runtime.total_layers = total_layers
    runtime.n_layers = shard_end - shard_start + 1
    runtime.is_swarm_ready = is_swarm_ready
    runtime.coverage_percent = coverage_percent
    runtime.peers = list(peers or [])
    runtime.model_runner = ShardModelRunner(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        shard_layer_start=shard_start,
        shard_layer_end=shard_end,
        total_layers=total_layers,
        n_layers=shard_end - shard_start + 1,
    )
    return {"Authorization": f"Bearer {token}"}


def _parse_sse_events(raw_body: str) -> tuple[list[dict[str, Any]], bool]:
    """Parse SSE `data:` lines into JSON dicts and report whether `data: [DONE]` was present."""
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


# ==============================================================================
# Bug 1: SSE Pre-Flight Validation & Mid-Stream Error Handling
# ==============================================================================


def test_preflight_rejects_non_head_node_with_400_before_sse_commit() -> None:
    """Non-Head Nodes (shard_layer_start != 0) must return HTTP 400 before HTTP 200 SSE."""
    headers = _configure_running_node(shard_start=16, shard_end=31, total_layers=32)

    stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert stream_res.status_code == 400
    assert "Head Node" in stream_res.json()["detail"]

    non_stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 4,
        },
    )
    assert non_stream_res.status_code == 400
    assert "Head Node" in non_stream_res.json()["detail"]

    gen_res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 4},
    )
    assert gen_res.status_code == 400
    assert "Head Node" in gen_res.json()["detail"]


def test_preflight_rejects_missing_downstream_peer_with_503_before_sse_commit() -> None:
    """Partial Head Nodes without a registered downstream peer must return HTTP 503 pre-flight."""
    headers = _configure_running_node(
        shard_start=0,
        shard_end=15,
        total_layers=32,
        is_swarm_ready=True,
        peers=[],
    )

    stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert stream_res.status_code == 503
    assert "no downstream peer" in stream_res.json()["detail"]

    non_stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 4,
        },
    )
    assert non_stream_res.status_code == 503
    assert "no downstream peer" in non_stream_res.json()["detail"]

    gen_res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 4},
    )
    assert gen_res.status_code == 503
    assert "no downstream peer" in gen_res.json()["detail"]


def test_openai_sse_emits_structured_error_and_done_on_mid_stream_peer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-stream downstream peer failure must emit a JSON SSE error chunk + [DONE] without crashing."""
    tail_peer = PeerNodeInfo(
        peer_id="tail-peer-1",
        ip_address="127.0.0.1",
        port=49991,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
    )
    headers = _configure_running_node(
        shard_start=0,
        shard_end=15,
        total_layers=32,
        is_swarm_ready=True,
        peers=[tail_peer],
    )

    call_count = 0

    async def _flaky_send_activations(
        peer_ip: str,
        byte_stream: bytes,
        peer_port: int = 49152,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        nonlocal call_count
        call_count += 1
        frame = deserialize_activation_tensor(byte_stream)
        if call_count == 1:
            # First token succeeds so SSE is genuinely mid-stream
            await runtime.handle_incoming_decoded_token(
                DecodedTokenFrame(
                    request_id=frame.request_id,
                    step_index=frame.step_index,
                    token_id=101,
                    token_text="Hello",
                    is_eos=False,
                )
            )
            return 1.0
        raise TimeoutError("Downstream peer timed out at step 1")

    monkeypatch.setattr(main_module, "send_activations", _flaky_send_activations)

    res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Stream test"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    events, saw_done = _parse_sse_events(res.text)
    assert saw_done is True
    # Must contain initial assistant role chunk, first token chunk, and final structured error chunk
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(
        "choices" in ev and ev["choices"][0]["delta"].get("content") == "Hello"
        for ev in events
    )
    error_events = [ev for ev in events if "error" in ev]
    assert len(error_events) == 1
    assert error_events[0]["error"]["code"] == "pipeline_stream_error"
    assert "All primary and backup peers failed" in error_events[0]["error"]["message"]
    # Must NOT emit finish_reason: "stop" after failure
    assert not any(
        "choices" in ev and ev["choices"][0].get("finish_reason") == "stop"
        for ev in events
    )
    # Request queue must be cleaned up cleanly
    assert len(runtime.pending_token_queues) == 0


def test_generate_sse_emits_structured_error_and_done_on_mid_stream_peer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /generate must catch mid-stream RuntimeError and emit structured error + [DONE]."""
    tail_peer = PeerNodeInfo(
        peer_id="tail-peer-gen",
        ip_address="127.0.0.1",
        port=49992,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
    )
    headers = _configure_running_node(
        shard_start=0,
        shard_end=15,
        total_layers=32,
        is_swarm_ready=True,
        peers=[tail_peer],
    )

    async def _failing_send_activations(
        peer_ip: str,
        byte_stream: bytes,
        peer_port: int = 49152,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        raise OSError("Connection reset by peer")

    monkeypatch.setattr(main_module, "send_activations", _failing_send_activations)

    res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Mid-stream fail", "max_tokens": 3},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    events, saw_done = _parse_sse_events(res.text)
    assert saw_done is True
    assert len(events) == 1
    assert "error" in events[0]
    assert events[0]["error"]["code"] == "pipeline_stream_error"
    assert "All primary and backup peers failed" in events[0]["error"]["message"]
    assert len(runtime.pending_token_queues) == 0


def test_non_streaming_chat_completions_returns_503_on_downstream_peer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-streaming /v1/chat/completions must return HTTP 503 JSON error instead of HTTP 500."""
    tail_peer = PeerNodeInfo(
        peer_id="tail-peer-nonstream",
        ip_address="127.0.0.1",
        port=49993,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
    )
    headers = _configure_running_node(
        shard_start=0,
        shard_end=15,
        total_layers=32,
        is_swarm_ready=True,
        peers=[tail_peer],
    )

    async def _failing_send_activations(
        peer_ip: str,
        byte_stream: bytes,
        peer_port: int = 49152,
        timeout_seconds: float = 5.0,
        ack_timeout_seconds: float | None = None,
    ) -> float:
        raise TimeoutError("Tail peer unreachable")

    monkeypatch.setattr(main_module, "send_activations", _failing_send_activations)

    res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Non-streaming fail"}],
            "stream": False,
            "max_tokens": 3,
        },
    )
    assert res.status_code == 503
    body = res.json()
    assert body["error"]["code"] == "pipeline_peer_failure"
    assert body["error"]["type"] == "service_unavailable_error"
    assert len(runtime.pending_token_queues) == 0


# ==============================================================================
# Bug 2: POST /generate Bearer Session Token Auth & is_swarm_ready 503 Guard
# ==============================================================================


def test_generate_enforces_stopped_503_auth_401_and_swarm_ready_503() -> None:
    """POST /generate must enforce stopped 503, Bearer token 401, and is_swarm_ready 503."""
    # 1. Node STOPPED -> 503
    stopped_res = client.post("/generate", json={"prompt": "Hello", "max_tokens": 2})
    assert stopped_res.status_code == 503

    # 2. Node RUNNING with coverage < 100% (is_swarm_ready == False)
    headers = _configure_running_node(
        shard_start=0,
        shard_end=15,
        total_layers=32,
        is_swarm_ready=False,
        coverage_percent=50.0,
    )

    # Missing Authorization header -> 401
    no_auth_res = client.post("/generate", json={"prompt": "Hello", "max_tokens": 2})
    assert no_auth_res.status_code == 401
    assert no_auth_res.headers.get("www-authenticate") == "Bearer"

    # Malformed Authorization header -> 401
    malformed_res = client.post(
        "/generate",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"prompt": "Hello", "max_tokens": 2},
    )
    assert malformed_res.status_code == 401

    # Wrong Bearer token -> 401
    wrong_token_res = client.post(
        "/generate",
        headers={"Authorization": "Bearer sk-shard-invalid"},
        json={"prompt": "Hello", "max_tokens": 2},
    )
    assert wrong_token_res.status_code == 401

    # Valid Bearer token but is_swarm_ready == False -> 503 swarm_insufficient_ram
    not_ready_res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 2},
    )
    assert not_ready_res.status_code == 503
    body = not_ready_res.json()
    assert body["error"]["code"] == "swarm_insufficient_ram"
    assert "50%" in body["error"]["message"]

    # 3. Full layers + is_swarm_ready == True + valid Bearer token -> 200 SSE
    ready_headers = _configure_running_node(
        shard_start=0,
        shard_end=31,
        total_layers=32,
        is_swarm_ready=True,
        coverage_percent=100.0,
    )
    ok_res = client.post(
        "/generate",
        headers=ready_headers,
        json={"prompt": "Hello ready swarm", "max_tokens": 3, "temperature": 0.0},
    )
    assert ok_res.status_code == 200
    assert "text/event-stream" in ok_res.headers["content-type"]
    events, saw_done = _parse_sse_events(ok_res.text)
    assert saw_done is True
    assert len(events) >= 1
    assert "token" in events[0]


# ==============================================================================
# Bug 3: Binary Temperature Propagation & Remote Tail Node Decoding
# ==============================================================================


def test_binary_activation_serializer_temperature_roundtrip_and_legacy_compat() -> None:
    """Binary ACTIVATION_TENSOR frames must preserve exact temperature and stay v1-compatible."""
    raw_bytes = array("f", [0.25, -0.5, 1.0, 2.5] * 16).tobytes()
    req_id = str(uuid.uuid4())

    for expected_temp in (0.0, 0.25, 0.7, 1.35, 2.0):
        frame_bytes = serialize_activation_tensor(
            tensor_data=raw_bytes,
            shape=(4, 16),
            request_id=req_id,
            origin_ip="127.0.0.1",
            origin_port=49152,
            current_layer=15,
            total_layers=32,
            step_index=2,
            max_tokens=16,
            force_compression=True,
            temperature=expected_temp,
        )
        unpacked = deserialize_activation_tensor(frame_bytes)
        assert unpacked.temperature == expected_temp
        assert unpacked.raw_tensor_bytes == raw_bytes
        assert unpacked.request_id == req_id

    # Invalid temperatures must be rejected on serialize
    for bad_temp in (-0.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="Invalid sampling temperature"):
            serialize_activation_tensor(
                tensor_data=raw_bytes,
                shape=(4, 16),
                request_id=req_id,
                origin_ip="127.0.0.1",
                origin_port=49152,
                current_layer=15,
                total_layers=32,
                step_index=0,
                temperature=bad_temp,
            )

    # Corrupted payload must still fail CRC32 check
    valid_v2 = bytearray(
        serialize_activation_tensor(
            tensor_data=raw_bytes,
            shape=(4, 16),
            request_id=req_id,
            origin_ip="127.0.0.1",
            origin_port=49152,
            current_layer=15,
            total_layers=32,
            step_index=0,
            temperature=0.0,
        )
    )
    valid_v2[-1] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        deserialize_activation_tensor(bytes(valid_v2))

    # Tampered NaN temperature in an otherwise valid v2 frame must be rejected
    # Build v2 header with NaN temperature
    nan_header = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(CompressionCodec.NONE),
        uuid.UUID(req_id).bytes,
        socket.inet_aton("127.0.0.1"),
        49152,
        15,
        32,
        0,
        16,
        4,
        16,
        len(raw_bytes),
        zlib.crc32(raw_bytes) & 0xFFFFFFFF,
        float("nan"),
    )
    with pytest.raises(ValueError, match="Invalid sampling temperature"):
        deserialize_activation_tensor(nan_header + raw_bytes)

    # Legacy 52-byte v1 frame (without float64 temperature) must deserialize with DEFAULT_TEMPERATURE
    legacy_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
    legacy_header = LEGACY_ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(CompressionCodec.NONE),
        uuid.UUID(req_id).bytes,
        socket.inet_aton("127.0.0.1"),
        49152,
        15,
        32,
        1,
        16,
        4,
        16,
        len(raw_bytes),
        legacy_crc,
    )
    legacy_unpacked = deserialize_activation_tensor(legacy_header + raw_bytes)
    assert math.isclose(legacy_unpacked.temperature, DEFAULT_TEMPERATURE)
    assert legacy_unpacked.raw_tensor_bytes == raw_bytes
    assert legacy_unpacked.step_index == 1


@pytest.mark.asyncio
async def test_three_node_head_middle_tail_propagates_temperature_to_tail_decoder() -> None:
    """Head -> Middle -> Tail P2P runtime pipeline must preserve client temperature to Tail decode."""
    head = LocalNodeRuntime()
    middle = LocalNodeRuntime()
    tail = LocalNodeRuntime()

    head.state = NodeState.RUNNING
    head.shard_layer_start = 0
    head.shard_layer_end = 9
    head.total_layers = 32
    head.model_runner = ShardModelRunner(
        shard_layer_start=0, shard_layer_end=9, total_layers=32, n_layers=10
    )

    middle.state = NodeState.RUNNING
    middle.shard_layer_start = 10
    middle.shard_layer_end = 19
    middle.total_layers = 32
    middle.model_runner = ShardModelRunner(
        shard_layer_start=10, shard_layer_end=19, total_layers=32, n_layers=10
    )

    tail.state = NodeState.RUNNING
    tail.shard_layer_start = 20
    tail.shard_layer_end = 31
    tail.total_layers = 32
    tail.model_runner = ShardModelRunner(
        shard_layer_start=20, shard_layer_end=31, total_layers=32, n_layers=12
    )

    observed_tail_temperatures: list[float] = []
    original_decode = tail.model_runner.decode_final_layer_to_token

    def _spy_decode(
        hidden_state_bytes: bytes,
        step_index: int,
        max_tokens: int = 16,
        request_id: str = "default",
        temperature: float = 0.7,
    ) -> tuple[int, str, bool]:
        observed_tail_temperatures.append(temperature)
        return original_decode(
            hidden_state_bytes,
            step_index=step_index,
            max_tokens=max_tokens,
            request_id=request_id,
            temperature=temperature,
        )

    tail.model_runner.decode_final_layer_to_token = _spy_decode  # type: ignore[method-assign]

    head_port = await head.start_p2p_listener(port=0)
    middle_port = await middle.start_p2p_listener(port=0)
    tail_port = await tail.start_p2p_listener(port=0)
    head.p2p_listen_port = head_port
    middle.p2p_listen_port = middle_port
    tail.p2p_listen_port = tail_port

    head.peers = [
        PeerNodeInfo(
            peer_id="middle-node",
            ip_address="127.0.0.1",
            port=middle_port,
            shard_layer_start=10,
            shard_layer_end=19,
            role=PeerRole.PRIMARY,
        )
    ]
    middle.peers = [
        PeerNodeInfo(
            peer_id="tail-node",
            ip_address="127.0.0.1",
            port=tail_port,
            shard_layer_start=20,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        )
    ]

    try:
        # 1. Deterministic temperature=0.0 across 2 tokens
        tokens_t0: list[str] = []
        async for _rid, _step, _tid, text, _eos in head.stream_p2p_pipeline_tokens(
            prompt="Deterministic test",
            max_tokens=2,
            temperature=0.0,
        ):
            tokens_t0.append(text)
        assert len(tokens_t0) == 2
        assert observed_tail_temperatures == [0.0, 0.0]

        # 2. Custom temperature=1.45 across 2 tokens
        observed_tail_temperatures.clear()
        tokens_t145: list[str] = []
        async for _rid, _step, _tid, text, _eos in head.stream_p2p_pipeline_tokens(
            prompt="Creative test",
            max_tokens=2,
            temperature=1.45,
        ):
            tokens_t145.append(text)
        assert len(tokens_t145) == 2
        assert observed_tail_temperatures == [1.45, 1.45]
        assert head_port > 0
    finally:
        await head.stop_p2p_listener()
        await middle.stop_p2p_listener()
        await tail.stop_p2p_listener()
        head.model_runner.release()
        middle.model_runner.release()
        tail.model_runner.release()


@pytest.mark.asyncio
async def test_two_node_real_pytorch_runtime_propagates_temperature_zero(
    tmp_path: Path,
) -> None:
    """Two real PyTorch LocalNodeRuntimes must produce deterministic greedy argmax tokens when temperature=0.0."""
    import torch
    from safetensors.torch import save_model
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(99)
    full_model = AutoModelForCausalLM.from_config(cfg).eval()
    cfg.save_pretrained(str(tmp_path))
    save_model(full_model, str(tmp_path / "model.safetensors"))

    head = LocalNodeRuntime()
    tail = LocalNodeRuntime()
    head.state = NodeState.RUNNING
    head.shard_layer_start = 0
    head.shard_layer_end = 1
    head.total_layers = 4
    head.model_runner = ShardModelRunner(
        model_id="test/tiny-qwen2-temp",
        model_path=str(tmp_path),
        shard_layer_start=0,
        shard_layer_end=1,
        total_layers=4,
        max_ram_mb=2048,
    )

    tail.state = NodeState.RUNNING
    tail.shard_layer_start = 2
    tail.shard_layer_end = 3
    tail.total_layers = 4
    tail.model_runner = ShardModelRunner(
        model_id="test/tiny-qwen2-temp",
        model_path=str(tmp_path),
        shard_layer_start=2,
        shard_layer_end=3,
        total_layers=4,
        max_ram_mb=2048,
    )

    assert head.model_runner.is_real_model is True
    assert tail.model_runner.is_real_model is True

    prompt_text = "deterministic greedy test"
    prompt_ids = [3, 11, 27, 42]
    head.model_runner.tokenize = lambda _prompt: list(prompt_ids)  # type: ignore[method-assign]
    expected_tokens: list[int] = []
    with torch.inference_mode():
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long)
        for _ in range(4):
            logits = full_model(cur_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            expected_tokens.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)

    head_port = await head.start_p2p_listener(port=0)
    tail_port = await tail.start_p2p_listener(port=0)
    head.p2p_listen_port = head_port
    tail.p2p_listen_port = tail_port
    head.peers = [
        PeerNodeInfo(
            peer_id="pytorch-tail",
            ip_address="127.0.0.1",
            port=tail_port,
            shard_layer_start=2,
            shard_layer_end=3,
            role=PeerRole.PRIMARY,
        )
    ]

    try:
        # Run twice with temperature=0.0 and confirm both runs match unsharded greedy argmax exactly
        for _ in range(2):
            produced_ids: list[int] = []
            async for _rid, _step, tok_id, _text, _eos in head.stream_p2p_pipeline_tokens(
                prompt=prompt_text,
                max_tokens=4,
                temperature=0.0,
            ):
                produced_ids.append(tok_id)
            assert produced_ids == expected_tokens
    finally:
        await head.stop_p2p_listener()
        await tail.stop_p2p_listener()
        head.model_runner.release()
        tail.model_runner.release()
