"""Regression tests for Tier 1 adaptive P2P timeouts, FIFO turn queue, heartbeats,
BF16 wire tensors, and Tier 2 GGUF hardware acceleration / weight materialization.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import struct
import time
from typing import Any
import uuid

import httpx
import numpy as np
import pytest
import torch

from main import LocalNodeRuntime, app, runtime
from models.gguf_shard_loader import (
    GGML_TYPE_F32,
    GgufQuantizedLinear,
    load_gguf_pytorch_shard,
    select_gguf_torch_device,
)
from models.shard_runner import ShardModelRunner
from p2p.serializer import (
    AUTO_BF16_WIRE_MIN_BYTES,
    CompressionCodec,
    InferenceHeartbeatFrame,
    PacketType,
    deserialize_activation_tensor,
    deserialize_inference_heartbeat,
    serialize_activation_tensor,
    serialize_inference_heartbeat,
)
from p2p.transport import dispatch_p2p_frame
from shard_engine.schemas import NodeState, PeerNodeInfo, PeerRole
from tests._mock_runner import make_mock_runner


async def _reset_runtime_async(target: LocalNodeRuntime) -> None:
    """Cleanly reset a LocalNodeRuntime instance after async socket tests."""
    await target.cleanup_proxy_state()
    await target.stop_p2p_listener()
    target.session_token = None
    target.is_swarm_ready = False
    target.coverage_percent = 0.0
    target.peers = []
    target.pending_token_queues.clear()
    target.active_request_steps.clear()
    target._peer_step_ema_ms.clear()
    target._evicted_peers.clear()
    target._promoted_peer_ids.clear()
    target._queued_pipeline_requests = 0
    if target.model_runner is not None:
        target.model_runner.release()
        target.model_runner = None
    target.memory_manager.release_all()
    target.state = NodeState.STOPPED
    target.node_id = None


@pytest.fixture(autouse=True)
async def _reset_global_runtime_around_tests() -> Any:
    orig_keepalive = runtime.sse_keepalive_interval_seconds
    await _reset_runtime_async(runtime)
    yield
    runtime.sse_keepalive_interval_seconds = orig_keepalive
    await _reset_runtime_async(runtime)


def _write_minimal_f32_qwen2_gguf(
    path: Path,
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    vocab_size: int = 64,
) -> None:
    """Create a minimal valid float32 Qwen2 GGUF v3 file for device & budget tests."""
    torch.manual_seed(42)

    def _pstr(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _kv_str(k: str, v: str) -> bytes:
        return _pstr(k) + struct.pack("<I", 8) + _pstr(v)

    def _kv_u32(k: str, v: int) -> bytes:
        return _pstr(k) + struct.pack("<II", 4, v)

    def _kv_f32(k: str, v: float) -> bytes:
        return _pstr(k) + struct.pack("<If", 6, v)

    tokens = [f"t{i}" for i in range(vocab_size)]
    tok_kv = _pstr("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, len(tokens))
    tok_kv = tok_kv + b"".join(_pstr(t) for t in tokens)

    kvs = [
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", num_layers),
        _kv_u32("qwen2.embedding_length", hidden_size),
        _kv_u32("qwen2.feed_forward_length", hidden_size * 2),
        _kv_u32("qwen2.context_length", 128),
        _kv_u32("qwen2.attention.head_count", 4),
        _kv_u32("qwen2.attention.head_count_kv", 2),
        _kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_f32("qwen2.rope.freq_base", 10000.0),
        _kv_u32("tokenizer.ggml.bos_token_id", 1),
        _kv_u32("tokenizer.ggml.eos_token_id", 2),
        tok_kv,
    ]

    tensors: list[tuple[str, tuple[int, ...], bytes]] = []

    def _add(name: str, shape: tuple[int, ...], fill_ones: bool = False) -> None:
        t = (
            torch.ones(shape, dtype=torch.float32)
            if fill_ones
            else (torch.randn(shape, dtype=torch.float32) * 0.05)
        )
        tensors.append((name, shape, t.numpy().astype("<f4").tobytes()))

    _add("token_embd.weight", (vocab_size, hidden_size))
    _add("output_norm.weight", (hidden_size,), fill_ones=True)
    _add("output.weight", (vocab_size, hidden_size))
    for i in range(num_layers):
        p = f"blk.{i}"
        _add(f"{p}.attn_norm.weight", (hidden_size,), fill_ones=True)
        _add(f"{p}.ffn_norm.weight", (hidden_size,), fill_ones=True)
        _add(f"{p}.attn_q.weight", (hidden_size, hidden_size))
        _add(f"{p}.attn_k.weight", (hidden_size // 2, hidden_size))
        _add(f"{p}.attn_v.weight", (hidden_size // 2, hidden_size))
        _add(f"{p}.attn_output.weight", (hidden_size, hidden_size))
        _add(f"{p}.ffn_gate.weight", (hidden_size * 2, hidden_size))
        _add(f"{p}.ffn_up.weight", (hidden_size * 2, hidden_size))
        _add(f"{p}.ffn_down.weight", (hidden_size, hidden_size * 2))

    alignment = 32
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs))
    kv_blob = b"".join(kvs)
    t_hdr = bytearray()
    d_blob = bytearray()
    for name, shape_pt, raw_bytes in tensors:
        dims_gguf = tuple(reversed(shape_pt))
        rel_offset = len(d_blob)
        t_hdr.extend(_pstr(name))
        t_hdr.extend(struct.pack("<I", len(dims_gguf)))
        t_hdr.extend(struct.pack(f"<{len(dims_gguf)}Q", *dims_gguf))
        t_hdr.extend(struct.pack("<IQ", GGML_TYPE_F32, rel_offset))
        d_blob.extend(raw_bytes)
        pad = (-len(d_blob)) % alignment
        if pad:
            d_blob.extend(b"\x00" * pad)
    pre_len = len(header) + len(kv_blob) + len(t_hdr)
    hdr_pad = (-pre_len) % alignment
    path.write_bytes(
        header + kv_blob + bytes(t_hdr) + (b"\x00" * hdr_pad) + bytes(d_blob)
    )


def test_inference_heartbeat_frame_roundtrip_and_dispatch() -> None:
    """INFERENCE_HEARTBEAT serializes cleanly, validates headers, and dispatches safely."""
    req_id = str(uuid.uuid4())
    queued_bytes = serialize_inference_heartbeat(
        request_id=req_id, is_computing=False, progress_index=0
    )
    queued = deserialize_inference_heartbeat(queued_bytes)
    assert queued == InferenceHeartbeatFrame(
        request_id=req_id, is_computing=False, progress_index=0
    )

    computing_bytes = serialize_inference_heartbeat(
        request_id=req_id, is_computing=True, progress_index=42
    )
    computing = deserialize_inference_heartbeat(computing_bytes)
    assert computing == InferenceHeartbeatFrame(
        request_id=req_id, is_computing=True, progress_index=42
    )

    with pytest.raises(ValueError, match="size"):
        deserialize_inference_heartbeat(computing_bytes[:-1])

    bad_magic = b"NOPE" + computing_bytes[4:]
    with pytest.raises(ValueError, match="Invalid header"):
        deserialize_inference_heartbeat(bad_magic)

    bad_type = computing_bytes[:4] + bytes([int(PacketType.DECODED_TOKEN)]) + computing_bytes[5:]
    with pytest.raises(ValueError, match="Invalid header"):
        deserialize_inference_heartbeat(bad_type)


@pytest.mark.asyncio
async def test_dispatch_p2p_frame_handles_heartbeat_with_and_without_callback() -> None:
    """dispatch_p2p_frame tolerates missing heartbeat callback and invokes callback when set."""
    req_id = str(uuid.uuid4())
    hb_bytes = serialize_inference_heartbeat(req_id, is_computing=True, progress_index=7)

    async def _noop_act(_frame: Any) -> None:
        return None

    await dispatch_p2p_frame(
        frame_bytes=hb_bytes,
        on_activation_frame=_noop_act,
        on_inference_heartbeat=None,
    )

    seen: list[InferenceHeartbeatFrame] = []

    async def _on_hb(frame: InferenceHeartbeatFrame) -> None:
        seen.append(frame)

    await dispatch_p2p_frame(
        frame_bytes=hb_bytes,
        on_activation_frame=_noop_act,
        on_inference_heartbeat=_on_hb,
    )
    assert seen == [InferenceHeartbeatFrame(req_id, True, 7)]


def test_bfloat16_wire_tensor_roundtrip_halves_bytes_and_preserves_dynamic_range() -> None:
    """bfloat16 wire mode cuts uncompressed payload by 50% and preserves exponent range."""
    req_id = str(uuid.uuid4())
    values = np.linspace(-3.0, 3.0, 4096, dtype=np.float32)
    # Values well beyond float16 max (65504) must not overflow under bfloat16
    values[0] = 1e10
    values[1] = -1e10
    values[2] = 0.0
    raw_f32 = values.tobytes()
    assert len(raw_f32) == AUTO_BF16_WIRE_MIN_BYTES

    bf16_wire = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=(1, 4096),
        request_id=req_id,
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=15,
        total_layers=32,
        step_index=0,
        force_compression=False,
        use_bfloat16_wire=True,
    )
    f32_wire = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=(1, 4096),
        request_id=req_id,
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=15,
        total_layers=32,
        step_index=0,
        force_compression=False,
        use_bfloat16_wire=False,
    )
    assert len(bf16_wire) < len(f32_wire)
    assert (len(bf16_wire) - 60) == len(raw_f32) // 2

    decoded_bf16 = deserialize_activation_tensor(bf16_wire)
    assert decoded_bf16.wire_bfloat16 is True
    assert decoded_bf16.compression_used == CompressionCodec.NONE
    restored = np.frombuffer(decoded_bf16.raw_tensor_bytes, dtype=np.float32)
    assert np.isfinite(restored[0]) and abs(restored[0] - 1e10) / 1e10 < 1e-2
    assert np.isfinite(restored[1]) and abs(restored[1] + 1e10) / 1e10 < 1e-2
    assert restored[2] == 0.0
    assert np.allclose(restored, values, rtol=1e-2, atol=2e-2)

    decoded_f32 = deserialize_activation_tensor(f32_wire)
    assert decoded_f32.wire_bfloat16 is False
    assert decoded_f32.raw_tensor_bytes == raw_f32


def test_resolve_step_timeout_adapts_by_stage_and_preserves_small_test_overrides() -> None:
    """Prefill (step 0), decode, slow-peer EMA, and small test overrides behave cleanly."""
    node = LocalNodeRuntime()
    assert node.resolve_step_timeout(0, 1, "peer-a") == 180.0
    assert node.resolve_step_timeout(0, 2, "peer-a") == 90.0
    assert node.resolve_step_timeout(1, 1, "peer-a") == 90.0
    assert node.resolve_step_timeout(1, 2, "peer-a") == 45.0

    # Slow peer latency history raises decode timeout up to the 300s safety ceiling
    node.record_peer_step_latency("slow-peer", 40_000.0)
    assert node.resolve_step_timeout(1, 1, "slow-peer") == pytest.approx(135.0)
    node.record_peer_step_latency("huge-peer", 200_000.0)
    assert node.resolve_step_timeout(1, 1, "huge-peer") == 300.0

    # Explicit sub-30s unit-test timeouts must remain untouched
    node.failover_token_timeout_seconds = 0.25
    assert node.resolve_step_timeout(0, 2, "slow-peer") == 0.25
    node.single_peer_token_timeout_seconds = 0.4
    assert node.resolve_step_timeout(0, 1, "slow-peer") == 0.4


@pytest.mark.asyncio
async def test_fifo_turn_queue_and_heartbeats_prevent_multiuser_timeout() -> None:
    """Concurrent User A + User B serialize cleanly on Head Node and heartbeats reset timeout."""
    head = LocalNodeRuntime()
    joining = LocalNodeRuntime()

    head.state = NodeState.RUNNING
    head.node_id = "head-fifo"
    head.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    head.room_id = "room-fifo"
    head.shard_layer_start = 0
    head.shard_layer_end = 15
    head.total_layers = 32
    head.is_swarm_ready = True
    head.coverage_percent = 100.0
    head.proxy_heartbeat_interval_seconds = 0.06
    head.single_peer_token_timeout_seconds = 5.0

    joining.state = NodeState.RUNNING
    joining.node_id = "joining-tail"
    joining.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    joining.room_id = "room-fifo"
    joining.shard_layer_start = 16
    joining.shard_layer_end = 31
    joining.total_layers = 32
    joining.is_swarm_ready = True
    joining.coverage_percent = 100.0
    joining.single_peer_token_timeout_seconds = 0.35

    head_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=15, total_layers=32
    )
    active_turns = 0
    max_turns = 0
    orig_forward = head_runner.forward_pass

    def _slow_serialized_forward(
        input_data: Any,
        *,
        shape: tuple[int, int] | None = None,
        request_id: str = "default",
        step_index: int = 0,
    ) -> tuple[bytes, tuple[int, int]]:
        nonlocal active_turns, max_turns
        active_turns += 1
        max_turns = max(max_turns, active_turns)
        try:
            time.sleep(0.22)
            return orig_forward(
                input_data, shape=shape, request_id=request_id, step_index=step_index
            )
        finally:
            active_turns -= 1

    head_runner.forward_pass = _slow_serialized_forward  # type: ignore[method-assign]
    head.model_runner = head_runner
    joining.model_runner = make_mock_runner(
        shard_layer_start=16, shard_layer_end=31, total_layers=32
    )

    head_port = await head.start_p2p_listener(port=0)
    join_port = await joining.start_p2p_listener(port=0)
    head_peer = PeerNodeInfo(
        peer_id="head-fifo",
        ip_address="127.0.0.1",
        port=head_port,
        shard_layer_start=0,
        shard_layer_end=15,
        role=PeerRole.PRIMARY,
    )
    join_peer = PeerNodeInfo(
        peer_id="joining-tail",
        ip_address="127.0.0.1",
        port=join_port,
        shard_layer_start=16,
        shard_layer_end=31,
        role=PeerRole.PRIMARY,
    )
    head.peers = [join_peer]
    joining.peers = [head_peer]

    async def _run_head_user() -> list[str]:
        out: list[str] = []
        async for _rid, _s, _tid, text, _eos in head.stream_p2p_pipeline_tokens(
            prompt="User A long prompt", max_tokens=2
        ):
            out.append(text)
        return out

    async def _run_joining_user() -> list[str]:
        out: list[str] = []
        async for _rid, _s, _tid, text, _eos in joining.stream_p2p_pipeline_tokens(
            prompt="User B concurrent prompt", max_tokens=2
        ):
            out.append(text)
        return out

    try:
        t0 = time.monotonic()
        res_a, res_b = await asyncio.gather(_run_head_user(), _run_joining_user())
        elapsed = time.monotonic() - t0
        # Total ~0.9s (> joining 0.35s timeout), yet succeeds because heartbeats keep it alive
        assert elapsed > 0.40
        assert len(res_a) == 2
        assert len(res_b) == 2
        assert max_turns == 1

        # Verify genuinely stalled Head (no heartbeats at all) still times out promptly
        head.proxy_heartbeat_interval_seconds = 5.0
        head_runner.forward_pass = lambda *a, **kw: (time.sleep(0.55), orig_forward(*a, **kw))[1]  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            await _run_joining_user()
    finally:
        await _reset_runtime_async(head)
        await _reset_runtime_async(joining)


@pytest.mark.asyncio
async def test_sse_stream_emits_keepalive_comments_while_waiting_for_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI SSE endpoint emits `: keepalive` comments during slow prefill before token arrives."""
    runtime.state = NodeState.RUNNING
    runtime.node_id = "sse-keepalive-head"
    runtime.model_id = "meta-llama/Llama-3.1-8B-Instruct"
    runtime.room_id = "room-sse"
    runtime.session_token = "sk-shard-keepalive-test"
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32
    runtime.n_layers = 32
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.sse_keepalive_interval_seconds = 0.04
    runtime.model_runner = make_mock_runner(
        shard_layer_start=0, shard_layer_end=31, total_layers=32
    )

    async def _slow_stream(**_kwargs: Any) -> Any:
        await asyncio.sleep(0.14)
        yield ("req-keep", 0, 1, "hello", True)

    monkeypatch.setattr(runtime, "stream_p2p_pipeline_tokens", _slow_stream)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as ac:
        res = await ac.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {runtime.session_token}"},
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "max_tokens": 4,
            },
        )
    assert res.status_code == 200
    assert ": keepalive" in res.text
    assert '"hello"' in res.text
    assert "data: [DONE]" in res.text


def test_gguf_materialization_budget_and_device_acceleration(tmp_path: Path) -> None:
    """Materialization speeds up linear passes while obeying max_ram_mb & device selection."""
    gguf_path = tmp_path / "tiny-qwen2-accel.gguf"
    _write_minimal_f32_qwen2_gguf(gguf_path, num_layers=2, hidden_size=32, vocab_size=64)

    # 1. Tiny models (< 8 MB) default to CPU in 'auto' mode; explicit 'cpu' honors CPU
    assert select_gguf_torch_device("auto", kept_bytes=1024).type == "cpu"
    assert select_gguf_torch_device("cpu", kept_bytes=64 * 1024 * 1024).type == "cpu"

    # 2. Materialized FP32 and BF16 linear parity + CPU BF16 multi-token prefill & single-token decode
    raw_u8 = torch.empty(0, dtype=torch.uint8)
    w_ref = torch.randn(32, 32, dtype=torch.float32) * 0.1
    lin = GgufQuantizedLinear(
        raw_weight=torch.frombuffer(
            bytearray(w_ref.numpy().astype("<f4").tobytes()), dtype=torch.uint8
        ).clone(),
        out_features=32,
        in_features=32,
        ggml_type=GGML_TYPE_F32,
    )
    del raw_u8
    x_prefill = torch.randn(1, 8, 32, dtype=torch.float32)
    expected = torch.nn.functional.linear(x_prefill, w_ref)

    assert lin._cached_weight is None
    y_unmat = lin(x_prefill)
    torch.testing.assert_close(y_unmat, expected, atol=1e-5, rtol=1e-5)

    lin.materialize(torch.float32, torch.device("cpu"))
    assert lin._cached_weight is not None and lin.raw_weight.numel() == 0
    y_f32 = lin(x_prefill)
    torch.testing.assert_close(y_f32, expected, atol=1e-5, rtol=1e-5)

    lin.materialize(torch.bfloat16, torch.device("cpu"))
    y_bf16_prefill = lin(x_prefill)
    y_bf16_decode = lin(x_prefill[:, :1, :])
    torch.testing.assert_close(y_bf16_prefill, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y_bf16_decode, expected[:, :1, :], atol=2e-2, rtol=2e-2)

    # 3. Load shard on CPU with ample budget -> materializes all decoder linears
    shard_cpu = load_gguf_pytorch_shard(
        gguf_path, shard_layer_start=0, shard_layer_end=1, max_ram_mb=512, device="cpu"
    )
    assert shard_cpu.device.type == "cpu"
    assert shard_cpu.materialized_tensors >= 14
    if shard_cpu.tokenizer is not None:
        shard_cpu.tokenizer.close()

    # 4. Tight budget forces partial / un-materialized fallback without error and still runs forward
    shard_tight = load_gguf_pytorch_shard(
        gguf_path, shard_layer_start=0, shard_layer_end=1, max_ram_mb=0, device="cpu"
    )
    assert shard_tight.materialized_tensors == 0
    if shard_tight.tokenizer is not None:
        shard_tight.tokenizer.close()

    # 5. Hardware MPS execution check when running on Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        runner_cpu = ShardModelRunner(
            model_id="tiny-qwen2-accel",
            model_path=str(gguf_path),
            shard_layer_start=0,
            shard_layer_end=0,
            total_layers=2,
            max_ram_mb=512,
        )
        shard_mps = load_gguf_pytorch_shard(
            gguf_path, shard_layer_start=0, shard_layer_end=1, max_ram_mb=512, device="mps"
        )
        assert shard_mps.device.type == "mps"
        assert shard_mps.layers[0].self_attn.q_proj._cached_weight.device.type == "mps"  # type: ignore[union-attr]
        runner_mps = ShardModelRunner(
            shard_layer_start=0, shard_layer_end=1, total_layers=2
        )
        runner_mps._is_real_pytorch = True
        runner_mps._use_simple_kv_cache = True
        runner_mps._hf_config = shard_mps.hf_config
        runner_mps.hidden_dim = shard_mps.hidden_dim
        runner_mps._shard_container = shard_mps.container
        runner_mps._layers = shard_mps.layers
        runner_mps._rotary_emb = shard_mps.rotary_emb
        runner_mps._embed_tokens = shard_mps.embed_tokens
        runner_mps._norm = shard_mps.norm
        runner_mps._lm_head = shard_mps.lm_head
        runner_mps._compute_dtype = shard_mps.compute_dtype
        runner_mps._device = shard_mps.device
        out_bytes, out_shape = runner_mps.forward_pass([1, 3, 5], request_id="mps-test", step_index=0)
        assert out_shape == (3, 32)
        tok_id, _txt, _eos = runner_mps.decode_final_layer_to_token(
            out_bytes, step_index=0, max_tokens=4, request_id="mps-test", temperature=0.0
        )
        assert 0 <= tok_id < 64
        runner_cpu.release()
        runner_mps.release()
