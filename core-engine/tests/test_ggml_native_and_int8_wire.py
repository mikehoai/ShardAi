"""Tests for native libggml quantized GGUF inference, growing KV-cache, and INT8 wire activations."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import time
import uuid

import numpy as np
import pytest
import torch

from models.ggml_native import GGML_NATIVE_AVAILABLE, GgmlNativeEngine
from models.gguf_shard_loader import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q8_0,
    GgufQuantizedLinear,
    GgufSimpleKVCache,
    dequantize_ggml_bytes,
    load_gguf_pytorch_shard,
)
from models.shard_runner import ShardModelRunner
from p2p.serializer import (
    AUTO_INT8_WIRE_MIN_BYTES,
    CompressionCodec,
    deserialize_activation_tensor,
    float32_bytes_to_int8_wire,
    int8_wire_to_float32_bytes,
    serialize_activation_tensor,
)
from tests.test_gguf_sharded_pipeline import (
    _encode_q4_0,
    _encode_q4_k,
    _encode_q5_0,
    _encode_q8_0,
    _pack_gguf_kv,
    _pack_gguf_str,
)

pytestmark = pytest.mark.skipif(
    not GGML_NATIVE_AVAILABLE, reason="llama_cpp bundled libggml not available"
)


def _write_quantized_llama_gguf(
    path: Path,
    *,
    num_layers: int = 4,
    hidden_size: int = 256,
    intermediate_size: int = 256,
    vocab_size: int = 256,
    num_heads: int = 4,
    num_kv_heads: int = 4,
) -> None:
    """Create a valid quantized Llama GGUF (Q4_K + Q8_0) with RoPE permutation."""
    rng = np.random.default_rng(1234)
    tokens = [f"tok{i}" for i in range(vocab_size)]
    kvs = [
        _pack_gguf_kv("general.architecture", 8, "llama"),
        _pack_gguf_kv("llama.block_count", 4, num_layers),
        _pack_gguf_kv("llama.embedding_length", 4, hidden_size),
        _pack_gguf_kv("llama.feed_forward_length", 4, intermediate_size),
        _pack_gguf_kv("llama.context_length", 4, 256),
        _pack_gguf_kv("llama.attention.head_count", 4, num_heads),
        _pack_gguf_kv("llama.attention.head_count_kv", 4, num_kv_heads),
        _pack_gguf_kv("llama.attention.layer_norm_rms_epsilon", 6, 1e-6),
        _pack_gguf_kv("llama.rope.freq_base", 6, 10000.0),
        _pack_gguf_kv("tokenizer.ggml.bos_token_id", 4, 1),
        _pack_gguf_kv("tokenizer.ggml.eos_token_id", 4, 2),
        _pack_gguf_kv("tokenizer.ggml.tokens", 9, tokens),
    ]

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def _add_f32(name: str, shape: tuple[int, ...], fill_ones: bool = False) -> None:
        arr = (
            np.ones(shape, dtype=np.float32)
            if fill_ones
            else (rng.standard_normal(shape).astype(np.float32) * 0.05)
        )
        tensors.append((name, shape, GGML_TYPE_F32, arr.astype("<f4").tobytes()))

    def _add_q8(name: str, shape: tuple[int, int]) -> None:
        arr = rng.standard_normal(shape).astype(np.float32) * 0.1
        raw, _ = _encode_q8_0(arr)
        tensors.append((name, shape, GGML_TYPE_Q8_0, raw))

    def _add_q4k(name: str, shape: tuple[int, int]) -> None:
        arr = rng.standard_normal(shape).astype(np.float32) * 0.1
        raw, _ = _encode_q4_k(arr)
        tensors.append((name, shape, GGML_TYPE_Q4_K, raw))

    _add_q8("token_embd.weight", (vocab_size, hidden_size))
    _add_f32("output_norm.weight", (hidden_size,), fill_ones=True)
    _add_q8("output.weight", (vocab_size, hidden_size))

    for i in range(num_layers):
        p = f"blk.{i}"
        _add_f32(f"{p}.attn_norm.weight", (hidden_size,), fill_ones=True)
        _add_f32(f"{p}.ffn_norm.weight", (hidden_size,), fill_ones=True)
        _add_q4k(f"{p}.attn_q.weight", (hidden_size, hidden_size))
        _add_q8(f"{p}.attn_k.weight", (hidden_size, hidden_size))
        _add_q8(f"{p}.attn_v.weight", (hidden_size, hidden_size))
        _add_q4k(f"{p}.attn_output.weight", (hidden_size, hidden_size))
        _add_q4k(f"{p}.ffn_gate.weight", (intermediate_size, hidden_size))
        _add_q4k(f"{p}.ffn_up.weight", (intermediate_size, hidden_size))
        _add_q4k(f"{p}.ffn_down.weight", (hidden_size, intermediate_size))

    alignment = 32
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs))
    kv_blob = b"".join(kvs)
    t_hdr = bytearray()
    d_blob = bytearray()
    for name, shape_pt, ggml_type, raw_bytes in tensors:
        dims_gguf = tuple(reversed(shape_pt))
        rel_offset = len(d_blob)
        t_hdr.extend(_pack_gguf_str(name))
        t_hdr.extend(struct.pack("<I", len(dims_gguf)))
        t_hdr.extend(struct.pack(f"<{len(dims_gguf)}Q", *dims_gguf))
        t_hdr.extend(struct.pack("<IQ", ggml_type, rel_offset))
        d_blob.extend(raw_bytes)
        pad = (-len(d_blob)) % alignment
        if pad:
            d_blob.extend(b"\x00" * pad)
    pre_len = len(header) + len(kv_blob) + len(t_hdr)
    hdr_pad = (-pre_len) % alignment
    path.write_bytes(
        header + kv_blob + bytes(t_hdr) + (b"\x00" * hdr_pad) + bytes(d_blob)
    )


@pytest.mark.parametrize(
    ("ggml_type", "encoder"),
    [
        (GGML_TYPE_Q4_0, _encode_q4_0),
        (GGML_TYPE_Q5_0, _encode_q5_0),
        (GGML_TYPE_Q8_0, _encode_q8_0),
        (GGML_TYPE_Q4_K, _encode_q4_k),
    ],
)
def test_ggml_native_linear_parity_decode_prefill_bias_and_unpermute(
    ggml_type: int,
    encoder: object,
) -> None:
    """Native libggml linear matches PyTorch dequantization on seq=1, seq=8, bias, and RoPE unpermute."""
    rng = np.random.default_rng(77 + ggml_type)
    out_features, in_features = 128, 256
    w_src = (rng.standard_normal((out_features, in_features)).astype(np.float32)) * 0.2
    raw_bytes, _ = encoder(w_src)  # type: ignore[operator]
    raw_u8 = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy())
    bias = torch.randn(out_features, dtype=torch.float32) * 0.05

    engine = GgmlNativeEngine(preferred="cpu")
    try:
        for unpermute_heads in (None, 4):
            lin_native = GgufQuantizedLinear(
                raw_weight=raw_u8.clone(),
                out_features=out_features,
                in_features=in_features,
                ggml_type=ggml_type,
                bias=bias.clone(),
                unpermute_heads=unpermute_heads,
            )
            lin_ref = GgufQuantizedLinear(
                raw_weight=raw_u8.clone(),
                out_features=out_features,
                in_features=in_features,
                ggml_type=ggml_type,
                bias=bias.clone(),
                unpermute_heads=unpermute_heads,
            )
            assert lin_native.attach_ggml_engine(engine) is True

            # Both decode (seq=1, cached graph) and prefill (seq=8)
            for seq_len in (1, 8):
                x = torch.randn(1, seq_len, in_features, dtype=torch.float32)
                y_native = lin_native(x)
                y_ref = lin_ref(x)
                # libggml quantizes activations on-the-fly (Q8_0/Q8_K) for SIMD integer dot products
                cos = torch.nn.functional.cosine_similarity(y_native, y_ref, dim=-1)
                assert float(cos.min()) > 0.999
                torch.testing.assert_close(y_native, y_ref, atol=0.15, rtol=0.05)
    finally:
        engine.close()
        assert engine.is_closed is True


def test_quantized_gguf_two_node_pipeline_uses_zero_copy_ggml_and_matches_single_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2-node sharded Q4_K/Q8_0 GGUF runs via native libggml with zero RAM expansion and exact parity."""
    monkeypatch.setenv("SHARDAI_GGML_BACKEND", "cpu")
    gguf_path = tmp_path / "sharded-llama-q4k.gguf"
    _write_quantized_llama_gguf(gguf_path, num_layers=4, hidden_size=256, intermediate_size=256)

    # Full single-node PyTorch fallback reference (native GGML disabled)
    monkeypatch.setenv("SHARDAI_GGML_BACKEND", "off")
    ref_runner = ShardModelRunner(
        model_id="test/quantized-llama",
        shard_layer_start=0,
        shard_layer_end=3,
        total_layers=4,
        max_ram_mb=512,
    )
    # Force pure PyTorch sharded path for reference
    ref_shard = load_gguf_pytorch_shard(
        gguf_path, shard_layer_start=0, shard_layer_end=3, max_ram_mb=512, device="cpu"
    )
    assert ref_shard.native_ggml_tensors == 0
    ref_runner._is_real_pytorch = True
    ref_runner._use_simple_kv_cache = True
    ref_runner._hf_config = ref_shard.hf_config
    ref_runner.hidden_dim = ref_shard.hidden_dim
    ref_runner._shard_container = ref_shard.container
    ref_runner._layers = ref_shard.layers
    ref_runner._rotary_emb = ref_shard.rotary_emb
    ref_runner._embed_tokens = ref_shard.embed_tokens
    ref_runner._norm = ref_shard.norm
    ref_runner._lm_head = ref_shard.lm_head
    ref_runner._compute_dtype = ref_shard.compute_dtype
    ref_runner._device = ref_shard.device

    # Sharded 2-node runners with native libggml enabled
    monkeypatch.setenv("SHARDAI_GGML_BACKEND", "cpu")
    head_runner = ShardModelRunner(
        model_id="test/quantized-llama",
        model_path=str(gguf_path),
        shard_layer_start=0,
        shard_layer_end=1,
        total_layers=4,
        max_ram_mb=512,
    )
    tail_runner = ShardModelRunner(
        model_id="test/quantized-llama",
        model_path=str(gguf_path),
        shard_layer_start=2,
        shard_layer_end=3,
        total_layers=4,
        max_ram_mb=512,
    )
    try:
        assert head_runner._ggml_engine is not None
        assert tail_runner._ggml_engine is not None
        # Zero materialization for quantized weights (no 3.5x-7x RAM blowup)
        assert head_runner._layers[0].self_attn.q_proj._cached_weight is None
        assert tail_runner._lm_head._cached_weight is None
        assert tail_runner._lm_head._ggml_handle is not None

        prompt_ids = [1, 5, 9, 13]
        for step in range(4):
            inp = prompt_ids if step == 0 else [sharded_tid]  # noqa: F821
            ref_h, ref_shape = ref_runner.forward_pass(inp, request_id="ref", step_index=step)
            inp_s = prompt_ids if step == 0 else [sharded_tid]
            h_mid, shape_mid = head_runner.forward_pass(inp_s, request_id="shd", step_index=step)
            h_end, end_shape = tail_runner.forward_pass(
                h_mid, shape=shape_mid, request_id="shd", step_index=step
            )
            assert end_shape == ref_shape
            ref_t = torch.frombuffer(bytearray(ref_h), dtype=torch.float32).reshape(-1)
            shd_t = torch.frombuffer(bytearray(h_end), dtype=torch.float32).reshape(-1)
            assert float(torch.nn.functional.cosine_similarity(shd_t, ref_t, dim=0)) > 0.999

            ref_tid, _, _ = ref_runner.decode_final_layer_to_token(
                ref_h, step_index=step, max_tokens=8, request_id="ref", temperature=0.0
            )
            sharded_tid, _, _ = tail_runner.decode_final_layer_to_token(
                h_end, step_index=step, max_tokens=8, request_id="shd", temperature=0.0
            )
            assert sharded_tid == ref_tid
    finally:
        head_engine = head_runner._ggml_engine
        tail_engine = tail_runner._ggml_engine
        head_runner.release()
        tail_runner.release()
        ref_shard.close()
        ref_runner.release()
        assert head_engine is not None and head_engine.is_closed is True
        assert tail_engine is not None and tail_engine.is_closed is True


def test_growing_kv_cache_matches_concat_across_300_steps() -> None:
    """Preallocated GgufSimpleKVCache grows seamlessly beyond initial capacity without altering tensors."""
    cache = GgufSimpleKVCache()
    ref_k_list: list[torch.Tensor] = []
    ref_v_list: list[torch.Tensor] = []

    # Prefill 12 tokens + 300 decode tokens (exceeds initial 256 capacity)
    k0 = torch.randn(1, 2, 12, 16, dtype=torch.float32)
    v0 = torch.randn(1, 2, 12, 16, dtype=torch.float32)
    ref_k_list.append(k0)
    ref_v_list.append(v0)
    k_out, v_out = cache.update(k0, v0, layer_idx=3)
    assert cache.get_seq_length(3) == 12
    torch.testing.assert_close(k_out, k0)
    torch.testing.assert_close(v_out, v0)

    for step in range(300):
        ks = torch.randn(1, 2, 1, 16, dtype=torch.float32)
        vs = torch.randn(1, 2, 1, 16, dtype=torch.float32)
        ref_k_list.append(ks)
        ref_v_list.append(vs)
        k_out, v_out = cache.update(ks, vs, layer_idx=3)
        assert cache.get_seq_length(3) == 13 + step

    torch.testing.assert_close(k_out, torch.cat(ref_k_list, dim=-2))
    torch.testing.assert_close(v_out, torch.cat(ref_v_list, dim=-2))
    cache.clear()
    assert cache.get_seq_length(3) == 0


def test_int8_wire_quantization_roundtrip_bandwidth_and_bounds() -> None:
    """INT8 wire quantization reduces payload ~4x vs FP32 (~2x vs BF16) with tight numerical bounds."""
    rng = np.random.default_rng(99)
    shape = (32, 2048)  # 262,144 bytes FP32
    mat = rng.standard_normal(shape).astype(np.float32) * 3.5
    mat[0, :] = 0.0  # verify all-zero row stability
    raw_f32 = mat.astype("<f4").tobytes()
    assert len(raw_f32) > AUTO_INT8_WIRE_MIN_BYTES

    packed_i8 = float32_bytes_to_int8_wire(raw_f32, shape)
    assert len(packed_i8) == (shape[0] * 4) + (shape[0] * shape[1])
    unpacked_f32 = int8_wire_to_float32_bytes(packed_i8, shape)
    reconstructed = np.frombuffer(unpacked_f32, dtype="<f4").reshape(shape)
    assert np.all(reconstructed[0] == 0.0)
    assert float(np.max(np.abs(reconstructed - mat))) < 0.15

    # Full P2P activation frame round-trip
    frame_fp32 = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=7,
        total_layers=32,
        step_index=0,
        force_compression=False,
    )
    frame_bf16 = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=7,
        total_layers=32,
        step_index=0,
        force_compression=False,
        use_bfloat16_wire=True,
    )
    frame_int8 = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=7,
        total_layers=32,
        step_index=0,
        force_compression=False,
        use_int8_wire=True,
    )
    assert len(frame_int8) < len(frame_bf16) * 0.55
    assert len(frame_int8) < len(frame_fp32) * 0.28

    decoded = deserialize_activation_tensor(frame_int8)
    assert decoded.wire_int8 is True
    assert decoded.wire_bfloat16 is False
    assert decoded.compression_used in (CompressionCodec.NONE, CompressionCodec.LZ4)
    assert decoded.shape == shape

    # Small activation (< 32 KB threshold) skips LZ4/Zlib when force_compression=False
    small_shape = (1, 1024)
    small_f32 = rng.standard_normal(small_shape).astype(np.float32).astype("<f4").tobytes()
    small_frame = serialize_activation_tensor(
        tensor_data=small_f32,
        shape=small_shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=0,
        total_layers=4,
        step_index=0,
        force_compression=False,
        use_int8_wire=True,
    )
    small_decoded = deserialize_activation_tensor(small_frame)
    assert small_decoded.wire_int8 is True
    assert small_decoded.compression_used == CompressionCodec.NONE
    assert small_decoded.shape == small_shape

    with pytest.raises(ValueError, match="does not match shape"):
        int8_wire_to_float32_bytes(b"\x00" * 10, shape)


def test_native_ggml_q4_k_dramatically_faster_than_pytorch_fallback() -> None:
    """Verify native libggml Q4_K single-token decode beats PyTorch fallback by >5x."""
    out_f, in_f = 4096, 2048
    n_blocks = (out_f * in_f) // 256
    raw_u8 = torch.zeros(n_blocks * 144, dtype=torch.uint8)
    raw_u8[::144] = 0x00
    raw_u8[1::144] = 0x38
    raw_u8[4::144] = 1
    x = torch.randn(1, 1, in_f, dtype=torch.float32)

    lin_pt = GgufQuantizedLinear(raw_u8.clone(), out_f, in_f, GGML_TYPE_Q4_K)
    t0 = time.perf_counter()
    _ = lin_pt(x)
    pt_ms = (time.perf_counter() - t0) * 1000.0

    engine = GgmlNativeEngine(preferred="cpu")
    try:
        lin_ggml = GgufQuantizedLinear(raw_u8.clone(), out_f, in_f, GGML_TYPE_Q4_K)
        assert lin_ggml.attach_ggml_engine(engine) is True
        _ = lin_ggml(x)  # warmup cached decode graph
        t1 = time.perf_counter()
        _ = lin_ggml(x)
        ggml_ms = (time.perf_counter() - t1) * 1000.0
        assert ggml_ms * 5.0 < pt_ms, f"Expected >5x speedup, got pt={pt_ms:.2f}ms ggml={ggml_ms:.2f}ms"
    finally:
        engine.close()
