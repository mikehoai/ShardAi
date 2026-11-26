"""Comprehensive tests for Issue #60: multi-node GGUF quantized PyTorch pipeline sharding."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import Any
from unittest.mock import AsyncMock

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import httpx
import numpy as np
import pytest
import torch  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config  # type: ignore[import-not-found]

import main as main_module
from main import LocalNodeRuntime, app, runtime, start_node, stop_node
from models.gguf_shard_loader import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GgufQuantizedEmbedding,
    GgufQuantizedLinear,
    _reverse_permute_rope_rows,
    can_load_gguf_pytorch_shard,
    dequantize_ggml_bytes,
    load_gguf_pytorch_shard,
    read_gguf_header_and_tensors,
)
from models.shard_runner import ShardModelRunner
from shard_engine.schemas import NetworkMode, NodeState, PeerNodeInfo, PeerRole, StartNodeRequest
from shard_engine.tracker_client import SignalingTrackerClient


# ---------------------------------------------------------------------------
# GGUF Test Writer & Block Encoders
# ---------------------------------------------------------------------------


def _pack_gguf_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_gguf_kv(key: str, val_type: int, value: Any) -> bytes:
    out = _pack_gguf_str(key) + struct.pack("<I", val_type)
    if val_type == 4:  # uint32
        return out + struct.pack("<I", int(value))
    if val_type == 5:  # int32
        return out + struct.pack("<i", int(value))
    if val_type == 6:  # float32
        return out + struct.pack("<f", float(value))
    if val_type == 7:  # bool
        return out + struct.pack("<?", bool(value))
    if val_type == 8:  # string
        return out + _pack_gguf_str(str(value))
    if val_type == 9:  # array of strings
        items = list(value)
        body = bytearray(struct.pack("<IQ", 8, len(items)))
        for item in items:
            body.extend(_pack_gguf_str(str(item)))
        return out + bytes(body)
    raise ValueError(f"Unsupported test KV type: {val_type}")


def _encode_q8_0(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize a 1D/2D float32 array to Q8_0 (block=32, 34B) and return (bytes, ref_f32)."""
    flat = arr.astype(np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.where(max_abs > 0, max_abs / 127.0, 0.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    quants = np.where(
        scales_f32 > 0, np.clip(np.round(flat / scales_f32), -127, 127), 0
    ).astype(np.int8)
    blocks = bytearray()
    for i in range(flat.shape[0]):
        blocks.extend(scales[i].tobytes())
        blocks.extend(quants[i].tobytes())
    ref = (scales_f32 * quants.astype(np.float32)).reshape(arr.shape)
    return bytes(blocks), ref


def _encode_q4_0(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize float32 array to Q4_0 (block=32, 18B) and return (bytes, ref_f32)."""
    flat = arr.astype(np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.where(max_abs > 0, max_abs / 7.0, 0.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    q = np.where(
        scales_f32 > 0, np.clip(np.round(flat / scales_f32) + 8, 0, 15), 8
    ).astype(np.uint8)
    packed = q[:, :16] | (q[:, 16:] << 4)
    blocks = bytearray()
    for i in range(flat.shape[0]):
        blocks.extend(scales[i].tobytes())
        blocks.extend(packed[i].tobytes())
    ref = (scales_f32 * (q.astype(np.float32) - 8.0)).reshape(arr.shape)
    return bytes(blocks), ref


def _encode_q5_0(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize float32 array to Q5_0 (block=32, 22B) and return (bytes, ref_f32)."""
    flat = arr.astype(np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    scales = np.where(max_abs > 0, max_abs / 15.0, 0.0).astype(np.float16)
    scales_f32 = scales.astype(np.float32)
    q = np.where(
        scales_f32 > 0, np.clip(np.round(flat / scales_f32) + 16, 0, 31), 16
    ).astype(np.uint32)
    packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)
    blocks = bytearray()
    for i in range(flat.shape[0]):
        qh = 0
        for bit in range(32):
            if int(q[i, bit]) & 0x10:
                qh |= 1 << bit
        blocks.extend(scales[i].tobytes())
        blocks.extend(struct.pack("<I", qh))
        blocks.extend(packed[i].astype(np.uint8).tobytes())
    ref = (scales_f32 * (q.astype(np.float32) - 16.0)).reshape(arr.shape)
    return bytes(blocks), ref


def _encode_q4_k(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Encode float32 array into Q4_K (block=256, 144B) with exact reference values."""
    flat = arr.astype(np.float32).reshape(-1, 256)
    blocks = bytearray()
    refs = np.zeros_like(flat, dtype=np.float32)
    for b in range(flat.shape[0]):
        row = flat[b].reshape(8, 32)
        d = np.float16(0.25)
        dmin = np.float16(0.1)
        d_f = float(d)
        dmin_f = float(dmin)
        scales_u6 = np.full(8, 4, dtype=np.uint8)
        mins_u6 = np.full(8, 2, dtype=np.uint8)
        scales_bytes = bytearray(12)
        for j in range(4):
            scales_bytes[j] = int(scales_u6[j]) & 63
            scales_bytes[j + 4] = int(mins_u6[j]) & 63
            scales_bytes[j + 8] = (int(scales_u6[j + 4]) & 0x0F) | ((int(mins_u6[j + 4]) & 0x0F) << 4)
            scales_bytes[j] |= (int(scales_u6[j + 4]) >> 4) << 6
            scales_bytes[j + 4] |= (int(mins_u6[j + 4]) >> 4) << 6
        q_vals = np.zeros((8, 32), dtype=np.uint8)
        for g in range(8):
            sc = d_f * float(scales_u6[g])
            mn = dmin_f * float(mins_u6[g])
            q_vals[g] = np.clip(np.round((row[g] + mn) / sc), 0, 15).astype(np.uint8)
            refs[b, g * 32 : (g + 1) * 32] = sc * q_vals[g].astype(np.float32) - mn
        qs = bytearray()
        for pair in range(4):
            lo = q_vals[2 * pair]
            hi = q_vals[2 * pair + 1]
            qs.extend((lo | (hi << 4)).astype(np.uint8).tobytes())
        blocks.extend(d.tobytes())
        blocks.extend(dmin.tobytes())
        blocks.extend(scales_bytes)
        blocks.extend(qs)
    return bytes(blocks), refs.reshape(arr.shape)


def _encode_q6_k(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Encode float32 array into Q6_K (block=256, 210B) with exact reference values."""
    flat = arr.astype(np.float32).reshape(-1, 256)
    blocks = bytearray()
    refs = np.zeros_like(flat, dtype=np.float32)
    for b in range(flat.shape[0]):
        d = np.float16(0.125)
        d_f = float(d)
        scales_i8 = np.full(16, 3, dtype=np.int8)
        row = flat[b].reshape(16, 16)
        q_vals = np.zeros((16, 16), dtype=np.int32)
        for g in range(16):
            sc = d_f * float(scales_i8[g])
            q_vals[g] = np.clip(np.round(row[g] / sc) + 32, 0, 63).astype(np.int32)
            refs[b, g * 16 : (g + 1) * 16] = sc * (q_vals[g] - 32).astype(np.float32)
        q_flat = q_vals.reshape(2, 4, 32)  # matches (low/high bit layout)
        # Pack ql (128 bytes: two 64B halves, low nibbles = shift0, high nibbles = shift4)
        ql = bytearray()
        qh = bytearray()
        for half in range(2):
            g0 = q_flat[half, 0]
            g1 = q_flat[half, 1]
            g2 = q_flat[half, 2]
            g3 = q_flat[half, 3]
            ql.extend(((g0 & 0x0F) | ((g2 & 0x0F) << 4)).astype(np.uint8).tobytes())
            ql.extend(((g1 & 0x0F) | ((g3 & 0x0F) << 4)).astype(np.uint8).tobytes())
            qh.extend(
                (
                    ((g0 >> 4) & 3)
                    | (((g1 >> 4) & 3) << 2)
                    | (((g2 >> 4) & 3) << 4)
                    | (((g3 >> 4) & 3) << 6)
                )
                .astype(np.uint8)
                .tobytes()
            )
        blocks.extend(ql)
        blocks.extend(qh)
        blocks.extend(scales_i8.tobytes())
        blocks.extend(d.tobytes())
    return bytes(blocks), refs.reshape(arr.shape)


def _write_gguf_file(
    path: Path,
    *,
    kvs: list[tuple[str, int, Any]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    alignment: int = 32,
) -> None:
    """Write a valid GGUF v3 binary file."""
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(kvs))
    kv_blob = bytearray()
    for key, vtype, val in kvs:
        kv_blob.extend(_pack_gguf_kv(key, vtype, val))

    tensor_hdr = bytearray()
    data_blob = bytearray()
    for name, shape_pt, ggml_type, raw_bytes in tensors:
        dims_gguf = tuple(reversed(shape_pt))
        rel_offset = len(data_blob)
        tensor_hdr.extend(_pack_gguf_str(name))
        tensor_hdr.extend(struct.pack("<I", len(dims_gguf)))
        tensor_hdr.extend(struct.pack(f"<{len(dims_gguf)}Q", *dims_gguf))
        tensor_hdr.extend(struct.pack("<IQ", ggml_type, rel_offset))
        data_blob.extend(raw_bytes)
        pad = (-len(data_blob)) % alignment
        if pad:
            data_blob.extend(b"\x00" * pad)

    pre_data_len = len(header) + len(kv_blob) + len(tensor_hdr)
    hdr_pad = (-pre_data_len) % alignment
    with path.open("wb") as f:
        f.write(header)
        f.write(kv_blob)
        f.write(tensor_hdr)
        if hdr_pad:
            f.write(b"\x00" * hdr_pad)
        f.write(data_blob)


def _build_tiny_qwen2_gguf(
    path: Path,
    *,
    seed: int = 17,
    mixed_quants: bool = True,
) -> tuple[Qwen2Config, Any]:
    """Create a tiny 4-layer Qwen2 GGUF and an unsharded HF reference model with exact dequantized weights."""
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
    )
    ref_model = AutoModelForCausalLM.from_config(cfg).eval()
    # Keep tiny weights in clean magnitude range for accurate quantization
    with torch.no_grad():
        for name, param in ref_model.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                param.mul_(0.3)

    vocab_tokens = [f"<tok_{i}>" for i in range(256)]
    vocab_tokens[0] = "<unk>"
    vocab_tokens[1] = "<s>"
    vocab_tokens[2] = "<|im_end|>"
    vocab_tokens[3] = "Hello"
    vocab_tokens[4] = " "
    vocab_tokens[5] = "world"

    kvs: list[tuple[str, int, Any]] = [
        ("general.architecture", 8, "qwen2"),
        ("general.alignment", 4, 32),
        ("qwen2.block_count", 4, cfg.num_hidden_layers),
        ("qwen2.embedding_length", 4, cfg.hidden_size),
        ("qwen2.feed_forward_length", 4, cfg.intermediate_size),
        ("qwen2.context_length", 4, cfg.max_position_embeddings),
        ("qwen2.attention.head_count", 4, cfg.num_attention_heads),
        ("qwen2.attention.head_count_kv", 4, cfg.num_key_value_heads),
        ("qwen2.attention.key_length", 4, cfg.head_dim),
        ("qwen2.attention.layer_norm_rms_epsilon", 6, float(cfg.rms_norm_eps)),
        ("qwen2.rope.freq_base", 6, 1000000.0),
        ("tokenizer.ggml.tokens", 9, vocab_tokens),
        ("tokenizer.ggml.bos_token_id", 4, 1),
        ("tokenizer.ggml.eos_token_id", 4, 2),
        ("tokenizer.ggml.add_bos_token", 7, False),
    ]

    def _quantize_for(name: str, t: torch.Tensor) -> tuple[int, bytes, torch.Tensor]:
        arr = t.detach().cpu().to(torch.float32).numpy()
        if not mixed_quants or t.ndim == 1:
            return GGML_TYPE_F32, arr.astype("<f4").tobytes(), torch.from_numpy(arr)
        if "embed_tokens" in name:
            raw, ref = _encode_q5_0(arr)
            return GGML_TYPE_Q5_0, raw, torch.from_numpy(ref)
        if "lm_head" in name or "v_proj" in name:
            raw, ref = _encode_q8_0(arr)
            return GGML_TYPE_Q8_0, raw, torch.from_numpy(ref)
        if "down_proj" in name:
            raw, ref = _encode_q6_k(arr)
            return GGML_TYPE_Q6_K, raw, torch.from_numpy(ref)
        if "gate_proj" in name or "up_proj" in name:
            raw, ref = _encode_q4_k(arr)
            return GGML_TYPE_Q4_K, raw, torch.from_numpy(ref)
        if "q_proj" in name:
            raw, ref = _encode_q4_0(arr)
            return GGML_TYPE_Q4_0, raw, torch.from_numpy(ref)
        raw, ref = _encode_q5_0(arr)
        return GGML_TYPE_Q5_0, raw, torch.from_numpy(ref)

    hf_to_gguf = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }
    gguf_tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    updated_state: dict[str, torch.Tensor] = {}

    for hf_name, tensor in ref_model.state_dict().items():
        if hf_name in hf_to_gguf:
            g_name = hf_to_gguf[hf_name]
        else:
            parts = hf_name.split(".", 3)
            idx = parts[2]
            leaf = parts[3]
            leaf_map = {
                "input_layernorm.weight": "attn_norm.weight",
                "post_attention_layernorm.weight": "ffn_norm.weight",
                "self_attn.q_proj.weight": "attn_q.weight",
                "self_attn.q_proj.bias": "attn_q.bias",
                "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.k_proj.bias": "attn_k.bias",
                "self_attn.v_proj.weight": "attn_v.weight",
                "self_attn.v_proj.bias": "attn_v.bias",
                "self_attn.o_proj.weight": "attn_output.weight",
                "mlp.gate_proj.weight": "ffn_gate.weight",
                "mlp.up_proj.weight": "ffn_up.weight",
                "mlp.down_proj.weight": "ffn_down.weight",
            }
            g_name = f"blk.{idx}.{leaf_map[leaf]}"

        gtype, raw, ref_t = _quantize_for(hf_name, tensor)
        gguf_tensors.append((g_name, tuple(tensor.shape), gtype, raw))
        updated_state[hf_name] = ref_t

    ref_model.load_state_dict(updated_state)
    _write_gguf_file(path, kvs=kvs, tensors=gguf_tensors)
    return cfg, ref_model


# ---------------------------------------------------------------------------
# 1. Dequantization Accuracy Tests for All 13 GGML Types
# ---------------------------------------------------------------------------


def test_dequantize_all_supported_ggml_types() -> None:
    """Verify vectorized dequantization accuracy for F32, F16, BF16, Q4_0..Q6_K."""
    rng = np.random.default_rng(42)

    # F32
    x_f32 = rng.standard_normal((2, 16), dtype=np.float32)
    out = dequantize_ggml_bytes(x_f32.tobytes(), (2, 16), GGML_TYPE_F32)
    torch.testing.assert_close(out, torch.from_numpy(x_f32), atol=1e-6, rtol=1e-6)

    # F16
    x_f16 = x_f32.astype(np.float16)
    out = dequantize_ggml_bytes(x_f16.tobytes(), (2, 16), GGML_TYPE_F16)
    torch.testing.assert_close(out, torch.from_numpy(x_f16.astype(np.float32)), atol=1e-4, rtol=1e-4)

    # BF16
    x_bf16 = torch.from_numpy(x_f32).to(torch.bfloat16)
    raw_bf16 = bytes(x_bf16.view(torch.uint16).numpy().tobytes())
    out = dequantize_ggml_bytes(raw_bf16, (2, 16), GGML_TYPE_BF16)
    torch.testing.assert_close(out, x_bf16.to(torch.float32), atol=1e-4, rtol=1e-4)

    # Q4_0
    x32 = rng.uniform(-1.0, 1.0, size=(2, 64)).astype(np.float32)
    raw, ref = _encode_q4_0(x32)
    out = dequantize_ggml_bytes(raw, (2, 64), GGML_TYPE_Q4_0)
    torch.testing.assert_close(out, torch.from_numpy(ref), atol=1e-4, rtol=1e-4)

    # Q4_1: f16 d, f16 m, 16B qs (low nibbles = 0..15, high nibbles = 16..31)
    d = np.float16(0.5)
    m = np.float16(-1.25)
    qs_lo = np.arange(16, dtype=np.uint8)
    qs_hi = (15 - np.arange(16, dtype=np.uint8))
    packed_41 = qs_lo | (qs_hi << 4)
    raw_41 = d.tobytes() + m.tobytes() + packed_41.tobytes()
    ref_41 = float(d) * np.concatenate([qs_lo, qs_hi]).astype(np.float32) + float(m)
    out_41 = dequantize_ggml_bytes(raw_41, (1, 32), GGML_TYPE_Q4_1)
    torch.testing.assert_close(out_41, torch.from_numpy(ref_41.reshape(1, 32)), atol=1e-4, rtol=1e-4)

    # Q5_0
    raw, ref = _encode_q5_0(x32)
    out = dequantize_ggml_bytes(raw, (2, 64), GGML_TYPE_Q5_0)
    torch.testing.assert_close(out, torch.from_numpy(ref), atol=1e-4, rtol=1e-4)

    # Q5_1: f16 d, f16 m, uint32 qh, 16B qs
    qh_51 = 0xAAAA5555
    raw_51 = d.tobytes() + m.tobytes() + struct.pack("<I", qh_51) + packed_41.tobytes()
    bits_51 = np.array([(qh_51 >> b) & 1 for b in range(32)], dtype=np.float32)
    nib_51 = np.concatenate([qs_lo, qs_hi]).astype(np.float32)
    ref_51 = float(d) * (nib_51 + 16.0 * bits_51) + float(m)
    out_51 = dequantize_ggml_bytes(raw_51, (1, 32), GGML_TYPE_Q5_1)
    torch.testing.assert_close(out_51, torch.from_numpy(ref_51.reshape(1, 32)), atol=1e-4, rtol=1e-4)

    # Q8_0
    raw, ref = _encode_q8_0(x32)
    out = dequantize_ggml_bytes(raw, (2, 64), GGML_TYPE_Q8_0)
    torch.testing.assert_close(out, torch.from_numpy(ref), atol=1e-4, rtol=1e-4)

    # Q2_K (block=256, 84B: 16B scales + 64B qs + f16 d + f16 dmin)
    scales_2k = bytes([(g % 7) | (((g + 1) % 5) << 4) for g in range(16)])
    qs_2k = bytes([(i * 37) & 0xFF for i in range(64)])
    d_2k = np.float16(0.5)
    dmin_2k = np.float16(0.25)
    raw_2k = scales_2k + qs_2k + d_2k.tobytes() + dmin_2k.tobytes()
    ref_2k = np.zeros(256, dtype=np.float32)
    is_idx = 0
    for half in range(2):
        q_slice = np.frombuffer(qs_2k[half * 32 : (half + 1) * 32], dtype=np.uint8)
        for shift in (0, 2, 4, 6):
            sc0 = float(d_2k) * (scales_2k[is_idx] & 0x0F)
            mn0 = float(dmin_2k) * (scales_2k[is_idx] >> 4)
            is_idx += 1
            for l_i in range(16):
                ref_2k[half * 128 + (shift // 2) * 32 + l_i] = sc0 * ((q_slice[l_i] >> shift) & 3) - mn0
            sc1 = float(d_2k) * (scales_2k[is_idx] & 0x0F)
            mn1 = float(dmin_2k) * (scales_2k[is_idx] >> 4)
            is_idx += 1
            for l_i in range(16):
                ref_2k[half * 128 + (shift // 2) * 32 + 16 + l_i] = sc1 * ((q_slice[l_i + 16] >> shift) & 3) - mn1
    out_2k = dequantize_ggml_bytes(raw_2k, (1, 256), GGML_TYPE_Q2_K)
    torch.testing.assert_close(out_2k, torch.from_numpy(ref_2k.reshape(1, 256)), atol=1e-4, rtol=1e-4)

    # Q3_K (block=256, 110B: 32B hmask + 64B qs + 12B scales + f16 d)
    hmask_3k = bytes([0xA5] * 32)
    qs_3k = bytes([(i * 19) & 0xFF for i in range(64)])
    scales_3k = bytes([0x31] * 8 + [0x95] * 4)
    d_3k = np.float16(0.25)
    raw_3k = hmask_3k + qs_3k + scales_3k + d_3k.tobytes()
    out_3k = dequantize_ggml_bytes(raw_3k, (1, 256), GGML_TYPE_Q3_K)
    assert out_3k.shape == (1, 256)
    assert torch.isfinite(out_3k).all()
    # Cross-check Q3_K against transformers reference formula
    from transformers.integrations.gguf.dequant import dequantize as hf_dequant  # type: ignore[import-not-found]

    ref_3k = hf_dequant(torch.tensor(list(raw_3k), dtype=torch.uint8), GGML_TYPE_Q3_K).reshape(1, 256)
    torch.testing.assert_close(out_3k, ref_3k, atol=1e-5, rtol=1e-5)

    # Q4_K
    x256 = rng.uniform(-0.5, 0.5, size=(2, 256)).astype(np.float32)
    raw_4k, ref_4k = _encode_q4_k(x256)
    out_4k = dequantize_ggml_bytes(raw_4k, (2, 256), GGML_TYPE_Q4_K)
    torch.testing.assert_close(out_4k, torch.from_numpy(ref_4k), atol=1e-4, rtol=1e-4)

    # Q5_K (block=256, 176B: f16 d, f16 dmin, 12B scales, 32B qh, 128B qs)
    raw_5k = (
        np.float16(0.25).tobytes()
        + np.float16(0.125).tobytes()
        + bytes([0x12] * 12)
        + bytes([0x5A] * 32)
        + bytes([(i * 23) & 0xFF for i in range(128)])
    )
    out_5k = dequantize_ggml_bytes(raw_5k, (1, 256), GGML_TYPE_Q5_K)
    ref_5k = hf_dequant(torch.tensor(list(raw_5k), dtype=torch.uint8), GGML_TYPE_Q5_K).reshape(1, 256)
    torch.testing.assert_close(out_5k, ref_5k, atol=1e-5, rtol=1e-5)

    # Q6_K
    raw_6k, ref_6k = _encode_q6_k(x256)
    out_6k = dequantize_ggml_bytes(raw_6k, (2, 256), GGML_TYPE_Q6_K)
    torch.testing.assert_close(out_6k, torch.from_numpy(ref_6k), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Memory-Safety & Layer Behavior Tests
# ---------------------------------------------------------------------------


def test_quantized_linear_and_embedding_memory_safety() -> None:
    """Quantized layers hold only packed uint8 buffers and chunk large projections cleanly."""
    rng = np.random.default_rng(7)
    w = rng.uniform(-0.5, 0.5, size=(12, 64)).astype(np.float32)
    raw_q8, ref_w = _encode_q8_0(w)
    raw_u8 = torch.from_numpy(np.frombuffer(raw_q8, dtype=np.uint8).copy())
    bias = torch.linspace(-0.2, 0.2, 12, dtype=torch.float32)

    lin_full = GgufQuantizedLinear(raw_u8, out_features=12, in_features=64, ggml_type=GGML_TYPE_Q8_0, bias=bias)
    lin_chunked = GgufQuantizedLinear(
        raw_u8, out_features=12, in_features=64, ggml_type=GGML_TYPE_Q8_0, bias=bias, chunk_out_rows=5
    )
    assert lin_full.raw_weight.dtype == torch.uint8
    assert not hasattr(lin_full, "weight")
    assert all(p.dtype == torch.float32 for p in lin_full.parameters())  # only bias

    x = torch.randn(2, 3, 64, dtype=torch.float32)
    expected = torch.nn.functional.linear(x, torch.from_numpy(ref_w), bias)
    y_full = lin_full(x)
    y_chunked = lin_chunked(x)
    assert y_full.requires_grad is False
    torch.testing.assert_close(y_full, expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y_chunked, expected, atol=1e-4, rtol=1e-4)

    # Quantized embedding gathers ONLY requested rows
    emb_w = rng.uniform(-0.5, 0.5, size=(20, 64)).astype(np.float32)
    emb_raw, emb_ref = _encode_q5_0(emb_w)
    emb = GgufQuantizedEmbedding(
        torch.from_numpy(np.frombuffer(emb_raw, dtype=np.uint8).copy()),
        vocab_size=20,
        embedding_dim=64,
        ggml_type=GGML_TYPE_Q5_0,
    )
    assert emb.raw_weight.dtype == torch.uint8
    ids = torch.tensor([[3, 17, 0]], dtype=torch.long)
    out_emb = emb(ids)
    torch.testing.assert_close(out_emb, torch.from_numpy(emb_ref)[[3, 17, 0]].unsqueeze(0), atol=1e-4, rtol=1e-4)


def test_llama_vs_qwen2_rope_permutation_behavior(tmp_path: Path) -> None:
    """`llama` reverse-permutes `attn_q`/`attn_k` weights & biases, while `qwen2` does not."""
    # Verify _reverse_permute_rope_rows exact round-trip of llama.cpp interleaved permutation
    n_head = 4
    head_dim = 8
    orig_w = torch.arange(n_head * head_dim * 32, dtype=torch.float32).reshape(n_head * head_dim, 32)
    interleaved_w = orig_w.reshape(n_head, 2, head_dim // 2, 32).swapaxes(1, 2).reshape_as(orig_w)
    restored_w = _reverse_permute_rope_rows(interleaved_w, n_head)
    torch.testing.assert_close(restored_w, orig_w)

    # Tiny Llama GGUF stores interleaved Q/K -> GgufQuantizedLinear must restore orig_w on forward
    llama_gguf = tmp_path / "tiny-llama.gguf"
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=True,
        tie_word_embeddings=True,
    )
    m = AutoModelForCausalLM.from_config(cfg).eval()
    kvs = [
        ("general.architecture", 8, "llama"),
        ("llama.block_count", 4, 2),
        ("llama.embedding_length", 4, 32),
        ("llama.feed_forward_length", 4, 64),
        ("llama.context_length", 4, 64),
        ("llama.attention.head_count", 4, 4),
        ("llama.attention.head_count_kv", 4, 2),
        ("llama.attention.key_length", 4, 8),
        ("llama.attention.layer_norm_rms_epsilon", 6, 1e-6),
        ("llama.rope.freq_base", 6, 10000.0),
    ]
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (64, 32), GGML_TYPE_F32, m.model.embed_tokens.weight.detach().numpy().astype("<f4").tobytes()),
        ("output_norm.weight", (32,), GGML_TYPE_F32, np.ones(32, dtype="<f4").tobytes()),
    ]
    for i in range(2):
        q_orig = m.model.layers[i].self_attn.q_proj.weight.detach()
        q_interleaved = q_orig.reshape(4, 2, 4, 32).swapaxes(1, 2).reshape(32, 32)
        k_orig = m.model.layers[i].self_attn.k_proj.weight.detach()
        k_interleaved = k_orig.reshape(2, 2, 4, 32).swapaxes(1, 2).reshape(16, 32)
        q_bias_orig = torch.arange(32, dtype=torch.float32)
        q_bias_inter = q_bias_orig.reshape(4, 2, 4).swapaxes(1, 2).reshape(32)
        tensors.extend(
            [
                (f"blk.{i}.attn_norm.weight", (32,), GGML_TYPE_F32, np.ones(32, dtype="<f4").tobytes()),
                (f"blk.{i}.ffn_norm.weight", (32,), GGML_TYPE_F32, np.ones(32, dtype="<f4").tobytes()),
                (f"blk.{i}.attn_q.weight", (32, 32), GGML_TYPE_F32, q_interleaved.numpy().astype("<f4").tobytes()),
                (f"blk.{i}.attn_q.bias", (32,), GGML_TYPE_F32, q_bias_inter.numpy().astype("<f4").tobytes()),
                (f"blk.{i}.attn_k.weight", (16, 32), GGML_TYPE_F32, k_interleaved.numpy().astype("<f4").tobytes()),
                (f"blk.{i}.attn_v.weight", (16, 32), GGML_TYPE_F32, m.model.layers[i].self_attn.v_proj.weight.detach().numpy().astype("<f4").tobytes()),
                (f"blk.{i}.attn_output.weight", (32, 32), GGML_TYPE_F32, m.model.layers[i].self_attn.o_proj.weight.detach().numpy().astype("<f4").tobytes()),
                (f"blk.{i}.ffn_gate.weight", (64, 32), GGML_TYPE_F32, m.model.layers[i].mlp.gate_proj.weight.detach().numpy().astype("<f4").tobytes()),
                (f"blk.{i}.ffn_up.weight", (64, 32), GGML_TYPE_F32, m.model.layers[i].mlp.up_proj.weight.detach().numpy().astype("<f4").tobytes()),
                (f"blk.{i}.ffn_down.weight", (32, 64), GGML_TYPE_F32, m.model.layers[i].mlp.down_proj.weight.detach().numpy().astype("<f4").tobytes()),
            ]
        )
    _write_gguf_file(llama_gguf, kvs=kvs, tensors=tensors)

    llama_shard = load_gguf_pytorch_shard(llama_gguf, shard_layer_start=0, shard_layer_end=0)
    assert llama_shard.layers[0].self_attn.q_proj.unpermute_heads == 4
    assert llama_shard.layers[0].self_attn.k_proj.unpermute_heads == 2
    torch.testing.assert_close(llama_shard.layers[0].self_attn.q_proj.bias, torch.arange(32, dtype=torch.float32))
    eye_x = torch.eye(32, dtype=torch.float32)
    q_eff = llama_shard.layers[0].self_attn.q_proj(eye_x)
    torch.testing.assert_close(
        q_eff,
        torch.nn.functional.linear(eye_x, m.model.layers[0].self_attn.q_proj.weight.detach(), torch.arange(32, dtype=torch.float32)),
        atol=1e-5,
        rtol=1e-5,
    )
    if llama_shard.tokenizer is not None:
        llama_shard.tokenizer.close()

    # Qwen2 must keep unpermute_heads=None
    qwen_gguf = tmp_path / "tiny-qwen2.gguf"
    _build_tiny_qwen2_gguf(qwen_gguf, mixed_quants=False)
    qwen_shard = load_gguf_pytorch_shard(qwen_gguf, shard_layer_start=0, shard_layer_end=1)
    assert qwen_shard.layers[0].self_attn.q_proj.unpermute_heads is None
    assert qwen_shard.layers[0].self_attn.k_proj.unpermute_heads is None
    if qwen_shard.tokenizer is not None:
        qwen_shard.tokenizer.close()


# ---------------------------------------------------------------------------
# 3. End-to-End 2-Node GGUF P2P Pipeline & OpenAI Gateway Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_node_gguf_p2p_pipeline_and_openai_gateway(tmp_path: Path) -> None:
    """2-node GGUF pipeline (Head 0..1, Tail 2..3) matches unsharded PyTorch and serves HTTP 200."""
    gguf_path = tmp_path / "qwen2-mixed-quants.gguf"
    _cfg, ref_model = _build_tiny_qwen2_gguf(gguf_path, seed=123, mixed_quants=True)

    head = LocalNodeRuntime()
    tail = LocalNodeRuntime()
    head.state = NodeState.RUNNING
    head.model_id = "Qwen/Qwen2.5-Test-GGUF"
    head.shard_layer_start = 0
    head.shard_layer_end = 1
    head.total_layers = 4
    head.model_runner = ShardModelRunner(
        model_id=head.model_id,
        model_path=str(gguf_path),
        shard_layer_start=0,
        shard_layer_end=1,
        total_layers=4,
        max_ram_mb=2048,
    )

    tail.state = NodeState.RUNNING
    tail.model_id = "Qwen/Qwen2.5-Test-GGUF"
    tail.shard_layer_start = 2
    tail.shard_layer_end = 3
    tail.total_layers = 4
    tail.model_runner = ShardModelRunner(
        model_id=tail.model_id,
        model_path=str(gguf_path),
        shard_layer_start=2,
        shard_layer_end=3,
        total_layers=4,
        max_ram_mb=2048,
    )

    assert head.model_runner.is_real_model is True
    assert head.model_runner.has_native_llama is False
    assert head.model_runner.supports_pipeline_sharding is True
    assert tail.model_runner.supports_pipeline_sharding is True
    # Confirm selective loading kept ONLY 2 layers per node and preserved layer_idx
    assert len(head.model_runner._layers) == 2
    assert head.model_runner._layers[0].self_attn.layer_idx == 0
    assert head.model_runner._layers[1].self_attn.layer_idx == 1
    assert len(tail.model_runner._layers) == 2
    assert tail.model_runner._layers[0].self_attn.layer_idx == 2
    assert tail.model_runner._layers[1].self_attn.layer_idx == 3

    prompt_ids = [3, 4, 5]
    expected_ids: list[int] = []
    with torch.inference_mode():
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long)
        for _ in range(4):
            logits = ref_model(cur_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            expected_ids.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)

    head_port = await head.start_p2p_listener(port=0)
    tail_port = await tail.start_p2p_listener(port=0)
    head.p2p_listen_port = head_port
    tail.p2p_listen_port = tail_port
    head.peers = [
        PeerNodeInfo(
            peer_id="gguf-tail",
            ip_address="127.0.0.1",
            port=tail_port,
            shard_layer_start=2,
            shard_layer_end=3,
            role=PeerRole.PRIMARY,
        )
    ]
    tail.peers = [
        PeerNodeInfo(
            peer_id="gguf-head",
            ip_address="127.0.0.1",
            port=head_port,
            shard_layer_start=0,
            shard_layer_end=1,
            role=PeerRole.PRIMARY,
        )
    ]

    try:
        produced_ids: list[int] = []
        async for _rid, _step, tok_id, _text, _eos in head.stream_p2p_pipeline_tokens(
            prompt="Hello world",
            max_tokens=4,
            temperature=0.0,
        ):
            produced_ids.append(tok_id)
        assert produced_ids == expected_ids

        # Wire global runtime to `head` and verify FastAPI `/v1/chat/completions` stream & non-stream
        token = "sk-shard-gguf-2node-test"
        runtime.state = NodeState.RUNNING
        runtime.node_id = "gguf-head"
        runtime.model_id = head.model_id
        runtime.room_id = "gguf-room"
        runtime.session_token = token
        runtime.shard_layer_start = 0
        runtime.shard_layer_end = 1
        runtime.total_layers = 4
        runtime.n_layers = 2
        runtime.is_swarm_ready = True
        runtime.coverage_percent = 100.0
        runtime.peers = list(head.peers)
        runtime.model_runner = head.model_runner
        runtime.p2p_listen_port = head_port
        # Ensure `runtime` receives incoming TAIL_TOKEN_ACK on the active listener
        runtime.pending_token_queues = head.pending_token_queues
        orig_enqueue = head.pending_token_queues

        # Delegate runtime.stream_p2p_pipeline_tokens via head listener port
        headers = {"Authorization": f"Bearer {token}"}
        # Use head's actual listener by swapping runtime._p2p_server temporarily?
        # Even simpler: forward incoming ACK frames from head's queue -> runtime's queue is unnecessary
        # when we point tail's head peer to head_port and share pending_token_queues.
        assert runtime.pending_token_queues is orig_enqueue

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as async_client:
            non_stream = await async_client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": head.model_id,
                    "messages": [{"role": "user", "content": "Hello world"}],
                    "stream": False,
                    "max_tokens": 3,
                    "temperature": 0.0,
                },
            )
            assert non_stream.status_code == 200
            body = non_stream.json()
            assert body["object"] == "chat.completion"
            assert len(body["choices"]) == 1

            stream_res = await async_client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": head.model_id,
                    "messages": [{"role": "user", "content": "Hello world"}],
                    "stream": True,
                    "max_tokens": 3,
                    "temperature": 0.0,
                },
            )
            assert stream_res.status_code == 200
            assert "data: [DONE]" in stream_res.text
    finally:
        runtime.model_runner = None
        runtime.peers = []
        runtime.session_token = None
        runtime.state = NodeState.STOPPED
        await head.stop_p2p_listener()
        await tail.stop_p2p_listener()
        head.model_runner.release()
        tail.model_runner.release()


# ---------------------------------------------------------------------------
# 4. Corrupt / Truncated GGUF Rejection & Release Cleanup
# ---------------------------------------------------------------------------


def test_corrupt_and_truncated_gguf_rejected_safely(tmp_path: Path) -> None:
    """Truncated GGUF payloads, invalid magic/versions, and 30-byte stubs are rejected."""
    stub = tmp_path / "offline-stub.gguf"
    stub.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 22)
    assert can_load_gguf_pytorch_shard(stub) is False

    # Valid header + tensor table whose data payload is truncated
    valid_gguf = tmp_path / "valid.gguf"
    _build_tiny_qwen2_gguf(valid_gguf, mixed_quants=False)
    idx = read_gguf_header_and_tensors(valid_gguf)
    assert idx.tensor_count > 0

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(valid_gguf.read_bytes()[: idx.data_offset + 16])
    with pytest.raises(ValueError, match="Truncated GGUF tensor"):
        read_gguf_header_and_tensors(truncated)
    assert can_load_gguf_pytorch_shard(truncated) is False


def test_runner_release_closes_tokenizer_and_clears_state(tmp_path: Path) -> None:
    """`ShardModelRunner.release()` closes the GGUF tokenizer and clears all shard references."""
    gguf_path = tmp_path / "release-test.gguf"
    _build_tiny_qwen2_gguf(gguf_path, mixed_quants=False)
    runner = ShardModelRunner(
        model_id="Qwen/Qwen2.5-Release-Test",
        model_path=str(gguf_path),
        shard_layer_start=0,
        shard_layer_end=1,
        total_layers=4,
        max_ram_mb=1024,
    )
    assert runner.is_real_model is True
    assert runner._tokenizer is not None

    closed = False
    orig_close = runner._tokenizer.close

    def _spy_close() -> None:
        nonlocal closed
        closed = True
        orig_close()

    runner._tokenizer.close = _spy_close  # type: ignore[method-assign]
    runner.release()

    assert closed is True
    assert runner._tokenizer is None
    assert runner._shard_container is None
    assert runner._layers == []
    assert runner._embed_tokens is None
    assert runner._norm is None
    assert runner._lm_head is None
    assert runner._is_real_pytorch is False


# ---------------------------------------------------------------------------
# 5. Post-Download Tracker Sync Re-Announces Real `final_total_layers`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_node_post_download_sync_reannounces_real_gguf_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`_start_node_unlocked` re-announces real `final_total_layers` from downloaded GGUF metadata."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0.0")

    gguf_path = tmp_path / "qwen2-4layers.gguf"
    _build_tiny_qwen2_gguf(gguf_path, mixed_quants=False)

    # First call (pre-download): probe_path is None -> returns default 32 layers.
    # Second call (post-download): local_model_path is gguf_path -> returns real 4 layers.
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=gguf_path),
    )
    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", AsyncMock(return_value=[]))
    monkeypatch.setattr(main_module.lan_mdns_service, "start_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(main_module.lan_mdns_service, "stop_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(main_module.zeroconf_discovery, "start_advertising", AsyncMock(return_value=None))
    monkeypatch.setattr(main_module.zeroconf_discovery, "stop_advertising", AsyncMock(return_value=None))

    announce_bodies: list[dict[str, Any]] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/announce":
            body = json.loads(req.content.decode("utf-8"))
            announce_bodies.append(body)
            total_l = int(body.get("total_model_layers", 32))
            return httpx.Response(
                200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": min(1, total_l - 1),
                    "role": "primary",
                    "peers": [],
                },
            )
        if req.method == "GET" and req.url.path == "/rooms":
            return httpx.Response(200, json={"rooms": []})
        if req.method == "GET" and req.url.path.startswith("/peers/"):
            return httpx.Response(200, json={"peers": []})
        if req.method == "POST" and req.url.path == "/nodes/leave":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"detail": "not found"})

    runtime.tracker_client = SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=1,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(_handler),
    )

    try:
        resp = await start_node(
            StartNodeRequest(
                node_id="post-dl-sync-node",
                model_id="Qwen/Qwen2.5-Test-GGUF",
                network_mode=NetworkMode.INTERNET,
                room_id="sync-room",
                ram_gb=8,
                shared_ram_gb=8,
                max_ram_mb=8192,
                p2p_listen_port=49490,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
            )
        )
        assert resp.state == NodeState.RUNNING
        assert resp.total_layers == 4
        assert len(announce_bodies) >= 2
        # Pre-download announce had 32 layers guess; post-download sync re-announced real 4 layers
        assert announce_bodies[0]["total_model_layers"] == 32
        assert announce_bodies[1]["total_model_layers"] == 4
        assert announce_bodies[1]["shard_layer_start"] == 0
        assert announce_bodies[1]["shard_layer_end"] == 1
        assert announce_bodies[1]["max_layers_can_hold"] >= 1
    finally:
        await stop_node()
