"""Regression tests for Issue #83: GGUF head metadata validation and embedding token ID bounds."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import struct
from typing import Any

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import numpy as np
import pytest
import torch  # type: ignore[import-not-found]

from models.gguf_shard_loader import (
    GGML_TYPE_F32,
    GGML_TYPE_Q8_0,
    GgufAttention,
    GgufFileIndex,
    GgufQuantizedEmbedding,
    GgufTensorInfo,
    GgufTokenizerAdapter,
    _build_model_config_from_gguf,
    load_gguf_pytorch_shard,
    read_gguf_header_and_tensors,
)
from models.shard_runner import ShardModelRunner


def _pack_gguf_str(text: str) -> bytes:
    raw = text.encode("utf-8")
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
    if val_type == 95:  # synthetic helper code: array of int32
        items = [int(v) for v in value]
        out = _pack_gguf_str(key) + struct.pack("<I", 9)
        body = bytearray(struct.pack("<IQ", 5, len(items)))
        for item in items:
            body.extend(struct.pack("<i", item))
        return out + bytes(body)
    raise ValueError(f"Unsupported test KV type: {val_type}")


def _encode_q8_0(arr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize a float32 array to Q8_0 (block=32, 34B) and return (bytes, ref_f32)."""
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


def _make_base_index(
    tmp_path: Path,
    *,
    arch: str = "llama",
    vocab_size: int = 64,
    hidden_size: int = 64,
    metadata_overrides: dict[str, Any] | None = None,
    omit_keys: tuple[str, ...] = (),
) -> GgufFileIndex:
    """Construct an in-memory `GgufFileIndex` with valid baseline metadata for unit testing."""
    meta: dict[str, Any] = {
        "general.architecture": arch,
        f"{arch}.block_count": 2,
        f"{arch}.embedding_length": hidden_size,
        f"{arch}.feed_forward_length": 128,
        f"{arch}.context_length": 256,
        f"{arch}.attention.head_count": 4,
        f"{arch}.attention.head_count_kv": 2,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.padding_token_id": 0,
    }
    if metadata_overrides:
        meta.update(metadata_overrides)
    for k in omit_keys:
        meta.pop(k, None)

    emb_info = GgufTensorInfo(
        name="token_embd.weight",
        dims_gguf=(hidden_size, vocab_size),
        shape_pt=(vocab_size, hidden_size),
        ggml_type=GGML_TYPE_F32,
        rel_offset=0,
        nbytes=vocab_size * hidden_size * 4,
    )
    return GgufFileIndex(
        path=tmp_path / "synthetic.gguf",
        version=3,
        alignment=32,
        tensor_count=1,
        kv_count=len(meta),
        metadata=meta,
        tensors={"token_embd.weight": emb_info},
        data_offset=64,
        file_size=64 + emb_info.nbytes,
        vocab_size_from_tokens=vocab_size,
    )


def _write_minimal_gguf_model(
    path: Path,
    *,
    arch: str = "qwen3",
    num_layers: int = 2,
    vocab_size: int = 64,
    token_list_len: int | None = None,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    explicit_key_length: int | None = None,
    extra_kvs: list[tuple[str, int, Any]] | None = None,
    omit_kv_keys: set[str] | None = None,
) -> None:
    """Write a complete runnable multi-layer GGUF file with configurable geometry and tokens."""
    rng = np.random.default_rng(2026)
    head_dim = explicit_key_length if explicit_key_length is not None else (hidden_size // num_heads)
    tok_count = token_list_len if token_list_len is not None else vocab_size
    vocab_tokens = [f"tok_{i}" for i in range(tok_count)]

    raw_kvs: list[tuple[str, int, Any]] = [
        ("general.architecture", 8, arch),
        ("general.alignment", 4, 32),
        (f"{arch}.block_count", 4, num_layers),
        (f"{arch}.embedding_length", 4, hidden_size),
        (f"{arch}.feed_forward_length", 4, intermediate_size),
        (f"{arch}.context_length", 4, 128),
        (f"{arch}.attention.head_count", 4, num_heads),
        (f"{arch}.attention.head_count_kv", 4, num_kv_heads),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, 1e-6),
        (f"{arch}.rope.freq_base", 6, 10000.0),
        ("tokenizer.ggml.tokens", 9, vocab_tokens),
        ("tokenizer.ggml.bos_token_id", 4, 1),
        ("tokenizer.ggml.eos_token_id", 4, 2),
    ]
    if explicit_key_length is not None:
        raw_kvs.append((f"{arch}.attention.key_length", 4, explicit_key_length))
    if omit_kv_keys:
        raw_kvs = [entry for entry in raw_kvs if entry[0] not in omit_kv_keys]
    if extra_kvs:
        override_keys = {entry[0] for entry in extra_kvs}
        raw_kvs = [entry for entry in raw_kvs if entry[0] not in override_keys]
        raw_kvs.extend(extra_kvs)

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def _f32_tensor(name: str, shape: tuple[int, ...], ones: bool = False) -> None:
        arr = (
            np.ones(shape, dtype=np.float32)
            if ones
            else (rng.standard_normal(shape).astype(np.float32) * 0.05)
        )
        tensors.append((name, shape, GGML_TYPE_F32, arr.astype("<f4").tobytes()))

    _f32_tensor("token_embd.weight", (vocab_size, hidden_size))
    _f32_tensor("output_norm.weight", (hidden_size,), ones=True)
    _f32_tensor("output.weight", (vocab_size, hidden_size))

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    for layer_idx in range(num_layers):
        prefix = f"blk.{layer_idx}"
        _f32_tensor(f"{prefix}.attn_norm.weight", (hidden_size,), ones=True)
        _f32_tensor(f"{prefix}.ffn_norm.weight", (hidden_size,), ones=True)
        _f32_tensor(f"{prefix}.attn_q.weight", (q_dim, hidden_size))
        _f32_tensor(f"{prefix}.attn_k.weight", (kv_dim, hidden_size))
        _f32_tensor(f"{prefix}.attn_v.weight", (kv_dim, hidden_size))
        _f32_tensor(f"{prefix}.attn_output.weight", (hidden_size, q_dim))
        _f32_tensor(f"{prefix}.ffn_gate.weight", (intermediate_size, hidden_size))
        _f32_tensor(f"{prefix}.ffn_up.weight", (intermediate_size, hidden_size))
        _f32_tensor(f"{prefix}.ffn_down.weight", (hidden_size, intermediate_size))

    _write_gguf_file(path, kvs=raw_kvs, tensors=tensors)


# ---------------------------------------------------------------------------
# 1. Required Metadata & Head Geometry Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_suffix",
    [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
    ],
)
def test_missing_or_invalid_required_gguf_metadata_raises_value_error(
    tmp_path: Path,
    missing_suffix: str,
) -> None:
    """Missing, None, boolean, non-integer, or <= 0 required GGUF metadata raises ValueError."""
    full_key = f"llama.{missing_suffix}"

    # Omitted entirely -> ValueError (never TypeError)
    idx_missing = _make_base_index(tmp_path, omit_keys=(full_key,))
    with pytest.raises(ValueError, match=missing_suffix):
        _build_model_config_from_gguf(idx_missing)

    # Invalid values -> ValueError (never TypeError)
    for bad_val in (None, True, False, "not-an-int", 3.5, 0, -4):
        idx_bad = _make_base_index(tmp_path, metadata_overrides={full_key: bad_val})
        with pytest.raises(ValueError, match=missing_suffix):
            _build_model_config_from_gguf(idx_bad)

    # Also verify end-to-end binary GGUF loading raises ValueError when omitted
    gguf_path = tmp_path / f"missing-{missing_suffix.replace('.', '_')}.gguf"
    _write_minimal_gguf_model(
        gguf_path,
        arch="qwen3",
        explicit_key_length=16,
        omit_kv_keys={f"qwen3.{missing_suffix}"},
    )
    with pytest.raises(ValueError, match=missing_suffix):
        load_gguf_pytorch_shard(gguf_path, shard_layer_start=0, shard_layer_end=0, device="cpu")


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_dim"),
    [
        (0, 2, 16),
        (-4, 2, 16),
        (4, 0, 16),
        (4, -2, 16),
        (4, 8, 16),  # head_count_kv > head_count
        (6, 4, 16),  # head_count % head_count_kv != 0
        (4, 3, 16),  # head_count % head_count_kv != 0
        (4, 2, 0),   # head_dim <= 0
        (4, 2, -8),  # head_dim < 0
    ],
)
def test_invalid_head_geometry_raises_value_error_in_config_and_attention(
    tmp_path: Path,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Invalid head counts, KV head counts, and head_dim fail fast with ValueError in both layers."""
    with pytest.raises(ValueError):
        GgufAttention(
            layer_idx=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=max(2, abs(head_dim)),
            attn_scale=0.25,
        )

    overrides: dict[str, Any] = {
        "llama.embedding_length": 96,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.attention.key_length": head_dim,
    }
    idx = _make_base_index(tmp_path, hidden_size=96, metadata_overrides=overrides)
    with pytest.raises(ValueError):
        _build_model_config_from_gguf(idx)


def test_non_divisible_embedding_length_without_explicit_key_length_raises_value_error(
    tmp_path: Path,
) -> None:
    """When `attention.key_length` is omitted, `embedding_length % head_count != 0` raises ValueError."""
    # 100 % 4 == 0 is True, whereas 100 % 6 != 0
    idx_non_div = _make_base_index(
        tmp_path,
        hidden_size=100,
        metadata_overrides={
            "llama.embedding_length": 100,
            "llama.attention.head_count": 6,
            "llama.attention.head_count_kv": 2,
        },
    )
    with pytest.raises(ValueError, match="divisible"):
        _build_model_config_from_gguf(idx_non_div)

    # embedding_length < head_count (would previously produce head_dim == 0)
    idx_zero_head_dim = _make_base_index(
        tmp_path,
        hidden_size=2,
        metadata_overrides={
            "llama.embedding_length": 2,
            "llama.attention.head_count": 4,
            "llama.attention.head_count_kv": 2,
        },
    )
    with pytest.raises(ValueError):
        _build_model_config_from_gguf(idx_zero_head_dim)


def test_explicit_key_length_allows_non_divisible_embedding_length_end_to_end(
    tmp_path: Path,
) -> None:
    """Models with explicit `attention.key_length` and `embedding_length % head_count != 0` work cleanly."""
    # hidden_size=160, num_heads=6 -> 160 % 6 == 4 != 0, with explicit key_length=32
    gguf_path = tmp_path / "qwen3-explicit-key-length.gguf"
    _write_minimal_gguf_model(
        gguf_path,
        arch="qwen3",
        num_layers=2,
        vocab_size=64,
        hidden_size=160,
        intermediate_size=256,
        num_heads=6,
        num_kv_heads=2,
        explicit_key_length=32,
    )

    index = read_gguf_header_and_tensors(gguf_path)
    cfg = _build_model_config_from_gguf(index)
    assert cfg.hidden_size == 160
    assert cfg.num_attention_heads == 6
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 32

    head = ShardModelRunner(
        model_id="Qwen/Qwen3-ExplicitKeyLen-Test",
        model_path=str(gguf_path),
        shard_layer_start=0,
        shard_layer_end=0,
        total_layers=2,
        max_ram_mb=512,
    )
    tail = ShardModelRunner(
        model_id="Qwen/Qwen3-ExplicitKeyLen-Test",
        model_path=str(gguf_path),
        shard_layer_start=1,
        shard_layer_end=1,
        total_layers=2,
        max_ram_mb=512,
    )
    try:
        h0, shape0 = head.forward_pass([1, 5, 9], request_id="exp-key", step_index=0)
        assert shape0 == (3, 160)
        h1, shape1 = tail.forward_pass(h0, shape=shape0, request_id="exp-key", step_index=0)
        assert shape1 == (3, 160)
        tok_id, _, _ = tail.decode_final_layer_to_token(
            h1, step_index=0, max_tokens=4, request_id="exp-key", temperature=0.0
        )
        assert 0 <= tok_id < 64
    finally:
        head.release()
        tail.release()


# ---------------------------------------------------------------------------
# 2. Special Token ID Bounds Sanitization
# ---------------------------------------------------------------------------


def test_out_of_bounds_special_token_ids_sanitized_in_config_and_tokenizer(
    tmp_path: Path,
) -> None:
    """Out-of-range bos_token_id, eos_token_id (scalar & list), and pad_token_id are sanitized."""
    vocab_size = 64

    # Out-of-bounds positive (>= vocab_size) and negative (< 0) scalar IDs -> None
    idx_oob = _make_base_index(
        tmp_path,
        vocab_size=vocab_size,
        metadata_overrides={
            "tokenizer.ggml.bos_token_id": 64,
            "tokenizer.ggml.eos_token_id": 9999,
            "tokenizer.ggml.padding_token_id": -1,
        },
    )
    cfg_oob = _build_model_config_from_gguf(idx_oob)
    assert cfg_oob.bos_token_id is None
    assert cfg_oob.eos_token_id is None
    assert cfg_oob.pad_token_id is None

    idx_neg = _make_base_index(
        tmp_path,
        vocab_size=vocab_size,
        metadata_overrides={
            "tokenizer.ggml.bos_token_id": -5,
            "tokenizer.ggml.eos_token_id": -1,
            "tokenizer.ggml.padding_token_id": 64,
        },
    )
    cfg_neg = _build_model_config_from_gguf(idx_neg)
    assert cfg_neg.bos_token_id is None
    assert cfg_neg.eos_token_id is None
    assert cfg_neg.pad_token_id is None

    # Valid boundary IDs (0 and vocab_size - 1) are preserved
    idx_valid = _make_base_index(
        tmp_path,
        vocab_size=vocab_size,
        metadata_overrides={
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": vocab_size - 1,
            "tokenizer.ggml.padding_token_id": vocab_size - 1,
        },
    )
    cfg_valid = _build_model_config_from_gguf(idx_valid)
    assert cfg_valid.bos_token_id == 0
    assert cfg_valid.eos_token_id == vocab_size - 1
    assert cfg_valid.pad_token_id == vocab_size - 1

    # Iterable EOS token IDs filter out-of-range values while keeping valid ones
    idx_list_eos = _make_base_index(
        tmp_path,
        vocab_size=vocab_size,
        metadata_overrides={
            "tokenizer.ggml.eos_token_id": [-1, 3, 64, 500, 7],
        },
    )
    cfg_list_eos = _build_model_config_from_gguf(idx_list_eos)
    assert cfg_list_eos.eos_token_id == [3, 7]

    idx_all_bad_list = _make_base_index(
        tmp_path,
        vocab_size=vocab_size,
        metadata_overrides={
            "tokenizer.ggml.eos_token_id": [-3, 64, 1024],
        },
    )
    assert _build_model_config_from_gguf(idx_all_bad_list).eos_token_id is None

    # End-to-end binary GGUF with out-of-bounds BOS and mixed-validity EOS int32 array
    gguf_path = tmp_path / "oob-special-tokens.gguf"
    _write_minimal_gguf_model(
        gguf_path,
        arch="qwen3",
        vocab_size=vocab_size,
        explicit_key_length=16,
        extra_kvs=[
            ("tokenizer.ggml.bos_token_id", 5, 500),
            ("tokenizer.ggml.eos_token_id", 95, [-2, 4, 64, 9, 999]),
            ("tokenizer.ggml.padding_token_id", 5, -1),
            ("tokenizer.ggml.add_bos_token", 7, True),
        ],
    )
    parsed_idx = read_gguf_header_and_tensors(gguf_path)
    parsed_cfg = _build_model_config_from_gguf(parsed_idx)
    assert parsed_cfg.bos_token_id is None
    assert parsed_cfg.eos_token_id == [4, 9]
    assert parsed_cfg.pad_token_id is None

    tok = GgufTokenizerAdapter(parsed_idx, vocab_size=vocab_size)
    try:
        assert tok.bos_token_id is None
        assert tok.eos_token_id == 4
        assert {4, 9}.issubset(tok._special_ids)
        assert 500 not in tok._special_ids
        assert 64 not in tok._special_ids
        encoded = tok.encode("tok_5", add_special_tokens=True)
        assert all(0 <= tid < vocab_size for tid in encoded)
    finally:
        tok.close()


# ---------------------------------------------------------------------------
# 3. GgufQuantizedEmbedding Bounds Guard & Exact Valid Row Preservation
# ---------------------------------------------------------------------------


def test_gguf_quantized_embedding_rejects_non_positive_vocab_and_guards_oob_ids(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """`GgufQuantizedEmbedding` rejects vocab_size <= 0 and clamps < 0 / >= vocab_size IDs safely."""
    vocab_size = 16
    emb_dim = 32
    rng = np.random.default_rng(101)
    weights_f32 = rng.standard_normal((vocab_size, emb_dim)).astype(np.float32)
    raw_q8, ref_q8 = _encode_q8_0(weights_f32)
    raw_u8 = torch.from_numpy(np.frombuffer(raw_q8, dtype=np.uint8).copy())

    for bad_vocab in (0, -1, False):
        with pytest.raises(ValueError, match="vocab_size > 0"):
            GgufQuantizedEmbedding(
                raw_weight=torch.empty(0, dtype=torch.uint8),
                vocab_size=bad_vocab,  # type: ignore[arg-type]
                embedding_dim=emb_dim,
                ggml_type=GGML_TYPE_Q8_0,
            )

    emb = GgufQuantizedEmbedding(
        raw_weight=raw_u8,
        vocab_size=vocab_size,
        embedding_dim=emb_dim,
        ggml_type=GGML_TYPE_Q8_0,
        embed_scale=2.0,
        output_dtype=torch.float32,
    )

    input_ids = torch.tensor([[-7, 3, vocab_size, 11, 9999, 0]], dtype=torch.long)
    original_snapshot = input_ids.clone()

    with caplog.at_level(logging.WARNING, logger="shardai.models.gguf_shard_loader"):
        out = emb(input_ids)

    # Caller tensor is never mutated in place
    assert torch.equal(input_ids, original_snapshot)
    assert out.shape == (1, 6, emb_dim)
    assert any("out-of-bounds token ID" in rec.message for rec in caplog.records)

    ref_tensor = torch.from_numpy(ref_q8) * 2.0
    # Valid IDs (3, 11, 0) preserve exact embeddings
    torch.testing.assert_close(out[0, 1], ref_tensor[3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 3], ref_tensor[11], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 5], ref_tensor[0], atol=1e-5, rtol=1e-5)
    # Out-of-bounds IDs (-7, 16, 9999) map cleanly to fallback row 0 without IndexError
    torch.testing.assert_close(out[0, 0], ref_tensor[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 2], ref_tensor[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[0, 4], ref_tensor[0], atol=1e-5, rtol=1e-5)

    # End-to-end runner forward_pass when tokenizer has more tokens than token_embd.weight rows
    gguf_mismatch = tmp_path / "vocab-mismatch.gguf"
    _write_minimal_gguf_model(
        gguf_mismatch,
        arch="qwen3",
        num_layers=2,
        vocab_size=64,
        token_list_len=80,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        explicit_key_length=16,
    )
    head_runner = ShardModelRunner(
        model_id="Qwen/Qwen3-VocabMismatch-Test",
        model_path=str(gguf_mismatch),
        shard_layer_start=0,
        shard_layer_end=0,
        total_layers=2,
        max_ram_mb=512,
    )
    try:
        raw_act, shape = head_runner.forward_pass([-1, 5, 64, 79], request_id="oob-e2e", step_index=0)
        assert shape == (4, 64)
        assert len(raw_act) == 4 * 64 * 4
    finally:
        head_runner.release()
