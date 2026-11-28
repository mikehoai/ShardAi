"""Regression tests for 32-byte GGML weight buffer alignment (Issue #78)."""

from __future__ import annotations

import gc
import io
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import pytest
import torch

from models.ggml_native import (
    GGML_NATIVE_AVAILABLE,
    GGML_TENSOR_ALIGNMENT,
    GgmlNativeEngine,
    _alloc_aligned_cpu_u8,
    _ensure_aligned_cpu_bytes,
)
from models.gguf_shard_loader import (
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q8_0,
    GgufFileIndex,
    GgufQuantizedLinear,
    GgufTensorInfo,
    _read_tensor_raw_u8,
)
from tests.test_gguf_sharded_pipeline import _encode_q4_k, _encode_q8_0


class _ChunkedBinaryReader(io.BytesIO):
    """BytesIO subclass simulating short partial `readinto` reads."""

    def __init__(self, data: bytes, max_chunk: int = 7) -> None:
        super().__init__(data)
        self._max_chunk = max_chunk

    def readinto(self, buffer: object) -> int:  # type: ignore[override]
        mv = memoryview(buffer)  # type: ignore[arg-type]
        if len(mv) > self._max_chunk:
            return super().readinto(mv[: self._max_chunk])
        return super().readinto(mv)


def _make_unaligned_slice(payload: bytes, misalign_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a contiguous 1D `torch.uint8` slice whose `data_ptr()` is intentionally unaligned."""
    nbytes = len(payload)
    backing, _ = _alloc_aligned_cpu_u8(nbytes + GGML_TENSOR_ALIGNMENT)
    assert int(backing.data_ptr()) % GGML_TENSOR_ALIGNMENT == 0
    sl = backing[misalign_offset : misalign_offset + nbytes]
    assert int(sl.data_ptr()) % GGML_TENSOR_ALIGNMENT == misalign_offset % GGML_TENSOR_ALIGNMENT
    assert int(sl.data_ptr()) % GGML_TENSOR_ALIGNMENT != 0
    sl.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
    return sl, backing


@pytest.mark.parametrize("nbytes", [0, 1, 17, 144, 4096])
def test_alloc_aligned_cpu_u8_guarantees_32_byte_alignment(nbytes: int) -> None:
    """`_alloc_aligned_cpu_u8` returns a contiguous 1D `torch.uint8` tensor aligned to 32 bytes."""
    aligned, backing = _alloc_aligned_cpu_u8(nbytes)
    assert aligned.dtype == torch.uint8
    assert aligned.ndim == 1
    assert aligned.is_contiguous()
    assert int(aligned.numel()) == nbytes
    assert int(backing.numel()) >= nbytes
    if nbytes > 0:
        assert int(aligned.data_ptr()) % GGML_TENSOR_ALIGNMENT == 0


@pytest.mark.parametrize("misalign_offset", [1, 16])
def test_ensure_aligned_cpu_bytes_fixes_unaligned_slices(misalign_offset: int) -> None:
    """`_ensure_aligned_cpu_bytes` realigns 1-byte and 16-byte offset slices while preserving bytes."""
    payload = bytes((i * 13 + 7) % 251 for i in range(512))
    unaligned, backing = _make_unaligned_slice(payload, misalign_offset)

    aligned, pinned = _ensure_aligned_cpu_bytes(unaligned)
    assert aligned.dtype == torch.uint8
    assert aligned.ndim == 1
    assert aligned.is_contiguous()
    assert int(aligned.data_ptr()) % GGML_TENSOR_ALIGNMENT == 0
    assert bytes(aligned.numpy().tobytes()) == payload
    assert pinned is not None

    # Drop the caller's unaligned source and verify pinned storage remains intact.
    del unaligned, backing
    gc.collect()
    assert bytes(aligned.numpy().tobytes()) == payload


def test_ensure_aligned_cpu_bytes_reuses_aligned_tensor_zero_copy() -> None:
    """Already 32-byte aligned CPU uint8 tensors are returned zero-copy without extra allocation."""
    aligned_src, _ = _alloc_aligned_cpu_u8(256)
    aligned_src.copy_(torch.arange(256, dtype=torch.uint8))

    out, backing = _ensure_aligned_cpu_bytes(aligned_src)
    assert int(out.data_ptr()) == int(aligned_src.data_ptr())
    assert int(backing.data_ptr()) == int(aligned_src.data_ptr())


def test_read_tensor_raw_u8_aligned_and_handles_partial_reads_and_eof() -> None:
    """`_read_tensor_raw_u8` reads directly into 32-byte aligned buffers and raises EOFError on truncation."""
    payload = bytes((i * 29 + 3) % 256 for i in range(1024))
    prefix = b"\xaa" * 37
    stream = _ChunkedBinaryReader(prefix + payload, max_chunk=11)

    t_info = GgufTensorInfo(
        name="blk.0.attn_q.weight",
        dims_gguf=(32, 32),
        shape_pt=(32, 32),
        ggml_type=GGML_TYPE_Q8_0,
        rel_offset=5,
        nbytes=len(payload),
    )
    index = GgufFileIndex(
        path=Path("dummy.gguf"),
        version=3,
        alignment=32,
        tensor_count=1,
        kv_count=0,
        metadata={},
        tensors={t_info.name: t_info},
        data_offset=32,
        file_size=len(prefix) + len(payload),
    )

    raw_u8 = _read_tensor_raw_u8(cast(BinaryIO, stream), index, t_info)
    assert raw_u8.dtype == torch.uint8
    assert raw_u8.ndim == 1
    assert raw_u8.is_contiguous()
    assert int(raw_u8.numel()) == len(payload)
    assert int(raw_u8.data_ptr()) % GGML_TENSOR_ALIGNMENT == 0
    assert raw_u8.numpy().tobytes() == payload

    # Zero-byte tensor edge case
    zero_info = GgufTensorInfo(
        name="empty.weight",
        dims_gguf=(0,),
        shape_pt=(0,),
        ggml_type=GGML_TYPE_Q8_0,
        rel_offset=0,
        nbytes=0,
    )
    empty_u8 = _read_tensor_raw_u8(cast(BinaryIO, io.BytesIO(b"")), index, zero_info)
    assert empty_u8.dtype == torch.uint8
    assert empty_u8.ndim == 1
    assert int(empty_u8.numel()) == 0

    # Truncated stream must raise EOFError
    truncated_stream = io.BytesIO(prefix + payload[:500])
    with pytest.raises(EOFError, match="Unexpected EOF reading tensor"):
        _read_tensor_raw_u8(cast(BinaryIO, truncated_stream), index, t_info)


@pytest.mark.skipif(
    not GGML_NATIVE_AVAILABLE, reason="llama_cpp bundled libggml not available"
)
@pytest.mark.parametrize("misalign_offset", [1, 16])
@pytest.mark.parametrize(
    ("ggml_type", "encoder"),
    [
        (GGML_TYPE_Q8_0, _encode_q8_0),
        (GGML_TYPE_Q4_K, _encode_q4_k),
    ],
)
def test_ggml_native_register_weight_accepts_unaligned_slices_without_sigabrt(
    misalign_offset: int,
    ggml_type: int,
    encoder: object,
) -> None:
    """Intentional 1-byte and 16-byte unaligned slices register and compute cleanly in libggml."""
    rng = np.random.default_rng(2026 + misalign_offset + ggml_type)
    out_features, in_features = 64, 256
    w_src = rng.standard_normal((out_features, in_features)).astype(np.float32) * 0.2
    raw_bytes, _ = encoder(w_src)  # type: ignore[operator]

    unaligned_slice, backing = _make_unaligned_slice(raw_bytes, misalign_offset)
    assert int(unaligned_slice.data_ptr()) % GGML_TENSOR_ALIGNMENT != 0

    engine = GgmlNativeEngine(preferred="cpu")
    try:
        lin_native = GgufQuantizedLinear(
            raw_weight=unaligned_slice,
            out_features=out_features,
            in_features=in_features,
            ggml_type=ggml_type,
        )
        lin_ref = GgufQuantizedLinear(
            raw_weight=unaligned_slice.clone(),
            out_features=out_features,
            in_features=in_features,
            ggml_type=ggml_type,
        )

        assert lin_native.attach_ggml_engine(engine) is True
        assert len(engine._weight_refs) == 1
        assert int(engine._weight_refs[0].numel()) >= len(raw_bytes)
        assert lin_native._ggml_handle is not None

        # Release caller references to the original unaligned backing and GC
        del unaligned_slice, backing
        gc.collect()

        x = torch.randn(1, 2, in_features, dtype=torch.float32)
        y_native = lin_native(x)
        y_ref = lin_ref(x)
        cos = torch.nn.functional.cosine_similarity(y_native, y_ref, dim=-1)
        assert float(cos.min()) > 0.999
        torch.testing.assert_close(y_native, y_ref, atol=0.15, rtol=0.05)
    finally:
        engine.close()
        assert len(engine._weight_refs) == 0
        assert engine.is_closed is True
