"""Security and regression tests for INT8/BF16 wire tensor bounds and scale validation (Issue #69)."""

from __future__ import annotations

import socket
from typing import Any
import uuid
import zlib

import lz4.frame as lz4_frame  # type: ignore[import-not-found]
import numpy as np
import pytest

import p2p.serializer as serializer_module
from p2p.serializer import (
    ACTIVATION_HEADER_STRUCT,
    MAGIC_HEADER,
    MAX_UNCOMPRESSED_TENSOR_BYTES,
    TENSOR_DTYPE_BFLOAT16_FLAG,
    TENSOR_DTYPE_INT8_FLAG,
    CompressionCodec,
    PacketType,
    bfloat16_bytes_to_float32_bytes,
    deserialize_activation_tensor,
    float32_bytes_to_bfloat16_bytes,
    float32_bytes_to_int8_wire,
    int8_wire_to_float32_bytes,
    serialize_activation_tensor,
)


def _pack_custom_v2_frame(
    payload: bytes,
    codec_flags: int,
    uncompressed_len: int,
    shape: tuple[int, int],
) -> bytes:
    """Build a valid-CRC v2 activation frame with explicit dtype/codec flags and shape."""
    crc: int = zlib.crc32(payload) & 0xFFFFFFFF
    header: bytes = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        codec_flags,
        uuid.UUID(str(uuid.uuid4())).bytes,
        socket.inet_aton("127.0.0.1"),
        49152,
        15,
        32,
        0,
        16,
        shape[0],
        shape[1],
        uncompressed_len,
        crc,
        0.7,
    )
    return header + payload


@pytest.mark.parametrize("codec", [CompressionCodec.LZ4, CompressionCodec.ZLIB])
def test_int8_oversized_post_expansion_rejected_before_decompression(
    monkeypatch: pytest.MonkeyPatch,
    codec: CompressionCodec,
) -> None:
    """INT8 frame <= 64 MiB on the wire whose expanded float32 > 64 MiB fails pre-decompression."""
    decompress_called: bool = False

    def _spy_decompress(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal decompress_called
        decompress_called = True
        return b""

    monkeypatch.setattr(serializer_module, "_decompress_buffer", _spy_decompress)

    # 2049 x 8192 -> 16,793,604 wire bytes (~16 MiB) -> 67,141,632 float32 bytes (> 64 MiB)
    shape = (2049, 8192)
    expected_wire_len = (shape[0] * 4) + (shape[0] * shape[1])
    assert expected_wire_len < MAX_UNCOMPRESSED_TENSOR_BYTES
    assert shape[0] * shape[1] * 4 > MAX_UNCOMPRESSED_TENSOR_BYTES

    raw_zero_wire = b"\x00" * expected_wire_len
    compressed_payload = (
        lz4_frame.compress(raw_zero_wire)
        if codec == CompressionCodec.LZ4
        else zlib.compress(raw_zero_wire, level=zlib.Z_BEST_SPEED)
    )
    frame = _pack_custom_v2_frame(
        payload=compressed_payload,
        codec_flags=int(codec) | TENSOR_DTYPE_INT8_FLAG,
        uncompressed_len=expected_wire_len,
        shape=shape,
    )

    with pytest.raises(ValueError, match="exceeding safety cap"):
        deserialize_activation_tensor(frame)
    assert decompress_called is False


@pytest.mark.parametrize("codec", [CompressionCodec.LZ4, CompressionCodec.ZLIB])
def test_bf16_oversized_post_expansion_rejected_before_decompression(
    monkeypatch: pytest.MonkeyPatch,
    codec: CompressionCodec,
) -> None:
    """BF16 frame <= 64 MiB on the wire whose expanded float32 > 64 MiB fails pre-decompression."""
    decompress_called: bool = False

    def _spy_decompress(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal decompress_called
        decompress_called = True
        return b""

    monkeypatch.setattr(serializer_module, "_decompress_buffer", _spy_decompress)

    # 4096 x 4097 -> 33,562,624 wire bytes (~32 MiB) -> 67,125,248 float32 bytes (> 64 MiB)
    shape = (4096, 4097)
    expected_wire_len = shape[0] * shape[1] * 2
    assert expected_wire_len < MAX_UNCOMPRESSED_TENSOR_BYTES
    assert shape[0] * shape[1] * 4 > MAX_UNCOMPRESSED_TENSOR_BYTES

    raw_zero_wire = b"\x00" * expected_wire_len
    compressed_payload = (
        lz4_frame.compress(raw_zero_wire)
        if codec == CompressionCodec.LZ4
        else zlib.compress(raw_zero_wire, level=zlib.Z_BEST_SPEED)
    )
    frame = _pack_custom_v2_frame(
        payload=compressed_payload,
        codec_flags=int(codec) | TENSOR_DTYPE_BFLOAT16_FLAG,
        uncompressed_len=expected_wire_len,
        shape=shape,
    )

    with pytest.raises(ValueError, match="exceeding safety cap"):
        deserialize_activation_tensor(frame)
    assert decompress_called is False


def test_empty_shape_and_mismatched_wire_len_rejected_before_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-shape BF16/INT8 bypasses and shape/wire length mismatches fail pre-decompression."""
    decompress_called: bool = False

    def _spy_decompress(*_args: Any, **_kwargs: Any) -> bytes:
        nonlocal decompress_called
        decompress_called = True
        return b""

    monkeypatch.setattr(serializer_module, "_decompress_buffer", _spy_decompress)

    small_lz4 = lz4_frame.compress(b"\x00" * 64)

    # Empty-shape BF16 claiming > 32 MiB wire payload (> 64 MiB expanded)
    bf16_empty_oversized = _pack_custom_v2_frame(
        payload=small_lz4,
        codec_flags=int(CompressionCodec.LZ4) | TENSOR_DTYPE_BFLOAT16_FLAG,
        uncompressed_len=(MAX_UNCOMPRESSED_TENSOR_BYTES // 2) + 2,
        shape=(0, 0),
    )
    with pytest.raises(ValueError, match="exceeding safety cap"):
        deserialize_activation_tensor(bf16_empty_oversized)
    assert decompress_called is False

    # Empty-shape INT8 frame rejected before decompression
    int8_empty_shape = _pack_custom_v2_frame(
        payload=small_lz4,
        codec_flags=int(CompressionCodec.LZ4) | TENSOR_DTYPE_INT8_FLAG,
        uncompressed_len=64,
        shape=(0, 0),
    )
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        deserialize_activation_tensor(int8_empty_shape)
    assert decompress_called is False

    # Mismatched INT8 wire length vs shape rejected before decompression
    int8_mismatched_len = _pack_custom_v2_frame(
        payload=small_lz4,
        codec_flags=int(CompressionCodec.LZ4) | TENSOR_DTYPE_INT8_FLAG,
        uncompressed_len=64,
        shape=(4, 16),
    )
    with pytest.raises(ValueError, match="does not match shape"):
        deserialize_activation_tensor(int8_mismatched_len)
    assert decompress_called is False

    # Mismatched BF16 wire length vs shape rejected before decompression
    bf16_mismatched_len = _pack_custom_v2_frame(
        payload=small_lz4,
        codec_flags=int(CompressionCodec.LZ4) | TENSOR_DTYPE_BFLOAT16_FLAG,
        uncompressed_len=64,
        shape=(4, 16),
    )
    with pytest.raises(ValueError, match="does not match shape"):
        deserialize_activation_tensor(bf16_mismatched_len)
    assert decompress_called is False


@pytest.mark.parametrize(
    "bad_scale",
    [float("nan"), float("inf"), float("-inf"), -0.5, -1e-6],
)
def test_int8_wire_rejects_non_finite_and_negative_scales(bad_scale: float) -> None:
    """INT8 frames with NaN, +/-Inf, or negative per-row scales must raise ValueError."""
    shape = (2, 8)
    scales = np.array([0.25, bad_scale], dtype="<f4")
    quantized = np.zeros(shape, dtype=np.int8)
    quantized[1, 0] = 10
    raw_int8_wire = scales.tobytes() + quantized.tobytes()

    with pytest.raises(ValueError, match="non-finite or negative"):
        int8_wire_to_float32_bytes(raw_int8_wire, shape)

    frame_uncompressed = _pack_custom_v2_frame(
        payload=raw_int8_wire,
        codec_flags=int(CompressionCodec.NONE) | TENSOR_DTYPE_INT8_FLAG,
        uncompressed_len=len(raw_int8_wire),
        shape=shape,
    )
    with pytest.raises(ValueError, match="non-finite or negative"):
        deserialize_activation_tensor(frame_uncompressed)

    compressed_payload = lz4_frame.compress(raw_int8_wire)
    frame_compressed = _pack_custom_v2_frame(
        payload=compressed_payload,
        codec_flags=int(CompressionCodec.LZ4) | TENSOR_DTYPE_INT8_FLAG,
        uncompressed_len=len(raw_int8_wire),
        shape=shape,
    )
    with pytest.raises(ValueError, match="non-finite or negative"):
        deserialize_activation_tensor(frame_compressed)


def test_direct_wire_expanders_enforce_64mib_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct calls to int8_wire_to_float32_bytes and bfloat16_bytes_to_float32_bytes enforce cap."""
    monkeypatch.setattr(serializer_module, "MAX_UNCOMPRESSED_TENSOR_BYTES", 1024)

    oversized_bf16 = b"\x00" * 514  # expands to 1028 > 1024
    with pytest.raises(ValueError, match="exceeding safety cap"):
        bfloat16_bytes_to_float32_bytes(oversized_bf16)

    oversized_i8_shape = (16, 17)  # 16 * 17 * 4 = 1088 > 1024
    oversized_i8_wire = b"\x00" * (16 * 4 + 16 * 17)
    with pytest.raises(ValueError, match="exceeding safety cap"):
        int8_wire_to_float32_bytes(oversized_i8_wire, oversized_i8_shape)


def test_valid_int8_and_bf16_frames_within_cap_roundtrip_accurately() -> None:
    """Valid INT8 and BF16 frames within 64 MiB expanded bounds continue round-tripping."""
    rng = np.random.default_rng(2026)
    shape = (16, 1024)  # 65,536 bytes FP32 (< 64 MiB)
    mat = (rng.standard_normal(shape).astype(np.float32) * 2.0).astype("<f4")
    mat[0, :] = 0.0
    raw_f32 = mat.tobytes()

    # INT8 compressed round-trip
    int8_frame = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=4,
        total_layers=16,
        step_index=0,
        force_compression=True,
        use_int8_wire=True,
    )
    decoded_int8 = deserialize_activation_tensor(int8_frame)
    assert decoded_int8.wire_int8 is True
    assert decoded_int8.wire_bfloat16 is False
    assert decoded_int8.shape == shape
    restored_int8 = np.frombuffer(decoded_int8.raw_tensor_bytes, dtype="<f4").reshape(shape)
    assert np.all(restored_int8[0] == 0.0)
    assert float(np.max(np.abs(restored_int8 - mat))) < 0.1

    # BF16 compressed round-trip
    bf16_frame = serialize_activation_tensor(
        tensor_data=raw_f32,
        shape=shape,
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=4,
        total_layers=16,
        step_index=0,
        force_compression=True,
        use_bfloat16_wire=True,
    )
    decoded_bf16 = deserialize_activation_tensor(bf16_frame)
    assert decoded_bf16.wire_bfloat16 is True
    assert decoded_bf16.wire_int8 is False
    assert decoded_bf16.shape == shape
    restored_bf16 = np.frombuffer(decoded_bf16.raw_tensor_bytes, dtype="<f4").reshape(shape)
    assert np.allclose(restored_bf16, mat, rtol=1e-2, atol=2e-2)

    # Direct helper checks within cap
    assert len(bfloat16_bytes_to_float32_bytes(float32_bytes_to_bfloat16_bytes(raw_f32))) == len(raw_f32)
    assert len(int8_wire_to_float32_bytes(float32_bytes_to_int8_wire(raw_f32, shape), shape)) == len(raw_f32)
