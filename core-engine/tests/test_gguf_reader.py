"""Unit tests for Issue #54: GGUF header metadata reader safety and accuracy."""

from __future__ import annotations

import io
from pathlib import Path
import struct

import pytest

import models.gguf_reader as gguf_reader
from models.gguf_reader import (
    _GGUF_MAGIC,
    _GGUF_TYPE_ARRAY,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_INT64,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT32,
    _GGUF_TYPE_UINT64,
    _MAX_GGUF_ARRAY_ELEMENTS,
    _MAX_GGUF_SKIP_DEPTH,
    _skip_gguf_value,
    read_gguf_total_layers,
)


def _pack_gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_kv_uint32(key: str, value: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<II", _GGUF_TYPE_UINT32, value)


def _pack_kv_int32(key: str, value: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<Ii", _GGUF_TYPE_INT32, value)


def _pack_kv_uint64(key: str, value: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<IQ", _GGUF_TYPE_UINT64, value)


def _pack_kv_int64(key: str, value: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<Iq", _GGUF_TYPE_INT64, value)


def _pack_kv_string(key: str, value: str) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<I", _GGUF_TYPE_STRING) + _pack_gguf_string(value)


def _build_nested_array_payload(nesting_levels: int) -> bytes:
    """Build nested 1-element GGUF arrays ending in a single UINT32 scalar."""
    payload = struct.pack("<I", 42)
    for idx in range(nesting_levels):
        elem_type = _GGUF_TYPE_UINT32 if idx == 0 else _GGUF_TYPE_ARRAY
        payload = struct.pack("<IQ", elem_type, 1) + payload
    return payload


def _build_gguf_header(kv_entries: list[bytes], version: int = 3) -> bytes:
    buf = bytearray()
    buf.extend(_GGUF_MAGIC)
    buf.extend(struct.pack("<IQQ", version, 0, len(kv_entries)))
    for entry in kv_entries:
        buf.extend(entry)
    return bytes(buf)


def test_read_gguf_total_layers_skips_262144_string_array_for_gemma3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure 262,144-element token arrays are skipped via seek without decoding elements."""
    token_count = 262_144
    token_entry = _pack_gguf_string("tok")
    tokens_kv = (
        _pack_gguf_string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", _GGUF_TYPE_ARRAY, _GGUF_TYPE_STRING, token_count)
        + (token_entry * token_count)
    )
    block_count_kv = _pack_kv_uint32("gemma3.block_count", 48)

    gguf_file = tmp_path / "gemma3-27b.gguf"
    gguf_file.write_bytes(_build_gguf_header([tokens_kv, block_count_kv]))

    read_string_calls = 0
    original_read_string = gguf_reader._read_gguf_string

    def spy_read_string(f: object) -> str:
        nonlocal read_string_calls
        read_string_calls += 1
        return original_read_string(f)  # type: ignore[arg-type]

    monkeypatch.setattr(gguf_reader, "_read_gguf_string", spy_read_string)

    assert read_gguf_total_layers(gguf_file) == 48
    assert read_string_calls == 2


def test_read_gguf_total_layers_skips_large_scalar_array(tmp_path: Path) -> None:
    """Verify scalar arrays exceeding 200K elements are skipped accurately."""
    score_count = 262_144
    scores_kv = (
        _pack_gguf_string("tokenizer.ggml.scores")
        + struct.pack("<IIQ", _GGUF_TYPE_ARRAY, _GGUF_TYPE_FLOAT32, score_count)
        + (b"\x00\x00\x80\x3f" * score_count)
    )
    block_count_kv = _pack_kv_uint32("gemma3.block_count", 48)

    gguf_file = tmp_path / "gemma3-scores.gguf"
    gguf_file.write_bytes(_build_gguf_header([scores_kv, block_count_kv]))

    assert read_gguf_total_layers(gguf_file) == 48


def test_skip_gguf_value_enforces_recursion_depth_limit(tmp_path: Path) -> None:
    """Verify nested GGUF arrays succeed up to max_depth and fail beyond it."""
    allowed_payload = _build_nested_array_payload(_MAX_GGUF_SKIP_DEPTH)
    stream_ok = io.BytesIO(allowed_payload)
    _skip_gguf_value(stream_ok, _GGUF_TYPE_ARRAY)
    assert stream_ok.tell() == len(allowed_payload)

    too_deep_payload = _build_nested_array_payload(_MAX_GGUF_SKIP_DEPTH + 2)
    stream_bad = io.BytesIO(too_deep_payload)
    with pytest.raises(ValueError, match="maximum recursion depth"):
        _skip_gguf_value(stream_bad, _GGUF_TYPE_ARRAY)

    nested_kv = _pack_gguf_string("custom.deep_nested") + struct.pack("<I", _GGUF_TYPE_ARRAY) + too_deep_payload
    block_count_kv = _pack_kv_uint32("gemma3.block_count", 48)
    gguf_file = tmp_path / "deep-nested.gguf"
    gguf_file.write_bytes(_build_gguf_header([nested_kv, block_count_kv]))

    assert read_gguf_total_layers(gguf_file) is None


def test_skip_gguf_value_rejects_oversized_array_cap(tmp_path: Path) -> None:
    """Verify arrays exceeding _MAX_GGUF_ARRAY_ELEMENTS raise ValueError instead of truncating."""
    oversized = _MAX_GGUF_ARRAY_ELEMENTS + 1

    for elem_type in (_GGUF_TYPE_STRING, _GGUF_TYPE_FLOAT32):
        stream = io.BytesIO(struct.pack("<IQ", elem_type, oversized))
        with pytest.raises(ValueError, match="safety cap"):
            _skip_gguf_value(stream, _GGUF_TYPE_ARRAY)

    oversized_kv = (
        _pack_gguf_string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", _GGUF_TYPE_ARRAY, _GGUF_TYPE_STRING, oversized)
    )
    block_count_kv = _pack_kv_uint32("gemma3.block_count", 48)
    gguf_file = tmp_path / "oversized-array.gguf"
    gguf_file.write_bytes(_build_gguf_header([oversized_kv, block_count_kv]))

    assert read_gguf_total_layers(gguf_file) is None


def test_read_gguf_total_layers_out_of_range_block_count_does_not_double_skip(
    tmp_path: Path,
) -> None:
    """Out-of-range *.block_count values must not double-advance stream before valid entry."""
    kv_entries = [
        _pack_kv_uint32("invalid_zero.block_count", 0),
        _pack_kv_int32("invalid_negative.block_count", -5),
        _pack_kv_uint64("invalid_large.block_count", 999),
        _pack_kv_int64("invalid_neg64.block_count", -1),
        _pack_kv_string("note.block_count", "non-numeric"),
        _pack_kv_uint32("gemma3.block_count", 48),
    ]
    gguf_file = tmp_path / "out-of-range-block-count.gguf"
    gguf_file.write_bytes(_build_gguf_header(kv_entries))

    assert read_gguf_total_layers(gguf_file) == 48
