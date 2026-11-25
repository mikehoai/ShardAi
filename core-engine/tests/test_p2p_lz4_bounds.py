"""Security and regression tests for bounded LZ4/Zlib P2P activation decompression (Issue #50)."""

from __future__ import annotations

from array import array
import asyncio
import socket
import struct
import threading
import tracemalloc
from typing import Any
import uuid
import zlib

import lz4.frame as lz4_frame  # type: ignore[import-not-found]
import pytest

import p2p.serializer as serializer_module
from p2p.serializer import (
    ACTIVATION_HEADER_STRUCT,
    LEGACY_ACTIVATION_HEADER_STRUCT,
    MAGIC_HEADER,
    MAX_UNCOMPRESSED_TENSOR_BYTES,
    CompressionCodec,
    PacketType,
    _decompress_buffer,
    deserialize_activation_tensor,
    serialize_activation_tensor,
)
import p2p.transport as transport_module
from p2p.transport import FRAME_LEN_STRUCT, ACK_BYTE, NACK_BYTE, receive_activations


def _pack_v2_activation_frame(
    payload: bytes,
    codec: CompressionCodec,
    uncompressed_len: int,
    shape: tuple[int, int] = (4, 4),
    request_id: str | None = None,
) -> bytes:
    """Pack a raw v2 SHRD activation frame with an explicit codec and declared size."""
    crc: int = zlib.crc32(payload) & 0xFFFFFFFF
    header: bytes = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(codec),
        uuid.UUID(request_id or str(uuid.uuid4())).bytes,
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


def _pack_v1_activation_frame(
    payload: bytes,
    codec: CompressionCodec,
    uncompressed_len: int,
    shape: tuple[int, int] = (4, 4),
) -> bytes:
    """Pack a legacy 52-byte v1 SHRD activation frame."""
    crc: int = zlib.crc32(payload) & 0xFFFFFFFF
    header: bytes = LEGACY_ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(codec),
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
    )
    return header + payload


def test_lz4_valid_round_trip_and_frame_variants() -> None:
    """Valid LZ4 frames (standard, multi-block without content_size, 0-byte, and v1) succeed."""
    seq_len, hidden_dim = 32, 1024
    raw_bytes: bytes = array("f", [0.25] * (seq_len * hidden_dim)).tobytes()
    req_id: str = str(uuid.uuid4())

    wire_frame: bytes = serialize_activation_tensor(
        tensor_data=raw_bytes,
        shape=(seq_len, hidden_dim),
        request_id=req_id,
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=15,
        total_layers=32,
        step_index=0,
        force_compression=True,
    )
    unpacked = deserialize_activation_tensor(wire_frame)
    assert unpacked.compression_used == CompressionCodec.LZ4
    assert unpacked.raw_tensor_bytes == raw_bytes
    assert unpacked.request_id == req_id

    # Multi-block LZ4 frame with store_size=False (content_size=0 in LZ4 header)
    multi_block_raw: bytes = b"SHRD" * 65_536  # 256 KB
    multi_block_comp: bytes = lz4_frame.compress(
        multi_block_raw,
        block_size=lz4_frame.BLOCKSIZE_MAX64KB,
        store_size=False,
    )
    v2_no_size = _pack_v2_activation_frame(
        payload=multi_block_comp,
        codec=CompressionCodec.LZ4,
        uncompressed_len=len(multi_block_raw),
        shape=(64, 1024),
    )
    unpacked_no_size = deserialize_activation_tensor(v2_no_size)
    assert unpacked_no_size.raw_tensor_bytes == multi_block_raw

    # Legacy v1 frame compressed with LZ4
    v1_frame = _pack_v1_activation_frame(
        payload=multi_block_comp,
        codec=CompressionCodec.LZ4,
        uncompressed_len=len(multi_block_raw),
        shape=(64, 1024),
    )
    unpacked_v1 = deserialize_activation_tensor(v1_frame)
    assert unpacked_v1.raw_tensor_bytes == multi_block_raw

    # Empty 0-byte LZ4 payload round-trip
    empty_comp: bytes = lz4_frame.compress(b"")
    assert _decompress_buffer(empty_comp, CompressionCodec.LZ4, 0) == b""


def test_deserialize_rejects_oversized_declared_length_before_lz4_decompression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames claiming uncompressed_len > MAX_UNCOMPRESSED_TENSOR_BYTES fail before decompression."""
    called: bool = False

    class _SpyDecompressor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal called
            called = True

    monkeypatch.setattr(lz4_frame, "LZ4FrameDecompressor", _SpyDecompressor)

    small_lz4: bytes = lz4_frame.compress(b"A" * 64)
    oversized_frame: bytes = _pack_v2_activation_frame(
        payload=small_lz4,
        codec=CompressionCodec.LZ4,
        uncompressed_len=MAX_UNCOMPRESSED_TENSOR_BYTES + 1,
    )
    with pytest.raises(ValueError, match="exceeding safety cap"):
        deserialize_activation_tensor(oversized_frame)
    assert called is False


@pytest.mark.parametrize("store_size", [True, False])
def test_lz4_bomb_with_small_declared_size_fails_fast_without_allocating_bomb(
    monkeypatch: pytest.MonkeyPatch,
    store_size: bool,
) -> None:
    """A 128 MB LZ4 bomb claiming a 64-byte tensor must fail fast without allocating 128 MB."""
    bomb_payload: bytes = lz4_frame.compress(b"\x00" * (128 * 1024 * 1024), store_size=store_size)
    assert len(bomb_payload) < 1024 * 1024  # Highly compressible bomb

    observed_max_lengths: list[int] = []
    orig_cls = lz4_frame.LZ4FrameDecompressor

    class _TrackingDecompressor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = orig_cls(*args, **kwargs)

        def decompress(self, data: bytes, max_length: int = -1) -> bytes:
            observed_max_lengths.append(max_length)
            return bytes(self._inner.decompress(data, max_length=max_length))

        @property
        def eof(self) -> bool:
            return bool(self._inner.eof)

        @property
        def needs_input(self) -> bool:
            return bool(self._inner.needs_input)

        @property
        def unused_data(self) -> bytes | None:
            return self._inner.unused_data

    monkeypatch.setattr(lz4_frame, "LZ4FrameDecompressor", _TrackingDecompressor)

    frame: bytes = _pack_v2_activation_frame(
        payload=bomb_payload,
        codec=CompressionCodec.LZ4,
        uncompressed_len=64,
    )

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceeded safety limit"):
            deserialize_activation_tensor(frame)
        _cur, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Peak allocation stays tiny (well under MAX_UNCOMPRESSED_TENSOR_BYTES / 128 MB)
    assert peak_bytes < 4 * 1024 * 1024
    if not store_size:
        assert observed_max_lengths == [64]
    else:
        # When content_size is present in LZ4 frame header, it is rejected before decompressing
        assert observed_max_lengths == []


def test_lz4_bomb_at_max_safety_cap_stops_at_max_uncompressed_tensor_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An LZ4 bomb without content_size claiming MAX_UNCOMPRESSED_TENSOR_BYTES stops at cap."""
    capped_limit: int = 4096
    monkeypatch.setattr(serializer_module, "MAX_UNCOMPRESSED_TENSOR_BYTES", capped_limit)

    # 1 MB bomb compressed without content_size, claiming exact cap (4096 bytes)
    bomb_payload: bytes = lz4_frame.compress(b"\x00" * (1024 * 1024), store_size=False)
    frame: bytes = _pack_v2_activation_frame(
        payload=bomb_payload,
        codec=CompressionCodec.LZ4,
        uncompressed_len=capped_limit,
    )

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceeded safety limit"):
            deserialize_activation_tensor(frame)
        _cur, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 512 * 1024


def test_lz4_rejects_truncated_trailing_concatenated_and_skippable_frames() -> None:
    """Truncated LZ4 frames, trailing bytes, extra frames, and skippable frames must raise ValueError."""
    raw: bytes = b"X" * 128
    comp: bytes = lz4_frame.compress(raw)

    # 1. Truncated LZ4 frame (missing endmark)
    truncated_frame: bytes = _pack_v2_activation_frame(
        payload=comp[:-4],
        codec=CompressionCodec.LZ4,
        uncompressed_len=len(raw),
    )
    with pytest.raises(ValueError, match="Incomplete LZ4 frame"):
        deserialize_activation_tensor(truncated_frame)

    # 2. Trailing garbage after end of LZ4 frame
    trailing_frame: bytes = _pack_v2_activation_frame(
        payload=comp + b"TRAILING",
        codec=CompressionCodec.LZ4,
        uncompressed_len=len(raw),
    )
    with pytest.raises(ValueError, match="Unexpected trailing data"):
        deserialize_activation_tensor(trailing_frame)

    # 3. Concatenated second LZ4 frame
    concat_frame: bytes = _pack_v2_activation_frame(
        payload=comp + lz4_frame.compress(b"EXTRA"),
        codec=CompressionCodec.LZ4,
        uncompressed_len=len(raw),
    )
    with pytest.raises(ValueError, match="Unexpected trailing data"):
        deserialize_activation_tensor(concat_frame)

    # 4. Skippable LZ4 frame (magic 0x184D2A50, little-endian size, payload)
    skippable_body: bytes = b"skip"
    skippable_payload: bytes = struct.pack("<II", 0x184D2A50, len(skippable_body)) + skippable_body
    skippable_frame: bytes = _pack_v2_activation_frame(
        payload=skippable_payload,
        codec=CompressionCodec.LZ4,
        uncompressed_len=0,
    )
    with pytest.raises(ValueError, match="skippable LZ4 frame"):
        deserialize_activation_tensor(skippable_frame)


def test_zlib_bounded_decompression_rejects_bombs_and_zero_length_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zlib decompression must reject bombs, trailing bytes, and 0-byte unbounded max_length bypass."""
    # Valid 0-byte and non-empty round-trips
    assert _decompress_buffer(zlib.compress(b""), CompressionCodec.ZLIB, 0) == b""
    sample: bytes = b"TENSOR" * 64
    assert _decompress_buffer(zlib.compress(sample), CompressionCodec.ZLIB, len(sample)) == sample

    # 0-byte declared size with non-empty zlib payload must NOT trigger zlib max_length=0 (unbounded)
    non_empty_zlib: bytes = zlib.compress(b"B" * 1024)
    zero_bypass_frame: bytes = _pack_v2_activation_frame(
        payload=non_empty_zlib,
        codec=CompressionCodec.ZLIB,
        uncompressed_len=0,
    )
    with pytest.raises(ValueError, match="exceeded safety limit|size mismatch"):
        deserialize_activation_tensor(zero_bypass_frame)

    # Zlib bomb capped at expected_uncompressed_size (< MAX_UNCOMPRESSED_TENSOR_BYTES)
    monkeypatch.setattr(serializer_module, "MAX_UNCOMPRESSED_TENSOR_BYTES", 4096)
    zlib_bomb: bytes = zlib.compress(b"\x00" * (256 * 1024))
    bomb_frame: bytes = _pack_v2_activation_frame(
        payload=zlib_bomb,
        codec=CompressionCodec.ZLIB,
        uncompressed_len=64,
    )
    with pytest.raises(ValueError, match="exceeded safety limit"):
        deserialize_activation_tensor(bomb_frame)


@pytest.mark.asyncio
async def test_handle_incoming_p2p_stream_offloads_deserialization_and_nacks_lz4_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incoming TCP stream handler must offload deserialization to a thread and NACK LZ4 bombs."""
    loop_thread_id: int = threading.get_ident()
    deserializer_threads: list[int] = []
    handler_calls: int = 0

    orig_deserialize = transport_module.deserialize_activation_tensor

    def _spy_deserialize(frame_bytes: bytes) -> Any:
        deserializer_threads.append(threading.get_ident())
        return orig_deserialize(frame_bytes)

    monkeypatch.setattr(transport_module, "deserialize_activation_tensor", _spy_deserialize)

    async def _on_activation(_frame: Any) -> None:
        nonlocal handler_calls
        handler_calls += 1

    server = await receive_activations(
        host="127.0.0.1",
        port=0,
        on_activation_frame=_on_activation,
    )
    assert server.sockets
    port: int = int(server.sockets[0].getsockname()[1])

    async def _send_raw(payload: bytes) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(FRAME_LEN_STRUCT.pack(len(payload)) + payload)
            await writer.drain()
            return await asyncio.wait_for(reader.readexactly(1), timeout=2.0)
        finally:
            writer.close()
            await writer.wait_closed()

    try:
        # 1. Valid compressed frame runs deserialization on a worker thread and ACKs
        valid_raw: bytes = array("f", [0.5] * 256).tobytes()
        valid_frame: bytes = serialize_activation_tensor(
            tensor_data=valid_raw,
            shape=(16, 16),
            request_id=str(uuid.uuid4()),
            origin_ip="127.0.0.1",
            origin_port=49152,
            current_layer=15,
            total_layers=32,
            step_index=0,
            force_compression=True,
        )
        assert await _send_raw(valid_frame) == ACK_BYTE
        assert handler_calls == 1
        assert len(deserializer_threads) == 1
        assert deserializer_threads[0] != loop_thread_id

        # 2. Oversized LZ4 bomb frame is NACKed on worker thread without invoking handler
        bomb_payload: bytes = lz4_frame.compress(b"\x00" * (128 * 1024 * 1024), store_size=False)
        bomb_frame: bytes = _pack_v2_activation_frame(
            payload=bomb_payload,
            codec=CompressionCodec.LZ4,
            uncompressed_len=64,
        )
        assert await _send_raw(bomb_frame) == NACK_BYTE
        assert handler_calls == 1
        assert len(deserializer_threads) == 2
        assert deserializer_threads[1] != loop_thread_id
    finally:
        server.close()
        await server.wait_closed()
