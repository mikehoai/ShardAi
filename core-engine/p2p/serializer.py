"""Ultra-Fast Binary Struct Tensor Serializer with Adaptive LZ4/Zlib Compression.

Complies with `.antigravity/rules.md` and `.antigravity/review.md`:
- Strictly NO JSON and NO `pickle` for tensor serialization.
- Uses fixed-size big-endian binary `struct` framing (`SHRD` magic header).
- Integrates automatic `lz4` (preferred if available) or `zlib` (`Z_BEST_SPEED`)
  compression before socket transmission to eliminate multi-megabyte hidden-state
  latency bottlenecks when network congestion or large tensors are detected.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import socket
import struct
from typing import Any
import uuid
import zlib

try:
    import lz4.frame as lz4_frame  # type: ignore[import-not-found]

    HAS_LZ4: bool = True
except ImportError:
    lz4_frame = None
    HAS_LZ4 = False

MAGIC_HEADER: bytes = b"SHRD"

# Safety cap: reject any P2P frame claiming > 64 MB uncompressed tensor size
MAX_UNCOMPRESSED_TENSOR_BYTES: int = 64 * 1024 * 1024

# Automatically compress hidden states >= 32 KB or whenever socket RTT indicates congestion
DEFAULT_COMPRESSION_THRESHOLD_BYTES: int = 32 * 1024


class PacketType(IntEnum):
    """Wire message types for the ShardAi binary P2P protocol."""

    ACTIVATION_TENSOR = 1
    DECODED_TOKEN = 2


class CompressionCodec(IntEnum):
    """Binary payload compression codec flags."""

    NONE = 0
    ZLIB = 1
    LZ4 = 2


DEFAULT_TEMPERATURE: float = 0.7

# Legacy 52-byte v1 big-endian binary header (no temperature field; kept for backward compatibility)
LEGACY_ACTIVATION_HEADER_STRUCT = struct.Struct("!4sBB16s4sHHHHHIIII")

# 60-byte v2 big-endian binary header for ACTIVATION_TENSOR frames:
# - 4s : magic ("SHRD")
# - B  : packet_type (PacketType.ACTIVATION_TENSOR = 1)
# - B  : compression_codec (0=NONE, 1=ZLIB, 2=LZ4)
# - 16s: request_id (16-byte UUID)
# - 4s : origin_ip (IPv4 packed bytes)
# - H  : origin_port (uint16)
# - H  : current_layer (uint16)
# - H  : total_layers (uint16)
# - H  : step_index (uint16)
# - H  : max_tokens (uint16)
# - I  : shape_dim0 / seq_len (uint32)
# - I  : shape_dim1 / hidden_dim (uint32)
# - I  : uncompressed_bytes (uint32)
# - I  : crc32_checksum (uint32)
# - d  : temperature (IEEE-754 float64)
ACTIVATION_HEADER_STRUCT = struct.Struct("!4sBB16s4sHHHHHIIIId")

# 28-byte big-endian binary header for DECODED_TOKEN return frames:
# - 4s : magic ("SHRD")
# - B  : packet_type (PacketType.DECODED_TOKEN = 2)
# - B  : is_eos (0 or 1)
# - 16s: request_id (16-byte UUID)
# - H  : step_index (uint16)
# - I  : token_id (uint32)
TOKEN_HEADER_STRUCT = struct.Struct("!4sBB16sHI")


@dataclass(frozen=True, slots=True)
class ActivationFrame:
    """Deserialized binary activation tensor frame received from an upstream peer."""

    request_id: str
    origin_ip: str
    origin_port: int
    current_layer: int
    total_layers: int
    step_index: int
    max_tokens: int
    shape: tuple[int, int]
    raw_tensor_bytes: bytes
    compression_used: CompressionCodec
    temperature: float = DEFAULT_TEMPERATURE


@dataclass(frozen=True, slots=True)
class DecodedTokenFrame:
    """Deserialized binary token packet returned from the Tail Node to the Head Node."""

    request_id: str
    step_index: int
    token_id: int
    token_text: str
    is_eos: bool


def extract_contiguous_float32_bytes(
    tensor_or_buffer: bytes | bytearray | memoryview | Any,
) -> bytes:
    """Convert a NumPy ndarray, PyTorch Tensor, or byte buffer into contiguous raw bytes."""
    if isinstance(tensor_or_buffer, bytes):
        return tensor_or_buffer
    if isinstance(tensor_or_buffer, (bytearray, memoryview)):
        return bytes(tensor_or_buffer)

    # Support PyTorch Tensor without hard dependency
    if hasattr(tensor_or_buffer, "detach") and hasattr(tensor_or_buffer, "cpu"):
        cpu_tensor = tensor_or_buffer.detach().contiguous().cpu()
        return bytes(cpu_tensor.numpy().tobytes())

    # Support NumPy ndarray or array.array
    if hasattr(tensor_or_buffer, "tobytes"):
        return bytes(tensor_or_buffer.tobytes())

    raise TypeError(
        f"Unsupported tensor object for binary serialization: {type(tensor_or_buffer)!r}"
    )


def _compress_buffer(
    raw_bytes: bytes,
    prefer_lz4: bool = True,
) -> tuple[bytes, CompressionCodec]:
    """Compress raw tensor bytes using LZ4 (if installed) or fast C-level Zlib (level 1)."""
    if prefer_lz4 and HAS_LZ4 and lz4_frame is not None:
        compressed: bytes = lz4_frame.compress(raw_bytes)
        return compressed, CompressionCodec.LZ4

    # Z_BEST_SPEED (level=1) maximizes throughput for real-time P2P hidden state streaming
    compressed = zlib.compress(raw_bytes, level=zlib.Z_BEST_SPEED)
    return compressed, CompressionCodec.ZLIB


def _decompress_buffer(
    payload_bytes: bytes,
    codec: CompressionCodec,
    expected_uncompressed_size: int,
) -> bytes:
    """Decompress a binary tensor buffer with strict bounds verification."""
    if expected_uncompressed_size > MAX_UNCOMPRESSED_TENSOR_BYTES:
        raise ValueError(
            f"Rejected P2P tensor exceeding safety cap: {expected_uncompressed_size} bytes"
        )

    if codec == CompressionCodec.NONE:
        raw_out = payload_bytes
    elif codec == CompressionCodec.LZ4:
        if not HAS_LZ4 or lz4_frame is None:
            raise RuntimeError("Received LZ4-compressed frame but 'lz4' package is not installed.")
        raw_out = lz4_frame.decompress(payload_bytes)
    elif codec == CompressionCodec.ZLIB:
        decomp = zlib.decompressobj()
        raw_out = decomp.decompress(payload_bytes, max_length=MAX_UNCOMPRESSED_TENSOR_BYTES)
        if decomp.unconsumed_tail:
            raise ValueError("Decompressed zlib payload exceeded safety limit.")
    else:
        raise ValueError(f"Unsupported compression codec: {codec}")

    if len(raw_out) != expected_uncompressed_size:
        raise ValueError(
            f"Tensor size mismatch after decompression: expected {expected_uncompressed_size}, got {len(raw_out)}"
        )
    return raw_out


def _validate_temperature(temperature: float) -> float:
    """Ensure sampling temperature is a finite non-negative float before packing or decoding."""
    temp_float: float = float(temperature)
    if not math.isfinite(temp_float) or temp_float < 0.0:
        raise ValueError(
            f"Invalid sampling temperature in activation frame: {temperature!r}"
        )
    return temp_float


def serialize_activation_tensor(
    tensor_data: bytes | bytearray | memoryview | Any,
    shape: tuple[int, int],
    request_id: str,
    origin_ip: str,
    origin_port: int,
    current_layer: int,
    total_layers: int,
    step_index: int,
    max_tokens: int = 16,
    network_congested: bool = False,
    force_compression: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
) -> bytes:
    """Pack a hidden-state tensor and routing metadata into a compact binary byte stream.

    Automatically engages LZ4 or Zlib compression when `force_compression` is True,
    `network_congested` is True, or the raw tensor size >= 32 KB.
    """
    validated_temperature: float = _validate_temperature(temperature)
    raw_bytes: bytes = extract_contiguous_float32_bytes(tensor_data)
    uncompressed_len: int = len(raw_bytes)
    if uncompressed_len > MAX_UNCOMPRESSED_TENSOR_BYTES:
        raise ValueError(
            f"Raw tensor buffer ({uncompressed_len} bytes) exceeds 64 MB maximum limit."
        )

    should_compress: bool = (
        force_compression
        or network_congested
        or uncompressed_len >= DEFAULT_COMPRESSION_THRESHOLD_BYTES
    )

    if should_compress and uncompressed_len > 0:
        wire_payload, codec = _compress_buffer(raw_bytes)
        # If incompressible noise slightly expands the buffer, send uncompressed
        if len(wire_payload) >= uncompressed_len:
            wire_payload = raw_bytes
            codec = CompressionCodec.NONE
    else:
        wire_payload = raw_bytes
        codec = CompressionCodec.NONE

    crc: int = zlib.crc32(wire_payload) & 0xFFFFFFFF
    req_uuid_bytes: bytes = uuid.UUID(request_id).bytes
    origin_ip_bytes: bytes = socket.inet_aton(origin_ip)

    header: bytes = ACTIVATION_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.ACTIVATION_TENSOR),
        int(codec),
        req_uuid_bytes,
        origin_ip_bytes,
        origin_port,
        current_layer,
        total_layers,
        step_index,
        max_tokens,
        shape[0],
        shape[1],
        uncompressed_len,
        crc,
        validated_temperature,
    )
    return header + wire_payload


def deserialize_activation_tensor(frame_bytes: bytes) -> ActivationFrame:
    """Unpack and verify a binary activation tensor frame from an upstream P2P node.

    Supports both the current 60-byte v2 header (with IEEE-754 float64 `temperature`)
    and legacy 52-byte v1 headers (defaulting `temperature` to `0.7`) via CRC32 verification.
    """
    if len(frame_bytes) < LEGACY_ACTIVATION_HEADER_STRUCT.size:
        raise ValueError(
            f"Truncated activation frame: expected >= {LEGACY_ACTIVATION_HEADER_STRUCT.size} bytes, "
            f"got {len(frame_bytes)}"
        )

    magic_prefix: bytes = frame_bytes[:4]
    if magic_prefix != MAGIC_HEADER:
        raise ValueError(f"Invalid P2P magic header: {magic_prefix!r}")
    pkt_type_byte: int = frame_bytes[4]
    if pkt_type_byte != int(PacketType.ACTIVATION_TENSOR):
        raise ValueError(
            f"Unexpected packet type for activation frame: {pkt_type_byte}"
        )

    v2_crc_mismatch: tuple[int, int] | None = None
    if len(frame_bytes) >= ACTIVATION_HEADER_STRUCT.size:
        header_slice = frame_bytes[: ACTIVATION_HEADER_STRUCT.size]
        payload_slice = frame_bytes[ACTIVATION_HEADER_STRUCT.size :]
        (
            _magic,
            _pkt_type,
            codec_val,
            req_uuid_bytes,
            origin_ip_bytes,
            origin_port,
            current_layer,
            total_layers,
            step_index,
            max_tokens,
            dim0,
            dim1,
            uncompressed_len,
            expected_crc,
            temperature,
        ) = ACTIVATION_HEADER_STRUCT.unpack(header_slice)

        actual_crc: int = zlib.crc32(payload_slice) & 0xFFFFFFFF
        if actual_crc == expected_crc:
            validated_temperature: float = _validate_temperature(temperature)
            codec = CompressionCodec(codec_val)
            raw_tensor = _decompress_buffer(payload_slice, codec, uncompressed_len)
            return ActivationFrame(
                request_id=str(uuid.UUID(bytes=req_uuid_bytes)),
                origin_ip=socket.inet_ntoa(origin_ip_bytes),
                origin_port=origin_port,
                current_layer=current_layer,
                total_layers=total_layers,
                step_index=step_index,
                max_tokens=max_tokens,
                shape=(dim0, dim1),
                raw_tensor_bytes=raw_tensor,
                compression_used=codec,
                temperature=validated_temperature,
            )
        v2_crc_mismatch = (expected_crc, actual_crc)

    # Fallback for legacy 52-byte v1 frames without trailing float64 temperature
    legacy_header_slice = frame_bytes[: LEGACY_ACTIVATION_HEADER_STRUCT.size]
    legacy_payload_slice = frame_bytes[LEGACY_ACTIVATION_HEADER_STRUCT.size :]
    (
        _magic,
        _pkt_type,
        codec_val,
        req_uuid_bytes,
        origin_ip_bytes,
        origin_port,
        current_layer,
        total_layers,
        step_index,
        max_tokens,
        dim0,
        dim1,
        uncompressed_len,
        expected_crc,
    ) = LEGACY_ACTIVATION_HEADER_STRUCT.unpack(legacy_header_slice)

    actual_legacy_crc: int = zlib.crc32(legacy_payload_slice) & 0xFFFFFFFF
    if actual_legacy_crc != expected_crc:
        reported_expected, reported_actual = (
            v2_crc_mismatch
            if v2_crc_mismatch is not None
            else (expected_crc, actual_legacy_crc)
        )
        raise ValueError(
            "Corrupted P2P tensor payload: CRC32 mismatch "
            f"(expected {reported_expected:#x}, got {reported_actual:#x})"
        )

    codec = CompressionCodec(codec_val)
    raw_tensor = _decompress_buffer(legacy_payload_slice, codec, uncompressed_len)

    return ActivationFrame(
        request_id=str(uuid.UUID(bytes=req_uuid_bytes)),
        origin_ip=socket.inet_ntoa(origin_ip_bytes),
        origin_port=origin_port,
        current_layer=current_layer,
        total_layers=total_layers,
        step_index=step_index,
        max_tokens=max_tokens,
        shape=(dim0, dim1),
        raw_tensor_bytes=raw_tensor,
        compression_used=codec,
        temperature=DEFAULT_TEMPERATURE,
    )


def serialize_decoded_token(
    request_id: str,
    step_index: int,
    token_id: int,
    token_text: str,
    is_eos: bool,
) -> bytes:
    """Pack a decoded token into a compact binary frame to send back to the Head Node."""
    req_uuid_bytes: bytes = uuid.UUID(request_id).bytes
    utf8_token: bytes = token_text.encode("utf-8")
    header: bytes = TOKEN_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.DECODED_TOKEN),
        1 if is_eos else 0,
        req_uuid_bytes,
        step_index,
        token_id,
    )
    return header + utf8_token


def deserialize_decoded_token(frame_bytes: bytes) -> DecodedTokenFrame:
    """Unpack a binary token frame received from the Tail Node."""
    if len(frame_bytes) < TOKEN_HEADER_STRUCT.size:
        raise ValueError("Truncated decoded token frame.")

    header_slice = frame_bytes[: TOKEN_HEADER_STRUCT.size]
    utf8_slice = frame_bytes[TOKEN_HEADER_STRUCT.size :]

    magic, pkt_type, eos_flag, req_uuid_bytes, step_index, token_id = (
        TOKEN_HEADER_STRUCT.unpack(header_slice)
    )
    if magic != MAGIC_HEADER or pkt_type != int(PacketType.DECODED_TOKEN):
        raise ValueError("Invalid header in decoded token frame.")

    return DecodedTokenFrame(
        request_id=str(uuid.UUID(bytes=req_uuid_bytes)),
        step_index=step_index,
        token_id=token_id,
        token_text=utf8_slice.decode("utf-8", errors="replace"),
        is_eos=bool(eos_flag),
    )
