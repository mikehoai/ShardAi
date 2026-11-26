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
import json
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

# Safety caps for proxied inference control frames
MAX_INFERENCE_REQUEST_PAYLOAD_BYTES: int = 2 * 1024 * 1024
MAX_INFERENCE_ERROR_BYTES: int = 16 * 1024
MAX_PROXIED_TOKENS: int = 4096

# Automatically compress hidden states >= 32 KB or whenever socket RTT indicates congestion
DEFAULT_COMPRESSION_THRESHOLD_BYTES: int = 32 * 1024


class PacketType(IntEnum):
    """Wire message types for the ShardAi binary P2P protocol."""

    ACTIVATION_TENSOR = 1
    DECODED_TOKEN = 2
    INFERENCE_REQUEST = 3
    INFERENCE_ERROR = 4


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

# 42-byte big-endian binary header for INFERENCE_REQUEST frames:
# - 4s : magic ("SHRD")
# - B  : packet_type (PacketType.INFERENCE_REQUEST = 3)
# - B  : flags (bit 0 = compress_activations)
# - 16s: request_id (16-byte UUID)
# - 4s : requester_ip (IPv4 packed bytes)
# - H  : requester_port (uint16)
# - H  : max_tokens (uint16)
# - d  : temperature (IEEE-754 float64)
# - I  : crc32_checksum (uint32 of UTF-8 JSON body)
INFERENCE_REQUEST_HEADER_STRUCT = struct.Struct("!4sBB16s4sHHdI")

# 24-byte big-endian binary header for INFERENCE_ERROR frames:
# - 4s : magic ("SHRD")
# - B  : packet_type (PacketType.INFERENCE_ERROR = 4)
# - B  : reserved (0)
# - 16s: request_id (16-byte UUID)
# - H  : status_code (uint16)
INFERENCE_ERROR_HEADER_STRUCT = struct.Struct("!4sBB16sH")


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


@dataclass(frozen=True, slots=True)
class InferenceRequestFrame:
    """Deserialized binary inference request proxied from a non-Head peer to a Head Node."""

    request_id: str
    requester_ip: str
    requester_port: int
    prompt: str
    messages: list[dict[str, Any]] | None
    max_tokens: int
    temperature: float
    compress_activations: bool


@dataclass(frozen=True, slots=True)
class InferenceErrorFrame:
    """Deserialized binary inference error packet returned from a Head Node to a requester."""

    request_id: str
    status_code: int
    detail: str


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


def _decompress_lz4_bounded(payload_bytes: bytes, max_allowed_bytes: int) -> bytes:
    """Stream-decompress an LZ4 frame while capping output to `max_allowed_bytes`."""
    if not HAS_LZ4 or lz4_frame is None:
        raise RuntimeError("Received LZ4-compressed frame but 'lz4' package is not installed.")

    try:
        frame_info: dict[str, Any] = lz4_frame.get_frame_info(payload_bytes)
    except RuntimeError as exc:
        raise ValueError(f"Malformed LZ4 frame header: {exc}") from exc

    if frame_info.get("skippable"):
        raise ValueError("Rejected skippable LZ4 frame in activation tensor payload.")

    declared_content_size: int = int(frame_info.get("content_size", 0))
    if declared_content_size > max_allowed_bytes:
        raise ValueError(
            "Decompressed LZ4 payload exceeded safety limit: "
            f"frame header declares {declared_content_size} bytes > {max_allowed_bytes} bytes."
        )

    decomp = lz4_frame.LZ4FrameDecompressor()
    try:
        raw_out: bytes = decomp.decompress(payload_bytes, max_length=max_allowed_bytes)
    except RuntimeError as exc:
        raise ValueError(f"Corrupted or invalid LZ4 payload: {exc}") from exc

    if not decomp.needs_input:
        raise ValueError(
            f"Decompressed LZ4 payload exceeded safety limit ({max_allowed_bytes} bytes)."
        )
    if not decomp.eof:
        raise ValueError("Incomplete LZ4 frame: missing end-of-frame marker.")
    if decomp.unused_data:
        raise ValueError("Unexpected trailing data after LZ4 frame.")

    return raw_out


def _decompress_zlib_bounded(payload_bytes: bytes, max_allowed_bytes: int) -> bytes:
    """Decompress a zlib stream while capping output to `max_allowed_bytes`."""
    decomp = zlib.decompressobj()
    # zlib.decompressobj.decompress treats max_length=0 as unbounded, so clamp to >= 1.
    zlib_max_length: int = max(1, max_allowed_bytes)
    try:
        raw_out: bytes = decomp.decompress(payload_bytes, max_length=zlib_max_length)
    except zlib.error as exc:
        raise ValueError(f"Corrupted or invalid zlib payload: {exc}") from exc

    if decomp.unconsumed_tail:
        raise ValueError(
            f"Decompressed zlib payload exceeded safety limit ({max_allowed_bytes} bytes)."
        )
    if not decomp.eof:
        raise ValueError("Incomplete zlib stream: missing end-of-stream marker.")
    if decomp.unused_data:
        raise ValueError("Unexpected trailing data after zlib stream.")
    if len(raw_out) > max_allowed_bytes:
        raise ValueError(
            f"Decompressed zlib payload exceeded safety limit ({max_allowed_bytes} bytes)."
        )
    return raw_out


def _decompress_buffer(
    payload_bytes: bytes,
    codec: CompressionCodec,
    expected_uncompressed_size: int,
) -> bytes:
    """Decompress a binary tensor buffer with strict bounds verification."""
    if expected_uncompressed_size < 0 or expected_uncompressed_size > MAX_UNCOMPRESSED_TENSOR_BYTES:
        raise ValueError(
            f"Rejected P2P tensor exceeding safety cap: {expected_uncompressed_size} bytes"
        )
    max_allowed_bytes: int = min(expected_uncompressed_size, MAX_UNCOMPRESSED_TENSOR_BYTES)

    if codec == CompressionCodec.NONE:
        raw_out = payload_bytes
    elif codec == CompressionCodec.LZ4:
        raw_out = _decompress_lz4_bounded(payload_bytes, max_allowed_bytes)
    elif codec == CompressionCodec.ZLIB:
        raw_out = _decompress_zlib_bounded(payload_bytes, max_allowed_bytes)
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


def serialize_inference_request(
    request_id: str,
    requester_ip: str,
    requester_port: int,
    prompt: str,
    messages: list[dict[str, Any]] | None = None,
    max_tokens: int = 256,
    temperature: float = DEFAULT_TEMPERATURE,
    compress_activations: bool = True,
) -> bytes:
    """Pack a proxied inference request into a bounded binary frame for a Head Node peer."""
    if not (1 <= requester_port <= 65535):
        raise ValueError(f"Invalid requester_port in inference request: {requester_port}")
    if not (1 <= max_tokens <= MAX_PROXIED_TOKENS):
        raise ValueError(
            f"max_tokens must be between 1 and {MAX_PROXIED_TOKENS}, got {max_tokens}"
        )
    validated_temperature: float = _validate_temperature(temperature)
    req_uuid_bytes: bytes = uuid.UUID(request_id).bytes
    requester_ip_bytes: bytes = socket.inet_aton(requester_ip)

    body_bytes: bytes = json.dumps(
        {"prompt": prompt, "messages": messages},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(body_bytes) > MAX_INFERENCE_REQUEST_PAYLOAD_BYTES:
        raise ValueError(
            f"Inference request payload ({len(body_bytes)} bytes) exceeds "
            f"{MAX_INFERENCE_REQUEST_PAYLOAD_BYTES} byte limit."
        )

    crc: int = zlib.crc32(body_bytes) & 0xFFFFFFFF
    flags: int = 0x01 if compress_activations else 0x00
    header: bytes = INFERENCE_REQUEST_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.INFERENCE_REQUEST),
        flags,
        req_uuid_bytes,
        requester_ip_bytes,
        requester_port,
        max_tokens,
        validated_temperature,
        crc,
    )
    return header + body_bytes


def deserialize_inference_request(frame_bytes: bytes) -> InferenceRequestFrame:
    """Unpack and validate a binary INFERENCE_REQUEST frame received from a peer."""
    if len(frame_bytes) < INFERENCE_REQUEST_HEADER_STRUCT.size:
        raise ValueError("Truncated inference request frame.")

    header_slice = frame_bytes[: INFERENCE_REQUEST_HEADER_STRUCT.size]
    payload_slice = frame_bytes[INFERENCE_REQUEST_HEADER_STRUCT.size :]

    (
        magic,
        pkt_type,
        flags,
        req_uuid_bytes,
        requester_ip_bytes,
        requester_port,
        max_tokens,
        temperature,
        expected_crc,
    ) = INFERENCE_REQUEST_HEADER_STRUCT.unpack(header_slice)

    if magic != MAGIC_HEADER or pkt_type != int(PacketType.INFERENCE_REQUEST):
        raise ValueError("Invalid header in inference request frame.")

    if len(payload_slice) > MAX_INFERENCE_REQUEST_PAYLOAD_BYTES:
        raise ValueError(
            f"Inference request payload ({len(payload_slice)} bytes) exceeds "
            f"{MAX_INFERENCE_REQUEST_PAYLOAD_BYTES} byte limit."
        )

    actual_crc: int = zlib.crc32(payload_slice) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            "Corrupted inference request payload: CRC32 mismatch "
            f"(expected {expected_crc:#x}, got {actual_crc:#x})"
        )

    if not (1 <= requester_port <= 65535):
        raise ValueError(f"Invalid requester_port in inference request: {requester_port}")
    if not (1 <= max_tokens <= MAX_PROXIED_TOKENS):
        raise ValueError(
            f"Invalid max_tokens in inference request: {max_tokens} (max {MAX_PROXIED_TOKENS})"
        )
    validated_temperature: float = _validate_temperature(temperature)

    try:
        decoded_body: Any = json.loads(payload_slice.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON payload in inference request frame: {exc}") from exc

    if not isinstance(decoded_body, dict):
        raise ValueError("Inference request payload must be a JSON object.")

    prompt = decoded_body.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Inference request JSON payload must include a string 'prompt'.")

    messages = decoded_body.get("messages")
    if messages is not None:
        if not isinstance(messages, list) or not all(
            isinstance(item, dict) for item in messages
        ):
            raise ValueError(
                "Inference request 'messages' must be None or a list of JSON objects."
            )

    return InferenceRequestFrame(
        request_id=str(uuid.UUID(bytes=req_uuid_bytes)),
        requester_ip=socket.inet_ntoa(requester_ip_bytes),
        requester_port=requester_port,
        prompt=prompt,
        messages=messages,
        max_tokens=max_tokens,
        temperature=validated_temperature,
        compress_activations=bool(flags & 0x01),
    )


def serialize_inference_error(
    request_id: str,
    status_code: int,
    detail: str,
) -> bytes:
    """Pack a terminal inference error frame to send back to a proxying peer."""
    if not (0 <= status_code <= 65535):
        raise ValueError(f"Invalid status_code in inference error frame: {status_code}")

    req_uuid_bytes: bytes = uuid.UUID(request_id).bytes
    detail_bytes: bytes = detail.encode("utf-8")
    if len(detail_bytes) > MAX_INFERENCE_ERROR_BYTES:
        detail_bytes = (
            detail_bytes[:MAX_INFERENCE_ERROR_BYTES]
            .decode("utf-8", errors="ignore")
            .encode("utf-8")
        )

    header: bytes = INFERENCE_ERROR_HEADER_STRUCT.pack(
        MAGIC_HEADER,
        int(PacketType.INFERENCE_ERROR),
        0,
        req_uuid_bytes,
        status_code,
    )
    return header + detail_bytes


def deserialize_inference_error(frame_bytes: bytes) -> InferenceErrorFrame:
    """Unpack a binary INFERENCE_ERROR frame received from a Head Node peer."""
    if len(frame_bytes) < INFERENCE_ERROR_HEADER_STRUCT.size:
        raise ValueError("Truncated inference error frame.")

    header_slice = frame_bytes[: INFERENCE_ERROR_HEADER_STRUCT.size]
    detail_slice = frame_bytes[INFERENCE_ERROR_HEADER_STRUCT.size :]

    magic, pkt_type, _reserved, req_uuid_bytes, status_code = (
        INFERENCE_ERROR_HEADER_STRUCT.unpack(header_slice)
    )
    if magic != MAGIC_HEADER or pkt_type != int(PacketType.INFERENCE_ERROR):
        raise ValueError("Invalid header in inference error frame.")

    if len(detail_slice) > MAX_INFERENCE_ERROR_BYTES:
        raise ValueError(
            f"Inference error detail ({len(detail_slice)} bytes) exceeds "
            f"{MAX_INFERENCE_ERROR_BYTES} byte limit."
        )

    return InferenceErrorFrame(
        request_id=str(uuid.UUID(bytes=req_uuid_bytes)),
        status_code=status_code,
        detail=detail_slice.decode("utf-8", errors="replace"),
    )
