"""Binary envelope framing and chunking for the Signaling Tracker relay protocol.

Every WebSocket binary message uses a fixed 28-byte big-endian header (`SHRL`)
followed by UTF-8 `src` node ID, UTF-8 `dst` node ID, and the frame payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
import uuid

from p2p.transport import MAX_WIRE_FRAME_BYTES

RELAY_MAGIC: bytes = b"SHRL"
RELAY_VERSION: int = 1
RELAY_HEADER_STRUCT: struct.Struct = struct.Struct("!4sBB16sHHBB")

MAX_DATA_CHUNK_BYTES: int = 512 * 1024
MAX_NACK_REASON_BYTES: int = 1024
MAX_CHUNKS: int = 256
MAX_NODE_ID_BYTES: int = 64
MAX_RELAY_MESSAGE_BYTES: int = 1024 * 1024


class RelayPacketKind(IntEnum):
    """Packet kinds carried in the Signaling Tracker relay binary envelope."""

    DATA = 1
    ACK = 2
    NACK = 3
    NO_ROUTE = 4
    PING = 5
    PONG = 6


@dataclass(frozen=True, slots=True)
class RelayEnvelope:
    """Decoded binary relay envelope."""

    kind: RelayPacketKind
    msg_id: bytes
    chunk_index: int
    chunk_count: int
    src: str
    dst: str
    payload: bytes


def _normalize_msg_id(msg_id: bytes | uuid.UUID | str) -> bytes:
    """Validate and normalize `msg_id` to a 16-byte UUID representation."""
    if isinstance(msg_id, uuid.UUID):
        return msg_id.bytes
    if isinstance(msg_id, str):
        return uuid.UUID(msg_id).bytes
    if isinstance(msg_id, (bytes, bytearray)):
        raw = bytes(msg_id)
        if len(raw) != 16:
            raise ValueError(f"msg_id must be exactly 16 bytes, got {len(raw)}")
        return raw
    raise TypeError(f"Unsupported msg_id type: {type(msg_id).__name__}")


def truncate_utf8_bytes(text: str, max_bytes: int = MAX_NACK_REASON_BYTES) -> bytes:
    """Encode `text` as UTF-8 and truncate cleanly at a character boundary <= `max_bytes`."""
    if max_bytes <= 0:
        return b""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return encoded
    clipped = encoded[:max_bytes]
    while clipped:
        try:
            clipped.decode("utf-8")
            return clipped
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return b""


def _validate_envelope_fields(
    *,
    kind: RelayPacketKind,
    msg_id: bytes,
    chunk_index: int,
    chunk_count: int,
    src_bytes: bytes,
    dst_bytes: bytes,
    payload: bytes,
) -> None:
    """Validate protocol invariants for both encoding and decoding."""
    if len(msg_id) != 16:
        raise ValueError(f"msg_id must be 16 bytes, got {len(msg_id)}")
    if not (1 <= chunk_count <= MAX_CHUNKS):
        raise ValueError(f"chunk_count out of range (1..{MAX_CHUNKS}): {chunk_count}")
    if not (0 <= chunk_index < chunk_count):
        raise ValueError(
            f"chunk_index {chunk_index} out of range for chunk_count {chunk_count}"
        )
    if not (1 <= len(src_bytes) <= MAX_NODE_ID_BYTES):
        raise ValueError(
            f"src node_id UTF-8 length must be 1..{MAX_NODE_ID_BYTES}, got {len(src_bytes)}"
        )
    if not (1 <= len(dst_bytes) <= MAX_NODE_ID_BYTES):
        raise ValueError(
            f"dst node_id UTF-8 length must be 1..{MAX_NODE_ID_BYTES}, got {len(dst_bytes)}"
        )

    if kind == RelayPacketKind.DATA:
        if len(payload) > MAX_DATA_CHUNK_BYTES:
            raise ValueError(
                f"DATA chunk payload {len(payload)} exceeds {MAX_DATA_CHUNK_BYTES} bytes"
            )
    elif kind == RelayPacketKind.NACK:
        if chunk_index != 0 or chunk_count != 1:
            raise ValueError("NACK envelopes must have chunk_index=0 and chunk_count=1")
        if len(payload) > MAX_NACK_REASON_BYTES:
            raise ValueError(
                f"NACK reason payload {len(payload)} exceeds {MAX_NACK_REASON_BYTES} bytes"
            )
    else:
        if chunk_index != 0 or chunk_count != 1:
            raise ValueError(
                f"{kind.name} envelopes must have chunk_index=0 and chunk_count=1"
            )
        if len(payload) != 0:
            raise ValueError(
                f"{kind.name} envelopes must have empty payload, got {len(payload)} bytes"
            )

    total_size = RELAY_HEADER_STRUCT.size + len(src_bytes) + len(dst_bytes) + len(payload)
    if total_size > MAX_RELAY_MESSAGE_BYTES:
        raise ValueError(
            f"Relay message size {total_size} exceeds {MAX_RELAY_MESSAGE_BYTES} bytes"
        )


def encode_envelope(
    kind: RelayPacketKind | int,
    msg_id: bytes | uuid.UUID | str,
    src: str,
    dst: str,
    payload: bytes = b"",
    *,
    chunk_index: int = 0,
    chunk_count: int = 1,
) -> bytes:
    """Serialize a `RelayEnvelope` into a wire-ready binary WebSocket message."""
    packet_kind = RelayPacketKind(kind)
    msg_id_bytes = _normalize_msg_id(msg_id)
    src_bytes = src.encode("utf-8")
    dst_bytes = dst.encode("utf-8")
    payload_bytes = bytes(payload)

    _validate_envelope_fields(
        kind=packet_kind,
        msg_id=msg_id_bytes,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        payload=payload_bytes,
    )

    header = RELAY_HEADER_STRUCT.pack(
        RELAY_MAGIC,
        RELAY_VERSION,
        int(packet_kind),
        msg_id_bytes,
        chunk_index,
        chunk_count,
        len(src_bytes),
        len(dst_bytes),
    )
    return header + src_bytes + dst_bytes + payload_bytes


def decode_envelope(data: bytes) -> RelayEnvelope:
    """Parse and validate a binary relay WebSocket message into a `RelayEnvelope`."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("Relay envelope must be binary bytes")
    raw = bytes(data)
    if len(raw) > MAX_RELAY_MESSAGE_BYTES:
        raise ValueError(
            f"Relay message length {len(raw)} exceeds {MAX_RELAY_MESSAGE_BYTES} bytes"
        )
    if len(raw) < RELAY_HEADER_STRUCT.size:
        raise ValueError(
            f"Relay message truncated: expected >= {RELAY_HEADER_STRUCT.size} bytes, got {len(raw)}"
        )

    (
        magic,
        version,
        raw_kind,
        msg_id_bytes,
        chunk_index,
        chunk_count,
        src_len,
        dst_len,
    ) = RELAY_HEADER_STRUCT.unpack_from(raw, 0)

    if magic != RELAY_MAGIC:
        raise ValueError(f"Invalid relay magic header: {magic!r}")
    if version != RELAY_VERSION:
        raise ValueError(f"Unsupported relay protocol version: {version}")
    try:
        packet_kind = RelayPacketKind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"Unsupported relay packet kind: {raw_kind}") from exc

    body_offset = RELAY_HEADER_STRUCT.size
    min_required = body_offset + src_len + dst_len
    if len(raw) < min_required:
        raise ValueError(
            f"Relay message truncated before src/dst node IDs: {len(raw)} < {min_required}"
        )

    src_bytes = raw[body_offset : body_offset + src_len]
    dst_bytes = raw[body_offset + src_len : min_required]
    payload = raw[min_required:]

    _validate_envelope_fields(
        kind=packet_kind,
        msg_id=msg_id_bytes,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        src_bytes=src_bytes,
        dst_bytes=dst_bytes,
        payload=payload,
    )

    try:
        src = src_bytes.decode("utf-8")
        dst = dst_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Relay src/dst node_id is not valid UTF-8") from exc

    if packet_kind == RelayPacketKind.NACK:
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Relay NACK reason payload is not valid UTF-8") from exc

    return RelayEnvelope(
        kind=packet_kind,
        msg_id=msg_id_bytes,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        src=src,
        dst=dst,
        payload=payload,
    )


def split_into_chunks(
    frame_bytes: bytes,
    chunk_size: int = MAX_DATA_CHUNK_BYTES,
) -> list[bytes]:
    """Split a P2P wire frame into bounded relay DATA chunks (1..256 chunks)."""
    if chunk_size <= 0 or chunk_size > MAX_DATA_CHUNK_BYTES:
        raise ValueError(
            f"chunk_size must be in 1..{MAX_DATA_CHUNK_BYTES}, got {chunk_size}"
        )
    total_len = len(frame_bytes)
    if total_len <= 0:
        raise ValueError("Cannot split an empty frame into relay chunks")
    if total_len > MAX_WIRE_FRAME_BYTES:
        raise ValueError(
            f"Frame size {total_len} exceeds MAX_WIRE_FRAME_BYTES ({MAX_WIRE_FRAME_BYTES})"
        )

    chunks = [
        bytes(frame_bytes[offset : offset + chunk_size])
        for offset in range(0, total_len, chunk_size)
    ]
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"Frame requires {len(chunks)} chunks, exceeding maximum of {MAX_CHUNKS}"
        )
    return chunks
