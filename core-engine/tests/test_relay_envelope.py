"""Unit tests for the binary relay envelope protocol (`p2p.relay_envelope`)."""

from __future__ import annotations

import struct
import uuid

import pytest

from p2p.relay_envelope import (
    MAX_CHUNKS,
    MAX_DATA_CHUNK_BYTES,
    MAX_NACK_REASON_BYTES,
    MAX_NODE_ID_BYTES,
    RELAY_HEADER_STRUCT,
    RELAY_MAGIC,
    RELAY_VERSION,
    RelayEnvelope,
    RelayPacketKind,
    decode_envelope,
    encode_envelope,
    split_into_chunks,
    truncate_utf8_bytes,
)
from p2p.transport import MAX_WIRE_FRAME_BYTES


def test_relay_header_struct_size_and_constants() -> None:
    """Header layout must be exactly 28 bytes with expected magic and version."""
    assert RELAY_MAGIC == b"SHRL"
    assert RELAY_VERSION == 1
    assert RELAY_HEADER_STRUCT.size == 28


@pytest.mark.parametrize(
    ("kind", "payload", "chunk_index", "chunk_count"),
    [
        (RelayPacketKind.DATA, b"hello-tensor-chunk", 0, 1),
        (RelayPacketKind.DATA, b"chunk-two", 1, 3),
        (RelayPacketKind.ACK, b"", 0, 1),
        (RelayPacketKind.NACK, b"unauthorized peer", 0, 1),
        (RelayPacketKind.NO_ROUTE, b"", 0, 1),
        (RelayPacketKind.PING, b"", 0, 1),
        (RelayPacketKind.PONG, b"", 0, 1),
    ],
)
def test_encode_decode_round_trip(
    kind: RelayPacketKind,
    payload: bytes,
    chunk_index: int,
    chunk_count: int,
) -> None:
    """All relay packet kinds must round-trip byte-for-byte."""
    msg_id = uuid.uuid4().bytes
    encoded = encode_envelope(
        kind=kind,
        msg_id=msg_id,
        src="node-head-01",
        dst="node-tail-02",
        payload=payload,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
    )
    decoded = decode_envelope(encoded)
    assert decoded == RelayEnvelope(
        kind=kind,
        msg_id=msg_id,
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        src="node-head-01",
        dst="node-tail-02",
        payload=payload,
    )


def test_split_into_chunks_and_reassembly() -> None:
    """Frames split into chunks must reassemble identically in order."""
    frame = b"SHRD" + bytes(range(256)) * 5000  # ~1.28 MB
    chunks = split_into_chunks(frame, chunk_size=MAX_DATA_CHUNK_BYTES)
    assert len(chunks) == 3
    assert all(len(c) <= MAX_DATA_CHUNK_BYTES for c in chunks)
    assert b"".join(chunks) == frame


def test_split_into_chunks_rejects_oversized_or_too_many_chunks() -> None:
    """Empty frames, > MAX_WIRE_FRAME_BYTES, or > 256 chunks must raise ValueError."""
    with pytest.raises(ValueError):
        split_into_chunks(b"")

    with pytest.raises(ValueError):
        split_into_chunks(b"x" * 1024, chunk_size=2)  # 512 chunks > MAX_CHUNKS (256)

    oversized = b"\x00" * (MAX_WIRE_FRAME_BYTES + 1)
    with pytest.raises(ValueError):
        split_into_chunks(oversized)


def test_encode_and_decode_reject_malformed_and_out_of_bounds_envelopes() -> None:
    """Malformed headers, bad bounds, and non-empty control payloads must fail validation."""
    msg_id = uuid.uuid4().bytes

    # Empty or oversized node IDs
    with pytest.raises(ValueError):
        encode_envelope(RelayPacketKind.ACK, msg_id, "", "dst-node")
    with pytest.raises(ValueError):
        encode_envelope(
            RelayPacketKind.ACK, msg_id, "a" * (MAX_NODE_ID_BYTES + 1), "dst-node"
        )

    # Non-empty ACK / NO_ROUTE / PING / PONG payload
    for empty_kind in (
        RelayPacketKind.ACK,
        RelayPacketKind.NO_ROUTE,
        RelayPacketKind.PING,
        RelayPacketKind.PONG,
    ):
        with pytest.raises(ValueError):
            encode_envelope(empty_kind, msg_id, "src-node", "dst-node", payload=b"bad")

    # Oversized DATA chunk & NACK reason
    with pytest.raises(ValueError):
        encode_envelope(
            RelayPacketKind.DATA,
            msg_id,
            "src-node",
            "dst-node",
            payload=b"x" * (MAX_DATA_CHUNK_BYTES + 1),
        )
    with pytest.raises(ValueError):
        encode_envelope(
            RelayPacketKind.NACK,
            msg_id,
            "src-node",
            "dst-node",
            payload=b"x" * (MAX_NACK_REASON_BYTES + 1),
        )

    # Chunk index / count bounds
    with pytest.raises(ValueError):
        encode_envelope(
            RelayPacketKind.DATA,
            msg_id,
            "src-node",
            "dst-node",
            payload=b"x",
            chunk_index=1,
            chunk_count=1,
        )
    with pytest.raises(ValueError):
        encode_envelope(
            RelayPacketKind.DATA,
            msg_id,
            "src-node",
            "dst-node",
            payload=b"x",
            chunk_index=0,
            chunk_count=MAX_CHUNKS + 1,
        )

    # Corrupted wire headers
    valid = encode_envelope(RelayPacketKind.PING, msg_id, "src-node", "dst-node")
    with pytest.raises(ValueError):
        decode_envelope(valid[:10])
    with pytest.raises(ValueError):
        decode_envelope(b"XXXX" + valid[4:])
    with pytest.raises(ValueError):
        bad_ver = bytearray(valid)
        bad_ver[4] = 99
        decode_envelope(bytes(bad_ver))
    with pytest.raises(ValueError):
        bad_kind = bytearray(valid)
        bad_kind[5] = 99
        decode_envelope(bytes(bad_kind))
    with pytest.raises(ValueError):
        bad_len = RELAY_HEADER_STRUCT.pack(
            RELAY_MAGIC, RELAY_VERSION, int(RelayPacketKind.PING), msg_id, 0, 1, 10, 10
        ) + b"short"
        decode_envelope(bad_len)

    # Invalid UTF-8 in NACK reason
    bad_nack = (
        RELAY_HEADER_STRUCT.pack(
            RELAY_MAGIC, RELAY_VERSION, int(RelayPacketKind.NACK), msg_id, 0, 1, 1, 1
        )
        + b"a"
        + b"b"
        + b"\xff\xfe"
    )
    with pytest.raises(ValueError):
        decode_envelope(bad_nack)


def test_truncate_utf8_bytes_preserves_multibyte_boundaries() -> None:
    """Truncating multi-byte UTF-8 text must never produce invalid UTF-8 sequences."""
    text = "Error: " + ("€" * 500)
    clipped = truncate_utf8_bytes(text, max_bytes=MAX_NACK_REASON_BYTES)
    assert len(clipped) <= MAX_NACK_REASON_BYTES
    assert clipped.decode("utf-8").startswith("Error: ")
