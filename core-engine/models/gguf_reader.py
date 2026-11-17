"""Binary GGUF Header Metadata Reader (`total_model_layers` extraction).

Parses the GGUF v2/v3 binary header key-value section to extract `<arch>.block_count`
(i.e. `total_model_layers`) without loading multi-gigabyte tensor weights into RAM.
"""

import logging
from pathlib import Path
import struct

logger = logging.getLogger("shardai.models.gguf_reader")

_GGUF_MAGIC: bytes = b"GGUF"

# GGUF metadata value type IDs (from ggml GGUF spec)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_SCALAR_SIZES: dict[int, int] = {
    _GGUF_TYPE_UINT8: 1,
    _GGUF_TYPE_INT8: 1,
    _GGUF_TYPE_UINT16: 2,
    _GGUF_TYPE_INT16: 2,
    _GGUF_TYPE_UINT32: 4,
    _GGUF_TYPE_INT32: 4,
    _GGUF_TYPE_FLOAT32: 4,
    _GGUF_TYPE_BOOL: 1,
    _GGUF_TYPE_UINT64: 8,
    _GGUF_TYPE_INT64: 8,
    _GGUF_TYPE_FLOAT64: 8,
}


def _read_gguf_string(f: object) -> str:
    """Read a length-prefixed UTF-8 string from a binary GGUF file stream."""
    len_bytes: bytes = f.read(8)  # type: ignore[attr-defined]
    if len(len_bytes) < 8:
        raise EOFError("Unexpected EOF reading GGUF string length")
    (str_len,) = struct.unpack("<Q", len_bytes)
    if str_len > 1_048_576:
        raise ValueError(f"Suspicious GGUF string length: {str_len}")
    raw: bytes = f.read(str_len)  # type: ignore[attr-defined]
    if len(raw) < str_len:
        raise EOFError("Unexpected EOF reading GGUF string payload")
    return raw.decode("utf-8", errors="replace")


def _skip_gguf_value(f: object, value_type: int) -> None:
    """Advance file cursor past an arbitrary GGUF KV metadata value."""
    if value_type in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[value_type], 1)  # type: ignore[attr-defined]
        return
    if value_type == _GGUF_TYPE_STRING:
        _read_gguf_string(f)
        return
    if value_type == _GGUF_TYPE_ARRAY:
        hdr: bytes = f.read(12)  # type: ignore[attr-defined]
        if len(hdr) < 12:
            raise EOFError("Unexpected EOF reading GGUF array header")
        elem_type, count = struct.unpack("<IQ", hdr)
        if elem_type in _SCALAR_SIZES:
            f.seek(_SCALAR_SIZES[elem_type] * count, 1)  # type: ignore[attr-defined]
            return
        for _ in range(min(count, 200_000)):
            _skip_gguf_value(f, elem_type)
        return
    raise ValueError(f"Unknown GGUF value type: {value_type}")


def read_gguf_total_layers(gguf_path: Path) -> int | None:
    """Read `<architecture>.block_count` (`total_model_layers`) from a GGUF file header.

    Returns `None` if the file is a lightweight dev stub or lacks valid GGUF KV metadata.
    """
    if not gguf_path.is_file() or gguf_path.stat().st_size < 24:
        return None

    try:
        with gguf_path.open("rb") as f:
            header = f.read(24)
            if len(header) < 24 or header[:4] != _GGUF_MAGIC:
                return None

            version, _tensor_count, kv_count = struct.unpack("<IQQ", header[4:24])
            if version not in (2, 3) or kv_count == 0 or kv_count > 10_000:
                return None

            for _ in range(kv_count):
                key: str = _read_gguf_string(f)
                type_bytes: bytes = f.read(4)
                if len(type_bytes) < 4:
                    break
                (val_type,) = struct.unpack("<I", type_bytes)

                if key.endswith(".block_count") and val_type in (
                    _GGUF_TYPE_UINT32,
                    _GGUF_TYPE_INT32,
                    _GGUF_TYPE_UINT64,
                    _GGUF_TYPE_INT64,
                ):
                    fmt = "<I" if val_type == _GGUF_TYPE_UINT32 else (
                        "<i" if val_type == _GGUF_TYPE_INT32 else (
                            "<Q" if val_type == _GGUF_TYPE_UINT64 else "<q"
                        )
                    )
                    size = struct.calcsize(fmt)
                    raw_val = f.read(size)
                    if len(raw_val) == size:
                        (block_count,) = struct.unpack(fmt, raw_val)
                        if 1 <= block_count <= 512:
                            logger.info(
                                "Extracted total_model_layers=%d from GGUF metadata (%s=%d)",
                                block_count,
                                key,
                                block_count,
                            )
                            return int(block_count)

                _skip_gguf_value(f, val_type)
    except Exception as exc:
        logger.debug("Could not parse GGUF KV metadata from %s: %s", gguf_path, exc)

    return None
