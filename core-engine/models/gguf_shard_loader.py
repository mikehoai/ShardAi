"""Selective Memory-Mapped GGUF Quantized PyTorch Shard Loader.

Enables distributed multi-node pipeline parallelism for `.gguf` models without
loading unowned layers into RAM or pre-dequantizing full weight matrices:
- Bounds-checked GGUF v2/v3 binary header & tensor table parser (`zero pickle`).
- Vectorized GGML dequantization for F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
  Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, and Q6_K.
- Memory-safe quantized `nn.Module` layers (`GgufQuantizedLinear` with optional
  row-chunked projection for large vocabularies and `GgufQuantizedEmbedding`)
  that store raw quantized `uint8` bytes and dequantize on the fly inside
  `torch.inference_mode()`.
- Pure-PyTorch decoder layer reconstruction (`GgufModelConfig`, `GgufRMSNorm`,
  `GgufRotaryEmbedding`, `GgufAttention`, `GgufMLP`, `GgufDecoderLayer`, and
  `GgufSimpleKVCache`) with zero external model-framework dependencies, loading
  ONLY each peer's assigned `[shard_layer_start..shard_layer_end]` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO

import numpy as np
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]

from models.ggml_native import (
    NATIVE_QUANTIZED_GGML_TYPES,
    GgmlNativeEngine,
    GgmlWeightHandle,
    is_ggml_native_enabled,
)
from models.gguf_reader import (
    _GGUF_MAGIC,
    _GGUF_TYPE_ARRAY,
    _GGUF_TYPE_BOOL,
    _GGUF_TYPE_FLOAT32,
    _GGUF_TYPE_FLOAT64,
    _GGUF_TYPE_INT8,
    _GGUF_TYPE_INT16,
    _GGUF_TYPE_INT32,
    _GGUF_TYPE_INT64,
    _GGUF_TYPE_STRING,
    _GGUF_TYPE_UINT8,
    _GGUF_TYPE_UINT16,
    _GGUF_TYPE_UINT32,
    _GGUF_TYPE_UINT64,
    _MAX_GGUF_ARRAY_ELEMENTS,
    _MAX_GGUF_STRING_BYTES,
    _SCALAR_SIZES,
    _read_gguf_string,
    _skip_gguf_value,
)

logger = logging.getLogger("shardai.models.gguf_shard_loader")

SUPPORTED_GGUF_ARCHITECTURES: frozenset[str] = frozenset(
    {"llama", "mistral", "qwen2", "qwen3", "gemma2", "gemma3", "phi3"}
)

_MAX_GGUF_KV_COUNT = 100_000
_MAX_GGUF_TENSOR_COUNT = 100_000
_DEFAULT_ALIGNMENT = 32
_DEFAULT_LM_HEAD_CHUNK_ROWS = 16384

# GGML tensor quantization type IDs (exact `enum ggml_type` from ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30

_DENSE_GGML_TYPES: frozenset[int] = frozenset(
    {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}
)

# ggml_type -> (block_elements, block_bytes)
GGML_TYPE_INFO: dict[int, tuple[int, int]] = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18),
    GGML_TYPE_Q4_1: (32, 20),
    GGML_TYPE_Q5_0: (32, 22),
    GGML_TYPE_Q5_1: (32, 24),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_Q3_K: (256, 110),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_Q5_K: (256, 176),
    GGML_TYPE_Q6_K: (256, 210),
    GGML_TYPE_IQ2_XXS: (256, 66),
    GGML_TYPE_IQ2_XS: (256, 74),
    GGML_TYPE_IQ3_XXS: (256, 98),
    GGML_TYPE_IQ1_S: (256, 50),
    GGML_TYPE_IQ4_NL: (32, 18),
    GGML_TYPE_IQ3_S: (256, 110),
    GGML_TYPE_IQ2_S: (256, 82),
    GGML_TYPE_IQ4_XS: (256, 136),
    GGML_TYPE_IQ1_M: (256, 56),
    GGML_TYPE_BF16: (1, 2),
}

_SCALAR_STRUCT_FMTS: dict[int, str] = {
    _GGUF_TYPE_UINT8: "<B",
    _GGUF_TYPE_INT8: "<b",
    _GGUF_TYPE_UINT16: "<H",
    _GGUF_TYPE_INT16: "<h",
    _GGUF_TYPE_UINT32: "<I",
    _GGUF_TYPE_INT32: "<i",
    _GGUF_TYPE_FLOAT32: "<f",
    _GGUF_TYPE_BOOL: "<?",
    _GGUF_TYPE_UINT64: "<Q",
    _GGUF_TYPE_INT64: "<q",
    _GGUF_TYPE_FLOAT64: "<d",
}

_SKIPPED_LARGE_ARRAY_KEYS: frozenset[str] = frozenset(
    {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    }
)


@dataclass(frozen=True)
class GgufTensorInfo:
    """Metadata and byte location of one tensor inside a GGUF file."""

    name: str
    dims_gguf: tuple[int, ...]
    shape_pt: tuple[int, ...]
    ggml_type: int
    rel_offset: int
    nbytes: int


@dataclass
class GgufFileIndex:
    """Parsed GGUF v2/v3 header metadata and tensor table without loading tensor payloads."""

    path: Path
    version: int
    alignment: int
    tensor_count: int
    kv_count: int
    metadata: dict[str, Any]
    tensors: dict[str, GgufTensorInfo]
    data_offset: int
    file_size: int
    tokens_value_offset: int | None = None
    vocab_size_from_tokens: int | None = None


@dataclass
class LoadedGgufShard:
    """Sliced PyTorch shard modules and metadata loaded selectively from a GGUF file."""

    hf_config: Any
    container: nn.Module
    layers: list[Any]
    rotary_emb: Any | None
    embed_tokens: Any | None
    norm: Any | None
    lm_head: Any | None
    tokenizer: Any | None
    total_layers: int
    hidden_dim: int
    shard_layer_start: int
    shard_layer_end: int
    compute_dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    materialized_tensors: int = 0
    ggml_engine: Any | None = None
    native_ggml_tensors: int = 0

    def close(self) -> None:
        """Release native GGML engine handles and tokenizer resources."""
        if self.ggml_engine is not None:
            close_engine = getattr(self.ggml_engine, "close", None)
            if callable(close_engine):
                close_engine()
            self.ggml_engine = None
        if self.tokenizer is not None:
            close_tok = getattr(self.tokenizer, "close", None)
            if callable(close_tok):
                close_tok()
            self.tokenizer = None


_AUTO_ACCELERATOR_MIN_BYTES: int = 8 * 1024 * 1024


def select_gguf_torch_device(
    preferred: str | None = None,
    kept_bytes: int = 0,
) -> torch.device:
    """Choose the best available PyTorch compute device (`cuda`, `mps`, or `cpu`)."""
    choice = (
        (preferred if preferred is not None else os.getenv("SHARDAI_TORCH_DEVICE", "auto"))
        .strip()
        .lower()
    )
    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    if choice == "cuda":
        selected = torch.device("cuda" if cuda_ok else "cpu")
    elif choice == "mps":
        selected = torch.device("mps" if mps_ok else "cpu")
    elif choice == "cpu":
        selected = torch.device("cpu")
    else:
        if kept_bytes < _AUTO_ACCELERATOR_MIN_BYTES:
            selected = torch.device("cpu")
        elif cuda_ok:
            selected = torch.device("cuda")
        elif mps_ok:
            selected = torch.device("mps")
        else:
            selected = torch.device("cpu")

    if selected.type == "cpu":
        cpu_threads = os.cpu_count() or 1
        if cpu_threads > 1 and torch.get_num_threads() == 1:
            torch.set_num_threads(min(8, cpu_threads))
    return selected


def _compute_ggml_nbytes(dims_gguf: tuple[int, ...], ggml_type: int) -> int:
    """Compute exact packed byte length for a GGUF tensor shape and GGML type."""
    if ggml_type not in GGML_TYPE_INFO:
        raise ValueError(f"Unsupported GGML tensor type ID: {ggml_type}")
    if not dims_gguf:
        raise ValueError("Tensor must have at least 1 dimension")
    block_elems, block_bytes = GGML_TYPE_INFO[ggml_type]
    inner_dim = dims_gguf[0]
    if inner_dim <= 0 or inner_dim % block_elems != 0:
        raise ValueError(
            f"Inner GGUF dimension {inner_dim} is not a positive multiple of block_elems={block_elems} "
            f"for ggml_type={ggml_type}"
        )
    outer_rows = 1
    for dim in dims_gguf[1:]:
        if dim <= 0:
            raise ValueError(f"Invalid non-positive GGUF dimension in {dims_gguf}")
        outer_rows *= dim
    return (inner_dim // block_elems) * block_bytes * outer_rows


def _read_scalar_value(f: BinaryIO, value_type: int) -> Any:
    """Read one GGUF scalar metadata value."""
    fmt = _SCALAR_STRUCT_FMTS[value_type]
    size = _SCALAR_SIZES[value_type]
    raw = f.read(size)
    if len(raw) < size:
        raise EOFError(f"Unexpected EOF reading GGUF scalar type {value_type}")
    (val,) = struct.unpack(fmt, raw)
    return val


def _should_store_metadata_key(key: str) -> bool:
    """Return True if `key` is needed for config or tokenizer construction."""
    if key in {
        "general.architecture",
        "general.alignment",
        "general.name",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.chat_template",
    }:
        return True
    suffixes = (
        ".block_count",
        ".embedding_length",
        ".feed_forward_length",
        ".context_length",
        ".attention.head_count",
        ".attention.head_count_kv",
        ".attention.key_length",
        ".attention.value_length",
        ".attention.layer_norm_rms_epsilon",
        ".attention.sliding_window",
        ".attention.query_pre_attn_scalar",
        ".rope.freq_base",
        ".rope.dimension_count",
        ".final_logit_softcapping",
        ".attn_logit_softcapping",
    )
    return any(key.endswith(suffix) for suffix in suffixes)


def _skip_large_string_array_fast(f: BinaryIO, count: int) -> None:
    """Quickly skip an array of GGUF strings with bounds checks."""
    if count > _MAX_GGUF_ARRAY_ELEMENTS:
        raise ValueError(
            f"GGUF array element count ({count}) exceeds safety cap ({_MAX_GGUF_ARRAY_ELEMENTS})"
        )
    for _ in range(count):
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise EOFError("Unexpected EOF skipping GGUF string array element")
        (slen,) = struct.unpack("<Q", raw_len)
        if slen > _MAX_GGUF_STRING_BYTES:
            raise ValueError(f"Suspicious GGUF string length: {slen}")
        if slen > 0:
            f.seek(slen, 1)


def read_gguf_header_and_tensors(gguf_path: Path) -> GgufFileIndex:
    """Parse GGUF v2/v3 header metadata and tensor index with strict bounds checks."""
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
    file_size = gguf_path.stat().st_size
    if file_size < 24:
        raise ValueError(f"GGUF file too small ({file_size} bytes): {gguf_path}")

    with gguf_path.open("rb") as f:
        header = f.read(24)
        if len(header) < 24 or header[:4] != _GGUF_MAGIC:
            raise ValueError("Invalid GGUF magic header")
        version, tensor_count, kv_count = struct.unpack("<IQQ", header[4:24])
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        if tensor_count == 0 or tensor_count > _MAX_GGUF_TENSOR_COUNT:
            raise ValueError(f"Invalid GGUF tensor_count: {tensor_count}")
        if kv_count > _MAX_GGUF_KV_COUNT:
            raise ValueError(f"Invalid GGUF kv_count: {kv_count}")

        metadata: dict[str, Any] = {}
        tokens_value_offset: int | None = None
        vocab_size_from_tokens: int | None = None

        for _ in range(kv_count):
            key = _read_gguf_string(f)
            type_raw = f.read(4)
            if len(type_raw) < 4:
                raise EOFError(f"Unexpected EOF reading value type for key '{key}'")
            (val_type,) = struct.unpack("<I", type_raw)

            if key in _SKIPPED_LARGE_ARRAY_KEYS and val_type == _GGUF_TYPE_ARRAY:
                if key == "tokenizer.ggml.tokens":
                    tokens_value_offset = f.tell()
                arr_hdr = f.read(12)
                if len(arr_hdr) < 12:
                    raise EOFError(f"Unexpected EOF reading array header for '{key}'")
                elem_type, count = struct.unpack("<IQ", arr_hdr)
                if count > _MAX_GGUF_ARRAY_ELEMENTS:
                    raise ValueError(
                        f"GGUF array element count ({count}) exceeds safety cap ({_MAX_GGUF_ARRAY_ELEMENTS})"
                    )
                if key == "tokenizer.ggml.tokens":
                    vocab_size_from_tokens = int(count)
                if elem_type in _SCALAR_SIZES:
                    f.seek(_SCALAR_SIZES[elem_type] * count, 1)
                elif elem_type == _GGUF_TYPE_STRING:
                    _skip_large_string_array_fast(f, int(count))
                else:
                    for _ in range(count):
                        _skip_gguf_value(f, elem_type, depth=1)
                continue

            if _should_store_metadata_key(key):
                if val_type in _SCALAR_STRUCT_FMTS:
                    metadata[key] = _read_scalar_value(f, val_type)
                    continue
                if val_type == _GGUF_TYPE_STRING:
                    metadata[key] = _read_gguf_string(f)
                    continue

            _skip_gguf_value(f, val_type)

        raw_alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
        alignment = int(raw_alignment) if isinstance(raw_alignment, int) else _DEFAULT_ALIGNMENT
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError(f"Invalid GGUF alignment: {alignment}")

        tensors: dict[str, GgufTensorInfo] = {}
        for _ in range(tensor_count):
            t_name = _read_gguf_string(f)
            nd_raw = f.read(4)
            if len(nd_raw) < 4:
                raise EOFError(f"Unexpected EOF reading ndims for tensor '{t_name}'")
            (n_dims,) = struct.unpack("<I", nd_raw)
            if n_dims == 0 or n_dims > 4:
                raise ValueError(f"Invalid GGUF tensor ndims={n_dims} for '{t_name}'")
            dims_raw = f.read(8 * n_dims)
            if len(dims_raw) < 8 * n_dims:
                raise EOFError(f"Unexpected EOF reading dims for tensor '{t_name}'")
            dims_gguf = tuple(int(d) for d in struct.unpack(f"<{n_dims}Q", dims_raw))
            meta_raw = f.read(12)
            if len(meta_raw) < 12:
                raise EOFError(f"Unexpected EOF reading type/offset for tensor '{t_name}'")
            ggml_type, rel_offset = struct.unpack("<IQ", meta_raw)
            nbytes = _compute_ggml_nbytes(dims_gguf, int(ggml_type))
            tensors[t_name] = GgufTensorInfo(
                name=t_name,
                dims_gguf=dims_gguf,
                shape_pt=tuple(reversed(dims_gguf)),
                ggml_type=int(ggml_type),
                rel_offset=int(rel_offset),
                nbytes=nbytes,
            )

        pos = f.tell()
        data_offset = ((pos + alignment - 1) // alignment) * alignment
        if data_offset > file_size:
            raise ValueError(
                f"GGUF data_offset ({data_offset}) exceeds file_size ({file_size})"
            )

        for t_info in tensors.values():
            end_pos = data_offset + t_info.rel_offset + t_info.nbytes
            if end_pos > file_size:
                raise ValueError(
                    f"Truncated GGUF tensor '{t_info.name}': end offset {end_pos} > file_size {file_size}"
                )

    return GgufFileIndex(
        path=gguf_path,
        version=int(version),
        alignment=alignment,
        tensor_count=int(tensor_count),
        kv_count=int(kv_count),
        metadata=metadata,
        tensors=tensors,
        data_offset=data_offset,
        file_size=file_size,
        tokens_value_offset=tokens_value_offset,
        vocab_size_from_tokens=vocab_size_from_tokens,
    )


def can_load_gguf_pytorch_shard(gguf_path: Path) -> bool:
    """Return True iff `gguf_path` is a valid GGUF v2/v3 with a supported architecture."""
    try:
        index = read_gguf_header_and_tensors(gguf_path)
    except Exception:
        return False
    arch = str(index.metadata.get("general.architecture") or "").strip().lower()
    return index.tensor_count > 0 and arch in SUPPORTED_GGUF_ARCHITECTURES


def _read_gguf_tokens_list(index: GgufFileIndex) -> list[str]:
    """Read `tokenizer.ggml.tokens` string list on demand when fallback tokenizer is used."""
    if index.tokens_value_offset is None:
        return []
    with index.path.open("rb") as f:
        f.seek(index.tokens_value_offset)
        arr_hdr = f.read(12)
        if len(arr_hdr) < 12:
            return []
        elem_type, count = struct.unpack("<IQ", arr_hdr)
        if elem_type != _GGUF_TYPE_STRING or count > _MAX_GGUF_ARRAY_ELEMENTS:
            return []
        return [_read_gguf_string(f) for _ in range(int(count))]


# ---------------------------------------------------------------------------
# Vectorized GGML Dequantization
# ---------------------------------------------------------------------------


def _half_scalar(blocks: torch.Tensor, start: int, dtype: torch.dtype) -> torch.Tensor:
    """Read little-endian float16 scalar per block as `(nb, 1)` in `dtype`."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).to(dtype)


def _unpack_k4_scales_mins(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack 12 bytes of 6-bit scales/mins shared by Q4_K and Q5_K (`get_scale_min_k4`)."""
    s0 = scales[:, 0:4] & 0x3F
    s1 = (scales[:, 8:12] & 0x0F) | ((scales[:, 0:4] >> 6) << 4)
    m0 = scales[:, 4:8] & 0x3F
    m1 = (scales[:, 8:12] >> 4) | ((scales[:, 4:8] >> 6) << 4)
    scale = torch.cat([s0, s1], dim=1).to(torch.float32)
    minimum = torch.cat([m0, m1], dim=1).to(torch.float32)
    return scale, minimum


def _extract_shifted_fields(
    data: torch.Tensor, shifts: tuple[int, ...], width: int
) -> torch.Tensor:
    """Extract bit fields at `shifts` across chunks of `width` bytes -> `(nb, -1, width)`."""
    shift = torch.tensor(shifts, device=data.device, dtype=torch.uint8).reshape(1, 1, -1, 1)
    return (data.reshape(data.shape[0], -1, 1, width) >> shift).reshape(data.shape[0], -1, width)


def _dequantize_blocks_u8(
    raw_u8: torch.Tensor,
    ggml_type: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized dequantization of a flat 1D `torch.uint8` tensor into 1D `dtype` values."""
    if ggml_type == GGML_TYPE_F32:
        return raw_u8.contiguous().view(torch.float32).to(dtype)
    if ggml_type == GGML_TYPE_F16:
        return raw_u8.contiguous().view(torch.float16).to(dtype)
    if ggml_type == GGML_TYPE_BF16:
        return raw_u8.contiguous().view(torch.bfloat16).to(dtype)

    if ggml_type not in GGML_TYPE_INFO:
        raise ValueError(f"Unsupported GGML quantization type ID: {ggml_type}")

    block_elems, block_bytes = GGML_TYPE_INFO[ggml_type]
    if raw_u8.numel() % block_bytes != 0:
        raise ValueError(
            f"Raw byte count {raw_u8.numel()} is not divisible by block_bytes={block_bytes} "
            f"for ggml_type={ggml_type}"
        )
    blocks = raw_u8.reshape(-1, block_bytes)
    nb = int(blocks.shape[0])

    if ggml_type == GGML_TYPE_Q4_0:
        d = _half_scalar(blocks, 0, dtype)
        qs = blocks[:, 2:18]
        nibbles = torch.cat([qs & 0x0F, qs >> 4], dim=1)
        out = d * (nibbles.to(torch.int8) - 8).to(dtype)
    elif ggml_type == GGML_TYPE_Q4_1:
        d = _half_scalar(blocks, 0, dtype)
        m = _half_scalar(blocks, 2, dtype)
        qs = blocks[:, 4:20]
        nibbles = torch.cat([qs & 0x0F, qs >> 4], dim=1)
        out = d * nibbles.to(dtype) + m
    elif ggml_type == GGML_TYPE_Q5_0:
        d = _half_scalar(blocks, 0, dtype)
        qh = blocks[:, 2:6].contiguous().view(torch.int32)
        qs = blocks[:, 6:22]
        bit_pos = torch.arange(32, device=blocks.device, dtype=torch.int32).reshape(1, 32)
        high_bits = (qh >> bit_pos) & 1
        nibbles = torch.cat([qs & 0x0F, qs >> 4], dim=1).to(torch.int32)
        quants = (nibbles | (high_bits << 4)) - 16
        out = d * quants.to(dtype)
    elif ggml_type == GGML_TYPE_Q5_1:
        d = _half_scalar(blocks, 0, dtype)
        m = _half_scalar(blocks, 2, dtype)
        qh = blocks[:, 4:8].contiguous().view(torch.int32)
        qs = blocks[:, 8:24]
        bit_pos = torch.arange(32, device=blocks.device, dtype=torch.int32).reshape(1, 32)
        high_bits = (qh >> bit_pos) & 1
        nibbles = torch.cat([qs & 0x0F, qs >> 4], dim=1).to(torch.int32)
        quants = nibbles | (high_bits << 4)
        out = d * quants.to(dtype) + m
    elif ggml_type == GGML_TYPE_Q8_0:
        d = _half_scalar(blocks, 0, dtype)
        qs = blocks[:, 2:34].view(torch.int8).to(dtype)
        out = d * qs
    elif ggml_type == GGML_TYPE_Q2_K:
        scales = blocks[:, 0:16]
        qs = blocks[:, 16:80]
        d = _half_scalar(blocks, 80, torch.float32)
        dmin = _half_scalar(blocks, 82, torch.float32)
        dl = (d * (scales & 0x0F).to(torch.float32)).reshape(nb, 16, 1)
        ml = (dmin * (scales >> 4).to(torch.float32)).reshape(nb, 16, 1)
        q = (_extract_shifted_fields(qs, (0, 2, 4, 6), 32) & 0x03).reshape(nb, 16, 16)
        out = (dl * q.to(torch.float32) - ml).to(dtype).reshape(nb, block_elems)
    elif ggml_type == GGML_TYPE_Q3_K:
        hmask = blocks[:, 0:32]
        qs = blocks[:, 32:96]
        scales = blocks[:, 96:108]
        d = _half_scalar(blocks, 108, torch.float32)
        low = _extract_shifted_fields(scales[:, :8], (0, 4), 8).reshape(nb, 16)
        high = _extract_shifted_fields(scales[:, 8:12], (0, 2, 4, 6), 4).reshape(nb, 16)
        scale = ((low & 0x0F) | ((high & 0x03) << 4)).to(torch.int8).to(torch.float32) - 32.0
        ql = (_extract_shifted_fields(qs, (0, 2, 4, 6), 32) & 0x03).reshape(nb, 16, 16)
        qh = (_extract_shifted_fields(hmask, tuple(range(8)), 32).reshape(nb, 16, 16) & 1) ^ 1
        q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
        out = ((d * scale)[..., None] * q.to(torch.float32)).to(dtype).reshape(nb, block_elems)
    elif ggml_type == GGML_TYPE_Q4_K:
        d = _half_scalar(blocks, 0, torch.float32)
        dmin = _half_scalar(blocks, 2, torch.float32)
        scale, minimum = _unpack_k4_scales_mins(blocks[:, 4:16])
        q = _extract_shifted_fields(blocks[:, 16:144], (0, 4), 32) & 0x0F
        out = (
            (d * scale)[..., None] * q.to(torch.float32) - (dmin * minimum)[..., None]
        ).to(dtype).reshape(nb, block_elems)
    elif ggml_type == GGML_TYPE_Q5_K:
        d = _half_scalar(blocks, 0, torch.float32)
        dmin = _half_scalar(blocks, 2, torch.float32)
        scale, minimum = _unpack_k4_scales_mins(blocks[:, 4:16])
        high = _extract_shifted_fields(blocks[:, 16:48], tuple(range(8)), 32) & 1
        low = _extract_shifted_fields(blocks[:, 48:176], (0, 4), 32) & 0x0F
        q = low | (high << 4)
        out = (
            (d * scale)[..., None] * q.to(torch.float32) - (dmin * minimum)[..., None]
        ).to(dtype).reshape(nb, block_elems)
    elif ggml_type == GGML_TYPE_Q6_K:
        ql = blocks[:, 0:128]
        qh = blocks[:, 128:192]
        scales = blocks[:, 192:208].view(torch.int8).to(torch.float32)
        d = _half_scalar(blocks, 208, torch.float32)
        scale = (d * scales).reshape(nb, 16, 1)
        low = (_extract_shifted_fields(ql, (0, 4), 64) & 0x0F).reshape(nb, 8, 32)
        high = (_extract_shifted_fields(qh, (0, 2, 4, 6), 32) & 0x03).reshape(nb, 8, 32)
        quants = (low | (high << 4)).to(torch.int8) - 32
        out = (scale * quants.reshape(nb, 16, 16).to(torch.float32)).to(dtype).reshape(
            nb, block_elems
        )
    else:
        raise ValueError(
            f"Unsupported GGML quantization type ID in PyTorch fallback: {ggml_type} "
            "(enable native libggml backend for IQ quantization types)"
        )

    return out.reshape(-1)


def dequantize_ggml_bytes(
    raw: bytes | memoryview,
    shape_pt: tuple[int, ...],
    ggml_type: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize raw GGML tensor bytes into a PyTorch tensor of `shape_pt` and `dtype`."""
    expected_elems = math.prod(shape_pt)
    raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
    raw_u8 = torch.from_numpy(arr)
    flat = _dequantize_blocks_u8(raw_u8, ggml_type=ggml_type, dtype=dtype)
    if int(flat.numel()) != expected_elems:
        raise ValueError(
            f"Dequantized element count {flat.numel()} != expected {expected_elems} "
            f"for shape {shape_pt} (ggml_type={ggml_type})"
        )
    return flat.reshape(shape_pt)


def _reverse_permute_rope_rows(tensor: torch.Tensor, n_head: int) -> torch.Tensor:
    """Undo llama.cpp's interleaved RoPE head permutation along axis 0 (`llama`/`mistral`)."""
    out_rows = int(tensor.shape[0])
    if n_head <= 0 or out_rows % (2 * n_head) != 0:
        return tensor
    head_half = out_rows // n_head // 2
    rest = tensor.shape[1:]
    return tensor.reshape(n_head, head_half, 2, *rest).swapaxes(2, 1).reshape(tensor.shape)


def _reverse_permute_rope_last_dim(tensor: torch.Tensor, n_head: int) -> torch.Tensor:
    """Undo llama.cpp's interleaved RoPE head permutation along the last output dimension."""
    out_dim = int(tensor.shape[-1])
    if n_head <= 0 or out_dim % (2 * n_head) != 0:
        return tensor
    head_half = out_dim // n_head // 2
    lead = tensor.shape[:-1]
    return (
        tensor.reshape(*lead, n_head, head_half, 2)
        .swapaxes(-2, -1)
        .reshape(tensor.shape)
    )


# ---------------------------------------------------------------------------
# Memory-Safe Quantized PyTorch Layers
# ---------------------------------------------------------------------------


class GgufQuantizedLinear(nn.Module):
    """Linear layer storing packed quantized `uint8` weights and dequantizing per forward."""

    def __init__(
        self,
        raw_weight: torch.Tensor,
        out_features: int,
        in_features: int,
        ggml_type: int,
        bias: torch.Tensor | None = None,
        unpermute_heads: int | None = None,
        chunk_out_rows: int | None = None,
    ) -> None:
        super().__init__()
        if raw_weight.dtype != torch.uint8 or raw_weight.ndim != 1:
            raise ValueError("GgufQuantizedLinear expects a 1D torch.uint8 buffer")
        block_elems, block_bytes = GGML_TYPE_INFO[ggml_type]
        if in_features % block_elems != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by block_elems={block_elems}"
            )
        self.out_features: int = out_features
        self.in_features: int = in_features
        self.ggml_type: int = ggml_type
        self.unpermute_heads: int | None = unpermute_heads
        self.chunk_out_rows: int | None = chunk_out_rows
        self._row_bytes: int = (in_features // block_elems) * block_bytes
        if int(raw_weight.numel()) != self.out_features * self._row_bytes:
            raise ValueError(
                f"raw_weight byte length {raw_weight.numel()} != expected "
                f"{self.out_features * self._row_bytes}"
            )

        self.register_buffer("raw_weight", raw_weight.detach().contiguous(), persistent=True)
        if bias is not None:
            self.register_parameter(
                "bias", nn.Parameter(bias.detach().contiguous(), requires_grad=False)
            )
        else:
            self.register_parameter("bias", None)
        self._cached_weight: torch.Tensor | None = None
        self._ggml_engine: GgmlNativeEngine | None = None
        self._ggml_handle: GgmlWeightHandle | None = None

    def attach_ggml_engine(self, engine: GgmlNativeEngine) -> bool:
        """Bind raw packed weights directly to a native `GgmlNativeEngine` without RAM expansion."""
        if (
            self._cached_weight is not None
            or self.raw_weight.numel() == 0
            or self.ggml_type not in NATIVE_QUANTIZED_GGML_TYPES
        ):
            return False
        self._ggml_handle = engine.register_weight(
            self.raw_weight,
            self.out_features,
            self.in_features,
            self.ggml_type,
        )
        self._ggml_engine = engine
        return True

    def materialize(self, dtype: torch.dtype, device: torch.device) -> int:
        """Pre-dequantize `raw_weight` onto `(device, dtype)` once and return net RAM byte delta."""
        if self._ggml_handle is not None:
            return 0
        if self._cached_weight is not None:
            if self._cached_weight.device == device and self._cached_weight.dtype == dtype:
                return 0
            old_bytes = int(self._cached_weight.numel() * self._cached_weight.element_size())
            self._cached_weight = self._cached_weight.to(device=device, dtype=dtype).contiguous()
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().to(device=device, dtype=dtype).contiguous(),
                    requires_grad=False,
                )
            new_bytes = int(self._cached_weight.numel() * self._cached_weight.element_size())
            return new_bytes - old_bytes

        old_raw_bytes = int(self.raw_weight.numel())
        with torch.inference_mode():
            w = _dequantize_blocks_u8(
                self.raw_weight, ggml_type=self.ggml_type, dtype=torch.float32
            ).reshape(self.out_features, self.in_features)
            if self.unpermute_heads is not None:
                w = _reverse_permute_rope_rows(w, self.unpermute_heads)
            cached = w.to(device=device, dtype=dtype).contiguous()
            del w
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().to(device=device, dtype=dtype).contiguous(),
                    requires_grad=False,
                )
        self._cached_weight = cached
        self.register_buffer("raw_weight", torch.empty(0, dtype=torch.uint8), persistent=True)
        new_bytes = int(cached.numel() * cached.element_size())
        return new_bytes - old_raw_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear projection via native libggml, materialized weights, or PyTorch dequant."""
        with torch.inference_mode():
            orig_device = x.device
            orig_dtype = x.dtype

            if (
                self._ggml_engine is not None
                and self._ggml_handle is not None
                and not self._ggml_engine.is_closed
            ):
                x_2d = (
                    x.reshape(-1, self.in_features)
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                out_np = self._ggml_engine.matmul(self._ggml_handle, x_2d.numpy())
                out = torch.from_numpy(out_np).reshape(*x.shape[:-1], self.out_features)
                if self.unpermute_heads is not None:
                    out = _reverse_permute_rope_last_dim(out, self.unpermute_heads)
                if self.bias is not None:
                    out = out + self.bias.to(device="cpu", dtype=torch.float32)
                if out.device != orig_device or out.dtype != orig_dtype:
                    out = out.to(device=orig_device, dtype=orig_dtype)
                return out

            if self._cached_weight is not None:
                w = self._cached_weight
                x_dev = x if x.device == w.device else x.to(device=w.device)
                # CPU BLAS SGEMM is dramatically faster than CPU BF16 GEMM on multi-token prefills
                if (
                    w.device.type == "cpu"
                    and w.dtype == torch.bfloat16
                    and x_dev.ndim >= 2
                    and x_dev.shape[-2] > 4
                ):
                    bias_f32 = (
                        self.bias.to(device=w.device, dtype=torch.float32)
                        if self.bias is not None
                        else None
                    )
                    out = F.linear(x_dev.to(torch.float32), w.to(torch.float32), bias_f32)
                else:
                    bias_w = (
                        self.bias.to(device=w.device, dtype=w.dtype)
                        if self.bias is not None
                        else None
                    )
                    x_cast = x_dev if x_dev.dtype == w.dtype else x_dev.to(dtype=w.dtype)
                    out = F.linear(x_cast, w, bias_w)
                if out.device != orig_device or out.dtype != orig_dtype:
                    out = out.to(device=orig_device, dtype=orig_dtype)
                return out

            work_dtype = torch.float32 if orig_device.type != "cpu" else orig_dtype
            x_cpu = x if orig_device.type == "cpu" else x.to(device="cpu", dtype=work_dtype)
            if x_cpu.dtype != work_dtype:
                x_cpu = x_cpu.to(dtype=work_dtype)
            bias_t = (
                self.bias.to(device="cpu", dtype=work_dtype)
                if self.bias is not None
                else None
            )
            rows_2d = self.raw_weight.view(self.out_features, self._row_bytes)

            if (
                self.chunk_out_rows is not None
                and self.chunk_out_rows > 0
                and self.out_features > self.chunk_out_rows
                and self.unpermute_heads is None
            ):
                out_shape = (*x_cpu.shape[:-1], self.out_features)
                output = torch.empty(out_shape, dtype=work_dtype, device="cpu")
                for r0 in range(0, self.out_features, self.chunk_out_rows):
                    r1 = min(self.out_features, r0 + self.chunk_out_rows)
                    chunk_u8 = rows_2d[r0:r1].reshape(-1)
                    w_chunk = _dequantize_blocks_u8(
                        chunk_u8, ggml_type=self.ggml_type, dtype=work_dtype
                    ).reshape(r1 - r0, self.in_features)
                    b_chunk = bias_t[r0:r1] if bias_t is not None else None
                    output[..., r0:r1] = F.linear(x_cpu, w_chunk, b_chunk)
                    del w_chunk
                if output.device != orig_device or output.dtype != orig_dtype:
                    output = output.to(device=orig_device, dtype=orig_dtype)
                return output

            w = _dequantize_blocks_u8(
                self.raw_weight, ggml_type=self.ggml_type, dtype=work_dtype
            ).reshape(self.out_features, self.in_features)
            if self.unpermute_heads is not None:
                w = _reverse_permute_rope_rows(w, self.unpermute_heads)
            out = F.linear(x_cpu, w, bias_t)
            del w
            if out.device != orig_device or out.dtype != orig_dtype:
                out = out.to(device=orig_device, dtype=orig_dtype)
            return out


class GgufQuantizedEmbedding(nn.Module):
    """Embedding layer storing packed `uint8` rows and dequantizing only gathered token IDs."""

    def __init__(
        self,
        raw_weight: torch.Tensor,
        vocab_size: int,
        embedding_dim: int,
        ggml_type: int,
        embed_scale: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if raw_weight.dtype != torch.uint8:
            raise ValueError("GgufQuantizedEmbedding expects torch.uint8 raw_weight")
        block_elems, block_bytes = GGML_TYPE_INFO[ggml_type]
        row_bytes = (embedding_dim // block_elems) * block_bytes
        if raw_weight.ndim == 1:
            raw_weight = raw_weight.view(vocab_size, row_bytes)
        if tuple(raw_weight.shape) != (vocab_size, row_bytes):
            raise ValueError(
                f"raw_weight shape {tuple(raw_weight.shape)} != expected {(vocab_size, row_bytes)}"
            )
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.ggml_type: int = ggml_type
        self.embed_scale: float = float(embed_scale)
        self.output_dtype: torch.dtype = output_dtype
        self._row_bytes: int = row_bytes
        self.target_device: torch.device = torch.device("cpu")
        self.register_buffer("raw_weight", raw_weight.detach().contiguous(), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather only `input_ids` rows and dequantize those rows on the fly."""
        with torch.inference_mode():
            ids_cpu = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
            selected_u8 = self.raw_weight.index_select(0, ids_cpu).reshape(-1)
            deq = _dequantize_blocks_u8(
                selected_u8, ggml_type=self.ggml_type, dtype=self.output_dtype
            ).reshape(*input_ids.shape, self.embedding_dim)
            if self.embed_scale != 1.0:
                deq = deq * self.embed_scale
            if self.target_device.type != "cpu":
                deq = deq.to(device=self.target_device)
            return deq


# ---------------------------------------------------------------------------
# Pure-PyTorch GGUF Model Config, KV-Cache & Decoder Modules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufModelConfig:
    """Pure-PyTorch model architecture configuration reconstructed from GGUF metadata."""

    arch: str
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    sliding_window: int | None
    final_logit_softcapping: float | None
    attn_logit_softcapping: float | None
    query_pre_attn_scalar: float
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None


_KV_INITIAL_CAP: int = 256
_KV_GROW_STEP: int = 256


class GgufSimpleKVCache:
    """Preallocated layer-indexed KV-cache supporting partial shards (`shard_layer_start > 0`)."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for `layer_idx` (or 0 if unpopulated)."""
        entry = self._cache.get(layer_idx)
        if entry is None:
            return 0
        return int(entry[2])

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key/value states for `layer_idx` into a growing buffer without O(N^2) copies."""
        bsz, n_kv_heads, seq_len, head_dim = key_states.shape
        entry = self._cache.get(layer_idx)
        if entry is None:
            cap = max(_KV_INITIAL_CAP, seq_len + _KV_GROW_STEP)
            k_buf = torch.empty(
                (bsz, n_kv_heads, cap, head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            v_buf = torch.empty(
                (bsz, n_kv_heads, cap, head_dim),
                dtype=value_states.dtype,
                device=value_states.device,
            )
            k_buf[:, :, :seq_len, :] = key_states
            v_buf[:, :, :seq_len, :] = value_states
            self._cache[layer_idx] = (k_buf, v_buf, seq_len)
            return k_buf[:, :, :seq_len, :], v_buf[:, :, :seq_len, :]

        k_buf, v_buf, prev_len = entry
        new_len = prev_len + seq_len
        cap = int(k_buf.shape[-2])
        if (
            new_len > cap
            or k_buf.device != key_states.device
            or k_buf.dtype != key_states.dtype
        ):
            new_cap = max(new_len + _KV_GROW_STEP, cap * 2)
            new_k = torch.empty(
                (bsz, n_kv_heads, new_cap, head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            new_v = torch.empty(
                (bsz, n_kv_heads, new_cap, head_dim),
                dtype=value_states.dtype,
                device=value_states.device,
            )
            if prev_len > 0:
                new_k[:, :, :prev_len, :] = k_buf[:, :, :prev_len, :].to(
                    device=key_states.device, dtype=key_states.dtype
                )
                new_v[:, :, :prev_len, :] = v_buf[:, :, :prev_len, :].to(
                    device=value_states.device, dtype=value_states.dtype
                )
            k_buf, v_buf = new_k, new_v

        k_buf[:, :, prev_len:new_len, :] = key_states
        v_buf[:, :, prev_len:new_len, :] = value_states
        self._cache[layer_idx] = (k_buf, v_buf, new_len)
        return k_buf[:, :, :new_len, :], v_buf[:, :, :new_len, :]

    def clear(self) -> None:
        """Release all cached key/value tensors."""
        self._cache.clear()


class GgufRMSNorm(nn.Module):
    """Pure-PyTorch RMSNorm reading GGUF effective norm weights directly."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = float(eps)
        self.weight: nn.Parameter = nn.Parameter(weight.detach().contiguous(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize along the last dimension in float32 and scale by `self.weight`."""
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.to(torch.float32)).to(x.dtype)


class GgufRotaryEmbedding(nn.Module):
    """Pure-PyTorch Rotary Position Embedding (RoPE) supporting full and partial dimensions."""

    def __init__(self, rotary_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.rotary_dim: int = int(rotary_dim)
        self.rope_theta: float = float(rope_theta)
        self.inv_freq: torch.Tensor = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / float(self.rotary_dim)
            )
        )

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute `(cos, sin)` tensors in `hidden_states.dtype` for `position_ids`."""
        inv_freq = self.inv_freq.to(device=hidden_states.device, dtype=torch.float32)
        freqs = position_ids.to(device=hidden_states.device, dtype=torch.float32)[..., None] * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=hidden_states.dtype)
        sin = emb.sin().to(dtype=hidden_states.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the negated first half."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE to the first `rotary_dim` channels of `q` and `k`."""
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    if rotary_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos_u) + (_rotate_half(q_rot) * sin_u)
        k_rot = (k_rot * cos_u) + (_rotate_half(k_rot) * sin_u)
        return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed


class GgufAttention(nn.Module):
    """Pure-PyTorch multi-head / grouped-query self-attention for GGUF decoder layers."""

    def __init__(
        self,
        *,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        attn_scale: float,
        attn_logit_softcapping: float | None = None,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx: int = layer_idx
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.num_kv_groups: int = max(1, num_heads // num_kv_heads)
        self.head_dim: int = head_dim
        self.rotary_dim: int = rotary_dim
        self.attn_scale: float = float(attn_scale)
        self.attn_logit_softcapping: float | None = attn_logit_softcapping
        self.sliding_window: int | None = sliding_window

        self.q_proj: nn.Module | None = None
        self.k_proj: nn.Module | None = None
        self.v_proj: nn.Module | None = None
        self.qkv_proj: nn.Module | None = None
        self.o_proj: nn.Module | None = None
        self.q_norm: GgufRMSNorm | None = None
        self.k_norm: GgufRMSNorm | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute GQA self-attention with RoPE, KV-cache, sliding window, and optional softcapping."""
        del position_ids
        bsz, seq_len, _ = hidden_states.shape
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(hidden_states)
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q_flat = qkv[..., :q_size]
            k_flat = qkv[..., q_size : q_size + kv_size]
            v_flat = qkv[..., q_size + kv_size :]
        else:
            if self.q_proj is None or self.k_proj is None or self.v_proj is None:
                raise RuntimeError(f"Attention projections missing on layer {self.layer_idx}")
            q_flat = self.q_proj(hidden_states)
            k_flat = self.k_proj(hidden_states)
            v_flat = self.v_proj(hidden_states)

        query_states = q_flat.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = v_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rotary_dim
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        kv_len = int(key_states.shape[-2])
        eff_mask = attention_mask
        if (
            self.sliding_window is not None
            and self.sliding_window > 0
            and kv_len > self.sliding_window
        ):
            past_len = kv_len - seq_len
            q_pos = torch.arange(past_len, kv_len, device=query_states.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
            too_old = (q_pos - k_pos) >= self.sliding_window
            if too_old.any():
                min_val = torch.finfo(query_states.dtype).min
                win_mask = torch.zeros(
                    (1, 1, seq_len, kv_len),
                    dtype=query_states.dtype,
                    device=query_states.device,
                ).masked_fill(too_old.unsqueeze(0).unsqueeze(0), min_val)
                eff_mask = win_mask if eff_mask is None else torch.minimum(eff_mask, win_mask)

        if self.attn_logit_softcapping is not None and self.attn_logit_softcapping > 0:
            cap = float(self.attn_logit_softcapping)
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(-2, -1)) * self.attn_scale
            )
            attn_weights = torch.tanh(attn_weights / cap) * cap
            if eff_mask is not None:
                attn_weights = attn_weights + eff_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=eff_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.attn_scale,
            )

        if self.o_proj is None:
            raise RuntimeError(f"Attention o_proj missing on layer {self.layer_idx}")
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, seq_len, self.num_heads * self.head_dim)
        )
        return self.o_proj(attn_output)


class GgufMLP(nn.Module):
    """Pure-PyTorch gated MLP supporting SwiGLU, GeGLU (`gemma2`/`gemma3`), and fused `phi3`."""

    def __init__(self, use_gelu_tanh: bool = False) -> None:
        super().__init__()
        self.use_gelu_tanh: bool = use_gelu_tanh
        self.gate_proj: nn.Module | None = None
        self.up_proj: nn.Module | None = None
        self.gate_up_proj: nn.Module | None = None
        self.down_proj: nn.Module | None = None

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gelu_tanh:
            return F.gelu(x, approximate="tanh")
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated activation and down projection."""
        if self.down_proj is None:
            raise RuntimeError("MLP down_proj is not initialized")
        if self.gate_up_proj is not None:
            up_states = self.gate_up_proj(x)
            gate, up = up_states.chunk(2, dim=-1)
            return self.down_proj(self._activate(gate) * up)
        if self.gate_proj is None or self.up_proj is None:
            raise RuntimeError("MLP gate_proj/up_proj are not initialized")
        return self.down_proj(self._activate(self.gate_proj(x)) * self.up_proj(x))


class GgufDecoderLayer(nn.Module):
    """Pure-PyTorch decoder block compatible with `ShardModelRunner.forward_pass`."""

    def __init__(
        self,
        *,
        self_attn: GgufAttention,
        mlp: GgufMLP,
        input_layernorm: GgufRMSNorm,
        post_attention_layernorm: GgufRMSNorm,
        pre_feedforward_layernorm: GgufRMSNorm | None = None,
        post_feedforward_layernorm: GgufRMSNorm | None = None,
    ) -> None:
        super().__init__()
        self.self_attn: GgufAttention = self_attn
        self.mlp: GgufMLP = mlp
        self.input_layernorm: GgufRMSNorm = input_layernorm
        self.post_attention_layernorm: GgufRMSNorm = post_attention_layernorm
        self.pre_feedforward_layernorm: GgufRMSNorm | None = pre_feedforward_layernorm
        self.post_feedforward_layernorm: GgufRMSNorm | None = post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = True,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor]:
        """Run pre-norm (or Gemma sandwich-norm) self-attention + MLP with residual connections."""
        del cache_position
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values if use_cache else None,
            position_embeddings=position_embeddings,
        )
        if (
            self.pre_feedforward_layernorm is not None
            and self.post_feedforward_layernorm is not None
        ):
            attn_out = self.post_attention_layernorm(attn_out)
            hidden_states = residual + attn_out
            residual = hidden_states
            ffn_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
            ffn_out = self.post_feedforward_layernorm(ffn_out)
            hidden_states = residual + ffn_out
        else:
            hidden_states = residual + attn_out
            residual = hidden_states
            ffn_out = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = residual + ffn_out
        return (hidden_states,)


def _build_model_config_from_gguf(index: GgufFileIndex) -> GgufModelConfig:
    """Reconstruct a pure-PyTorch `GgufModelConfig` from GGUF header metadata."""
    arch = str(index.metadata.get("general.architecture") or "").strip().lower()
    if arch not in SUPPORTED_GGUF_ARCHITECTURES:
        raise ValueError(f"Unsupported GGUF architecture: '{arch}'")

    def _meta(suffix: str, default: Any = None) -> Any:
        return index.metadata.get(f"{arch}.{suffix}", default)

    num_hidden_layers = int(_meta("block_count"))
    hidden_size = int(_meta("embedding_length"))
    intermediate_size = int(_meta("feed_forward_length"))
    max_position_embeddings = int(_meta("context_length", 2048))
    num_attention_heads = int(_meta("attention.head_count"))
    num_key_value_heads = int(_meta("attention.head_count_kv", num_attention_heads))

    head_dim_raw = _meta("attention.key_length")
    if head_dim_raw is not None and int(head_dim_raw) > 0:
        head_dim = int(head_dim_raw)
    elif (
        "blk.0.attn_q.weight" in index.tensors
        and int(index.tensors["blk.0.attn_q.weight"].shape_pt[0]) > 0
        and int(index.tensors["blk.0.attn_q.weight"].shape_pt[0]) % num_attention_heads == 0
    ):
        head_dim = int(index.tensors["blk.0.attn_q.weight"].shape_pt[0]) // num_attention_heads
    else:
        head_dim = hidden_size // num_attention_heads

    rms_norm_eps = float(_meta("attention.layer_norm_rms_epsilon", 1e-6))
    rope_theta = float(_meta("rope.freq_base", 10000.0))

    rope_dim_raw = _meta("rope.dimension_count")
    if rope_dim_raw is not None and 0 < int(rope_dim_raw) <= head_dim:
        rotary_dim = int(rope_dim_raw)
    else:
        rotary_dim = head_dim

    sliding_window_raw = _meta("attention.sliding_window")
    sliding_window = (
        int(sliding_window_raw)
        if sliding_window_raw is not None and int(sliding_window_raw) > 0
        else None
    )

    final_softcap_raw = _meta("final_logit_softcapping")
    final_logit_softcapping = (
        float(final_softcap_raw)
        if final_softcap_raw is not None and float(final_softcap_raw) > 0.0
        else None
    )

    attn_softcap_raw = _meta("attn_logit_softcapping")
    attn_logit_softcapping = (
        float(attn_softcap_raw)
        if attn_softcap_raw is not None and float(attn_softcap_raw) > 0.0
        else None
    )

    q_scalar_raw = _meta("attention.query_pre_attn_scalar")
    if q_scalar_raw is not None and float(q_scalar_raw) > 0.0:
        query_pre_attn_scalar = float(q_scalar_raw)
    elif arch in ("gemma2", "gemma3"):
        query_pre_attn_scalar = 256.0
    else:
        query_pre_attn_scalar = float(head_dim)

    if "token_embd.weight" in index.tensors:
        vocab_size = int(index.tensors["token_embd.weight"].shape_pt[0])
    elif index.vocab_size_from_tokens is not None:
        vocab_size = int(index.vocab_size_from_tokens)
    elif "output.weight" in index.tensors:
        vocab_size = int(index.tensors["output.weight"].shape_pt[0])
    else:
        raise ValueError("Unable to determine vocab_size from GGUF tensors or metadata")

    tie_word_embeddings = "output.weight" not in index.tensors
    bos_raw = index.metadata.get("tokenizer.ggml.bos_token_id")
    eos_raw = index.metadata.get("tokenizer.ggml.eos_token_id")
    pad_raw = index.metadata.get("tokenizer.ggml.padding_token_id")

    bos_token_id = int(bos_raw) if bos_raw is not None else None
    eos_token_id = int(eos_raw) if eos_raw is not None else None
    pad_token_id = (
        int(pad_raw) if pad_raw is not None and 0 <= int(pad_raw) < vocab_size else None
    )

    return GgufModelConfig(
        arch=arch,
        model_type=arch,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rotary_dim=rotary_dim,
        sliding_window=sliding_window,
        final_logit_softcapping=final_logit_softcapping,
        attn_logit_softcapping=attn_logit_softcapping,
        query_pre_attn_scalar=query_pre_attn_scalar,
        tie_word_embeddings=tie_word_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def _read_tensor_raw_u8(f: BinaryIO, index: GgufFileIndex, t_info: GgufTensorInfo) -> torch.Tensor:
    """Seek-read raw tensor bytes as a 1D `torch.uint8` tensor without loading other tensors."""
    f.seek(index.data_offset + t_info.rel_offset)
    raw_bytes = f.read(t_info.nbytes)
    if len(raw_bytes) < t_info.nbytes:
        raise EOFError(f"Unexpected EOF reading tensor '{t_info.name}'")
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
    return torch.from_numpy(arr)


def _read_tensor_dense(
    f: BinaryIO,
    index: GgufFileIndex,
    t_info: GgufTensorInfo,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Seek-read and dequantize one tensor (used for small 1D norms & biases)."""
    raw_u8 = _read_tensor_raw_u8(f, index, t_info)
    return _dequantize_blocks_u8(raw_u8, ggml_type=t_info.ggml_type, dtype=dtype).reshape(
        t_info.shape_pt
    )


def _make_linear_from_gguf(
    f: BinaryIO,
    index: GgufFileIndex,
    weight_name: str,
    bias_name: str | None = None,
    *,
    unpermute_heads: int | None = None,
    chunk_out_rows: int | None = None,
) -> GgufQuantizedLinear:
    """Construct a `GgufQuantizedLinear` by seek-reading only its packed bytes."""
    w_info = index.tensors[weight_name]
    if len(w_info.shape_pt) != 2:
        raise ValueError(f"Expected 2D weight for '{weight_name}', got {w_info.shape_pt}")
    out_features, in_features = w_info.shape_pt
    raw_u8 = _read_tensor_raw_u8(f, index, w_info)
    bias_tensor: torch.Tensor | None = None
    if bias_name is not None and bias_name in index.tensors:
        b_info = index.tensors[bias_name]
        bias_tensor = _read_tensor_dense(f, index, b_info, dtype=torch.float32)
        if unpermute_heads is not None:
            bias_tensor = _reverse_permute_rope_rows(bias_tensor, unpermute_heads)
    return GgufQuantizedLinear(
        raw_weight=raw_u8,
        out_features=out_features,
        in_features=in_features,
        ggml_type=w_info.ggml_type,
        bias=bias_tensor,
        unpermute_heads=unpermute_heads,
        chunk_out_rows=chunk_out_rows,
    )


def _make_norm_from_gguf(
    f: BinaryIO,
    index: GgufFileIndex,
    tensor_name: str,
    *,
    fallback_dim: int,
    eps: float,
) -> GgufRMSNorm:
    """Load 1D RMSNorm effective weights directly from GGUF."""
    if tensor_name in index.tensors:
        w = _read_tensor_dense(f, index, index.tensors[tensor_name], dtype=torch.float32)
    else:
        w = torch.ones(fallback_dim, dtype=torch.float32)
    return GgufRMSNorm(weight=w, eps=eps)


def _build_layer_from_gguf(
    layer_idx: int,
    cfg: GgufModelConfig,
    f: BinaryIO,
    index: GgufFileIndex,
) -> GgufDecoderLayer:
    """Construct one `GgufDecoderLayer` by selectively seek-reading its quantized weights."""
    arch = cfg.arch
    gemma_sandwich = arch in ("gemma2", "gemma3")
    is_llama_rope_perm = arch in ("llama", "mistral")
    prefix = f"blk.{layer_idx}"

    if arch == "gemma2":
        layer_sw = cfg.sliding_window if ((layer_idx + 1) % 2) else None
    elif arch == "gemma3":
        layer_sw = cfg.sliding_window if ((layer_idx + 1) % 6) else None
    else:
        layer_sw = cfg.sliding_window

    attn = GgufAttention(
        layer_idx=layer_idx,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rotary_dim=cfg.rotary_dim,
        attn_scale=cfg.query_pre_attn_scalar**-0.5,
        attn_logit_softcapping=cfg.attn_logit_softcapping,
        sliding_window=layer_sw,
    )

    if f"{prefix}.attn_q_norm.weight" in index.tensors:
        attn.q_norm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.attn_q_norm.weight",
            fallback_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
        )
    if f"{prefix}.attn_k_norm.weight" in index.tensors:
        attn.k_norm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.attn_k_norm.weight",
            fallback_dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
        )

    if arch == "phi3" and f"{prefix}.attn_qkv.weight" in index.tensors:
        attn.qkv_proj = _make_linear_from_gguf(
            f, index, f"{prefix}.attn_qkv.weight", f"{prefix}.attn_qkv.bias"
        )
    else:
        if f"{prefix}.attn_q.weight" in index.tensors:
            attn.q_proj = _make_linear_from_gguf(
                f,
                index,
                f"{prefix}.attn_q.weight",
                f"{prefix}.attn_q.bias",
                unpermute_heads=cfg.num_attention_heads if is_llama_rope_perm else None,
            )
        if f"{prefix}.attn_k.weight" in index.tensors:
            attn.k_proj = _make_linear_from_gguf(
                f,
                index,
                f"{prefix}.attn_k.weight",
                f"{prefix}.attn_k.bias",
                unpermute_heads=cfg.num_key_value_heads if is_llama_rope_perm else None,
            )
        if f"{prefix}.attn_v.weight" in index.tensors:
            attn.v_proj = _make_linear_from_gguf(
                f, index, f"{prefix}.attn_v.weight", f"{prefix}.attn_v.bias"
            )

    if f"{prefix}.attn_output.weight" in index.tensors:
        attn.o_proj = _make_linear_from_gguf(
            f, index, f"{prefix}.attn_output.weight", f"{prefix}.attn_output.bias"
        )

    mlp = GgufMLP(use_gelu_tanh=gemma_sandwich)
    if (
        arch == "phi3"
        and f"{prefix}.ffn_up.weight" in index.tensors
        and f"{prefix}.ffn_gate.weight" not in index.tensors
    ):
        mlp.gate_up_proj = _make_linear_from_gguf(
            f, index, f"{prefix}.ffn_up.weight", f"{prefix}.ffn_up.bias"
        )
    else:
        if f"{prefix}.ffn_gate.weight" in index.tensors:
            mlp.gate_proj = _make_linear_from_gguf(
                f, index, f"{prefix}.ffn_gate.weight", f"{prefix}.ffn_gate.bias"
            )
        if f"{prefix}.ffn_up.weight" in index.tensors:
            mlp.up_proj = _make_linear_from_gguf(
                f, index, f"{prefix}.ffn_up.weight", f"{prefix}.ffn_up.bias"
            )
    if f"{prefix}.ffn_down.weight" in index.tensors:
        mlp.down_proj = _make_linear_from_gguf(
            f, index, f"{prefix}.ffn_down.weight", f"{prefix}.ffn_down.bias"
        )

    input_layernorm = _make_norm_from_gguf(
        f,
        index,
        f"{prefix}.attn_norm.weight",
        fallback_dim=cfg.hidden_size,
        eps=cfg.rms_norm_eps,
    )
    if gemma_sandwich:
        post_attention_layernorm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.post_attention_norm.weight",
            fallback_dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
        )
        pre_feedforward_layernorm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.ffn_norm.weight",
            fallback_dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
        )
        post_feedforward_layernorm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.post_ffw_norm.weight",
            fallback_dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
        )
    else:
        post_attention_layernorm = _make_norm_from_gguf(
            f,
            index,
            f"{prefix}.ffn_norm.weight",
            fallback_dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
        )
        pre_feedforward_layernorm = None
        post_feedforward_layernorm = None

    return GgufDecoderLayer(
        self_attn=attn,
        mlp=mlp,
        input_layernorm=input_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        pre_feedforward_layernorm=pre_feedforward_layernorm,
        post_feedforward_layernorm=post_feedforward_layernorm,
    )


# ---------------------------------------------------------------------------
# Lightweight GGUF Tokenizer Adapter (Head / Tail Nodes Only)
# ---------------------------------------------------------------------------


class GgufTokenizerAdapter:
    """Lightweight tokenizer backed by `llama_cpp(vocab_only=True)` or GGUF token fallback."""

    def __init__(self, index: GgufFileIndex) -> None:
        self.chat_template: str | None = (
            str(index.metadata["tokenizer.chat_template"])
            if index.metadata.get("tokenizer.chat_template")
            else None
        )
        self.bos_token_id: int | None = (
            int(index.metadata["tokenizer.ggml.bos_token_id"])
            if "tokenizer.ggml.bos_token_id" in index.metadata
            else None
        )
        self.eos_token_id: int | None = (
            int(index.metadata["tokenizer.ggml.eos_token_id"])
            if "tokenizer.ggml.eos_token_id" in index.metadata
            else None
        )
        self.pad_token_id: int | None = (
            int(index.metadata["tokenizer.ggml.padding_token_id"])
            if "tokenizer.ggml.padding_token_id" in index.metadata
            else None
        )
        self._add_bos_token: bool = bool(index.metadata.get("tokenizer.ggml.add_bos_token", False))
        self._llama: Any | None = None
        self._vocab_tokens: list[str] = []
        self._token_to_id: dict[str, int] = {}
        self._special_ids: set[int] = set()

        for tid in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
            if tid is not None and tid >= 0:
                self._special_ids.add(tid)

        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]

            self._llama = Llama(
                model_path=str(index.path),
                vocab_only=True,
                use_mmap=True,
                verbose=False,
            )
            if self.bos_token_id is None and hasattr(self._llama, "token_bos"):
                bos = int(self._llama.token_bos())
                if bos >= 0:
                    self.bos_token_id = bos
                    self._special_ids.add(bos)
            if self.eos_token_id is None and hasattr(self._llama, "token_eos"):
                eos = int(self._llama.token_eos())
                if eos >= 0:
                    self.eos_token_id = eos
                    self._special_ids.add(eos)
            for special_tok in ("<|im_end|>", "<|eot_id|>", "<|im_start|>", "<|endoftext|>", "</s>"):
                sid = self.convert_tokens_to_ids(special_tok)
                if sid is not None and sid >= 0:
                    self._special_ids.add(sid)
        except Exception as exc:
            logger.debug(
                "Falling back to GGUF metadata tokenizer for %s (%s)",
                index.path.name,
                exc,
            )
            self._llama = None
            self._vocab_tokens = _read_gguf_tokens_list(index)
            self._token_to_id = {tok: idx for idx, tok in enumerate(self._vocab_tokens)}
            for special_tok in ("<|im_end|>", "<|eot_id|>", "<|im_start|>", "<|endoftext|>", "</s>", "<s>"):
                if special_tok in self._token_to_id:
                    self._special_ids.add(self._token_to_id[special_tok])

    def convert_tokens_to_ids(self, token: str) -> int | None:
        """Map a single token string (e.g. `<|im_end|>`) to its integer ID."""
        if self._llama is not None:
            try:
                ids = self._llama.tokenize(
                    token.encode("utf-8"), add_bos=False, special=True
                )
                if len(ids) == 1:
                    return int(ids[0])
            except Exception:
                return None
            return None
        return self._token_to_id.get(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text into integer token IDs."""
        if self._llama is not None:
            add_bos = self._add_bos_token if add_special_tokens else False
            ids = self._llama.tokenize(
                text.encode("utf-8"), add_bos=add_bos, special=True
            )
            return [int(t) for t in ids]

        tokens: list[int] = []
        if add_special_tokens and self._add_bos_token and self.bos_token_id is not None:
            tokens.append(self.bos_token_id)

        if not self._token_to_id:
            return tokens or [self.bos_token_id or 0]

        i = 0
        max_tok_len = max((len(t) for t in self._token_to_id), default=1)
        n = len(text)
        while i < n:
            matched_id: int | None = None
            matched_len = 0
            for span in range(min(max_tok_len, n - i), 0, -1):
                piece = text[i : i + span]
                if piece in self._token_to_id:
                    matched_id = self._token_to_id[piece]
                    matched_len = span
                    break
            if matched_id is not None:
                tokens.append(matched_id)
                i += matched_len
            else:
                tokens.append(0)
                i += 1
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode integer token IDs back to UTF-8 text."""
        filtered = (
            [int(t) for t in token_ids if int(t) not in self._special_ids]
            if skip_special_tokens
            else [int(t) for t in token_ids]
        )
        if not filtered:
            return ""
        if self._llama is not None:
            raw_bytes: bytes = self._llama.detokenize(filtered)
            return raw_bytes.decode("utf-8", errors="replace")

        parts: list[str] = []
        for tid in filtered:
            if 0 <= tid < len(self._vocab_tokens):
                parts.append(self._vocab_tokens[tid])
        return "".join(parts)

    def _render_chatml_fallback(
        self, messages: list[dict[str, str]], add_generation_prompt: bool
    ) -> str:
        """Render safe ChatML prompt format when Jinja template is absent or fails."""
        chunks: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
        return "".join(chunks)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> list[int] | str:
        """Render `messages` using the GGUF Jinja chat template (sandboxed) or ChatML fallback."""
        rendered: str | None = None
        if self.chat_template:
            try:
                from jinja2.sandbox import SandboxedEnvironment  # type: ignore[import-not-found]

                env = SandboxedEnvironment()
                bos_str = ""
                eos_str = ""
                if self.bos_token_id is not None:
                    bos_str = self.decode([self.bos_token_id], skip_special_tokens=False)
                if self.eos_token_id is not None:
                    eos_str = self.decode([self.eos_token_id], skip_special_tokens=False)

                def _raise_exc(msg: str) -> None:
                    raise ValueError(msg)

                tmpl = env.from_string(self.chat_template)
                rendered = tmpl.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token=bos_str,
                    eos_token=eos_str,
                    raise_exception=_raise_exc,
                )
            except Exception as exc:
                logger.debug("Chat template Jinja render fallback to ChatML: %s", exc)
                rendered = None

        if rendered is None:
            rendered = self._render_chatml_fallback(messages, add_generation_prompt)

        if not tokenize:
            return rendered
        return self.encode(rendered, add_special_tokens=True)

    def close(self) -> None:
        """Release underlying `llama_cpp.Llama` vocab handle if open."""
        if self._llama is not None:
            close_fn = getattr(self._llama, "close", None)
            if callable(close_fn):
                close_fn()
            self._llama = None


# ---------------------------------------------------------------------------
# Public Selective Shard Entrypoint
# ---------------------------------------------------------------------------


def _iter_layer_linears(layer: GgufDecoderLayer) -> list[GgufQuantizedLinear]:
    """Return all `GgufQuantizedLinear` projections belonging to one decoder layer."""
    modules = (
        layer.self_attn.qkv_proj,
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_up_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    )
    return [m for m in modules if isinstance(m, GgufQuantizedLinear)]


def _move_norm_module(
    mod: GgufRMSNorm | None, device: torch.device, dtype: torch.dtype
) -> None:
    """Move a lightweight `GgufRMSNorm` module onto `(device, dtype)` cleanly."""
    if mod is not None:
        mod.to(device=device, dtype=dtype)


def load_gguf_pytorch_shard(
    gguf_path: Path,
    shard_layer_start: int,
    shard_layer_end: int,
    max_ram_mb: int = 4096,
    device: str | None = None,
) -> LoadedGgufShard:
    """Selectively load ONLY `[shard_layer_start..shard_layer_end]` from a GGUF file into pure PyTorch."""
    index = read_gguf_header_and_tensors(gguf_path)
    arch = str(index.metadata.get("general.architecture") or "").strip().lower()
    if arch not in SUPPORTED_GGUF_ARCHITECTURES:
        raise ValueError(f"Unsupported GGUF architecture: '{arch}'")

    cfg = _build_model_config_from_gguf(index)
    total_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    clamped_start = max(0, min(shard_layer_start, total_layers - 1))
    clamped_end = max(clamped_start, min(shard_layer_end, total_layers - 1))
    is_head = clamped_start == 0
    is_tail = clamped_end >= (total_layers - 1)

    container = nn.Module()
    rotary_emb = GgufRotaryEmbedding(rotary_dim=cfg.rotary_dim, rope_theta=cfg.rope_theta)
    container.rotary_emb = rotary_emb

    embed_tokens: GgufQuantizedEmbedding | None = None
    norm: GgufRMSNorm | None = None
    lm_head: GgufQuantizedLinear | None = None

    embed_scale = math.sqrt(float(hidden_dim)) if arch in ("gemma2", "gemma3") else 1.0

    # Estimate kept tensor weight size to choose hardware device, compute dtype, and materialization budget
    kept_bytes = 0
    for idx in range(clamped_start, clamped_end + 1):
        prefix = f"blk.{idx}."
        for name, t_info in index.tensors.items():
            if name.startswith(prefix):
                kept_bytes += t_info.nbytes
    if is_head and "token_embd.weight" in index.tensors:
        kept_bytes += index.tensors["token_embd.weight"].nbytes
    if is_tail:
        if "output.weight" in index.tensors:
            kept_bytes += index.tensors["output.weight"].nbytes
        elif "token_embd.weight" in index.tensors:
            kept_bytes += index.tensors["token_embd.weight"].nbytes

    target_device = select_gguf_torch_device(device, kept_bytes=kept_bytes)
    budget_bytes = max(1, int(max_ram_mb * 0.80 * 1024 * 1024))

    kept_layers: list[GgufDecoderLayer] = []
    with gguf_path.open("rb") as f:
        for idx in range(clamped_start, clamped_end + 1):
            kept_layers.append(_build_layer_from_gguf(idx, cfg, f, index))

        container.layers = nn.ModuleList(kept_layers)

        if is_head and "token_embd.weight" in index.tensors:
            emb_info = index.tensors["token_embd.weight"]
            vocab_size, emb_dim = emb_info.shape_pt
            emb_raw = _read_tensor_raw_u8(f, index, emb_info)
            embed_tokens = GgufQuantizedEmbedding(
                raw_weight=emb_raw,
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                ggml_type=emb_info.ggml_type,
                embed_scale=embed_scale,
                output_dtype=torch.float32,
            )
            container.embed_tokens = embed_tokens

        if is_tail:
            norm = _make_norm_from_gguf(
                f,
                index,
                "output_norm.weight",
                fallback_dim=hidden_dim,
                eps=cfg.rms_norm_eps,
            )
            container.norm = norm
            lm_tensor_name = (
                "output.weight"
                if "output.weight" in index.tensors
                else ("token_embd.weight" if "token_embd.weight" in index.tensors else None)
            )
            if lm_tensor_name is not None:
                lm_head = _make_linear_from_gguf(
                    f,
                    index,
                    lm_tensor_name,
                    chunk_out_rows=_DEFAULT_LM_HEAD_CHUNK_ROWS,
                )
                container.lm_head = lm_head

    decoder_linears: list[GgufQuantizedLinear] = []
    for layer in kept_layers:
        decoder_linears.extend(_iter_layer_linears(layer))

    all_quantized_linears: list[GgufQuantizedLinear] = [
        lin for lin in [*decoder_linears, *([lm_head] if lm_head is not None else [])]
        if lin.ggml_type in NATIVE_QUANTIZED_GGML_TYPES
    ]

    ggml_engine: GgmlNativeEngine | None = None
    native_ggml_count = 0
    if all_quantized_linears and is_ggml_native_enabled():
        try:
            ggml_engine = GgmlNativeEngine()
        except Exception as exc:
            logger.warning(
                "Could not initialize native libggml engine (%s); falling back to PyTorch.",
                exc,
            )
            ggml_engine = None

    try:
        if ggml_engine is not None and ggml_engine.is_cpu and device is None:
            # Keep activations on CPU in float32 to eliminate per-op GPU transfer overhead
            target_device = torch.device("cpu")
            compute_dtype = torch.float32
            mat_dtype = torch.float32

        if ggml_engine is not None:
            for lin in all_quantized_linears:
                if lin.attach_ggml_engine(ggml_engine):
                    native_ggml_count += 1
            if native_ggml_count == 0:
                ggml_engine.close()
                ggml_engine = None

        dense_linears = [
            lin for lin in decoder_linears if lin._ggml_handle is None
        ]
        decoder_elems = sum(lin.out_features * lin.in_features for lin in dense_linears)
        decoder_raw_bytes = sum(int(lin.raw_weight.numel()) for lin in dense_linears)
        non_decoder_bytes = kept_bytes

        if ggml_engine is not None and ggml_engine.is_cpu and device is None:
            compute_dtype = torch.float32
            mat_dtype = torch.float32
        elif target_device.type == "cuda":
            bf16_supported = bool(
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
            compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
            mat_dtype = compute_dtype
        elif target_device.type == "mps":
            compute_dtype = torch.bfloat16
            mat_dtype = torch.bfloat16
        else:
            f32_total = non_decoder_bytes - decoder_raw_bytes + (decoder_elems * 4)
            if f32_total <= budget_bytes:
                compute_dtype = torch.float32
                mat_dtype = torch.float32
            else:
                compute_dtype = torch.bfloat16
                mat_dtype = torch.bfloat16

        for layer in kept_layers:
            _move_norm_module(layer.input_layernorm, target_device, compute_dtype)
            _move_norm_module(layer.post_attention_layernorm, target_device, compute_dtype)
            _move_norm_module(layer.pre_feedforward_layernorm, target_device, compute_dtype)
            _move_norm_module(layer.post_feedforward_layernorm, target_device, compute_dtype)
            _move_norm_module(layer.self_attn.q_norm, target_device, compute_dtype)
            _move_norm_module(layer.self_attn.k_norm, target_device, compute_dtype)
        _move_norm_module(norm, target_device, compute_dtype)
        rotary_emb.to(device=target_device)

        if embed_tokens is not None:
            embed_tokens.output_dtype = compute_dtype
            embed_tokens.target_device = target_device

        used_bytes = non_decoder_bytes
        elem_bytes = 4 if mat_dtype == torch.float32 else 2
        materialized_count = 0
        for lin in dense_linears:
            projected_delta = (lin.out_features * lin.in_features * elem_bytes) - int(
                lin.raw_weight.numel()
            )
            if (used_bytes + projected_delta) > budget_bytes:
                continue
            try:
                actual_delta = lin.materialize(dtype=mat_dtype, device=target_device)
                used_bytes += actual_delta
                materialized_count += 1
            except RuntimeError as exc:
                logger.warning(
                    "Stopping weight materialization early on %s after %d tensors (%s)",
                    target_device,
                    materialized_count,
                    exc,
                )
                break

        if (
            lm_head is not None
            and lm_head._ggml_handle is None
            and target_device.type != "mps"
            and lm_head.out_features <= _DEFAULT_LM_HEAD_CHUNK_ROWS
        ):
            lm_delta = (lm_head.out_features * lm_head.in_features * elem_bytes) - int(
                lm_head.raw_weight.numel()
            )
            if (used_bytes + lm_delta) <= int(budget_bytes * 0.7):
                try:
                    used_bytes += lm_head.materialize(dtype=mat_dtype, device=target_device)
                    materialized_count += 1
                except RuntimeError:
                    pass

        container.eval()
        for param in container.parameters():
            param.requires_grad_(False)

        logger.info(
            "GGUF shard %d..%d prepared on %s (dtype=%s, ggml_native=%s [%d linears], materialized=%d/%d, est_ram=%.1f MB)",
            clamped_start,
            clamped_end,
            target_device,
            compute_dtype,
            ggml_engine.backend_kind if ggml_engine is not None else "off",
            native_ggml_count,
            materialized_count,
            len(decoder_linears),
            used_bytes / (1024.0 * 1024.0),
        )

        tokenizer: GgufTokenizerAdapter | None = None
        if is_head or is_tail:
            tokenizer = GgufTokenizerAdapter(index)

        return LoadedGgufShard(
            hf_config=cfg,
            container=container,
            layers=kept_layers,
            rotary_emb=rotary_emb,
            embed_tokens=embed_tokens,
            norm=norm,
            lm_head=lm_head,
            tokenizer=tokenizer,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            shard_layer_start=clamped_start,
            shard_layer_end=clamped_end,
            compute_dtype=compute_dtype,
            device=target_device,
            materialized_tensors=materialized_count,
            ggml_engine=ggml_engine,
            native_ggml_tensors=native_ggml_count,
        )
    except BaseException:
        if ggml_engine is not None:
            try:
                ggml_engine.close()
            except Exception:
                pass
        raise
