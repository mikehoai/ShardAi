"""Hugging Face Hub Model & Shard Downloader with Real-Time SSE Progress Broadcasting.

Complies with `.antigravity/rules.md` and `.antigravity/review.md`:
- Uses `huggingface_hub.hf_hub_download` to fetch `.safetensors` layer shards (or `.gguf` files).
- Supports selective shard downloading via `model.safetensors.index.json` so each peer only
  downloads the `.safetensors` files containing its assigned transformer layers.
- Verifies canonical paths (`Path.resolve()`) inside `cache_models/` to prevent Path Traversal.
- Broadcasts live percentage updates (`0%` .. `100%`) over Server-Sent Events (SSE) for the UI.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files  # type: ignore[import-not-found]
except ImportError:
    HfApi = None
    hf_hub_download = None
    list_repo_files = None

logger = logging.getLogger("shardai.models.downloader")

_HF_ESSENTIAL_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)


def is_offline_mode() -> bool:
    """Return True when strict offline simulation mode is enabled via `SHARDAI_OFFLINE_MODE`."""
    return os.getenv("SHARDAI_OFFLINE_MODE", "false").lower() == "true"


@dataclass(frozen=True, slots=True)
class GgufModelSpec:
    """Mapping from logical `model_id` to its Hugging Face repository and hardware specs."""

    model_id: str
    hf_repo_id: str
    gguf_filename: str
    total_layers: int
    estimated_total_size_gb: float
    base_overhead_gb: float = 0.6


SUPPORTED_MODELS: dict[str, GgufModelSpec] = {
    "Qwen/Qwen2.5-0.5B-Instruct": GgufModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        gguf_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        total_layers=24,
        estimated_total_size_gb=1.0,
        base_overhead_gb=0.35,
    ),
    "HuggingFaceTB/SmolLM2-360M-Instruct": GgufModelSpec(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        hf_repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        gguf_filename="smollm2-360m-instruct-q4_k_m.gguf",
        total_layers=32,
        estimated_total_size_gb=0.75,
        base_overhead_gb=0.30,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": GgufModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        hf_repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        gguf_filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        total_layers=32,
        estimated_total_size_gb=5.0,
        base_overhead_gb=0.6,
    ),
    "Qwen/Qwen2.5-7B-Instruct": GgufModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        hf_repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        gguf_filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        total_layers=28,
        estimated_total_size_gb=4.7,
        base_overhead_gb=0.6,
    ),
}

FINGERPRINT_HEAD_BYTES: int = 10 * 1024 * 1024  # 10 MB prefix for fast fingerprint hashing
FINGERPRINT_CHUNK_BYTES: int = 1_048_576  # 1 MiB bounded read buffer
GGUF_MAGIC: bytes = b"GGUF"
MIN_ONLINE_GGUF_BYTES: int = 1_048_576
MAX_DIRECT_URL_REDIRECTS: int = 5
_REDIRECT_STATUS_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_FORBIDDEN_DIRECT_URL_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)


def is_forbidden_ssrf_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if `ip_obj` is private, loopback, link-local, multicast, reserved, or non-global."""
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return is_forbidden_ssrf_ip(ip_obj.ipv4_mapped)
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
        or not ip_obj.is_global
    )


def _resolve_hostname_ips(hostname: str, port: int) -> list[str]:
    """Resolve `hostname:port` via system DNS into deduplicated IP strings."""
    infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    seen: list[str] = []
    for _family, _socktype, _proto, _canonname, sockaddr in infos:
        raw_ip = str(sockaddr[0]).split("%", 1)[0]
        if raw_ip and raw_ip not in seen:
            seen.append(raw_ip)
    return seen


def validate_direct_url_syntax_and_host(url: str) -> tuple[str, int]:
    """Validate HTTP/HTTPS `direct_url` syntax and reject restricted hostnames and IP literals."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"Direct URL must be a valid http:// or https:// URL: '{url}'")

    host = parsed.hostname.rstrip(".").lower()
    if host in _FORBIDDEN_DIRECT_URL_HOSTS or host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise PermissionError(f"Rejected restricted direct_url hostname '{host}'.")

    try:
        ip_obj = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        ip_obj = None

    if ip_obj is not None and is_forbidden_ssrf_ip(ip_obj):
        raise PermissionError(f"Rejected non-public direct_url IP address '{host}'.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def has_gguf_magic(file_path: Path) -> bool:
    """Return True iff `file_path` exists and begins with the 4-byte `GGUF` magic header."""
    if not file_path.is_file():
        return False
    try:
        with file_path.open("rb") as handle:
            return handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC
    except OSError:
        return False


def is_valid_gguf_file(file_path: Path, min_bytes: int = MIN_ONLINE_GGUF_BYTES) -> bool:
    """Return True iff `file_path` meets `min_bytes` and begins with the `GGUF` magic header."""
    if not file_path.is_file():
        return False
    try:
        required_bytes = max(len(GGUF_MAGIC), min_bytes)
        if file_path.stat().st_size < required_bytes:
            return False
    except OSError:
        return False
    return has_gguf_magic(file_path)


def parse_estimated_ram_gb(value: Any, default: float = 5.0) -> float:
    """Parse `estimated_ram_gb` defensively, falling back to `default` on malformed input."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        try:
            parsed = float(cleaned)
        except ValueError:
            return default
    else:
        return default

    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return max(0.1, parsed)


def sanitize_catalog_entry(entry: Any) -> dict[str, Any] | None:
    """Normalize a raw catalog dictionary, returning `None` when disabled or missing `repo_id`."""
    if not isinstance(entry, dict):
        return None
    if entry.get("enabled", True) is not True:
        return None

    raw_repo_id = entry.get("repo_id")
    if not isinstance(raw_repo_id, str) or not raw_repo_id.strip():
        return None
    repo_id = raw_repo_id.strip()

    raw_id = entry.get("id")
    entry_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else repo_id

    raw_name = entry.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else repo_id

    raw_filename = entry.get("filename")
    filename = raw_filename.strip() if isinstance(raw_filename, str) and raw_filename.strip() else None

    return {
        "id": entry_id,
        "name": name,
        "repo_id": repo_id,
        "filename": filename,
        "estimated_ram_gb": parse_estimated_ram_gb(entry.get("estimated_ram_gb"), default=5.0),
        "enabled": True,
    }


def sanitize_catalog_entries(entries: Any) -> list[dict[str, Any]]:
    """Filter and normalize a raw catalog sequence into valid enabled model entries."""
    if not isinstance(entries, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for entry in entries:
        cleaned = sanitize_catalog_entry(entry)
        if cleaned is not None:
            sanitized.append(cleaned)
    return sanitized

DEFAULT_CATALOG_MODELS: list[dict[str, Any]] = [
    {
        "id": "deepseek-r1-distill-llama-8b",
        "name": "DeepSeek-R1 Distill Llama 8B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "estimated_ram_gb": 5.2,
        "enabled": True,
    },
    {
        "id": "deepseek-r1-distill-qwen-14b",
        "name": "DeepSeek-R1 Distill Qwen 14B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "estimated_ram_gb": 9.2,
        "enabled": True,
    },
    {
        "id": "qwen-2.5-coder-7b",
        "name": "Qwen 2.5 Coder 7B",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "estimated_ram_gb": 4.9,
        "enabled": True,
    },
    {
        "id": "qwen-2.5-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "filename": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "estimated_ram_gb": 20.0,
        "enabled": True,
    },
    {
        "id": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "estimated_ram_gb": 5.1,
        "enabled": True,
    },
    {
        "id": "gemma-2-9b",
        "name": "Gemma 2 9B",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "estimated_ram_gb": 5.8,
        "enabled": True,
    },
]


def get_shardai_home() -> Path:
    """Return the per-user ShardAi data directory (`~/.shardai` or `SHARDAI_HOME_DIR`)."""
    override = os.getenv("SHARDAI_HOME_DIR", "").strip()
    base = Path(override).expanduser() if override else (Path.home() / ".shardai")
    return base.resolve()


def resolve_default_cache_dir(shardai_home: Path | None = None) -> Path:
    """Resolve a persistent, writable default directory for cached model weights.

    Precedence:
    1. Explicit `SHARDAI_CACHE_DIR` environment override.
    2. Frozen PyInstaller executables (`<shardai_home>/cache_models`) to avoid ephemeral `_MEI` dirs.
    3. Repository `core-engine/cache_models` when writable during local development/tests.
    4. Linux `$XDG_CACHE_HOME/shardai/cache_models` or `<shardai_home>/cache_models` fallback.
    """
    env_cache = os.getenv("SHARDAI_CACHE_DIR", "").strip()
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    resolved_home = shardai_home if shardai_home is not None else get_shardai_home()
    if getattr(sys, "frozen", False):
        return (resolved_home / "cache_models").resolve()

    source_cache = (Path(__file__).resolve().parent.parent / "cache_models").resolve()
    try:
        source_cache.mkdir(parents=True, exist_ok=True)
        if os.access(source_cache, os.W_OK):
            return source_cache
    except OSError:
        pass

    if platform.system() == "Linux":
        xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
        if xdg_cache:
            return (Path(xdg_cache).expanduser() / "shardai" / "cache_models").resolve()

    return (resolved_home / "cache_models").resolve()


def _feed_file_prefix(
    file_path: Path,
    max_bytes: int,
    update: Callable[[bytes], None],
) -> None:
    """Stream up to `max_bytes` from `file_path` into `update` using bounded chunks."""
    remaining = max(0, max_bytes)
    if remaining == 0:
        return
    with file_path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(remaining, FINGERPRINT_CHUNK_BYTES))
            if not chunk:
                break
            update(chunk)
            remaining -= len(chunk)


def compute_model_fingerprint(model_path: Path) -> str:
    """Compute a fast SHA-256 fingerprint from the first 10 MB + total size of a model artifact."""
    hasher = hashlib.sha256()
    if model_path.is_file():
        total_size = model_path.stat().st_size
        hasher.update(f"file:{total_size}:".encode("utf-8"))
        _feed_file_prefix(model_path, FINGERPRINT_HEAD_BYTES, hasher.update)
        return hasher.hexdigest()

    if model_path.is_dir():
        weight_files = sorted(
            p for p in model_path.iterdir() if p.is_file() and not p.name.startswith(".")
        )
        manifest = "|".join(f"{p.name}:{p.stat().st_size}" for p in weight_files)
        hasher.update(f"dir:{manifest}:".encode("utf-8"))
        for p in weight_files:
            if p.suffix.lower() in (".safetensors", ".gguf") or p.name == "config.json":
                _feed_file_prefix(
                    p,
                    min(FINGERPRINT_HEAD_BYTES, FINGERPRINT_CHUNK_BYTES),
                    hasher.update,
                )
                break
        return hasher.hexdigest()

    raise FileNotFoundError(f"Cannot compute model_fingerprint; path not found: {model_path}")


async def compute_model_fingerprint_async(model_path: Path) -> str:
    """Offload `compute_model_fingerprint` to a worker thread to avoid blocking the event loop."""
    return await asyncio.to_thread(compute_model_fingerprint, model_path)


def select_best_gguf_filename(repo_files: list[str]) -> str | None:
    """Pick the best `.gguf` file from a Hugging Face repo, prioritizing `Q4_K_M` quantization."""
    gguf_files = [
        f
        for f in repo_files
        if f.lower().endswith(".gguf") and not f.lower().endswith(".gguf.part")
    ]
    if not gguf_files:
        return None

    non_mmproj = [f for f in gguf_files if "mmproj" not in Path(f).name.lower()]
    pool = non_mmproj or gguf_files

    quant_ranks: tuple[str, ...] = (
        "q4_k_m",
        "q4_k_s",
        "q4_k",
        "q5_k_m",
        "q4_0",
        "q5_k_s",
        "q6_k",
        "q8_0",
    )

    def _score(filename: str) -> tuple[int, int, str]:
        lowered = filename.lower()
        split_penalty = 1 if re.search(r"-\d{5}-of-\d{5}\.gguf$", lowered) else 0
        quant_rank = len(quant_ranks)
        for idx, token in enumerate(quant_ranks):
            if token in lowered:
                quant_rank = idx
                break
        return (quant_rank, split_penalty, filename)

    return min(pool, key=_score)


def _list_hf_repo_files(repo_id: str) -> list[str]:
    """List files in a Hugging Face repo via `HfApi().list_repo_files(repo_id)`."""
    if HfApi is not None:
        return list(HfApi().list_repo_files(repo_id=repo_id))
    if list_repo_files is not None:
        return list(list_repo_files(repo_id=repo_id))
    raise RuntimeError("`huggingface_hub` is not installed.")


@dataclass(frozen=True, slots=True)
class DownloadProgressEvent:
    """Progress snapshot pushed to the React UI via Server-Sent Events (SSE)."""

    model_id: str
    status: str  # "checking" | "downloading" | "cached" | "completed" | "error"
    progress_percent: float
    downloaded_bytes: int = 0
    total_bytes: int = 0
    local_path: str | None = None
    message: str = ""


class _HfTqdmProgressBridge:
    """Custom `tqdm_class` hook passed to `hf_hub_download` to capture byte-level progress."""

    def __init__(
        self,
        *args: Any,  # Untyped args forwarded by huggingface_hub tqdm callback
        on_progress: Callable[[int, int], None] | None = None,
        **kwargs: Any,  # Untyped kwargs forwarded by huggingface_hub tqdm callback
    ) -> None:
        self.total: int = int(kwargs.get("total") or 0)
        self.n: int = int(kwargs.get("initial") or 0)
        self.transfer_n: int = self.n
        self._reported_n: int = self.n
        self.desc: str = str(kwargs.get("desc") or "")
        self.postfix: str = str(kwargs.get("postfix") or "")
        self.disable: bool = bool(kwargs.get("disable", False))
        self.pos: int = int(kwargs.get("position") or 0)
        self._failed: bool = False
        self._on_progress: Callable[[int, int], None] | None = on_progress
        if self._on_progress is not None and self.total > 0:
            self._emit_progress()

    @classmethod
    def with_callback(
        cls,
        on_progress: Callable[[int, int], None],
    ) -> type["_HfTqdmProgressBridge"]:
        """Return a `tqdm_class` subclass with `on_progress` bound at class level."""

        class _BoundHfTqdmProgressBridge(_HfTqdmProgressBridge):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("on_progress", on_progress)
                super().__init__(*args, **kwargs)

        return _BoundHfTqdmProgressBridge

    def _emit_progress(self) -> None:
        if self._on_progress is None:
            return
        total = int(self.total or 0)
        best_n = max(self.n, self.transfer_n)
        if total > 0:
            best_n = min(best_n, total)
        self._reported_n = max(self._reported_n, best_n)
        self._on_progress(self._reported_n, total)

    def update(self, n: int | float | None = 1) -> None:
        delta = int(n or 0)
        self.n = max(0, self.n + delta)
        if delta < 0:
            self._reported_n = min(self._reported_n, self.n)
        self._emit_progress()

    def update_transfer(self, n: int | float | None = 1) -> None:
        delta = int(n or 0)
        self.transfer_n = max(0, self.transfer_n + delta)
        if delta < 0:
            self._reported_n = min(self._reported_n, max(self.n, self.transfer_n))
        self._emit_progress()

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        self._emit_progress()

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        self.postfix = s
        if refresh:
            self.refresh()

    def set_transfer_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        self.postfix = s
        if refresh:
            self.refresh()

    def set_postfix(
        self,
        ordered_dict: Any = None,
        refresh: bool = True,
        **kwargs: Any,
    ) -> None:
        if ordered_dict is not None:
            self.postfix = str(ordered_dict)
        elif kwargs:
            self.postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        if refresh:
            self.refresh()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if desc is not None:
            self.desc = desc
        if refresh:
            self.refresh()

    def set_description_str(self, desc: str | None = None, refresh: bool = True) -> None:
        if desc is not None:
            self.desc = desc
        if refresh:
            self.refresh()

    def clear(self, *args: Any, **kwargs: Any) -> None:
        pass

    def display(self, *args: Any, **kwargs: Any) -> None:
        pass

    def reset(self, total: int | None = None) -> None:
        self.n = 0
        self.transfer_n = 0
        self._reported_n = 0
        if total is not None:
            self.total = int(total)

    @property
    def format_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "total": self.total,
            "rate": None,
            "elapsed": 0.0,
        }

    def close(self) -> None:
        total = int(self.total or 0)
        if not self._failed and self._on_progress is not None and total > 0:
            self._reported_n = total
            self._on_progress(total, total)

    def __enter__(self) -> "_HfTqdmProgressBridge":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._failed = True
        self.close()


def normalize_hf_repo_id(model_id: str) -> str:
    """Normalize a Hugging Face repo ID or full URL into `owner/repo` format."""
    cleaned: str = model_id.strip()
    if cleaned.startswith("https://huggingface.co/") or cleaned.startswith(
        "http://huggingface.co/"
    ):
        lowered = cleaned.lower()
        if "/resolve/" not in lowered and "/blob/" not in lowered and not lowered.endswith(".gguf"):
            parts = cleaned.split("huggingface.co/", 1)[1].strip("/").split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
    return cleaned


def filter_required_safetensors_files(
    weight_map: dict[str, str],
    shard_layer_start: int,
    shard_layer_end: int,
    total_layers: int,
    tie_word_embeddings: bool = False,
) -> list[str]:
    """Select only the `.safetensors` filenames required for `[shard_layer_start..shard_layer_end]`."""
    is_head: bool = shard_layer_start == 0
    is_tail: bool = shard_layer_end >= (total_layers - 1)
    layer_prefixes: tuple[str, ...] = tuple(
        f"model.layers.{idx}."
        for idx in range(shard_layer_start, shard_layer_end + 1)
    )

    selected_files: set[str] = set()
    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith(layer_prefixes):
            selected_files.add(shard_file)
        elif tensor_name.startswith("model.rotary_emb."):
            selected_files.add(shard_file)
        elif is_head and (
            tensor_name.startswith("model.embed_tokens.")
            or (tie_word_embeddings and tensor_name.startswith("lm_head."))
        ):
            selected_files.add(shard_file)
        elif is_tail and (
            tensor_name.startswith("model.norm.")
            or tensor_name.startswith("lm_head.")
            or (tie_word_embeddings and tensor_name.startswith("model.embed_tokens."))
        ):
            selected_files.add(shard_file)

    return sorted(selected_files)


def read_hf_config_summary(model_dir: Path) -> tuple[int, bool]:
    """Read `(num_hidden_layers, tie_word_embeddings)` from a cached `config.json`."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return 32, False
    data = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = int(data.get("num_hidden_layers") or 32)
    tie_embeddings = bool(data.get("tie_word_embeddings", False))
    return num_layers, tie_embeddings


class ModelDownloader:
    """Manages local `.safetensors` / `.gguf` cache resolution, multi-source downloads, and catalog cache."""

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        models_dir: Path | str | None = None,
        catalog_cache_path: Path | str | None = None,
        http_transport: httpx.AsyncBaseTransport | None = None,
        allowed_local_dirs: list[Path | str] | None = None,
    ) -> None:
        self.shardai_home: Path = get_shardai_home()
        default_dir = resolve_default_cache_dir(self.shardai_home)
        self.cache_dir: Path = Path(cache_dir).resolve() if cache_dir else default_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir: Path = (
            Path(models_dir).resolve()
            if models_dir
            else (self.shardai_home / "models").resolve()
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_cache_path: Path = (
            Path(catalog_cache_path).resolve()
            if catalog_cache_path
            else (self.shardai_home / "models_cache.json").resolve()
        )
        self.hf_shards_dir: Path = (self.cache_dir / "hf_shards").resolve()
        self.hf_shards_dir.mkdir(parents=True, exist_ok=True)
        self._http_transport: httpx.AsyncBaseTransport | None = http_transport
        self._extra_allowed_local_dirs: list[Path] = [
            Path(p).expanduser().resolve() for p in (allowed_local_dirs or [])
        ]
        self._subscribers: list[asyncio.Queue[DownloadProgressEvent]] = []
        self._latest_event: dict[str, DownloadProgressEvent] = {}

    def get_approved_local_roots(self) -> list[Path]:
        """Return canonical directories permitted for local model file/directory loading."""
        roots: list[Path] = [
            self.models_dir,
            self.cache_dir,
            *self._extra_allowed_local_dirs,
        ]
        env_dirs = os.getenv("SHARDAI_ALLOWED_MODEL_DIRS", "").strip()
        if env_dirs:
            for raw_dir in env_dirs.split(os.pathsep):
                cleaned = raw_dir.strip()
                if cleaned:
                    roots.append(Path(cleaned).expanduser().resolve())
        unique_roots: list[Path] = []
        for root in roots:
            if root not in unique_roots:
                unique_roots.append(root)
        return unique_roots

    def resolve_safe_local_model_path(self, raw_path: str) -> Path:
        """Validate canonical containment within approved roots and `.gguf`/HF directory format."""
        cleaned = raw_path.strip()
        if not cleaned or "\x00" in cleaned:
            raise ValueError("Local model path cannot be empty or contain NUL bytes.")

        resolved = Path(cleaned).expanduser().resolve()
        approved_roots = self.get_approved_local_roots()
        if not any(resolved.is_relative_to(root) for root in approved_roots):
            raise PermissionError(
                "Rejected local model path outside approved model directories."
            )

        if resolved.is_dir():
            if not (resolved / "config.json").is_file():
                raise FileNotFoundError(
                    f"Local model directory missing config.json: {resolved}"
                )
            return resolved

        if not resolved.is_file():
            raise FileNotFoundError(f"Local .gguf model file not found: {resolved}")
        if resolved.suffix.lower() != ".gguf":
            raise ValueError(f"Expected a .gguf model file, got: {resolved.name}")
        if not has_gguf_magic(resolved):
            raise ValueError(
                f"Local model file '{resolved.name}' is not a valid GGUF binary "
                "(missing 4-byte GGUF magic header)."
            )
        return resolved

    def save_catalog_cache(self, models: list[dict[str, Any]]) -> None:
        """Persist synchronized online model catalog to `~/.shardai/models_cache.json`."""
        self.catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time()),
            "models": models,
        }
        tmp_path = self.catalog_cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.catalog_cache_path)

    def load_catalog_cache(self) -> list[dict[str, Any]]:
        """Read cached model catalog from `~/.shardai/models_cache.json`, or empty list."""
        if not self.catalog_cache_path.is_file():
            return []
        try:
            raw = json.loads(self.catalog_cache_path.read_text(encoding="utf-8"))
            items = raw.get("models", []) if isinstance(raw, dict) else raw
            return sanitize_catalog_entries(items)
        except Exception as exc:
            logger.warning("Failed to read models cache %s: %s", self.catalog_cache_path, exc)
            return []

    def lookup_catalog_entry(self, model_id: str) -> dict[str, Any] | None:
        """Match a catalog `id` or `repo_id` from `~/.shardai/models_cache.json` or fallback defaults."""
        cleaned = model_id.strip()
        if not cleaned:
            return None
        normalized = normalize_hf_repo_id(cleaned)
        for entry in [*self.load_catalog_cache(), *DEFAULT_CATALOG_MODELS]:
            if entry.get("id") in (cleaned, normalized) or entry.get("repo_id") in (
                cleaned,
                normalized,
            ):
                return entry
        return None

    def subscribe(self) -> asyncio.Queue[DownloadProgressEvent]:
        """Register an SSE consumer queue to receive live download progress events."""
        queue: asyncio.Queue[DownloadProgressEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[DownloadProgressEvent]) -> None:
        """Remove an SSE consumer queue on disconnect."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _broadcast(self, event: DownloadProgressEvent) -> None:
        """Push a progress update to all active SSE subscribers."""
        self._latest_event[event.model_id] = event
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def resolve_safe_model_dir(self, model_id: str) -> tuple[str, Path]:
        """Validate and resolve a canonical directory for Hugging Face Safetensors shards."""
        cleaned_id: str = model_id.strip()
        if not cleaned_id:
            raise ValueError("model_id cannot be empty.")

        local_candidate = Path(cleaned_id)
        if local_candidate.is_dir() and (local_candidate / "config.json").is_file():
            resolved_local = self.resolve_safe_local_model_path(cleaned_id)
            return cleaned_id, resolved_local

        repo_id: str = normalize_hf_repo_id(cleaned_id)
        safe_slug: str = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id).strip("_")
        if not safe_slug:
            raise ValueError(f"Invalid model identifier: '{model_id}'")

        target_dir: Path = (self.hf_shards_dir / safe_slug).resolve()
        if not target_dir.is_relative_to(self.cache_dir):
            raise PermissionError("Rejected model directory outside designated cache directory.")
        return repo_id, target_dir

    def is_pytorch_model_cached(self, model_id: str) -> tuple[bool, Path]:
        """Check whether `config.json` and at least one `.safetensors` file exist locally."""
        _, model_dir = self.resolve_safe_model_dir(model_id)
        if not (model_dir / "config.json").is_file():
            return False, model_dir
        safetensor_files = list(model_dir.glob("*.safetensors"))
        has_weights = any(p.is_file() and p.stat().st_size > 1_048_576 for p in safetensor_files)
        return has_weights, model_dir

    def resolve_safe_gguf_path(self, model_id: str) -> tuple[GgufModelSpec, Path]:
        """Validate `model_id` (preset, custom HF repo/URL, or local `.gguf` path).

        Enforces Path Traversal protection and `.gguf` allowlist as required by `.antigravity/review.md`.
        """
        cleaned_id: str = model_id.strip()
        if not cleaned_id:
            raise ValueError("model_id cannot be empty.")

        # 1. Preset models
        spec = SUPPORTED_MODELS.get(cleaned_id)
        if spec is not None:
            candidate_path: Path = (self.cache_dir / spec.gguf_filename).resolve()
            if not candidate_path.is_relative_to(self.cache_dir):
                raise PermissionError("Rejected model path outside designated cache directory.")
            return spec, candidate_path

        # 2. Custom local `.gguf` file path
        local_candidate = Path(cleaned_id)
        if local_candidate.suffix.lower() == ".gguf" and (
            local_candidate.is_absolute()
            or local_candidate.expanduser().exists()
            or cleaned_id.startswith((".", "~"))
            or "\\" in cleaned_id
        ):
            resolved_local = self.resolve_safe_local_model_path(cleaned_id)
            custom_spec = GgufModelSpec(
                model_id=cleaned_id,
                hf_repo_id="local",
                gguf_filename=resolved_local.name,
                total_layers=32,
                estimated_total_size_gb=max(1.0, resolved_local.stat().st_size / float(1024**3)),
                base_overhead_gb=0.6,
            )
            return custom_spec, resolved_local

        # 3. Custom HuggingFace Repo ID or URL
        repo_id = normalize_hf_repo_id(cleaned_id)
        safe_slug: str = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_id).strip("_")
        if not safe_slug:
            raise ValueError(f"Invalid model identifier: '{model_id}'")

        custom_filename: str = f"{safe_slug}.gguf"
        candidate_path = (self.cache_dir / custom_filename).resolve()
        if not candidate_path.is_relative_to(self.cache_dir):
            raise PermissionError("Rejected model path outside designated cache directory.")

        custom_spec = GgufModelSpec(
            model_id=cleaned_id,
            hf_repo_id=repo_id,
            gguf_filename=custom_filename,
            total_layers=32,
            estimated_total_size_gb=5.0,
            base_overhead_gb=0.6,
        )
        return custom_spec, candidate_path

    def is_model_cached(self, model_id: str) -> tuple[bool, Path]:
        """Check whether a real `.gguf` (>= 1 MB or offline stub) with `GGUF` magic exists locally."""
        _, target_path = self.resolve_safe_gguf_path(model_id)
        min_bytes = len(GGUF_MAGIC) if is_offline_mode() else MIN_ONLINE_GGUF_BYTES
        return is_valid_gguf_file(target_path, min_bytes=min_bytes), target_path

    def _find_cached_pytorch_shard_files(
        self,
        model_dir: Path,
        shard_layer_start: int | None,
        shard_layer_end: int | None,
    ) -> tuple[list[Path], int, int]:
        """Return locally cached `.safetensors` files for the requested layer range, if complete."""
        if not (model_dir / "config.json").is_file():
            return [], 0, 0

        total_layers, tie_embeddings = read_hf_config_summary(model_dir)
        start_layer: int = max(0, shard_layer_start if shard_layer_start is not None else 0)
        end_layer: int = min(
            total_layers - 1,
            shard_layer_end if shard_layer_end is not None else (total_layers - 1),
        )

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                index_data = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map: dict[str, str] = index_data.get("weight_map", {})
                required_files = filter_required_safetensors_files(
                    weight_map=weight_map,
                    shard_layer_start=start_layer,
                    shard_layer_end=end_layer,
                    total_layers=total_layers,
                    tie_word_embeddings=tie_embeddings,
                )
                if not required_files:
                    required_files = sorted(set(weight_map.values()))
                resolved = [(model_dir / fname).resolve() for fname in required_files]
                cached = [
                    p
                    for p in resolved
                    if p.is_relative_to(model_dir) and p.is_file() and p.stat().st_size > 0
                ]
                if required_files and len(cached) == len(required_files):
                    return cached, start_layer, end_layer
            except Exception:
                return [], start_layer, end_layer
            return [], start_layer, end_layer

        cached_weights = sorted(
            p for p in model_dir.glob("*.safetensors") if p.is_file() and p.stat().st_size > 0
        )
        return cached_weights, start_layer, end_layer

    async def ensure_pytorch_shard_downloaded(
        self,
        model_id: str,
        shard_layer_start: int | None = None,
        shard_layer_end: int | None = None,
        on_event: Callable[[DownloadProgressEvent], None] | None = None,
    ) -> Path:
        """Download only the config, tokenizer, and `.safetensors` files needed for this shard."""
        repo_id, model_dir = self.resolve_safe_model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        def _emit(evt: DownloadProgressEvent) -> None:
            self._broadcast(evt)
            if on_event is not None:
                on_event(evt)

        _emit(
            DownloadProgressEvent(
                model_id=model_id,
                status="checking",
                progress_percent=0.0,
                message=f"Inspecting Hugging Face repository '{repo_id}'...",
            )
        )

        cached_files, start_layer, end_layer = self._find_cached_pytorch_shard_files(
            model_dir=model_dir,
            shard_layer_start=shard_layer_start,
            shard_layer_end=shard_layer_end,
        )
        if cached_files:
            total_bytes = sum(p.stat().st_size for p in cached_files)
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="cached",
                    progress_percent=100.0,
                    downloaded_bytes=total_bytes,
                    total_bytes=total_bytes,
                    local_path=str(model_dir),
                    message=(
                        f"Shard ready from cache for '{repo_id}' (layers {start_layer}..{end_layer}, "
                        f"{len(cached_files)} weight file(s))."
                    ),
                )
            )
            return model_dir

        if is_offline_mode() or hf_hub_download is None or list_repo_files is None:
            raise RuntimeError(
                "Online Hugging Face download is disabled (`SHARDAI_OFFLINE_MODE=true`) "
                "or `huggingface_hub` is not installed."
            )

        repo_files: list[str] = await asyncio.to_thread(list_repo_files, repo_id=repo_id)
        repo_file_set: set[str] = set(repo_files)

        if "config.json" not in repo_file_set:
            raise FileNotFoundError(
                f"Hugging Face repo '{repo_id}' does not contain config.json for PyTorch sharding."
            )

        loop = asyncio.get_running_loop()

        async def _download_file(filename: str, base_pct: float, span_pct: float) -> Path:
            local_file = (model_dir / filename).resolve()
            if not local_file.is_relative_to(model_dir):
                raise PermissionError(f"Rejected unsafe filename '{filename}'.")
            if local_file.is_file() and local_file.stat().st_size > 0:
                return local_file

            def _progress_hook(downloaded: int, total: int) -> None:
                ratio = (downloaded / total) if total > 0 else 0.0
                pct = min(99.9, round(base_pct + (ratio * span_pct), 1))
                evt = DownloadProgressEvent(
                    model_id=model_id,
                    status="downloading",
                    progress_percent=pct,
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                    message=f"Downloading {filename} ({pct:.1f}%)...",
                )
                loop.call_soon_threadsafe(_emit, evt)

            tqdm_bridge_cls = _HfTqdmProgressBridge.with_callback(_progress_hook)

            logger.info("Downloading %s/%s into %s", repo_id, filename, model_dir)
            downloaded_str: str = await asyncio.to_thread(
                hf_hub_download,
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                tqdm_class=tqdm_bridge_cls,
            )
            return Path(downloaded_str).resolve()

        # 1. Download config + tokenizer metadata
        for meta_file in _HF_ESSENTIAL_FILES:
            if meta_file in repo_file_set:
                await _download_file(meta_file, base_pct=2.0, span_pct=8.0)

        total_layers, tie_embeddings = read_hf_config_summary(model_dir)
        start_layer: int = max(0, shard_layer_start if shard_layer_start is not None else 0)
        end_layer: int = min(
            total_layers - 1,
            shard_layer_end if shard_layer_end is not None else (total_layers - 1),
        )

        # 2. Select either sharded `.safetensors` files for this layer range or single `model.safetensors`
        weight_files: list[str]
        if "model.safetensors.index.json" in repo_file_set:
            index_path = await _download_file(
                "model.safetensors.index.json", base_pct=10.0, span_pct=5.0
            )
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map: dict[str, str] = index_data.get("weight_map", {})
            weight_files = filter_required_safetensors_files(
                weight_map=weight_map,
                shard_layer_start=start_layer,
                shard_layer_end=end_layer,
                total_layers=total_layers,
                tie_word_embeddings=tie_embeddings,
            )
            if not weight_files:
                weight_files = sorted(set(weight_map.values()))
        elif "model.safetensors" in repo_file_set:
            weight_files = ["model.safetensors"]
        else:
            safetensors_candidates = sorted(
                f for f in repo_files if f.lower().endswith(".safetensors")
            )
            if not safetensors_candidates:
                raise FileNotFoundError(
                    f"No .safetensors weights found in Hugging Face repo '{repo_id}'."
                )
            weight_files = safetensors_candidates

        # 3. Download required weight files
        per_file_span = 80.0 / float(max(1, len(weight_files)))
        total_bytes = 0
        for idx, wf in enumerate(weight_files):
            downloaded_path = await _download_file(
                wf,
                base_pct=15.0 + (idx * per_file_span),
                span_pct=per_file_span,
            )
            if downloaded_path.is_file():
                total_bytes += downloaded_path.stat().st_size

        _emit(
            DownloadProgressEvent(
                model_id=model_id,
                status="completed",
                progress_percent=100.0,
                downloaded_bytes=total_bytes,
                total_bytes=total_bytes,
                local_path=str(model_dir),
                message=(
                    f"Shard ready for '{repo_id}' (layers {start_layer}..{end_layer}, "
                    f"{len(weight_files)} weight file(s))."
                ),
            )
        )
        return model_dir

    def resolve_direct_url_path(self, url: str, filename: str | None = None) -> Path:
        """Validate an HTTP/HTTPS `.gguf` URL and resolve a safe destination inside `~/.shardai/models/`."""
        validate_direct_url_syntax_and_host(url)
        parsed = urlparse(url.strip())

        raw_name = (filename or "").strip() or unquote(Path(parsed.path).name)
        safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw_name).name).strip("._")
        if not safe_base or not safe_base.lower().endswith(".gguf"):
            url_tag = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
            stem = safe_base[:-5] if safe_base.lower().endswith(".gguf") else (safe_base or "model")
            safe_base = f"{stem}_{url_tag}.gguf"

        target_path = (self.models_dir / safe_base).resolve()
        if not target_path.is_relative_to(self.models_dir):
            raise PermissionError("Rejected direct_url destination outside ~/.shardai/models.")
        return target_path

    async def _validate_direct_url_target(self, url: str) -> None:
        """Resolve `url` hostname and reject any private, loopback, link-local, or reserved IP."""
        host, port = validate_direct_url_syntax_and_host(url)
        try:
            ipaddress.ip_address(host.split("%", 1)[0])
            return
        except ValueError:
            pass

        try:
            resolved_ips = await asyncio.to_thread(_resolve_hostname_ips, host, port)
        except OSError as exc:
            if self._http_transport is not None and host.endswith(".test"):
                return
            raise PermissionError(
                f"Could not safely resolve direct_url host '{host}': {exc}"
            ) from exc

        if not resolved_ips:
            raise PermissionError(f"direct_url host '{host}' did not resolve to any IP address.")

        for raw_ip in resolved_ips:
            cleaned_ip = raw_ip.split("%", 1)[0]
            try:
                ip_obj = ipaddress.ip_address(cleaned_ip)
            except ValueError as exc:
                raise PermissionError(
                    f"Resolved direct_url IP '{raw_ip}' is invalid."
                ) from exc
            if is_forbidden_ssrf_ip(ip_obj):
                raise PermissionError(
                    f"Rejected direct_url host '{host}' resolving to restricted IP '{cleaned_ip}'."
                )

    def resolve_gguf_in_models_dir(self, repo_id: str, filename: str | None) -> Path:
        """Resolve a safe `.gguf` target inside `~/.shardai/models/` (or `cache_dir` fallback)."""
        effective_repo_id: str = repo_id.strip()
        effective_filename: str | None = (
            filename.strip() if filename and filename.strip() else None
        )
        if effective_filename is None:
            catalog_entry = self.lookup_catalog_entry(effective_repo_id)
            if catalog_entry is not None:
                entry_filename = catalog_entry.get("filename")
                if isinstance(entry_filename, str) and entry_filename.strip():
                    effective_filename = entry_filename.strip()
                entry_repo_id = catalog_entry.get("repo_id")
                if isinstance(entry_repo_id, str) and entry_repo_id.strip():
                    effective_repo_id = entry_repo_id.strip()

        if effective_filename:
            raw_parts = [
                part
                for part in effective_filename.replace("\\", "/").split("/")
                if part not in ("", ".", "..")
            ]
            safe_parts = [
                cleaned
                for cleaned in (
                    re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip(".") for part in raw_parts
                )
                if cleaned
            ]
            if safe_parts:
                in_models = (self.models_dir / Path(*safe_parts)).resolve()
                if in_models.is_relative_to(self.models_dir) and in_models.suffix.lower() == ".gguf":
                    return in_models

        _, legacy_path = self.resolve_safe_gguf_path(effective_repo_id)
        fallback_target = (self.models_dir / legacy_path.name).resolve()
        if not fallback_target.is_relative_to(self.models_dir):
            raise PermissionError("Rejected model destination outside ~/.shardai/models.")
        return fallback_target

    def _emit_helper(
        self,
        on_event: Callable[[DownloadProgressEvent], None] | None,
    ) -> Callable[[DownloadProgressEvent], None]:
        def _emit(evt: DownloadProgressEvent) -> None:
            self._broadcast(evt)
            if on_event is not None:
                on_event(evt)

        return _emit

    async def _simulate_offline_gguf(
        self,
        model_id: str,
        target_path: Path,
        _emit: Callable[[DownloadProgressEvent], None],
    ) -> Path:
        """Produce a lightweight deterministic GGUF stub when `SHARDAI_OFFLINE_MODE=true`."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        simulated_total = 4_680_000_000
        for step_pct in (15.0, 40.0, 70.0, 95.0):
            await asyncio.sleep(0.05)
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="downloading",
                    progress_percent=step_pct,
                    downloaded_bytes=int(simulated_total * (step_pct / 100.0)),
                    total_bytes=simulated_total,
                    message=f"Downloading {target_path.name} ({step_pct:.0f}%)...",
                )
            )
        target_path.write_bytes(b"GGUF\x03\x00\x00\x00" + model_id.encode("utf-8"))
        final_size = target_path.stat().st_size
        _emit(
            DownloadProgressEvent(
                model_id=model_id,
                status="completed",
                progress_percent=100.0,
                downloaded_bytes=final_size,
                total_bytes=final_size,
                local_path=str(target_path),
                message=f"Model '{model_id}' ready at {target_path.name}.",
            )
        )
        return target_path

    async def _ensure_direct_url_downloaded(
        self,
        url: str,
        filename: str | None,
        _emit: Callable[[DownloadProgressEvent], None],
    ) -> Path:
        """Stream a `.gguf` file from a validated public HTTP/HTTPS URL into `~/.shardai/models/`."""
        target_path = self.resolve_direct_url_path(url, filename=filename)
        await self._validate_direct_url_target(url)
        min_bytes = (
            len(GGUF_MAGIC)
            if (is_offline_mode() and self._http_transport is None)
            else MIN_ONLINE_GGUF_BYTES
        )

        _emit(
            DownloadProgressEvent(
                model_id=url,
                status="checking",
                progress_percent=0.0,
                message=f"Checking ~/.shardai/models for {target_path.name}...",
            )
        )

        if is_valid_gguf_file(target_path, min_bytes=min_bytes):
            size_bytes = target_path.stat().st_size
            _emit(
                DownloadProgressEvent(
                    model_id=url,
                    status="cached",
                    progress_percent=100.0,
                    downloaded_bytes=size_bytes,
                    total_bytes=size_bytes,
                    local_path=str(target_path),
                    message=f"Model ready from ~/.shardai/models/{target_path.name}.",
                )
            )
            return target_path

        if target_path.is_file():
            logger.warning(
                "Cached file '%s' failed GGUF validation (size=%d); re-downloading.",
                target_path.name,
                target_path.stat().st_size,
            )

        if is_offline_mode() and self._http_transport is None:
            return await self._simulate_offline_gguf(url, target_path, _emit)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = target_path.with_suffix(target_path.suffix + ".part")
        last_emit_ts = 0.0
        downloaded = 0
        total_bytes = 0

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(timeout=None, connect=15.0, read=60.0),
                transport=self._http_transport,
            ) as client:
                current_url = url.strip()
                redirects_followed = 0
                while True:
                    await self._validate_direct_url_target(current_url)
                    async with client.stream("GET", current_url) as response:
                        if response.status_code in _REDIRECT_STATUS_CODES:
                            location = response.headers.get("location", "").strip()
                            if not location:
                                raise ValueError(
                                    f"Redirect response (HTTP {response.status_code}) missing Location header."
                                )
                            redirects_followed += 1
                            if redirects_followed > MAX_DIRECT_URL_REDIRECTS:
                                raise PermissionError(
                                    f"Exceeded maximum direct_url redirects ({MAX_DIRECT_URL_REDIRECTS})."
                                )
                            next_url = urljoin(current_url, location)
                            await self._validate_direct_url_target(next_url)
                            current_url = next_url
                            continue

                        response.raise_for_status()
                        raw_content_length = response.headers.get("content-length")
                        if raw_content_length:
                            try:
                                parsed_len = int(raw_content_length)
                                if parsed_len > 0:
                                    total_bytes = parsed_len
                            except ValueError:
                                total_bytes = 0
                        with part_path.open("wb") as handle:
                            async for chunk in response.aiter_bytes(chunk_size=1_048_576):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                downloaded += len(chunk)
                                now = time.monotonic()
                                if now - last_emit_ts >= 0.15 or (total_bytes > 0 and downloaded >= total_bytes):
                                    last_emit_ts = now
                                    pct = (
                                        min(99.9, round((downloaded / float(total_bytes)) * 100.0, 1))
                                        if total_bytes > 0
                                        else 0.0
                                    )
                                    _emit(
                                        DownloadProgressEvent(
                                            model_id=url,
                                            status="downloading",
                                            progress_percent=pct,
                                            downloaded_bytes=downloaded,
                                            total_bytes=total_bytes,
                                            message=f"Downloading {target_path.name} ({pct:.1f}%)...",
                                        )
                                    )
                        break
            if total_bytes > 0 and downloaded != total_bytes:
                raise RuntimeError(
                    f"Incomplete GGUF download for '{target_path.name}': "
                    f"received {downloaded} of {total_bytes} bytes."
                )
            if not has_gguf_magic(part_path):
                raise ValueError(
                    f"Downloaded file '{target_path.name}' is not a valid GGUF binary "
                    "(missing 4-byte GGUF magic header)."
                )
            os.replace(part_path, target_path)
        except Exception:
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            raise

        final_size = target_path.stat().st_size
        _emit(
            DownloadProgressEvent(
                model_id=url,
                status="completed",
                progress_percent=100.0,
                downloaded_bytes=final_size,
                total_bytes=final_size,
                local_path=str(target_path),
                message=f"Downloaded {target_path.name} to ~/.shardai/models.",
            )
        )
        return target_path

    async def _ensure_hf_or_catalog_downloaded(
        self,
        model_id: str,
        filename: str | None,
        shard_layer_start: int | None,
        shard_layer_end: int | None,
        _emit: Callable[[DownloadProgressEvent], None],
    ) -> Path:
        """Download `.gguf` (via `HfApi().list_repo_files` + `hf_hub_download`) or Safetensors."""
        catalog_entry = self.lookup_catalog_entry(model_id)
        repo_id = normalize_hf_repo_id(
            str(catalog_entry["repo_id"]) if catalog_entry else model_id.strip()
        )
        resolved_filename = (
            filename.strip()
            if filename and filename.strip()
            else (
                str(catalog_entry["filename"]).strip()
                if catalog_entry and catalog_entry.get("filename")
                else None
            )
        )

        # In offline simulation mode, check both ~/.shardai/models and legacy cache_models
        if is_offline_mode():
            target_path = self.resolve_gguf_in_models_dir(repo_id, resolved_filename)
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="checking",
                    progress_percent=0.0,
                    message=f"Checking local cache for {target_path.name}...",
                )
            )
            if is_valid_gguf_file(target_path, min_bytes=len(GGUF_MAGIC)):
                size_bytes = target_path.stat().st_size
                _emit(
                    DownloadProgressEvent(
                        model_id=model_id,
                        status="cached",
                        progress_percent=100.0,
                        downloaded_bytes=size_bytes,
                        total_bytes=size_bytes,
                        local_path=str(target_path),
                        message=f"Model '{model_id}' ready from cache ({target_path.name}).",
                    )
                )
                return target_path
            if target_path.is_file():
                logger.warning(
                    "Offline cached GGUF '%s' failed validation (size=%d); regenerating stub.",
                    target_path.name,
                    target_path.stat().st_size,
                )
                target_path.unlink(missing_ok=True)
            cached, existing_path = self.is_model_cached(model_id)
            if cached:
                size_bytes = existing_path.stat().st_size
                _emit(
                    DownloadProgressEvent(
                        model_id=model_id,
                        status="cached",
                        progress_percent=100.0,
                        downloaded_bytes=size_bytes,
                        total_bytes=size_bytes,
                        local_path=str(existing_path),
                        message=f"Model '{model_id}' ready from cache ({existing_path.name}).",
                    )
                )
                return existing_path
            return await self._simulate_offline_gguf(model_id, target_path, _emit)

        if hf_hub_download is None:
            raise RuntimeError("`huggingface_hub` is not installed.")

        # Auto-discover best .gguf (Q4_K_M priority) if caller only gave repo_id
        if resolved_filename is None:
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="checking",
                    progress_percent=0.0,
                    message=f"Listing files in '{repo_id}' via HfApi...",
                )
            )
            repo_files = await asyncio.to_thread(_list_hf_repo_files, repo_id)
            best_gguf = select_best_gguf_filename(repo_files)
            if best_gguf is not None:
                resolved_filename = best_gguf
            elif "config.json" in repo_files:
                return await self.ensure_pytorch_shard_downloaded(
                    model_id=repo_id,
                    shard_layer_start=shard_layer_start,
                    shard_layer_end=shard_layer_end,
                    on_event=_emit,
                )
            else:
                raise FileNotFoundError(
                    f"No .gguf or Safetensors weights found in Hugging Face repo '{repo_id}'."
                )

        assert resolved_filename is not None
        target_path = self.resolve_gguf_in_models_dir(repo_id, resolved_filename)
        _emit(
            DownloadProgressEvent(
                model_id=model_id,
                status="checking",
                progress_percent=0.0,
                message=f"Checking ~/.shardai/models for {target_path.name}...",
            )
        )
        if is_valid_gguf_file(target_path, min_bytes=MIN_ONLINE_GGUF_BYTES):
            size_bytes = target_path.stat().st_size
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="cached",
                    progress_percent=100.0,
                    downloaded_bytes=size_bytes,
                    total_bytes=size_bytes,
                    local_path=str(target_path),
                    message=f"Model '{resolved_filename}' ready from ~/.shardai/models.",
                )
            )
            return target_path

        if target_path.is_file():
            logger.warning(
                "Cached GGUF file '%s' failed validation (size=%d); removing and re-downloading.",
                target_path.name,
                target_path.stat().st_size,
            )
            target_path.unlink(missing_ok=True)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()

        def _progress_hook(downloaded: int, total: int) -> None:
            pct = min(99.9, round((downloaded / float(total)) * 100.0, 1)) if total > 0 else 0.0
            evt = DownloadProgressEvent(
                model_id=model_id,
                status="downloading",
                progress_percent=pct,
                downloaded_bytes=downloaded,
                total_bytes=total,
                message=f"Downloading {resolved_filename} ({pct:.1f}%)...",
            )
            loop.call_soon_threadsafe(_emit, evt)

        tqdm_bridge_cls = _HfTqdmProgressBridge.with_callback(_progress_hook)
        downloaded_str: str = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename=resolved_filename,
            local_dir=str(self.models_dir),
            tqdm_class=tqdm_bridge_cls,
        )
        final_path = Path(downloaded_str).resolve()
        if not has_gguf_magic(final_path):
            if final_path.is_file() and final_path.is_relative_to(self.models_dir):
                final_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file '{final_path.name}' is not a valid GGUF binary "
                "(missing 4-byte GGUF magic header)."
            )
        final_size = final_path.stat().st_size if final_path.is_file() else 0
        _emit(
            DownloadProgressEvent(
                model_id=model_id,
                status="completed",
                progress_percent=100.0,
                downloaded_bytes=final_size,
                total_bytes=final_size,
                local_path=str(final_path),
                message=f"Downloaded {final_path.name} into ~/.shardai/models.",
            )
        )
        return final_path

    async def ensure_model_downloaded(
        self,
        model_id: str,
        on_event: Callable[[DownloadProgressEvent], None] | None = None,
        shard_layer_start: int | None = None,
        shard_layer_end: int | None = None,
        source_type: str = "catalog",
        filename: str | None = None,
    ) -> Path:
        """Prepare any model source (`catalog`, `huggingface`, `direct_url`, or `local_file`)."""
        cleaned_id: str = model_id.strip()
        if not cleaned_id:
            raise ValueError("model_id cannot be empty.")

        _emit = self._emit_helper(on_event)
        lowered = cleaned_id.lower()
        effective_source = (source_type or "catalog").strip().lower()

        # Infer direct_url or local_file when caller uses default source_type with a URL / .gguf path
        if effective_source in ("catalog", "huggingface"):
            if (lowered.startswith("http://") or lowered.startswith("https://")) and (
                "huggingface.co/" not in lowered
                or "/resolve/" in lowered
                or lowered.endswith(".gguf")
            ):
                effective_source = "direct_url"
            elif Path(cleaned_id).suffix.lower() == ".gguf" and (
                Path(cleaned_id).is_absolute()
                or Path(cleaned_id).expanduser().exists()
                or cleaned_id.startswith((".", "~"))
                or "\\" in cleaned_id
            ):
                effective_source = "local_file"

        if effective_source == "local_file":
            local_path = self.resolve_safe_local_model_path(cleaned_id)
            if local_path.is_dir():
                _emit(
                    DownloadProgressEvent(
                        model_id=model_id,
                        status="cached",
                        progress_percent=100.0,
                        local_path=str(local_path),
                        message=f"Local model directory ready ({local_path.name}).",
                    )
                )
                return local_path
            size_bytes = local_path.stat().st_size
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="cached",
                    progress_percent=100.0,
                    downloaded_bytes=size_bytes,
                    total_bytes=size_bytes,
                    local_path=str(local_path),
                    message=f"Local GGUF ready ({local_path.name}).",
                )
            )
            return local_path

        if effective_source == "direct_url":
            return await self._ensure_direct_url_downloaded(
                url=cleaned_id,
                filename=filename,
                _emit=_emit,
            )

        # Support already-downloaded local HF Safetensors directories inside approved roots
        local_candidate = Path(cleaned_id)
        if local_candidate.is_dir() and (local_candidate / "config.json").is_file():
            resolved_dir = self.resolve_safe_local_model_path(cleaned_id)
            _emit(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="cached",
                    progress_percent=100.0,
                    local_path=str(resolved_dir),
                    message=f"Local model directory ready ({resolved_dir.name}).",
                )
            )
            return resolved_dir

        return await self._ensure_hf_or_catalog_downloaded(
            model_id=cleaned_id,
            filename=filename,
            shard_layer_start=shard_layer_start,
            shard_layer_end=shard_layer_end,
            _emit=_emit,
        )
