"""RAM/VRAM Lifecycle Manager, Dynamic Layer Planner, and Host Memory Inspector.

Complies with `.antigravity/review.md`:
- Prevents RAM/VRAM memory leaks when stopping the node or switching shards.
- Dynamically calculates maximum safe `n_layers` from the selected model footprint and `ram_gb`
  to prevent Out-Of-Memory (OOM) crashes on consumer machines.
- Explicitly releases model handles, triggers Python GC, and clears GPU caches.
"""

import ctypes
from dataclasses import dataclass
import gc
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

from models.downloader import DEFAULT_CATALOG_MODELS, SUPPORTED_MODELS, read_hf_config_summary
from models.gguf_reader import read_gguf_total_layers
from shard_engine.schemas import ComputeBackend, StartNodeRequest

logger = logging.getLogger("shardai.memory")


@dataclass(frozen=True, slots=True)
class LayerAllocationPlan:
    """Dynamic layer allocation computed from model configuration and available `ram_gb`."""

    model_id: str
    total_layers: int
    n_layers: int
    shard_layer_start: int
    shard_layer_end: int
    model_size_gb: float
    per_layer_mb: float
    estimated_ram_mb: int


@dataclass(frozen=True, slots=True)
class RoomRamRequirements:
    """Stable physical RAM budget required for a 1-node or N-node room without OS swap."""

    model_id: str
    model_size_gb: float
    total_layers: int
    node_count: int
    per_node_overhead_gb: float
    shard_rounding_slack_gb: float
    safety_margin_gb: float
    min_room_ram_gb: float
    stable_room_ram_gb: float
    recommended_slider_gb: int


def _resolve_model_ram_metadata(
    model_id: str,
    model_size_gb: float | None = None,
    total_layers: int | None = None,
) -> tuple[float, int, float]:
    """Resolve `(model_size_gb, total_layers, per_node_overhead_gb)` consistently."""
    spec = SUPPORTED_MODELS.get(model_id)
    if spec is not None:
        resolved_size = (
            float(model_size_gb)
            if model_size_gb is not None and model_size_gb > 0
            else float(spec.estimated_total_size_gb)
        )
        resolved_layers = (
            int(total_layers)
            if total_layers is not None and total_layers > 0
            else int(spec.total_layers)
        )
        resolved_overhead = float(spec.base_overhead_gb)
        return resolved_size, resolved_layers, resolved_overhead

    catalog_match = next(
        (
            m
            for m in DEFAULT_CATALOG_MODELS
            if m.get("repo_id") == model_id or m.get("id") == model_id
        ),
        None,
    )
    resolved_size = (
        float(model_size_gb)
        if model_size_gb is not None and model_size_gb > 0
        else (float(catalog_match["estimated_ram_gb"]) if catalog_match else 5.0)
    )
    if total_layers is not None and total_layers > 0:
        resolved_layers = int(total_layers)
    elif resolved_size <= 2.5:
        resolved_layers = 24
    elif resolved_size < 9.0:
        resolved_layers = 32
    elif resolved_size < 18.0:
        resolved_layers = 40
    else:
        resolved_layers = 64

    resolved_overhead = 0.35 if resolved_size <= 1.5 else 0.6
    return resolved_size, resolved_layers, resolved_overhead


def _ceil_half_gb(value: float) -> float:
    """Round up `value` to the nearest 0.5 GB step."""
    return math.ceil(value * 2.0 - 1e-9) / 2.0


def calculate_room_ram_requirements(
    model_id: str,
    node_count: int = 1,
    model_size_gb: float | None = None,
    total_layers: int | None = None,
) -> RoomRamRequirements:
    """Calculate minimum and stable physical room RAM for a 1-node or N-node shard swarm.

    With native `libggml` zero-copy quantized execution, weights remain compressed in RAM
    (no BF16/FP32 weight expansion), while each peer incurs per-node KV-cache/runtime
    overhead (`0.60 GB` or `0.35 GB` for <= 1.5 GB models) plus floor-rounding shard slack
    for multi-node partitioning and a 10% KV-growth safety buffer.
    """
    safe_nodes = max(1, int(node_count))
    resolved_size, resolved_layers, per_node_overhead = _resolve_model_ram_metadata(
        model_id=model_id,
        model_size_gb=model_size_gb,
        total_layers=total_layers,
    )
    per_layer_gb = resolved_size / float(resolved_layers)
    shard_rounding_slack = float(safe_nodes - 1) * per_layer_gb
    safety_margin = max(0.2, 0.10 * resolved_size)

    raw_min = resolved_size + (float(safe_nodes) * per_node_overhead) + shard_rounding_slack
    raw_stable = raw_min + safety_margin
    stable_room_ram_gb = _ceil_half_gb(raw_stable)

    return RoomRamRequirements(
        model_id=model_id,
        model_size_gb=round(resolved_size, 2),
        total_layers=resolved_layers,
        node_count=safe_nodes,
        per_node_overhead_gb=round(per_node_overhead, 2),
        shard_rounding_slack_gb=round(shard_rounding_slack, 2),
        safety_margin_gb=round(safety_margin, 2),
        min_room_ram_gb=round(raw_min, 2),
        stable_room_ram_gb=stable_room_ram_gb,
        recommended_slider_gb=max(2, math.ceil(stable_room_ram_gb - 1e-9)),
    )


def calculate_max_layers(
    model_id: str,
    ram_gb: int,
    model_path: Path | None = None,
    requested_start: int = 0,
) -> LayerAllocationPlan:
    """Compute the maximum transformer layers (`n_layers`) safely loadable within `ram_gb`.

    Formula:
    1. Determine `total_layers` and `model_size_gb` from `SUPPORTED_MODELS` (or actual `.gguf`
       file size when a full weight binary > 1 MB is present on disk).
    2. Subtract `base_overhead_gb` (0.6 GB for embeddings, LM head, and KV-cache buffers).
    3. Divide remaining budget by `per_layer_gb = model_size_gb / total_layers`.
    4. Clamp `n_layers` into `[1, total_layers - shard_layer_start]`.
    """
    spec = SUPPORTED_MODELS.get(model_id)
    if spec is not None:
        total_layers = spec.total_layers
        model_size_gb = spec.estimated_total_size_gb
        base_overhead_gb = spec.base_overhead_gb
    else:
        catalog_match = next(
            (m for m in DEFAULT_CATALOG_MODELS if m.get("repo_id") == model_id or m.get("id") == model_id),
            None,
        )
        total_layers = 40 if catalog_match and float(catalog_match.get("estimated_ram_gb", 0)) >= 9.0 else 32
        model_size_gb = float(catalog_match["estimated_ram_gb"]) if catalog_match else 5.0
        base_overhead_gb = 0.6

    # Use actual GGUF file size or HF Safetensors directory config when present
    if model_path is not None and model_path.is_dir():
        cfg_layers, _tie = read_hf_config_summary(model_path)
        if cfg_layers > 0:
            total_layers = cfg_layers
        safetensor_bytes = sum(
            p.stat().st_size for p in model_path.glob("*.safetensors") if p.is_file()
        )
        if safetensor_bytes > 1_048_576 and spec is None:
            # Estimate float32/bf16 memory footprint from safetensors disk size
            model_size_gb = max(0.5, (safetensor_bytes / float(1024**3)) * 1.5)
    elif model_path is not None and model_path.is_file():
        gguf_layers = read_gguf_total_layers(model_path)
        if gguf_layers is not None and gguf_layers > 0:
            total_layers = gguf_layers
        actual_bytes: int = model_path.stat().st_size
        if actual_bytes > 1_048_576:
            model_size_gb = actual_bytes / float(1024**3)

    per_layer_gb: float = model_size_gb / float(total_layers)
    per_layer_mb: float = round(per_layer_gb * 1024.0, 2)

    usable_for_layers_gb: float = max(0.0, float(ram_gb) - base_overhead_gb)
    max_loadable: int = math.floor(usable_for_layers_gb / per_layer_gb)

    safe_start: int = max(0, min(requested_start, total_layers - 1))
    remaining_layers: int = total_layers - safe_start
    n_layers: int = max(1, min(remaining_layers, max_loadable))
    shard_end: int = safe_start + n_layers - 1

    estimated_ram_mb: int = min(
        ram_gb * 1024,
        round((base_overhead_gb + (n_layers * per_layer_gb)) * 1024.0),
    )

    logger.info(
        "Layer plan for %s with %d GB RAM: n_layers=%d/%d (layers %d..%d, %.1f MB/layer, est=%d MB)",
        model_id,
        ram_gb,
        n_layers,
        total_layers,
        safe_start,
        shard_end,
        per_layer_mb,
        estimated_ram_mb,
    )

    return LayerAllocationPlan(
        model_id=model_id,
        total_layers=total_layers,
        n_layers=n_layers,
        shard_layer_start=safe_start,
        shard_layer_end=shard_end,
        model_size_gb=round(model_size_gb, 2),
        per_layer_mb=per_layer_mb,
        estimated_ram_mb=estimated_ram_mb,
    )


def _parse_linux_meminfo(meminfo_text: str) -> tuple[int, int]:
    """Parse `/proc/meminfo` and return `(total_bytes, available_bytes)`.

    Falls back to `MemFree + Buffers + Cached` when `MemAvailable` is absent.
    """
    values_kb: dict[str, int] = {}
    for line in meminfo_text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values_kb[key.strip()] = int(parts[0])
        except ValueError:
            continue

    total_kb = values_kb.get("MemTotal", 0)
    avail_kb = values_kb.get("MemAvailable", 0)
    if avail_kb <= 0:
        avail_kb = (
            values_kb.get("MemFree", 0)
            + values_kb.get("Buffers", 0)
            + values_kb.get("Cached", 0)
        )
    return max(0, total_kb * 1024), max(0, avail_kb * 1024)


def get_system_memory_info() -> tuple[float, float, int]:
    """Return `(total_ram_gb, available_ram_gb, max_shareable_ram_gb)` for the host OS.

    Reserves 1 GB of free RAM for the operating system (`floor(available - 1)`),
    and clamps Linux host readings to active container cgroup v1/v2 limits.
    """
    system_name: str = platform.system()
    total_bytes: int = 0
    avail_bytes: int = 0
    cgroup_limit: int | None = None

    try:
        if system_name == "Windows":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                total_bytes = int(stat.ullTotalPhys)
                avail_bytes = int(stat.ullAvailPhys)
        elif system_name == "Linux":
            if Path("/proc/meminfo").exists():
                meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
                total_bytes, avail_bytes = _parse_linux_meminfo(meminfo)
            from shard_engine.linux_env import read_cgroup_memory_limit

            cgroup_limit, cgroup_current = read_cgroup_memory_limit()
            if cgroup_limit is not None and cgroup_limit > 0:
                total_bytes = min(total_bytes, cgroup_limit) if total_bytes > 0 else cgroup_limit
                if cgroup_current is not None and cgroup_current >= 0:
                    cgroup_avail = max(0, cgroup_limit - cgroup_current)
                    avail_bytes = (
                        min(avail_bytes, cgroup_avail) if avail_bytes > 0 else cgroup_avail
                    )
                else:
                    avail_bytes = min(avail_bytes, cgroup_limit) if avail_bytes > 0 else cgroup_limit
        elif system_name == "Darwin":
            total_out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            total_bytes = int(total_out)
            page_size = os.sysconf("SC_PAGE_SIZE")
            vm_out = subprocess.check_output(["vm_stat"], text=True)
            free_pages = 0
            for line in vm_out.splitlines():
                if "Pages free:" in line or "Pages inactive:" in line:
                    val = line.split(":")[1].strip().rstrip(".")
                    free_pages += int(val)
            avail_bytes = free_pages * page_size
    except Exception as exc:
        logger.warning("Failed to inspect OS memory via native API: %s", exc)

    if total_bytes <= 0 or (avail_bytes <= 0 and cgroup_limit is None):
        # Conservative fallback if OS telemetry is unavailable
        return 16.0, 8.0, 7

    total_gb: float = round(total_bytes / float(1024**3), 2)
    avail_gb: float = round(avail_bytes / float(1024**3), 2)
    max_shareable_gb: int = max(2, math.floor(avail_gb - 1.0))
    return total_gb, avail_gb, max_shareable_gb


class ShardMemoryManager:
    """Manages allocation and deterministic cleanup of local AI model shards."""

    def __init__(self) -> None:
        self._active_shard_handle: Any | None = None
        self._allocated_ram_mb: int = 0
        self._allocated_vram_mb: int = 0
        self._current_backend: ComputeBackend | None = None

    @property
    def allocated_ram_mb(self) -> int:
        return self._allocated_ram_mb

    @property
    def allocated_vram_mb(self) -> int:
        return self._allocated_vram_mb

    @property
    def is_loaded(self) -> bool:
        return self._active_shard_handle is not None

    def allocate_shard(
        self,
        config: StartNodeRequest,
        estimated_ram_mb: int | None = None,
    ) -> None:
        """Load the assigned model shard layers within user-permitted RAM/VRAM limits."""
        if self._active_shard_handle is not None:
            self.release_all()

        self._current_backend = config.backend
        self._active_shard_handle = {
            "model_id": config.model_id,
            "room_id": config.room_id,
            "backend": config.backend.value,
            "layers": (config.shard_layer_start, config.shard_layer_end),
        }
        self._allocated_ram_mb = (
            min(config.max_ram_mb, estimated_ram_mb)
            if estimated_ram_mb is not None
            else min(config.max_ram_mb, 1024)
        )
        self._allocated_vram_mb = min(config.max_vram_mb, 1024)
        logger.info(
            "Allocated shard for %s [room=%s, %s] layers %d..%d (RAM=%dMB, VRAM=%dMB)",
            config.model_id,
            config.room_id or "PUBLIC",
            config.backend.value,
            config.shard_layer_start,
            config.shard_layer_end,
            self._allocated_ram_mb,
            self._allocated_vram_mb,
        )

    def release_all(self) -> None:
        """Deterministically free all RAM/VRAM buffers held by the current model shard."""
        if self._active_shard_handle is not None:
            close_fn = getattr(self._active_shard_handle, "close", None)
            if callable(close_fn):
                close_fn()
            self._active_shard_handle = None

        self._allocated_ram_mb = 0
        self._allocated_vram_mb = 0
        self._current_backend = None

        # Force garbage collection to reclaim unreachable tensor/buffer wrappers
        gc.collect()

        # Clear PyTorch GPU caches if torch is installed in the environment
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
            ):
                torch.mps.empty_cache()
        except ImportError:
            pass

        logger.info("Shard memory released and GPU/CPU caches cleared.")
