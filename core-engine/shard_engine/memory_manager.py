"""RAM/VRAM Lifecycle Manager for LLM Shards (PyTorch & llama.cpp).

Complies with `.antigravity/review.md`:
- Prevents RAM/VRAM memory leaks when stopping the node or switching shards.
- Explicitly releases model handles, triggers Python GC, and clears GPU caches.
"""

import gc
import logging
from typing import Any

from shard_engine.schemas import ComputeBackend, StartNodeRequest

logger = logging.getLogger("shardai.memory")


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

    def allocate_shard(self, config: StartNodeRequest) -> None:
        """Load the assigned model shard layers within user-permitted RAM/VRAM limits."""
        if self._active_shard_handle is not None:
            self.release_all()

        self._current_backend = config.backend
        # Placeholder for actual llama.cpp / PyTorch shard initialization
        self._active_shard_handle = {
            "model_id": config.model_id,
            "backend": config.backend.value,
            "layers": (config.shard_layer_start, config.shard_layer_end),
        }
        self._allocated_ram_mb = min(config.max_ram_mb, 1024)
        self._allocated_vram_mb = min(config.max_vram_mb, 1024)
        logger.info(
            "Allocated shard for %s [%s] layers %d..%d (RAM=%dMB, VRAM=%dMB)",
            config.model_id,
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
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except ImportError:
            pass

        logger.info("Shard memory released and GPU/CPU caches cleared.")
