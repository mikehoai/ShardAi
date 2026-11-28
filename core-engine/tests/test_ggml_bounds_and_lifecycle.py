"""Regression tests for Issue #70: GGML weight bounds validation, GPU RAM pinning, and lifecycle cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import models.ggml_native as ggml_native_mod
from models.ggml_native import (
    GGML_NATIVE_AVAILABLE,
    GgmlNativeEngine,
)
import models.gguf_shard_loader as shard_loader_mod
from models.gguf_shard_loader import (
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_K,
    load_gguf_pytorch_shard,
)
from tests.test_gguf_sharded_pipeline import _build_tiny_qwen2_gguf


def _make_mock_api(
    *,
    best_backend: int = 0x1001,
    best_dev: int = 0x2001,
    best_dev_type: int = 1,
    best_name: bytes = b"CUDA0",
    cpu_backend: int = 0x3001,
) -> SimpleNamespace:
    """Create a deterministic mock `_GgmlCApi` without loading real C libraries."""
    lib_base = MagicMock()
    lib_base.ggml_backend_get_device.return_value = best_dev
    lib_base.ggml_backend_dev_type.return_value = best_dev_type
    lib_base.ggml_backend_dev_name.return_value = best_name
    lib_base.ggml_backend_name.return_value = best_name
    lib_base.ggml_backend_get_default_buffer_type.return_value = 0x4001
    lib_base.ggml_blck_size.side_effect = lambda t: {0: 1, 2: 32, 4: 0, 12: 256}.get(int(t), 0)
    lib_base.ggml_type_size.side_effect = lambda t: {0: 4, 2: 18, 4: 0, 12: 144}.get(int(t), 0)
    lib_base.ggml_init.return_value = 0x5001
    lib_base.ggml_new_tensor_2d.return_value = 0x6001
    lib_base.ggml_backend_cpu_buffer_from_ptr.return_value = 0x7001
    lib_base.ggml_backend_tensor_alloc.return_value = 0
    lib_base.ggml_backend_alloc_ctx_tensors.return_value = 0x8001
    lib_base.ggml_gallocr_new.return_value = 0x9001
    lib_base.ggml_mul_mat.return_value = 0xA001
    lib_base.ggml_new_graph.return_value = 0xB001
    lib_base.ggml_gallocr_alloc_graph.return_value = True

    lib_cpu = MagicMock()
    lib_cpu.ggml_backend_cpu_init.return_value = cpu_backend

    lib_ggml = MagicMock()
    lib_ggml.ggml_backend_dev_count.return_value = 1
    lib_ggml.ggml_backend_dev_get.return_value = best_dev
    lib_ggml.ggml_backend_init_best.return_value = best_backend

    return SimpleNamespace(
        lib_base=lib_base,
        lib_cpu=lib_cpu,
        lib_ggml=lib_ggml,
        tensor_overhead=128,
        graph_overhead=512,
    )


def test_register_weight_rejects_invalid_inputs_before_c_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid dimensions, types, alignment, and byte lengths fail before any C alloc."""
    mock_api = _make_mock_api()
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: mock_api)
    engine = GgmlNativeEngine(preferred="cpu")
    try:
        valid_q4_0 = torch.zeros(18, dtype=torch.uint8)

        # Invalid dimensions
        with pytest.raises(ValueError, match="positive weight dimensions"):
            engine.register_weight(valid_q4_0, out_features=0, in_features=32, ggml_type=GGML_TYPE_Q4_0)
        with pytest.raises(ValueError, match="positive weight dimensions"):
            engine.register_weight(valid_q4_0, out_features=-2, in_features=32, ggml_type=GGML_TYPE_Q4_0)
        with pytest.raises(ValueError, match="positive weight dimensions"):
            engine.register_weight(valid_q4_0, out_features=1, in_features=0, ggml_type=GGML_TYPE_Q4_0)

        # Invalid / removed ggml_type IDs (negative, out-of-range, removed=4)
        with pytest.raises(ValueError, match="ggml_type"):
            engine.register_weight(valid_q4_0, out_features=1, in_features=32, ggml_type=-1)
        with pytest.raises(ValueError, match="ggml_type"):
            engine.register_weight(valid_q4_0, out_features=1, in_features=32, ggml_type=999)
        with pytest.raises(ValueError, match="Unsupported ggml_type=4"):
            engine.register_weight(valid_q4_0, out_features=1, in_features=32, ggml_type=4)

        # Misaligned in_features (Q4_0 block size is 32)
        with pytest.raises(ValueError, match="divisible by blck_size=32"):
            engine.register_weight(valid_q4_0, out_features=1, in_features=48, ggml_type=GGML_TYPE_Q4_0)

        # Truncated and oversized buffers
        with pytest.raises(ValueError, match="byte length"):
            engine.register_weight(
                torch.zeros(17, dtype=torch.uint8),
                out_features=1,
                in_features=32,
                ggml_type=GGML_TYPE_Q4_0,
            )
        with pytest.raises(ValueError, match="byte length"):
            engine.register_weight(
                torch.zeros(36, dtype=torch.uint8),
                out_features=1,
                in_features=32,
                ggml_type=GGML_TYPE_Q4_0,
            )

        assert mock_api.lib_base.ggml_init.call_count == 0
        assert mock_api.lib_base.ggml_new_tensor_2d.call_count == 0
        assert mock_api.lib_base.ggml_backend_cpu_buffer_from_ptr.call_count == 0
        assert mock_api.lib_base.ggml_backend_alloc_ctx_tensors.call_count == 0
    finally:
        engine.close()


def test_matmul_rejects_zero_rows_before_compute_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """matmul() rejects 0-row inputs with ValueError before allocating a GGML graph."""
    mock_api = _make_mock_api()
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: mock_api)
    engine = GgmlNativeEngine(preferred="cpu")
    try:
        handle = engine.register_weight(
            torch.zeros(18, dtype=torch.uint8),
            out_features=1,
            in_features=32,
            ggml_type=GGML_TYPE_Q4_0,
        )
        mock_api.lib_base.ggml_init.reset_mock()

        empty_x = np.empty((0, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="non-empty 2D input"):
            engine.matmul(handle, empty_x)

        assert mock_api.lib_base.ggml_gallocr_new.call_count == 0
        assert mock_api.lib_base.ggml_init.call_count == 0
    finally:
        engine.close()


def test_gpu_vs_cpu_weight_ref_pinning_and_best_backend_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU backend does not pin raw_cpu host RAM; CPU fallback from init_best configures CPU threads."""
    # 1. True GPU backend -> is_cpu=False, _weight_refs stays empty
    gpu_api = _make_mock_api(best_dev_type=1, best_name=b"CUDA0")
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: gpu_api)
    gpu_engine = GgmlNativeEngine(preferred="gpu")
    try:
        assert gpu_engine.is_cpu is False
        assert gpu_engine.backend_kind == "gpu(CUDA0)"
        raw = torch.zeros(144, dtype=torch.uint8)
        gpu_engine.register_weight(raw, out_features=1, in_features=256, ggml_type=GGML_TYPE_Q4_K)
        assert len(gpu_engine._handles) == 1
        assert len(gpu_engine._weight_refs) == 0
        assert gpu_api.lib_base.ggml_backend_alloc_ctx_tensors.call_count == 1
        assert gpu_api.lib_base.ggml_backend_tensor_set.call_count == 1
    finally:
        gpu_engine.close()

    # 2. CPU backend -> is_cpu=True, _weight_refs pins raw_cpu for zero-copy pointer lifetime
    cpu_api = _make_mock_api()
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: cpu_api)
    cpu_engine = GgmlNativeEngine(preferred="cpu")
    try:
        assert cpu_engine.is_cpu is True
        raw = torch.zeros(144, dtype=torch.uint8)
        cpu_engine.register_weight(raw, out_features=1, in_features=256, ggml_type=GGML_TYPE_Q4_K)
        assert len(cpu_engine._handles) == 1
        assert len(cpu_engine._weight_refs) == 1
    finally:
        cpu_engine.close()

    # 3. ggml_backend_init_best() falling back to CPU (dev_type=0) frees best and uses CPU init+threads
    fallback_api = _make_mock_api(best_backend=0xBEEF, best_dev_type=0, best_name=b"CPU")
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: fallback_api)
    fb_engine = GgmlNativeEngine(preferred="gpu")
    try:
        assert fb_engine.is_cpu is True
        assert fb_engine.backend_kind.startswith("cpu(")
        fallback_api.lib_base.ggml_backend_free.assert_called_once_with(0xBEEF)
        fallback_api.lib_cpu.ggml_backend_cpu_init.assert_called_once()
        fallback_api.lib_cpu.ggml_backend_cpu_set_n_threads.assert_called_once()
    finally:
        fb_engine.close()

    # 4. ggml_backend_get_device() returning NULL also frees best and falls back to CPU
    null_dev_api = _make_mock_api(best_backend=0xDEAD, best_dev=0, best_dev_type=1)
    monkeypatch.setattr(ggml_native_mod, "_get_ggml_api", lambda: null_dev_api)
    null_dev_engine = GgmlNativeEngine(preferred="gpu")
    try:
        assert null_dev_engine.is_cpu is True
        null_dev_api.lib_base.ggml_backend_free.assert_called_once_with(0xDEAD)
        null_dev_api.lib_cpu.ggml_backend_cpu_init.assert_called_once()
    finally:
        null_dev_engine.close()


@pytest.mark.skipif(not GGML_NATIVE_AVAILABLE, reason="llama_cpp bundled libggml not available")
def test_real_ggml_engine_bounds_and_zero_row_safety() -> None:
    """Verify bounds checks and 0-row matmul guard against the real bundled libggml without crashing."""
    engine = GgmlNativeEngine(preferred="cpu")
    try:
        valid_q4_0 = torch.zeros(18, dtype=torch.uint8)
        for bad_type in (-1, 999, 4):
            with pytest.raises(ValueError):
                engine.register_weight(valid_q4_0, out_features=1, in_features=32, ggml_type=bad_type)

        with pytest.raises(ValueError):
            engine.register_weight(valid_q4_0, out_features=1, in_features=48, ggml_type=GGML_TYPE_Q4_0)
        with pytest.raises(ValueError):
            engine.register_weight(torch.zeros(10, dtype=torch.uint8), out_features=1, in_features=32, ggml_type=GGML_TYPE_Q4_0)

        handle = engine.register_weight(valid_q4_0, out_features=1, in_features=32, ggml_type=GGML_TYPE_Q4_0)
        with pytest.raises(ValueError):
            engine.matmul(handle, np.empty((0, 32), dtype=np.float32))
    finally:
        engine.close()


def test_load_gguf_pytorch_shard_closes_engine_on_midway_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_gguf_pytorch_shard() calls ggml_engine.close() if loading fails after engine init."""
    gguf_path = tmp_path / "tiny-fail.gguf"
    _build_tiny_qwen2_gguf(gguf_path, seed=29, mixed_quants=True)

    created_engines: list[MagicMock] = []

    class _SpyEngine:
        def __init__(self) -> None:
            self.is_cpu = True
            self.backend_kind = "cpu(4t)"
            self.close = MagicMock()
            created_engines.append(self)

        def register_weight(
            self,
            raw_weight: torch.Tensor,
            out_features: int,
            in_features: int,
            ggml_type: int,
        ) -> object:
            return object()

    monkeypatch.setattr(shard_loader_mod, "is_ggml_native_enabled", lambda: True)
    monkeypatch.setattr(shard_loader_mod, "GgmlNativeEngine", _SpyEngine)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Simulated mid-load tokenizer failure")

    monkeypatch.setattr(shard_loader_mod, "GgufTokenizerAdapter", _boom)

    with pytest.raises(RuntimeError, match="Simulated mid-load tokenizer failure"):
        load_gguf_pytorch_shard(
            gguf_path,
            shard_layer_start=0,
            shard_layer_end=1,
            max_ram_mb=512,
            device="cpu",
        )

    assert len(created_engines) == 1
    created_engines[0].close.assert_called_once()
