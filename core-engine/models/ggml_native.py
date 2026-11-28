"""Native C++ `libggml` Quantized MatMul Engine for Sharded GGUF Inference.

Directly invokes the `ggml` shared libraries bundled with `llama_cpp` via `ctypes`
so sharded GGUF linear projections and `lm_head` execute on raw packed quantized
weights (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
`Q6_K`, and `IQ*`) with:
- Zero RAM expansion (zero-copy `ggml_backend_cpu_buffer_from_ptr` on CPU, or
  VRAM copy on discrete CUDA/ROCm/Vulkan GPUs).
- Cached single-token decode graphs (`n_rows == 1`) eliminating per-token Python
  allocation overhead.
- Thread-safe deterministic resource cleanup on shard release.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import threading
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found]

logger = logging.getLogger("shardai.models.ggml_native")

# Supported GGML quantized tensor type IDs (`enum ggml_type` in ggml.h)
NATIVE_QUANTIZED_GGML_TYPES: frozenset[int] = frozenset(
    {
        2,   # Q4_0
        3,   # Q4_1
        6,   # Q5_0
        7,   # Q5_1
        8,   # Q8_0
        10,  # Q2_K
        11,  # Q3_K
        12,  # Q4_K
        13,  # Q5_K
        14,  # Q6_K
        16,  # IQ2_XXS
        17,  # IQ2_XS
        18,  # IQ3_XXS
        19,  # IQ1_S
        20,  # IQ4_NL
        21,  # IQ3_S
        22,  # IQ2_S
        23,  # IQ4_XS
        29,  # IQ1_M
    }
)
GGML_TYPE_F32_ID: int = 0
_GGML_BACKEND_DEVICE_TYPE_GPU: int = 1
_GGML_TYPE_COUNT: int = 40


class _GgmlInitParams(ctypes.Structure):
    """C struct `ggml_init_params` passed by value to `ggml_init`."""

    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class _GgmlCApi:
    """Bound ctypes function pointers for `libggml-base`, `libggml-cpu`, and `libggml`."""

    def __init__(self, lib_dir: Path) -> None:
        system = platform.system()
        if system == "Darwin":
            base_name, cpu_name, ggml_name = (
                "libggml-base.dylib",
                "libggml-cpu.dylib",
                "libggml.dylib",
            )
        elif system == "Windows":
            base_name, cpu_name, ggml_name = (
                "ggml-base.dll",
                "ggml-cpu.dll",
                "ggml.dll",
            )
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(lib_dir))
        else:
            base_name, cpu_name, ggml_name = (
                "libggml-base.so",
                "libggml-cpu.so",
                "libggml.so",
            )

        base_path = lib_dir / base_name
        cpu_path = lib_dir / cpu_name
        ggml_path = lib_dir / ggml_name
        if not (base_path.is_file() and cpu_path.is_file() and ggml_path.is_file()):
            raise FileNotFoundError(
                f"Required ggml shared libraries not found in {lib_dir}"
            )

        self.lib_base = ctypes.CDLL(str(base_path))
        self.lib_cpu = ctypes.CDLL(str(cpu_path))
        self.lib_ggml = ctypes.CDLL(str(ggml_path))

        # Backend discovery / loading
        self.lib_ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        self.lib_ggml.ggml_backend_load_all_from_path.restype = None
        self.lib_ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
        self.lib_ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        self.lib_ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
        self.lib_ggml.ggml_backend_init_best.restype = ctypes.c_void_p

        self.lib_base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_dev_name.restype = ctypes.c_char_p
        self.lib_base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_dev_type.restype = ctypes.c_int
        self.lib_base.ggml_backend_get_device.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_get_device.restype = ctypes.c_void_p
        self.lib_base.ggml_backend_name.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_name.restype = ctypes.c_char_p

        # Context & tensor graph construction
        self.lib_base.ggml_init.argtypes = [_GgmlInitParams]
        self.lib_base.ggml_init.restype = ctypes.c_void_p
        self.lib_base.ggml_free.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_free.restype = None
        self.lib_base.ggml_tensor_overhead.restype = ctypes.c_size_t
        self.lib_base.ggml_graph_overhead.restype = ctypes.c_size_t
        self.lib_base.ggml_new_tensor_2d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib_base.ggml_new_tensor_2d.restype = ctypes.c_void_p
        self.lib_base.ggml_mul_mat.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_mul_mat.restype = ctypes.c_void_p
        self.lib_base.ggml_new_graph.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_new_graph.restype = ctypes.c_void_p
        self.lib_base.ggml_build_forward_expand.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_build_forward_expand.restype = None

        # Backend buffers, allocators, and execution
        self.lib_base.ggml_backend_cpu_buffer_from_ptr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib_base.ggml_backend_cpu_buffer_from_ptr.restype = ctypes.c_void_p
        self.lib_base.ggml_backend_tensor_alloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_backend_tensor_alloc.restype = ctypes.c_int
        self.lib_base.ggml_backend_alloc_ctx_tensors.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_backend_alloc_ctx_tensors.restype = ctypes.c_void_p
        self.lib_base.ggml_backend_buffer_free.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_buffer_free.restype = None
        self.lib_base.ggml_backend_tensor_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib_base.ggml_backend_tensor_set.restype = None
        self.lib_base.ggml_backend_tensor_get.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self.lib_base.ggml_backend_tensor_get.restype = None
        self.lib_base.ggml_backend_graph_compute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_backend_graph_compute.restype = ctypes.c_int
        self.lib_base.ggml_backend_free.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_free.restype = None
        self.lib_base.ggml_backend_get_default_buffer_type.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_backend_get_default_buffer_type.restype = ctypes.c_void_p
        self.lib_base.ggml_gallocr_new.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_gallocr_new.restype = ctypes.c_void_p
        self.lib_base.ggml_gallocr_alloc_graph.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib_base.ggml_gallocr_alloc_graph.restype = ctypes.c_bool
        self.lib_base.ggml_gallocr_free.argtypes = [ctypes.c_void_p]
        self.lib_base.ggml_gallocr_free.restype = None
        self.lib_base.ggml_blck_size.argtypes = [ctypes.c_int]
        self.lib_base.ggml_blck_size.restype = ctypes.c_int64
        self.lib_base.ggml_type_size.argtypes = [ctypes.c_int]
        self.lib_base.ggml_type_size.restype = ctypes.c_size_t

        # CPU backend init & thread count
        self.lib_cpu.ggml_backend_cpu_init.restype = ctypes.c_void_p
        self.lib_cpu.ggml_backend_cpu_set_n_threads.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib_cpu.ggml_backend_cpu_set_n_threads.restype = None

        self.lib_ggml.ggml_backend_load_all_from_path(str(lib_dir).encode("utf-8"))
        self.tensor_overhead: int = int(self.lib_base.ggml_tensor_overhead())
        self.graph_overhead: int = int(self.lib_base.ggml_graph_overhead())


_API_LOCK: threading.Lock = threading.Lock()
_CACHED_API: _GgmlCApi | None = None
_API_INIT_ERROR: str | None = None


def _try_load_ggml_api() -> _GgmlCApi | None:
    global _CACHED_API, _API_INIT_ERROR
    with _API_LOCK:
        if _CACHED_API is not None:
            return _CACHED_API
        if _API_INIT_ERROR is not None:
            return None
        try:
            import llama_cpp  # type: ignore[import-not-found]

            lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
            _CACHED_API = _GgmlCApi(lib_dir)
            return _CACHED_API
        except Exception as exc:
            _API_INIT_ERROR = str(exc)
            logger.debug("Native libggml backend unavailable: %s", exc)
            return None


def _get_ggml_api() -> _GgmlCApi:
    api = _try_load_ggml_api()
    if api is None:
        raise RuntimeError(f"Native libggml backend unavailable: {_API_INIT_ERROR}")
    return api


GGML_NATIVE_AVAILABLE: bool = _try_load_ggml_api() is not None


def is_ggml_native_enabled() -> bool:
    """Return True when bundled `libggml` is available and not disabled by env."""
    if not GGML_NATIVE_AVAILABLE:
        return False
    choice = os.getenv("SHARDAI_GGML_BACKEND", "auto").strip().lower()
    return choice not in {"off", "false", "0", "pytorch", "disabled"}


@dataclass
class _CachedGraph:
    """Reusable GGML graph and allocator for single-token decode (`n_rows == 1`)."""

    g_ctx: int
    gallocr: int
    gf: int
    x_tensor: int
    y_tensor: int


@dataclass
class GgmlWeightHandle:
    """Owns one quantized weight tensor's GGML context, buffer, and cached decode graph."""

    out_features: int
    in_features: int
    ggml_type: int
    w_ctx: int
    w_buf: int
    w_tensor: int
    decode_graph: _CachedGraph | None = None


class GgmlNativeEngine:
    """Thread-safe native `libggml` backend executing quantized `ggml_mul_mat` kernels."""

    def __init__(self, preferred: str | None = None) -> None:
        api = _get_ggml_api()
        self._api: _GgmlCApi = api
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False
        self._backend: int | None = None
        self._handles: list[GgmlWeightHandle] = []
        self._weight_refs: list[torch.Tensor] = []

        choice = (
            (preferred if preferred is not None else os.getenv("SHARDAI_GGML_BACKEND", "auto"))
            .strip()
            .lower()
        )
        use_gpu = False
        if choice in {"gpu", "cuda", "metal", "vulkan", "rocm"}:
            use_gpu = True
        elif choice == "auto":
            use_gpu = self._has_discrete_gpu(api)

        backend: int | None = None
        is_cpu = True
        backend_kind = "cpu"
        if use_gpu:
            best = api.lib_ggml.ggml_backend_init_best()
            if best:
                dev = api.lib_base.ggml_backend_get_device(best)
                dev_type = int(api.lib_base.ggml_backend_dev_type(dev)) if dev else -1
                if dev_type == _GGML_BACKEND_DEVICE_TYPE_GPU:
                    backend = int(best)
                    is_cpu = False
                    raw_name = api.lib_base.ggml_backend_name(best)
                    bname = (
                        raw_name.decode("utf-8", errors="replace").strip()
                        if isinstance(raw_name, (bytes, bytearray))
                        else ""
                    )
                    backend_kind = f"gpu({bname})" if bname else "gpu"
                else:
                    # ggml_backend_init_best fell back to non-GPU (e.g., CPU/ACCEL); free and use configured CPU path
                    api.lib_base.ggml_backend_free(best)

        if backend is None:
            cpu_backend = api.lib_cpu.ggml_backend_cpu_init()
            if not cpu_backend:
                raise RuntimeError("Failed to initialize ggml CPU backend")
            cpu_threads = os.cpu_count() or 4
            env_threads = os.getenv("SHARDAI_GGML_THREADS", "").strip()
            if env_threads.isdigit() and int(env_threads) > 0:
                cpu_threads = int(env_threads)
            n_threads = max(1, min(16, cpu_threads))
            api.lib_cpu.ggml_backend_cpu_set_n_threads(cpu_backend, n_threads)
            backend = int(cpu_backend)
            is_cpu = True
            backend_kind = f"cpu({n_threads}t)"

        try:
            default_buft = int(
                api.lib_base.ggml_backend_get_default_buffer_type(backend)
            )
            if not default_buft:
                raise RuntimeError("ggml_backend_get_default_buffer_type returned null")
        except Exception:
            try:
                api.lib_base.ggml_backend_free(backend)
            except Exception:
                pass
            raise

        self._backend = backend
        self._is_cpu: bool = is_cpu
        self._backend_kind: str = backend_kind
        self._default_buft: int = default_buft
        logger.info("Initialized native libggml engine (%s)", self._backend_kind)

    @staticmethod
    def _has_discrete_gpu(api: _GgmlCApi) -> bool:
        """Return True if a non-Metal discrete GPU GGML device (e.g. CUDA/Vulkan/ROCm) exists."""
        dev_count = int(api.lib_ggml.ggml_backend_dev_count())
        for idx in range(dev_count):
            dev = api.lib_ggml.ggml_backend_dev_get(idx)
            if not dev:
                continue
            dev_type = int(api.lib_base.ggml_backend_dev_type(dev))
            if dev_type != _GGML_BACKEND_DEVICE_TYPE_GPU:
                continue
            raw_name = api.lib_base.ggml_backend_dev_name(dev)
            name = raw_name.decode("utf-8", errors="replace") if raw_name else ""
            # On Apple Silicon, multi-threaded CPU SIMD beats Metal per-op dispatch for single-linear sharding
            if not name.upper().startswith("MTL"):
                return True
        return False

    @property
    def is_cpu(self) -> bool:
        return self._is_cpu

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    def register_weight(
        self,
        raw_weight: torch.Tensor,
        out_features: int,
        in_features: int,
        ggml_type: int,
    ) -> GgmlWeightHandle:
        """Register a packed 1D `torch.uint8` weight tensor on the GGML backend (zero-copy on CPU)."""
        if raw_weight.dtype != torch.uint8 or raw_weight.ndim != 1:
            raise ValueError("Expected 1D torch.uint8 raw_weight tensor")
        out_f = int(out_features)
        in_f = int(in_features)
        if out_f <= 0 or in_f <= 0:
            raise ValueError(
                f"Expected positive weight dimensions, got out_features={out_f}, in_features={in_f}"
            )
        nbytes = int(raw_weight.numel())
        if nbytes <= 0:
            raise ValueError("Cannot register empty weight buffer in GgmlNativeEngine")

        gtype = int(ggml_type)
        # Guard C type_traits[] indexing against out-of-bounds type IDs
        if gtype < 0 or gtype >= _GGML_TYPE_COUNT:
            raise ValueError(f"Unsupported or invalid ggml_type: {gtype}")
        api = self._api
        blck_size = int(api.lib_base.ggml_blck_size(gtype))
        type_size = int(api.lib_base.ggml_type_size(gtype))
        if blck_size <= 0 or type_size <= 0:
            raise ValueError(
                f"Unsupported ggml_type={gtype} (blck_size={blck_size}, type_size={type_size})"
            )
        if in_f % blck_size != 0:
            raise ValueError(
                f"in_features={in_f} must be divisible by blck_size={blck_size} for ggml_type={gtype}"
            )
        expected_nbytes = (in_f // blck_size) * type_size * out_f
        if nbytes != expected_nbytes:
            raise ValueError(
                f"Weight buffer byte length {nbytes} != expected {expected_nbytes} "
                f"for shape=({out_f}, {in_f}) and ggml_type={gtype}"
            )

        raw_cpu = raw_weight.detach().to(device="cpu").contiguous()

        with self._lock:
            if self._closed or self._backend is None:
                raise RuntimeError("GgmlNativeEngine is closed")
            ctx_bytes = max(4096, api.tensor_overhead * 4)
            w_ctx = api.lib_base.ggml_init(_GgmlInitParams(ctx_bytes, None, True))
            if not w_ctx:
                raise RuntimeError("ggml_init failed for weight context")

            w_buf: int | None = None
            try:
                w_tensor = api.lib_base.ggml_new_tensor_2d(
                    w_ctx, gtype, in_f, out_f
                )
                if not w_tensor:
                    raise RuntimeError(
                        f"ggml_new_tensor_2d failed (type={gtype}, shape=({in_f}, {out_f}))"
                    )

                ptr = int(raw_cpu.data_ptr())
                if self._is_cpu:
                    w_buf = api.lib_base.ggml_backend_cpu_buffer_from_ptr(ptr, nbytes)
                    if not w_buf:
                        raise RuntimeError("ggml_backend_cpu_buffer_from_ptr returned null")
                    rc = int(api.lib_base.ggml_backend_tensor_alloc(w_buf, w_tensor, ptr))
                    if rc != 0:
                        raise RuntimeError(f"ggml_backend_tensor_alloc failed with code {rc}")
                else:
                    w_buf = api.lib_base.ggml_backend_alloc_ctx_tensors(w_ctx, self._backend)
                    if not w_buf:
                        raise RuntimeError("ggml_backend_alloc_ctx_tensors failed for GPU weight")
                    api.lib_base.ggml_backend_tensor_set(w_tensor, ptr, 0, nbytes)

                handle = GgmlWeightHandle(
                    out_features=out_f,
                    in_features=in_f,
                    ggml_type=gtype,
                    w_ctx=int(w_ctx),
                    w_buf=int(w_buf),
                    w_tensor=int(w_tensor),
                )
                # Pin host tensor only on CPU zero-copy path; GPU copies data into VRAM
                if self._is_cpu:
                    self._weight_refs.append(raw_cpu)
                self._handles.append(handle)
                return handle
            except Exception:
                if w_buf:
                    api.lib_base.ggml_backend_buffer_free(w_buf)
                api.lib_base.ggml_free(w_ctx)
                raise

    def _build_compute_graph(self, handle: GgmlWeightHandle, n_rows: int) -> _CachedGraph:
        """Build and allocate a `ggml_mul_mat(w, x)` compute graph for `n_rows` input vectors."""
        api = self._api
        ctx_size = max(8192, api.tensor_overhead * 16 + api.graph_overhead)
        gallocr = api.lib_base.ggml_gallocr_new(self._default_buft)
        if not gallocr:
            raise RuntimeError("ggml_gallocr_new failed")

        g_ctx = api.lib_base.ggml_init(_GgmlInitParams(ctx_size, None, True))
        if not g_ctx:
            api.lib_base.ggml_gallocr_free(gallocr)
            raise RuntimeError("ggml_init failed for compute graph context")

        try:
            x_tensor = api.lib_base.ggml_new_tensor_2d(
                g_ctx, GGML_TYPE_F32_ID, int(handle.in_features), int(n_rows)
            )
            y_tensor = api.lib_base.ggml_mul_mat(g_ctx, handle.w_tensor, x_tensor)
            gf = api.lib_base.ggml_new_graph(g_ctx)
            if not x_tensor or not y_tensor or not gf:
                raise RuntimeError("Failed to construct ggml_mul_mat forward graph")
            api.lib_base.ggml_build_forward_expand(gf, y_tensor)
            ok = bool(api.lib_base.ggml_gallocr_alloc_graph(gallocr, gf))
            if not ok:
                raise RuntimeError("ggml_gallocr_alloc_graph failed")
            return _CachedGraph(
                g_ctx=int(g_ctx),
                gallocr=int(gallocr),
                gf=int(gf),
                x_tensor=int(x_tensor),
                y_tensor=int(y_tensor),
            )
        except Exception:
            api.lib_base.ggml_free(g_ctx)
            api.lib_base.ggml_gallocr_free(gallocr)
            raise

    def _free_graph(self, graph: _CachedGraph) -> None:
        api = self._api
        api.lib_base.ggml_free(graph.g_ctx)
        api.lib_base.ggml_gallocr_free(graph.gallocr)

    def matmul(self, handle: GgmlWeightHandle, x_2d_f32: np.ndarray) -> np.ndarray:
        """Compute `x_2d_f32 @ W.T` directly against quantized `handle` weights."""
        if (
            x_2d_f32.ndim != 2
            or int(x_2d_f32.shape[0]) <= 0
            or int(x_2d_f32.shape[1]) != handle.in_features
        ):
            raise ValueError(
                f"Expected non-empty 2D input with width {handle.in_features}, got {x_2d_f32.shape}"
            )
        x_contig = np.ascontiguousarray(x_2d_f32, dtype=np.float32)
        n_rows = int(x_contig.shape[0])
        out_np = np.empty((n_rows, handle.out_features), dtype=np.float32)

        with self._lock:
            if self._closed or self._backend is None:
                raise RuntimeError("GgmlNativeEngine is closed")
            api = self._api
            ephemeral_graph: _CachedGraph | None = None
            if n_rows == 1:
                if handle.decode_graph is None:
                    handle.decode_graph = self._build_compute_graph(handle, 1)
                graph = handle.decode_graph
            else:
                ephemeral_graph = self._build_compute_graph(handle, n_rows)
                graph = ephemeral_graph

            try:
                api.lib_base.ggml_backend_tensor_set(
                    graph.x_tensor, x_contig.ctypes.data, 0, x_contig.nbytes
                )
                rc = int(api.lib_base.ggml_backend_graph_compute(self._backend, graph.gf))
                if rc != 0:
                    raise RuntimeError(f"ggml_backend_graph_compute failed with status {rc}")
                api.lib_base.ggml_backend_tensor_get(
                    graph.y_tensor, out_np.ctypes.data, 0, out_np.nbytes
                )
            finally:
                if ephemeral_graph is not None:
                    self._free_graph(ephemeral_graph)

        return out_np

    def close(self) -> None:
        """Release all cached graphs, weight buffers, GGML contexts, and backend handle."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            api = self._api
            for handle in self._handles:
                if handle.decode_graph is not None:
                    try:
                        self._free_graph(handle.decode_graph)
                    except Exception:
                        pass
                    handle.decode_graph = None
                try:
                    api.lib_base.ggml_backend_buffer_free(handle.w_buf)
                except Exception:
                    pass
                try:
                    api.lib_base.ggml_free(handle.w_ctx)
                except Exception:
                    pass
            self._handles.clear()
            self._weight_refs.clear()
            if self._backend is not None:
                try:
                    api.lib_base.ggml_backend_free(self._backend)
                except Exception:
                    pass
                self._backend = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
