"""Low-level Shard Forward Pass & Token Decoding Engine (`core-engine/models/`).

Supports true distributed PyTorch pipeline parallelism across transformer layers
(`shard_layer_start` .. `shard_layer_end`) loaded selectively from Hugging Face
`.safetensors` shards, as well as a lightweight deterministic simulation fallback
for offline unit tests without physical model weights.

Complies with `.antigravity/rules.md` and `.antigravity/review.md`:
- 100% Python type hints.
- Executes all PyTorch forward/decode operations inside `torch.inference_mode()`.
- Bounds per-request KV-caches with strict LRU capacity and TTL expiration.
- Explicitly releases model handles, KV-caches, and triggers `gc.collect()` on cleanup.
"""

from array import array
from collections import OrderedDict
import ctypes
from dataclasses import dataclass, field
import gc
import logging
import math
from pathlib import Path
import struct
import time
from typing import Any

logger = logging.getLogger("shardai.models.runner")

DEFAULT_HIDDEN_DIM: int = 4096
DEFAULT_TOTAL_LAYERS: int = 32
MAX_KV_CACHE_ENTRIES: int = 8
KV_CACHE_TTL_SECONDS: float = 120.0

# Vocabulary sample bank used ONLY when running offline simulation without real weights
_SYNTHETIC_VOCAB: tuple[str, ...] = (
    "ShardAi",
    " distributed",
    " P2P",
    " pipeline",
    " inference",
    " streams",
    " hidden",
    " state",
    " tensors",
    " across",
    " peer",
    " nodes",
    " with",
    " binary",
    " struct",
    " serialization",
    " and",
    " adaptive",
    " compression.",
)


@dataclass
class _RequestSession:
    """Bounded per-request KV-cache and incremental UTF-8 decoder state."""

    created_at: float
    past_key_values: Any  # `transformers.cache_utils.DynamicCache` when PyTorch is active
    generated_ids: list[int] = field(default_factory=list)
    prev_decoded_text: str = ""


def _resolve_safetensors_files(model_dir: Path) -> list[Path]:
    """Return sorted `.safetensors` weight files in a cached model directory."""
    return sorted(p for p in model_dir.glob("*.safetensors") if p.is_file() and p.stat().st_size > 0)


def _extract_eos_token_ids(config: Any, tokenizer: Any) -> set[int]:
    """Collect all EOS / stop token IDs from Hugging Face config and tokenizer."""
    eos_ids: set[int] = set()
    cfg_eos = getattr(config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple, set)):
        for item in cfg_eos:
            if isinstance(item, int):
                eos_ids.add(item)

    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tok_eos, int):
        eos_ids.add(tok_eos)

    for special_tok in ("<|im_end|>", "<|eot_id|>", "</s>"):
        tok_id = tokenizer.convert_tokens_to_ids(special_tok)
        if isinstance(tok_id, int) and tok_id >= 0:
            eos_ids.add(tok_id)

    return eos_ids


class ShardModelRunner:
    """Executes forward passes across the transformer layers assigned to this node."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        model_path: str | None = None,
        shard_layer_start: int = 0,
        shard_layer_end: int = 15,
        total_layers: int = DEFAULT_TOTAL_LAYERS,
        n_layers: int | None = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_ram_mb: int = 4096,
        native_handle: Any | None = None,  # Optional external PyTorch/llama_cpp handle
        hf_config: Any | None = None,  # Optional preloaded PretrainedConfig for unit tests
        hf_model: Any | None = None,  # Optional preloaded PreTrainedModel for sharding unit tests
        hf_tokenizer: Any | None = None,  # Optional preloaded tokenizer for unit tests
    ) -> None:
        self.model_id: str = model_id
        self.model_path: str | None = model_path
        self.shard_layer_start: int = shard_layer_start
        self.shard_layer_end: int = shard_layer_end
        self.total_layers: int = total_layers
        self.n_layers: int = (
            n_layers
            if n_layers is not None
            else max(1, shard_layer_end - shard_layer_start + 1)
        )
        self.hidden_dim: int = hidden_dim
        self.max_ram_mb: int = max_ram_mb
        self._native_handle: Any | None = native_handle

        # PyTorch layer-sharded components
        self._is_real_pytorch: bool = False
        self._hf_config: Any | None = None
        self._tokenizer: Any | None = None
        self._eos_token_ids: set[int] = set()
        self._embed_tokens: Any | None = None
        self._layers: list[Any] = []
        self._rotary_emb: Any | None = None
        self._norm: Any | None = None
        self._lm_head: Any | None = None
        self._shard_container: Any | None = None
        self._compute_dtype: Any | None = None
        self._sessions: OrderedDict[str, _RequestSession] = OrderedDict()

        if hf_model is not None and hf_config is not None:
            self._init_from_existing_hf_model(
                hf_model=hf_model,
                hf_config=hf_config,
                hf_tokenizer=hf_tokenizer,
            )
        elif self._native_handle is None and self.model_path is not None:
            candidate = Path(self.model_path)
            if candidate.is_dir() and (candidate / "config.json").is_file():
                self._init_pytorch_shard_from_dir(candidate)
            elif candidate.is_file() and candidate.suffix.lower() == ".gguf" and candidate.stat().st_size > 1_048_576:
                self._init_llama_cpp(str(candidate), self.n_layers)

    @property
    def is_real_model(self) -> bool:
        """Return True when executing real PyTorch sharded weights (not stub simulation)."""
        return self._is_real_pytorch

    @property
    def is_head_node(self) -> bool:
        """Return True if this node holds the embedding + initial transformer layer (layer 0)."""
        return self.shard_layer_start == 0

    @property
    def is_tail_node(self) -> bool:
        """Return True if this node holds the final transformer layer + LM head."""
        return self.shard_layer_end >= (self.total_layers - 1)

    def _init_from_existing_hf_model(
        self,
        hf_model: Any,
        hf_config: Any,
        hf_tokenizer: Any | None = None,
    ) -> None:
        """Slice an in-memory Hugging Face causal LM into only this node's assigned layers."""
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]

        self._hf_config = hf_config
        self.total_layers = int(getattr(hf_config, "num_hidden_layers", self.total_layers))
        self.hidden_dim = int(getattr(hf_config, "hidden_size", self.hidden_dim))
        self.shard_layer_end = min(self.shard_layer_end, self.total_layers - 1)
        self.n_layers = max(1, self.shard_layer_end - self.shard_layer_start + 1)

        inner_model = getattr(hf_model, "model", hf_model)
        all_layers = list(inner_model.layers)
        kept_layers = all_layers[self.shard_layer_start : self.shard_layer_end + 1]

        container = nn.Module()
        self._layers = list(kept_layers)
        container.layers = nn.ModuleList(kept_layers)

        if hasattr(inner_model, "rotary_emb") and inner_model.rotary_emb is not None:
            self._rotary_emb = inner_model.rotary_emb
            container.rotary_emb = self._rotary_emb

        if self.is_head_node and hasattr(inner_model, "embed_tokens"):
            self._embed_tokens = inner_model.embed_tokens
            container.embed_tokens = self._embed_tokens

        if self.is_tail_node:
            if hasattr(inner_model, "norm"):
                self._norm = inner_model.norm
                container.norm = self._norm
            if hasattr(hf_model, "lm_head"):
                self._lm_head = hf_model.lm_head
                container.lm_head = self._lm_head

        container.eval()
        for param in container.parameters():
            param.requires_grad_(False)

        self._compute_dtype = torch.float32
        container.to(dtype=self._compute_dtype, device="cpu")
        self._shard_container = container
        self._tokenizer = hf_tokenizer
        if hf_tokenizer is not None:
            self._eos_token_ids = _extract_eos_token_ids(hf_config, hf_tokenizer)
        else:
            cfg_eos = getattr(hf_config, "eos_token_id", None)
            if isinstance(cfg_eos, int):
                self._eos_token_ids = {cfg_eos}
        self._is_real_pytorch = True

    def _init_pytorch_shard_from_dir(self, model_dir: Path) -> None:
        """Load ONLY the assigned transformer layers from `.safetensors` using a meta skeleton."""
        safetensor_files = _resolve_safetensors_files(model_dir)
        if not safetensor_files:
            logger.info("No .safetensors files in %s; keeping simulation mode.", model_dir)
            return

        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
        from safetensors import safe_open  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        self._hf_config = config
        self.total_layers = int(getattr(config, "num_hidden_layers", self.total_layers))
        self.hidden_dim = int(getattr(config, "hidden_size", self.hidden_dim))
        self.shard_layer_start = max(0, min(self.shard_layer_start, self.total_layers - 1))
        self.shard_layer_end = max(
            self.shard_layer_start, min(self.shard_layer_end, self.total_layers - 1)
        )
        self.n_layers = max(1, self.shard_layer_end - self.shard_layer_start + 1)

        if self.is_head_node or self.is_tail_node:
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            self._eos_token_ids = _extract_eos_token_ids(config, self._tokenizer)

        # Build zero-RAM model skeleton on PyTorch meta device and discard unowned layers
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(config)

        inner_meta = getattr(meta_model, "model", meta_model)
        kept_layers = [
            inner_meta.layers[idx]
            for idx in range(self.shard_layer_start, self.shard_layer_end + 1)
        ]

        container = nn.Module()
        container.layers = nn.ModuleList(kept_layers)
        self._layers = kept_layers

        if hasattr(inner_meta, "rotary_emb") and inner_meta.rotary_emb is not None:
            rotary_cls = type(inner_meta.rotary_emb)
            self._rotary_emb = rotary_cls(config=config)
            container.rotary_emb = self._rotary_emb

        if self.is_head_node and hasattr(inner_meta, "embed_tokens"):
            self._embed_tokens = inner_meta.embed_tokens
            container.embed_tokens = self._embed_tokens

        if self.is_tail_node:
            if hasattr(inner_meta, "norm"):
                self._norm = inner_meta.norm
                container.norm = self._norm
            if hasattr(meta_model, "lm_head"):
                self._lm_head = meta_model.lm_head
                container.lm_head = self._lm_head

        del meta_model

        # Map safetensors keys strictly for kept submodules so unowned layers never enter RAM
        target_keys: set[str] = set(container.state_dict().keys())
        loaded_state: dict[str, torch.Tensor] = {}

        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as handle:
                for raw_key in handle.keys():
                    mapped_key: str | None = None
                    if raw_key.startswith("model.layers."):
                        parts = raw_key.split(".", 3)
                        if len(parts) == 4 and parts[2].isdigit():
                            layer_num = int(parts[2])
                            if self.shard_layer_start <= layer_num <= self.shard_layer_end:
                                local_idx = layer_num - self.shard_layer_start
                                mapped_key = f"layers.{local_idx}.{parts[3]}"
                    elif self.is_head_node and raw_key.startswith("model.embed_tokens."):
                        mapped_key = raw_key[len("model.") :]
                    elif self.is_tail_node and raw_key.startswith("model.norm."):
                        mapped_key = raw_key[len("model.") :]
                    elif self.is_tail_node and raw_key.startswith("lm_head."):
                        mapped_key = raw_key

                    if mapped_key is not None and mapped_key in target_keys:
                        loaded_state[mapped_key] = handle.get_tensor(raw_key)

                    # Handle bidirectional weight tying (`tie_word_embeddings=True`)
                    if (
                        self.is_tail_node
                        and raw_key == "model.embed_tokens.weight"
                        and "lm_head.weight" in target_keys
                        and "lm_head.weight" not in loaded_state
                    ):
                        loaded_state["lm_head.weight"] = handle.get_tensor(raw_key)
                    elif (
                        self.is_head_node
                        and raw_key == "lm_head.weight"
                        and "embed_tokens.weight" in target_keys
                        and "embed_tokens.weight" not in loaded_state
                    ):
                        loaded_state["embed_tokens.weight"] = handle.get_tensor(raw_key)

        if not loaded_state:
            raise RuntimeError(
                f"No matching shard tensors found in {model_dir} for layers "
                f"{self.shard_layer_start}..{self.shard_layer_end}."
            )

        container.load_state_dict(loaded_state, strict=False, assign=True)
        loaded_state.clear()

        # Choose float32 on CPU when budget fits (3x faster on x86), else bfloat16 for half RAM
        shard_bytes_bf16 = sum(
            p.numel() * 2 for p in container.parameters() if p.device.type != "meta"
        )
        shard_mb_f32 = (shard_bytes_bf16 * 2) / (1024.0 * 1024.0)
        self._compute_dtype = (
            torch.float32 if shard_mb_f32 <= (self.max_ram_mb * 0.85) else torch.bfloat16
        )

        container.to(dtype=self._compute_dtype, device="cpu")
        container.eval()
        for param in container.parameters():
            param.requires_grad_(False)

        self._shard_container = container
        self._is_real_pytorch = True
        logger.info(
            "Loaded PyTorch shard for %s: layers %d..%d/%d (dtype=%s, role=%s)",
            self.model_id,
            self.shard_layer_start,
            self.shard_layer_end,
            self.total_layers,
            self._compute_dtype,
            "HEAD+TAIL"
            if (self.is_head_node and self.is_tail_node)
            else ("HEAD" if self.is_head_node else ("TAIL" if self.is_tail_node else "MIDDLE")),
        )

    def _init_llama_cpp(self, gguf_path: str, n_layers: int) -> None:
        """Initialize `llama_cpp.Llama` when an explicit real `.gguf` file is provided."""
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]

            self._native_handle = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_layers,
                embeddings=True,
                verbose=False,
            )
            logger.info("Initialized llama_cpp with %s (n_layers=%d)", gguf_path, n_layers)
        except Exception as exc:
            logger.info(
                "Using lightweight float32 shard runner for %s (n_layers=%d): %s",
                gguf_path,
                n_layers,
                exc,
            )

    def _evict_expired_sessions(self) -> None:
        """Enforce TTL and LRU capacity bounds on cached per-request KV states."""
        now = time.monotonic()
        expired = [
            req_id
            for req_id, sess in self._sessions.items()
            if (now - sess.created_at) > KV_CACHE_TTL_SECONDS
        ]
        for req_id in expired:
            self._sessions.pop(req_id, None)

        while len(self._sessions) > MAX_KV_CACHE_ENTRIES:
            self._sessions.popitem(last=False)

    def _get_or_create_session(self, request_id: str, step_index: int) -> _RequestSession:
        """Retrieve or create a bounded KV-cache session for `request_id`."""
        self._evict_expired_sessions()
        if step_index == 0 or request_id not in self._sessions:
            from transformers.cache_utils import DynamicCache  # type: ignore[import-not-found]

            session = _RequestSession(
                created_at=time.monotonic(),
                past_key_values=DynamicCache(config=self._hf_config),
            )
            self._sessions[request_id] = session
            self._evict_expired_sessions()
            return session

        self._sessions.move_to_end(request_id)
        return self._sessions[request_id]

    def release_request_cache(self, request_id: str) -> None:
        """Drop KV-cache and decoding history for a completed/aborted request."""
        self._sessions.pop(request_id, None)

    def tokenize_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply chat template (when available) and tokenize chat messages into token IDs."""
        if self._tokenizer is not None:
            if getattr(self._tokenizer, "chat_template", None):
                encoded = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if hasattr(encoded, "get") and "input_ids" in encoded:
                    encoded = encoded["input_ids"]
                elif hasattr(encoded, "input_ids"):
                    encoded = encoded.input_ids
                if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
                    encoded = encoded[0]
                return [int(t) for t in encoded]

            joined = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
            return [int(t) for t in self._tokenizer.encode(joined, add_special_tokens=True)]

        joined_fallback = "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
        return self.tokenize(joined_fallback)

    def tokenize(self, prompt: str) -> list[int]:
        """Convert input prompt text into token IDs (Head Node only)."""
        if not prompt.strip():
            return [1]

        if self._tokenizer is not None:
            if getattr(self._tokenizer, "chat_template", None):
                return self.tokenize_chat([{"role": "user", "content": prompt}])
            return [int(t) for t in self._tokenizer.encode(prompt, add_special_tokens=True)]

        if self._native_handle is not None and hasattr(self._native_handle, "tokenize"):
            raw_tokens: list[int] = self._native_handle.tokenize(
                prompt.encode("utf-8"), add_bos=True
            )
            return raw_tokens

        words: list[str] = prompt.strip().split()
        return [1] + [(abs(hash(word)) % 31999) + 2 for word in words]

    def embed_tokens_to_raw_tensor(self, token_ids: list[int]) -> tuple[bytes, tuple[int, int]]:
        """Project input token IDs into an initial contiguous float32 tensor buffer."""
        if self._is_real_pytorch and self._embed_tokens is not None:
            import torch  # type: ignore[import-not-found]

            with torch.inference_mode():
                ids_tensor = torch.tensor([token_ids], dtype=torch.long)
                embedded = self._embed_tokens(ids_tensor).squeeze(0).to(torch.float32).contiguous()
                return embedded.numpy().tobytes(), (int(embedded.shape[0]), int(embedded.shape[1]))

        seq_len: int = max(1, min(len(token_ids), 16))
        seed_val: float = float(sum(token_ids) % 997) * 0.001
        base_row = array(
            "f",
            (math.sin(seed_val + (i * 0.01)) * 0.1 for i in range(self.hidden_dim)),
        )
        full_buffer: bytes = base_row.tobytes() * seq_len
        return full_buffer, (seq_len, self.hidden_dim)

    def forward_pass(
        self,
        input_data: bytes | memoryview | list[int],
        shape: tuple[int, int] | None = None,
        request_id: str = "default",
        step_index: int = 0,
    ) -> tuple[bytes, tuple[int, int]]:
        """Run input tokens or upstream hidden-state tensor through this node's assigned layers."""
        # Real PyTorch Layer-Sharded Execution Path
        if self._is_real_pytorch:
            import torch  # type: ignore[import-not-found]

            with torch.inference_mode():
                session = self._get_or_create_session(request_id, step_index)
                if isinstance(input_data, list):
                    if self._embed_tokens is None:
                        raise RuntimeError("Head node embedding layer is not loaded.")
                    ids_tensor = torch.tensor([input_data], dtype=torch.long)
                    hidden_states = self._embed_tokens(ids_tensor)
                else:
                    raw_in = bytes(input_data)
                    if shape is not None:
                        seq_len, hidden_dim = shape
                    else:
                        seq_len = max(1, (len(raw_in) // 4) // self.hidden_dim)
                        hidden_dim = self.hidden_dim
                    hidden_states = (
                        torch.frombuffer(bytearray(raw_in), dtype=torch.float32)
                        .reshape(1, seq_len, hidden_dim)
                        .to(dtype=self._compute_dtype)
                    )

                seq_len = int(hidden_states.shape[1])
                past_len: int = int(
                    session.past_key_values.get_seq_length(self.shard_layer_start)
                )
                position_ids = torch.arange(
                    past_len, past_len + seq_len, dtype=torch.long
                ).unsqueeze(0)
                cache_position = torch.arange(past_len, past_len + seq_len, dtype=torch.long)

                position_embeddings = (
                    self._rotary_emb(hidden_states, position_ids)
                    if self._rotary_emb is not None
                    else None
                )

                attention_mask: torch.Tensor | None = None
                if seq_len > 1:
                    total_kv_len = past_len + seq_len
                    min_val = torch.finfo(hidden_states.dtype).min
                    mask = torch.full(
                        (1, 1, seq_len, total_kv_len),
                        fill_value=min_val,
                        dtype=hidden_states.dtype,
                    )
                    attention_mask = torch.triu(mask, diagonal=1 + past_len)

                for layer in self._layers:
                    layer_out = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=session.past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                out_f32 = hidden_states.squeeze(0).to(torch.float32).contiguous()
                return out_f32.numpy().tobytes(), (int(out_f32.shape[0]), int(out_f32.shape[1]))

        # Simulation / Fallback Path (offline unit tests without physical weights)
        if isinstance(input_data, list):
            raw_bytes, tensor_shape = self.embed_tokens_to_raw_tensor(input_data)
        else:
            raw_bytes = bytes(input_data)
            if shape is not None:
                tensor_shape = shape
            else:
                total_floats = len(raw_bytes) // 4
                seq_len = max(1, total_floats // self.hidden_dim)
                tensor_shape = (seq_len, self.hidden_dim)

        if self._native_handle is not None and hasattr(self._native_handle, "_ctx"):
            ctx_ptr = getattr(self._native_handle._ctx, "ctx", None)
            if ctx_ptr is not None:
                try:
                    import llama_cpp  # type: ignore[import-not-found]

                    embd_ptr = llama_cpp.llama_get_embeddings(ctx_ptr)
                    if embd_ptr:
                        num_bytes = tensor_shape[0] * tensor_shape[1] * ctypes.sizeof(ctypes.c_float)
                        extracted: bytes = ctypes.string_at(embd_ptr, num_bytes)
                        return extracted, tensor_shape
                except Exception as exc:
                    logger.debug("Fallback from native llama_cpp embedding pointer: %s", exc)

        buf = bytearray(raw_bytes)
        if len(buf) >= 16:
            f0, f1, f2, f3 = struct.unpack_from("<ffff", buf, 0)
            layer_span = float(self.shard_layer_end - self.shard_layer_start + 1)
            scale = 1.0 + (layer_span / float(max(1, self.total_layers))) * 0.05
            struct.pack_into(
                "<ffff",
                buf,
                0,
                f0 * scale + 0.001 * self.shard_layer_start,
                f1 * scale + 0.001 * self.shard_layer_end,
                f2 * scale,
                f3 + layer_span * 0.01,
            )

        return bytes(buf), tensor_shape

    def decode_final_layer_to_token(
        self,
        hidden_state_bytes: bytes,
        step_index: int,
        max_tokens: int = 16,
        request_id: str = "default",
        temperature: float = 0.7,
    ) -> tuple[int, str, bool]:
        """Decode the final layer's output hidden-state tensor into a text token (Tail Node only)."""
        if self._is_real_pytorch:
            if self._norm is None or self._lm_head is None:
                raise RuntimeError("Tail node norm/lm_head is not loaded.")

            import torch  # type: ignore[import-not-found]

            with torch.inference_mode():
                num_floats = len(hidden_state_bytes) // 4
                seq_len = max(1, num_floats // self.hidden_dim)
                last_offset = (seq_len - 1) * self.hidden_dim * 4
                last_slice = hidden_state_bytes[last_offset : last_offset + self.hidden_dim * 4]
                last_hidden = (
                    torch.frombuffer(bytearray(last_slice), dtype=torch.float32)
                    .reshape(1, 1, self.hidden_dim)
                    .to(dtype=self._compute_dtype)
                )

                normed = self._norm(last_hidden)
                logits = self._lm_head(normed)[0, -1, :].to(torch.float32)

                session = self._sessions.get(request_id)
                if session is None:
                    session = self._get_or_create_session(request_id, step_index)

                if temperature <= 0.05:
                    token_id = int(torch.argmax(logits).item())
                else:
                    # Mild repetition penalty during sampling to prevent small models from looping
                    if session.generated_ids:
                        recent_ids = torch.tensor(
                            list(set(session.generated_ids[-32:])), dtype=torch.long
                        )
                        logits_at_recent = logits[recent_ids]
                        logits[recent_ids] = torch.where(
                            logits_at_recent > 0,
                            logits_at_recent / 1.12,
                            logits_at_recent * 1.12,
                        )
                    scaled = logits / float(temperature)
                    top_k = min(40, int(scaled.shape[-1]))
                    top_vals, top_idx = torch.topk(scaled, k=top_k)
                    probs = torch.softmax(top_vals, dim=-1)
                    sampled_pos = int(torch.multinomial(probs, num_samples=1).item())
                    token_id = int(top_idx[sampled_pos].item())

                is_eos: bool = (token_id in self._eos_token_ids) or (
                    (step_index + 1) >= max_tokens
                )

                session.generated_ids.append(token_id)
                if self._tokenizer is not None:
                    full_decoded: str = self._tokenizer.decode(
                        session.generated_ids,
                        skip_special_tokens=True,
                    )
                    token_text: str = full_decoded[len(session.prev_decoded_text) :]
                    session.prev_decoded_text = full_decoded
                else:
                    token_text = "" if token_id in self._eos_token_ids else f"<{token_id}>"

                if is_eos:
                    self.release_request_cache(request_id)

                return token_id, token_text, is_eos

        if len(hidden_state_bytes) < 4:
            return 2, "", True

        first_float: float = struct.unpack_from("<f", hidden_state_bytes, 0)[0]
        vocab_idx: int = (step_index + int(abs(first_float) * 10)) % len(_SYNTHETIC_VOCAB)
        ordered_idx: int = step_index % len(_SYNTHETIC_VOCAB)
        token_text = _SYNTHETIC_VOCAB[ordered_idx]
        token_id = 1000 + vocab_idx
        is_eos = (step_index + 1) >= max_tokens or (step_index + 1) >= len(_SYNTHETIC_VOCAB)
        return token_id, token_text, is_eos

    def release(self) -> None:
        """Explicitly release any native model handles and KV-caches to prevent RAM leaks."""
        self._sessions.clear()
        self._embed_tokens = None
        self._layers = []
        self._rotary_emb = None
        self._norm = None
        self._lm_head = None
        self._shard_container = None
        self._hf_config = None
        self._tokenizer = None
        self._is_real_pytorch = False

        if self._native_handle is not None:
            close_fn = getattr(self._native_handle, "close", None)
            if callable(close_fn):
                close_fn()
            self._native_handle = None

        gc.collect()
