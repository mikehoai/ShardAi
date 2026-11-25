"""Test-only mock helpers for ShardModelRunner without synthetic production fallbacks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import main as main_module
from models.shard_runner import ShardModelRunner


class FakeNativeLlama:
    """Minimal fake `llama_cpp.Llama` yielding deterministic tokens for unit tests."""

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[dict[str, Any]]:
        _ = messages, max_tokens, temperature, stream
        yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    def create_completion(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[dict[str, Any]]:
        _ = prompt, max_tokens, temperature, stream
        yield {"choices": [{"text": "ok", "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}

    def close(self) -> None:
        return None


class _MockShardModelRunner(ShardModelRunner):
    """Test-only ShardModelRunner allowing mock P2P hidden-state simulation."""

    @property
    def supports_pipeline_sharding(self) -> bool:
        return True


def make_mock_runner(**kwargs: Any) -> ShardModelRunner:
    """Create a `ShardModelRunner` backed by `FakeNativeLlama` and a test decoder stub."""
    kwargs.pop("model_path", None)
    kwargs.setdefault("native_handle", FakeNativeLlama())
    runner = _MockShardModelRunner(**kwargs)

    def _test_decode(
        hidden_state_bytes: bytes,
        step_index: int,
        max_tokens: int = 16,
        request_id: str = "default",
        temperature: float = 0.7,
    ) -> tuple[int, str, bool]:
        _ = hidden_state_bytes, request_id, temperature
        is_eos = (step_index + 1) >= max_tokens
        return 1000 + step_index, f"tok{step_index}", is_eos

    runner.decode_final_layer_to_token = _test_decode  # type: ignore[method-assign]
    return runner


def patch_start_node_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch `main.ShardModelRunner` so `start_node` tests use `make_mock_runner`."""

    def _factory(**kwargs: Any) -> ShardModelRunner:
        return make_mock_runner(**kwargs)

    monkeypatch.setattr(main_module, "ShardModelRunner", _factory)
