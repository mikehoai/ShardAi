"""Regression tests for Issue #53: remove unsafe llama_cpp embedding reads and guard GGUF multi-node sharding."""

from __future__ import annotations

from array import array
import os
import sys
import types
from typing import Any
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from main import LocalNodeRuntime, app, runtime
from models.shard_runner import ShardModelRunner
from p2p.serializer import deserialize_activation_tensor, serialize_activation_tensor
from shard_engine.schemas import NodeState, PeerNodeInfo, PeerRole

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_global_runtime() -> Any:
    """Ensure clean global runtime state before and after each test."""
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.peers = []
    runtime.pending_token_queues.clear()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    yield
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.peers = []
    runtime.pending_token_queues.clear()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None


class _FakeLlamaWithCtx:
    """Fake native llama_cpp.Llama handle exposing `_ctx.ctx` and completion methods."""

    def __init__(self) -> None:
        self._ctx = types.SimpleNamespace(ctx=object())

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Any:
        _ = messages, max_tokens, temperature, stream
        yield {"choices": [{"delta": {"content": "solo-ok"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    def create_completion(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Any:
        _ = prompt, max_tokens, temperature, stream
        yield {"choices": [{"text": "solo-ok", "finish_reason": None}]}
        yield {"choices": [{"text": "", "finish_reason": "stop"}]}

    def close(self) -> None:
        return None


def test_forward_pass_never_reads_llama_get_embeddings_or_ctypes_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`forward_pass` must not call `llama_get_embeddings` and must reject native GGUF sharding."""
    called = False

    def _forbidden_get_embeddings(_ctx: Any) -> int:
        nonlocal called
        called = True
        return 0x1234

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.llama_get_embeddings = _forbidden_get_embeddings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)

    runner = ShardModelRunner(
        model_id="test/gguf-model",
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
        hidden_dim=4096,
        native_handle=_FakeLlamaWithCtx(),
    )
    assert runner.has_native_llama is True
    assert runner.supports_pipeline_sharding is False

    with pytest.raises(RuntimeError, match="Safetensors/PyTorch"):
        runner.forward_pass([1, 2, 3])
    assert called is False

    with pytest.raises(RuntimeError, match="Safetensors/PyTorch"):
        runner.forward_pass(b"\x00" * 64, shape=(1, 16))
    assert called is False


def test_forward_pass_offline_simulation_respects_tensor_shape_bounds() -> None:
    """Offline simulation `forward_pass` (no native/PyTorch handle) keeps exact shape byte size."""
    runner = ShardModelRunner(
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
        hidden_dim=16,
    )
    assert runner.has_native_llama is False
    raw_in, in_shape = runner.embed_tokens_to_raw_tensor([10, 20, 30])
    assert in_shape == (3, 16)
    assert len(raw_in) == 3 * 16 * 4

    out_bytes, out_shape = runner.forward_pass(raw_in, shape=in_shape)
    assert out_shape == (3, 16)
    assert len(out_bytes) == 3 * 16 * 4


def test_decode_final_layer_rejects_native_gguf_runner_explicitly() -> None:
    """`decode_final_layer_to_token` must explain GGUF/llama_cpp sharding is unsupported."""
    runner = ShardModelRunner(
        model_id="test/gguf-tail",
        shard_layer_start=16,
        shard_layer_end=31,
        total_layers=32,
        native_handle=_FakeLlamaWithCtx(),
    )
    with pytest.raises(RuntimeError, match="Safetensors/PyTorch"):
        runner.decode_final_layer_to_token(b"\x00" * 16, step_index=0, max_tokens=4)


def test_endpoints_reject_partial_gguf_head_node_with_422_before_sse() -> None:
    """Partial/multi-node GGUF Head Nodes must fail pre-flight with HTTP 422 before SSE."""
    token = "sk-shard-gguf-guard-test"
    runtime.state = NodeState.RUNNING
    runtime.node_id = "gguf-head-node"
    runtime.model_id = "bartowski/Llama-3.1-8B-Instruct-GGUF"
    runtime.room_id = "gguf-room"
    runtime.session_token = token
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.n_layers = 16
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.peers = [
        PeerNodeInfo(
            peer_id="gguf-tail-peer",
            ip_address="127.0.0.1",
            port=49901,
            shard_layer_start=16,
            shard_layer_end=31,
            role=PeerRole.PRIMARY,
        )
    ]
    runtime.model_runner = ShardModelRunner(
        model_id="bartowski/Llama-3.1-8B-Instruct-GGUF",
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
        n_layers=16,
        native_handle=_FakeLlamaWithCtx(),
    )
    headers = {"Authorization": f"Bearer {token}"}

    stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "bartowski/Llama-3.1-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 4,
        },
    )
    assert stream_res.status_code == 422
    assert "Safetensors/PyTorch" in stream_res.json()["detail"]
    assert "single-node GGUF" in stream_res.json()["detail"]

    non_stream_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "bartowski/Llama-3.1-8B-Instruct-GGUF",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "max_tokens": 4,
        },
    )
    assert non_stream_res.status_code == 422
    assert "Safetensors/PyTorch" in non_stream_res.json()["detail"]

    gen_res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 4},
    )
    assert gen_res.status_code == 422
    assert "Safetensors/PyTorch" in gen_res.json()["detail"]


def test_endpoints_reject_partial_gguf_without_peers_with_422() -> None:
    """Even before peers register, a partial-layer GGUF Head Node must fail with HTTP 422."""
    token = "sk-shard-gguf-partial-nopeer"
    runtime.state = NodeState.RUNNING
    runtime.node_id = "gguf-partial-head"
    runtime.model_id = "bartowski/Llama-3.1-8B-Instruct-GGUF"
    runtime.room_id = "gguf-room"
    runtime.session_token = token
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 15
    runtime.total_layers = 32
    runtime.n_layers = 16
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.peers = []
    runtime.model_runner = ShardModelRunner(
        model_id="bartowski/Llama-3.1-8B-Instruct-GGUF",
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
        n_layers=16,
        native_handle=_FakeLlamaWithCtx(),
    )
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 4},
    )
    assert res.status_code == 422
    assert "single-node GGUF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_stream_p2p_pipeline_tokens_allows_solo_gguf_and_rejects_multinode_gguf() -> None:
    """Single-node GGUF streams natively, while multi-node GGUF raises HTTP 422."""
    node = LocalNodeRuntime()
    node.state = NodeState.RUNNING
    node.shard_layer_start = 0
    node.shard_layer_end = 31
    node.total_layers = 32
    node.model_runner = ShardModelRunner(
        model_id="test/solo-gguf",
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        n_layers=32,
        native_handle=_FakeLlamaWithCtx(),
    )

    solo_tokens: list[str] = []
    async for _rid, _step, _tid, text, _eos in node.stream_p2p_pipeline_tokens(
        prompt="Hi",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=4,
    ):
        if text:
            solo_tokens.append(text)
    assert solo_tokens == ["solo-ok"]

    # Adding a downstream peer turns it into multi-node GGUF -> must reject with HTTP 422
    node.peers = [
        PeerNodeInfo(
            peer_id="unexpected-downstream",
            ip_address="127.0.0.1",
            port=49902,
            shard_layer_start=32,
            shard_layer_end=63,
            role=PeerRole.PRIMARY,
        )
    ]
    node.shard_layer_end = 15
    with pytest.raises(HTTPException) as exc_info:
        async for _ in node.stream_p2p_pipeline_tokens(prompt="Hi", max_tokens=4):
            pass
    assert exc_info.value.status_code == 422
    assert "Safetensors/PyTorch" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_handle_incoming_activation_rejects_gguf_tail_node() -> None:
    """Incoming P2P activation frames must be rejected cleanly on a native GGUF node."""
    tail = LocalNodeRuntime()
    tail.state = NodeState.RUNNING
    tail.shard_layer_start = 16
    tail.shard_layer_end = 31
    tail.total_layers = 32
    tail.model_runner = ShardModelRunner(
        model_id="test/gguf-tail",
        shard_layer_start=16,
        shard_layer_end=31,
        total_layers=32,
        n_layers=16,
        native_handle=_FakeLlamaWithCtx(),
    )
    tail.peers = [
        PeerNodeInfo(
            peer_id="head-peer",
            ip_address="127.0.0.1",
            port=49910,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
        )
    ]

    raw = serialize_activation_tensor(
        tensor_data=array("f", [0.1] * 16).tobytes(),
        shape=(1, 16),
        request_id=str(uuid.uuid4()),
        origin_ip="127.0.0.1",
        origin_port=49910,
        current_layer=15,
        total_layers=32,
        step_index=0,
        max_tokens=4,
    )
    frame = deserialize_activation_tensor(raw)

    with pytest.raises(RuntimeError, match="Safetensors/PyTorch"):
        await tail.handle_incoming_activation(frame)
