"""Regression tests for Issue #24: Online-to-LAN/Solo fallback and native Llama compute lock & cancellation."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

import main as main_module
from main import LocalNodeRuntime, app, runtime, start_node
from models.shard_runner import ShardModelRunner
from shard_engine.schemas import NetworkMode, NetworkStatus, NodeState, StartNodeRequest
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import patch_start_node_runner

client = TestClient(app)


def _make_mock_tracker(handler: Any) -> SignalingTrackerClient:
    return SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=1,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Restore runtime state before and after each test."""
    original_tracker = runtime.tracker_client
    monkeypatch.setenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "0")
    monkeypatch.setenv("SHARDAI_ENABLE_ZEROCONF", "false")
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []
    runtime.network_mode = None
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    yield
    if runtime.state != NodeState.STOPPED:
        client.post("/stop-node")
    runtime.tracker_client = original_tracker
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []
    runtime.network_mode = None
    runtime.network_status = None
    runtime.last_tracker_reachable = None


@pytest.mark.asyncio
async def test_online_node_falls_back_to_offline_solo_and_lan_only_on_outage_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tracker outage transitions ONLINE -> OFFLINE_SOLO / LAN_ONLY, drops stale cloud peers, and recovers."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    monkeypatch.setenv("SHARDAI_HEARTBEAT_INTERVAL_SECONDS", "60.0")

    fake_model = tmp_path / "fallback-model.gguf"
    fake_model.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fake_model),
    )

    tracker_up = True
    announce_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal announce_count
        if not tracker_up:
            raise httpx.ConnectError("Simulated internet disconnect", request=request)
        if request.method == "POST" and request.url.path == "/announce":
            announce_count += 1
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 15,
                    "role": "primary",
                    "peers": [
                        {
                            "node_id": "cloud-peer-01",
                            "ip": "198.51.100.10",
                            "port": 49510,
                            "assigned_start_layer": 16,
                            "assigned_end_layer": 31,
                            "role": "primary",
                        }
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/rooms":
            return httpx.Response(
                status_code=200,
                json={
                    "rooms": [
                        {
                            "room_id": "fallback-room",
                            "model": "meta-llama/Llama-3.1-8B-Instruct",
                            "nodes_online": 2,
                            "total_shared_ram_gb": 12.0,
                            "is_private": False,
                            "coverage_percent": 100.0,
                            "redundancy_factor": 1,
                        }
                    ]
                },
            )
        if request.method == "GET" and request.url.path.startswith("/peers"):
            return httpx.Response(
                status_code=200,
                json={
                    "peers": [
                        {
                            "node_id": "cloud-peer-01",
                            "ip": "198.51.100.10",
                            "port": 49510,
                            "assigned_start_layer": 16,
                            "assigned_end_layer": 31,
                            "role": "primary",
                        }
                    ]
                },
            )
        if request.method == "POST" and request.url.path == "/nodes/leave":
            return httpx.Response(status_code=200, json={"success": True})
        return httpx.Response(status_code=404)

    lan_beacons_to_return: list[dict[str, Any]] = []

    async def _fake_scan_lan(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return list(lan_beacons_to_return)

    monkeypatch.setattr(main_module, "_scan_all_lan_beacons", _fake_scan_lan)
    patch_start_node_runner(monkeypatch)
    runtime.tracker_client = _make_mock_tracker(_handler)

    start_resp = await start_node(
        StartNodeRequest(
            node_id="fallback-node-01",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            network_mode=NetworkMode.INTERNET,
            room_id="fallback-room",
            ram_gb=4,
            shared_ram_gb=4,
            max_ram_mb=4096,
            p2p_listen_port=49511,
            shard_layer_start=0,
            shard_layer_end=15,
            total_layers=32,
        )
    )
    assert start_resp.network_status == NetworkStatus.ONLINE
    assert runtime.last_tracker_reachable is True
    assert [p.peer_id for p in runtime.peers] == ["cloud-peer-01"]
    initial_announces = announce_count

    # 1. Tracker outage with zero LAN peers -> transitions to OFFLINE_SOLO and clears stale cloud peers
    tracker_up = False
    await runtime.refresh_room_coverage()
    assert runtime.network_status == NetworkStatus.OFFLINE_SOLO
    assert runtime.last_tracker_reachable is False
    assert runtime.peers == []

    # 2. Tracker still down, LAN peer appears -> transitions to LAN_ONLY with only LAN peer
    lan_beacons_to_return.append(
        {
            "node_id": "lan-peer-02",
            "room_id": "fallback-room",
            "ip_address": "192.168.1.55",
            "p2p_port": 49512,
            "shard_layer_start": 16,
            "shard_layer_end": 31,
            "role": "primary",
        }
    )
    await runtime.refresh_room_coverage()
    assert runtime.network_status == NetworkStatus.LAN_ONLY
    assert runtime.last_tracker_reachable is False
    assert [p.peer_id for p in runtime.peers] == ["lan-peer-02"]

    # 3. Tracker recovers -> restores ONLINE, re-announces, and restores cloud peers
    tracker_up = True
    lan_beacons_to_return.clear()
    await runtime.refresh_room_coverage()
    assert runtime.network_status == NetworkStatus.ONLINE
    assert runtime.last_tracker_reachable is True
    assert announce_count > initial_announces
    assert [p.peer_id for p in runtime.peers] == ["cloud-peer-01"]


@pytest.mark.asyncio
async def test_native_llama_streaming_serializes_concurrent_requests_via_compute_lock() -> None:
    """Concurrent `_stream_native_async` requests must serialize strictly via `_compute_lock`."""
    node = LocalNodeRuntime()
    node.state = NodeState.RUNNING
    node.shard_layer_start = 0
    node.shard_layer_end = 31
    node.total_layers = 32

    active_calls = 0
    max_concurrent = 0
    overlap_lock = threading.Lock()
    call_log: list[str] = []

    class _FakeLlamaHandle:
        def create_chat_completion(
            self,
            messages: list[dict[str, str]],
            max_tokens: int,
            temperature: float,
            stream: bool,
        ) -> Any:
            nonlocal active_calls, max_concurrent
            user_prompt = messages[0]["content"]
            with overlap_lock:
                active_calls += 1
                max_concurrent = max(max_concurrent, active_calls)
                call_log.append(f"start:{user_prompt}")
            try:
                for token in (f"{user_prompt}-1", f"{user_prompt}-2"):
                    time.sleep(0.03)
                    yield {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            finally:
                with overlap_lock:
                    call_log.append(f"end:{user_prompt}")
                    active_calls -= 1

    node.model_runner = ShardModelRunner(
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        native_handle=_FakeLlamaHandle(),
    )

    async def _collect_stream(prompt: str) -> list[str]:
        tokens: list[str] = []
        async for _rid, _step, _tid, text, _eos in node.stream_p2p_pipeline_tokens(
            prompt=prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
        ):
            if text:
                tokens.append(text)
        return tokens

    res_a, res_b = await asyncio.gather(_collect_stream("reqA"), _collect_stream("reqB"))
    assert res_a == ["reqA-1", "reqA-2"]
    assert res_b == ["reqB-1", "reqB-2"]
    assert max_concurrent == 1
    assert call_log in (
        ["start:reqA", "end:reqA", "start:reqB", "end:reqB"],
        ["start:reqB", "end:reqB", "start:reqA", "end:reqA"],
    )


@pytest.mark.asyncio
async def test_native_llama_stream_abort_cancels_worker_and_releases_compute_lock() -> None:
    """Closing an in-flight native stream stops the worker thread promptly and frees `_compute_lock`."""
    node = LocalNodeRuntime()
    node.state = NodeState.RUNNING
    node.shard_layer_start = 0
    node.shard_layer_end = 31
    node.total_layers = 32

    produced_steps: list[int] = []
    stream_closed = threading.Event()

    class _SlowNativeStream:
        def __init__(self, total_steps: int) -> None:
            self._total_steps = total_steps
            self._idx = 0

        def __iter__(self) -> "_SlowNativeStream":
            return self

        def __next__(self) -> dict[str, Any]:
            if self._idx >= self._total_steps:
                raise StopIteration
            step = self._idx
            self._idx += 1
            if step > 0:
                time.sleep(0.03)
            produced_steps.append(step)
            return {"choices": [{"text": f"tok-{step}", "finish_reason": None}]}

        def close(self) -> None:
            stream_closed.set()

    class _FakeLlamaHandle:
        def create_completion(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float,
            stream: bool,
        ) -> Any:
            return _SlowNativeStream(total_steps=100)

    node.model_runner = ShardModelRunner(
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        native_handle=_FakeLlamaHandle(),
    )

    gen = node.stream_p2p_pipeline_tokens(prompt="Abort early", max_tokens=100)
    first_item = await anext(gen)
    assert first_item[3] == "tok-0"

    await gen.aclose()
    assert stream_closed.is_set()
    assert len(produced_steps) < 15
    assert not node._compute_lock.locked()

    # Subsequent request acquires `_compute_lock` immediately without deadlock
    follow_up_gen = node.stream_p2p_pipeline_tokens(prompt="Follow up", max_tokens=2)
    second_item = await asyncio.wait_for(anext(follow_up_gen), timeout=2.0)
    assert second_item[3].startswith("tok-")
    await follow_up_gen.aclose()


def test_corrupt_gguf_raises_reload_model_error(tmp_path: Path) -> None:
    """Loading a corrupt >1 MB .gguf must raise RuntimeError instead of falling back."""
    corrupt_gguf = tmp_path / "corrupt-model.gguf"
    corrupt_gguf.write_bytes(b"GGUF" + b"\x00" * 1_100_000)

    with pytest.raises(RuntimeError, match="reload the model"):
        ShardModelRunner(
            model_id="test/corrupt-gguf",
            model_path=str(corrupt_gguf),
            shard_layer_start=0,
            shard_layer_end=15,
            total_layers=16,
            n_layers=16,
        )


def test_decode_without_loaded_weights_raises_reload_model_error() -> None:
    """decode_final_layer_to_token must never emit synthetic fallback tokens."""
    runner = ShardModelRunner(
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=16,
    )
    assert runner.is_real_model is False

    with pytest.raises(RuntimeError, match="reload the model"):
        runner.decode_final_layer_to_token(b"\x00" * 16, step_index=0, max_tokens=4)


def test_inference_endpoints_return_503_when_model_not_loaded() -> None:
    """/v1/chat/completions and /generate must return HTTP 503 asking to reload the model."""
    token = "sk-shard-unloaded-model-token"
    runtime.state = NodeState.RUNNING
    runtime.node_id = "unloaded-node"
    runtime.model_id = "test/unloaded-model"
    runtime.room_id = "public"
    runtime.session_token = token
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32
    runtime.n_layers = 32
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.peers = []
    runtime.model_runner = ShardModelRunner(
        model_id="test/unloaded-model",
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        n_layers=32,
    )
    headers = {"Authorization": f"Bearer {token}"}

    chat_res = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "test/unloaded-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert chat_res.status_code == 503
    assert "reload the model" in chat_res.json()["detail"].lower()

    gen_res = client.post(
        "/generate",
        headers=headers,
        json={"prompt": "Hello", "max_tokens": 4},
    )
    assert gen_res.status_code == 503
    assert "reload the model" in gen_res.json()["detail"].lower()


def test_start_node_corrupt_gguf_returns_500_and_sets_error_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """/start-node must return HTTP 500 with reload-model prompt and transition to ERROR on load failure."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")
    corrupt_gguf = tmp_path / "broken-weights.gguf"
    corrupt_gguf.write_bytes(b"GGUF" + b"\x00" * 1_100_000)

    monkeypatch.setattr(
        runtime.model_downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=corrupt_gguf),
    )

    res = client.post(
        "/start-node",
        json={
            "node_id": "corrupt-start-node",
            "model_id": "test/corrupt-gguf",
            "network_mode": "lan",
            "room_id": "public",
            "ram_gb": 4,
            "shared_ram_gb": 4,
            "max_ram_mb": 4096,
            "p2p_listen_port": 49599,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
            "total_layers": 16,
        },
    )
    assert res.status_code == 500
    assert "reload the model" in res.json()["detail"].lower()
    assert runtime.state == NodeState.ERROR
