"""Regression tests for Issue #30: Native Llama stream cancellation, bounded queue backpressure, and thread cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import aclosing
import json
import os
import threading
import time
from typing import Any

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import httpx
import pytest
from starlette.requests import ClientDisconnect

from main import (
    NATIVE_STREAM_CANCEL_JOIN_TIMEOUT_SECONDS,
    NATIVE_STREAM_QUEUE_MAXSIZE,
    app,
    runtime,
)
from models.shard_runner import ShardModelRunner
from shard_engine.schemas import ComputeBackend, NodeState, StartNodeRequest


class FakeNativeLlamaHandle:
    """Thread-safe fake `llama_cpp.Llama` handle tracking stream lifecycle and close races."""

    def __init__(
        self,
        total_tokens: int = 2048,
        chunk_delay_seconds: float = 0.0,
        fail_after_chunks: int | None = None,
        token_prefix: str = "tok",
    ) -> None:
        self.total_tokens: int = total_tokens
        self.chunk_delay_seconds: float = chunk_delay_seconds
        self.fail_after_chunks: int | None = fail_after_chunks
        self.token_prefix: str = token_prefix

        self._state_lock: threading.Lock = threading.Lock()
        self.calls: int = 0
        self.chat_calls: int = 0
        self.completion_calls: int = 0
        self.generated_chunks: int = 0
        self.generator_closed: bool = False
        self.close_called_count: int = 0
        self.close_during_decode: bool = False
        self.in_decode: bool = False
        self.active_calls: int = 0
        self.max_concurrent_calls: int = 0
        self.first_chunk_emitted_event: threading.Event = threading.Event()
        self.worker_exited_event: threading.Event = threading.Event()
        self.last_stop_step: int = -1

    def _stream_chunks(
        self,
        *,
        is_chat: bool,
        max_tokens: int,
    ) -> Iterator[dict[str, Any]]:
        limit = min(max_tokens, self.total_tokens)
        with self._state_lock:
            self.calls += 1
            if is_chat:
                self.chat_calls += 1
            else:
                self.completion_calls += 1
            self.active_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self.active_calls)
            self.in_decode = True
            self.generator_closed = False

        try:
            for idx in range(limit):
                if self.chunk_delay_seconds > 0:
                    time.sleep(self.chunk_delay_seconds)
                if self.fail_after_chunks is not None and idx >= self.fail_after_chunks:
                    raise RuntimeError("native decode failure")
                is_last = idx == (limit - 1)
                with self._state_lock:
                    self.generated_chunks += 1
                    self.last_stop_step = idx
                self.first_chunk_emitted_event.set()
                piece = f" {self.token_prefix}{idx}"
                if is_chat:
                    yield {
                        "choices": [
                            {
                                "delta": {"content": piece},
                                "finish_reason": "stop" if is_last else None,
                            }
                        ]
                    }
                else:
                    yield {
                        "choices": [
                            {
                                "text": piece,
                                "finish_reason": "stop" if is_last else None,
                            }
                        ]
                    }
        finally:
            with self._state_lock:
                self.in_decode = False
                self.active_calls -= 1
                self.generator_closed = True
            self.worker_exited_event.set()

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool = True,
    ) -> Iterator[dict[str, Any]]:
        assert stream is True
        assert len(messages) >= 1
        assert temperature >= 0.0
        return self._stream_chunks(is_chat=True, max_tokens=max_tokens)

    def create_completion(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool = True,
    ) -> Iterator[dict[str, Any]]:
        assert stream is True
        assert isinstance(prompt, str)
        assert temperature >= 0.0
        return self._stream_chunks(is_chat=False, max_tokens=max_tokens)

    def close(self) -> None:
        with self._state_lock:
            self.close_called_count += 1
            if self.in_decode:
                self.close_during_decode = True


def _reset_runtime_sync() -> None:
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.peers = []
    runtime.pending_token_queues.clear()
    runtime.active_request_steps.clear()
    runtime.native_stream_queue_maxsize = NATIVE_STREAM_QUEUE_MAXSIZE
    runtime.native_cancel_wait_timeout_seconds = NATIVE_STREAM_CANCEL_JOIN_TIMEOUT_SECONDS
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.model_id = None
    runtime.room_id = None
    runtime.room_key_hash = None


@pytest.fixture(autouse=True)
def clean_runtime_fixture() -> Iterator[None]:
    """Ensure clean runtime and memory state before and after each test."""
    _reset_runtime_sync()
    yield
    _reset_runtime_sync()


def _configure_native_runtime(
    handle: FakeNativeLlamaHandle,
    *,
    token: str = "sk-shard-issue30-token",
    allocate_memory: bool = False,
) -> dict[str, str]:
    runtime.state = NodeState.RUNNING
    runtime.node_id = "node-issue30-test"
    runtime.model_id = "test-native-llama-gguf"
    runtime.room_id = "issue30-room"
    runtime.session_token = token
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32
    runtime.n_layers = 32
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.peers = []
    runtime.model_runner = ShardModelRunner(
        model_id="test-native-llama-gguf",
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        n_layers=32,
        native_handle=handle,
    )
    if allocate_memory:
        runtime.memory_manager.allocate_shard(
            StartNodeRequest(
                node_id="node-issue30-test",
                model_id="test-native-llama-gguf",
                room_id="issue30-room",
                max_ram_mb=1024,
                max_vram_mb=512,
                shard_layer_start=0,
                shard_layer_end=31,
                total_layers=32,
                backend=ComputeBackend.LLAMA_CPP,
            ),
            estimated_ram_mb=512,
        )
    return {"Authorization": f"Bearer {token}"}


def test_stream_native_stops_early_and_closes_underlying_generator_chat_and_prompt() -> None:
    """`stream_native` must honor `stop_event` immediately and close underlying generator."""
    # 1. Pre-cancelled stop_event makes zero native calls
    pre_cancelled = threading.Event()
    pre_cancelled.set()
    pre_handle = FakeNativeLlamaHandle(total_tokens=100)
    runner_pre = ShardModelRunner(native_handle=pre_handle)
    tokens_pre = list(
        runner_pre.stream_native(
            prompt="ignored",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            stop_event=pre_cancelled,
        )
    )
    assert tokens_pre == []
    assert pre_handle.calls == 0

    # 2. Chat mode cancellation via stop_event after 3 tokens
    chat_handle = FakeNativeLlamaHandle(total_tokens=500)
    runner_chat = ShardModelRunner(native_handle=chat_handle)
    chat_stop = threading.Event()
    chat_out: list[tuple[int, str, bool]] = []
    for item in runner_chat.stream_native(
        prompt="hi",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=500,
        stop_event=chat_stop,
    ):
        chat_out.append(item)
        if len(chat_out) == 3:
            chat_stop.set()

    assert len(chat_out) == 3
    assert chat_handle.chat_calls == 1
    assert chat_handle.completion_calls == 0
    assert chat_handle.generated_chunks == 3
    assert chat_handle.generator_closed is True

    # 3. Prompt completion mode early break + explicit close
    prompt_handle = FakeNativeLlamaHandle(total_tokens=500)
    runner_prompt = ShardModelRunner(native_handle=prompt_handle)
    prompt_stop = threading.Event()
    gen = runner_prompt.stream_native(
        prompt="hello world",
        messages=None,
        max_tokens=500,
        stop_event=prompt_stop,
    )
    first_two = [next(gen), next(gen)]
    gen.close()

    assert len(first_two) == 2
    assert prompt_handle.completion_calls == 1
    assert prompt_handle.chat_calls == 0
    assert prompt_handle.generated_chunks == 2
    assert prompt_handle.generator_closed is True

    runner_pre.release()
    runner_chat.release()
    runner_prompt.release()


@pytest.mark.asyncio
async def test_stream_native_async_aborts_worker_thread_on_cancel_and_aclose() -> None:
    """Task cancellation and early generator `aclose` must abort worker threads immediately."""
    # Case A: Cancel consumer task after 5 tokens
    handle_a = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.005)
    _configure_native_runtime(handle_a)

    emitted_a: list[str] = []
    started_cancel = asyncio.Event()

    async def _consume() -> None:
        async for _step, text, _eos in runtime._stream_native_async(
            prompt="hello",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2048,
            temperature=0.7,
        ):
            emitted_a.append(text)
            if len(emitted_a) >= 5:
                started_cancel.set()
                await asyncio.sleep(10.0)

    consumer_task = asyncio.create_task(_consume())
    await asyncio.wait_for(started_cancel.wait(), timeout=2.0)
    consumer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer_task

    assert await asyncio.to_thread(handle_a.worker_exited_event.wait, 2.0)
    assert handle_a.generator_closed is True
    assert len(emitted_a) == 5
    assert handle_a.generated_chunks < 30

    # Case B: Early break inside `async with aclosing(...)`
    handle_b = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.005)
    _configure_native_runtime(handle_b)

    emitted_b: list[str] = []
    async with aclosing(
        runtime._stream_native_async(
            prompt="follow up",
            messages=None,
            max_tokens=2048,
            temperature=0.5,
        )
    ) as stream_b:
        async for _step, text, _eos in stream_b:
            emitted_b.append(text)
            if len(emitted_b) >= 4:
                break

    assert await asyncio.to_thread(handle_b.worker_exited_event.wait, 2.0)
    assert handle_b.generator_closed is True
    assert len(emitted_b) == 4
    assert handle_b.generated_chunks < 30

    # Case C: Follow-up normal completion still succeeds immediately
    handle_c = FakeNativeLlamaHandle(total_tokens=3)
    _configure_native_runtime(handle_c)
    emitted_c: list[str] = []
    async with aclosing(
        runtime._stream_native_async(
            prompt="final",
            messages=[{"role": "user", "content": "final"}],
            max_tokens=3,
            temperature=0.7,
        )
    ) as stream_c:
        async for _step, text, _eos in stream_c:
            emitted_c.append(text)

    assert len(emitted_c) == 3
    assert handle_c.generator_closed is True


@pytest.mark.asyncio
async def test_stream_native_async_enforces_bounded_queue_backpressure_and_unblocks_on_cancel() -> None:
    """Fast native producer must pause under bounded queue backpressure and unblock on cancel."""
    handle = FakeNativeLlamaHandle(total_tokens=1000, chunk_delay_seconds=0.0)
    _configure_native_runtime(handle)
    runtime.native_stream_queue_maxsize = 4

    received: list[str] = []
    async with aclosing(
        runtime._stream_native_async(
            prompt="fast",
            messages=[{"role": "user", "content": "fast"}],
            max_tokens=1000,
            temperature=0.7,
        )
    ) as stream:
        async for _step, text, _eos in stream:
            received.append(text)
            if len(received) == 1:
                # Allow worker to produce up to the bounded queue limit and block on condition
                await asyncio.sleep(0.08)
                assert handle.generated_chunks <= 7
                break

    assert await asyncio.to_thread(handle.worker_exited_event.wait, 2.0)
    assert handle.generator_closed is True
    assert handle.generated_chunks < 20


async def _drive_asgi_sse_disconnect(
    *,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    spec_version: str,
    disconnect_after_chunks: int = 5,
) -> list[dict[str, Any]]:
    """Run ASGI request directly against `app` and simulate client disconnect after N SSE chunks."""
    body_bytes = json.dumps(payload).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body_bytes)).encode("ascii")),
        (b"authorization", headers["Authorization"].encode("ascii")),
    ]
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", 8000),
    }

    request_sent = False
    disconnect_event = asyncio.Event()
    sent_messages: list[dict[str, Any]] = []
    content_chunk_count = 0

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        await disconnect_event.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal content_chunk_count
        sent_messages.append(message)
        if message["type"] != "http.response.body":
            return
        raw_body: bytes = message.get("body", b"")
        text = raw_body.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :].strip()
            if data_str == "[DONE]":
                continue
            parsed = json.loads(data_str)
            is_chat_delta = bool(
                parsed.get("choices", [{}])[0].get("delta", {}).get("content")
            )
            is_generate_token = "token" in parsed
            if is_chat_delta or is_generate_token:
                content_chunk_count += 1
                if content_chunk_count >= disconnect_after_chunks:
                    if spec_version >= "2.4":
                        raise OSError("Simulated client TCP disconnect (ASGI 2.4)")
                    disconnect_event.set()

    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
    except (ClientDisconnect, OSError):
        pass

    return sent_messages


@pytest.mark.asyncio
async def test_sse_chat_completions_and_generate_disconnect_stops_native_worker() -> None:
    """SSE client disconnects (ASGI 2.3 and 2.4+) must abort worker threads on both SSE routes."""
    # 1. /v1/chat/completions on ASGI 2.3 (http.disconnect)
    handle_23 = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.005)
    headers = _configure_native_runtime(handle_23)
    await _drive_asgi_sse_disconnect(
        path="/v1/chat/completions",
        payload={
            "model": "test-native-llama-gguf",
            "messages": [{"role": "user", "content": "Stream many tokens"}],
            "stream": True,
            "max_tokens": 2048,
        },
        headers=headers,
        spec_version="2.3",
        disconnect_after_chunks=5,
    )
    assert await asyncio.to_thread(handle_23.worker_exited_event.wait, 2.0)
    assert handle_23.generator_closed is True
    assert handle_23.generated_chunks < 30
    assert runtime.pending_token_queues == {}
    assert runtime.active_request_steps == {}

    # 2. /v1/chat/completions on ASGI 2.4 (OSError -> ClientDisconnect)
    handle_24 = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.005)
    headers = _configure_native_runtime(handle_24)
    await _drive_asgi_sse_disconnect(
        path="/v1/chat/completions",
        payload={
            "model": "test-native-llama-gguf",
            "messages": [{"role": "user", "content": "Stream many tokens"}],
            "stream": True,
            "max_tokens": 2048,
        },
        headers=headers,
        spec_version="2.4",
        disconnect_after_chunks=5,
    )
    assert await asyncio.to_thread(handle_24.worker_exited_event.wait, 2.0)
    assert handle_24.generator_closed is True
    assert handle_24.generated_chunks < 30
    assert runtime.pending_token_queues == {}
    assert runtime.active_request_steps == {}

    # 3. /generate on ASGI 2.3 (http.disconnect)
    handle_gen = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.005)
    headers = _configure_native_runtime(handle_gen)
    await _drive_asgi_sse_disconnect(
        path="/generate",
        payload={"prompt": "Stream raw tokens", "max_tokens": 2048},
        headers=headers,
        spec_version="2.3",
        disconnect_after_chunks=5,
    )
    assert await asyncio.to_thread(handle_gen.worker_exited_event.wait, 2.0)
    assert handle_gen.generator_closed is True
    assert handle_gen.generated_chunks < 30
    assert runtime.pending_token_queues == {}
    assert runtime.active_request_steps == {}

    # 4. Follow-up request succeeds cleanly with HTTP 200
    handle_followup = FakeNativeLlamaHandle(total_tokens=4)
    headers = _configure_native_runtime(handle_followup)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    ) as client:
        followup_res = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "test-native-llama-gguf",
                "messages": [{"role": "user", "content": "Short follow-up"}],
                "stream": False,
                "max_tokens": 4,
            },
        )
    assert followup_res.status_code == 200
    assert "tok0" in followup_res.json()["choices"][0]["message"]["content"]
    assert handle_followup.generator_closed is True


@pytest.mark.asyncio
async def test_stop_node_and_release_cancel_active_native_stream_without_close_race() -> None:
    """Calling `/stop-node` or `runner.release()` mid-stream must cancel stream before `close()`."""
    # Part A: POST /stop-node mid-stream
    handle_stop = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.01)
    _configure_native_runtime(handle_stop, allocate_memory=True)
    assert runtime.memory_manager.allocated_ram_mb > 0
    assert runtime.memory_manager.allocated_vram_mb > 0

    stop_event = threading.Event()
    collected: list[str] = []

    async def _run_stream() -> None:
        async with aclosing(
            runtime.stream_p2p_pipeline_tokens(
                prompt="long inference",
                max_tokens=2048,
                messages=[{"role": "user", "content": "long inference"}],
                stop_event=stop_event,
            )
        ) as token_stream:
            async for _req_id, _step, _tok_id, text, _eos in token_stream:
                collected.append(text)

    stream_task = asyncio.create_task(_run_stream())
    assert await asyncio.to_thread(handle_stop.first_chunk_emitted_event.wait, 2.0)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    ) as client:
        stop_res = await client.post("/stop-node")

    await asyncio.wait_for(stream_task, timeout=2.0)
    assert stop_res.status_code == 200
    assert runtime.state == NodeState.STOPPED
    assert runtime.model_runner is None
    assert runtime.memory_manager.allocated_ram_mb == 0
    assert runtime.memory_manager.allocated_vram_mb == 0
    assert handle_stop.close_called_count == 1
    assert handle_stop.close_during_decode is False
    assert handle_stop.generator_closed is True
    assert handle_stop.generated_chunks < 2048

    # Part B: Direct runner.release() while worker thread is actively streaming
    handle_direct = FakeNativeLlamaHandle(total_tokens=2048, chunk_delay_seconds=0.01)
    runner = ShardModelRunner(native_handle=handle_direct)
    direct_tokens: list[tuple[int, str, bool]] = []

    def _consume_sync() -> None:
        for item in runner.stream_native(prompt="direct", max_tokens=2048):
            direct_tokens.append(item)

    worker = threading.Thread(target=_consume_sync, daemon=True)
    worker.start()
    assert handle_direct.first_chunk_emitted_event.wait(timeout=2.0)

    runner.release()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert handle_direct.close_called_count == 1
    assert handle_direct.close_during_decode is False
    assert handle_direct.generator_closed is True
    assert handle_direct.generated_chunks < 2048


@pytest.mark.asyncio
async def test_stream_native_async_propagates_worker_exception_and_sse_error_chunk() -> None:
    """Mid-stream native engine exception must propagate cleanly and emit SSE error + [DONE]."""
    handle_async = FakeNativeLlamaHandle(total_tokens=50, fail_after_chunks=2)
    _configure_native_runtime(handle_async)

    seen_tokens: list[str] = []
    with pytest.raises(RuntimeError, match="native decode failure"):
        async with aclosing(
            runtime._stream_native_async(
                prompt="trigger error",
                messages=[{"role": "user", "content": "trigger error"}],
                max_tokens=50,
                temperature=0.7,
            )
        ) as stream:
            async for _step, text, _eos in stream:
                seen_tokens.append(text)

    assert len(seen_tokens) == 2
    assert await asyncio.to_thread(handle_async.worker_exited_event.wait, 2.0)
    assert handle_async.generator_closed is True

    # SSE endpoint must emit initial tokens, structured error JSON, and [DONE]
    handle_sse = FakeNativeLlamaHandle(total_tokens=50, fail_after_chunks=2)
    headers = _configure_native_runtime(handle_sse)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1:8000",
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "test-native-llama-gguf",
                "messages": [{"role": "user", "content": "trigger sse error"}],
                "stream": True,
                "max_tokens": 50,
            },
        )

    assert res.status_code == 200
    events: list[dict[str, Any]] = []
    saw_done = False
    for line in res.text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[len("data: ") :].strip()
        if raw == "[DONE]":
            saw_done = True
            continue
        events.append(json.loads(raw))

    assert saw_done is True
    assert any(
        "error" in ev and "native decode failure" in ev["error"]["message"]
        for ev in events
    )
    assert handle_sse.generator_closed is True


@pytest.mark.asyncio
async def test_stream_native_concurrent_requests_serialized_by_native_lock() -> None:
    """Concurrent native streams sharing one `ShardModelRunner` must serialize via `_native_lock`."""
    handle = FakeNativeLlamaHandle(total_tokens=6, chunk_delay_seconds=0.01)
    _configure_native_runtime(handle)

    async def _run_one(label: str) -> list[str]:
        tokens: list[str] = []
        async with aclosing(
            runtime._stream_native_async(
                prompt=label,
                messages=[{"role": "user", "content": label}],
                max_tokens=6,
                temperature=0.7,
            )
        ) as stream:
            async for _step, text, _eos in stream:
                tokens.append(text)
        return tokens

    res_a, res_b = await asyncio.gather(_run_one("req-a"), _run_one("req-b"))
    assert len(res_a) == 6
    assert len(res_b) == 6
    assert handle.calls == 2
    assert handle.max_concurrent_calls == 1
    assert handle.generator_closed is True
