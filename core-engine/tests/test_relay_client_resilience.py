"""Regression tests for `RelayClient` chunk scaling and auth failure reconnect backoff."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest
import websockets.asyncio.client as ws_client
from websockets.exceptions import InvalidStatus
from websockets.http11 import Headers
from websockets.http11 import Response as WsHttpResponse
from websockets.protocol import State

import p2p.relay_client as relay_client_mod
from p2p.relay_client import (
    DEFAULT_RELAY_SEND_CHUNK_BYTES,
    RELAY_RECONNECT_INITIAL_SECONDS,
    RELAY_RECONNECT_MAX_SECONDS,
    RelayClient,
    _resolve_send_chunk_size,
)
from p2p.relay_envelope import (
    MAX_CHUNKS,
    MAX_DATA_CHUNK_BYTES,
    RelayPacketKind,
    decode_envelope,
    encode_envelope,
)
from p2p.transport import MAX_WIRE_FRAME_BYTES


class _FakeSendWebSocket:
    """In-memory WebSocket double that captures outgoing relay envelopes and emits ACKs."""

    def __init__(self, sender: RelayClient, receiver: RelayClient) -> None:
        self.state = State.OPEN
        self._sender = sender
        self._receiver = receiver
        self.sent_envelopes_chunk_sizes: list[int] = []
        self.sent_chunk_counts: list[int] = []
        self.reassembled_frames: list[bytes] = []

    async def send(self, payload: bytes) -> None:
        env = decode_envelope(payload)
        assert env.kind == RelayPacketKind.DATA
        self.sent_envelopes_chunk_sizes.append(len(env.payload))
        self.sent_chunk_counts.append(env.chunk_count)

        completed, err = self._receiver._reassemble_data_chunk(env)
        assert err is None
        if completed is not None:
            self.reassembled_frames.append(completed)
            ack = encode_envelope(
                kind=RelayPacketKind.ACK,
                msg_id=env.msg_id,
                src=env.dst,
                dst=env.src,
            )
            await self._sender._handle_incoming_envelope(decode_envelope(ack))

    async def close(self) -> None:
        self.state = State.CLOSED


def _install_fast_backoff_wait_for(
    monkeypatch: pytest.MonkeyPatch,
    client: RelayClient,
    stop_after_waits: int,
) -> tuple[list[float], Any]:
    """Patch `asyncio.wait_for` to record `_stop_event` backoff timeouts without sleeping."""
    real_wait_for = asyncio.wait_for
    recorded_waits: list[float] = []

    def _is_backoff_stop_wait(awaitable: Any) -> bool:
        frame = getattr(awaitable, "cr_frame", None)
        return (
            frame is not None
            and asyncio.current_task() is client._runner_task
            and client._stop_event is not None
            and frame.f_locals.get("self") is client._stop_event
        )

    async def _fast_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        if _is_backoff_stop_wait(awaitable) and timeout is not None:
            recorded_waits.append(float(timeout))
            if hasattr(awaitable, "close"):
                awaitable.close()
            if len(recorded_waits) >= stop_after_waits:
                client._running = False
                if client._stop_event is not None:
                    client._stop_event.set()
                await asyncio.sleep(0)
                return True
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(relay_client_mod.asyncio, "wait_for", _fast_wait_for)
    return recorded_waits, real_wait_for


def test_resolve_send_chunk_size_bounds_and_scaling() -> None:
    """Chunk size stays at 128 KiB up to 32 MiB and scales up cleanly through 68 MiB."""
    assert _resolve_send_chunk_size(1) == DEFAULT_RELAY_SEND_CHUNK_BYTES
    assert (
        _resolve_send_chunk_size(DEFAULT_RELAY_SEND_CHUNK_BYTES * MAX_CHUNKS)
        == DEFAULT_RELAY_SEND_CHUNK_BYTES
    )

    for frame_len in (
        DEFAULT_RELAY_SEND_CHUNK_BYTES * MAX_CHUNKS + 1,
        33 * 1024 * 1024,
        MAX_WIRE_FRAME_BYTES,
    ):
        chunk_size = _resolve_send_chunk_size(frame_len)
        assert DEFAULT_RELAY_SEND_CHUNK_BYTES <= chunk_size <= MAX_DATA_CHUNK_BYTES
        assert math.ceil(frame_len / chunk_size) <= MAX_CHUNKS

    with pytest.raises(ValueError):
        _resolve_send_chunk_size(0)
    with pytest.raises(ValueError):
        _resolve_send_chunk_size(MAX_WIRE_FRAME_BYTES + 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame_len",
    [
        33 * 1024 * 1024,
        MAX_WIRE_FRAME_BYTES,
    ],
)
async def test_relay_client_send_frame_transmits_33mib_and_68mib_frames(
    frame_len: int,
) -> None:
    """`RelayClient.send_frame()` chunks and transmits valid 33 MiB and 68 MiB frames."""
    sender = RelayClient()
    receiver = RelayClient()
    fake_ws = _FakeSendWebSocket(sender, receiver)

    sender._running = True
    sender._node_id = "node-sender"
    sender._send_lock = asyncio.Lock()
    sender._ws = fake_ws  # type: ignore[assignment]

    frame = (b"SHRD\x01\x02\x03\x04" * (frame_len // 8 + 1))[:frame_len]
    rtt_ms = await sender.send_frame("node-receiver", frame, ack_timeout=10.0)

    assert rtt_ms >= 0.0
    chunk_count = len(fake_ws.sent_envelopes_chunk_sizes)
    assert 1 <= chunk_count <= MAX_CHUNKS
    assert all(count == chunk_count for count in fake_ws.sent_chunk_counts)
    assert all(size <= MAX_DATA_CHUNK_BYTES for size in fake_ws.sent_envelopes_chunk_sizes)
    assert len(fake_ws.reassembled_frames) == 1
    assert len(fake_ws.reassembled_frames[0]) == frame_len
    assert fake_ws.reassembled_frames[0] == frame


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_relay_client_auth_failure_allows_one_immediate_retry_then_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    """Persistent 401/403 with mutating token allows 1 immediate retry then exponential backoff."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    client = RelayClient()
    seen_tokens: list[str] = []
    refresh_count = 0

    class _RejectingConnect:
        def __init__(self, _url: str, **kwargs: Any) -> None:
            auth = kwargs.get("additional_headers", {}).get("Authorization", "")
            seen_tokens.append(auth)

        async def __aenter__(self) -> Any:
            raise InvalidStatus(
                WsHttpResponse(status_code, "Rejected", Headers(), b"auth rejected")
            )

        async def __aexit__(self, *_args: Any) -> None:
            return None

    async def _mutating_auth_refresh() -> None:
        nonlocal refresh_count
        refresh_count += 1
        client.update_credentials(
            "ws://127.0.0.1:8765/relay/room-1",
            f"refreshed-token-{refresh_count}",
        )

    monkeypatch.setattr(ws_client, "connect", _RejectingConnect)
    recorded_waits, real_wait_for = _install_fast_backoff_wait_for(
        monkeypatch, client, stop_after_waits=4
    )

    await client.start(
        url="ws://127.0.0.1:8765/relay/room-1",
        token="initial-token",
        node_id="node-a",
        on_frame=lambda _b, _s: asyncio.sleep(0),
        allowed_host="127.0.0.1",
        on_auth_failure=_mutating_auth_refresh,
    )
    if client._runner_task is not None:
        await real_wait_for(client._runner_task, timeout=2.0)
    await client.stop()

    # Attempt 1 fails -> immediate retry (attempt 2).
    # Attempts 2..5 fail -> 4 backoff sleeps: 1.0, 2.0, 4.0, 8.0.
    assert seen_tokens == [
        "Bearer initial-token",
        "Bearer refreshed-token-1",
        "Bearer refreshed-token-2",
        "Bearer refreshed-token-3",
        "Bearer refreshed-token-4",
    ]
    assert recorded_waits == [
        RELAY_RECONNECT_INITIAL_SECONDS,
        min(RELAY_RECONNECT_MAX_SECONDS, RELAY_RECONNECT_INITIAL_SECONDS * 2.0),
        min(RELAY_RECONNECT_MAX_SECONDS, RELAY_RECONNECT_INITIAL_SECONDS * 4.0),
        min(RELAY_RECONNECT_MAX_SECONDS, RELAY_RECONNECT_INITIAL_SECONDS * 8.0),
    ]


@pytest.mark.asyncio
async def test_relay_client_resets_auth_retry_budget_only_after_open_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a connection opens successfully, both backoff and the immediate auth retry reset."""
    monkeypatch.setenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "true")

    client = RelayClient()
    attempt = 0

    class _FakeOpenWs:
        state = State.OPEN

        def __aiter__(self) -> _FakeOpenWs:
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

        async def close(self) -> None:
            self.state = State.CLOSED

    class _ScriptedConnect:
        def __init__(self, _url: str, **_kwargs: Any) -> None:
            nonlocal attempt
            attempt += 1
            self._attempt = attempt

        async def __aenter__(self) -> _FakeOpenWs:
            # 1: 401 -> immediate retry
            # 2: 401 -> backoff wait #1 (1.0s)
            # 3: 401 -> backoff wait #2 (2.0s)
            # 4: open & close cleanly -> resets immediate retry budget & backoff, wait #3 (1.0s -> 2.0s)
            # 5: 401 -> fresh immediate retry allowed
            # 6: 401 -> backoff wait #4 (2.0s, vs 4.0s+ if not reset), then stop
            if self._attempt in (1, 2, 3, 5, 6):
                raise InvalidStatus(
                    WsHttpResponse(401, "Unauthorized", Headers(), b"bad token")
                )
            return _FakeOpenWs()

        async def __aexit__(self, *_args: Any) -> None:
            return None

    refresh_seq = 0

    async def _refresh_token() -> None:
        nonlocal refresh_seq
        refresh_seq += 1
        client.update_credentials(
            "ws://127.0.0.1:8765/relay/room-1",
            f"tok-{refresh_seq}",
        )

    monkeypatch.setattr(ws_client, "connect", _ScriptedConnect)
    recorded_waits, real_wait_for = _install_fast_backoff_wait_for(
        monkeypatch, client, stop_after_waits=4
    )

    await client.start(
        url="ws://127.0.0.1:8765/relay/room-1",
        token="tok-0",
        node_id="node-a",
        on_frame=lambda _b, _s: asyncio.sleep(0),
        allowed_host="127.0.0.1",
        on_auth_failure=_refresh_token,
    )
    if client._runner_task is not None:
        await real_wait_for(client._runner_task, timeout=2.0)
    await client.stop()

    assert attempt == 6
    assert recorded_waits == [1.0, 2.0, 1.0, 2.0]
