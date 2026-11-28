"""Regression tests for Issue #73: QUIC connect-lock eviction and incoming connection limits."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

import pytest

import p2p.quic_transport as quic_transport_mod
from p2p.quic_transport import (
    MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS,
    MAX_CONNECT_LOCKS,
    QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS,
    IrohP2PTransport,
)
from p2p.transport import PeerUnreachableError


class _FakeConn:
    """Minimal fake QUIC connection for lock-lifecycle and pool eviction tests."""

    def __init__(self, remote_id: str, *, open_bi_error: Exception | None = None) -> None:
        self._remote_id = remote_id
        self._closed_reason: str | None = None
        self._close_event = asyncio.Event()
        self._open_bi_error = open_bi_error
        self.close_calls = 0

    def remote_id(self) -> str:
        return self._remote_id

    def close_reason(self) -> str | None:
        return self._closed_reason

    def close(self, _code: int = 0, _reason: bytes = b"") -> None:
        self.close_calls += 1
        self._closed_reason = "closed"
        self._close_event.set()

    async def accept_bi(self) -> Any:
        await self._close_event.wait()
        raise EOFError("connection closed")

    async def open_bi(self) -> Any:
        if self._open_bi_error is not None:
            raise self._open_bi_error
        raise AssertionError("open_bi unexpected in this test")


class _FakeEndpoint:
    """Minimal fake `iroh.Endpoint` driving `_accept_loop` and `_get_or_connect`."""

    def __init__(
        self,
        *,
        incoming_items: list[Any] | None = None,
        connect_impl: Any = None,
    ) -> None:
        self._incoming_queue: asyncio.Queue[Any] = asyncio.Queue()
        for item in incoming_items or ():
            self._incoming_queue.put_nowait(item)
        self._connect_impl = connect_impl
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    async def accept_next(self) -> Any:
        return await self._incoming_queue.get()

    async def connect(self, addr: Any, alpn: bytes) -> Any:
        if self._connect_impl is not None:
            return await self._connect_impl(addr, alpn)
        raise ConnectionError("dial failed")

    async def close(self) -> None:
        self._closed = True


class _RefusableIncoming:
    """Fake incoming connection implementing `accept()` and `refuse()`."""

    def __init__(
        self,
        *,
        remote_id: str = "peer-in",
        accept_gate: asyncio.Event | None = None,
        conn_hold_event: asyncio.Event | None = None,
    ) -> None:
        self._remote_id = remote_id
        self._accept_gate = accept_gate
        self._conn_hold_event = conn_hold_event
        self.accepted = False
        self.refused = False

    async def accept(self) -> _RefusableAccepting:
        self.accepted = True
        if self._accept_gate is not None:
            await self._accept_gate.wait()
        return _RefusableAccepting(self._remote_id, self._conn_hold_event)

    async def refuse(self) -> None:
        self.refused = True


class _RefusableAccepting:
    """Fake `iroh.Accepting` returning a fake connection that stays open until released."""

    def __init__(self, remote_id: str, conn_hold_event: asyncio.Event | None) -> None:
        self._remote_id = remote_id
        self._conn_hold_event = conn_hold_event

    async def connect(self) -> _FakeConn:
        conn = _FakeConn(self._remote_id)
        if self._conn_hold_event is not None:
            async def _close_on_signal() -> None:
                assert self._conn_hold_event is not None
                await self._conn_hold_event.wait()
                conn.close()

            asyncio.create_task(_close_on_signal())
        return conn


class _AcceptOnlyIncoming:
    """Fake incoming connection with only `accept()` (no `refuse`/`ignore`/`close`)."""

    def __init__(self) -> None:
        self.accept_calls = 0

    async def accept(self) -> Any:
        self.accept_calls += 1
        raise AssertionError("excess accept-only incoming should never call accept()")


def test_module_limit_constants_defined() -> None:
    """Verify required module-level QUIC safety constants exist with expected types/values."""
    assert MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS == 64
    assert MAX_CONNECT_LOCKS == 256
    assert QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS == 10.0


@pytest.mark.asyncio
async def test_connect_lock_evicted_when_reader_loop_finishes() -> None:
    """Per-peer `_connect_locks` entry is evicted when `_connection_reader_loop` exits."""
    transport = IrohP2PTransport()
    remote_id = "peer-disconnect-1"
    conn = _FakeConn(remote_id)
    transport._endpoint = _FakeEndpoint()
    transport._running = True
    transport._register_connection(remote_id, conn, replace=True)
    lock = transport._get_connect_lock(remote_id)
    assert transport._connect_locks.get(remote_id) is lock

    reader_task = asyncio.create_task(transport._connection_reader_loop(conn, remote_id))
    await asyncio.sleep(0)
    assert remote_id in transport._connect_locks

    conn.close()
    await reader_task

    assert remote_id not in transport._connections
    assert remote_id not in transport._connect_locks


@pytest.mark.asyncio
async def test_connect_lock_evicted_after_failed_outbound_dial_and_stale_retry() -> None:
    """Failed `_get_or_connect` and stale `_open_bi_with_retry` redial failures evict `_connect_locks`."""
    dial_attempts = 0

    async def _failing_connect(_addr: Any, _alpn: bytes) -> Any:
        nonlocal dial_attempts
        dial_attempts += 1
        await asyncio.sleep(0)
        raise ConnectionRefusedError("unreachable")

    transport = IrohP2PTransport()
    transport._endpoint = _FakeEndpoint(connect_impl=_failing_connect)
    transport._running = True

    with pytest.raises(PeerUnreachableError):
        await transport._get_or_connect("peer-fail-1", object(), connect_timeout=1.0)
    assert dial_attempts == 1
    assert "peer-fail-1" not in transport._connect_locks

    # Concurrent waiters on the same peer also leave zero leaked locks after failure
    results = await asyncio.gather(
        transport._get_or_connect("peer-concurrent-fail", object(), connect_timeout=1.0),
        transport._get_or_connect("peer-concurrent-fail", object(), connect_timeout=1.0),
        return_exceptions=True,
    )
    assert all(isinstance(res, PeerUnreachableError) for res in results)
    assert "peer-concurrent-fail" not in transport._connect_locks

    # Stale pooled connection whose open_bi fails and redial fails evicts both conn and lock
    stale_conn = _FakeConn("peer-stale", open_bi_error=RuntimeError("stream reset"))
    transport._register_connection("peer-stale", stale_conn, replace=True)
    transport._get_connect_lock("peer-stale")
    with pytest.raises(PeerUnreachableError):
        await transport._open_bi_with_retry("peer-stale", object(), connect_timeout=1.0)

    assert stale_conn.close_calls == 1
    assert "peer-stale" not in transport._connections
    assert "peer-stale" not in transport._connect_locks


@pytest.mark.asyncio
async def test_prune_idle_connect_locks_respects_cap_and_preserves_active_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idle pruning enforces `MAX_CONNECT_LOCKS` without removing locked or connected peers."""
    monkeypatch.setattr(quic_transport_mod, "MAX_CONNECT_LOCKS", 4)
    transport = IrohP2PTransport()

    # 1. Idle lock with no connection -> must be pruned
    transport._get_connect_lock("idle-no-conn")
    # 2. Idle lock with closed connection -> must be pruned
    closed_conn = _FakeConn("idle-closed-conn")
    closed_conn.close()
    transport._register_connection("idle-closed-conn", closed_conn, replace=True)
    transport._get_connect_lock("idle-closed-conn")
    # 3. Actively locked peer -> must NOT be pruned
    locked_lock = transport._get_connect_lock("active-locked")
    await locked_lock.acquire()
    # 4. Live open pooled connection -> must NOT be pruned
    live_conn = _FakeConn("active-open")
    transport._register_connection("active-open", live_conn, replace=True)
    open_lock = transport._get_connect_lock("active-open")

    try:
        assert len(transport._connect_locks) == 4
        # Direct cleanup checks must respect active lock and open connection
        transport._cleanup_connect_lock("active-locked")
        transport._cleanup_connect_lock("active-open")
        assert "active-locked" in transport._connect_locks
        assert "active-open" in transport._connect_locks

        # Requesting a 5th lock triggers idle pruning first
        new_lock = transport._get_connect_lock("peer-new")
        assert isinstance(new_lock, asyncio.Lock)
        assert "idle-no-conn" not in transport._connect_locks
        assert "idle-closed-conn" not in transport._connect_locks
        assert transport._connect_locks.get("active-locked") is locked_lock
        assert transport._connect_locks.get("active-open") is open_lock
        assert transport._connect_locks.get("peer-new") is new_lock
    finally:
        locked_lock.release()
        transport._cleanup_connect_lock("active-locked")
        assert "active-locked" not in transport._connect_locks


@pytest.mark.asyncio
async def test_accept_loop_refuses_excess_incoming_connections_at_cap(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`_accept_loop` drops/refuses excess incoming connections beyond the cap without task growth."""
    monkeypatch.setattr(quic_transport_mod, "MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS", 3)

    release_gate = asyncio.Event()
    endpoint = _FakeEndpoint()
    transport = IrohP2PTransport()
    transport._endpoint = endpoint
    transport._running = True

    # Pre-populate outbound `_conn_tasks` to ensure outbound tasks do not block incoming slots
    outbound_blocker = asyncio.create_task(release_gate.wait())
    transport._conn_tasks.add(outbound_blocker)

    caplog.set_level(logging.WARNING, logger="shardai.p2p.quic")
    accept_task = asyncio.create_task(transport._accept_loop())

    try:
        accepted_incomings = [
            _RefusableIncoming(
                remote_id=f"peer-in-{idx}",
                conn_hold_event=release_gate,
            )
            for idx in range(3)
        ]
        for inc in accepted_incomings:
            endpoint._incoming_queue.put_nowait(inc)

        # Wait until all 3 incoming connections are active in `_incoming_tasks`
        for _ in range(50):
            if len(transport._incoming_tasks) == 3:
                break
            await asyncio.sleep(0.01)
        assert len(transport._incoming_tasks) == 3
        assert all(inc.accepted and not inc.refused for inc in accepted_incomings)

        # Send excess incoming connections: one refusable, one accept-only (no refuse/close method)
        excess_refusable = _RefusableIncoming(remote_id="peer-excess-1")
        excess_accept_only = _AcceptOnlyIncoming()
        endpoint._incoming_queue.put_nowait(excess_refusable)
        endpoint._incoming_queue.put_nowait(excess_accept_only)

        # Allow loop to process both excess arrivals
        for _ in range(50):
            if excess_refusable.refused:
                break
            await asyncio.sleep(0.01)

        assert len(transport._incoming_tasks) == 3
        assert excess_refusable.refused is True
        assert excess_refusable.accepted is False
        assert excess_accept_only.accept_calls == 0
        assert any("active incoming limit (3) reached" in rec.message for rec in caplog.records)

        # Release active connections; slots drain and a new incoming connection is accepted
        release_gate.set()
        for _ in range(50):
            if len(transport._incoming_tasks) == 0:
                break
            await asyncio.sleep(0.01)
        assert len(transport._incoming_tasks) == 0

        hold_after = asyncio.Event()
        after_inc = _RefusableIncoming(remote_id="peer-after", conn_hold_event=hold_after)
        endpoint._incoming_queue.put_nowait(after_inc)
        for _ in range(50):
            if after_inc.accepted:
                break
            await asyncio.sleep(0.01)
        assert after_inc.accepted is True
        assert after_inc.refused is False
        hold_after.set()
    finally:
        release_gate.set()
        endpoint._incoming_queue.put_nowait(None)
        await accept_task
        transport._accept_task = None
        await transport.stop()


@pytest.mark.asyncio
async def test_accept_one_connection_times_out_stalled_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stalled `incoming.accept()` and `accepting.connect()` time out at handshake limit."""
    monkeypatch.setattr(quic_transport_mod, "QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS", 0.05)

    class _StalledAcceptIncoming:
        async def accept(self) -> Any:
            await asyncio.sleep(5.0)
            return None

    class _StalledConnectAccepting:
        async def connect(self) -> Any:
            await asyncio.sleep(5.0)
            return None

    class _StalledConnectIncoming:
        async def accept(self) -> _StalledConnectAccepting:
            return _StalledConnectAccepting()

    transport = IrohP2PTransport()
    transport._running = True

    for stalled_incoming in (_StalledAcceptIncoming(), _StalledConnectIncoming()):
        task = asyncio.create_task(transport._accept_one_connection(stalled_incoming))
        transport._incoming_tasks.add(task)
        task.add_done_callback(transport._incoming_tasks.discard)
        await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0)
        assert len(transport._incoming_tasks) == 0
        assert len(transport._connections) == 0
