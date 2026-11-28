"""Production Iroh QUIC P2P Transport with UDP Hole Punching and E2EE Relay Fallback.

Provides NAT-traversed peer-to-peer binary frame delivery over QUIC streams:
- Uses `iroh` (`Endpoint`, `EndpointTicket`, `open_bi`/`accept_bi`) for automatic
  UDP hole punching across CGNAT/home routers, TLS 1.3 peer authentication by
  cryptographic `EndpointId`, and encrypted relay fallback when UDP hole punching
  is blocked by symmetric NATs or strict firewalls.
- Pools one persistent QUIC connection per remote `EndpointId` (reusing both
  outbound and inbound connections) and multiplexes each P2P binary frame over a
  lightweight bidirectional QUIC stream (`4-byte length + frame` -> `ACK/NACK`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
import logging
import os
import time
from typing import Any

from p2p.relay_envelope import MAX_NACK_REASON_BYTES, truncate_utf8_bytes
from p2p.transport import (
    ACK_BYTE,
    FRAME_LEN_STRUCT,
    MAX_WIRE_FRAME_BYTES,
    NACK_BYTE,
    PeerUnreachableError,
    congestion_monitor,
)

logger = logging.getLogger("shardai.p2p.quic")

try:
    import iroh  # type: ignore[import-not-found]

    IROH_AVAILABLE: bool = True
except ImportError:
    iroh = None  # type: ignore[assignment]
    IROH_AVAILABLE: bool = False

SHARDAI_QUIC_ALPN: bytes = b"shardai/p2p/1"
MAX_CONCURRENT_QUIC_STREAMS: int = 64
MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS: int = 64
MAX_CONNECT_LOCKS: int = 256
MAX_QUIC_NACK_REASON_BYTES: int = MAX_NACK_REASON_BYTES
QUIC_CONNECT_TIMEOUT_SECONDS: float = 12.0
QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS: float = 10.0
QUIC_ONLINE_WAIT_SECONDS: float = 4.0
QUIC_STREAM_IO_TIMEOUT_SECONDS: float = 45.0


def is_quic_p2p_enabled() -> bool:
    """Return True when `iroh` is installed and `SHARDAI_P2P_QUIC` is not disabled."""
    if not IROH_AVAILABLE:
        return False
    raw = os.getenv("SHARDAI_P2P_QUIC", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _build_relay_mode(relay_mode: str | None) -> Any:
    """Resolve the `iroh.RelayMode` from an explicit mode string or environment."""
    if not IROH_AVAILABLE or iroh is None:
        raise RuntimeError("iroh package is not installed")

    env_urls = os.getenv("SHARDAI_IROH_RELAY_URLS", "").strip()
    if env_urls:
        urls = [u.strip() for u in env_urls.split(",") if u.strip()]
        if urls:
            return iroh.RelayMode.custom_from_urls(urls)

    effective = (
        relay_mode
        if relay_mode is not None
        else os.getenv("SHARDAI_IROH_RELAY_MODE", "default")
    ).strip().lower()

    if effective in {"disabled", "disable", "off", "lan"}:
        return iroh.RelayMode.disabled()
    if effective == "staging":
        return iroh.RelayMode.staging()
    return iroh.RelayMode.default_mode()


def _is_connection_closed(conn: Any) -> bool:
    """Return True when an `iroh.Connection` reports a terminal close reason."""
    if conn is None:
        return True
    try:
        return conn.close_reason() is not None
    except Exception:
        return True


async def _refuse_incoming(incoming: Any) -> None:
    """Best-effort refusal or drop of an excess incoming QUIC connection."""
    for method_name in ("refuse", "ignore", "close"):
        method = getattr(incoming, method_name, None)
        if callable(method):
            try:
                result = method()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=1.0)
            except Exception:
                pass
            return


def _resolve_endpoint_addr(peer_ticket_or_id: str) -> tuple[str, Any]:
    """Decode an `EndpointTicket` string into `(endpoint_id_str, EndpointAddr)`."""
    if not IROH_AVAILABLE or iroh is None:
        raise PeerUnreachableError("iroh QUIC transport is unavailable")
    cleaned = peer_ticket_or_id.strip()
    if not cleaned:
        raise PeerUnreachableError("Missing QUIC peer ticket")
    try:
        ticket = iroh.EndpointTicket.from_string(cleaned)
        addr = ticket.endpoint_addr()
        return str(addr.id()), addr
    except Exception as exc:
        raise PeerUnreachableError(
            f"Invalid QUIC EndpointTicket ({exc.__class__.__name__}: {exc})"
        ) from exc


class IrohP2PTransport:
    """Manages an `iroh.Endpoint`, connection pool, and QUIC bi-stream framing."""

    def __init__(self) -> None:
        self._endpoint: Any | None = None
        self._endpoint_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_frame: Callable[[bytes, str], Awaitable[None]] | None = None
        self._running: bool = False
        self._accept_task: asyncio.Task[None] | None = None
        self._incoming_tasks: set[asyncio.Task[None]] = set()
        self._conn_tasks: set[asyncio.Task[None]] = set()
        self._stream_tasks: set[asyncio.Task[None]] = set()
        self._stream_semaphore: asyncio.Semaphore | None = None
        self._connections: dict[str, Any] = {}
        self._connect_locks: dict[str, asyncio.Lock] = {}

    @property
    def is_running(self) -> bool:
        """Return True when the QUIC endpoint is bound and accepting streams."""
        return (
            self._running
            and self._endpoint is not None
            and not bool(self._endpoint.is_closed())
        )

    @property
    def endpoint_id(self) -> str | None:
        """Return the local cryptographic `EndpointId` string when running."""
        return self._endpoint_id if self.is_running else None

    def get_ticket(self) -> str | None:
        """Build a fresh base32 `EndpointTicket` carrying current direct & relay addresses."""
        if not self.is_running or self._endpoint is None or iroh is None:
            return None
        try:
            addr = self._endpoint.addr()
            return str(iroh.EndpointTicket.from_addr(addr))
        except Exception as exc:
            logger.debug("Failed to serialize Iroh EndpointTicket: %s", exc)
            return None

    def get_peer_path_kind(self, endpoint_id: str) -> str | None:
        """Return `'direct_quic'` or `'relay_quic'` for a pooled peer connection, else `None`."""
        conn = self._connections.get(endpoint_id)
        if _is_connection_closed(conn):
            return None
        try:
            paths = conn.paths()
        except Exception:
            return None

        selected_paths = [p for p in paths if getattr(p, "is_selected", False)]
        candidates = selected_paths if selected_paths else paths
        for path in candidates:
            if getattr(path, "is_ip", False):
                return "direct_quic"
        for path in candidates:
            if getattr(path, "is_relay", False):
                return "relay_quic"
        return None

    def get_direct_peers_count(self) -> int:
        """Return the number of active pooled peers using a direct UDP QUIC path."""
        count = 0
        for endpoint_id, conn in list(self._connections.items()):
            if _is_connection_closed(conn):
                continue
            if self.get_peer_path_kind(endpoint_id) == "direct_quic":
                count += 1
        return count

    async def start(
        self,
        on_frame: Callable[[bytes, str], Awaitable[None]],
        *,
        relay_mode: str | None = None,
        wait_online_seconds: float = QUIC_ONLINE_WAIT_SECONDS,
    ) -> tuple[str, str]:
        """Bind the Iroh QUIC endpoint, optionally wait for relay registration, and start accepting."""
        if not IROH_AVAILABLE or iroh is None:
            raise RuntimeError("iroh package is not installed")

        if self.is_running:
            self._on_frame = on_frame
            ticket = self.get_ticket()
            assert self._endpoint_id is not None and ticket is not None
            return self._endpoint_id, ticket

        await self.stop()

        loop = asyncio.get_running_loop()
        # Keep _UNIFFI_GLOBAL_EVENT_LOOP unset so every coroutine resolves its active loop safely.
        iroh.iroh_ffi.uniffi_set_event_loop(None)

        resolved_relay = _build_relay_mode(relay_mode)
        preset = (
            iroh.preset_n0_disable_relay()
            if (relay_mode or "").strip().lower() in {"disabled", "disable", "off", "lan"}
            else iroh.preset_n0()
        )
        options = iroh.EndpointOptions(
            preset=preset,
            alpns=[SHARDAI_QUIC_ALPN],
            relay_mode=resolved_relay,
        )
        ep = await iroh.Endpoint.bind(options)
        self._endpoint = ep
        self._endpoint_id = str(ep.id())
        self._loop = loop
        self._on_frame = on_frame
        self._stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUIC_STREAMS)
        self._running = True

        effective_relay = (relay_mode or os.getenv("SHARDAI_IROH_RELAY_MODE", "default")).strip().lower()
        if effective_relay not in {"disabled", "disable", "off", "lan"} and wait_online_seconds > 0:
            try:
                await asyncio.wait_for(ep.online(), timeout=wait_online_seconds)
            except asyncio.TimeoutError:
                logger.debug(
                    "Iroh endpoint %s timed out waiting %.1fs for home relay; continuing in background.",
                    self._endpoint_id,
                    wait_online_seconds,
                )
            except Exception as exc:
                logger.debug("Iroh online() notice: %s", exc)

        self._accept_task = asyncio.create_task(self._accept_loop())
        ticket = self.get_ticket() or ""
        logger.info(
            "Iroh QUIC P2P transport active (endpoint_id=%s, sockets=%s)",
            self._endpoint_id,
            ep.bound_sockets(),
        )
        return self._endpoint_id, ticket

    async def stop(self) -> None:
        """Stop accepting connections, close pooled QUIC sessions, and shut down the endpoint."""
        self._running = False

        if self._accept_task is not None:
            if not self._accept_task.done():
                self._accept_task.cancel()
            try:
                await self._accept_task
            except (asyncio.CancelledError, Exception):
                pass
            self._accept_task = None

        for task_set in (self._stream_tasks, self._incoming_tasks, self._conn_tasks):
            tasks = list(task_set)
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            task_set.clear()

        for conn in list(self._connections.values()):
            try:
                conn.close(0, b"shutdown")
            except Exception:
                pass
        self._connections.clear()
        self._connect_locks.clear()

        ep = self._endpoint
        ep_loop = self._loop
        self._endpoint = None
        self._endpoint_id = None
        self._loop = None
        self._on_frame = None
        self._stream_semaphore = None
        if ep is not None:
            try:
                running_loop = asyncio.get_running_loop()
                same_loop = ep_loop is None or ep_loop is running_loop
                if same_loop and not ep.is_closed():
                    await asyncio.wait_for(ep.close(), timeout=3.0)
                elif ep_loop is not None and not ep_loop.is_closed() and not ep.is_closed():
                    asyncio.run_coroutine_threadsafe(ep.close(), ep_loop)
            except Exception as exc:
                logger.debug("Error closing Iroh endpoint: %s", exc)

    def _cleanup_connect_lock(self, remote_id: str) -> None:
        """Evict an idle per-peer connect lock when no live pooled connection exists."""
        lock = self._connect_locks.get(remote_id)
        if lock is None or lock.locked():
            return
        if getattr(lock, "_waiters", None):
            return
        conn = self._connections.get(remote_id)
        if conn is not None and not _is_connection_closed(conn):
            return
        self._connect_locks.pop(remote_id, None)

    def _prune_idle_connect_locks(self) -> None:
        """Remove unlocked connect locks that have no open pooled connection."""
        for peer_id in list(self._connect_locks):
            self._cleanup_connect_lock(peer_id)

    def _get_connect_lock(self, remote_id: str) -> asyncio.Lock:
        lock = self._connect_locks.get(remote_id)
        if lock is None:
            if len(self._connect_locks) >= MAX_CONNECT_LOCKS:
                self._prune_idle_connect_locks()
            lock = asyncio.Lock()
            self._connect_locks[remote_id] = lock
        return lock

    def _register_connection(self, remote_id: str, conn: Any, *, replace: bool = False) -> None:
        existing = self._connections.get(remote_id)
        if existing is None or _is_connection_closed(existing) or replace:
            self._connections[remote_id] = conn

    async def _accept_loop(self) -> None:
        """Accept incoming QUIC connections on the bound endpoint."""
        while self._running and self._endpoint is not None:
            try:
                incoming = await self._endpoint.accept_next()
                if incoming is None:
                    break
                if len(self._incoming_tasks) >= MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS:
                    logger.warning(
                        "Rejecting incoming QUIC connection: active incoming limit (%d) reached",
                        MAX_CONCURRENT_INCOMING_QUIC_CONNECTIONS,
                    )
                    await _refuse_incoming(incoming)
                    continue
                task = asyncio.create_task(self._accept_one_connection(incoming))
                self._incoming_tasks.add(task)
                task.add_done_callback(self._incoming_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running or self._endpoint is None or self._endpoint.is_closed():
                    break
                logger.warning("Error accepting incoming Iroh QUIC connection: %s", exc)
                await asyncio.sleep(0.05)

    async def _accept_one_connection(self, incoming: Any) -> None:
        """Complete QUIC handshake for an incoming connection and serve its streams."""
        async def _complete_handshake() -> Any:
            accepting = await incoming.accept()
            return await accepting.connect()

        try:
            conn = await asyncio.wait_for(
                _complete_handshake(),
                timeout=QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError):
            logger.debug(
                "Incoming QUIC handshake timed out after %.1fs",
                QUIC_INCOMING_HANDSHAKE_TIMEOUT_SECONDS,
            )
            return
        except Exception as exc:
            logger.debug("Incoming QUIC handshake failed: %s", exc)
            return

        remote_id = str(conn.remote_id())
        self._register_connection(remote_id, conn, replace=False)
        await self._connection_reader_loop(conn, remote_id)

    async def _connection_reader_loop(self, conn: Any, remote_id: str) -> None:
        """Accept incoming bidirectional QUIC streams on an established connection."""
        try:
            while self._running and not _is_connection_closed(conn):
                bi = await conn.accept_bi()
                task = asyncio.create_task(self._handle_bi_stream(bi, remote_id))
                self._stream_tasks.add(task)
                task.add_done_callback(self._stream_tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running and not _is_connection_closed(conn):
                logger.debug("QUIC stream loop ended for %s: %s", remote_id, exc)
        finally:
            if self._connections.get(remote_id) is conn:
                self._connections.pop(remote_id, None)
            self._cleanup_connect_lock(remote_id)

    async def _handle_bi_stream(self, bi: Any, remote_id: str) -> None:
        """Read one length-prefixed frame, invoke `on_frame`, and reply with ACK or NACK."""
        recv = bi.recv()
        send = bi.send()
        sem = self._stream_semaphore
        if sem is not None and sem.locked() and getattr(sem, "_value", 1) <= 0:
            try:
                await send.write_all(NACK_BYTE + b"QUIC receiver busy (max concurrent streams)")
                await send.finish()
            except Exception:
                pass
            return

        async def _process() -> None:
            try:
                len_bytes = await asyncio.wait_for(
                    recv.read_exact(FRAME_LEN_STRUCT.size),
                    timeout=QUIC_STREAM_IO_TIMEOUT_SECONDS,
                )
                (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
                if frame_len <= 0 or frame_len > MAX_WIRE_FRAME_BYTES:
                    raise ValueError(f"Rejected invalid QUIC frame length: {frame_len} bytes")

                frame_bytes = await asyncio.wait_for(
                    recv.read_exact(frame_len),
                    timeout=QUIC_STREAM_IO_TIMEOUT_SECONDS,
                )

                on_frame = self._on_frame
                if on_frame is None:
                    raise RuntimeError("No QUIC frame handler registered")

                await on_frame(bytes(frame_bytes), remote_id)
                await asyncio.wait_for(
                    send.write_all(ACK_BYTE),
                    timeout=QUIC_STREAM_IO_TIMEOUT_SECONDS,
                )
                await asyncio.wait_for(
                    send.finish(),
                    timeout=QUIC_STREAM_IO_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("QUIC incoming frame error from %s: %s", remote_id, exc)
                reason = truncate_utf8_bytes(
                    str(exc) or exc.__class__.__name__,
                    MAX_QUIC_NACK_REASON_BYTES,
                )
                try:
                    await send.write_all(NACK_BYTE + reason)
                    await send.finish()
                except Exception:
                    pass

        if sem is not None:
            async with sem:
                await _process()
        else:
            await _process()

    async def _get_or_connect(
        self,
        remote_id: str,
        addr: Any,
        *,
        connect_timeout: float,
    ) -> Any:
        """Return an open pooled QUIC connection or dial a fresh one with hole punching."""
        existing = self._connections.get(remote_id)
        if not _is_connection_closed(existing):
            return existing

        lock = self._get_connect_lock(remote_id)
        try:
            async with lock:
                existing = self._connections.get(remote_id)
                if not _is_connection_closed(existing):
                    return existing
                if existing is not None and self._connections.get(remote_id) is existing:
                    self._connections.pop(remote_id, None)

                if not self.is_running or self._endpoint is None:
                    raise PeerUnreachableError("Iroh QUIC transport is not running")

                try:
                    conn = await asyncio.wait_for(
                        self._endpoint.connect(addr, SHARDAI_QUIC_ALPN),
                        timeout=connect_timeout,
                    )
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise PeerUnreachableError(
                        f"QUIC connect to {remote_id} timed out after {connect_timeout:.1f}s"
                    ) from exc
                except Exception as exc:
                    raise PeerUnreachableError(
                        f"Unable to establish QUIC connection to {remote_id}: {exc}"
                    ) from exc

                self._register_connection(remote_id, conn, replace=True)
                reader_task = asyncio.create_task(self._connection_reader_loop(conn, remote_id))
                self._conn_tasks.add(reader_task)
                reader_task.add_done_callback(self._conn_tasks.discard)
                return conn
        finally:
            self._cleanup_connect_lock(remote_id)

    async def _open_bi_with_retry(
        self,
        remote_id: str,
        addr: Any,
        *,
        connect_timeout: float,
    ) -> Any:
        """Open a bidirectional stream, redialing once if a pooled connection went stale."""
        conn = await self._get_or_connect(remote_id, addr, connect_timeout=connect_timeout)
        try:
            return await asyncio.wait_for(conn.open_bi(), timeout=connect_timeout)
        except Exception:
            if self._connections.get(remote_id) is conn:
                self._connections.pop(remote_id, None)
            try:
                conn.close(0, b"stale")
            except Exception:
                pass
            self._cleanup_connect_lock(remote_id)
            fresh_conn = await self._get_or_connect(
                remote_id, addr, connect_timeout=connect_timeout
            )
            try:
                return await asyncio.wait_for(fresh_conn.open_bi(), timeout=connect_timeout)
            except Exception as exc:
                if (
                    _is_connection_closed(fresh_conn)
                    and self._connections.get(remote_id) is fresh_conn
                ):
                    self._connections.pop(remote_id, None)
                self._cleanup_connect_lock(remote_id)
                raise PeerUnreachableError(
                    f"Failed opening QUIC stream to {remote_id}: {exc}"
                ) from exc

    async def send_frame(
        self,
        peer_ticket_or_id: str,
        frame: bytes,
        *,
        connect_timeout: float = QUIC_CONNECT_TIMEOUT_SECONDS,
        ack_timeout: float = 60.0,
    ) -> float:
        """Send a binary P2P frame over QUIC to `peer_ticket_or_id` and await ACK/NACK."""
        if not self.is_running:
            raise PeerUnreachableError("Iroh QUIC transport is not running")
        frame_len = len(frame)
        if frame_len <= 0 or frame_len > MAX_WIRE_FRAME_BYTES:
            raise ValueError(f"Invalid QUIC frame length: {frame_len} bytes")

        remote_id, addr = _resolve_endpoint_addr(peer_ticket_or_id)
        start_ns = time.perf_counter_ns()
        bi = await self._open_bi_with_retry(
            remote_id, addr, connect_timeout=connect_timeout
        )
        send = bi.send()
        recv = bi.recv()

        payload = FRAME_LEN_STRUCT.pack(frame_len) + frame
        try:
            await asyncio.wait_for(send.write_all(payload), timeout=max(connect_timeout, ack_timeout))
            await asyncio.wait_for(send.finish(), timeout=max(connect_timeout, ack_timeout))
        except Exception as exc:
            raise PeerUnreachableError(
                f"QUIC stream write failed for {remote_id}: {exc}"
            ) from exc

        try:
            ack = await asyncio.wait_for(recv.read_exact(1), timeout=ack_timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise RuntimeError(
                f"Timed out awaiting QUIC ACK from peer {remote_id} after {ack_timeout:.1f}s"
            ) from exc
        except Exception as exc:
            raise PeerUnreachableError(
                f"QUIC peer {remote_id} closed stream before sending ACK: {exc}"
            ) from exc

        if ack == NACK_BYTE:
            reason_raw = b""
            try:
                reason_raw = await asyncio.wait_for(
                    recv.read_to_end(MAX_QUIC_NACK_REASON_BYTES),
                    timeout=2.0,
                )
            except Exception:
                pass
            reason = reason_raw.decode("utf-8", errors="replace").strip()
            detail = f": {reason}" if reason else ""
            raise RuntimeError(f"Peer {remote_id} rejected QUIC frame with NACK{detail}")

        if ack != ACK_BYTE:
            raise RuntimeError(f"Peer {remote_id} returned invalid QUIC ACK: {ack!r}")

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        congestion_monitor.record_rtt(elapsed_ms)
        return elapsed_ms
