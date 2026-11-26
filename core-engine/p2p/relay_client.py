"""Async WebSocket Relay Client for NAT-traversed Internet P2P frame delivery.

Connects to the Signaling Tracker relay endpoint using bearer authentication,
splits large P2P binary frames into bounded chunks, reassembles incoming frames,
and dispatches handlers asynchronously to avoid receive-loop deadlocks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import logging
import os
import time
from urllib.parse import urlparse
import uuid

import websockets.asyncio.client as ws_client
from websockets.exceptions import WebSocketException
from websockets.protocol import State

from p2p.relay_envelope import (
    MAX_NACK_REASON_BYTES,
    RelayEnvelope,
    RelayPacketKind,
    decode_envelope,
    encode_envelope,
    split_into_chunks,
    truncate_utf8_bytes,
)
from p2p.transport import MAX_WIRE_FRAME_BYTES, PeerUnreachableError

logger = logging.getLogger("shardai.p2p.relay")

MAX_CONCURRENT_PARTIALS_PER_SOURCE: int = 8
MAX_TRACKED_PARTIAL_SOURCES: int = 256
PARTIAL_MESSAGE_TTL_SECONDS: float = 30.0
MAX_CONCURRENT_INCOMING_TASKS: int = 64
RELAY_RECONNECT_INITIAL_SECONDS: float = 1.0
RELAY_RECONNECT_MAX_SECONDS: float = 30.0
RELAY_WS_MAX_SIZE_BYTES: int = 1_100_000
RELAY_WS_PING_INTERVAL_SECONDS: float = 20.0
_LOCAL_INSECURE_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})


def is_insecure_local_relay_allowed() -> bool:
    """Return True when plaintext `ws://` relay URLs are explicitly enabled for localhost tests."""
    return os.getenv("SHARDAI_RELAY_ALLOW_INSECURE_LOCAL", "").strip().lower() == "true"


def validate_relay_url(url: str, allowed_host: str | None = None) -> str:
    """Validate relay WebSocket URL scheme and host against configured tracker host."""
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("Relay URL must not be empty")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError(f"Relay URL is missing a valid hostname: {cleaned!r}")

    if scheme == "wss":
        pass
    elif scheme == "ws":
        if not is_insecure_local_relay_allowed() or host not in _LOCAL_INSECURE_HOSTS:
            raise ValueError(
                "Insecure ws:// relay URLs are rejected unless SHARDAI_RELAY_ALLOW_INSECURE_LOCAL=true "
                "for 127.0.0.1 or localhost"
            )
    else:
        raise ValueError(f"Unsupported relay URL scheme {parsed.scheme!r}; expected wss://")

    if allowed_host is not None:
        expected_host = allowed_host.strip().lower()
        if not expected_host or host != expected_host:
            raise ValueError(
                f"Relay URL host {host!r} does not match configured tracker host {expected_host!r}"
            )

    return cleaned


@dataclass(slots=True)
class _PartialMessageBuffer:
    """Tracks chunks for a single multi-chunk `(src, msg_id)` relay transfer."""

    chunk_count: int
    created_at: float
    chunks: dict[int, bytes] = field(default_factory=dict)
    total_bytes: int = 0


class RelayClient:
    """Persistent WebSocket relay client with chunking, ACK/NACK, and reconnect backoff."""

    def __init__(self) -> None:
        self._url: str | None = None
        self._token: str | None = None
        self._node_id: str | None = None
        self._allowed_host: str | None = None
        self._on_frame: Callable[[bytes, str], Awaitable[None]] | None = None

        self._ws: ws_client.ClientConnection | None = None
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._initial_connect_event: asyncio.Event | None = None
        self._send_lock: asyncio.Lock | None = None
        self._running: bool = False

        self._pending_acks: dict[bytes, asyncio.Future[None]] = {}
        self._partials: dict[str, dict[bytes, _PartialMessageBuffer]] = {}
        self._incoming_tasks: set[asyncio.Task[None]] = set()

    @property
    def is_started(self) -> bool:
        """Return True if the relay client lifecycle has been started."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Return True when the WebSocket relay link is currently open."""
        return (
            self._running
            and self._ws is not None
            and self._ws.state == State.OPEN
        )

    async def start(
        self,
        url: str,
        token: str,
        node_id: str,
        on_frame: Callable[[bytes, str], Awaitable[None]],
        allowed_host: str | None = None,
    ) -> None:
        """Start the relay client and wait briefly for the initial WebSocket handshake."""
        validated_url = validate_relay_url(url, allowed_host=allowed_host)
        cleaned_token = token.strip()
        if not cleaned_token:
            raise ValueError("Relay token must not be empty")
        cleaned_node_id = node_id.strip()
        if not cleaned_node_id:
            raise ValueError("Relay node_id must not be empty")

        if self._running:
            same_identity = (
                self._node_id == cleaned_node_id
                and self._allowed_host == (allowed_host.strip().lower() if allowed_host else None)
            )
            if same_identity:
                self._on_frame = on_frame
                self.update_credentials(validated_url, cleaned_token)
                return
            await self.stop()

        self._url = validated_url
        self._token = cleaned_token
        self._node_id = cleaned_node_id
        self._allowed_host = allowed_host.strip().lower() if allowed_host else None
        self._on_frame = on_frame
        self._stop_event = asyncio.Event()
        self._initial_connect_event = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._running = True
        self._runner_task = asyncio.create_task(self._connection_loop())

        try:
            await asyncio.wait_for(self._initial_connect_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            # Connection continues retrying in the background without blocking node startup.
            pass

    def update_credentials(self, url: str, token: str) -> None:
        """Update the active relay URL and bearer token for current or future reconnects."""
        validated_url = validate_relay_url(url, allowed_host=self._allowed_host)
        cleaned_token = token.strip()
        if not cleaned_token:
            raise ValueError("Relay token must not be empty")

        url_changed = self._url is not None and self._url != validated_url
        self._url = validated_url
        self._token = cleaned_token

        if url_changed and self._ws is not None and self._ws.state == State.OPEN:
            # Force immediate reconnect to the updated URL without logging credentials.
            asyncio.create_task(self._close_active_ws())

    async def _close_active_ws(self) -> None:
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop the relay client, close the WebSocket, and cancel pending tasks cleanly."""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._initial_connect_event is not None:
            self._initial_connect_event.set()

        await self._close_active_ws()

        if self._runner_task is not None:
            if not self._runner_task.done():
                self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._runner_task = None

        incoming = list(self._incoming_tasks)
        for task in incoming:
            if not task.done():
                task.cancel()
        if incoming:
            await asyncio.gather(*incoming, return_exceptions=True)
        self._incoming_tasks.clear()

        self._fail_pending_acks(RuntimeError("Relay client stopped"))
        self._partials.clear()
        self._ws = None
        self._stop_event = None
        self._initial_connect_event = None
        self._send_lock = None

    def _fail_pending_acks(self, exc: Exception) -> None:
        """Reject all outstanding ACK futures with `exc`."""
        for msg_id, fut in list(self._pending_acks.items()):
            if not fut.done():
                fut.set_exception(exc)
            self._pending_acks.pop(msg_id, None)

    async def _send_raw_ws(self, payload: bytes) -> None:
        """Send a single binary WebSocket message using the active connection."""
        ws = self._ws
        if ws is None or ws.state != State.OPEN or not self._running:
            raise PeerUnreachableError("Relay WebSocket is not connected")
        lock = self._send_lock
        try:
            if lock is not None:
                async with lock:
                    await ws.send(payload)
            else:
                await ws.send(payload)
        except (OSError, WebSocketException) as exc:
            raise PeerUnreachableError(f"Relay WebSocket send failed: {exc}") from exc

    async def send_frame(
        self,
        dst_node_id: str,
        frame: bytes,
        ack_timeout: float = 10.0,
    ) -> float:
        """Send a binary P2P frame through the relay to `dst_node_id` and await ACK."""
        if not self.is_connected or self._node_id is None:
            raise PeerUnreachableError("Relay client is not connected")

        chunks = split_into_chunks(frame)
        msg_id = uuid.uuid4().bytes
        loop = asyncio.get_running_loop()
        ack_future: asyncio.Future[None] = loop.create_future()
        self._pending_acks[msg_id] = ack_future

        start_ns = time.perf_counter_ns()
        try:
            chunk_count = len(chunks)
            for idx, chunk in enumerate(chunks):
                envelope_bytes = encode_envelope(
                    kind=RelayPacketKind.DATA,
                    msg_id=msg_id,
                    src=self._node_id,
                    dst=dst_node_id,
                    payload=chunk,
                    chunk_index=idx,
                    chunk_count=chunk_count,
                )
                await self._send_raw_ws(envelope_bytes)

            await asyncio.wait_for(ack_future, timeout=ack_timeout)
            return (time.perf_counter_ns() - start_ns) / 1_000_000.0
        finally:
            self._pending_acks.pop(msg_id, None)

    async def _send_control_envelope(
        self,
        *,
        kind: RelayPacketKind,
        msg_id: bytes,
        dst: str,
        payload: bytes = b"",
    ) -> None:
        """Best-effort send of an ACK, NACK, or PONG control envelope."""
        if not self.is_connected or self._node_id is None:
            return
        try:
            encoded = encode_envelope(
                kind=kind,
                msg_id=msg_id,
                src=self._node_id,
                dst=dst,
                payload=payload,
                chunk_index=0,
                chunk_count=1,
            )
            await self._send_raw_ws(encoded)
        except Exception as exc:
            logger.debug("Failed to send relay %s to %s: %s", kind.name, dst, exc)

    def _prune_expired_partials(self, now: float) -> None:
        """Remove partial reassembly buffers older than `PARTIAL_MESSAGE_TTL_SECONDS`."""
        expired_sources: list[str] = []
        for src, msg_map in self._partials.items():
            expired_ids = [
                msg_id
                for msg_id, buf in msg_map.items()
                if (now - buf.created_at) > PARTIAL_MESSAGE_TTL_SECONDS
            ]
            for msg_id in expired_ids:
                msg_map.pop(msg_id, None)
            if not msg_map:
                expired_sources.append(src)
        for src in expired_sources:
            self._partials.pop(src, None)

    def _reassemble_data_chunk(self, env: RelayEnvelope) -> tuple[bytes | None, str | None]:
        """Store an incoming DATA chunk and return `(completed_frame, error_reason)`."""
        now = time.monotonic()
        self._prune_expired_partials(now)

        if env.chunk_count == 1:
            if len(env.payload) > MAX_WIRE_FRAME_BYTES:
                return None, "Frame exceeds maximum wire frame size"
            return env.payload, None

        src_map = self._partials.get(env.src)
        if src_map is None:
            if len(self._partials) >= MAX_TRACKED_PARTIAL_SOURCES:
                return None, "Too many concurrent relay sources"
            src_map = {}
            self._partials[env.src] = src_map

        buf = src_map.get(env.msg_id)
        if buf is None:
            if len(src_map) >= MAX_CONCURRENT_PARTIALS_PER_SOURCE:
                return None, "Too many concurrent partial messages from source"
            buf = _PartialMessageBuffer(chunk_count=env.chunk_count, created_at=now)
            src_map[env.msg_id] = buf
        elif buf.chunk_count != env.chunk_count:
            src_map.pop(env.msg_id, None)
            if not src_map:
                self._partials.pop(env.src, None)
            return None, "Inconsistent chunk_count across chunks"

        existing = buf.chunks.get(env.chunk_index)
        delta_bytes = len(env.payload) - (len(existing) if existing is not None else 0)
        if buf.total_bytes + delta_bytes > MAX_WIRE_FRAME_BYTES:
            src_map.pop(env.msg_id, None)
            if not src_map:
                self._partials.pop(env.src, None)
            return None, "Reassembled frame exceeds maximum wire frame size"

        buf.chunks[env.chunk_index] = env.payload
        buf.total_bytes += delta_bytes

        if len(buf.chunks) < buf.chunk_count:
            return None, None

        src_map.pop(env.msg_id, None)
        if not src_map:
            self._partials.pop(env.src, None)

        frame_bytes = b"".join(buf.chunks[idx] for idx in range(buf.chunk_count))
        return frame_bytes, None

    async def _dispatch_and_ack(
        self,
        frame_bytes: bytes,
        src_node_id: str,
        msg_id: bytes,
    ) -> None:
        """Invoke `on_frame` off the receive loop and send ACK or NACK back to `src_node_id`."""
        on_frame = self._on_frame
        if on_frame is None:
            await self._send_control_envelope(
                kind=RelayPacketKind.NACK,
                msg_id=msg_id,
                dst=src_node_id,
                payload=b"No relay frame handler registered",
            )
            return

        try:
            await on_frame(frame_bytes, src_node_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason_bytes = truncate_utf8_bytes(str(exc) or exc.__class__.__name__, MAX_NACK_REASON_BYTES)
            await self._send_control_envelope(
                kind=RelayPacketKind.NACK,
                msg_id=msg_id,
                dst=src_node_id,
                payload=reason_bytes,
            )
            return

        await self._send_control_envelope(
            kind=RelayPacketKind.ACK,
            msg_id=msg_id,
            dst=src_node_id,
        )

    async def _handle_incoming_envelope(self, env: RelayEnvelope) -> None:
        """Handle a decoded relay envelope without blocking the WebSocket receive loop."""
        if env.kind == RelayPacketKind.ACK:
            fut = self._pending_acks.pop(env.msg_id, None)
            if fut is not None and not fut.done():
                fut.set_result(None)
            return

        if env.kind == RelayPacketKind.NACK:
            fut = self._pending_acks.pop(env.msg_id, None)
            if fut is not None and not fut.done():
                reason = env.payload.decode("utf-8", errors="replace")
                fut.set_exception(
                    RuntimeError(f"Peer {env.src} rejected relayed frame with NACK: {reason}")
                )
            return

        if env.kind == RelayPacketKind.NO_ROUTE:
            fut = self._pending_acks.pop(env.msg_id, None)
            if fut is not None and not fut.done():
                fut.set_exception(
                    RuntimeError(f"Relay reported NO_ROUTE for peer {env.src}")
                )
            return

        if env.kind == RelayPacketKind.PING:
            await self._send_control_envelope(
                kind=RelayPacketKind.PONG,
                msg_id=env.msg_id,
                dst=env.src,
            )
            return

        if env.kind == RelayPacketKind.PONG:
            return

        if env.kind == RelayPacketKind.DATA:
            completed_frame, error_reason = self._reassemble_data_chunk(env)
            if error_reason is not None:
                await self._send_control_envelope(
                    kind=RelayPacketKind.NACK,
                    msg_id=env.msg_id,
                    dst=env.src,
                    payload=truncate_utf8_bytes(error_reason, MAX_NACK_REASON_BYTES),
                )
                return
            if completed_frame is None:
                return

            if len(self._incoming_tasks) >= MAX_CONCURRENT_INCOMING_TASKS:
                await self._send_control_envelope(
                    kind=RelayPacketKind.NACK,
                    msg_id=env.msg_id,
                    dst=env.src,
                    payload=b"Node relay receiver busy (max concurrent incoming frames)",
                )
                return

            task = asyncio.create_task(
                self._dispatch_and_ack(completed_frame, env.src, env.msg_id)
            )
            self._incoming_tasks.add(task)
            task.add_done_callback(self._incoming_tasks.discard)

    async def _connection_loop(self) -> None:
        """Maintain the WebSocket relay connection with exponential reconnect backoff."""
        backoff = RELAY_RECONNECT_INITIAL_SECONDS
        stop_event = self._stop_event
        assert stop_event is not None

        while self._running and not stop_event.is_set():
            url = self._url
            token = self._token
            if not url or not token:
                break

            connected_at: float | None = None
            try:
                async with ws_client.connect(
                    url,
                    additional_headers={"Authorization": f"Bearer {token}"},
                    max_size=RELAY_WS_MAX_SIZE_BYTES,
                    ping_interval=RELAY_WS_PING_INTERVAL_SECONDS,
                    open_timeout=5.0,
                ) as ws:
                    self._ws = ws
                    connected_at = time.monotonic()
                    backoff = RELAY_RECONNECT_INITIAL_SECONDS
                    if self._initial_connect_event is not None:
                        self._initial_connect_event.set()
                    logger.info("Connected to Signaling Tracker relay (%s)", urlparse(url).hostname)

                    async for raw_message in ws:
                        if not self._running:
                            break
                        if not isinstance(raw_message, (bytes, bytearray)):
                            logger.debug("Ignoring non-binary WebSocket relay message")
                            continue
                        try:
                            envelope = decode_envelope(bytes(raw_message))
                        except ValueError as exc:
                            logger.warning("Ignoring malformed relay envelope: %s", exc)
                            continue
                        await self._handle_incoming_envelope(envelope)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._initial_connect_event is not None:
                    self._initial_connect_event.set()
                if self._running:
                    logger.warning(
                        "Relay WebSocket disconnected (%s); reconnecting in %.1fs...",
                        exc.__class__.__name__,
                        backoff,
                    )
            finally:
                self._ws = None
                self._fail_pending_acks(PeerUnreachableError("Relay connection closed"))
                self._partials.clear()

            if not self._running or stop_event.is_set():
                break

            if connected_at is not None and (time.monotonic() - connected_at) >= 5.0:
                backoff = RELAY_RECONNECT_INITIAL_SECONDS

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                break
            except asyncio.TimeoutError:
                backoff = min(RELAY_RECONNECT_MAX_SECONDS, backoff * 2.0)
