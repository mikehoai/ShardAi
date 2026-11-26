"""Async P2P Binary Socket Transport (`send_activations` & `receive_activations`).

Handles length-prefixed binary TCP streaming of intermediate LLM hidden states
between ShardAi nodes with `TCP_NODELAY`, explicit timeouts, and congestion detection.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
import ipaddress
import logging
import socket
import struct
import time

from p2p.serializer import (
    MAGIC_HEADER,
    ActivationFrame,
    DecodedTokenFrame,
    InferenceErrorFrame,
    InferenceHeartbeatFrame,
    InferenceRequestFrame,
    PacketType,
    deserialize_activation_tensor,
    deserialize_decoded_token,
    deserialize_inference_error,
    deserialize_inference_heartbeat,
    deserialize_inference_request,
)

logger = logging.getLogger("shardai.p2p.transport")

# 4-byte unsigned big-endian length prefix for TCP stream framing
FRAME_LEN_STRUCT = struct.Struct("!I")
ACK_BYTE: bytes = b"\x06"
NACK_BYTE: bytes = b"\x15"
DEFAULT_P2P_PORT: int = 49152
MAX_WIRE_FRAME_BYTES: int = 68 * 1024 * 1024  # 68 MB max wire frame
CONGESTION_RTT_THRESHOLD_MS: float = 25.0

current_relay_source: ContextVar[str | None] = ContextVar(
    "current_relay_source", default=None
)


class PeerUnreachableError(OSError):
    """Raised when a TCP connection to a peer cannot be established."""


class BandwidthCongestionMonitor:
    """Tracks socket Round-Trip Time (RTT) to trigger adaptive LZ4/Zlib compression."""

    def __init__(self, rtt_threshold_ms: float = CONGESTION_RTT_THRESHOLD_MS) -> None:
        self.rtt_threshold_ms: float = rtt_threshold_ms
        self.ema_rtt_ms: float = 0.0

    def record_rtt(self, rtt_ms: float) -> None:
        if self.ema_rtt_ms == 0.0:
            self.ema_rtt_ms = rtt_ms
        else:
            self.ema_rtt_ms = (0.7 * self.ema_rtt_ms) + (0.3 * rtt_ms)

    @property
    def is_congested(self) -> bool:
        return self.ema_rtt_ms >= self.rtt_threshold_ms


congestion_monitor = BandwidthCongestionMonitor()


def _parse_host_port(peer_endpoint: str, default_port: int = DEFAULT_P2P_PORT) -> tuple[str, int]:
    """Parse IPv4, IPv6 (bare or bracketed), or hostname endpoints into `(host, port)`."""
    cleaned = peer_endpoint.strip()
    if not cleaned:
        raise ValueError("Peer endpoint must not be empty")

    if cleaned.startswith("["):
        close_idx = cleaned.find("]")
        if close_idx == -1:
            raise ValueError(f"Malformed bracketed IPv6 endpoint: {peer_endpoint!r}")
        host_part = cleaned[1:close_idx].strip()
        if not host_part:
            raise ValueError(f"Empty host in bracketed endpoint: {peer_endpoint!r}")
        ip_base = host_part.split("%", 1)[0]
        addr = ipaddress.ip_address(ip_base)
        if addr.version != 6:
            raise ValueError(f"Bracketed host must be an IPv6 address: {peer_endpoint!r}")
        rest = cleaned[close_idx + 1 :].strip()
        if not rest:
            return host_part, default_port
        if not rest.startswith(":"):
            raise ValueError(f"Unexpected trailing characters in endpoint: {peer_endpoint!r}")
        port_part = rest[1:].strip()
        if not port_part:
            raise ValueError(f"Missing port in bracketed endpoint: {peer_endpoint!r}")
        port = int(port_part)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port out of range (1..65535): {port}")
        return host_part, port

    if "%" in cleaned:
        ip_base, zone_part = cleaned.split("%", 1)
        if zone_part and ":" in ip_base:
            try:
                addr = ipaddress.ip_address(ip_base)
                if addr.version == 6:
                    return cleaned, default_port
            except ValueError:
                pass
    else:
        try:
            ipaddress.ip_address(cleaned)
            return cleaned, default_port
        except ValueError:
            pass

    colon_count = cleaned.count(":")
    if colon_count == 0:
        return cleaned, default_port
    if colon_count == 1:
        host_part, port_part = cleaned.split(":", 1)
        host_part = host_part.strip()
        port_part = port_part.strip()
        if not host_part or not port_part:
            raise ValueError(f"Malformed host:port endpoint: {peer_endpoint!r}")
        port = int(port_part)
        if not (1 <= port <= 65535):
            raise ValueError(f"Port out of range (1..65535): {port}")
        return host_part, port

    raise ValueError(f"Malformed multi-colon peer endpoint: {peer_endpoint!r}")


async def send_activations(
    peer_ip: str,
    byte_stream: bytes,
    peer_port: int = DEFAULT_P2P_PORT,
    timeout_seconds: float = 5.0,
    ack_timeout_seconds: float | None = None,
) -> float:
    """Push a binary activation frame (or decoded token frame) to a peer over a TCP socket.

    Args:
        peer_ip: Destination peer IP address (or `'ip:port'` string).
        byte_stream: Binary frame packed by `p2p.serializer`.
        peer_port: Destination TCP port (used if `peer_ip` does not include `:port`).
        timeout_seconds: Explicit connect and write timeout in seconds.
        ack_timeout_seconds: Optional timeout in seconds for awaiting the peer's post-handler
            `ACK_BYTE` / `NACK_BYTE` response (defaults to `timeout_seconds`).

    Returns:
        float: Measured socket transmission Round-Trip Time (RTT) in milliseconds.
    """
    ack_wait_timeout: float = (
        timeout_seconds if ack_timeout_seconds is None else ack_timeout_seconds
    )
    start_ns: int = time.perf_counter_ns()
    host = peer_ip
    resolved_port = peer_port

    try:
        host, resolved_port = _parse_host_port(peer_ip, peer_port)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, resolved_port),
            timeout=timeout_seconds,
        )
    except (OSError, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
        raise PeerUnreachableError(
            f"Unable to connect to peer {host}:{resolved_port}: {exc}"
        ) from exc
    try:
        # Disable Nagle's algorithm for immediate low-latency tensor streaming
        raw_sock: socket.SocketType | None = writer.get_extra_info("socket")
        if raw_sock is not None:
            raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        length_prefix: bytes = FRAME_LEN_STRUCT.pack(len(byte_stream))
        writer.write(length_prefix + byte_stream)
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)

        # Wait for 1-byte ACK/NACK confirming peer validated and processed the frame
        try:
            ack: bytes = await asyncio.wait_for(
                reader.readexactly(1), timeout=ack_wait_timeout
            )
        except asyncio.IncompleteReadError as exc:
            raise RuntimeError(
                f"Peer {host}:{resolved_port} closed connection before sending ACK"
            ) from exc

        if ack == NACK_BYTE:
            raise RuntimeError(
                f"Peer {host}:{resolved_port} rejected frame with NACK"
            )
        if ack != ACK_BYTE:
            raise RuntimeError(f"Peer {host}:{resolved_port} returned invalid ACK: {ack!r}")

        elapsed_ms: float = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        congestion_monitor.record_rtt(elapsed_ms)
        return elapsed_ms
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:
            pass


async def dispatch_p2p_frame(
    frame_bytes: bytes,
    on_activation_frame: Callable[[ActivationFrame], Awaitable[None]],
    on_token_frame: Callable[[DecodedTokenFrame], Awaitable[None]] | None = None,
    on_inference_request: Callable[[InferenceRequestFrame], Awaitable[None]] | None = None,
    on_inference_error: Callable[[InferenceErrorFrame], Awaitable[None]] | None = None,
    on_inference_heartbeat: Callable[[InferenceHeartbeatFrame], Awaitable[None]] | None = None,
) -> None:
    """Validate, deserialize, and dispatch a single binary P2P frame."""
    frame_len = len(frame_bytes)
    if frame_len <= 0 or frame_len > MAX_WIRE_FRAME_BYTES:
        raise ValueError(f"Rejected invalid P2P frame length: {frame_len} bytes")

    if frame_len < 6 or frame_bytes[:4] != MAGIC_HEADER:
        raise ValueError("Received malformed frame with invalid magic bytes.")

    pkt_type: int = frame_bytes[4]
    if pkt_type == int(PacketType.ACTIVATION_TENSOR):
        activation_frame = await asyncio.to_thread(
            deserialize_activation_tensor,
            frame_bytes,
        )
        await on_activation_frame(activation_frame)
    elif pkt_type == int(PacketType.DECODED_TOKEN):
        if on_token_frame is None:
            raise ValueError(
                "Received DECODED_TOKEN frame but no on_token_frame handler is registered."
            )
        token_frame = deserialize_decoded_token(frame_bytes)
        await on_token_frame(token_frame)
    elif pkt_type == int(PacketType.INFERENCE_REQUEST):
        if on_inference_request is None:
            raise ValueError(
                "Received INFERENCE_REQUEST frame but no on_inference_request handler is registered."
            )
        inference_req_frame = deserialize_inference_request(frame_bytes)
        await on_inference_request(inference_req_frame)
    elif pkt_type == int(PacketType.INFERENCE_ERROR):
        if on_inference_error is None:
            raise ValueError(
                "Received INFERENCE_ERROR frame but no on_inference_error handler is registered."
            )
        inference_err_frame = deserialize_inference_error(frame_bytes)
        await on_inference_error(inference_err_frame)
    elif pkt_type == int(PacketType.INFERENCE_HEARTBEAT):
        hb_frame = deserialize_inference_heartbeat(frame_bytes)
        if on_inference_heartbeat is not None:
            await on_inference_heartbeat(hb_frame)
    else:
        raise ValueError(f"Unsupported P2P packet type: {pkt_type}")


async def handle_incoming_p2p_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    on_activation_frame: Callable[[ActivationFrame], Awaitable[None]],
    on_token_frame: Callable[[DecodedTokenFrame], Awaitable[None]] | None = None,
    timeout_seconds: float = 45.0,
    on_inference_request: Callable[[InferenceRequestFrame], Awaitable[None]] | None = None,
    on_inference_error: Callable[[InferenceErrorFrame], Awaitable[None]] | None = None,
    on_inference_heartbeat: Callable[[InferenceHeartbeatFrame], Awaitable[None]] | None = None,
) -> None:
    """Read, validate, deserialize, and process a binary P2P frame before sending ACK/NACK."""
    try:
        raw_sock: socket.SocketType | None = writer.get_extra_info("socket")
        if raw_sock is not None:
            raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        len_bytes: bytes = await asyncio.wait_for(
            reader.readexactly(FRAME_LEN_STRUCT.size),
            timeout=timeout_seconds,
        )
        (frame_len,) = FRAME_LEN_STRUCT.unpack(len_bytes)
        if frame_len <= 0 or frame_len > MAX_WIRE_FRAME_BYTES:
            raise ValueError(f"Rejected invalid P2P frame length: {frame_len} bytes")

        frame_bytes: bytes = await asyncio.wait_for(
            reader.readexactly(frame_len),
            timeout=timeout_seconds,
        )

        await dispatch_p2p_frame(
            frame_bytes=frame_bytes,
            on_activation_frame=on_activation_frame,
            on_token_frame=on_token_frame,
            on_inference_request=on_inference_request,
            on_inference_error=on_inference_error,
            on_inference_heartbeat=on_inference_heartbeat,
        )

        # Acknowledge ONLY after validation, deserialization, and handler execution succeed
        writer.write(ACK_BYTE)
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)
    except Exception as exc:
        logger.error("Error processing incoming P2P activation stream: %s", exc)
        try:
            if not writer.is_closing():
                writer.write(NACK_BYTE)
                await asyncio.wait_for(writer.drain(), timeout=1.0)
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
        except Exception:
            pass


async def receive_activations(
    host: str,
    port: int,
    on_activation_frame: Callable[[ActivationFrame], Awaitable[None]],
    on_token_frame: Callable[[DecodedTokenFrame], Awaitable[None]] | None = None,
    on_inference_request: Callable[[InferenceRequestFrame], Awaitable[None]] | None = None,
    on_inference_error: Callable[[InferenceErrorFrame], Awaitable[None]] | None = None,
    on_inference_heartbeat: Callable[[InferenceHeartbeatFrame], Awaitable[None]] | None = None,
) -> asyncio.AbstractServer:
    """Start an async TCP server listening for incoming binary activation streams from peers.

    Args:
        host: Local interface to bind (`'0.0.0.0'` for P2P or `'127.0.0.1'` for local tests).
        port: Local TCP port for receiving activation frames.
        on_activation_frame: Async callback invoked with each deserialized `ActivationFrame`.
        on_token_frame: Async callback invoked when a Tail Node returns a `DecodedTokenFrame`.
        on_inference_request: Async callback invoked when a non-Head peer proxies an inference request.
        on_inference_error: Async callback invoked when a Head Node returns a terminal inference error.
        on_inference_heartbeat: Async callback invoked when a Head Node sends a progress heartbeat.

    Returns:
        asyncio.AbstractServer: Running asyncio TCP server instance.
    """

    async def _client_connected(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await handle_incoming_p2p_stream(
            reader=reader,
            writer=writer,
            on_activation_frame=on_activation_frame,
            on_token_frame=on_token_frame,
            on_inference_request=on_inference_request,
            on_inference_error=on_inference_error,
            on_inference_heartbeat=on_inference_heartbeat,
        )

    server = await asyncio.start_server(_client_connected, host=host, port=port)
    logger.info("P2P activation listener started on %s:%d", host, port)
    return server
