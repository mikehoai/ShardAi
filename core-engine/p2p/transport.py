"""Async P2P Binary Socket Transport (`send_activations` & `receive_activations`).

Handles length-prefixed binary TCP streaming of intermediate LLM hidden states
between ShardAi nodes with `TCP_NODELAY`, explicit timeouts, and congestion detection.
"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import socket
import struct
import time

from p2p.serializer import (
    MAGIC_HEADER,
    ActivationFrame,
    DecodedTokenFrame,
    PacketType,
    deserialize_activation_tensor,
    deserialize_decoded_token,
)

logger = logging.getLogger("shardai.p2p.transport")

# 4-byte unsigned big-endian length prefix for TCP stream framing
FRAME_LEN_STRUCT = struct.Struct("!I")
ACK_BYTE: bytes = b"\x06"
DEFAULT_P2P_PORT: int = 49152
MAX_WIRE_FRAME_BYTES: int = 68 * 1024 * 1024  # 68 MB max wire frame
CONGESTION_RTT_THRESHOLD_MS: float = 25.0


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
    """Parse either `'192.0.2.10'` or `'192.0.2.10:49153'` into `(host, port)`."""
    if ":" in peer_endpoint:
        host_part, port_part = peer_endpoint.rsplit(":", 1)
        return host_part.strip(), int(port_part)
    return peer_endpoint.strip(), default_port


async def send_activations(
    peer_ip: str,
    byte_stream: bytes,
    peer_port: int = DEFAULT_P2P_PORT,
    timeout_seconds: float = 5.0,
) -> float:
    """Push a binary activation frame (or decoded token frame) to a peer over a TCP socket.

    Args:
        peer_ip: Destination peer IP address (or `'ip:port'` string).
        byte_stream: Binary frame packed by `p2p.serializer`.
        peer_port: Destination TCP port (used if `peer_ip` does not include `:port`).
        timeout_seconds: Explicit connect + write + ACK timeout in seconds.

    Returns:
        float: Measured socket transmission Round-Trip Time (RTT) in milliseconds.
    """
    host, resolved_port = _parse_host_port(peer_ip, peer_port)
    start_ns: int = time.perf_counter_ns()

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, resolved_port),
        timeout=timeout_seconds,
    )
    try:
        # Disable Nagle's algorithm for immediate low-latency tensor streaming
        raw_sock: socket.SocketType | None = writer.get_extra_info("socket")
        if raw_sock is not None:
            raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        length_prefix: bytes = FRAME_LEN_STRUCT.pack(len(byte_stream))
        writer.write(length_prefix + byte_stream)
        await asyncio.wait_for(writer.drain(), timeout=timeout_seconds)

        # Wait for 1-byte ACK confirming peer received the full tensor frame
        ack: bytes = await asyncio.wait_for(reader.readexactly(1), timeout=timeout_seconds)
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


async def handle_incoming_p2p_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    on_activation_frame: Callable[[ActivationFrame], Awaitable[None]],
    on_token_frame: Callable[[DecodedTokenFrame], Awaitable[None]] | None = None,
    timeout_seconds: float = 10.0,
) -> None:
    """Read a length-prefixed binary P2P frame, deserialize into Tensor/Token, and dispatch."""
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

        # Acknowledge receipt immediately so upstream peer can release its socket
        writer.write(ACK_BYTE)
        await writer.drain()

        if len(frame_bytes) < 6 or frame_bytes[:4] != MAGIC_HEADER:
            raise ValueError("Received malformed frame with invalid magic bytes.")

        pkt_type: int = frame_bytes[4]
        if pkt_type == int(PacketType.ACTIVATION_TENSOR):
            activation_frame = deserialize_activation_tensor(frame_bytes)
            await on_activation_frame(activation_frame)
        elif pkt_type == int(PacketType.DECODED_TOKEN):
            if on_token_frame is not None:
                token_frame = deserialize_decoded_token(frame_bytes)
                await on_token_frame(token_frame)
        else:
            raise ValueError(f"Unsupported P2P packet type: {pkt_type}")
    except Exception as exc:
        logger.error("Error processing incoming P2P activation stream: %s", exc)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def receive_activations(
    host: str,
    port: int,
    on_activation_frame: Callable[[ActivationFrame], Awaitable[None]],
    on_token_frame: Callable[[DecodedTokenFrame], Awaitable[None]] | None = None,
) -> asyncio.AbstractServer:
    """Start an async TCP server listening for incoming binary activation streams from peers.

    Args:
        host: Local interface to bind (`'0.0.0.0'` for P2P or `'127.0.0.1'` for local tests).
        port: Local TCP port for receiving activation frames.
        on_activation_frame: Async callback invoked with each deserialized `ActivationFrame`.
        on_token_frame: Async callback invoked when a Tail Node returns a `DecodedTokenFrame`.

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
        )

    server = await asyncio.start_server(_client_connected, host=host, port=port)
    logger.info("P2P activation listener started on %s:%d", host, port)
    return server
