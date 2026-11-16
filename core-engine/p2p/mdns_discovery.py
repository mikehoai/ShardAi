"""Lightweight LAN mDNS / Multicast Room Discovery for ShardAi Nodes.

Allows nodes on the same local subnet to advertise and discover active rooms
without requiring internet access to the central Cloudflare Signaling Server.
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import socket
import struct
from typing import Any

logger = logging.getLogger("shardai.p2p.mdns")

MDNS_MULTICAST_GROUP: str = "224.0.0.251"
SHARDAI_MDNS_PORT: int = 5353
FALLBACK_LAN_PORT: int = 49535
PROBE_MAGIC: bytes = b"SHARDAI_MDNS_DISCOVER_V1"


@dataclass(frozen=True, slots=True)
class LanRoomBeacon:
    """Room metadata advertised over LAN mDNS multicast (never includes secret keys)."""

    node_id: str
    room_id: str
    model: str
    ram_gb: float
    is_private: bool
    is_unlisted: bool
    p2p_port: int


class _MdnsResponderProtocol(asyncio.DatagramProtocol):
    """UDP multicast responder replying to ShardAi LAN room discovery probes."""

    def __init__(self, beacon: LanRoomBeacon) -> None:
        self.beacon: LanRoomBeacon = beacon
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if data != PROBE_MAGIC or self.beacon.is_unlisted or self.transport is None:
            return
        reply_payload = json.dumps(
            {
                "type": "SHARDAI_MDNS_ROOM_V1",
                "node_id": self.beacon.node_id,
                "room_id": self.beacon.room_id,
                "model": self.beacon.model,
                "ram_gb": self.beacon.ram_gb,
                "is_private": self.beacon.is_private,
                "p2p_port": self.beacon.p2p_port,
            }
        ).encode("utf-8")
        try:
            self.transport.sendto(reply_payload, addr)
        except Exception:
            pass


class LanMdnsService:
    """Manages local mDNS advertisement and non-blocking LAN room scanning."""

    def __init__(self) -> None:
        self._transport: asyncio.DatagramTransport | None = None
        self._active_beacon: LanRoomBeacon | None = None

    async def start_advertising(self, beacon: LanRoomBeacon) -> None:
        """Start responding to LAN mDNS room probes if the room is not unlisted."""
        await self.stop_advertising()
        self._active_beacon = beacon
        if beacon.is_unlisted:
            return

        loop = asyncio.get_running_loop()
        for port in (SHARDAI_MDNS_PORT, FALLBACK_LAN_PORT):
            sock: socket.socket | None = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("", port))
                mreq = struct.pack(
                    "4s4s",
                    socket.inet_aton(MDNS_MULTICAST_GROUP),
                    socket.inet_aton("0.0.0.0"),
                )
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                sock.setblocking(False)

                transport, _ = await loop.create_datagram_endpoint(
                    lambda: _MdnsResponderProtocol(beacon),
                    sock=sock,
                )
                self._transport = transport  # type: ignore[assignment]
                logger.info(
                    "LAN mDNS room advertiser active on %s:%d (room=%s)",
                    MDNS_MULTICAST_GROUP,
                    port,
                    beacon.room_id,
                )
                return
            except OSError as exc:
                if sock is not None:
                    sock.close()
                logger.debug("Could not bind mDNS port %d: %s", port, exc)

    async def stop_advertising(self) -> None:
        """Stop advertising LAN rooms and release UDP multicast socket."""
        self._active_beacon = None
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception as exc:
                logger.debug("Ignoring mDNS transport close error: %s", exc)
            finally:
                self._transport = None

    async def scan_lan_rooms(self, timeout_seconds: float = 0.25) -> list[dict[str, Any]]:
        """Send an mDNS multicast probe and collect active LAN room beacons within timeout."""
        discovered: dict[str, dict[str, Any]] = {}

        # Include local active beacon if advertising on this host
        if self._active_beacon is not None and not self._active_beacon.is_unlisted:
            discovered[f"{self._active_beacon.room_id}:{self._active_beacon.node_id}"] = {
                "node_id": self._active_beacon.node_id,
                "room_id": self._active_beacon.room_id,
                "model": self._active_beacon.model,
                "ram_gb": self._active_beacon.ram_gb,
                "is_private": self._active_beacon.is_private,
                "p2p_port": self._active_beacon.p2p_port,
            }

        def _blocking_probe() -> list[dict[str, Any]]:
            found: list[dict[str, Any]] = []
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                sock.settimeout(timeout_seconds)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
                for target_port in (SHARDAI_MDNS_PORT, FALLBACK_LAN_PORT):
                    try:
                        sock.sendto(PROBE_MAGIC, (MDNS_MULTICAST_GROUP, target_port))
                    except OSError:
                        pass

                loop_deadline = asyncio.get_event_loop().time() + timeout_seconds if False else None
                _ = loop_deadline
                while True:
                    try:
                        packet, _addr = sock.recvfrom(4096)
                        data = json.loads(packet.decode("utf-8"))
                        if isinstance(data, dict) and data.get("type") == "SHARDAI_MDNS_ROOM_V1":
                            found.append(data)
                    except (TimeoutError, socket.timeout, OSError, ValueError):
                        break
            finally:
                sock.close()
            return found

        try:
            probed_items = await asyncio.to_thread(_blocking_probe)
            for item in probed_items:
                key = f"{item.get('room_id')}:{item.get('node_id')}"
                discovered[key] = item
        except Exception as exc:
            logger.debug("LAN mDNS scan skipped/failed: %s", exc)

        return list(discovered.values())


lan_mdns_service = LanMdnsService()
