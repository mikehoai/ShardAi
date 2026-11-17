"""Zeroconf (RFC 6762 mDNS / RFC 6763 DNS-SD) LAN discovery for ShardAi Offline Mode.

When the Cloudflare Tracker is unreachable, each node registers a DNS-SD service of type
`_shardai._tcp.local.` on every local IPv4 interface. TXT records carry the (non-secret)
room metadata. A persistent `AsyncServiceBrowser` keeps an in-memory table of every ShardAi
node on the same Wi-Fi / Ethernet segment, so machines form a "LAN Room" automatically.

Security: TXT records never contain the room key — only `is_private` and a SHA-256 hash prefix.
Everything degrades gracefully to a no-op if `zeroconf` is not installed or multicast is blocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
from typing import TYPE_CHECKING, Any

from p2p.mdns_discovery import LanRoomBeacon, _list_local_ipv4_addresses

if TYPE_CHECKING:
    from zeroconf import ServiceStateChange, Zeroconf
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

logger = logging.getLogger("shardai.p2p.zeroconf")

SHARDAI_SERVICE_TYPE: str = "_shardai._tcp.local."
RESOLVE_TIMEOUT_MS: int = 1500

try:
    from zeroconf import IPVersion, ServiceStateChange as _ServiceStateChange
    from zeroconf.asyncio import (
        AsyncServiceBrowser as _AsyncServiceBrowser,
        AsyncServiceInfo as _AsyncServiceInfo,
        AsyncZeroconf as _AsyncZeroconf,
    )

    ZEROCONF_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    ZEROCONF_AVAILABLE = False


def is_zeroconf_enabled() -> bool:
    """Return True when the zeroconf library is installed and not disabled via env."""
    return ZEROCONF_AVAILABLE and os.getenv("SHARDAI_ENABLE_ZEROCONF", "true").lower() == "true"


def _sanitize_instance_name(node_id: str) -> str:
    """DNS-SD instance labels must be <= 63 bytes; keep it ASCII-safe."""
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", node_id).strip("-") or "shardai-node"
    return cleaned[:63]


def _beacon_to_txt(beacon: LanRoomBeacon) -> dict[str, str]:
    return {
        "v": "1",
        "node_id": beacon.node_id,
        "room_id": beacon.room_id,
        "model": beacon.model[:200],  # TXT entries are capped at 255 bytes each
        "ram_gb": f"{beacon.ram_gb:.2f}",
        "is_private": "1" if beacon.is_private else "0",
        "p2p_port": str(beacon.p2p_port),
        "shard_layer_start": str(beacon.shard_layer_start),
        "shard_layer_end": str(beacon.shard_layer_end),
        "total_layers": str(beacon.total_layers),
        "role": beacon.role,
    }


def _decode_txt(properties: dict[bytes, bytes | None]) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for raw_key, raw_value in properties.items():
        key = raw_key.decode("utf-8", errors="ignore")
        decoded[key] = raw_value.decode("utf-8", errors="ignore") if raw_value else ""
    return decoded


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


class ZeroconfLanDiscovery:
    """Advertise this node and browse peers over standard mDNS / DNS-SD using `zeroconf`."""

    def __init__(self) -> None:
        self._azc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._registered_info: AsyncServiceInfo | None = None
        self._peers: dict[str, dict[str, Any]] = {}  # keyed by DNS-SD service name
        self._resolve_tasks: set[asyncio.Task[None]] = set()
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._azc is not None

    @property
    def is_advertising(self) -> bool:
        return self._registered_info is not None

    async def ensure_started(self) -> bool:
        """Lazily start the shared AsyncZeroconf instance + persistent service browser."""
        if not is_zeroconf_enabled():
            return False
        async with self._lock:
            if self._azc is not None:
                return True
            try:
                self._azc = _AsyncZeroconf(ip_version=IPVersion.V4Only)
                self._browser = _AsyncServiceBrowser(
                    self._azc.zeroconf,
                    [SHARDAI_SERVICE_TYPE],
                    handlers=[self._on_service_state_change],
                )
                logger.info("Zeroconf LAN browser started for %s", SHARDAI_SERVICE_TYPE)
                return True
            except Exception as exc:  # multicast blocked, no interfaces, etc.
                logger.warning("Zeroconf unavailable on this host: %s", exc)
                self._azc = None
                self._browser = None
                return False

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        """Browser callback (runs on the event loop): resolve added/updated, drop removed."""
        if state_change is _ServiceStateChange.Removed:
            removed = self._peers.pop(name, None)
            if removed is not None:
                logger.info("LAN peer left: %s (room=%s)", removed.get("node_id"), removed.get("room_id"))
            return
        task = asyncio.ensure_future(self._resolve_service(zeroconf, service_type, name))
        self._resolve_tasks.add(task)
        task.add_done_callback(self._resolve_tasks.discard)

    async def _resolve_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        info = _AsyncServiceInfo(service_type, name)
        try:
            if not await info.async_request(zeroconf, RESOLVE_TIMEOUT_MS):
                return
        except Exception as exc:
            logger.debug("Zeroconf resolve failed for %s: %s", name, exc)
            return

        txt = _decode_txt(info.properties or {})
        node_id = txt.get("node_id", "")
        if not node_id:
            return
        addresses = info.parsed_addresses(IPVersion.V4Only)
        is_new = name not in self._peers
        self._peers[name] = {
            "node_id": node_id,
            "room_id": txt.get("room_id") or "public",
            "model": txt.get("model", ""),
            "ram_gb": _safe_float(txt.get("ram_gb"), 0.0),
            "is_private": txt.get("is_private") == "1",
            "p2p_port": _safe_int(txt.get("p2p_port"), info.port or 49152),
            "shard_layer_start": _safe_int(txt.get("shard_layer_start"), 0),
            "shard_layer_end": _safe_int(txt.get("shard_layer_end"), 0),
            "total_layers": _safe_int(txt.get("total_layers"), 32),
            "role": txt.get("role") or "primary",
            "ip_address": addresses[0] if addresses else "127.0.0.1",
            "source": "zeroconf",
        }
        if is_new:
            logger.info(
                "LAN peer discovered via zeroconf: %s @ %s (room=%s)",
                node_id,
                self._peers[name]["ip_address"],
                self._peers[name]["room_id"],
            )

    async def start_advertising(self, beacon: LanRoomBeacon) -> bool:
        """Register this node as `_shardai._tcp.local.` (skipped for unlisted rooms)."""
        await self.stop_advertising()
        if beacon.is_unlisted or not await self.ensure_started():
            return False
        assert self._azc is not None

        ips = _list_local_ipv4_addresses() or ["127.0.0.1"]
        instance = _sanitize_instance_name(beacon.node_id)
        info = _AsyncServiceInfo(
            SHARDAI_SERVICE_TYPE,
            f"{instance}.{SHARDAI_SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip) for ip in ips],
            port=beacon.p2p_port,
            properties=_beacon_to_txt(beacon),
            server=f"{instance}.local.",
        )
        try:
            await self._azc.async_register_service(info, allow_name_change=True)
            self._registered_info = info
            logger.info(
                "Zeroconf broadcasting %s on %s:%d (room=%s, layers %d..%d)",
                instance,
                ",".join(ips),
                beacon.p2p_port,
                beacon.room_id,
                beacon.shard_layer_start,
                beacon.shard_layer_end,
            )
            return True
        except Exception as exc:
            logger.warning("Zeroconf service registration failed: %s", exc)
            return False

    async def stop_advertising(self) -> None:
        """Unregister this node's DNS-SD record (sends goodbye packets to the LAN)."""
        info, self._registered_info = self._registered_info, None
        if info is not None and self._azc is not None:
            try:
                await self._azc.async_unregister_service(info)
            except Exception as exc:
                logger.debug("Ignoring zeroconf unregister error: %s", exc)

    async def discover_peers(self, wait_seconds: float = 0.0) -> list[dict[str, Any]]:
        """Return all currently known LAN peers (optionally waiting for fresh responses)."""
        if not await self.ensure_started():
            return []
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        own_name = self._registered_info.name if self._registered_info is not None else None
        return [dict(peer) for name, peer in self._peers.items() if name != own_name]

    async def close(self) -> None:
        """Stop browsing/advertising and release the multicast sockets."""
        await self.stop_advertising()
        for task in list(self._resolve_tasks):
            task.cancel()
        if self._browser is not None:
            try:
                await self._browser.async_cancel()
            except Exception:
                pass
            self._browser = None
        if self._azc is not None:
            try:
                await self._azc.async_close()
            except Exception:
                pass
            self._azc = None
        self._peers.clear()


zeroconf_discovery = ZeroconfLanDiscovery()
