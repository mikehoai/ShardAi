"""Best-effort UPnP Internet Gateway Device (IGD) port-mapping client.

Uses SSDP M-SEARCH discovery constrained strictly to private RFC1918 IPv4 gateways
(preventing SSRF against public or metadata addresses) and bounded XML/SOAP calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ipaddress
import logging
import os
import socket
import time
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import httpx

logger = logging.getLogger("shardai.p2p.upnp")

SSDP_MULTICAST_HOST: str = "239.255.255.250"
SSDP_MULTICAST_PORT: int = 1900
SSDP_TIMEOUT_SECONDS: float = 2.0
MAX_UPNP_DESCRIPTION_BYTES: int = 64 * 1024
DEFAULT_LEASE_SECONDS: int = 3600

IGD_SEARCH_TARGETS: tuple[str, ...] = (
    "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
    "urn:schemas-upnp-org:device:InternetGatewayDevice:2",
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


def is_upnp_enabled() -> bool:
    """Return True unless `SHARDAI_ENABLE_UPNP` is explicitly disabled."""
    raw = os.getenv("SHARDAI_ENABLE_UPNP", "true").strip().lower()
    return raw not in {"false", "0", "no", "off"}


def is_allowed_upnp_location(location_url: str) -> bool:
    """Verify a UPnP `LOCATION` URL uses http(s) and points to a private RFC1918 IPv4 address."""
    try:
        parsed = urlparse(location_url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip()
    if not host:
        return False

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False

    return (
        addr.version == 4
        and addr.is_private
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_multicast
        and not addr.is_unspecified
    )


def _extract_ssdp_location(raw_packet: bytes) -> str | None:
    """Parse an SSDP M-SEARCH response packet and extract the `LOCATION` header."""
    try:
        text = raw_packet.decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        if key.strip().lower() == "location":
            candidate = val.strip()
            return candidate if candidate else None
    return None


def _ssdp_discover_locations_sync(timeout_seconds: float = SSDP_TIMEOUT_SECONDS) -> list[str]:
    """Send SSDP M-SEARCH probes over UDP and return deduplicated valid private LOCATION URLs."""
    locations: list[str] = []
    seen: set[str] = set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.settimeout(min(0.5, max(0.1, timeout_seconds)))
        for st in IGD_SEARCH_TARGETS:
            probe = (
                "M-SEARCH * HTTP/1.1\r\n"
                f"HOST: {SSDP_MULTICAST_HOST}:{SSDP_MULTICAST_PORT}\r\n"
                'MAN: "ssdp:discover"\r\n'
                "MX: 1\r\n"
                f"ST: {st}\r\n"
                "\r\n"
            ).encode("ascii")
            try:
                sock.sendto(probe, (SSDP_MULTICAST_HOST, SSDP_MULTICAST_PORT))
            except OSError:
                continue

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(4096)
            except (socket.timeout, TimeoutError):
                break
            except OSError:
                break
            loc = _extract_ssdp_location(data)
            if loc and loc not in seen and is_allowed_upnp_location(loc):
                seen.add(loc)
                locations.append(loc)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return locations


def _strip_xml_ns(tag: str) -> str:
    """Strip XML namespace prefix from an ElementTree tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _has_unsafe_xml_declarations(raw_bytes: bytes) -> bool:
    """Return True if raw XML bytes contain `<!DOCTYPE` or `<!ENTITY` declarations."""
    upper = raw_bytes.upper()
    return b"<!DOCTYPE" in upper or b"<!ENTITY" in upper


async def _read_bounded_response(
    resp: httpx.Response,
    max_bytes: int,
) -> bytes | None:
    """Stream an HTTP response body up to `max_bytes`, returning `None` if exceeded."""
    content_len = resp.headers.get("content-length")
    if content_len is not None:
        try:
            parsed_len = int(content_len.strip())
        except ValueError:
            return None
        if parsed_len < 0 or parsed_len > max_bytes:
            return None

    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.aiter_bytes(chunk_size=8192):
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@dataclass(frozen=True, slots=True)
class UpnpServiceEndpoint:
    """Resolved UPnP WANIPConnection/WANPPPConnection SOAP control endpoint."""

    service_type: str
    control_url: str


def _parse_igd_control_endpoint(
    xml_bytes: bytes, base_location_url: str
) -> UpnpServiceEndpoint | None:
    """Parse an IGD device description XML document and extract a valid WAN control URL."""
    if len(xml_bytes) > MAX_UPNP_DESCRIPTION_BYTES:
        return None
    if _has_unsafe_xml_declarations(xml_bytes):
        return None
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    for elem in root.iter():
        if _strip_xml_ns(elem.tag) != "service":
            continue
        service_type: str | None = None
        control_url: str | None = None
        for child in elem:
            name = _strip_xml_ns(child.tag)
            text = (child.text or "").strip()
            if name == "serviceType":
                service_type = text
            elif name == "controlURL":
                control_url = text
        if (
            service_type
            and control_url
            and ("WANIPConnection" in service_type or "WANPPPConnection" in service_type)
        ):
            resolved_url = urljoin(base_location_url, control_url)
            if is_allowed_upnp_location(resolved_url):
                return UpnpServiceEndpoint(
                    service_type=service_type,
                    control_url=resolved_url,
                )
    return None


def _get_local_ipv4_for_gateway(gateway_url: str) -> str:
    """Determine the local IPv4 interface address used to reach `gateway_url`."""
    parsed = urlparse(gateway_url)
    host = parsed.hostname or "192.168.1.1"
    port = parsed.port or 80
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, port))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class UpnpClient:
    """Minimal async UPnP IGD SOAP client."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    async def discover_service(self) -> UpnpServiceEndpoint | None:
        """Discover a local RFC1918 UPnP IGD gateway and locate its WAN control URL."""
        if not is_upnp_enabled():
            return None

        locations = await asyncio.to_thread(
            _ssdp_discover_locations_sync, self._timeout_seconds
        )
        if not locations:
            return None

        timeout = httpx.Timeout(timeout=self._timeout_seconds, connect=self._timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=False,
        ) as client:
            for loc in locations:
                if not is_allowed_upnp_location(loc):
                    continue
                try:
                    async with client.stream("GET", loc) as resp:
                        if resp.status_code != 200:
                            continue
                        body = await _read_bounded_response(resp, MAX_UPNP_DESCRIPTION_BYTES)
                        if not body:
                            continue
                        endpoint = _parse_igd_control_endpoint(body, loc)
                        if endpoint is not None:
                            return endpoint
                except Exception:
                    continue
        return None

    async def _soap_call(
        self,
        endpoint: UpnpServiceEndpoint,
        action: str,
        inner_xml: str,
    ) -> str | None:
        """Execute a bounded SOAP XML POST against `endpoint.control_url`."""
        if not is_allowed_upnp_location(endpoint.control_url):
            return None
        envelope = (
            '<?xml version="1.0"?>'
            '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
            's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
            "<s:Body>"
            f'<u:{xml_escape(action)} xmlns:u="{xml_escape(endpoint.service_type)}">'
            f"{inner_xml}"
            f"</u:{xml_escape(action)}>"
            "</s:Body>"
            "</s:Envelope>"
        )
        headers = {
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPAction": f'"{endpoint.service_type}#{action}"',
        }
        timeout = httpx.Timeout(timeout=self._timeout_seconds, connect=self._timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            follow_redirects=False,
        ) as client:
            async with client.stream(
                "POST",
                endpoint.control_url,
                content=envelope.encode("utf-8"),
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    return None
                raw_bytes = await _read_bounded_response(resp, MAX_UPNP_DESCRIPTION_BYTES)
                if raw_bytes is None:
                    return None
                charset = resp.encoding or "utf-8"
                try:
                    return raw_bytes.decode(charset, errors="replace")
                except LookupError:
                    return raw_bytes.decode("utf-8", errors="replace")

    async def add_port_mapping(
        self,
        endpoint: UpnpServiceEndpoint,
        port: int,
        local_ip: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Register a TCP port forwarding rule (`ShardAi P2P`) on the UPnP router."""
        if not (1 <= port <= 65535):
            return False
        body = (
            "<NewRemoteHost></NewRemoteHost>"
            f"<NewExternalPort>{port}</NewExternalPort>"
            "<NewProtocol>TCP</NewProtocol>"
            f"<NewInternalPort>{port}</NewInternalPort>"
            f"<NewInternalClient>{xml_escape(local_ip)}</NewInternalClient>"
            "<NewEnabled>1</NewEnabled>"
            "<NewPortMappingDescription>ShardAi P2P</NewPortMappingDescription>"
            f"<NewLeaseDuration>{max(0, int(lease_seconds))}</NewLeaseDuration>"
        )
        result = await self._soap_call(endpoint, "AddPortMapping", body)
        return result is not None

    async def get_external_ip(self, endpoint: UpnpServiceEndpoint) -> str | None:
        """Query the router's external IPv4/IPv6 address via `GetExternalIPAddress`."""
        text = await self._soap_call(endpoint, "GetExternalIPAddress", "")
        if not text:
            return None
        raw_bytes = text.encode("utf-8")
        if _has_unsafe_xml_declarations(raw_bytes):
            return None
        try:
            root = ET.fromstring(raw_bytes)
        except ET.ParseError:
            return None
        for elem in root.iter():
            if _strip_xml_ns(elem.tag) == "NewExternalIPAddress":
                raw_ip = (elem.text or "").strip()
                if not raw_ip:
                    return None
                try:
                    return str(ipaddress.ip_address(raw_ip))
                except ValueError:
                    return None
        return None

    async def delete_port_mapping(
        self,
        endpoint: UpnpServiceEndpoint,
        port: int,
    ) -> bool:
        """Remove a previously mapped TCP port forwarding rule from the UPnP router."""
        if not (1 <= port <= 65535):
            return False
        body = (
            "<NewRemoteHost></NewRemoteHost>"
            f"<NewExternalPort>{port}</NewExternalPort>"
            "<NewProtocol>TCP</NewProtocol>"
        )
        result = await self._soap_call(endpoint, "DeletePortMapping", body)
        return result is not None


class UpnpPortMapper:
    """High-level non-raising lifecycle wrapper around `UpnpClient`."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._client = UpnpClient(transport=transport, timeout_seconds=timeout_seconds)
        self._endpoint: UpnpServiceEndpoint | None = None
        self._mapped_port: int | None = None

    @property
    def mapped_port(self) -> int | None:
        """Return the currently mapped TCP port, or `None`."""
        return self._mapped_port

    async def ensure_port_mapped(
        self,
        port: int,
        local_ip: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Attempt UPnP discovery and TCP port mapping; never raises on failure."""
        if not is_upnp_enabled():
            return False
        try:
            endpoint = self._endpoint or await self._client.discover_service()
            if endpoint is None:
                return False
            self._endpoint = endpoint
            resolved_local_ip = local_ip or _get_local_ipv4_for_gateway(endpoint.control_url)
            ok = await self._client.add_port_mapping(
                endpoint=endpoint,
                port=port,
                local_ip=resolved_local_ip,
                lease_seconds=lease_seconds,
            )
            if ok:
                self._mapped_port = port
            return ok
        except Exception as exc:
            logger.debug("UPnP port mapping unavailable: %s", exc)
            return False

    async def get_external_ip(self) -> str | None:
        """Best-effort query of the UPnP router's external IP; returns None on error."""
        if not is_upnp_enabled():
            return None
        try:
            endpoint = self._endpoint or await self._client.discover_service()
            if endpoint is None:
                return None
            self._endpoint = endpoint
            return await self._client.get_external_ip(endpoint)
        except Exception as exc:
            logger.debug("UPnP external IP query failed: %s", exc)
            return None

    async def cleanup(self) -> None:
        """Best-effort deletion of any active UPnP mapping; never raises."""
        port = self._mapped_port
        endpoint = self._endpoint
        self._mapped_port = None
        self._endpoint = None
        if port is None or endpoint is None:
            return
        try:
            await self._client.delete_port_mapping(endpoint=endpoint, port=port)
        except Exception as exc:
            logger.debug("UPnP cleanup failed: %s", exc)
