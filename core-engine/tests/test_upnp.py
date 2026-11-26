"""Unit tests for the best-effort UPnP IGD port-mapping client (`p2p.upnp`)."""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

import p2p.upnp as upnp_module
from p2p.upnp import UpnpClient, UpnpPortMapper, is_allowed_upnp_location

_DEVICE_DESC_XML = b"""<?xml version="1.0"?>
<root xmlns="urn:schemas-upnp-org:device-1-0">
  <device>
    <deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType>
    <deviceList>
      <device>
        <deviceType>urn:schemas-upnp-org:device:WANDevice:1</deviceType>
        <deviceList>
          <device>
            <deviceType>urn:schemas-upnp-org:device:WANConnectionDevice:1</deviceType>
            <serviceList>
              <service>
                <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
                <controlURL>/ctl/IPConn</controlURL>
              </service>
            </serviceList>
          </device>
        </deviceList>
      </device>
    </deviceList>
  </device>
</root>
"""


class _FakeSsdpSocket:
    """Minimal mock UDP socket returning scripted SSDP M-SEARCH replies."""

    def __init__(self, responses: list[bytes], *, timeout_on_recv: bool = False) -> None:
        self._responses = list(responses)
        self._timeout_on_recv = timeout_on_recv
        self.sent_probes: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def settimeout(self, _timeout: float) -> None:
        return None

    def sendto(self, data: bytes, addr: tuple[str, int]) -> int:
        self.sent_probes.append((data, addr))
        return len(data)

    def recvfrom(self, _bufsize: int) -> tuple[bytes, tuple[str, int]]:
        if self._timeout_on_recv or not self._responses:
            raise TimeoutError("timed out")
        return self._responses.pop(0), ("192.168.1.1", 1900)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://192.168.1.1:5000/rootDesc.xml", True),
        ("http://10.0.0.1:49152/igd.xml", True),
        ("http://172.16.5.1:1900/desc.xml", True),
        ("http://8.8.8.8:5000/rootDesc.xml", False),
        ("http://127.0.0.1:5000/rootDesc.xml", False),
        ("http://169.254.169.254/latest/meta-data", False),
        ("http://router.local/rootDesc.xml", False),
        ("ftp://192.168.1.1/rootDesc.xml", False),
        ("http://[::1]:5000/rootDesc.xml", False),
    ],
)
def test_is_allowed_upnp_location_enforces_private_rfc1918_ipv4(
    url: str, expected: bool
) -> None:
    """SSRF guard must accept only private RFC1918 IPv4 HTTP(S) locations."""
    assert is_allowed_upnp_location(url) is expected


@pytest.mark.asyncio
async def test_upnp_add_external_ip_and_delete_port_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked SSDP + SOAP transport must discover IGD, map port, read external IP, and delete."""
    ssdp_reply = (
        b"HTTP/1.1 200 OK\r\n"
        b"LOCATION: http://192.168.1.1:5000/rootDesc.xml\r\n"
        b"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
    )
    fake_ssdp = _FakeSsdpSocket([ssdp_reply])
    monkeypatch.setattr(upnp_module.socket, "socket", lambda *_args, **_kwargs: fake_ssdp)

    soap_actions: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and str(request.url) == "http://192.168.1.1:5000/rootDesc.xml":
            return httpx.Response(status_code=200, content=_DEVICE_DESC_XML)
        if request.method == "POST" and str(request.url) == "http://192.168.1.1:5000/ctl/IPConn":
            action = request.headers.get("SOAPAction", "")
            body = request.content.decode("utf-8")
            soap_actions.append((action, body))
            if "AddPortMapping" in action:
                assert "<NewExternalPort>49152</NewExternalPort>" in body
                assert "<NewInternalPort>49152</NewInternalPort>" in body
                assert "<NewProtocol>TCP</NewProtocol>" in body
                assert "<NewInternalClient>192.168.1.42</NewInternalClient>" in body
                assert "<NewPortMappingDescription>ShardAi P2P</NewPortMappingDescription>" in body
                return httpx.Response(status_code=200, content=b"<s:Envelope/>")
            if "GetExternalIPAddress" in action:
                return httpx.Response(
                    status_code=200,
                    content=(
                        b'<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
                        b"<s:Body>"
                        b'<u:GetExternalIPAddressResponse xmlns:u="urn:schemas-upnp-org:service:WANIPConnection:1">'
                        b"<NewExternalIPAddress>203.0.113.77</NewExternalIPAddress>"
                        b"</u:GetExternalIPAddressResponse>"
                        b"</s:Body></s:Envelope>"
                    ),
                )
            if "DeletePortMapping" in action:
                assert "<NewExternalPort>49152</NewExternalPort>" in body
                return httpx.Response(status_code=200, content=b"<s:Envelope/>")
        return httpx.Response(status_code=404)

    mapper = UpnpPortMapper(transport=httpx.MockTransport(_handler), timeout_seconds=1.0)
    mapped = await mapper.ensure_port_mapped(49152, local_ip="192.168.1.42", lease_seconds=3600)
    assert mapped is True
    assert mapper.mapped_port == 49152

    ext_ip = await mapper.get_external_ip()
    assert ext_ip == "203.0.113.77"

    await mapper.cleanup()
    assert mapper.mapped_port is None
    assert len(soap_actions) == 3
    assert fake_ssdp.closed is True


@pytest.mark.asyncio
async def test_upnp_rejects_public_location_and_oversized_xml_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public LOCATION URLs, oversized XML, and network errors must degrade cleanly without raising."""
    public_reply = (
        b"HTTP/1.1 200 OK\r\n"
        b"LOCATION: http://93.184.216.34:5000/evil.xml\r\n"
        b"ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
    )
    fake_ssdp = _FakeSsdpSocket([public_reply])
    monkeypatch.setattr(upnp_module.socket, "socket", lambda *_args, **_kwargs: fake_ssdp)

    http_called = False

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(status_code=200, content=_DEVICE_DESC_XML)

    mapper = UpnpPortMapper(transport=httpx.MockTransport(_handler), timeout_seconds=1.0)
    assert await mapper.ensure_port_mapped(49152, local_ip="192.168.1.42") is False
    assert http_called is False

    # Now test private location returning > 64 KiB XML
    private_reply = (
        b"HTTP/1.1 200 OK\r\n"
        b"LOCATION: http://192.168.1.1:5000/huge.xml\r\n\r\n"
    )
    monkeypatch.setattr(
        upnp_module.socket,
        "socket",
        lambda *_args, **_kwargs: _FakeSsdpSocket([private_reply]),
    )

    def _huge_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"<root>" + b"x" * 70_000 + b"</root>")

    client = UpnpClient(transport=httpx.MockTransport(_huge_handler), timeout_seconds=1.0)
    assert await client.discover_service() is None

    # Socket errors / timeouts must never raise
    def _boom_socket(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("UDP disabled")

    monkeypatch.setattr(upnp_module.socket, "socket", _boom_socket)
    mapper_err = UpnpPortMapper(timeout_seconds=0.2)
    assert await mapper_err.ensure_port_mapped(49152) is False
    assert await mapper_err.get_external_ip() is None
    await mapper_err.cleanup()
