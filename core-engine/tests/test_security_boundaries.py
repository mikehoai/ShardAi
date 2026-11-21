"""Security boundary tests for Issue #19: SSRF, local path confinement, P2P origin, and local IPC auth."""

from __future__ import annotations

from array import array
from pathlib import Path
import struct
from typing import Any
import uuid

from fastapi.testclient import TestClient
import httpx
import pytest

import main as main_module
from main import LocalNodeRuntime, app, runtime
import models.downloader as downloader_module
from models.downloader import GGUF_MAGIC, ModelDownloader
from models.shard_runner import ShardModelRunner
from p2p.serializer import deserialize_activation_tensor, serialize_activation_tensor
from shard_engine.schemas import NodeState, PeerNodeInfo, PeerRole

client = TestClient(app)


def _write_valid_gguf(target: Path, size: int = 1_100_000) -> Path:
    """Create a minimal valid GGUF binary file at `target`."""
    header = bytearray(GGUF_MAGIC)
    header.extend(struct.pack("<IQQ", 3, 0, 0))
    if size > len(header):
        header.extend(b"\x00" * (size - len(header)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(header))
    return target


def _make_downloader(
    tmp_path: Path,
    transport: httpx.AsyncBaseTransport | None = None,
    allowed_local_dirs: list[Path | str] | None = None,
) -> ModelDownloader:
    return ModelDownloader(
        cache_dir=tmp_path / "cache_models",
        models_dir=tmp_path / "models",
        catalog_cache_path=tmp_path / "models_cache.json",
        http_transport=transport,
        allowed_local_dirs=allowed_local_dirs,
    )


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Reset runtime state and security env vars around each test."""
    orig_downloader = runtime.model_downloader
    orig_tracker = runtime.tracker_client
    monkeypatch.delenv("SHARDAI_LOCAL_IPC_TOKEN", raising=False)
    monkeypatch.delenv("SHARDAI_ALLOWED_MODEL_DIRS", raising=False)
    runtime.session_token = None
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []
    yield
    if runtime.state != NodeState.STOPPED:
        client.post("/stop-node")
    runtime.model_downloader = orig_downloader
    runtime.tracker_client = orig_tracker
    runtime.session_token = None
    runtime.state = NodeState.STOPPED
    runtime.peers = []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://127.0.0.1:8000/secret.gguf",
        "http://[::1]:8000/secret.gguf",
        "http://[::ffff:127.0.0.1]/secret.gguf",
        "http://localhost:9000/model.gguf",
        "http://169.254.169.254/latest/meta-data/model.gguf",
        "http://10.0.0.5/internal.gguf",
        "http://172.16.0.10/internal.gguf",
        "http://192.168.1.20/router.gguf",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
async def test_direct_url_rejects_ssrf_loopback_metadata_and_private_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_url: str,
) -> None:
    """Direct URL downloads must reject loopback, link-local metadata, and private IP targets."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    called = False

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(status_code=200, content=b"GGUF" + b"\x00" * 1_100_000)

    downloader = _make_downloader(tmp_path, transport=httpx.MockTransport(_handler))

    with pytest.raises(PermissionError):
        downloader.resolve_direct_url_path(blocked_url)

    with pytest.raises(PermissionError):
        await downloader.ensure_model_downloaded(blocked_url, source_type="direct_url")

    assert called is False


@pytest.mark.asyncio
async def test_direct_url_rejects_public_hostname_resolving_to_private_ip_and_malicious_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS resolution to private/metadata IPs and redirects to restricted IPs must be blocked."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")

    # Case 1: Hostname resolves via DNS to cloud metadata IP 169.254.169.254
    monkeypatch.setattr(
        downloader_module,
        "_resolve_hostname_ips",
        lambda _host, _port: ["169.254.169.254"],
    )
    called_dns = False

    def _dns_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called_dns
        called_dns = True
        return httpx.Response(status_code=200, content=b"GGUF")

    downloader_dns = _make_downloader(tmp_path / "dns", transport=httpx.MockTransport(_dns_handler))
    with pytest.raises(PermissionError, match="restricted IP"):
        await downloader_dns.ensure_model_downloaded(
            "https://models.public.example/evil.gguf",
            source_type="direct_url",
        )
    assert called_dns is False

    # Case 2: Public domain resolves to public IP, then returns 302 redirect to 169.254.169.254 / 127.0.0.1
    monkeypatch.setattr(
        downloader_module,
        "_resolve_hostname_ips",
        lambda host, _port: ["93.184.216.34"] if host == "models.public.example" else ["127.0.0.1"],
    )
    seen_urls: list[str] = []

    def _redirect_handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if str(request.url).startswith("https://models.public.example/"):
            return httpx.Response(
                status_code=302,
                headers={"Location": "http://169.254.169.254/latest/meta-data"},
            )
        return httpx.Response(status_code=200, content=b"SECRET")

    downloader_redir = _make_downloader(
        tmp_path / "redir",
        transport=httpx.MockTransport(_redirect_handler),
    )
    with pytest.raises(PermissionError):
        await downloader_redir.ensure_model_downloaded(
            "https://models.public.example/start.gguf",
            source_type="direct_url",
        )
    assert seen_urls == ["https://models.public.example/start.gguf"]


@pytest.mark.asyncio
async def test_local_file_enforces_approved_root_confinement_and_gguf_magic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`local_file` must reject traversal/outside files and non-GGUF binaries, while allowing approved roots."""
    downloader = _make_downloader(tmp_path)

    outside_dir = tmp_path / "outside"
    outside_gguf = _write_valid_gguf(outside_dir / "secret-model.gguf", size=128)
    inside_gguf = _write_valid_gguf(downloader.models_dir / "allowed-model.gguf", size=128)
    bad_magic_inside = downloader.models_dir / "not-gguf.gguf"
    bad_magic_inside.write_bytes(b"<!DOCTYPE html>not gguf")

    # Outside file rejected via ensure_model_downloaded
    with pytest.raises(PermissionError, match="outside approved model directories"):
        await downloader.ensure_model_downloaded(str(outside_gguf), source_type="local_file")

    # Relative path traversal escaping models_dir rejected
    traversal_path = str(downloader.models_dir / ".." / "outside" / "secret-model.gguf")
    with pytest.raises(PermissionError, match="outside approved model directories"):
        await downloader.ensure_model_downloaded(traversal_path, source_type="local_file")

    # Non-GGUF inside approved root rejected
    with pytest.raises(ValueError, match="missing 4-byte GGUF magic header"):
        await downloader.ensure_model_downloaded(str(bad_magic_inside), source_type="local_file")

    # Valid GGUF inside models_dir succeeds
    resolved_ok = await downloader.ensure_model_downloaded(str(inside_gguf), source_type="local_file")
    assert resolved_ok == inside_gguf.resolve()

    # Explicitly allowed directory via SHARDAI_ALLOWED_MODEL_DIRS succeeds
    monkeypatch.setenv("SHARDAI_ALLOWED_MODEL_DIRS", str(outside_dir))
    resolved_env = await downloader.ensure_model_downloaded(
        str(outside_gguf), source_type="local_file"
    )
    assert resolved_env == outside_gguf.resolve()

    # HTTP endpoints also reject outside local_file with 403 Forbidden
    monkeypatch.delenv("SHARDAI_ALLOWED_MODEL_DIRS", raising=False)
    runtime.model_downloader = downloader
    stream_res = client.get(
        "/models/download-stream",
        params={"model_id": str(outside_gguf), "source_type": "local_file"},
    )
    assert stream_res.status_code == 403

    start_res = client.post(
        "/start-node",
        json={
            "node_id": "traversal-test-node",
            "model_id": str(outside_gguf),
            "source_type": "local_file",
            "network_mode": "lan",
            "ram_gb": 4,
        },
    )
    assert start_res.status_code == 403


@pytest.mark.asyncio
async def test_p2p_activation_rejects_unauthorized_origin_before_outgoing_tcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tail/middle node must reject activation frames with unverified origin_ip/port before send_activations."""
    sent_targets: list[tuple[str, int]] = []

    async def _spy_send_activations(
        peer_ip: str,
        byte_stream: bytes,
        peer_port: int = 49152,
        timeout_seconds: float = 10.0,
    ) -> float:
        sent_targets.append((peer_ip, peer_port))
        return 1.0

    monkeypatch.setattr(main_module, "send_activations", _spy_send_activations)

    tail = LocalNodeRuntime()
    tail.state = NodeState.RUNNING
    tail.shard_layer_start = 16
    tail.shard_layer_end = 31
    tail.total_layers = 32
    tail.model_runner = ShardModelRunner(
        shard_layer_start=16,
        shard_layer_end=31,
        total_layers=32,
    )

    tensor_bytes = array("f", [0.1] * 16).tobytes()
    req_id = str(uuid.uuid4())

    def _make_frame(origin_ip: str, origin_port: int) -> Any:
        raw = serialize_activation_tensor(
            tensor_data=tensor_bytes,
            shape=(1, 16),
            request_id=req_id,
            origin_ip=origin_ip,
            origin_port=origin_port,
            current_layer=15,
            total_layers=32,
            step_index=0,
            max_tokens=4,
        )
        return deserialize_activation_tensor(raw)

    # 1. Empty peer routing table -> reject even 127.0.0.1
    tail.peers = []
    with pytest.raises(RuntimeError, match="unauthorized origin"):
        await tail.handle_incoming_activation(_make_frame("127.0.0.1", 49152))
    assert sent_targets == []

    # 2. Register legitimate head node (layers 0..15) and another middle node
    tail.peers = [
        PeerNodeInfo(
            peer_id="legit-head",
            ip_address="127.0.0.1",
            port=49501,
            shard_layer_start=0,
            shard_layer_end=15,
            role=PeerRole.PRIMARY,
        ),
        PeerNodeInfo(
            peer_id="other-peer",
            ip_address="203.0.113.50",
            port=49502,
            shard_layer_start=8,
            shard_layer_end=15,
            role=PeerRole.BACKUP,
        ),
    ]

    # Spoofed private/metadata target IP or mismatched port must be rejected without TCP send
    for bad_ip, bad_port in [
        ("169.254.169.254", 80),
        ("10.0.0.99", 6379),
        ("127.0.0.1", 9999),
        ("203.0.113.50", 49502),
    ]:
        with pytest.raises(RuntimeError, match="unauthorized origin"):
            await tail.handle_incoming_activation(_make_frame(bad_ip, bad_port))
    assert sent_targets == []

    # 3. Authorized head node origin succeeds and sends token back to head
    await tail.handle_incoming_activation(_make_frame("127.0.0.1", 49501))
    assert sent_targets == [("127.0.0.1", 49501)]


def test_local_ipc_token_enforcement_on_mutating_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When SHARDAI_LOCAL_IPC_TOKEN is set, mutating local endpoints require X-ShardAi-Local-Token."""
    monkeypatch.setenv("SHARDAI_LOCAL_IPC_TOKEN", "local-secret-token-xyz")
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "true")

    # Missing or wrong token returns 401
    assert (
        client.get(
            "/models/download-stream",
            params={"model_id": "Qwen/Qwen2.5-7B-Instruct"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/start-node",
            headers={"X-ShardAi-Local-Token": "wrong-token"},
            json={"model_id": "Qwen/Qwen2.5-7B-Instruct", "network_mode": "lan", "ram_gb": 4},
        ).status_code
        == 401
    )
    assert client.post("/stop-node").status_code == 401
    assert (
        client.post(
            "/peers/register",
            json={
                "peer_id": "p1",
                "ip_address": "127.0.0.1",
                "port": 49152,
                "shard_layer_start": 0,
                "shard_layer_end": 15,
            },
        ).status_code
        == 401
    )

    # Correct X-ShardAi-Local-Token succeeds
    valid_headers = {"X-ShardAi-Local-Token": "local-secret-token-xyz"}
    reg_res = client.post(
        "/peers/register",
        headers=valid_headers,
        json={
            "peer_id": "p1",
            "ip_address": "127.0.0.1",
            "port": 49152,
            "shard_layer_start": 0,
            "shard_layer_end": 15,
        },
    )
    assert reg_res.status_code == 200
    assert len(reg_res.json()) == 1

    stop_res = client.post("/stop-node", headers=valid_headers)
    assert stop_res.status_code == 200
