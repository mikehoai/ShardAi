"""Unit and integration tests for Issue #8: model downloader catalog resolution, GGUF validation, and cache fallback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import hashlib
import json
from pathlib import Path
import struct
import threading
from typing import Any, get_args
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

import main as main_module
from main import app, runtime
import models.downloader as downloader_module
from models.downloader import (
    DEFAULT_CATALOG_MODELS,
    FINGERPRINT_HEAD_BYTES,
    GGUF_MAGIC,
    VALID_CATALOG_TASKS,
    DownloadProgressEvent,
    ModelDownloader,
    compute_model_fingerprint,
    compute_model_fingerprint_async,
    has_gguf_magic,
    is_valid_gguf_file,
    sanitize_catalog_entry,
)
from shard_engine.schemas import CatalogModelItem, CatalogModelTask, NodeState
from shard_engine.tracker_client import SignalingTrackerClient
from tests._mock_runner import patch_start_node_runner

client = TestClient(app)


def _build_synthetic_gguf_bytes(
    *,
    block_count: int = 40,
    total_size: int = 1_100_000,
) -> bytes:
    """Create a valid GGUF v3 binary header with `<arch>.block_count` padded to `total_size`."""
    key_bytes = b"llama.block_count"
    header = bytearray()
    header.extend(GGUF_MAGIC)
    header.extend(struct.pack("<IQQ", 3, 0, 1))  # version=3, tensor_count=0, kv_count=1
    header.extend(struct.pack("<Q", len(key_bytes)))
    header.extend(key_bytes)
    header.extend(struct.pack("<II", 4, block_count))  # GGUF_TYPE_UINT32 = 4
    if total_size > len(header):
        header.extend(b"\x00" * (total_size - len(header)))
    return bytes(header)


def _write_synthetic_gguf(
    target_path: Path,
    *,
    block_count: int = 40,
    total_size: int = 1_100_000,
) -> Path:
    """Write a synthetic valid GGUF v3 file to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(
        _build_synthetic_gguf_bytes(block_count=block_count, total_size=total_size)
    )
    return target_path


def _make_downloader(tmp_path: Path, transport: httpx.AsyncBaseTransport | None = None) -> ModelDownloader:
    """Create a completely isolated `ModelDownloader` inside `tmp_path`."""
    return ModelDownloader(
        cache_dir=tmp_path / "cache_models",
        models_dir=tmp_path / "models",
        catalog_cache_path=tmp_path / "models_cache.json",
        http_transport=transport,
    )


def _make_mock_tracker(handler: Any) -> SignalingTrackerClient:
    return SignalingTrackerClient(
        base_url="https://mock.tracker.local",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=1,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture(autouse=True)
def isolate_runtime_state() -> Any:
    """Isolate `runtime.model_downloader`, `runtime.tracker_client`, and node lifecycle per test."""
    original_downloader = runtime.model_downloader
    original_tracker = runtime.tracker_client
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []
    runtime.network_status = None
    runtime.last_tracker_reachable = None
    yield
    if runtime.state != NodeState.STOPPED:
        client.post("/stop-node")
    runtime.model_downloader = original_downloader
    runtime.tracker_client = original_tracker
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


def test_resolve_gguf_in_models_dir_resolves_catalog_id_and_repo_id_when_filename_omitted(
    tmp_path: Path,
) -> None:
    """Catalog `id`, `repo_id`, and HF URL resolve canonical filenames while blocking path traversal."""
    downloader = _make_downloader(tmp_path)

    expected = (downloader.models_dir / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf").resolve()
    assert downloader.resolve_gguf_in_models_dir("deepseek-r1-distill-llama-8b", None) == expected
    assert downloader.resolve_gguf_in_models_dir("deepseek-r1-distill-llama-8b", "   ") == expected
    assert (
        downloader.resolve_gguf_in_models_dir("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", None)
        == expected
    )
    assert (
        downloader.resolve_gguf_in_models_dir(
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
            None,
        )
        == expected
    )

    # Custom catalog cache entry with a potentially dangerous relative filename stays inside models_dir
    downloader.save_catalog_cache(
        [
            {
                "id": "custom-cached-model",
                "name": "Custom Cached Model",
                "repo_id": "custom-org/Custom-Model-GGUF",
                "filename": "../../evil_sub/Custom-Model-Q4_K_M.gguf",
                "estimated_ram_gb": 4.2,
                "enabled": True,
            }
        ]
    )
    resolved_custom = downloader.resolve_gguf_in_models_dir("custom-cached-model", None)
    assert resolved_custom.is_relative_to(downloader.models_dir)
    assert resolved_custom.name == "Custom-Model-Q4_K_M.gguf"
    assert (
        downloader.resolve_gguf_in_models_dir("custom-org/Custom-Model-GGUF", None)
        == resolved_custom
    )

    # Explicit filename override still takes precedence and prevents traversal outside models_dir
    explicit = downloader.resolve_gguf_in_models_dir(
        "deepseek-r1-distill-llama-8b",
        "../../override-quant.gguf",
    )
    assert explicit.is_relative_to(downloader.models_dir)
    assert explicit == (downloader.models_dir / "override-quant.gguf").resolve()


def test_start_node_step1_resolves_cached_catalog_gguf_fingerprint_and_header_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POST /start-node` Step 1 locates cached catalog GGUF with `filename=None` for fingerprint and layer count."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    patch_start_node_runner(monkeypatch)

    downloader = _make_downloader(tmp_path)
    gguf_path = downloader.models_dir / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    _write_synthetic_gguf(gguf_path, block_count=40, total_size=1_100_000)
    expected_fp = compute_model_fingerprint(gguf_path)

    runtime.model_downloader = downloader
    announce_payloads: list[dict[str, Any]] = []

    def _tracker_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/announce"):
            payload = json.loads(request.content.decode("utf-8"))
            announce_payloads.append(payload)
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 39,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and "/rooms" in request.url.path:
            return httpx.Response(status_code=200, json={"rooms": []})
        return httpx.Response(status_code=200, json={"peers": []})

    runtime.tracker_client = _make_mock_tracker(_tracker_handler)

    res = client.post(
        "/start-node",
        json={
            "node_id": "catalog-step1-node",
            "model_id": "deepseek-r1-distill-llama-8b",
            "filename": None,
            "source_type": "catalog",
            "network_mode": "internet",
            "room_id": "catalog-probe-room",
            "ram_gb": 8,
            "p2p_listen_port": 49410,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["total_layers"] == 40
    assert len(announce_payloads) >= 1
    initial_announce = announce_payloads[0]
    assert initial_announce["model_fingerprint"] == expected_fp
    assert initial_announce["total_model_layers"] == 40

    client.post("/stop-node")

    # Corrupted non-GGUF file >= 1 MB at the same path must NOT be probed or fingerprinted in Step 1
    gguf_path.write_bytes(b"<html>bad gateway error page</html>" + (b"x" * 1_100_000))
    announce_payloads.clear()

    fallback_valid = _write_synthetic_gguf(
        tmp_path / "repaired.gguf",
        block_count=32,
        total_size=1_100_000,
    )
    monkeypatch.setattr(
        downloader,
        "ensure_model_downloaded",
        AsyncMock(return_value=fallback_valid),
    )

    res_corrupt = client.post(
        "/start-node",
        json={
            "node_id": "catalog-step1-corrupt-node",
            "model_id": "deepseek-r1-distill-llama-8b",
            "filename": None,
            "source_type": "catalog",
            "network_mode": "internet",
            "room_id": "catalog-probe-room-2",
            "ram_gb": 8,
            "p2p_listen_port": 49411,
        },
    )
    assert res_corrupt.status_code == 200
    assert len(announce_payloads) >= 1
    assert announce_payloads[0].get("model_fingerprint") is None


@pytest.mark.asyncio
async def test_direct_url_rejects_truncated_download_and_cleans_up_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncated direct downloads (`downloaded != Content-Length`) fail and leave no partial artifacts."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    url = "https://cdn.example.test/models/tiny-q4.gguf"
    partial_gguf = _build_synthetic_gguf_bytes(block_count=24, total_size=1_200_000)

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": "2097152"},
            content=partial_gguf,
        )

    downloader = _make_downloader(tmp_path, transport=httpx.MockTransport(_handler))
    target_path = downloader.resolve_direct_url_path(url)
    part_path = target_path.with_suffix(target_path.suffix + ".part")

    with pytest.raises(RuntimeError, match="Incomplete GGUF download"):
        await downloader.ensure_model_downloaded(url, source_type="direct_url")

    assert not target_path.exists()
    assert not part_path.exists()


@pytest.mark.asyncio
async def test_direct_url_rejects_non_gguf_payload_over_1mb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTML/JSON payloads >= 1 MB without `GGUF` magic are rejected and never cached."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    url = "https://cdn.example.test/models/error-page.gguf"
    html_payload = b"<!DOCTYPE html><html><body>502 Bad Gateway</body></html>" + (b" " * 1_200_000)

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": str(len(html_payload))},
            content=html_payload,
        )

    downloader = _make_downloader(tmp_path, transport=httpx.MockTransport(_handler))
    target_path = downloader.resolve_direct_url_path(url)
    part_path = target_path.with_suffix(target_path.suffix + ".part")

    with pytest.raises(ValueError, match="not a valid GGUF binary"):
        await downloader.ensure_model_downloaded(url, source_type="direct_url")

    assert not target_path.exists()
    assert not part_path.exists()


@pytest.mark.asyncio
async def test_direct_url_chunked_without_content_length_enforces_gguf_magic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunked direct URL responses without `Content-Length` still enforce `GGUF` magic."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")

    class _NoLengthStream(httpx.AsyncByteStream):
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self._payload[:512_000]
            yield self._payload[512_000:]

    valid_url = "https://cdn.example.test/models/chunked-valid.gguf"
    valid_bytes = _build_synthetic_gguf_bytes(block_count=24, total_size=1_100_000)

    def _valid_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, stream=_NoLengthStream(valid_bytes))

    downloader_ok = _make_downloader(tmp_path / "ok", transport=httpx.MockTransport(_valid_handler))
    ok_path = await downloader_ok.ensure_model_downloaded(valid_url, source_type="direct_url")
    assert ok_path.is_file()
    assert has_gguf_magic(ok_path)

    bad_url = "https://cdn.example.test/models/chunked-bad.gguf"
    bad_bytes = b"not-gguf-chunked-stream" + (b"z" * 1_100_000)

    def _bad_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, stream=_NoLengthStream(bad_bytes))

    downloader_bad = _make_downloader(tmp_path / "bad", transport=httpx.MockTransport(_bad_handler))
    bad_target = downloader_bad.resolve_direct_url_path(bad_url)
    with pytest.raises(ValueError, match="not a valid GGUF binary"):
        await downloader_bad.ensure_model_downloaded(bad_url, source_type="direct_url")
    assert not bad_target.exists()


@pytest.mark.asyncio
async def test_direct_url_redownloads_when_cached_file_over_1mb_lacks_gguf_magic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing >= 1 MB non-GGUF cache files trigger re-download instead of cache hit."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    url = "https://cdn.example.test/models/valid-model.gguf"
    valid_bytes = _build_synthetic_gguf_bytes(block_count=32, total_size=1_150_000)
    requests_seen = 0

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests_seen
        requests_seen += 1
        return httpx.Response(
            status_code=200,
            headers={"Content-Length": str(len(valid_bytes))},
            content=valid_bytes,
        )

    downloader = _make_downloader(tmp_path, transport=httpx.MockTransport(_handler))
    target_path = downloader.resolve_direct_url_path(url)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b'{"error": "corrupted json"}' + (b"0" * 1_200_000))
    assert not is_valid_gguf_file(target_path)

    events: list[DownloadProgressEvent] = []
    result_path = await downloader.ensure_model_downloaded(
        url,
        source_type="direct_url",
        on_event=events.append,
    )

    assert requests_seen == 1
    assert result_path == target_path
    assert has_gguf_magic(result_path)
    assert result_path.stat().st_size == len(valid_bytes)
    statuses = [evt.status for evt in events]
    assert "cached" not in statuses
    assert statuses[-1] == "completed"


@pytest.mark.asyncio
async def test_hf_catalog_and_is_model_cached_reject_non_gguf_over_1mb_and_redownload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`is_model_cached` and `_ensure_hf_or_catalog_downloaded` reject >= 1 MB non-GGUF files."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    downloader = _make_downloader(tmp_path)

    # 1. `is_model_cached` rejects >= 1 MB corrupted file in cache_dir
    _, legacy_path = downloader.resolve_safe_gguf_path("custom/bad-cache-model")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(b"NOT_GGUF_DATA" + (b"\x01" * 1_100_000))
    assert downloader.is_model_cached("custom/bad-cache-model")[0] is False

    # 2. Corrupted >= 1 MB file in models_dir is unlinked and re-downloaded via hf_hub_download
    catalog_target = downloader.resolve_gguf_in_models_dir("deepseek-r1-distill-llama-8b", None)
    catalog_target.parent.mkdir(parents=True, exist_ok=True)
    catalog_target.write_bytes(b"<html>corrupted cache</html>" + (b"x" * 1_100_000))

    hf_calls: list[tuple[str, str, str]] = []
    valid_bytes = _build_synthetic_gguf_bytes(block_count=32, total_size=1_120_000)

    def _fake_hf_hub_download(**kwargs: Any) -> str:
        repo_id = str(kwargs["repo_id"])
        filename = str(kwargs["filename"])
        local_dir = Path(str(kwargs["local_dir"]))
        hf_calls.append((repo_id, filename, str(local_dir)))
        assert not (local_dir / filename).exists(), "Corrupted file must be unlinked before re-download"
        out_path = local_dir / filename
        out_path.write_bytes(valid_bytes)
        return str(out_path)

    monkeypatch.setattr(downloader_module, "hf_hub_download", _fake_hf_hub_download)

    events: list[DownloadProgressEvent] = []
    downloaded_path = await downloader.ensure_model_downloaded(
        "deepseek-r1-distill-llama-8b",
        source_type="catalog",
        on_event=events.append,
    )

    assert len(hf_calls) == 1
    assert hf_calls[0][0] == "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"
    assert hf_calls[0][1] == "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    assert downloaded_path == catalog_target
    assert has_gguf_magic(downloaded_path)
    statuses = [evt.status for evt in events]
    assert "cached" not in statuses
    assert statuses[-1] == "completed"

    # 3. If hf_hub_download returns a non-GGUF payload, reject and unlink it
    catalog_target.unlink(missing_ok=True)

    def _bad_hf_hub_download(**kwargs: Any) -> str:
        out_path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        out_path.write_bytes(b"<html>404 from mirror</html>" + (b"y" * 1_100_000))
        return str(out_path)

    monkeypatch.setattr(downloader_module, "hf_hub_download", _bad_hf_hub_download)
    with pytest.raises(ValueError, match="not a valid GGUF binary"):
        await downloader.ensure_model_downloaded(
            "deepseek-r1-distill-llama-8b",
            source_type="catalog",
        )
    assert not catalog_target.exists()


def test_catalog_endpoint_handles_malformed_estimated_ram_gb_without_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /models/catalog` safely defaults invalid `estimated_ram_gb` values without HTTP 500."""
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "false")
    downloader = _make_downloader(tmp_path)
    downloader.catalog_cache_path.parent.mkdir(parents=True, exist_ok=True)
    downloader.catalog_cache_path.write_text(
        json.dumps(
            {
                "updated_at": 1700000000,
                "models": [
                    {
                        "id": "bad-str",
                        "name": "Bad String RAM",
                        "repo_id": "org/bad-str-gguf",
                        "estimated_ram_gb": "invalid",
                        "enabled": True,
                    },
                    {
                        "id": "bad-nan",
                        "name": "NaN RAM",
                        "repo_id": "org/bad-nan-gguf",
                        "estimated_ram_gb": "NaN",
                        "enabled": True,
                    },
                    {
                        "id": "bad-inf",
                        "name": "Inf RAM",
                        "repo_id": "org/bad-inf-gguf",
                        "estimated_ram_gb": "inf",
                        "enabled": True,
                    },
                    {
                        "id": "bad-neg",
                        "name": "Negative RAM",
                        "repo_id": "org/bad-neg-gguf",
                        "estimated_ram_gb": -2.5,
                        "enabled": True,
                    },
                    {
                        "id": "bad-bool",
                        "name": "Bool RAM",
                        "repo_id": "org/bad-bool-gguf",
                        "estimated_ram_gb": True,
                        "enabled": True,
                    },
                    {
                        "id": "bad-obj",
                        "name": "Dict RAM",
                        "repo_id": "org/bad-obj-gguf",
                        "estimated_ram_gb": {"bad": 1},
                        "enabled": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime.model_downloader = downloader

    res = client.get("/models/catalog")
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "cache"
    assert body["count"] == 6
    assert all(item["estimated_ram_gb"] == 5.0 for item in body["models"])


def test_catalog_endpoint_falls_back_to_default_when_cache_lacks_valid_repo_id_or_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /models/catalog` falls back to `DEFAULT_CATALOG_MODELS` when cache has no valid `repo_id` or bad JSON."""
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "false")
    downloader = _make_downloader(tmp_path)
    runtime.model_downloader = downloader

    # 1. Cache has entries, but none have a valid non-empty string `repo_id`
    downloader.catalog_cache_path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "missing-repo", "name": "Missing Repo", "estimated_ram_gb": 4.0},
                    {"id": "blank-repo", "repo_id": "   ", "enabled": True},
                    {"id": "int-repo", "repo_id": 12345, "enabled": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert downloader.load_catalog_cache() == []

    res_missing = client.get("/models/catalog")
    assert res_missing.status_code == 200
    body_missing = res_missing.json()
    assert body_missing["source"] == "default"
    assert body_missing["count"] == len(DEFAULT_CATALOG_MODELS)
    assert len(body_missing["models"]) == len(DEFAULT_CATALOG_MODELS)

    # 2. Corrupted JSON in models_cache.json
    downloader.catalog_cache_path.write_text("{corrupted-json", encoding="utf-8")
    assert downloader.load_catalog_cache() == []

    res_corrupt = client.get("/models/catalog")
    assert res_corrupt.status_code == 200
    body_corrupt = res_corrupt.json()
    assert body_corrupt["source"] == "default"
    assert body_corrupt["count"] == len(DEFAULT_CATALOG_MODELS)

    # 3. Cloud tracker returning only malformed entries must fall back cleanly without clobbering valid cache
    valid_cached_payload = {
        "updated_at": 1700000000,
        "models": [
            {
                "id": "cached-safe-model",
                "name": "Cached Safe Model",
                "repo_id": "safe-org/Safe-Model-GGUF",
                "filename": "safe-model.gguf",
                "estimated_ram_gb": 6.5,
                "enabled": True,
            }
        ],
    }
    downloader.catalog_cache_path.write_text(
        json.dumps(valid_cached_payload),
        encoding="utf-8",
    )
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")

    def _malformed_cloud_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "models": [
                    {"id": "broken-cloud", "name": "No Repo"},
                    {"id": "blank-cloud", "repo_id": ""},
                ]
            },
        )

    runtime.tracker_client = _make_mock_tracker(_malformed_cloud_handler)
    res_cloud = client.get("/models/catalog")
    assert res_cloud.status_code == 200
    body_cloud = res_cloud.json()
    assert body_cloud["source"] == "cache"
    assert body_cloud["count"] == 1
    assert body_cloud["models"][0]["repo_id"] == "safe-org/Safe-Model-GGUF"


@pytest.mark.asyncio
async def test_start_node_fingerprint_computation_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 1 and Step 4 fingerprint disk reads run off the event loop without stalling `/health` or `/status`."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    patch_start_node_runner(monkeypatch)

    downloader = _make_downloader(tmp_path)
    gguf_path = downloader.models_dir / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    _write_synthetic_gguf(gguf_path, block_count=40, total_size=1_100_000)
    expected_fp = compute_model_fingerprint(gguf_path)
    runtime.model_downloader = downloader

    def _tracker_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/announce"):
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 39,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and "/rooms" in request.url.path:
            return httpx.Response(status_code=200, json={"rooms": []})
        return httpx.Response(status_code=200, json={"peers": []})

    runtime.tracker_client = _make_mock_tracker(_tracker_handler)

    loop_thread_id = threading.get_ident()
    worker_thread_ids: list[int] = []
    step1_entered = threading.Event()
    step1_release = threading.Event()
    step4_entered = threading.Event()
    step4_release = threading.Event()
    real_fingerprint = compute_model_fingerprint

    def _blocking_fingerprint_probe(model_path: Path) -> str:
        call_idx = len(worker_thread_ids)
        worker_thread_ids.append(threading.get_ident())
        if call_idx == 0:
            step1_entered.set()
            assert step1_release.wait(timeout=5.0)
        elif call_idx == 1:
            step4_entered.set()
            assert step4_release.wait(timeout=5.0)
        return real_fingerprint(model_path)

    monkeypatch.setattr(main_module, "compute_model_fingerprint", _blocking_fingerprint_probe)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as async_client:
        start_task = asyncio.create_task(
            async_client.post(
                "/start-node",
                json={
                    "node_id": "fp-nonblock-node",
                    "model_id": "deepseek-r1-distill-llama-8b",
                    "filename": None,
                    "source_type": "catalog",
                    "network_mode": "internet",
                    "room_id": "fp-nonblock-room",
                    "ram_gb": 8,
                    "p2p_listen_port": 49431,
                },
            )
        )
        try:
            assert await asyncio.to_thread(step1_entered.wait, 5.0)
            health_res = await asyncio.wait_for(async_client.get("/health"), timeout=1.0)
            status_res = await asyncio.wait_for(async_client.get("/status"), timeout=1.0)
            assert health_res.status_code == 200
            assert status_res.status_code == 200
            assert status_res.json()["state"] == NodeState.STARTING.value
            step1_release.set()

            assert await asyncio.to_thread(step4_entered.wait, 5.0)
            health_res_step4 = await asyncio.wait_for(async_client.get("/health"), timeout=1.0)
            status_res_step4 = await asyncio.wait_for(async_client.get("/status"), timeout=1.0)
            assert health_res_step4.status_code == 200
            assert status_res_step4.status_code == 200
            assert status_res_step4.json()["state"] == NodeState.STARTING.value
            step4_release.set()

            start_res = await asyncio.wait_for(start_task, timeout=5.0)
            assert start_res.status_code == 200
            assert start_res.json()["model_fingerprint"] == expected_fp
            assert len(worker_thread_ids) == 2
            assert all(tid != loop_thread_id for tid in worker_thread_ids)
        finally:
            step1_release.set()
            step4_release.set()
            if not start_task.done():
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass
            if runtime.state != NodeState.STOPPED:
                await async_client.post("/stop-node")


@pytest.mark.asyncio
async def test_compute_model_fingerprint_chunked_large_file_dir_and_async_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunked prefix hashing matches reference SHA-256 for large files, dirs, and async offload."""
    payload = (
        (b"\x11" * (5 * 1024 * 1024))
        + (b"\x22" * (FINGERPRINT_HEAD_BYTES - 5 * 1024 * 1024))
        + (b"\x33" * 4096)
    )
    large_file = tmp_path / "large-model.gguf"
    large_file.write_bytes(payload)

    expected_file_fp = hashlib.sha256(
        f"file:{len(payload)}:".encode("utf-8") + payload[:FINGERPRINT_HEAD_BYTES]
    ).hexdigest()
    assert compute_model_fingerprint(large_file) == expected_file_fp

    # Mutating tail bytes beyond FINGERPRINT_HEAD_BYTES (with same total size) preserves digest
    tail_modified = tmp_path / "large-model-tail-modified.gguf"
    tail_modified.write_bytes(payload[:-128] + (b"\x99" * 128))
    assert compute_model_fingerprint(tail_modified) == expected_file_fp

    # Mutating a byte inside the 10 MB prefix (beyond the first 1 MiB chunk) changes digest
    mid_modified = tmp_path / "large-model-mid-modified.gguf"
    mid_bytes = bytearray(payload)
    mid_bytes[5 * 1024 * 1024] ^= 0xFF
    mid_modified.write_bytes(bytes(mid_bytes))
    assert compute_model_fingerprint(mid_modified) != expected_file_fp

    # Directory artifact: manifest + first 1 MiB of first weight/config file
    model_dir = tmp_path / "hf-dir"
    model_dir.mkdir()
    config_file = model_dir / "config.json"
    config_bytes = b'{"num_hidden_layers": 32}'
    config_file.write_bytes(config_bytes)
    shard_file = model_dir / "model-00001-of-00002.safetensors"
    shard_bytes = b"SAFETENSORS_HEADER" + (b"\x44" * 2_000_000)
    shard_file.write_bytes(shard_bytes)

    manifest = f"{config_file.name}:{len(config_bytes)}|{shard_file.name}:{len(shard_bytes)}"
    expected_dir_fp = hashlib.sha256(
        f"dir:{manifest}:".encode("utf-8") + config_bytes
    ).hexdigest()
    assert compute_model_fingerprint(model_dir) == expected_dir_fp

    with pytest.raises(FileNotFoundError):
        compute_model_fingerprint(tmp_path / "does-not-exist.gguf")

    # Verify async wrapper runs off the event loop thread and returns exact digest
    loop_thread_id = threading.get_ident()
    observed_threads: list[int] = []
    real_sync_fp = downloader_module.compute_model_fingerprint

    def _spy_fp(path: Path) -> str:
        observed_threads.append(threading.get_ident())
        return real_sync_fp(path)

    monkeypatch.setattr(downloader_module, "compute_model_fingerprint", _spy_fp)
    async_fp = await compute_model_fingerprint_async(large_file)
    assert async_fp == expected_file_fp
    assert len(observed_threads) == 1
    assert observed_threads[0] != loop_thread_id


@pytest.mark.asyncio
async def test_start_node_fingerprint_step1_error_fallback_and_step4_error_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 1 worker failure falls back to `initial_fingerprint=None`; Step 4 failure releases resources."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    patch_start_node_runner(monkeypatch)

    downloader = _make_downloader(tmp_path)
    gguf_path = downloader.models_dir / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
    _write_synthetic_gguf(gguf_path, block_count=40, total_size=1_100_000)
    expected_fp = compute_model_fingerprint(gguf_path)
    runtime.model_downloader = downloader

    announce_payloads: list[dict[str, Any]] = []

    def _tracker_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/announce"):
            announce_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                status_code=200,
                json={
                    "assigned_start_layer": 0,
                    "assigned_end_layer": 39,
                    "role": "primary",
                    "peers": [],
                },
            )
        if request.method == "GET" and "/rooms" in request.url.path:
            return httpx.Response(status_code=200, json={"rooms": []})
        return httpx.Response(status_code=200, json={"peers": []})

    runtime.tracker_client = _make_mock_tracker(_tracker_handler)
    real_fingerprint = compute_model_fingerprint

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as async_client:
        # Case A: Step 1 worker failure falls back cleanly and Step 4 syncs fingerprint
        calls_case_a = 0

        def _fail_step1_only(model_path: Path) -> str:
            nonlocal calls_case_a
            calls_case_a += 1
            if calls_case_a == 1:
                raise OSError("slow disk read error")
            return real_fingerprint(model_path)

        monkeypatch.setattr(main_module, "compute_model_fingerprint", _fail_step1_only)
        res_a = await async_client.post(
            "/start-node",
            json={
                "node_id": "fp-step1-fallback-node",
                "model_id": "deepseek-r1-distill-llama-8b",
                "filename": None,
                "source_type": "catalog",
                "network_mode": "internet",
                "room_id": "fp-fallback-room",
                "ram_gb": 8,
                "p2p_listen_port": 49432,
            },
        )
        assert res_a.status_code == 200
        assert res_a.json()["model_fingerprint"] == expected_fp
        assert len(announce_payloads) == 2
        assert "model_fingerprint" not in announce_payloads[0]
        assert announce_payloads[1]["model_fingerprint"] == expected_fp
        await async_client.post("/stop-node")

        # Case B: Step 4 worker failure aborts with HTTP 500, NodeState.ERROR, and 0 allocated RAM/VRAM
        announce_payloads.clear()
        calls_case_b = 0

        def _fail_step4_only(model_path: Path) -> str:
            nonlocal calls_case_b
            calls_case_b += 1
            if calls_case_b == 2:
                raise OSError("disk disappeared")
            return real_fingerprint(model_path)

        monkeypatch.setattr(main_module, "compute_model_fingerprint", _fail_step4_only)
        try:
            res_b = await async_client.post(
                "/start-node",
                json={
                    "node_id": "fp-step4-error-node",
                    "model_id": "deepseek-r1-distill-llama-8b",
                    "filename": None,
                    "source_type": "catalog",
                    "network_mode": "internet",
                    "room_id": "fp-error-room",
                    "ram_gb": 8,
                    "p2p_listen_port": 49433,
                },
            )
            assert res_b.status_code == 500
            assert "disk disappeared" in res_b.json()["detail"]
            assert runtime.state == NodeState.ERROR
            assert runtime.memory_manager.allocated_ram_mb == 0
            assert runtime.memory_manager.allocated_vram_mb == 0
        finally:
            if runtime.state != NodeState.STOPPED:
                await async_client.post("/stop-node")


def test_sanitize_catalog_entry_preserves_task_and_category_and_handles_invalid_inputs() -> None:
    """`sanitize_catalog_entry()` normalizes valid `task`/`category` and drops corrupt values safely."""
    assert VALID_CATALOG_TASKS == frozenset(get_args(CatalogModelTask))

    for task_input, expected_task in [
        ("text", "text"),
        ("  IMAGE  ", "image"),
        ("Video", "video"),
        ("multimodal", "multimodal"),
    ]:
        cleaned = sanitize_catalog_entry(
            {
                "id": f"model-{expected_task}",
                "name": f"Model {expected_task}",
                "repo_id": f"org/model-{expected_task}",
                "task": task_input,
                "category": "  image-generation  ",
                "enabled": True,
            }
        )
        assert cleaned is not None
        assert cleaned["task"] == expected_task
        assert cleaned["category"] == "image-generation"
        validated = CatalogModelItem.model_validate(cleaned)
        assert validated.task == expected_task
        assert validated.category == "image-generation"

    for bad_task in ("audio", "", "   ", 123, True, ["image"], None):
        for bad_category in ("", "   ", 456, False, {"x": 1}, None):
            cleaned_bad = sanitize_catalog_entry(
                {
                    "id": "safe-fallback-entry",
                    "name": "Safe Fallback Entry",
                    "repo_id": "org/safe-fallback",
                    "task": bad_task,
                    "category": bad_category,
                    "enabled": True,
                }
            )
            assert cleaned_bad is not None
            assert cleaned_bad["task"] is None
            assert cleaned_bad["category"] is None
            validated_bad = CatalogModelItem.model_validate(cleaned_bad)
            assert validated_bad.task is None
            assert validated_bad.category is None


def test_catalog_endpoint_and_cache_preserve_task_and_category_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /models/catalog` and `models_cache.json` persist and return `task` and `category`."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "true")
    downloader = _make_downloader(tmp_path)
    runtime.model_downloader = downloader

    remote_models = [
        {
            "id": "llama-3.1-8b",
            "name": "Llama 3.1 8B",
            "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "estimated_ram_gb": 5.1,
            "enabled": True,
            "task": "TEXT",
            "category": "reasoning",
        },
        {
            "id": "flux-schnell",
            "name": "FLUX.1 Schnell",
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "estimated_ram_gb": 12.0,
            "enabled": True,
            "task": "image",
            "category": "  image-generation  ",
        },
        {
            "id": "wan-2.1",
            "name": "Wan 2.1",
            "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "estimated_ram_gb": 8.5,
            "enabled": True,
            "task": "video",
            "category": "video-generation",
        },
        {
            "id": "qwen-vl",
            "name": "Qwen2.5-VL 7B",
            "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "estimated_ram_gb": 7.2,
            "enabled": True,
            "task": "multimodal",
            "category": "vision-language",
        },
        {
            "id": "unknown-meta",
            "name": "Unknown Meta",
            "repo_id": "org/unknown-meta",
            "estimated_ram_gb": 4.0,
            "enabled": True,
            "task": "audio",
            "category": "   ",
        },
    ]

    def _cloud_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/config/models"):
            return httpx.Response(status_code=200, json={"models": remote_models})
        return httpx.Response(status_code=404)

    runtime.tracker_client = _make_mock_tracker(_cloud_handler)

    # 1. Cloud sync preserves task & category in HTTP response and writes them to models_cache.json
    cloud_res = client.get("/models/catalog")
    assert cloud_res.status_code == 200
    cloud_body = cloud_res.json()
    assert cloud_body["source"] == "cloud"
    assert cloud_body["count"] == 5

    by_id = {item["id"]: item for item in cloud_body["models"]}
    assert by_id["llama-3.1-8b"]["task"] == "text"
    assert by_id["llama-3.1-8b"]["category"] == "reasoning"
    assert by_id["flux-schnell"]["task"] == "image"
    assert by_id["flux-schnell"]["category"] == "image-generation"
    assert by_id["wan-2.1"]["task"] == "video"
    assert by_id["wan-2.1"]["category"] == "video-generation"
    assert by_id["qwen-vl"]["task"] == "multimodal"
    assert by_id["qwen-vl"]["category"] == "vision-language"
    assert by_id["unknown-meta"]["task"] is None
    assert by_id["unknown-meta"]["category"] is None

    persisted_cache = json.loads(downloader.catalog_cache_path.read_text(encoding="utf-8"))
    persisted_by_id = {item["id"]: item for item in persisted_cache["models"]}
    assert persisted_by_id["flux-schnell"]["task"] == "image"
    assert persisted_by_id["flux-schnell"]["category"] == "image-generation"
    assert persisted_by_id["unknown-meta"]["task"] is None
    assert persisted_by_id["unknown-meta"]["category"] is None

    # 2. Cached reload when tracker disabled still preserves task & category
    monkeypatch.setenv("SHARDAI_ENABLE_TRACKER", "false")
    cached_res = client.get("/models/catalog")
    assert cached_res.status_code == 200
    cached_body = cached_res.json()
    assert cached_body["source"] == "cache"
    cached_by_id = {item["id"]: item for item in cached_body["models"]}
    assert cached_by_id["flux-schnell"]["task"] == "image"
    assert cached_by_id["flux-schnell"]["category"] == "image-generation"
    assert cached_by_id["wan-2.1"]["task"] == "video"
    assert cached_by_id["qwen-vl"]["task"] == "multimodal"

    # 3. Default catalog fallback includes task="text"
    downloader.catalog_cache_path.unlink()
    default_res = client.get("/models/catalog")
    assert default_res.status_code == 200
    default_body = default_res.json()
    assert default_body["source"] == "default"
    assert default_body["count"] == len(DEFAULT_CATALOG_MODELS)
    assert all(item["task"] == "text" for item in default_body["models"])
