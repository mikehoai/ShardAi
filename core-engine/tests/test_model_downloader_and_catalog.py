"""Unit and integration tests for Issue #8: model downloader catalog resolution, GGUF validation, and cache fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from pathlib import Path
import struct
from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import httpx
import pytest

from main import app, runtime
import models.downloader as downloader_module
from models.downloader import (
    DEFAULT_CATALOG_MODELS,
    GGUF_MAGIC,
    DownloadProgressEvent,
    ModelDownloader,
    compute_model_fingerprint,
    has_gguf_magic,
    is_valid_gguf_file,
)
from shard_engine.schemas import NodeState
from shard_engine.tracker_client import SignalingTrackerClient

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
