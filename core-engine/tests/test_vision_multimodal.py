"""Unit and integration tests for Vision / Multimodal (mmproj + OpenAI Vision) support."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import types
from typing import Any

from fastapi.testclient import TestClient
import pytest

from main import (
    _extract_chat_messages_dicts,
    _extract_prompt_from_messages,
    app,
    runtime,
)
import models.downloader as downloader_module
from models.downloader import (
    GGUF_MAGIC,
    DownloadedModelArtifacts,
    ModelDownloader,
    sanitize_catalog_entry,
    select_best_mmproj_filename,
)
from models.shard_runner import ShardModelRunner
from shard_engine.schemas import ChatMessage, NodeState

client = TestClient(app)


def _write_synthetic_gguf(target_path: Path, total_size: int = 1_100_000) -> Path:
    """Create a valid GGUF v3 file with minimal metadata padded to `total_size`."""
    key_bytes = b"llama.block_count"
    header = bytearray()
    header.extend(GGUF_MAGIC)
    header.extend(struct.pack("<IQQ", 3, 0, 1))
    header.extend(struct.pack("<Q", len(key_bytes)))
    header.extend(key_bytes)
    header.extend(struct.pack("<II", 4, 32))
    if total_size > len(header):
        header.extend(b"\x00" * (total_size - len(header)))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(bytes(header))
    return target_path


@pytest.fixture(autouse=True)
def reset_runtime_vision_state() -> Any:
    """Reset runtime state before and after each test."""
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.mmproj_path = None
    runtime.peers = []
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    yield
    runtime.session_token = None
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    runtime.mmproj_path = None
    runtime.peers = []
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


def test_sanitize_catalog_entry_preserves_vision_modality_and_mmproj() -> None:
    """`sanitize_catalog_entry()` preserves `modality='vision'` and `mmproj_filename`."""
    cleaned = sanitize_catalog_entry(
        {
            "id": "qwen2.5-vl-7b",
            "name": "Qwen2.5-VL 7B Instruct",
            "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
            "filename": "qwen2.5-vl-7b-instruct-q4_k_m.gguf",
            "mmproj_filename": "  mmproj-model-f16.gguf  ",
            "modality": "  VISION  ",
            "estimated_ram_gb": 7.5,
            "enabled": True,
        }
    )
    assert cleaned is not None
    assert cleaned["modality"] == "vision"
    assert cleaned["mmproj_filename"] == "mmproj-model-f16.gguf"

    inferred = sanitize_catalog_entry(
        {
            "id": "llava-custom",
            "name": "LLaVA Custom",
            "repo_id": "org/llava-v1.6-gguf",
            "mmproj_filename": "mmproj-vicuna-f16.gguf",
            "enabled": True,
        }
    )
    assert inferred is not None
    assert inferred["modality"] == "vision"


def test_select_best_mmproj_filename_priority_and_filtering() -> None:
    """`select_best_mmproj_filename()` prefers f16 -> bf16 -> q8_0 -> f32 and skips base GGUFs."""
    files = [
        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj-model-q8_0.gguf",
        "mmproj-model-bf16.gguf",
        "mmproj-model-f16.gguf",
        "mmproj-partial.gguf.part",
    ]
    assert select_best_mmproj_filename(files) == "mmproj-model-f16.gguf"
    assert (
        select_best_mmproj_filename(
            ["model-Q4_K_M.gguf", "mmproj-model-q8_0.gguf", "mmproj-model-bf16.gguf"]
        )
        == "mmproj-model-bf16.gguf"
    )
    assert select_best_mmproj_filename(["Llama-3.1-8B-Q4_K_M.gguf"]) is None


@pytest.mark.asyncio
async def test_downloader_fetches_explicit_and_autodiscovered_mmproj(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Downloader fetches companion `mmproj-*.gguf` for vision models and skips for text models."""
    monkeypatch.setenv("SHARDAI_OFFLINE_MODE", "false")
    downloaded_calls: list[tuple[str, str]] = []

    def _fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        local_dir: str,
        **_kwargs: Any,
    ) -> str:
        downloaded_calls.append((repo_id, filename))
        out = Path(local_dir) / filename
        _write_synthetic_gguf(out)
        return str(out)

    monkeypatch.setattr(downloader_module, "hf_hub_download", _fake_hf_hub_download)

    downloader = ModelDownloader(
        cache_dir=tmp_path / "cache",
        models_dir=tmp_path / "models",
        catalog_cache_path=tmp_path / "models_cache.json",
    )

    # 1. Explicit mmproj_filename from catalog / caller
    explicit_artifacts: DownloadedModelArtifacts = (
        await downloader.ensure_model_artifacts_downloaded(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
            source_type="catalog",
            filename="qwen2.5-vl-7b-q4_k_m.gguf",
            mmproj_filename="mmproj-qwen2.5-vl-7b-f16.gguf",
        )
    )
    assert explicit_artifacts.model_path.name == "qwen2.5-vl-7b-q4_k_m.gguf"
    assert explicit_artifacts.mmproj_path is not None
    assert explicit_artifacts.mmproj_path.name == "mmproj-qwen2.5-vl-7b-f16.gguf"
    assert explicit_artifacts.mmproj_path.is_file()
    assert downloader.get_resolved_mmproj_path("Qwen/Qwen2.5-VL-7B-Instruct-GGUF") == (
        explicit_artifacts.mmproj_path
    )

    # 2. Custom HF vision repo with no explicit filename/mmproj_filename auto-discovers both
    monkeypatch.setattr(
        downloader_module,
        "_list_hf_repo_files",
        lambda _repo_id: [
            "llava-v1.6-mistral-7b-Q4_K_M.gguf",
            "mmproj-model-f16.gguf",
        ],
    )
    auto_artifacts = await downloader.ensure_model_artifacts_downloaded(
        model_id="cjpais/llava-v1.6-mistral-7b-gguf",
        source_type="huggingface",
    )
    assert auto_artifacts.model_path.name == "llava-v1.6-mistral-7b-Q4_K_M.gguf"
    assert auto_artifacts.mmproj_path is not None
    assert auto_artifacts.mmproj_path.name == "mmproj-model-f16.gguf"

    # 3. Standard text-only HF repo returns mmproj_path=None
    monkeypatch.setattr(
        downloader_module,
        "_list_hf_repo_files",
        lambda _repo_id: ["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
    )
    text_artifacts = await downloader.ensure_model_artifacts_downloaded(
        model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        source_type="huggingface",
    )
    assert text_artifacts.model_path.name == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    assert text_artifacts.mmproj_path is None
    assert ("Qwen/Qwen2.5-VL-7B-Instruct-GGUF", "mmproj-qwen2.5-vl-7b-f16.gguf") in downloaded_calls
    assert ("cjpais/llava-v1.6-mistral-7b-gguf", "mmproj-model-f16.gguf") in downloaded_calls


def test_shard_runner_loads_vision_chat_handler_only_on_head_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`ShardModelRunner` passes `chat_handler` with `clip_model_path` only when layer 0 is held."""
    gguf_path = _write_synthetic_gguf(tmp_path / "vision-model-q4_k_m.gguf")
    mmproj_path = _write_synthetic_gguf(tmp_path / "mmproj-model-f16.gguf")

    llama_calls: list[dict[str, Any]] = []
    handler_calls: list[tuple[str, str]] = []

    class _FakeHandler:
        def __init__(self, *, name: str, clip_model_path: str, verbose: bool = False) -> None:
            self.name = name
            self.clip_model_path = clip_model_path
            self.verbose = verbose
            handler_calls.append((name, clip_model_path))

    class _FakeQwen25VLChatHandler(_FakeHandler):
        def __init__(self, *, clip_model_path: str, verbose: bool = False) -> None:
            super().__init__(
                name="Qwen25VLChatHandler",
                clip_model_path=clip_model_path,
                verbose=verbose,
            )

    class _FakeLlava15ChatHandler(_FakeHandler):
        def __init__(self, *, clip_model_path: str, verbose: bool = False) -> None:
            super().__init__(
                name="Llava15ChatHandler",
                clip_model_path=clip_model_path,
                verbose=verbose,
            )

    class _FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            llama_calls.append(kwargs)

    fake_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    fake_chat_format.Qwen25VLChatHandler = _FakeQwen25VLChatHandler  # type: ignore[attr-defined]
    fake_chat_format.Llava15ChatHandler = _FakeLlava15ChatHandler  # type: ignore[attr-defined]

    fake_llama_cpp = types.ModuleType("llama_cpp")
    fake_llama_cpp.Llama = _FakeLlama  # type: ignore[attr-defined]
    fake_llama_cpp.llama_chat_format = fake_chat_format  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", fake_chat_format)

    # 1. Head node (shard_layer_start=0) with Qwen2.5-VL selects Qwen25VLChatHandler
    qwen_head = ShardModelRunner(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        model_path=str(gguf_path),
        mmproj_path=str(mmproj_path),
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
        n_layers=16,
    )
    assert qwen_head.has_native_llama is True
    assert "chat_handler" in llama_calls[-1]
    assert isinstance(llama_calls[-1]["chat_handler"], _FakeQwen25VLChatHandler)
    assert llama_calls[-1]["chat_handler"].clip_model_path == str(mmproj_path)

    # 2. Head node (shard_layer_start=0) with LLaVA 1.5 falls back to Llava15ChatHandler
    llava_head = ShardModelRunner(
        model_id="mys/ggml_llava-v1.5-7b",
        model_path=str(gguf_path),
        mmproj_path=str(mmproj_path),
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        n_layers=32,
    )
    assert llava_head.has_native_llama is True
    assert isinstance(llama_calls[-1]["chat_handler"], _FakeLlava15ChatHandler)

    # 3. Worker node (shard_layer_start=16) must NOT instantiate chat_handler
    worker_runner = ShardModelRunner(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        model_path=str(gguf_path),
        mmproj_path=str(mmproj_path),
        shard_layer_start=16,
        shard_layer_end=31,
        total_layers=32,
        n_layers=16,
    )
    assert worker_runner.has_native_llama is True
    assert "chat_handler" not in llama_calls[-1]
    assert len(handler_calls) == 2


def test_openai_chat_completions_preserves_vision_multimodal_payload() -> None:
    """`POST /v1/chat/completions` preserves OpenAI Vision content blocks end-to-end."""
    captured_messages: list[Any] = []

    class _FakeNativeLlama:
        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            stream: bool,
        ) -> Any:
            captured_messages.append(messages)
            return iter(
                [
                    {"choices": [{"delta": {"content": "A cat on a sofa."}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ]
            )

    runner = ShardModelRunner(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        shard_layer_start=0,
        shard_layer_end=31,
        total_layers=32,
        n_layers=32,
        native_handle=_FakeNativeLlama(),
    )

    runtime.state = NodeState.RUNNING
    runtime.node_id = "vision-head-01"
    runtime.model_id = "Qwen/Qwen2.5-VL-7B-Instruct-GGUF"
    runtime.session_token = "sk-shard-vision-test"
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 31
    runtime.total_layers = 32
    runtime.n_layers = 32
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.model_runner = runner

    vision_data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/"
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-GGUF",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this frame?"},
                    {"type": "image_url", "image_url": {"url": vision_data_url, "detail": "low"}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 64,
    }

    res = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer sk-shard-vision-test"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "A cat on a sofa."
    assert len(captured_messages) == 1
    sent_content = captured_messages[0][0]["content"]
    assert isinstance(sent_content, list)
    assert sent_content[0] == {"type": "text", "text": "What is in this frame?"}
    assert sent_content[1] == {
        "type": "image_url",
        "image_url": {"url": vision_data_url, "detail": "low"},
    }

    # Prompt fallback flattens image_url cleanly for text-only paths
    flattened = _extract_prompt_from_messages(
        [ChatMessage.model_validate(payload["messages"][0])]
    )
    assert "[user] What is in this frame? [image]" == flattened

    # Text-only array content normalizes back to string for backward compatibility
    text_only = _extract_chat_messages_dicts(
        [ChatMessage(role="user", content=[{"type": "text", "text": "Hello"}])]
    )
    assert text_only == [{"role": "user", "content": "Hello"}]


def test_text_only_model_rejects_image_attachment_with_clear_400_error(tmp_path: Path) -> None:
    """Text-only models without a vision projector reject image_url payloads cleanly with HTTP 400."""

    class _TextOnlyFakeLlama:
        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            stream: bool,
        ) -> Any:
            _ = messages, max_tokens, temperature, stream
            raise AssertionError("Should not invoke text-only Llama for image payload")

    dummy_gguf = tmp_path / "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    dummy_gguf.write_bytes(b"GGUF")

    runner = ShardModelRunner(
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        model_path=str(dummy_gguf),
        shard_layer_start=0,
        shard_layer_end=23,
        total_layers=24,
        n_layers=24,
        native_handle=_TextOnlyFakeLlama(),
    )
    assert runner.has_vision_support is False

    runtime.state = NodeState.RUNNING
    runtime.node_id = "text-only-node"
    runtime.model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
    runtime.session_token = "sk-shard-text-only-test"
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 23
    runtime.total_layers = 24
    runtime.n_layers = 24
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.model_runner = runner

    payload = {
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "max_tokens": 32,
    }
    headers = {"Authorization": "Bearer sk-shard-text-only-test"}

    for stream_flag in (False, True):
        res = client.post(
            "/v1/chat/completions",
            json={**payload, "stream": stream_flag},
            headers=headers,
        )
        assert res.status_code == 400
        body = res.json()
        assert body["error"]["code"] == "model_vision_not_supported"
        assert "does not support image or vision attachments" in body["error"]["message"]


def test_chat_template_list_concatenation_typeerror_surfaces_model_error() -> None:
    """If a model's chat template raises TypeError on list content, surface HTTP 400 with detail."""

    class _TemplateTypeErrorLlama:
        def create_chat_completion(
            self,
            *,
            messages: list[dict[str, Any]],
            max_tokens: int,
            temperature: float,
            stream: bool,
        ) -> Any:
            _ = messages, max_tokens, temperature, stream
            raise TypeError('can only concatenate str (not "list") to str')

    runner = ShardModelRunner(
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        shard_layer_start=0,
        shard_layer_end=23,
        total_layers=24,
        n_layers=24,
        native_handle=_TemplateTypeErrorLlama(),
    )

    runtime.state = NodeState.RUNNING
    runtime.node_id = "template-err-node"
    runtime.model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
    runtime.session_token = "sk-shard-template-err"
    runtime.shard_layer_start = 0
    runtime.shard_layer_end = 23
    runtime.total_layers = 24
    runtime.n_layers = 24
    runtime.is_swarm_ready = True
    runtime.coverage_percent = 100.0
    runtime.model_runner = runner

    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
            "stream": False,
        },
        headers={"Authorization": "Bearer sk-shard-template-err"},
    )
    assert res.status_code == 400
    body = res.json()
    assert body["error"]["code"] == "model_vision_not_supported"
    assert "does not support this attachment format" in body["error"]["message"]
    assert 'can only concatenate str (not "list") to str' in body["error"]["message"]
