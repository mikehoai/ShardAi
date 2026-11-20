"""Pytest suite for `core-engine` local API, P2P binary serialization, and pipeline inference."""

from array import array
import asyncio
import json
import os
from pathlib import Path
import uuid

os.environ.setdefault("SHARDAI_OFFLINE_MODE", "true")

from fastapi.testclient import TestClient
import httpx
import pytest

from main import app, runtime
from models.downloader import filter_required_safetensors_files
from models.shard_runner import ShardModelRunner
from p2p.serializer import (
    CompressionCodec,
    DecodedTokenFrame,
    deserialize_activation_tensor,
    serialize_activation_tensor,
    serialize_decoded_token,
)
from p2p.transport import receive_activations, send_activations
from shard_engine.memory_manager import calculate_max_layers
from shard_engine.schemas import NodeState, StartNodeRequest
from shard_engine.tracker_client import SignalingTrackerClient, hash_room_key

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    """Ensure every test starts and ends with a stopped node and zero allocated RAM/VRAM."""
    runtime.session_token = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED
    runtime.node_id = None
    runtime.peers = []


def test_health_check_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "shardai-core-engine"}


def test_models_supported_and_download_stream() -> None:
    res = client.get("/models/supported")
    assert res.status_code == 200
    models = res.json()
    assert "meta-llama/Llama-3.1-8B-Instruct" in models
    assert "Qwen/Qwen2.5-7B-Instruct" in models

    stream_res = client.get(
        "/models/download-stream",
        params={"model_id": "Qwen/Qwen2.5-7B-Instruct"},
    )
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.headers["content-type"]
    assert "data: [DONE]" in stream_res.text


def test_binary_struct_serializer_compresses_megabyte_hidden_state() -> None:
    """Verify multi-megabyte hidden states are packed via binary struct and compressed."""
    # Simulate a (64, 4096) float32 hidden-state tensor (~1.05 MB)
    seq_len, hidden_dim = 64, 4096
    raw_floats = array("f", [0.125] * (seq_len * hidden_dim))
    raw_bytes = raw_floats.tobytes()
    req_id = str(uuid.uuid4())

    wire_frame = serialize_activation_tensor(
        tensor_data=raw_bytes,
        shape=(seq_len, hidden_dim),
        request_id=req_id,
        origin_ip="127.0.0.1",
        origin_port=49152,
        current_layer=15,
        total_layers=32,
        step_index=0,
        max_tokens=8,
        network_congested=True,
        force_compression=True,
    )

    # Wire frame must start with binary magic header b"SHRD" and be heavily compressed
    assert wire_frame[:4] == b"SHRD"
    assert len(wire_frame) < len(raw_bytes) // 10

    unpacked = deserialize_activation_tensor(wire_frame)
    assert unpacked.request_id == req_id
    assert unpacked.shape == (seq_len, hidden_dim)
    assert unpacked.current_layer == 15
    assert unpacked.compression_used in (CompressionCodec.ZLIB, CompressionCodec.LZ4)
    assert unpacked.raw_tensor_bytes == raw_bytes


@pytest.mark.asyncio
async def test_p2p_socket_send_and_receive_activations_with_forward_pass() -> None:
    """Verify end-to-end TCP socket streaming (`send_activations` -> `receive_activations`)."""
    received_frames = []
    runner = ShardModelRunner(shard_layer_start=16, shard_layer_end=31, total_layers=32)

    async def _on_activation(frame) -> None:
        out_bytes, out_shape = runner.forward_pass(frame.raw_tensor_bytes, shape=frame.shape)
        received_frames.append((frame, out_bytes, out_shape))

    server = await receive_activations(
        host="127.0.0.1",
        port=49299,
        on_activation_frame=_on_activation,
    )
    try:
        req_id = str(uuid.uuid4())
        in_bytes, shape = runner.embed_tokens_to_raw_tensor([101, 202, 303])
        binary_stream = serialize_activation_tensor(
            tensor_data=in_bytes,
            shape=shape,
            request_id=req_id,
            origin_ip="127.0.0.1",
            origin_port=49298,
            current_layer=15,
            total_layers=32,
            step_index=0,
        )

        rtt_ms = await send_activations("127.0.0.1", binary_stream, peer_port=49299)
        assert rtt_ms >= 0.0
        assert len(received_frames) == 1
        frame, out_bytes, out_shape = received_frames[0]
        assert frame.request_id == req_id
        assert out_shape == shape
        assert len(out_bytes) == len(in_bytes)
    finally:
        server.close()
        await server.wait_closed()


def test_dynamic_layer_calculation_by_ram_gb() -> None:
    """Verify n_layers scales safely with ram_gb to prevent OOM crashes."""
    low_ram_plan = calculate_max_layers(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=2,
    )
    mid_ram_plan = calculate_max_layers(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=4,
    )
    full_ram_plan = calculate_max_layers(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        ram_gb=8,
    )

    assert low_ram_plan.total_layers == 32
    assert 1 <= low_ram_plan.n_layers < mid_ram_plan.n_layers < full_ram_plan.n_layers
    assert full_ram_plan.n_layers == 32
    assert full_ram_plan.shard_layer_end == 31


def test_system_memory_endpoint_returns_safe_bounds() -> None:
    res = client.get("/system/memory")
    assert res.status_code == 200
    body = res.json()
    assert body["total_ram_gb"] > 0
    assert body["available_ram_gb"] > 0
    assert body["max_shareable_ram_gb"] >= 2


def test_openai_gateway_lifecycle_auth_and_streaming() -> None:
    # 1. Calling /v1/models while STOPPED must fail with 503
    stopped_res = client.get("/v1/models")
    assert stopped_res.status_code == 503

    # 2. Start node with custom HuggingFace model & LAN mode
    start_payload: dict[str, object] = {
        "node_id": "test-gateway-node",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "network_mode": "lan",
        "room_id": "dev-room",
        "backend": "llama_cpp",
        "ram_gb": 6,
        "shared_ram_gb": 6,
        "max_ram_mb": 6144,
        "max_vram_mb": 2048,
        "p2p_listen_port": 49311,
        "shard_layer_start": 0,
        "shard_layer_end": 31,
        "total_layers": 32,
    }
    start_res = client.post("/start-node", json=start_payload)
    assert start_res.status_code == 200
    start_data = start_res.json()
    session_token = start_data["session_token"]
    assert isinstance(session_token, str)
    assert session_token.startswith("sk-shard-")
    assert start_data["api_base_url"] == "http://127.0.0.1:8000/v1"
    assert start_data["network_mode"] == "lan"
    assert start_data["room_id"] == "dev-room"

    # 3. Invalid or missing Bearer token must be rejected with 401
    assert client.get("/v1/models").status_code == 401
    assert (
        client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer sk-shard-wrong"},
            json={"model": "mistralai/Mistral-7B-Instruct-v0.3", "messages": [{"role": "user", "content": "Hi"}]},
        ).status_code
        == 401
    )

    auth_headers = {"Authorization": f"Bearer {session_token}"}

    # 4. GET /v1/models returns active model
    models_res = client.get("/v1/models", headers=auth_headers)
    assert models_res.status_code == 200
    assert models_res.json()["data"][0]["id"] == "mistralai/Mistral-7B-Instruct-v0.3"

    # 5. POST /v1/chat/completions (stream=true) returns OpenAI delta chunks
    sse_res = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": "Hello IDE"}],
            "stream": True,
            "max_tokens": 3,
        },
    )
    assert sse_res.status_code == 200
    assert "text/event-stream" in sse_res.headers["content-type"]
    assert '"delta": {"content":' in sse_res.text
    assert "data: [DONE]" in sse_res.text

    # 6. POST /v1/chat/completions (stream=false) returns OpenAI chat completion JSON
    json_res = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": "Hello IDE"}],
            "stream": False,
            "max_tokens": 3,
        },
    )
    assert json_res.status_code == 200
    body = json_res.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"][0]["message"]["content"]) > 0

    # 7. Stopping node revokes session token
    stop_res = client.post("/stop-node")
    assert stop_res.status_code == 200
    assert stop_res.json()["session_token"] is None
    assert client.get("/v1/models", headers=auth_headers).status_code == 503


def test_start_generate_sse_and_stop_node_with_private_room() -> None:
    start_payload: dict[str, object] = {
        "node_id": "test-node-alpha",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "room_id": "secret-lab-42",
        "backend": "llama_cpp",
        "ram_gb": 6,
        "shared_ram_gb": 6,
        "max_ram_mb": 6144,
        "max_vram_mb": 2048,
        "p2p_listen_port": 49310,
        "shard_layer_start": 0,
        "shard_layer_end": 31,
        "total_layers": 32,
    }

    start_res = client.post("/start-node", json=start_payload)
    assert start_res.status_code == 200
    start_data = start_res.json()
    assert start_data["state"] == "RUNNING"
    assert start_data["room_id"] == "secret-lab-42"
    assert start_data["n_layers"] == 32

    # Unauthenticated /generate must be rejected with 401
    unauth_gen_res = client.post(
        "/generate",
        json={"prompt": "Hello ShardAi P2P", "max_tokens": 4, "compress_activations": True},
    )
    assert unauth_gen_res.status_code == 401

    # Stream tokens via SSE from /generate with valid Bearer session token
    auth_headers = {"Authorization": f"Bearer {start_data['session_token']}"}
    gen_res = client.post(
        "/generate",
        headers=auth_headers,
        json={"prompt": "Hello ShardAi P2P", "max_tokens": 4, "compress_activations": True},
    )
    assert gen_res.status_code == 200
    assert "text/event-stream" in gen_res.headers["content-type"]
    assert "data: [DONE]" in gen_res.text

    stop_res = client.post("/stop-node")
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["state"] == "STOPPED"
    assert stop_data["allocated_ram_mb"] == 0
    assert stop_data["room_id"] is None


@pytest.mark.asyncio
async def test_tracker_client_sends_room_id_and_retries_on_transient_503() -> None:
    attempts: int = 0
    captured_payload: dict[str, object] = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts, captured_payload
        attempts += 1
        captured_payload = json.loads(request.content.decode("utf-8"))
        if attempts < 2:
            return httpx.Response(status_code=503, json={"error": "Service unavailable"})
        return httpx.Response(
            status_code=200,
            json={
                "peers": [
                    {
                        "peer_id": "peer-02",
                        "ip_address": "203.0.113.42",
                        "port": 49152,
                        "shard_layer_start": 16,
                        "shard_layer_end": 31,
                        "available_vram_mb": 4096,
                        "room_id": "team-alpha",
                    }
                ]
            },
        )

    tracker = SignalingTrackerClient(
        base_url="https://tracker.shardai.test/api/v1",
        timeout_seconds=2.0,
        connect_timeout_seconds=1.0,
        max_retries=3,
        base_backoff_seconds=0.01,
        transport=httpx.MockTransport(mock_handler),
    )

    peers = await tracker.register_node_and_fetch_peers(
        StartNodeRequest(
            room_id="team-alpha",
            room_key="my-super-secret",
            is_unlisted=True,
            ram_gb=4,
        )
    )
    assert attempts == 2
    assert captured_payload["room_id"] == "team-alpha"
    assert captured_payload["room_key_hash"] == hash_room_key("my-super-secret")
    assert "room_key" not in captured_payload
    assert captured_payload["is_unlisted"] is True
    assert captured_payload["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert len(peers) == 1
    assert peers[0].peer_id == "peer-02"
    assert peers[0].room_id == "team-alpha"


def test_available_rooms_lobby_and_unlisted_visibility() -> None:
    # 1. Start a listed private room -> MUST appear in GET /available-rooms with is_private=True
    res_listed = client.post(
        "/start-node",
        json={
            "node_id": "lobby-node-01",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "network_mode": "lan",
            "room_id": "lan-cluster-a",
            "room_key": "secret-pass",
            "is_unlisted": False,
            "ram_gb": 6,
            "p2p_listen_port": 49320,
        },
    )
    assert res_listed.status_code == 200
    assert res_listed.json()["is_private"] is True

    rooms_res = client.get("/available-rooms")
    assert rooms_res.status_code == 200
    rooms = rooms_res.json()
    matched = [r for r in rooms if r["room_id"] == "lan-cluster-a"]
    assert len(matched) == 1
    assert matched[0]["is_private"] is True
    assert matched[0]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    client.post("/stop-node")

    # 2. Start an unlisted room -> MUST NOT appear in GET /available-rooms
    res_unlisted = client.post(
        "/start-node",
        json={
            "node_id": "lobby-node-02",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "network_mode": "lan",
            "room_id": "hidden-ghost-room",
            "room_key": "secret-pass",
            "is_unlisted": True,
            "ram_gb": 6,
            "p2p_listen_port": 49321,
        },
    )
    assert res_unlisted.status_code == 200
    assert res_unlisted.json()["is_unlisted"] is True

    rooms_after = client.get("/available-rooms").json()
    assert all(r["room_id"] != "hidden-ghost-room" for r in rooms_after)
    client.post("/stop-node")


@pytest.mark.asyncio
async def test_tracker_fetch_available_rooms() -> None:
    async def mock_rooms_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "success": True,
                "rooms": [
                    {
                        "room_id": "public",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "nodes_online": 4,
                        "total_shared_ram_gb": 24.0,
                        "is_private": False,
                        "network_scope": "internet",
                    },
                    {
                        "room_id": "vault-99",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "nodes_online": 2,
                        "total_shared_ram_gb": 12.0,
                        "is_private": True,
                        "network_scope": "internet",
                    },
                ],
            },
        )

    tracker = SignalingTrackerClient(
        base_url="https://tracker.shardai.test/api/v1",
        transport=httpx.MockTransport(mock_rooms_handler),
    )
    rooms = await tracker.fetch_available_rooms()
    assert len(rooms) == 2
    assert rooms[0].room_id == "public"
    assert rooms[0].is_private is False
    assert rooms[1].room_id == "vault-99"
    assert rooms[1].is_private is True


def test_filter_required_safetensors_files_selects_only_assigned_shard() -> None:
    """Verify each node downloads only the `.safetensors` files for its layer range."""
    weight_map: dict[str, str] = {
        "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
        "model.layers.15.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
        "model.layers.16.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
        "model.layers.31.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
        "model.norm.weight": "model-00004-of-00004.safetensors",
        "lm_head.weight": "model-00004-of-00004.safetensors",
    }

    head_files = filter_required_safetensors_files(
        weight_map=weight_map,
        shard_layer_start=0,
        shard_layer_end=15,
        total_layers=32,
    )
    assert head_files == [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
    ]

    tail_files = filter_required_safetensors_files(
        weight_map=weight_map,
        shard_layer_start=16,
        shard_layer_end=31,
        total_layers=32,
    )
    assert tail_files == [
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]


@pytest.mark.asyncio
async def test_two_node_pytorch_p2p_pipeline_inference_over_tcp(tmp_path: Path) -> None:
    """Verify 2-node PyTorch layer-sharded pipeline over TCP matches unsharded greedy decoding."""
    import torch
    from safetensors.torch import save_model
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    torch.manual_seed(42)
    full_model = AutoModelForCausalLM.from_config(cfg).eval()
    cfg.save_pretrained(str(tmp_path))
    save_model(full_model, str(tmp_path / "model.safetensors"))

    prompt_ids = [3, 11, 27, 42]
    expected_tokens: list[int] = []
    with torch.inference_mode():
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long)
        for _ in range(4):
            logits = full_model(cur_ids).logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            expected_tokens.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)

    head_runner = ShardModelRunner(
        model_id="test/tiny-qwen2",
        model_path=str(tmp_path),
        shard_layer_start=0,
        shard_layer_end=1,
        total_layers=4,
        max_ram_mb=2048,
    )
    tail_runner = ShardModelRunner(
        model_id="test/tiny-qwen2",
        model_path=str(tmp_path),
        shard_layer_start=2,
        shard_layer_end=3,
        total_layers=4,
        max_ram_mb=2048,
    )

    assert head_runner.is_real_model is True
    assert tail_runner.is_real_model is True
    assert head_runner.is_head_node and not head_runner.is_tail_node
    assert tail_runner.is_tail_node and not tail_runner.is_head_node

    token_queue: asyncio.Queue[DecodedTokenFrame] = asyncio.Queue()

    async def _on_head_token(tok_frame: DecodedTokenFrame) -> None:
        await token_queue.put(tok_frame)

    async def _on_tail_activation(frame) -> None:
        out_bytes, _shape = tail_runner.forward_pass(
            frame.raw_tensor_bytes,
            shape=frame.shape,
            request_id=frame.request_id,
            step_index=frame.step_index,
        )
        tok_id, tok_text, is_eos = tail_runner.decode_final_layer_to_token(
            out_bytes,
            step_index=frame.step_index,
            max_tokens=frame.max_tokens,
            request_id=frame.request_id,
            temperature=frame.temperature,
        )
        packet = serialize_decoded_token(
            request_id=frame.request_id,
            step_index=frame.step_index,
            token_id=tok_id,
            token_text=tok_text,
            is_eos=is_eos,
        )
        await send_activations("127.0.0.1", packet, peer_port=frame.origin_port)

    head_server = await receive_activations(
        host="127.0.0.1",
        port=49331,
        on_activation_frame=lambda _f: asyncio.sleep(0),
        on_token_frame=_on_head_token,
    )
    tail_server = await receive_activations(
        host="127.0.0.1",
        port=49332,
        on_activation_frame=_on_tail_activation,
    )

    try:
        req_id = str(uuid.uuid4())
        generated_tokens: list[int] = []
        current_input = list(prompt_ids)
        for step_idx in range(4):
            step_tokens = current_input if step_idx == 0 else [current_input[-1]]
            hidden_bytes, shape = head_runner.forward_pass(
                step_tokens,
                request_id=req_id,
                step_index=step_idx,
            )
            wire_frame = serialize_activation_tensor(
                tensor_data=hidden_bytes,
                shape=shape,
                request_id=req_id,
                origin_ip="127.0.0.1",
                origin_port=49331,
                current_layer=1,
                total_layers=4,
                step_index=step_idx,
                max_tokens=4,
                force_compression=True,
                temperature=0.0,
            )
            await send_activations("127.0.0.1", wire_frame, peer_port=49332)
            decoded = await asyncio.wait_for(token_queue.get(), timeout=5.0)
            generated_tokens.append(decoded.token_id)
            current_input.append(decoded.token_id)

        assert generated_tokens == expected_tokens
    finally:
        head_server.close()
        tail_server.close()
        await head_server.wait_closed()
        await tail_server.wait_closed()
        head_runner.release()
        tail_runner.release()
        assert head_runner.is_real_model is False
        assert tail_runner.is_real_model is False
