"""ShardAi Core Engine — Local FastAPI P2P AI Gateway & OpenAI-Compatible API Server.

Security & Architecture Notes:
- Binds strictly to `127.0.0.1` (loopback only) as mandated in `.antigravity/review.md`.
- Generates an in-memory cryptographic `session_token` (`sk-shard-...`) on `/start-node`
  and destroys it immediately on `/stop-node` or shutdown.
- Exposes OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions` protected by
  Bearer token + active node state verification.
- Connects directly to the P2P Pipeline Inference engine (`ShardModelRunner` + `p2p` binary stream).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import time
from typing import Any
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from models.downloader import (
    SUPPORTED_MODELS,
    DownloadProgressEvent,
    ModelDownloader,
    is_offline_mode,
    read_hf_config_summary,
)
from models.shard_runner import ShardModelRunner
from p2p.mdns_discovery import LanRoomBeacon, lan_mdns_service
from p2p.serializer import (
    ActivationFrame,
    DecodedTokenFrame,
    serialize_activation_tensor,
    serialize_decoded_token,
)
from p2p.transport import (
    congestion_monitor,
    receive_activations,
    send_activations,
)
from shard_engine.memory_manager import (
    ShardMemoryManager,
    calculate_max_layers,
    get_system_memory_info,
)
from shard_engine.schemas import (
    ChatMessage,
    ComputeBackend,
    GenerateRequest,
    NetworkMode,
    NodeState,
    NodeStatusResponse,
    OpenAIChatCompletionRequest,
    OpenAIModelListResponse,
    OpenAIModelObject,
    PeerNodeInfo,
    RoomLobbyItem,
    StartNodeRequest,
    SystemMemoryResponse,
)
from shard_engine.tracker_client import SignalingTrackerClient, hash_room_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shardai.main")


def _extract_chat_messages_dicts(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert OpenAI `ChatMessage` models into simple `{role, content}` dicts."""
    normalized: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg.content, str):
            text = msg.content.strip()
        else:
            text = " ".join(
                str(block.get("text", ""))
                for block in msg.content
                if isinstance(block, dict)
            ).strip()
        if text:
            normalized.append({"role": msg.role, "content": text})
    return normalized or [{"role": "user", "content": "Hello"}]


def _extract_prompt_from_messages(messages: list[ChatMessage]) -> str:
    """Flatten OpenAI chat messages into a single prompt string for fallback tokenizers."""
    parts: list[str] = [
        f"[{item['role']}] {item['content']}"
        for item in _extract_chat_messages_dicts(messages)
    ]
    return "\n".join(parts) if parts else "Hello"


def _resolve_advertise_ip(target_peer_ip: str, target_peer_port: int) -> str:
    """Resolve the local IPv4 address reachable by `target_peer_ip` for return token frames."""
    explicit_ip = os.getenv("SHARDAI_ADVERTISE_IP", "").strip()
    if explicit_ip:
        return explicit_ip
    if target_peer_ip.startswith("127.") or target_peer_ip == "localhost":
        return "127.0.0.1"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_peer_ip, target_peer_port))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class LocalNodeRuntime:
    """Encapsulates runtime state, session auth, P2P listener, and model shard pipeline."""

    def __init__(self) -> None:
        self.state: NodeState = NodeState.STOPPED
        self.node_id: str | None = None
        self.model_id: str | None = None
        self.model_path: str | None = None
        self.network_mode: NetworkMode | None = None
        self.room_id: str | None = None
        self.is_private: bool = False
        self.is_unlisted: bool = False
        self.backend: ComputeBackend | None = None
        self.p2p_listen_port: int | None = None
        self.n_layers: int = 0
        self.shard_layer_start: int = 0
        self.shard_layer_end: int = 15
        self.total_layers: int = 32
        self.peers: list[PeerNodeInfo] = []
        self.session_token: str | None = None
        self.memory_manager: ShardMemoryManager = ShardMemoryManager()
        self.model_downloader: ModelDownloader = ModelDownloader()
        self.model_runner: ShardModelRunner | None = None
        self.p2p_server: asyncio.AbstractServer | None = None
        self.pending_token_queues: dict[str, asyncio.Queue[DecodedTokenFrame]] = {}
        self._compute_lock: asyncio.Lock = asyncio.Lock()
        self.tracker_client: SignalingTrackerClient = SignalingTrackerClient(
            base_url=os.getenv(
                "SHARDAI_TRACKER_URL", "https://tracker.shardai.network/api/v1"
            ),
            timeout_seconds=8.0,
            connect_timeout_seconds=4.0,
            max_retries=3,
        )

    @property
    def api_base_url(self) -> str:
        port: int = int(os.getenv("SHARDAI_ENGINE_PORT", "8000"))
        return f"http://127.0.0.1:{port}/v1"

    def find_next_hop_peer(self, completed_layer: int) -> PeerNodeInfo | None:
        """Locate the downstream P2P peer responsible for layer `completed_layer + 1`."""
        target_next_layer: int = completed_layer + 1
        for peer in self.peers:
            if peer.shard_layer_start <= target_next_layer <= peer.shard_layer_end:
                return peer
        candidates = [p for p in self.peers if p.shard_layer_start > completed_layer]
        if candidates:
            candidates.sort(key=lambda p: p.shard_layer_start)
            return candidates[0]
        return None

    async def handle_incoming_activation(self, frame: ActivationFrame) -> None:
        """Callback triggered by `receive_activations` when an upstream peer pushes a tensor."""
        if self.model_runner is None or self.state != NodeState.RUNNING:
            logger.warning("Ignoring incoming activation frame because node is not RUNNING.")
            return

        async with self._compute_lock:
            output_tensor_bytes, output_shape = await asyncio.to_thread(
                self.model_runner.forward_pass,
                frame.raw_tensor_bytes,
                shape=frame.shape,
                request_id=frame.request_id,
                step_index=frame.step_index,
            )

        completed_layer: int = self.shard_layer_end
        is_final_layer: bool = completed_layer >= (frame.total_layers - 1)
        next_peer: PeerNodeInfo | None = (
            None if is_final_layer else self.find_next_hop_peer(completed_layer)
        )

        if is_final_layer or (not self.model_runner.is_real_model and next_peer is None):
            async with self._compute_lock:
                token_id, token_text, is_eos = await asyncio.to_thread(
                    self.model_runner.decode_final_layer_to_token,
                    output_tensor_bytes,
                    step_index=frame.step_index,
                    max_tokens=frame.max_tokens,
                    request_id=frame.request_id,
                )
            token_packet: bytes = serialize_decoded_token(
                request_id=frame.request_id,
                step_index=frame.step_index,
                token_id=token_id,
                token_text=token_text,
                is_eos=is_eos,
            )
            await send_activations(
                peer_ip=frame.origin_ip,
                byte_stream=token_packet,
                peer_port=frame.origin_port,
            )
        elif next_peer is not None:
            next_frame_bytes: bytes = serialize_activation_tensor(
                tensor_data=output_tensor_bytes,
                shape=output_shape,
                request_id=frame.request_id,
                origin_ip=frame.origin_ip,
                origin_port=frame.origin_port,
                current_layer=completed_layer,
                total_layers=frame.total_layers,
                step_index=frame.step_index,
                max_tokens=frame.max_tokens,
                network_congested=congestion_monitor.is_congested,
                force_compression=True,
            )
            await send_activations(
                peer_ip=next_peer.ip_address,
                byte_stream=next_frame_bytes,
                peer_port=next_peer.port,
            )
        else:
            logger.error(
                "Middle node completed layer %d/%d but no downstream peer is registered for layer %d.",
                completed_layer,
                frame.total_layers - 1,
                completed_layer + 1,
            )

    async def handle_incoming_decoded_token(self, token_frame: DecodedTokenFrame) -> None:
        """Callback triggered on the Head Node when the Tail Node returns a decoded token."""
        queue = self.pending_token_queues.get(token_frame.request_id)
        if queue is not None:
            await queue.put(token_frame)

    async def stream_p2p_pipeline_tokens(
        self,
        prompt: str,
        max_tokens: int,
        compress_activations: bool = True,
        messages: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[tuple[str, int, int, str, bool]]:
        """Execute distributed P2P pipeline inference and yield `(request_id, step, token_id, text, is_eos)`."""
        if self.state != NodeState.RUNNING or self.model_runner is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Node must be RUNNING before invoking pipeline inference.",
            )
        if not self.model_runner.is_head_node:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"This node hosts layers {self.shard_layer_start}..{self.shard_layer_end}. "
                    "Inference requests must enter via a Head Node holding layer 0."
                ),
            )

        next_peer_initial: PeerNodeInfo | None = self.find_next_hop_peer(
            self.shard_layer_end
        )
        if (
            self.model_runner.is_real_model
            and self.shard_layer_end < (self.total_layers - 1)
            and next_peer_initial is None
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"This node hosts layers {self.shard_layer_start}..{self.shard_layer_end} of "
                    f"{self.total_layers}, and no downstream peer is registered for layer "
                    f"{self.shard_layer_end + 1}. Register peers via POST /peers/register or "
                    "allocate enough RAM to host all layers."
                ),
            )

        request_id: str = str(uuid.uuid4())
        token_queue: asyncio.Queue[DecodedTokenFrame] = asyncio.Queue()
        self.pending_token_queues[request_id] = token_queue

        try:
            if messages and self.model_runner.is_real_model:
                token_ids: list[int] = self.model_runner.tokenize_chat(messages)
            else:
                token_ids = self.model_runner.tokenize(prompt)
            origin_port: int = self.p2p_listen_port or 49152

            for step_idx in range(max_tokens):
                step_input_ids: list[int] = (
                    token_ids
                    if (step_idx == 0 or not self.model_runner.is_real_model)
                    else [token_ids[-1]]
                )
                async with self._compute_lock:
                    hidden_bytes, tensor_shape = await asyncio.to_thread(
                        self.model_runner.forward_pass,
                        step_input_ids,
                        request_id=request_id,
                        step_index=step_idx,
                    )

                next_peer: PeerNodeInfo | None = self.find_next_hop_peer(
                    self.shard_layer_end
                )

                if (
                    next_peer is not None
                    and self.shard_layer_end < (self.total_layers - 1)
                ):
                    advertise_ip: str = _resolve_advertise_ip(
                        next_peer.ip_address, next_peer.port
                    )
                    binary_frame: bytes = serialize_activation_tensor(
                        tensor_data=hidden_bytes,
                        shape=tensor_shape,
                        request_id=request_id,
                        origin_ip=advertise_ip,
                        origin_port=origin_port,
                        current_layer=self.shard_layer_end,
                        total_layers=self.total_layers,
                        step_index=step_idx,
                        max_tokens=max_tokens,
                        network_congested=congestion_monitor.is_congested,
                        force_compression=compress_activations,
                    )
                    await send_activations(
                        peer_ip=next_peer.ip_address,
                        byte_stream=binary_frame,
                        peer_port=next_peer.port,
                    )
                    decoded_frame: DecodedTokenFrame = await asyncio.wait_for(
                        token_queue.get(),
                        timeout=60.0,
                    )
                    token_id = decoded_frame.token_id
                    token_text = decoded_frame.token_text
                    is_eos = decoded_frame.is_eos
                else:
                    async with self._compute_lock:
                        token_id, token_text, is_eos = await asyncio.to_thread(
                            self.model_runner.decode_final_layer_to_token,
                            hidden_bytes,
                            step_index=step_idx,
                            max_tokens=max_tokens,
                            request_id=request_id,
                            temperature=temperature,
                        )
                    if not self.model_runner.is_real_model:
                        await asyncio.sleep(0.015)

                yield request_id, step_idx, token_id, token_text, is_eos

                token_ids.append(token_id)
                if is_eos:
                    break
        finally:
            self.pending_token_queues.pop(request_id, None)
            if self.model_runner is not None:
                self.model_runner.release_request_cache(request_id)

    async def start_p2p_listener(
        self,
        port: int,
        network_mode: NetworkMode = NetworkMode.INTERNET,
    ) -> None:
        """Start the async TCP P2P listener (`receive_activations`)."""
        await self.stop_p2p_listener()
        default_bind = "0.0.0.0" if network_mode == NetworkMode.LAN else "127.0.0.1"
        bind_host: str = os.getenv("SHARDAI_P2P_BIND_HOST", default_bind)
        self.p2p_server = await receive_activations(
            host=bind_host,
            port=port,
            on_activation_frame=self.handle_incoming_activation,
            on_token_frame=self.handle_incoming_decoded_token,
        )

    async def stop_p2p_listener(self) -> None:
        """Close the async TCP P2P listener cleanly."""
        if self.p2p_server is not None:
            try:
                self.p2p_server.close()
                await asyncio.wait_for(self.p2p_server.wait_closed(), timeout=2.0)
            except Exception as exc:
                logger.debug("Error while closing P2P server: %s", exc)
            finally:
                self.p2p_server = None

    def to_status_response(self, message: str = "") -> NodeStatusResponse:
        """Build a typed status snapshot for the Tauri UI."""
        is_running: bool = self.state == NodeState.RUNNING
        return NodeStatusResponse(
            state=self.state,
            node_id=self.node_id,
            model_id=self.model_id,
            model_path=self.model_path,
            network_mode=self.network_mode,
            room_id=self.room_id,
            is_private=self.is_private,
            is_unlisted=self.is_unlisted,
            backend=self.backend,
            n_layers=self.n_layers,
            total_layers=self.total_layers,
            shard_layer_start=self.shard_layer_start if is_running else None,
            shard_layer_end=self.shard_layer_end if is_running else None,
            allocated_ram_mb=self.memory_manager.allocated_ram_mb,
            allocated_vram_mb=self.memory_manager.allocated_vram_mb,
            p2p_listen_port=self.p2p_listen_port,
            connected_peers_count=len(self.peers),
            peers=self.peers,
            session_token=self.session_token if is_running else None,
            api_base_url=self.api_base_url if is_running else None,
            message=message,
        )


runtime = LocalNodeRuntime()


async def verify_openai_bearer_token(
    authorization: str | None = Header(default=None),
) -> str:
    """FastAPI Dependency enforcing active node status and Bearer session token auth."""
    if runtime.state != NodeState.RUNNING or runtime.session_token is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ShardAi node is not active. Start sharing RAM to unlock the Local OpenAI Gateway.",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <session_token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token: str = authorization[len("Bearer ") :].strip()
    if not provided_token or not secrets.compare_digest(
        provided_token, runtime.session_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_token


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Ensure model shards, session tokens, and P2P listeners are released on shutdown."""
    logger.info("ShardAi P2P AI Gateway starting up on loopback interface...")
    yield
    logger.info("ShardAi P2P AI Gateway shutting down; releasing RAM/VRAM and session token...")
    runtime.session_token = None
    await lan_mdns_service.stop_advertising()
    await runtime.stop_p2p_listener()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


app = FastAPI(
    title="ShardAi Local P2P AI Gateway",
    description="Local OpenAI-compatible AI Gateway backed by distributed P2P LLM sharding.",
    version="0.3.0",
    lifespan=lifespan,
)

# Restrict CORS exclusively to local Tauri Webview and local Vite dev server
ALLOWED_ORIGINS: list[str] = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-ShardAi-Local-Token"],
)


# ==============================================================================
# Control Plane Endpoints (Tauri UI <-> Python Core)
# ==============================================================================


@app.get("/health", response_model=dict[str, str])
async def health_check() -> dict[str, str]:
    """Lightweight liveness probe for Tauri sidecar supervision."""
    return {"status": "ok", "service": "shardai-core-engine"}


@app.get("/status", response_model=NodeStatusResponse)
async def get_node_status() -> NodeStatusResponse:
    """Return current node state, RAM/VRAM usage, active session token, and connected peers."""
    return runtime.to_status_response(message="Node status retrieved successfully.")


@app.get("/system/memory", response_model=SystemMemoryResponse)
async def get_host_system_memory() -> SystemMemoryResponse:
    """Return host OS total/free RAM and safe max shareable RAM (free RAM - 1 GB)."""
    total_gb, avail_gb, max_shareable_gb = get_system_memory_info()
    return SystemMemoryResponse(
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        max_shareable_ram_gb=max_shareable_gb,
    )


def _is_tracker_enabled() -> bool:
    """Return True when outbound Cloudflare Signaling Tracker calls are enabled."""
    return os.getenv("SHARDAI_ENABLE_TRACKER", "false").lower() == "true" and not is_offline_mode()


@app.get("/available-rooms", response_model=list[RoomLobbyItem])
async def get_available_rooms() -> list[RoomLobbyItem]:
    """Fetch active rooms from Cloudflare Tracker (GET /rooms) merged with LAN mDNS scan."""
    merged: dict[str, RoomLobbyItem] = {}

    # 1. Query Cloudflare Signaling Tracker when enabled
    if _is_tracker_enabled():
        try:
            cloud_rooms = await runtime.tracker_client.fetch_available_rooms()
            for item in cloud_rooms:
                merged[item.room_id] = item
        except Exception as exc:
            logger.warning("Failed to fetch rooms from Cloudflare Tracker: %s", exc)

    # 2. Scan local LAN subnet via mDNS multicast
    lan_beacons = await lan_mdns_service.scan_lan_rooms(timeout_seconds=0.25)
    for beacon in lan_beacons:
        r_id = str(beacon.get("room_id") or "public")
        model = str(beacon.get("model") or "meta-llama/Llama-3.1-8B-Instruct")
        ram_gb = float(beacon.get("ram_gb") or 0.0)
        is_private = bool(beacon.get("is_private", False))

        if r_id in merged:
            existing = merged[r_id]
            merged[r_id] = RoomLobbyItem(
                room_id=r_id,
                model=existing.model,
                nodes_online=existing.nodes_online + 1,
                total_shared_ram_gb=round(existing.total_shared_ram_gb + ram_gb, 2),
                is_private=existing.is_private or is_private,
                network_scope="hybrid",
            )
        else:
            merged[r_id] = RoomLobbyItem(
                room_id=r_id,
                model=model,
                nodes_online=1,
                total_shared_ram_gb=round(ram_gb, 2),
                is_private=is_private,
                network_scope="lan",
            )

    rooms_list = sorted(merged.values(), key=lambda r: r.nodes_online, reverse=True)
    return rooms_list


@app.get("/models/supported", response_model=list[str])
async def list_supported_models() -> list[str]:
    """Return suggested model IDs available in the UI dropdown."""
    return list(SUPPORTED_MODELS.keys())


@app.get("/models/download-stream")
async def stream_model_download(
    model_id: str,
    shard_layer_start: int | None = None,
    shard_layer_end: int | None = None,
    ram_gb: int | None = None,
) -> StreamingResponse:
    """Ensure any requested preset, custom HF repo, or local `.gguf` model is ready."""
    try:
        if not is_offline_mode() and Path(model_id.strip()).suffix.lower() != ".gguf":
            runtime.model_downloader.resolve_safe_model_dir(model_id)
        else:
            runtime.model_downloader.resolve_safe_gguf_path(model_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    resolved_start = shard_layer_start
    resolved_end = shard_layer_end
    if resolved_start is None and resolved_end is None and ram_gb is not None:
        pre_plan = calculate_max_layers(model_id=model_id, ram_gb=max(2, ram_gb))
        resolved_start = pre_plan.shard_layer_start
        resolved_end = pre_plan.shard_layer_end

    progress_queue: asyncio.Queue[DownloadProgressEvent | None] = asyncio.Queue()

    def _on_progress(event: DownloadProgressEvent) -> None:
        progress_queue.put_nowait(event)

    async def _run_download() -> None:
        try:
            await runtime.model_downloader.ensure_model_downloaded(
                model_id=model_id,
                on_event=_on_progress,
                shard_layer_start=resolved_start,
                shard_layer_end=resolved_end,
            )
        except Exception as exc:
            progress_queue.put_nowait(
                DownloadProgressEvent(
                    model_id=model_id,
                    status="error",
                    progress_percent=0.0,
                    message=f"Download failed: {exc}",
                )
            )
        finally:
            progress_queue.put_nowait(None)

    download_task = asyncio.create_task(_run_download())

    async def _sse_generator() -> AsyncIterator[str]:
        try:
            while True:
                evt = await progress_queue.get()
                if evt is None:
                    break
                payload = json.dumps(
                    {
                        "model_id": evt.model_id,
                        "status": evt.status,
                        "progress": evt.progress_percent,
                        "downloaded_bytes": evt.downloaded_bytes,
                        "total_bytes": evt.total_bytes,
                        "local_path": evt.local_path,
                        "message": evt.message,
                    }
                )
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not download_task.done():
                download_task.cancel()

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/start-node",
    response_model=NodeStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def start_node(request: StartNodeRequest) -> NodeStatusResponse:
    """Load model shard, join LAN/Internet room, and issue an OpenAI Gateway session token."""
    if runtime.state == NodeState.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Node is already running. Stop the node before starting with a new configuration.",
        )

    if request.shard_layer_end < request.shard_layer_start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="shard_layer_end must be greater than or equal to shard_layer_start.",
        )

    effective_ram_gb: int = request.ram_gb
    request.shared_ram_gb = effective_ram_gb
    request.max_ram_mb = effective_ram_gb * 1024

    runtime.state = NodeState.STARTING
    try:
        # 1. Pre-plan shard range from `ram_gb` (or respect explicit P2P split range)
        pre_plan = calculate_max_layers(
            model_id=request.model_id,
            ram_gb=effective_ram_gb,
            requested_start=request.shard_layer_start,
        )
        target_start: int = pre_plan.shard_layer_start
        target_end: int = pre_plan.shard_layer_end
        if request.exact_shard_range:
            max_allowed_end = min(
                pre_plan.total_layers - 1,
                request.shard_layer_start + pre_plan.n_layers - 1,
            )
            target_start = request.shard_layer_start
            target_end = min(request.shard_layer_end, max_allowed_end)

        # 2. Download/resolve Safetensors shard directory (online HF) or `.gguf` model
        local_model_path = await runtime.model_downloader.ensure_model_downloaded(
            request.model_id,
            shard_layer_start=target_start,
            shard_layer_end=target_end,
        )

        # 3. Recompute exact layer plan from downloaded `config.json` / binary size
        layer_plan = calculate_max_layers(
            model_id=request.model_id,
            ram_gb=effective_ram_gb,
            model_path=local_model_path,
            requested_start=request.shard_layer_start,
        )
        if request.exact_shard_range:
            cfg_total = (
                read_hf_config_summary(local_model_path)[0]
                if local_model_path.is_dir()
                else layer_plan.total_layers
            )
            final_start = max(0, min(request.shard_layer_start, cfg_total - 1))
            max_allowed_end = min(
                cfg_total - 1,
                final_start + layer_plan.n_layers - 1,
            )
            final_end = max(final_start, min(request.shard_layer_end, max_allowed_end))
            final_n_layers = final_end - final_start + 1
            final_total_layers = cfg_total
            final_est_ram = min(
                request.max_ram_mb,
                round(
                    (0.6 + (final_n_layers * (layer_plan.per_layer_mb / 1024.0))) * 1024.0
                ),
            )
        else:
            final_start = layer_plan.shard_layer_start
            final_end = layer_plan.shard_layer_end
            final_n_layers = layer_plan.n_layers
            final_total_layers = layer_plan.total_layers
            final_est_ram = layer_plan.estimated_ram_mb

        request.total_layers = final_total_layers
        request.shard_layer_start = final_start
        request.shard_layer_end = final_end

        # 4. Allocate local RAM/VRAM & load ONLY the assigned layers into ShardModelRunner
        runtime.memory_manager.allocate_shard(
            request,
            estimated_ram_mb=final_est_ram,
        )
        runtime.model_runner = await asyncio.to_thread(
            ShardModelRunner,
            model_id=request.model_id,
            model_path=str(local_model_path),
            shard_layer_start=final_start,
            shard_layer_end=final_end,
            total_layers=final_total_layers,
            n_layers=final_n_layers,
            max_ram_mb=request.max_ram_mb,
        )
        effective_backend = (
            ComputeBackend.PYTORCH
            if runtime.model_runner.is_real_model
            else request.backend
        )

        # 5. Start P2P binary activation listener & LAN mDNS advertiser
        await runtime.start_p2p_listener(
            request.p2p_listen_port,
            network_mode=request.network_mode,
        )
        room_key_hash: str | None = hash_room_key(request.room_key)
        is_private_room: bool = room_key_hash is not None

        await lan_mdns_service.start_advertising(
            LanRoomBeacon(
                node_id=request.node_id,
                room_id=request.room_id,
                model=request.model_id,
                ram_gb=float(effective_ram_gb),
                is_private=is_private_room,
                is_unlisted=request.is_unlisted,
                p2p_port=request.p2p_listen_port,
            )
        )

        # 6. Announce to Cloudflare Tracker when enabled
        discovered_peers: list[PeerNodeInfo] = []
        if _is_tracker_enabled():
            try:
                discovered_peers = await runtime.tracker_client.register_node_and_fetch_peers(
                    request
                )
            except httpx.HTTPStatusError as http_exc:
                if http_exc.response.status_code == status.HTTP_403_FORBIDDEN:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Invalid room key for private room '{request.room_id}'.",
                    ) from http_exc
                logger.warning("Tracker registration HTTP error: %s", http_exc)
            except Exception as tracker_exc:
                logger.warning("Tracker unreachable; continuing in local/LAN P2P mode: %s", tracker_exc)

        # 7. Issue cryptographic Session Token stored strictly in process RAM
        runtime.session_token = f"sk-shard-{secrets.token_urlsafe(24)}"
        runtime.node_id = request.node_id
        runtime.model_id = request.model_id
        runtime.model_path = str(local_model_path)
        runtime.network_mode = request.network_mode
        runtime.room_id = request.room_id
        runtime.is_private = is_private_room
        runtime.is_unlisted = request.is_unlisted
        runtime.backend = effective_backend
        runtime.p2p_listen_port = request.p2p_listen_port
        runtime.n_layers = final_n_layers
        runtime.shard_layer_start = final_start
        runtime.shard_layer_end = final_end
        runtime.total_layers = final_total_layers
        runtime.peers = discovered_peers
        runtime.state = NodeState.RUNNING

        return runtime.to_status_response(
            message=(
                f"Gateway Active ({request.network_mode.value.upper()} / Room: '{request.room_id}') — "
                f"layers {final_start}..{final_end} ({final_n_layers}/{final_total_layers}) loaded "
                f"[{effective_backend.value}]."
            )
        )
    except HTTPException:
        runtime.session_token = None
        await lan_mdns_service.stop_advertising()
        await runtime.stop_p2p_listener()
        if runtime.model_runner is not None:
            runtime.model_runner.release()
            runtime.model_runner = None
        runtime.memory_manager.release_all()
        runtime.state = NodeState.ERROR
        raise
    except Exception as exc:
        runtime.session_token = None
        await lan_mdns_service.stop_advertising()
        await runtime.stop_p2p_listener()
        if runtime.model_runner is not None:
            runtime.model_runner.release()
            runtime.model_runner = None
        runtime.memory_manager.release_all()
        runtime.state = NodeState.ERROR
        logger.exception("Failed to start ShardAi node: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start node: {exc}",
        ) from exc


@app.post(
    "/stop-node",
    response_model=NodeStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def stop_node() -> NodeStatusResponse:
    """Stop P2P listener, invalidate session_token, and release all RAM/VRAM shards."""
    if runtime.state == NodeState.STOPPED:
        return runtime.to_status_response(message="Node is already stopped.")

    runtime.state = NodeState.STOPPING
    active_node_id: str | None = runtime.node_id
    active_room_id: str | None = runtime.room_id

    try:
        if (
            active_node_id is not None
            and _is_tracker_enabled()
        ):
            await runtime.tracker_client.unregister_node(
                active_node_id,
                room_id=active_room_id,
            )
    except Exception as exc:
        logger.warning("Could not unregister node from signaling tracker: %s", exc)
    finally:
        runtime.session_token = None
        await lan_mdns_service.stop_advertising()
        await runtime.stop_p2p_listener()
        if runtime.model_runner is not None:
            runtime.model_runner.release()
            runtime.model_runner = None
        runtime.memory_manager.release_all()
        runtime.pending_token_queues.clear()
        runtime.peers = []
        runtime.node_id = None
        runtime.model_id = None
        runtime.model_path = None
        runtime.network_mode = None
        runtime.room_id = None
        runtime.is_private = False
        runtime.is_unlisted = False
        runtime.backend = None
        runtime.p2p_listen_port = None
        runtime.n_layers = 0
        runtime.state = NodeState.STOPPED

    return runtime.to_status_response(
        message="Node stopped, session token revoked, and all RAM/VRAM shards released."
    )


@app.get("/peers", response_model=list[PeerNodeInfo])
async def list_peers() -> list[PeerNodeInfo]:
    """Return the list of direct P2P peers currently known to this node."""
    return runtime.peers


@app.post("/peers/register", response_model=list[PeerNodeInfo])
async def register_peer(peer: PeerNodeInfo) -> list[PeerNodeInfo]:
    """Register or update a direct P2P peer in the local routing table."""
    runtime.peers = [p for p in runtime.peers if p.peer_id != peer.peer_id]
    runtime.peers.append(peer)
    runtime.peers.sort(key=lambda p: p.shard_layer_start)
    return runtime.peers


# ==============================================================================
# OpenAI-Compatible Local AI Gateway Endpoints (/v1/models, /v1/chat/completions)
# ==============================================================================


@app.get("/v1/models", response_model=OpenAIModelListResponse)
async def openai_list_models(
    _token: str = Depends(verify_openai_bearer_token),
) -> OpenAIModelListResponse:
    """OpenAI-compatible model listing protected by active Bearer session token."""
    active_model: str = runtime.model_id or "shardai-p2p-model"
    return OpenAIModelListResponse(
        data=[
            OpenAIModelObject(
                id=active_model,
                created=int(time.time()),
                owned_by="shardai-p2p",
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=None)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    _token: str = Depends(verify_openai_bearer_token),
) -> StreamingResponse | dict[str, Any]:
    """OpenAI-compatible Chat Completions endpoint routed through P2P Pipeline Inference.

    Supports both `stream: true` (standard `choices[0].delta.content` SSE chunks)
    and `stream: false` (single JSON completion payload) for IDE / curl integration.
    """
    prompt_text: str = _extract_prompt_from_messages(request.messages)
    chat_messages: list[dict[str, str]] = _extract_chat_messages_dicts(request.messages)
    max_tokens: int = request.max_tokens or 256
    temperature: float = request.temperature if request.temperature is not None else 0.7
    completion_id: str = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts: int = int(time.time())
    served_model: str = request.model or runtime.model_id or "shardai-p2p-model"

    if request.stream:
        async def _openai_sse_stream() -> AsyncIterator[str]:
            # Emit initial assistant role delta chunk per OpenAI streaming spec
            role_chunk = json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": served_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield f"data: {role_chunk}\n\n"

            async for _req_id, _step, _tok_id, token_text, _is_eos in (
                runtime.stream_p2p_pipeline_tokens(
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    compress_activations=True,
                    messages=chat_messages,
                    temperature=temperature,
                )
            ):
                if not token_text:
                    continue
                content_chunk = json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": served_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield f"data: {content_chunk}\n\n"

            stop_chunk = json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": served_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield f"data: {stop_chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _openai_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming completion: aggregate tokens from P2P pipeline
    collected_tokens: list[str] = []
    async for _req_id, _step, _tok_id, token_text, _is_eos in (
        runtime.stream_p2p_pipeline_tokens(
            prompt=prompt_text,
            max_tokens=max_tokens,
            compress_activations=True,
            messages=chat_messages,
            temperature=temperature,
        )
    ):
        collected_tokens.append(token_text)

    full_text: str = "".join(collected_tokens).strip()
    prompt_token_count: int = max(1, len(prompt_text.split()))
    completion_token_count: int = len(collected_tokens)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": served_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": prompt_token_count + completion_token_count,
        },
    }


@app.post("/generate")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Internal SSE token stream endpoint backed by the P2P Pipeline Inference runner."""

    async def _sse_token_generator() -> AsyncIterator[str]:
        async for req_id, step_idx, token_id, token_text, is_eos in (
            runtime.stream_p2p_pipeline_tokens(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                compress_activations=request.compress_activations,
            )
        ):
            event_payload: str = json.dumps(
                {
                    "request_id": req_id,
                    "step": step_idx,
                    "token_id": token_id,
                    "token": token_text,
                    "is_eos": is_eos,
                }
            )
            yield f"data: {event_payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse_token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    # SECURITY: Bind strictly to 127.0.0.1 so only local Tauri UI & local IDE can invoke APIs
    local_port: int = int(os.getenv("SHARDAI_ENGINE_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=local_port,
        reload=False,
        log_level="info",
    )
