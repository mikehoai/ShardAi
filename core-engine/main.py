"""ShardAi Core Engine — Local FastAPI Server & P2P Pipeline Inference Coordinator.

Security & Architecture Notes:
- Binds strictly to `127.0.0.1` (loopback only) as mandated in `.antigravity/review.md`.
- Uses `ShardModelRunner` (`core-engine/models/`) for low-level shard `forward_pass`.
- Uses `p2p.serializer` and `p2p.transport` (`core-engine/p2p/`) for binary `struct`
  tensor serialization (with adaptive LZ4/Zlib compression) and TCP socket streaming.
- Streams generated tokens to the React UI via Server-Sent Events (SSE) on `/generate`.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from models.downloader import (
    SUPPORTED_MODELS,
    DownloadProgressEvent,
    ModelDownloader,
)
from models.shard_runner import ShardModelRunner
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
from shard_engine.memory_manager import ShardMemoryManager
from shard_engine.schemas import (
    ComputeBackend,
    GenerateRequest,
    NodeState,
    NodeStatusResponse,
    PeerNodeInfo,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shardai.main")


class LocalNodeRuntime:
    """Encapsulates the runtime state, P2P listener, and model shard pipeline of this node."""

    def __init__(self) -> None:
        self.state: NodeState = NodeState.STOPPED
        self.node_id: str | None = None
        self.model_id: str | None = None
        self.model_path: str | None = None
        self.backend: ComputeBackend | None = None
        self.p2p_listen_port: int | None = None
        self.shard_layer_start: int = 0
        self.shard_layer_end: int = 15
        self.total_layers: int = 32
        self.peers: list[PeerNodeInfo] = []
        self.memory_manager: ShardMemoryManager = ShardMemoryManager()
        self.model_downloader: ModelDownloader = ModelDownloader()
        self.model_runner: ShardModelRunner | None = None
        self.p2p_server: asyncio.AbstractServer | None = None
        self.pending_token_queues: dict[str, asyncio.Queue[DecodedTokenFrame]] = {}
        self.tracker_client: SignalingTrackerClient = SignalingTrackerClient(
            base_url=os.getenv(
                "SHARDAI_TRACKER_URL", "https://tracker.shardai.network/api/v1"
            ),
            timeout_seconds=8.0,
            connect_timeout_seconds=4.0,
            max_retries=3,
        )

    def find_next_Hop_peer(self, completed_layer: int) -> PeerNodeInfo | None:
        """Locate the downstream P2P peer responsible for layer `completed_layer + 1`."""
        target_next_layer: int = completed_layer + 1
        for peer in self.peers:
            if peer.shard_layer_start <= target_next_layer <= peer.shard_layer_end:
                return peer
        # Fallback: pick the first peer whose shard starts after completed_layer
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

        # 1. Run forward pass through the transformer layers hosted by this node
        output_tensor_bytes, output_shape = self.model_runner.forward_pass(
            frame.raw_tensor_bytes,
            shape=frame.shape,
        )
        completed_layer: int = self.shard_layer_end

        # 2. Routing decision: Is this the Tail Node (final layer) or an Intermediate Node?
        is_final_layer: bool = completed_layer >= (frame.total_layers - 1)
        next_peer: PeerNodeInfo | None = (
            None if is_final_layer else self.find_next_Hop_peer(completed_layer)
        )

        if is_final_layer or next_peer is None:
            # Tail Node: decode the final hidden state into a text token and send back to origin
            token_id, token_text, is_eos = self.model_runner.decode_final_layer_to_token(
                output_tensor_bytes,
                step_index=frame.step_index,
                max_tokens=frame.max_tokens,
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
        else:
            # Intermediate Node: serialize new hidden states and forward to the next peer
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

    async def handle_incoming_decoded_token(self, token_frame: DecodedTokenFrame) -> None:
        """Callback triggered on the Head Node when the Tail Node returns a decoded token."""
        queue = self.pending_token_queues.get(token_frame.request_id)
        if queue is not None:
            await queue.put(token_frame)

    async def start_p2p_listener(self, port: int) -> None:
        """Start the async TCP P2P listener (`receive_activations`)."""
        await self.stop_p2p_listener()
        bind_host: str = os.getenv("SHARDAI_P2P_BIND_HOST", "127.0.0.1")
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
        return NodeStatusResponse(
            state=self.state,
            node_id=self.node_id,
            model_id=self.model_id,
            model_path=self.model_path,
            backend=self.backend,
            allocated_ram_mb=self.memory_manager.allocated_ram_mb,
            allocated_vram_mb=self.memory_manager.allocated_vram_mb,
            p2p_listen_port=self.p2p_listen_port,
            connected_peers_count=len(self.peers),
            peers=self.peers,
            message=message,
        )


runtime = LocalNodeRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Ensure model shards and P2P listeners are cleanly released on server shutdown."""
    logger.info("ShardAi Core Engine starting up on loopback interface...")
    yield
    logger.info("ShardAi Core Engine shutting down; releasing P2P listener and RAM/VRAM shards...")
    await runtime.stop_p2p_listener()
    if runtime.model_runner is not None:
        runtime.model_runner.release()
        runtime.model_runner = None
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


app = FastAPI(
    title="ShardAi Local Core Engine API",
    description="Local control plane & P2P pipeline inference engine called by the Tauri Desktop UI.",
    version="0.2.0",
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
    allow_headers=["Content-Type", "X-ShardAi-Local-Token"],
)


@app.get("/health", response_model=dict[str, str])
async def health_check() -> dict[str, str]:
    """Lightweight liveness probe for Tauri sidecar supervision."""
    return {"status": "ok", "service": "shardai-core-engine"}


@app.get("/status", response_model=NodeStatusResponse)
async def get_node_status() -> NodeStatusResponse:
    """Return current node state, RAM/VRAM usage, and connected P2P peers."""
    return runtime.to_status_response(message="Node status retrieved successfully.")


@app.get("/models/supported", response_model=list[str])
async def list_supported_models() -> list[str]:
    """Return the list of supported model_ids available for selection in the UI."""
    return list(SUPPORTED_MODELS.keys())


@app.get("/models/download-stream")
async def stream_model_download(model_id: str) -> StreamingResponse:
    """Ensure the requested `.gguf` model is in local cache or download via `hf_hub_download`.

    Streams real-time percentage updates (`0%` .. `100%`) as Server-Sent Events (SSE)
    so the React UI can render a live Progress Bar before `/start-node` initializes `llama_cpp`.
    """
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported model_id '{model_id}'. Supported: {list(SUPPORTED_MODELS.keys())}",
        )

    progress_queue: asyncio.Queue[DownloadProgressEvent | None] = asyncio.Queue()

    def _on_progress(event: DownloadProgressEvent) -> None:
        progress_queue.put_nowait(event)

    async def _run_download() -> None:
        try:
            await runtime.model_downloader.ensure_model_downloaded(
                model_id=model_id,
                on_event=_on_progress,
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
    """Download/verify `.gguf` model, load assigned shard layers in `llama_cpp`, and announce to Tracker."""
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

    if request.model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported model_id '{request.model_id}'. Supported: {list(SUPPORTED_MODELS.keys())}",
        )

    # Synchronize max_ram_mb if ram_gb was passed from the UI slider
    effective_ram_gb: int = request.ram_gb
    request.shared_ram_gb = effective_ram_gb
    request.max_ram_mb = effective_ram_gb * 1024

    runtime.state = NodeState.STARTING
    try:
        # 1. Ensure `.gguf` model is in local cache (or download via huggingface_hub)
        local_gguf_path = await runtime.model_downloader.ensure_model_downloaded(
            request.model_id
        )

        # 2. Allocate local RAM/VRAM & initialize `llama_cpp` with local_gguf_path and shared layers
        runtime.memory_manager.allocate_shard(request)
        runtime.model_runner = ShardModelRunner(
            model_id=request.model_id,
            model_path=str(local_gguf_path),
            shard_layer_start=request.shard_layer_start,
            shard_layer_end=request.shard_layer_end,
            total_layers=request.total_layers,
        )

        # 3. Start P2P binary activation listener on the configured port
        await runtime.start_p2p_listener(request.p2p_listen_port)

        # 4. Call POST /announce on Cloudflare Signaling Tracker with `model = request.model_id`
        discovered_peers: list[PeerNodeInfo] = []
        if os.getenv("SHARDAI_OFFLINE_MODE", "true").lower() != "true":
            discovered_peers = await runtime.tracker_client.register_node_and_fetch_peers(
                request
            )

        runtime.node_id = request.node_id
        runtime.model_id = request.model_id
        runtime.model_path = str(local_gguf_path)
        runtime.backend = request.backend
        runtime.p2p_listen_port = request.p2p_listen_port
        runtime.shard_layer_start = request.shard_layer_start
        runtime.shard_layer_end = request.shard_layer_end
        runtime.total_layers = request.total_layers
        runtime.peers = discovered_peers
        runtime.state = NodeState.RUNNING

        return runtime.to_status_response(
            message=(
                f"Node '{request.node_id}' started with model '{request.model_id}' "
                f"({local_gguf_path.name}) on P2P port {request.p2p_listen_port}."
            )
        )
    except Exception as exc:
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
    """Stop P2P listener, unregister from Signaling Server, and release all RAM/VRAM shards."""
    if runtime.state == NodeState.STOPPED:
        return runtime.to_status_response(message="Node is already stopped.")

    runtime.state = NodeState.STOPPING
    active_node_id: str | None = runtime.node_id

    try:
        if (
            active_node_id is not None
            and os.getenv("SHARDAI_OFFLINE_MODE", "true").lower() != "true"
        ):
            await runtime.tracker_client.unregister_node(active_node_id)
    except Exception as exc:
        logger.warning("Could not unregister node from signaling tracker: %s", exc)
    finally:
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
        runtime.backend = None
        runtime.p2p_listen_port = None
        runtime.state = NodeState.STOPPED

    return runtime.to_status_response(
        message="Node stopped and all RAM/VRAM shards released."
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


@app.post("/generate")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Execute distributed P2P Pipeline Inference and stream tokens via Server-Sent Events (SSE).

    Pipeline Routing Logic:
    - Requires this node to be RUNNING and holding Layer 0 (`shard_layer_start == 0`).
    - Tokenizes the prompt -> runs local `forward_pass` through `0..shard_layer_end`.
    - If a downstream P2P peer holds `shard_layer_end + 1`:
      Serializes the output hidden state into a compressed binary `struct` stream (`SHRD`),
      pushes it via `send_activations`, and awaits the decoded token returned over P2P
      from the Tail Node (holding the final layer).
    - If this node also holds the final layer (or no remote peer is registered):
      Decodes the output tensor into a token locally and streams it via SSE.
    """
    if runtime.state != NodeState.RUNNING or runtime.model_runner is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Node must be RUNNING before invoking /generate.",
        )

    if not runtime.model_runner.is_head_node:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"This node hosts layers {runtime.shard_layer_start}..{runtime.shard_layer_end}. "
                "Prompt initiation (/generate) must be sent to a Head Node holding layer 0."
            ),
        )

    request_id: str = str(uuid.uuid4())
    token_queue: asyncio.Queue[DecodedTokenFrame] = asyncio.Queue()
    runtime.pending_token_queues[request_id] = token_queue

    async def _sse_token_generator() -> AsyncIterator[str]:
        try:
            assert runtime.model_runner is not None
            token_ids: list[int] = runtime.model_runner.tokenize(request.prompt)
            origin_port: int = runtime.p2p_listen_port or 49152

            for step_idx in range(request.max_tokens):
                # 1. Run forward pass across Layer 0 .. shard_layer_end
                hidden_bytes, tensor_shape = runtime.model_runner.forward_pass(token_ids)
                next_peer: PeerNodeInfo | None = runtime.find_next_Hop_peer(
                    runtime.shard_layer_end
                )

                if (
                    next_peer is not None
                    and runtime.shard_layer_end < (runtime.total_layers - 1)
                ):
                    # 2A. Distributed P2P Pipeline: serialize binary tensor & push to next peer
                    binary_frame: bytes = serialize_activation_tensor(
                        tensor_data=hidden_bytes,
                        shape=tensor_shape,
                        request_id=request_id,
                        origin_ip="127.0.0.1",
                        origin_port=origin_port,
                        current_layer=runtime.shard_layer_end,
                        total_layers=runtime.total_layers,
                        step_index=step_idx,
                        max_tokens=request.max_tokens,
                        network_congested=congestion_monitor.is_congested,
                        force_compression=request.compress_activations,
                    )
                    await send_activations(
                        peer_ip=next_peer.ip_address,
                        byte_stream=binary_frame,
                        peer_port=next_peer.port,
                    )

                    # Wait for the Tail Node to decode the final layer and stream the token back
                    decoded_frame: DecodedTokenFrame = await asyncio.wait_for(
                        token_queue.get(),
                        timeout=10.0,
                    )
                    token_id = decoded_frame.token_id
                    token_text = decoded_frame.token_text
                    is_eos = decoded_frame.is_eos
                else:
                    # 2B. Local / Single-Node Pipeline completion
                    token_id, token_text, is_eos = (
                        runtime.model_runner.decode_final_layer_to_token(
                            hidden_bytes,
                            step_index=step_idx,
                            max_tokens=request.max_tokens,
                        )
                    )
                    await asyncio.sleep(0.02)

                event_payload: str = json.dumps(
                    {
                        "request_id": request_id,
                        "step": step_idx,
                        "token_id": token_id,
                        "token": token_text,
                        "is_eos": is_eos,
                    }
                )
                yield f"data: {event_payload}\n\n"

                token_ids.append(token_id)
                if is_eos:
                    break

            yield "data: [DONE]\n\n"
        finally:
            runtime.pending_token_queues.pop(request_id, None)

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
    # SECURITY: Bind strictly to 127.0.0.1 so only local Tauri UI can invoke control endpoints
    local_port: int = int(os.getenv("SHARDAI_ENGINE_PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=local_port,
        reload=False,
        log_level="info",
    )
