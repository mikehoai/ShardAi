"""ShardAi Core Engine — Local FastAPI Server for Tauri IPC & P2P Node Control.

Security & Architecture Notes:
- Binds strictly to `127.0.0.1` (loopback only) as mandated in `.antigravity/review.md`.
- Uses `SignalingTrackerClient` (with explicit timeout + retry) for Cloudflare Tracker calls.
- Uses `ShardMemoryManager` to guarantee zero RAM/VRAM leak on `/stop-node` or app exit.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from shard_engine.memory_manager import ShardMemoryManager
from shard_engine.schemas import (
    ComputeBackend,
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
    """Encapsulates the runtime state of the local ShardAi Node."""

    def __init__(self) -> None:
        self.state: NodeState = NodeState.STOPPED
        self.node_id: str | None = None
        self.model_id: str | None = None
        self.backend: ComputeBackend | None = None
        self.p2p_listen_port: int | None = None
        self.peers: list[PeerNodeInfo] = []
        self.memory_manager: ShardMemoryManager = ShardMemoryManager()
        self.tracker_client: SignalingTrackerClient = SignalingTrackerClient(
            base_url=os.getenv(
                "SHARDAI_TRACKER_URL", "https://tracker.shardai.network/api/v1"
            ),
            timeout_seconds=8.0,
            connect_timeout_seconds=4.0,
            max_retries=3,
        )

    def to_status_response(self, message: str = "") -> NodeStatusResponse:
        """Build a typed status snapshot for the Tauri UI."""
        return NodeStatusResponse(
            state=self.state,
            node_id=self.node_id,
            model_id=self.model_id,
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
    """Ensure model shards and P2P resources are cleanly released on server shutdown."""
    logger.info("ShardAi Core Engine starting up on loopback interface...")
    yield
    logger.info("ShardAi Core Engine shutting down; releasing RAM/VRAM shards...")
    runtime.memory_manager.release_all()
    runtime.state = NodeState.STOPPED


app = FastAPI(
    title="ShardAi Local Core Engine API",
    description="Local control plane called by the Tauri Desktop UI to manage LLM shards and P2P networking.",
    version="0.1.0",
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


@app.post(
    "/start-node",
    response_model=NodeStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def start_node(request: StartNodeRequest) -> NodeStatusResponse:
    """Load the assigned LLM shard into RAM/VRAM, announce to Signaling Server, and start P2P node."""
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

    runtime.state = NodeState.STARTING
    try:
        # 1. Allocate local RAM/VRAM for the model shard
        runtime.memory_manager.allocate_shard(request)

        # 2. Announce to central Signaling Server (Cloudflare Tracker) & fetch peer IPs
        # In offline/local-dev mode without a live tracker, gracefully fallback to empty peer list
        discovered_peers: list[PeerNodeInfo] = []
        if os.getenv("SHARDAI_OFFLINE_MODE", "true").lower() != "true":
            discovered_peers = await runtime.tracker_client.register_node_and_fetch_peers(
                request
            )

        runtime.node_id = request.node_id
        runtime.model_id = request.model_id
        runtime.backend = request.backend
        runtime.p2p_listen_port = request.p2p_listen_port
        runtime.peers = discovered_peers
        runtime.state = NodeState.RUNNING

        return runtime.to_status_response(
            message=f"Node '{request.node_id}' started and ready for direct P2P sharding."
        )
    except Exception as exc:
        # Guarantee no partial RAM/VRAM leak if startup or signaling fails
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
    """Stop P2P connections, unregister from Signaling Server, and release all RAM/VRAM shards."""
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
        # Always free RAM/VRAM regardless of network state
        runtime.memory_manager.release_all()
        runtime.peers = []
        runtime.node_id = None
        runtime.model_id = None
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
