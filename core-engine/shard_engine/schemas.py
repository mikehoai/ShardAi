"""Pydantic schemas with strict typing for Tauri UI IPC and Signaling Server payloads."""

from enum import StrEnum
from pydantic import BaseModel, Field


class NodeState(StrEnum):
    """Lifecycle states of the local ShardAi Node."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class ComputeBackend(StrEnum):
    """Supported AI inference backends for executing model shards."""

    LLAMA_CPP = "llama_cpp"
    PYTORCH = "pytorch"


class PeerNodeInfo(BaseModel):
    """Peer metadata returned by the central Signaling Server (Cloudflare Tracker)."""

    peer_id: str = Field(..., description="Unique identifier of the peer node")
    ip_address: str = Field(..., description="Direct P2P reachable IP address")
    port: int = Field(..., ge=1, le=65535, description="Direct P2P listening port")
    shard_layer_start: int = Field(..., ge=0, description="Starting transformer layer index")
    shard_layer_end: int = Field(..., ge=0, description="Ending transformer layer index")
    available_vram_mb: int = Field(default=0, ge=0, description="Available VRAM shared by peer")


class StartNodeRequest(BaseModel):
    """Payload sent from Tauri UI to start the local node and register with the tracker."""

    node_id: str = Field(
        default="local-node-01",
        min_length=3,
        max_length=64,
        description="Unique identifier for this client node",
    )
    model_id: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Target distributed LLM identifier (e.g., meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct)",
    )
    backend: ComputeBackend = Field(
        default=ComputeBackend.LLAMA_CPP,
        description="Inference runtime used to load the assigned shard",
    )
    ram_gb: int = Field(
        default=4,
        ge=2,
        le=16,
        description="Shared system RAM in GB selected on the UI slider (2GB - 16GB)",
    )
    shared_ram_gb: int = Field(
        default=4,
        ge=2,
        le=16,
        description="Alias for ram_gb selected on the Tauri UI slider (2GB - 16GB)",
    )
    max_ram_mb: int = Field(
        default=4096,
        ge=512,
        le=262144,
        description="Maximum system RAM (MB) allowed for model shards",
    )
    max_vram_mb: int = Field(
        default=2048,
        ge=0,
        le=131072,
        description="Maximum GPU VRAM (MB) allowed for model shards",
    )
    p2p_listen_port: int = Field(
        default=49152,
        ge=1024,
        le=65535,
        description="Local TCP port for direct P2P tensor transmission",
    )
    shard_layer_start: int = Field(default=0, ge=0, description="First layer hosted locally")
    shard_layer_end: int = Field(default=15, ge=0, description="Last layer hosted locally")
    total_layers: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Total transformer layers across the distributed model pipeline",
    )


class GenerateRequest(BaseModel):
    """Payload sent from Tauri/React UI to trigger distributed P2P pipeline inference."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=16384,
        description="User prompt text to run through the distributed LLM pipeline",
    )
    max_tokens: int = Field(
        default=16,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate and stream via SSE",
    )
    compress_activations: bool = Field(
        default=True,
        description="Enable adaptive LZ4/Zlib compression on intermediate hidden-state tensors",
    )


class NodeStatusResponse(BaseModel):
    """Current runtime status of the local ShardAi Node returned to Tauri UI."""

    state: NodeState
    node_id: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    backend: ComputeBackend | None = None
    allocated_ram_mb: int = 0
    allocated_vram_mb: int = 0
    p2p_listen_port: int | None = None
    connected_peers_count: int = 0
    peers: list[PeerNodeInfo] = Field(default_factory=list)
    message: str = ""
