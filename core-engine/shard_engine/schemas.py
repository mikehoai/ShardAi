"""Pydantic schemas with strict typing for Tauri UI IPC, Signaling Server, and OpenAI Gateway."""

from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


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


class NetworkMode(StrEnum):
    """Transport scope for P2P node discovery and direct tensor streaming."""

    LAN = "lan"
    INTERNET = "internet"


class PeerNodeInfo(BaseModel):
    """Peer metadata returned by the central Signaling Server (Cloudflare Tracker)."""

    peer_id: str = Field(..., description="Unique identifier of the peer node")
    ip_address: str = Field(..., description="Direct P2P reachable IP address")
    port: int = Field(..., ge=1, le=65535, description="Direct P2P listening port")
    shard_layer_start: int = Field(..., ge=0, description="Starting transformer layer index")
    shard_layer_end: int = Field(..., ge=0, description="Ending transformer layer index")
    available_vram_mb: int = Field(default=0, ge=0, description="Available VRAM shared by peer")
    room_id: str = Field(
        default="public",
        description="Room identifier ('public' or custom private room passphrase)",
    )


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
        min_length=1,
        max_length=512,
        description="HuggingFace Repo ID, HuggingFace URL, or local .gguf file path",
    )
    network_mode: NetworkMode = Field(
        default=NetworkMode.INTERNET,
        description="Network scope: 'lan' or 'internet'",
    )
    room_id: str = Field(
        default="public",
        max_length=128,
        description="Room ID ('public' by default, or custom room name)",
    )
    room_key: str | None = Field(
        default=None,
        max_length=256,
        description="Optional secret key for private rooms; hashed with SHA-256 before transmission",
    )
    is_unlisted: bool = Field(
        default=False,
        description="If true, hide this room from the public Room Lobby browser",
    )
    backend: ComputeBackend = Field(
        default=ComputeBackend.LLAMA_CPP,
        description="Inference runtime used to load the assigned shard",
    )
    ram_gb: int = Field(
        default=4,
        ge=2,
        le=256,
        description="Shared system RAM in GB selected on the UI slider",
    )
    shared_ram_gb: int = Field(
        default=4,
        ge=2,
        le=256,
        description="Alias for ram_gb selected on the Tauri UI slider",
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
    exact_shard_range: bool = Field(
        default=False,
        description="If true, load exact [shard_layer_start..shard_layer_end] capped by RAM limit",
    )

    @field_validator("room_id", mode="before")
    @classmethod
    def _normalize_room_id(cls, value: str | None) -> str:
        if value is None:
            return "public"
        cleaned = value.strip()
        return cleaned if cleaned else "public"

    @field_validator("room_key", mode="before")
    @classmethod
    def _normalize_room_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned if cleaned else None

    @field_validator("model_id", mode="before")
    @classmethod
    def _normalize_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        # Normalize full HuggingFace URLs into owner/repo format
        if cleaned.startswith("https://huggingface.co/") or cleaned.startswith(
            "http://huggingface.co/"
        ):
            parts = cleaned.split("huggingface.co/", 1)[1].strip("/").split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        return cleaned


class GenerateRequest(BaseModel):
    """Payload to trigger internal P2P pipeline SSE generation."""

    prompt: str = Field(..., min_length=1, max_length=16384)
    max_tokens: int = Field(default=16, ge=1, le=2048)
    compress_activations: bool = Field(default=True)


class SystemMemoryResponse(BaseModel):
    """Host system physical memory snapshot for safe RAM slider bounds."""

    total_ram_gb: float
    available_ram_gb: float
    max_shareable_ram_gb: int


class RoomLobbyItem(BaseModel):
    """Public or Private (locked) room summary displayed in the Network Rooms Lobby."""

    room_id: str
    model: str
    nodes_online: int = Field(..., ge=1)
    total_shared_ram_gb: float = Field(..., ge=0.0)
    is_private: bool = False
    network_scope: Literal["internet", "lan", "hybrid"] = "internet"


class NodeStatusResponse(BaseModel):
    """Current runtime status of the local ShardAi Node returned to Tauri UI."""

    state: NodeState
    node_id: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    network_mode: NetworkMode | None = None
    room_id: str | None = None
    is_private: bool = False
    is_unlisted: bool = False
    backend: ComputeBackend | None = None
    n_layers: int = 0
    total_layers: int = 32
    shard_layer_start: int | None = None
    shard_layer_end: int | None = None
    allocated_ram_mb: int = 0
    allocated_vram_mb: int = 0
    p2p_listen_port: int | None = None
    connected_peers_count: int = 0
    peers: list[PeerNodeInfo] = Field(default_factory=list)
    session_token: str | None = None
    api_base_url: str | None = None
    message: str = ""


# ==============================================================================
# OpenAI-Compatible API Schemas (/v1/models & /v1/chat/completions)
# ==============================================================================


class ChatMessage(BaseModel):
    """Single message entry in an OpenAI chat completions request."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible request payload for POST /v1/chat/completions."""

    model: str = Field(..., min_length=1, description="Model identifier")
    messages: list[ChatMessage] = Field(..., min_length=1)
    stream: bool = Field(default=False, description="Enable OpenAI SSE token streaming")
    max_tokens: int | None = Field(default=256, ge=1, le=4096)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)


class OpenAIModelObject(BaseModel):
    """OpenAI-compatible model metadata entry."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "shardai-p2p"


class OpenAIModelListResponse(BaseModel):
    """OpenAI-compatible response for GET /v1/models."""

    object: Literal["list"] = "list"
    data: list[OpenAIModelObject]
