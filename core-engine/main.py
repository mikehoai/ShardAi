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
import re
import secrets
import socket
import time
from typing import Any
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
    PeerRole,
    RoomLobbyItem,
    StartNodeRequest,
    SystemMemoryResponse,
)
from shard_engine.tracker_client import (
    AnnounceAssignment,
    SignalingTrackerClient,
    hash_room_key,
)

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
        self.role: PeerRole = PeerRole.PRIMARY
        self.backend: ComputeBackend | None = None
        self.p2p_listen_port: int | None = None
        self.n_layers: int = 0
        self.shard_layer_start: int = 0
        self.shard_layer_end: int = 15
        self.total_layers: int = 32
        self.peers: list[PeerNodeInfo] = []
        self.session_token: str | None = None
        self.coverage_percent: float = 0.0
        self.is_swarm_ready: bool = False
        self._coverage_poll_task: asyncio.Task[None] | None = None
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

    def find_candidate_peers_for_next_layer(
        self, completed_layer: int
    ) -> list[PeerNodeInfo]:
        """Return ordered P2P peers covering `completed_layer + 1` (Primary first, then Backups)."""
        target_next_layer: int = completed_layer + 1
        covering = [
            peer
            for peer in self.peers
            if peer.shard_layer_start <= target_next_layer <= peer.shard_layer_end
        ]
        if not covering:
            covering = [p for p in self.peers if p.shard_layer_start > completed_layer]

        # Prioritize role == "primary" (0) over role == "backup" (1)
        covering.sort(
            key=lambda p: (
                0 if p.role == PeerRole.PRIMARY else 1,
                p.shard_layer_start,
                -p.shard_layer_end,
            )
        )
        return covering

    def find_next_hop_peer(self, completed_layer: int) -> PeerNodeInfo | None:
        """Locate the highest-priority downstream P2P peer for `completed_layer + 1`."""
        candidates = self.find_candidate_peers_for_next_layer(completed_layer)
        return candidates[0] if candidates else None

    async def send_with_failover(
        self,
        completed_layer: int,
        frame_bytes: bytes,
        timeout_seconds: float = 5.0,
    ) -> PeerNodeInfo:
        """Push binary activation tensors to the primary peer, failing over to backup peers on error."""
        candidates = self.find_candidate_peers_for_next_layer(completed_layer)
        if not candidates:
            raise RuntimeError(
                f"No downstream peer available for layer {completed_layer + 1}."
            )

        last_exc: Exception | None = None
        for idx, candidate in enumerate(candidates):
            try:
                await send_activations(
                    peer_ip=candidate.ip_address,
                    byte_stream=frame_bytes,
                    peer_port=candidate.port,
                    timeout_seconds=timeout_seconds,
                )
                if idx > 0:
                    logger.info(
                        "Failover succeeded: routed layer %d..%d activation to %s peer %s (%s:%d)",
                        candidate.shard_layer_start,
                        candidate.shard_layer_end,
                        candidate.role.value,
                        candidate.peer_id,
                        candidate.ip_address,
                        candidate.port,
                    )
                return candidate
            except (TimeoutError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
                last_exc = exc
                logger.warning(
                    "Downstream %s peer %s (%s:%d) failed (%s); failing over to next backup peer...",
                    candidate.role.value,
                    candidate.peer_id,
                    candidate.ip_address,
                    candidate.port,
                    exc,
                )

        raise RuntimeError(
            f"All primary and backup peers failed for layer {completed_layer + 1}: {last_exc}"
        ) from last_exc

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
            await self.send_with_failover(
                completed_layer=completed_layer,
                frame_bytes=next_frame_bytes,
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
                    candidates = self.find_candidate_peers_for_next_layer(
                        self.shard_layer_end
                    )
                    decoded_frame: DecodedTokenFrame | None = None
                    last_step_err: Exception | None = None

                    for idx, candidate in enumerate(candidates):
                        advertise_ip: str = _resolve_advertise_ip(
                            candidate.ip_address, candidate.port
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
                        try:
                            await send_activations(
                                peer_ip=candidate.ip_address,
                                byte_stream=binary_frame,
                                peer_port=candidate.port,
                                timeout_seconds=5.0,
                            )
                            decoded_frame = await asyncio.wait_for(
                                token_queue.get(),
                                timeout=60.0 if len(candidates) == 1 else 10.0,
                            )
                            if idx > 0:
                                logger.info(
                                    "Pipeline failover succeeded via %s peer %s (%s:%d)",
                                    candidate.role.value,
                                    candidate.peer_id,
                                    candidate.ip_address,
                                    candidate.port,
                                )
                            break
                        except (
                            TimeoutError,
                            asyncio.TimeoutError,
                            OSError,
                            RuntimeError,
                        ) as step_exc:
                            last_step_err = step_exc
                            logger.warning(
                                "Peer %s (%s) dropped/timed out at step %d (%s); attempting backup failover...",
                                candidate.peer_id,
                                candidate.role.value,
                                step_idx,
                                step_exc,
                            )

                    if decoded_frame is None:
                        raise RuntimeError(
                            f"All primary and backup peers failed at step {step_idx}: {last_step_err}"
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
    ) -> int:
        """Start the async TCP P2P listener (`receive_activations`) and return the bound port."""
        await self.stop_p2p_listener()
        default_bind = "0.0.0.0" if network_mode == NetworkMode.LAN else "127.0.0.1"
        bind_host: str = os.getenv("SHARDAI_P2P_BIND_HOST", default_bind)

        candidates = [port] + [
            port + offset for offset in range(1, 32) if port + offset <= 65535
        ] + [0]
        last_exc: OSError | None = None
        for candidate_port in candidates:
            try:
                self.p2p_server = await receive_activations(
                    host=bind_host,
                    port=candidate_port,
                    on_activation_frame=self.handle_incoming_activation,
                    on_token_frame=self.handle_incoming_decoded_token,
                )
                actual_port = (
                    int(self.p2p_server.sockets[0].getsockname()[1])
                    if self.p2p_server is not None and self.p2p_server.sockets
                    else candidate_port
                )
                if actual_port != port:
                    logger.info(
                        "Requested P2P port %d unavailable; bound P2P listener on %s:%d",
                        port,
                        bind_host,
                        actual_port,
                    )
                return actual_port
            except OSError as exc:
                last_exc = exc
                logger.debug(
                    "Could not bind P2P listener on %s:%d (%s); trying next port...",
                    bind_host,
                    candidate_port,
                    exc,
                )

        assert last_exc is not None
        raise last_exc

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

    def recompute_local_coverage(self) -> float:
        """Compute current model layer coverage percentage from this node + known peers."""
        if self.state != NodeState.RUNNING or self.total_layers <= 0:
            self.coverage_percent = 0.0
            self.is_swarm_ready = False
            return 0.0

        covered: set[int] = set(
            range(
                max(0, self.shard_layer_start),
                min(self.total_layers, self.shard_layer_end + 1),
            )
        )
        for peer in self.peers:
            p_start = max(0, peer.shard_layer_start)
            p_end = min(self.total_layers - 1, peer.shard_layer_end)
            covered.update(range(p_start, p_end + 1))

        pct = round((len(covered) / float(self.total_layers)) * 100.0, 1)
        self.coverage_percent = min(100.0, max(0.0, pct))
        self.is_swarm_ready = self.coverage_percent >= 100.0
        return self.coverage_percent

    async def refresh_room_coverage(self) -> None:
        """Refresh `coverage_percent` and `is_swarm_ready` from Cloud Tracker and LAN mDNS."""
        if self.state != NodeState.RUNNING:
            return

        cloud_coverage: float | None = None
        if _is_tracker_enabled() and self.room_id is not None:
            try:
                rooms = await self.tracker_client.fetch_available_rooms()
                for room in rooms:
                    if room.room_id == self.room_id:
                        cloud_coverage = room.coverage_percent
                        break
                # Also refresh peer routing table for this room/model
                if self.model_id is not None:
                    refreshed_peers = await self.tracker_client.fetch_peers(
                        model_id=self.model_id,
                        room_id=self.room_id,
                    )
                    if refreshed_peers:
                        self.peers = [
                            p for p in refreshed_peers if p.peer_id != self.node_id
                        ]
            except Exception as exc:
                logger.debug("Background coverage poll skipped Cloud Tracker: %s", exc)

        # Merge any newly discovered LAN peers for this room
        try:
            lan_beacons = await lan_mdns_service.scan_lan_rooms(timeout_seconds=0.15)
            known_ids = {p.peer_id for p in self.peers}
            for beacon in lan_beacons:
                p_id = str(beacon.get("node_id") or "")
                if (
                    not p_id
                    or p_id == self.node_id
                    or p_id in known_ids
                    or str(beacon.get("room_id") or "public") != self.room_id
                ):
                    continue
                known_ids.add(p_id)
                raw_role = str(beacon.get("role") or PeerRole.PRIMARY.value).lower()
                self.peers.append(
                    PeerNodeInfo(
                        peer_id=p_id,
                        ip_address=str(beacon.get("ip_address") or "127.0.0.1"),
                        port=int(beacon.get("p2p_port") or 49152),
                        shard_layer_start=int(beacon.get("shard_layer_start") or 0),
                        shard_layer_end=int(
                            beacon.get("shard_layer_end") or (self.total_layers - 1)
                        ),
                        role=(
                            PeerRole.BACKUP
                            if raw_role == PeerRole.BACKUP.value
                            else PeerRole.PRIMARY
                        ),
                        available_vram_mb=0,
                        room_id=self.room_id or "public",
                    )
                )
        except Exception as exc:
            logger.debug("Background LAN scan error: %s", exc)

        local_pct = self.recompute_local_coverage()
        if cloud_coverage is not None:
            self.coverage_percent = round(min(100.0, max(0.0, cloud_coverage)), 1)
            self.is_swarm_ready = self.coverage_percent >= 100.0
        else:
            self.coverage_percent = local_pct
            self.is_swarm_ready = local_pct >= 100.0

    async def _poll_room_coverage_loop(self, interval_seconds: float = 5.0) -> None:
        """Background loop polling Cloud room coverage every 5 seconds."""
        try:
            while self.state == NodeState.RUNNING:
                await asyncio.sleep(interval_seconds)
                if self.state != NodeState.RUNNING:
                    break
                await self.refresh_room_coverage()
        except asyncio.CancelledError:
            pass

    def start_coverage_monitor(self, interval_seconds: float = 5.0) -> None:
        """Launch the 5-second background room coverage polling task."""
        if self._coverage_poll_task is not None and not self._coverage_poll_task.done():
            self._coverage_poll_task.cancel()
        self._coverage_poll_task = asyncio.create_task(
            self._poll_room_coverage_loop(interval_seconds=interval_seconds)
        )

    async def stop_coverage_monitor(self) -> None:
        """Cancel the background room coverage polling task cleanly."""
        if self._coverage_poll_task is not None:
            self._coverage_poll_task.cancel()
            try:
                await self._coverage_poll_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._coverage_poll_task = None

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
            role=self.role,
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
            coverage_percent=self.coverage_percent if is_running else 0.0,
            is_swarm_ready=self.is_swarm_ready if is_running else False,
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
    runtime.is_swarm_ready = False
    runtime.coverage_percent = 0.0
    await runtime.stop_coverage_monitor()
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

    # 2. Scan local LAN subnet via mDNS multicast and aggregate layer coverage/redundancy
    lan_beacons = await lan_mdns_service.scan_lan_rooms(timeout_seconds=0.25)
    lan_groups: dict[str, list[dict[str, Any]]] = {}
    for beacon in lan_beacons:
        r_id = str(beacon.get("room_id") or "public")
        lan_groups.setdefault(r_id, []).append(beacon)

    for r_id, beacons in lan_groups.items():
        model = str(beacons[0].get("model") or "meta-llama/Llama-3.1-8B-Instruct")
        total_ram = round(sum(float(b.get("ram_gb") or 0.0) for b in beacons), 2)
        is_private = any(bool(b.get("is_private", False)) for b in beacons)
        total_layers = max(1, int(beacons[0].get("total_layers") or 32))

        layer_replication: dict[int, int] = {idx: 0 for idx in range(total_layers)}
        for b in beacons:
            b_start = max(0, int(b.get("shard_layer_start") or 0))
            b_end = min(total_layers - 1, int(b.get("shard_layer_end") or (total_layers - 1)))
            for layer_idx in range(b_start, b_end + 1):
                layer_replication[layer_idx] = layer_replication.get(layer_idx, 0) + 1

        covered_layers = sum(1 for cnt in layer_replication.values() if cnt > 0)
        coverage_pct = round((covered_layers / float(total_layers)) * 100.0, 1)
        min_rep = min(layer_replication.values()) if covered_layers == total_layers else 1
        redundancy = max(1, min_rep)

        if r_id in merged:
            existing = merged[r_id]
            merged[r_id] = RoomLobbyItem(
                room_id=r_id,
                model=existing.model,
                nodes_online=existing.nodes_online + len(beacons),
                total_shared_ram_gb=round(existing.total_shared_ram_gb + total_ram, 2),
                is_private=existing.is_private or is_private,
                network_scope="hybrid",
                coverage_percent=max(existing.coverage_percent, coverage_pct),
                redundancy_factor=max(existing.redundancy_factor, redundancy),
            )
        else:
            merged[r_id] = RoomLobbyItem(
                room_id=r_id,
                model=model,
                nodes_online=len(beacons),
                total_shared_ram_gb=total_ram,
                is_private=is_private,
                network_scope="lan",
                coverage_percent=coverage_pct,
                redundancy_factor=redundancy,
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
        await stop_node()

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
        # 1. Inspect existing local model metadata (GGUF header / config.json) to compute
        #    `total_model_layers` and `max_layers_can_hold` from `ram_gb`.
        probe_path: Path | None = None
        try:
            _, candidate_gguf = runtime.model_downloader.resolve_safe_gguf_path(
                request.model_id
            )
            if candidate_gguf.is_file():
                probe_path = candidate_gguf
            else:
                candidate_dir = runtime.model_downloader.resolve_safe_model_dir(
                    request.model_id
                )
                if (candidate_dir / "config.json").is_file():
                    probe_path = candidate_dir
        except Exception:
            probe_path = None

        pre_plan = calculate_max_layers(
            model_id=request.model_id,
            ram_gb=effective_ram_gb,
            model_path=probe_path,
            requested_start=request.shard_layer_start,
        )
        total_model_layers: int = pre_plan.total_layers
        max_layers_can_hold: int = pre_plan.n_layers

        target_start: int = pre_plan.shard_layer_start
        target_end: int = pre_plan.shard_layer_end
        if request.exact_shard_range:
            max_allowed_end = min(
                total_model_layers - 1,
                request.shard_layer_start + max_layers_can_hold - 1,
            )
            target_start = request.shard_layer_start
            target_end = min(request.shard_layer_end, max_allowed_end)

        # 2. Bind P2P listening port before announcing so Tracker gets the active port
        bound_port = await runtime.start_p2p_listener(
            request.p2p_listen_port,
            network_mode=request.network_mode,
        )
        request.p2p_listen_port = bound_port
        if request.node_id in ("shardai-client-01", "local-node-01"):
            host_slug = (
                re.sub(r"[^A-Za-z0-9-]+", "-", socket.gethostname().split(".")[0]).strip("-")
                or "host"
            )
            request.node_id = f"{request.node_id}-{host_slug}-{bound_port}"[:64]

        # 3. Announce to Cloudflare Tracker with `total_model_layers` + `max_layers_can_hold`
        #    and receive Cloud-orchestrated `assigned_start_layer`, `assigned_end_layer`, `role`.
        assigned_role: PeerRole = PeerRole.PRIMARY
        discovered_peers: list[PeerNodeInfo] = []
        if _is_tracker_enabled():
            try:
                assignment: AnnounceAssignment = (
                    await runtime.tracker_client.register_node_with_assignment(
                        request,
                        total_model_layers=total_model_layers,
                        max_layers_can_hold=max_layers_can_hold,
                    )
                )
                discovered_peers = assignment.peers
                assigned_role = assignment.role
                if not request.exact_shard_range:
                    target_start = assignment.assigned_start_layer
                    target_end = min(
                        assignment.assigned_end_layer,
                        min(total_model_layers - 1, target_start + max_layers_can_hold - 1),
                    )
            except httpx.HTTPStatusError as http_exc:
                if http_exc.response.status_code == status.HTTP_403_FORBIDDEN:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Invalid room key for private room '{request.room_id}'.",
                    ) from http_exc
                logger.warning("Tracker registration HTTP error: %s", http_exc)
            except Exception as tracker_exc:
                logger.warning(
                    "Tracker unreachable; continuing in local/LAN P2P mode: %s", tracker_exc
                )

        # 4. Download/resolve ONLY the Cloud-assigned shard layers [target_start..target_end]
        local_model_path = await runtime.model_downloader.ensure_model_downloaded(
            request.model_id,
            shard_layer_start=target_start,
            shard_layer_end=target_end,
        )

        # 5. Recompute final layer bounds from actual downloaded config.json / GGUF metadata
        layer_plan = calculate_max_layers(
            model_id=request.model_id,
            ram_gb=effective_ram_gb,
            model_path=local_model_path,
            requested_start=target_start,
        )
        final_total_layers = layer_plan.total_layers
        if local_model_path.is_dir():
            cfg_total = read_hf_config_summary(local_model_path)[0]
            if cfg_total > 0:
                final_total_layers = cfg_total

        final_start = max(0, min(target_start, final_total_layers - 1))
        max_allowed_end = min(
            final_total_layers - 1,
            final_start + layer_plan.n_layers - 1,
        )
        final_end = max(final_start, min(target_end, max_allowed_end))
        final_n_layers = final_end - final_start + 1
        final_est_ram = min(
            request.max_ram_mb,
            round((0.6 + (final_n_layers * (layer_plan.per_layer_mb / 1024.0))) * 1024.0),
        )

        request.total_layers = final_total_layers
        request.shard_layer_start = final_start
        request.shard_layer_end = final_end

        # 6. Allocate local RAM/VRAM & load ONLY the assigned layers into llama_cpp / PyTorch
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

        room_key_hash: str | None = hash_room_key(request.room_key)
        is_private_room: bool = room_key_hash is not None

        # 7. Advertise on LAN mDNS & merge LAN-discovered peers (preserving primary/backup roles)
        await lan_mdns_service.start_advertising(
            LanRoomBeacon(
                node_id=request.node_id,
                room_id=request.room_id,
                model=request.model_id,
                ram_gb=float(effective_ram_gb),
                is_private=is_private_room,
                is_unlisted=request.is_unlisted,
                p2p_port=bound_port,
                shard_layer_start=final_start,
                shard_layer_end=final_end,
                total_layers=final_total_layers,
                role=assigned_role.value,
            )
        )

        known_peer_ids: set[str] = {peer.peer_id for peer in discovered_peers}
        lan_peers_raw = await lan_mdns_service.scan_lan_rooms(timeout_seconds=0.2)
        for beacon in lan_peers_raw:
            peer_id = str(beacon.get("node_id") or "")
            if (
                not peer_id
                or peer_id == request.node_id
                or peer_id in known_peer_ids
                or str(beacon.get("room_id") or "public") != request.room_id
            ):
                continue
            known_peer_ids.add(peer_id)
            raw_peer_role = str(beacon.get("role") or PeerRole.PRIMARY.value).lower()
            peer_role = (
                PeerRole.BACKUP if raw_peer_role == PeerRole.BACKUP.value else PeerRole.PRIMARY
            )
            discovered_peers.append(
                PeerNodeInfo(
                    peer_id=peer_id,
                    ip_address=str(beacon.get("ip_address") or "127.0.0.1"),
                    port=int(beacon.get("p2p_port") or 49152),
                    shard_layer_start=int(beacon.get("shard_layer_start") or 0),
                    shard_layer_end=int(beacon.get("shard_layer_end") or (final_total_layers - 1)),
                    role=peer_role,
                    available_vram_mb=0,
                    room_id=request.room_id,
                )
            )

        # 8. Issue cryptographic Session Token stored strictly in process RAM
        runtime.session_token = f"sk-shard-{secrets.token_urlsafe(24)}"
        runtime.node_id = request.node_id
        runtime.model_id = request.model_id
        runtime.model_path = str(local_model_path)
        runtime.network_mode = request.network_mode
        runtime.room_id = request.room_id
        runtime.is_private = is_private_room
        runtime.is_unlisted = request.is_unlisted
        runtime.role = assigned_role
        runtime.backend = effective_backend
        runtime.p2p_listen_port = request.p2p_listen_port
        runtime.n_layers = final_n_layers
        runtime.shard_layer_start = final_start
        runtime.shard_layer_end = final_end
        runtime.total_layers = final_total_layers
        runtime.peers = discovered_peers
        runtime.state = NodeState.RUNNING

        # Compute initial coverage_percent & is_swarm_ready, then monitor every 5s
        await runtime.refresh_room_coverage()
        runtime.start_coverage_monitor(interval_seconds=5.0)

        role_display = (
            "Primary Seeder" if assigned_role == PeerRole.PRIMARY else "Backup Node"
        )
        return runtime.to_status_response(
            message=(
                f"Gateway Active ({request.network_mode.value.upper()} / Room: '{request.room_id}') — "
                f"Role: {role_display} (Layers {final_start}..{final_end}/{final_total_layers}) "
                f"[{effective_backend.value}]."
            )
        )
    except HTTPException:
        runtime.session_token = None
        runtime.is_swarm_ready = False
        runtime.coverage_percent = 0.0
        await runtime.stop_coverage_monitor()
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
        runtime.is_swarm_ready = False
        runtime.coverage_percent = 0.0
        await runtime.stop_coverage_monitor()
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
        runtime.is_swarm_ready = False
        runtime.coverage_percent = 0.0
        await runtime.stop_coverage_monitor()
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
        runtime.role = PeerRole.PRIMARY
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
    runtime.recompute_local_coverage()
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
) -> StreamingResponse | JSONResponse | dict[str, Any]:
    """OpenAI-compatible Chat Completions endpoint routed through P2P Pipeline Inference.

    Rejects inference with standard OpenAI 503 JSON error while `is_swarm_ready == False`
    (room RAM/layer coverage < 100%).
    """
    if not runtime.is_swarm_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "message": (
                        f"Swarm is not ready (current layer coverage: {runtime.coverage_percent:.0f}%). "
                        f"Room '{runtime.room_id or 'public'}' needs additional RAM/nodes to cover all "
                        f"{runtime.total_layers} layers of '{runtime.model_id}'. Invite more nodes to join."
                    ),
                    "type": "service_unavailable_error",
                    "param": None,
                    "code": "swarm_insufficient_ram",
                }
            },
        )
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
