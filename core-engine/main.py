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
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import socket
import time
from typing import Any, Literal
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import typer
import uvicorn

from models.downloader import (
    DEFAULT_CATALOG_MODELS,
    GGUF_MAGIC,
    MIN_ONLINE_GGUF_BYTES,
    SUPPORTED_MODELS,
    DownloadProgressEvent,
    ModelDownloader,
    compute_model_fingerprint,
    is_offline_mode,
    is_valid_gguf_file,
    parse_estimated_ram_gb,
    read_hf_config_summary,
    sanitize_catalog_entries,
)
from models.shard_runner import ShardModelRunner
from p2p.discovery import ZEROCONF_AVAILABLE, is_zeroconf_enabled, zeroconf_discovery
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
    CatalogModelItem,
    ChatMessage,
    CheckRoomJoinRequest,
    CheckRoomJoinResponse,
    ComputeBackend,
    GenerateRequest,
    ModelCatalogResponse,
    ModelSourceType,
    NetworkMode,
    NetworkStatus,
    NetworkStatusResponse,
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
    TrackerUnreachableError,
    extract_tracker_error_detail,
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


async def _scan_all_lan_beacons(
    udp_timeout_seconds: float = 0.2,
    zeroconf_wait_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge LAN peers from the legacy UDP multicast probe and zeroconf DNS-SD browser.

    Deduplicated by `node_id`; zeroconf entries win because they carry the routable interface IP.
    """
    udp_task = asyncio.create_task(
        lan_mdns_service.scan_lan_rooms(timeout_seconds=udp_timeout_seconds)
    )
    zc_beacons: list[dict[str, Any]] = []
    try:
        zc_beacons = await zeroconf_discovery.discover_peers(wait_seconds=zeroconf_wait_seconds)
    except Exception as exc:
        logger.debug("Zeroconf discovery error: %s", exc)
    udp_beacons: list[dict[str, Any]] = []
    try:
        udp_beacons = await udp_task
    except Exception as exc:
        logger.debug("UDP LAN scan error: %s", exc)

    merged: dict[str, dict[str, Any]] = {}
    for beacon in [*udp_beacons, *zc_beacons]:
        node_key = str(beacon.get("node_id") or "")
        if node_key:
            merged[node_key] = beacon
    return list(merged.values())


def _beacon_to_peer(beacon: dict[str, Any], room_id: str, total_layers: int) -> PeerNodeInfo:
    """Convert a LAN beacon dict (UDP or zeroconf) into a routable `PeerNodeInfo`."""
    raw_role = str(beacon.get("role") or PeerRole.PRIMARY.value).lower()
    return PeerNodeInfo(
        peer_id=str(beacon.get("node_id")),
        ip_address=str(beacon.get("ip_address") or "127.0.0.1"),
        port=int(beacon.get("p2p_port") or 49152),
        shard_layer_start=int(beacon.get("shard_layer_start") or 0),
        shard_layer_end=int(beacon.get("shard_layer_end") or (total_layers - 1)),
        role=PeerRole.BACKUP if raw_role == PeerRole.BACKUP.value else PeerRole.PRIMARY,
        available_vram_mb=0,
        room_id=room_id,
    )


def _lan_peers_for_room(
    beacons: list[dict[str, Any]],
    room_id: str,
    self_node_id: str | None,
    total_layers: int,
) -> list[PeerNodeInfo]:
    """Filter LAN beacons to peers in `room_id` (excluding this node)."""
    peers: list[PeerNodeInfo] = []
    seen: set[str] = set()
    for beacon in beacons:
        peer_id = str(beacon.get("node_id") or "")
        if (
            not peer_id
            or peer_id == self_node_id
            or peer_id in seen
            or str(beacon.get("room_id") or "public") != room_id
        ):
            continue
        seen.add(peer_id)
        peers.append(_beacon_to_peer(beacon, room_id, total_layers))
    return peers


def _first_uncovered_layer(peers: list[PeerNodeInfo], total_layers: int) -> int | None:
    """Return the first model layer not hosted by any LAN peer (None if fully covered)."""
    covered: set[int] = set()
    for peer in peers:
        covered.update(
            range(max(0, peer.shard_layer_start), min(total_layers, peer.shard_layer_end + 1))
        )
    for layer_idx in range(total_layers):
        if layer_idx not in covered:
            return layer_idx
    return None


def _lan_discovery_wait_seconds() -> float:
    """How long to listen for zeroconf answers before deciding LAN vs Solo on startup."""
    try:
        return max(0.0, float(os.getenv("SHARDAI_LAN_DISCOVERY_WAIT_SECONDS", "1.5")))
    except ValueError:
        return 1.5


class LocalNodeRuntime:
    """Encapsulates runtime state, session auth, P2P listener, and model shard pipeline."""

    def __init__(self) -> None:
        self.state: NodeState = NodeState.STOPPED
        self.node_id: str | None = None
        self.model_id: str | None = None
        self.model_path: str | None = None
        self.network_mode: NetworkMode | None = None
        self.room_id: str | None = None
        self.room_key_hash: str | None = None
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
        # Offline / Solo Mode state (auto-fallback chain: Cloud -> LAN -> Solo)
        self.network_status: NetworkStatus | None = None
        self.last_tracker_reachable: bool | None = None
        self.use_mmap: bool = False
        self.ssd_offload_active: bool = False
        self.source_type: ModelSourceType | None = None
        self.model_fingerprint: str | None = None
        self._coverage_poll_task: asyncio.Task[None] | None = None
        self.memory_manager: ShardMemoryManager = ShardMemoryManager()
        self.model_downloader: ModelDownloader = ModelDownloader()
        self.model_runner: ShardModelRunner | None = None
        self.p2p_server: asyncio.AbstractServer | None = None
        self.pending_token_queues: dict[str, asyncio.Queue[DecodedTokenFrame]] = {}
        self._compute_lock: asyncio.Lock = asyncio.Lock()
        self.tracker_client: SignalingTrackerClient = SignalingTrackerClient(
            base_url=os.getenv(
                "SHARDAI_TRACKER_URL", "https://network.shardai.online"
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

        is_online: bool = self.network_status == NetworkStatus.ONLINE
        cloud_coverage: float | None = None
        if is_online and _is_tracker_enabled() and self.room_id is not None:
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
                        room_key_hash=self.room_key_hash,
                    )
                    if refreshed_peers:
                        self.peers = [
                            p for p in refreshed_peers if p.peer_id != self.node_id
                        ]
                self.last_tracker_reachable = True
            except Exception as exc:
                logger.debug("Background coverage poll skipped Cloud Tracker: %s", exc)

        # LAN peers (UDP beacon + zeroconf). Offline nodes rebuild the routing table from the
        # LAN on every tick so peers that left the Wi-Fi are dropped automatically.
        try:
            lan_beacons = await _scan_all_lan_beacons(udp_timeout_seconds=0.15)
            lan_peers = _lan_peers_for_room(
                lan_beacons,
                room_id=self.room_id or "public",
                self_node_id=self.node_id,
                total_layers=self.total_layers,
            )
            if is_online:
                known_ids = {p.peer_id for p in self.peers}
                self.peers.extend(p for p in lan_peers if p.peer_id not in known_ids)
            else:
                self.peers = lan_peers
                new_status = NetworkStatus.LAN_ONLY if lan_peers else NetworkStatus.OFFLINE_SOLO
                if new_status != self.network_status:
                    logger.info(
                        "Network status changed: %s -> %s (%d LAN peers)",
                        self.network_status,
                        new_status,
                        len(lan_peers),
                    )
                    self.network_status = new_status
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
            network_status=self.network_status if is_running else None,
            is_solo_mode=is_running and self.network_status == NetworkStatus.OFFLINE_SOLO,
            use_mmap=self.use_mmap if is_running else False,
            ssd_offload_active=self.ssd_offload_active if is_running else False,
            source_type=self.source_type if is_running else None,
            model_fingerprint=self.model_fingerprint if is_running else None,
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
    await zeroconf_discovery.close()
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
    return os.getenv("SHARDAI_ENABLE_TRACKER", "true").lower() == "true" and not is_offline_mode()


@app.get("/network-status", response_model=NetworkStatusResponse)
async def get_network_status() -> NetworkStatusResponse:
    """Probe connectivity for the UI indicator: Online (Cloud) / LAN Only / Offline (Solo).

    While the node is running the authoritative `runtime.network_status` decided at start-up
    (and refreshed by the 5s monitor) is returned; when stopped, a fresh probe is performed.
    """
    lan_beacons = await _scan_all_lan_beacons(udp_timeout_seconds=0.15)
    lan_count = len(
        [b for b in lan_beacons if str(b.get("node_id") or "") != (runtime.node_id or "")]
    )

    if runtime.state == NodeState.RUNNING and runtime.network_status is not None:
        tracker_ok = runtime.network_status == NetworkStatus.ONLINE
        effective = runtime.network_status
    else:
        tracker_ok = _is_tracker_enabled() and await runtime.tracker_client.ping(
            timeout_seconds=3.0
        )
        runtime.last_tracker_reachable = tracker_ok
        if tracker_ok:
            effective = NetworkStatus.ONLINE
        elif lan_count > 0:
            effective = NetworkStatus.LAN_ONLY
        else:
            effective = NetworkStatus.OFFLINE_SOLO

    return NetworkStatusResponse(
        network_status=effective,
        tracker_reachable=tracker_ok,
        tracker_url=runtime.tracker_client.base_url,
        lan_peers_count=lan_count,
        zeroconf_available=ZEROCONF_AVAILABLE and is_zeroconf_enabled(),
    )


@app.get("/available-rooms", response_model=list[RoomLobbyItem])
async def get_available_rooms() -> list[RoomLobbyItem]:
    """Fetch active rooms from Cloudflare Tracker (GET /rooms) merged with LAN mDNS scan."""
    merged: dict[str, RoomLobbyItem] = {}

    # 1. Query Cloudflare Signaling Tracker when enabled (skipped while running offline)
    node_is_offline = (
        runtime.state == NodeState.RUNNING
        and runtime.network_status not in (None, NetworkStatus.ONLINE)
    )
    if _is_tracker_enabled() and not node_is_offline:
        try:
            cloud_rooms = await runtime.tracker_client.fetch_available_rooms()
            for item in cloud_rooms:
                merged[item.room_id] = item
            runtime.last_tracker_reachable = True
        except Exception as exc:
            runtime.last_tracker_reachable = False
            logger.warning("Failed to fetch rooms from Cloudflare Tracker: %s", exc)

    # 2. Scan local LAN subnet via UDP beacon + zeroconf and aggregate layer coverage/redundancy
    lan_beacons = await _scan_all_lan_beacons(udp_timeout_seconds=0.25)
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


async def _verify_cloud_room_join_permission(
    room_id: str,
    model_id: str,
    network_mode: NetworkMode = NetworkMode.INTERNET,
    room_key: str | None = None,
) -> tuple[bool, str]:
    """Verify with the Cloud Signaling Tracker whether this node may join `room_id`.

    Raises `HTTPException` (403 or 409) immediately on deterministic tracker rejection so
    no model download or offline fallback occurs. Genuine network/5xx errors allow
    continuation into the existing LAN / Offline Solo fallback chain.
    """
    if not _is_tracker_enabled() or network_mode != NetworkMode.INTERNET:
        return True, "Cloud pre-flight skipped for LAN/offline mode."

    try:
        await runtime.tracker_client.check_room_join(
            model_id=model_id,
            room_id=room_id,
            room_key=room_key,
        )
        runtime.last_tracker_reachable = True
        return True, f"Room '{room_id}' is available to join."
    except httpx.HTTPStatusError as http_exc:
        status_code = http_exc.response.status_code
        if status_code == status.HTTP_403_FORBIDDEN:
            detail = extract_tracker_error_detail(
                http_exc.response,
                f"Invalid room key for private room '{room_id}'.",
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=detail,
            ) from http_exc
        if status_code == status.HTTP_409_CONFLICT:
            detail = extract_tracker_error_detail(
                http_exc.response,
                f"Room '{room_id}' is full or currently busy and cannot be joined.",
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail,
            ) from http_exc
        logger.warning(
            "Tracker room check returned HTTP %d; allowing offline/LAN fallback.",
            status_code,
        )
        return True, "Cloud tracker returned a transient error; offline/LAN fallback enabled."
    except (TrackerUnreachableError, httpx.HTTPError, OSError) as tracker_exc:
        runtime.last_tracker_reachable = False
        logger.warning(
            "Cloudflare Tracker unreachable during room pre-flight (%s); allowing offline/LAN fallback.",
            tracker_exc,
        )
        return True, "Cloud tracker unreachable; offline/LAN fallback enabled."


@app.post(
    "/rooms/check-join",
    response_model=CheckRoomJoinResponse,
    status_code=status.HTTP_200_OK,
)
async def check_room_join(request: CheckRoomJoinRequest) -> CheckRoomJoinResponse:
    """Pre-flight admission check before downloading model weights or joining a room."""
    allowed, msg = await _verify_cloud_room_join_permission(
        room_id=request.room_id,
        model_id=request.model_id,
        network_mode=request.network_mode,
        room_key=request.room_key,
    )
    return CheckRoomJoinResponse(
        allowed=allowed,
        room_id=request.room_id,
        message=msg,
    )


@app.get("/models/catalog", response_model=ModelCatalogResponse)
async def get_model_catalog() -> ModelCatalogResponse:
    """Sync enabled models from Cloud `GET /config/models` and cache to `~/.shardai/models_cache.json`."""
    raw_models: list[dict[str, Any]] = []
    source: Literal["cloud", "cache", "default"] = "default"

    if _is_tracker_enabled():
        try:
            cloud_models = await runtime.tracker_client.fetch_model_catalog()
            sanitized_cloud = sanitize_catalog_entries(cloud_models)
            if sanitized_cloud:
                runtime.model_downloader.save_catalog_cache(sanitized_cloud)
                raw_models = sanitized_cloud
                source = "cloud"
                runtime.last_tracker_reachable = True
        except Exception as exc:
            logger.warning("Failed to fetch /config/models from Cloud (%s); using local cache.", exc)

    if not raw_models:
        cached = runtime.model_downloader.load_catalog_cache()
        if cached:
            raw_models = cached
            source = "cache"
        else:
            raw_models = sanitize_catalog_entries(DEFAULT_CATALOG_MODELS)
            source = "default"

    def _build_catalog_items(entries: list[dict[str, Any]]) -> list[CatalogModelItem]:
        built: list[CatalogModelItem] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("enabled", True) is not True:
                continue
            raw_repo_id = entry.get("repo_id")
            if not isinstance(raw_repo_id, str) or not raw_repo_id.strip():
                continue
            repo_id = raw_repo_id.strip()
            raw_id = entry.get("id")
            entry_id = raw_id.strip() if isinstance(raw_id, str) and raw_id.strip() else repo_id
            raw_name = entry.get("name")
            entry_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else repo_id
            raw_filename = entry.get("filename")
            entry_filename = (
                raw_filename.strip()
                if isinstance(raw_filename, str) and raw_filename.strip()
                else None
            )
            try:
                built.append(
                    CatalogModelItem(
                        id=entry_id,
                        name=entry_name,
                        repo_id=repo_id,
                        filename=entry_filename,
                        estimated_ram_gb=parse_estimated_ram_gb(
                            entry.get("estimated_ram_gb"),
                            default=5.0,
                        ),
                        enabled=True,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed catalog entry '%s': %s", repo_id, exc)
        return built

    items = _build_catalog_items(raw_models)
    if not items:
        items = _build_catalog_items(sanitize_catalog_entries(DEFAULT_CATALOG_MODELS))
        source = "default"

    return ModelCatalogResponse(source=source, count=len(items), models=items)


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
    source_type: ModelSourceType = ModelSourceType.CATALOG,
    filename: str | None = None,
    room_id: str | None = None,
    room_key: str | None = None,
    network_mode: NetworkMode = NetworkMode.INTERNET,
) -> StreamingResponse:
    """Ensure any requested catalog, HF repo, direct URL, or local `.gguf` model is ready."""
    cleaned_room_id = room_id.strip() if room_id is not None else ""
    if cleaned_room_id:
        await _verify_cloud_room_join_permission(
            room_id=cleaned_room_id,
            model_id=model_id.strip(),
            network_mode=network_mode,
            room_key=room_key,
        )

    try:
        cleaned_id = model_id.strip()
        if not cleaned_id:
            raise ValueError("model_id cannot be empty.")
        if source_type == ModelSourceType.DIRECT_URL or (
            source_type in (ModelSourceType.CATALOG, ModelSourceType.HUGGINGFACE)
            and (cleaned_id.lower().startswith("http://") or cleaned_id.lower().startswith("https://"))
            and ("huggingface.co/" not in cleaned_id.lower() or "/resolve/" in cleaned_id.lower() or cleaned_id.lower().endswith(".gguf"))
        ):
            runtime.model_downloader.resolve_direct_url_path(cleaned_id, filename=filename)
        elif source_type == ModelSourceType.LOCAL_FILE:
            local_path = Path(cleaned_id).expanduser().resolve()
            if not local_path.is_file() and not (local_path.is_dir() and (local_path / "config.json").is_file()):
                raise FileNotFoundError(f"Local .gguf model file not found: {local_path}")
        elif not is_offline_mode() and Path(cleaned_id).suffix.lower() != ".gguf":
            runtime.model_downloader.resolve_safe_model_dir(cleaned_id)
        else:
            runtime.model_downloader.resolve_safe_gguf_path(cleaned_id)
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
                source_type=source_type.value,
                filename=filename,
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
        # 1. Inspect existing local model metadata (GGUF header / config.json) & compute
        #    initial `model_fingerprint` (SHA-256 of first 10MB + total size) when cached/local.
        probe_path: Path | None = None
        initial_fingerprint: str | None = None
        if request.source_type == ModelSourceType.LOCAL_FILE:
            local_candidate = Path(request.model_id.strip()).expanduser().resolve()
            if local_candidate.is_dir() and (local_candidate / "config.json").is_file():
                probe_path = local_candidate
            elif local_candidate.is_file():
                if local_candidate.suffix.lower() != ".gguf":
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Expected a .gguf model file, got: {local_candidate.name}",
                    )
                probe_path = local_candidate
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Local .gguf model file not found: {local_candidate}",
                )
        else:
            try:
                min_gguf_bytes = (
                    len(GGUF_MAGIC) if is_offline_mode() else MIN_ONLINE_GGUF_BYTES
                )
                if request.source_type == ModelSourceType.DIRECT_URL:
                    direct_target = runtime.model_downloader.resolve_direct_url_path(
                        request.model_id, filename=request.filename
                    )
                    if is_valid_gguf_file(direct_target, min_bytes=min_gguf_bytes):
                        probe_path = direct_target
                else:
                    in_models = runtime.model_downloader.resolve_gguf_in_models_dir(
                        request.model_id, request.filename
                    )
                    if is_valid_gguf_file(in_models, min_bytes=min_gguf_bytes):
                        probe_path = in_models
                    else:
                        _, candidate_gguf = runtime.model_downloader.resolve_safe_gguf_path(
                            request.model_id
                        )
                        if is_valid_gguf_file(candidate_gguf, min_bytes=min_gguf_bytes):
                            probe_path = candidate_gguf
                        else:
                            _, candidate_dir = runtime.model_downloader.resolve_safe_model_dir(
                                request.model_id
                            )
                            if (candidate_dir / "config.json").is_file():
                                probe_path = candidate_dir
            except Exception:
                probe_path = None

        if probe_path is not None and probe_path.exists():
            try:
                initial_fingerprint = compute_model_fingerprint(probe_path)
            except Exception:
                initial_fingerprint = None

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
        #    AUTO-FALLBACK: timeout / DNS / no route -> OFFLINE_MODE (LAN zeroconf or Solo).
        assigned_role: PeerRole = PeerRole.PRIMARY
        discovered_peers: list[PeerNodeInfo] = []
        cloud_online: bool = False
        if _is_tracker_enabled() and request.network_mode == NetworkMode.INTERNET:
            try:
                assignment: AnnounceAssignment = (
                    await runtime.tracker_client.register_node_with_assignment(
                        request,
                        total_model_layers=total_model_layers,
                        max_layers_can_hold=max_layers_can_hold,
                        model_fingerprint=initial_fingerprint,
                    )
                )
                cloud_online = True
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
                    detail = extract_tracker_error_detail(
                        http_exc.response,
                        f"Invalid room key for private room '{request.room_id}'.",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=detail,
                    ) from http_exc
                if http_exc.response.status_code == status.HTTP_409_CONFLICT:
                    detail = extract_tracker_error_detail(
                        http_exc.response,
                        f"Room '{request.room_id}' is full or currently busy and cannot be joined.",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=detail,
                    ) from http_exc
                logger.warning(
                    "Tracker rejected announce (HTTP %d); falling back to OFFLINE_MODE.",
                    http_exc.response.status_code,
                )
            except (TrackerUnreachableError, httpx.HTTPError, OSError) as tracker_exc:
                logger.warning(
                    "Cloudflare Tracker unreachable (%s); switching to OFFLINE_MODE "
                    "(LAN zeroconf discovery -> Solo fallback).",
                    tracker_exc,
                )
            except Exception as tracker_exc:  # malformed response etc. — never block the node
                logger.warning(
                    "Unexpected Tracker error (%s); falling back to OFFLINE_MODE.", tracker_exc
                )
        runtime.last_tracker_reachable = cloud_online

        # 3b. OFFLINE_MODE: discover LAN peers via zeroconf (+ UDP beacon) before choosing
        #     which layers to load. No peers -> SOLO MODE (this node hosts 100% of layers).
        network_status: NetworkStatus = NetworkStatus.ONLINE
        use_mmap: bool = False
        ssd_offload_active: bool = False
        if not cloud_online:
            if request.network_mode != NetworkMode.LAN:
                # Re-bind P2P listener on 0.0.0.0 so LAN peers can reach this node.
                bound_port = await runtime.start_p2p_listener(
                    bound_port, network_mode=NetworkMode.LAN
                )
                request.p2p_listen_port = bound_port

            lan_beacons = await _scan_all_lan_beacons(
                udp_timeout_seconds=0.25,
                zeroconf_wait_seconds=_lan_discovery_wait_seconds(),
            )
            discovered_peers = _lan_peers_for_room(
                lan_beacons,
                room_id=request.room_id,
                self_node_id=request.node_id,
                total_layers=total_model_layers,
            )

            if discovered_peers:
                network_status = NetworkStatus.LAN_ONLY
                if not request.exact_shard_range:
                    # Fill the first layer gap left by LAN peers; if fully covered, be a Backup.
                    gap_start = _first_uncovered_layer(discovered_peers, total_model_layers)
                    if gap_start is None:
                        assigned_role = PeerRole.BACKUP
                        gap_start = 0
                    target_start = gap_start
                    target_end = min(
                        total_model_layers - 1, target_start + max_layers_can_hold - 1
                    )
                logger.info(
                    "LAN Room '%s': %d peer(s) found; hosting layers %d..%d as %s.",
                    request.room_id,
                    len(discovered_peers),
                    target_start,
                    target_end,
                    assigned_role.value,
                )
            else:
                network_status = NetworkStatus.OFFLINE_SOLO
                fits_in_ram = max_layers_can_hold >= total_model_layers
                if not request.exact_shard_range:
                    if fits_in_ram or request.allow_ssd_offload:
                        target_start = 0
                        target_end = total_model_layers - 1
                    else:
                        target_start = 0
                        target_end = max_layers_can_hold - 1
                if not fits_in_ram and request.allow_ssd_offload:
                    # RAM < model size: memory-map the GGUF so llama_cpp pages the missing
                    # weights from SSD on demand instead of crashing with OOM.
                    use_mmap = True
                    ssd_offload_active = True
                logger.info(
                    "SOLO MODE: no Cloud, no LAN peers. Layers %d..%d/%d (ram_gb=%d, "
                    "fits_in_ram=%s, use_mmap=%s).",
                    target_start,
                    target_end,
                    total_model_layers,
                    effective_ram_gb,
                    fits_in_ram,
                    use_mmap,
                )

        # 4. Download/resolve ONLY the Cloud-assigned shard layers [target_start..target_end]
        local_model_path = await runtime.model_downloader.ensure_model_downloaded(
            request.model_id,
            shard_layer_start=target_start,
            shard_layer_end=target_end,
            source_type=request.source_type.value,
            filename=request.filename,
        )
        model_fingerprint: str = compute_model_fingerprint(local_model_path)
        if cloud_online and model_fingerprint != initial_fingerprint:
            try:
                await runtime.tracker_client.register_node_with_assignment(
                    request,
                    total_model_layers=total_model_layers,
                    max_layers_can_hold=max_layers_can_hold,
                    model_fingerprint=model_fingerprint,
                )
            except Exception as fp_exc:
                logger.debug("Optional model_fingerprint sync skipped: %s", fp_exc)

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
        ram_capped_end = min(
            final_total_layers - 1,
            final_start + layer_plan.n_layers - 1,
        )
        # Re-check with the REAL on-disk model size: Solo + SSD offload consent -> mmap all layers.
        if (
            network_status == NetworkStatus.OFFLINE_SOLO
            and request.allow_ssd_offload
            and ram_capped_end < final_total_layers - 1
        ):
            use_mmap = True
            ssd_offload_active = True
            if not request.exact_shard_range:
                target_end = final_total_layers - 1
        max_allowed_end = final_total_layers - 1 if ssd_offload_active else ram_capped_end
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
            use_mmap=use_mmap,
        )
        effective_backend = (
            ComputeBackend.PYTORCH
            if runtime.model_runner.is_real_model
            else request.backend
        )

        room_key_hash: str | None = hash_room_key(request.room_key)
        is_private_room: bool = room_key_hash is not None

        # 7. Advertise on LAN (UDP beacon always; zeroconf DNS-SD broadcast in OFFLINE_MODE or
        #    explicit LAN mode) & merge LAN-discovered peers (preserving primary/backup roles).
        lan_beacon = LanRoomBeacon(
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
        await lan_mdns_service.start_advertising(lan_beacon)
        if network_status != NetworkStatus.ONLINE or request.network_mode == NetworkMode.LAN:
            await zeroconf_discovery.start_advertising(lan_beacon)

        if network_status == NetworkStatus.ONLINE:
            known_peer_ids: set[str] = {peer.peer_id for peer in discovered_peers}
            lan_peers = _lan_peers_for_room(
                await _scan_all_lan_beacons(udp_timeout_seconds=0.2),
                room_id=request.room_id,
                self_node_id=request.node_id,
                total_layers=final_total_layers,
            )
            discovered_peers.extend(p for p in lan_peers if p.peer_id not in known_peer_ids)

        # 8. Issue cryptographic Session Token stored strictly in process RAM
        #    (issued in every mode — Online, LAN Only and Offline Solo).
        runtime.session_token = f"sk-shard-{secrets.token_urlsafe(24)}"
        runtime.network_status = network_status
        runtime.use_mmap = use_mmap
        runtime.ssd_offload_active = ssd_offload_active
        runtime.source_type = request.source_type
        runtime.model_fingerprint = model_fingerprint
        runtime.node_id = request.node_id
        runtime.model_id = request.model_id
        runtime.model_path = str(local_model_path)
        runtime.network_mode = request.network_mode
        runtime.room_id = request.room_id
        runtime.room_key_hash = room_key_hash
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
        status_label = {
            NetworkStatus.ONLINE: "Online (Cloud)",
            NetworkStatus.LAN_ONLY: "LAN Only",
            NetworkStatus.OFFLINE_SOLO: "Offline (Solo)",
        }[network_status]
        offload_note = " + SSD Offload (use_mmap)" if ssd_offload_active else ""
        return runtime.to_status_response(
            message=(
                f"Gateway Active [{status_label}] (Room: '{request.room_id}') — "
                f"Role: {role_display} (Layers {final_start}..{final_end}/{final_total_layers}) "
                f"[{effective_backend.value}{offload_note}]."
            )
        )
    except HTTPException as http_err:
        runtime.session_token = None
        runtime.is_swarm_ready = False
        runtime.coverage_percent = 0.0
        await runtime.stop_coverage_monitor()
        await lan_mdns_service.stop_advertising()
        await zeroconf_discovery.stop_advertising()
        await runtime.stop_p2p_listener()
        if runtime.model_runner is not None:
            runtime.model_runner.release()
            runtime.model_runner = None
        runtime.memory_manager.release_all()
        runtime.state = (
            NodeState.STOPPED
            if http_err.status_code in (status.HTTP_403_FORBIDDEN, status.HTTP_409_CONFLICT)
            else NodeState.ERROR
        )
        raise
    except Exception as exc:
        runtime.session_token = None
        runtime.is_swarm_ready = False
        runtime.coverage_percent = 0.0
        await runtime.stop_coverage_monitor()
        await lan_mdns_service.stop_advertising()
        await zeroconf_discovery.stop_advertising()
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
        await zeroconf_discovery.stop_advertising()
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
        runtime.room_key_hash = None
        runtime.network_status = None
        runtime.use_mmap = False
        runtime.ssd_offload_active = False
        runtime.source_type = None
        runtime.model_fingerprint = None
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
                        + (
                            " Running Offline (Solo): restart with 'allow_ssd_offload' enabled to "
                            "page the remaining layers from SSD (use_mmap)."
                            if runtime.network_status == NetworkStatus.OFFLINE_SOLO
                            else ""
                        )
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


cli_app = typer.Typer(
    name="core-engine",
    help="ShardAi Core Engine — P2P AI Gateway (Daemon sidecar for Tauri UI or standalone CLI).",
    add_completion=False,
)


async def _serve_cli_mode(
    host: str,
    port: int,
    request: StartNodeRequest,
) -> None:
    """Boot Uvicorn and immediately join the configured ShardAi room in headless CLI mode."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    server = uvicorn.Server(config)

    async def _bootstrap_cli_node() -> None:
        while not server.started:
            await asyncio.sleep(0.05)
        try:
            status_resp = await start_node(request)
            readiness = "API READY" if status_resp.is_swarm_ready else f"STANDBY ({status_resp.coverage_percent:.0f}%)"
            typer.echo("\n" + "=" * 68)
            typer.echo(f" ShardAi Headless CLI Gateway — {readiness}")
            typer.echo(f" Model        : {status_resp.model_id}")
            typer.echo(f" Room         : {status_resp.room_id} ({status_resp.network_status or request.network_mode.value})")
            typer.echo(f" Role / Layers: {status_resp.role.value.upper()} ({status_resp.shard_layer_start}..{status_resp.shard_layer_end}/{status_resp.total_layers})")
            typer.echo(f" OpenAI URL   : {status_resp.api_base_url}")
            typer.echo(f" API Key      : {status_resp.session_token}")
            typer.echo(f" Use mmap SSD : {status_resp.ssd_offload_active}")
            typer.echo("=" * 68 + "\n")
        except Exception as exc:
            typer.echo(f"[ShardAi CLI] Failed to join room: {exc}", err=True)
            server.should_exit = True

    bootstrap_task = asyncio.create_task(_bootstrap_cli_node())
    try:
        await server.serve()
    finally:
        if not bootstrap_task.done():
            bootstrap_task.cancel()


@cli_app.callback(invoke_without_command=True)
def run_entrypoint(
    mode: str = typer.Option(
        os.getenv("SHARDAI_RUN_MODE", "daemon"),
        "--mode",
        "-m",
        help="Execution mode: 'daemon' (Tauri UI sidecar) or 'cli' (headless node).",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="HTTP API bind host (loopback 127.0.0.1 recommended for security).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", "8000")),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local HTTP OpenAI Gateway port.",
    ),
    ram: int = typer.Option(
        4,
        "--ram",
        min=2,
        max=256,
        help="[cli mode] RAM contribution in GB.",
    ),
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--model",
        help="[cli mode] Model Repo ID, Direct .gguf URL, or Local .gguf path.",
    ),
    source_type: ModelSourceType = typer.Option(
        ModelSourceType.CATALOG,
        "--source-type",
        help="[cli mode] Model source: catalog, huggingface, direct_url, or local_file.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        help="[cli mode] Optional specific .gguf filename in HF repo.",
    ),
    room: str = typer.Option(
        "public",
        "--room",
        help="[cli mode] Room ID to create or join.",
    ),
    room_key: str | None = typer.Option(
        None,
        "--room-key",
        help="[cli mode] Optional secret passphrase for private rooms.",
    ),
    network: NetworkMode = typer.Option(
        NetworkMode.INTERNET,
        "--network",
        help="[cli mode] Network scope: internet or lan.",
    ),
    p2p_port: int = typer.Option(
        49152,
        "--p2p-port",
        min=1024,
        max=65535,
        help="[cli mode] P2P tensor listener port.",
    ),
    ssd_offload: bool = typer.Option(
        False,
        "--ssd-offload/--no-ssd-offload",
        help="[cli mode] Enable llama_cpp use_mmap=True SSD offloading in Solo Mode.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE environment setting.",
    ),
) -> None:
    """Launch ShardAi Core Engine as a Tauri daemon sidecar or standalone CLI node."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in ("daemon", "cli"):
        raise typer.BadParameter("mode must be either 'daemon' or 'cli'.")

    if offline is not None:
        os.environ["SHARDAI_OFFLINE_MODE"] = "true" if offline else "false"
    os.environ["SHARDAI_ENGINE_PORT"] = str(port)

    if normalized_mode == "cli":
        cli_request = StartNodeRequest(
            node_id="shardai-cli-node",
            model_id=model,
            source_type=source_type,
            filename=filename,
            network_mode=network,
            room_id=room,
            room_key=room_key,
            ram_gb=ram,
            shared_ram_gb=ram,
            max_ram_mb=ram * 1024,
            p2p_listen_port=p2p_port,
            allow_ssd_offload=ssd_offload,
        )
        asyncio.run(_serve_cli_mode(host=host, port=port, request=cli_request))
        return

    # Daemon mode (default when launched with no arguments or by Tauri sidecar)
    # Pass `app` object directly rather than `"main:app"` so PyInstaller --onefile works cleanly.
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli_app()
