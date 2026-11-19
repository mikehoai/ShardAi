"""HTTP Signaling Client for communicating with the central Cloudflare Tracker REST API.

Complies with `.antigravity/rules.md`:
- 100% Python Type Hints.
- Mandatory explicit timeout on all external HTTP requests.
- Exponential backoff retry on transient network/server errors.
"""

import asyncio
from dataclasses import dataclass
import hashlib
import logging
import random
from typing import Any
from urllib.parse import quote

import httpx

from shard_engine.schemas import PeerNodeInfo, PeerRole, RoomLobbyItem, StartNodeRequest

logger = logging.getLogger("shardai.tracker")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class TrackerUnreachableError(RuntimeError):
    """Raised when the Cloudflare Tracker cannot be reached (timeout / DNS / no route / 5xx)."""


@dataclass(frozen=True, slots=True)
class AnnounceAssignment:
    """Cloud-orchestrated shard assignment returned by `POST /announce`."""

    assigned_start_layer: int
    assigned_end_layer: int
    role: PeerRole
    peers: list[PeerNodeInfo]


def _normalize_peer(
    raw: dict[str, Any],
    default_room_id: str,
    default_end_layer: int = 31,
) -> PeerNodeInfo:
    """Normalize peer payloads across different `shardai-cloud` response shapes."""
    raw_role = str(raw.get("role") or PeerRole.PRIMARY.value).lower()
    role = PeerRole.BACKUP if raw_role == PeerRole.BACKUP.value else PeerRole.PRIMARY
    start_val = raw.get("assigned_start_layer")
    if start_val is None:
        start_val = raw.get("shard_layer_start")
    end_val = raw.get("assigned_end_layer")
    if end_val is None:
        end_val = raw.get("shard_layer_end")

    return PeerNodeInfo(
        peer_id=str(raw.get("peer_id") or raw.get("node_id") or "unknown"),
        ip_address=str(raw.get("ip_address") or raw.get("ip") or "127.0.0.1"),
        port=int(raw.get("port") or 49152),
        shard_layer_start=int(start_val if start_val is not None else 0),
        shard_layer_end=int(end_val if end_val is not None else default_end_layer),
        role=role,
        available_vram_mb=int(raw.get("available_vram_mb") or 0),
        room_id=str(raw.get("room_id") or default_room_id),
    )


def hash_room_key(room_key: str | None) -> str | None:
    """Compute a deterministic SHA-256 hex digest of a private room secret key."""
    if room_key is None:
        return None
    cleaned = room_key.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def extract_tracker_error_detail(
    response: httpx.Response,
    fallback: str,
) -> str:
    """Extract a human-readable error message from a Signaling Tracker JSON error response."""
    try:
        payload: Any = response.json()
    except Exception:
        return fallback

    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            nested_msg = error_obj.get("message")
            if isinstance(nested_msg, str) and nested_msg.strip():
                return nested_msg.strip()
        for key in ("message", "detail"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return fallback


class SignalingTrackerClient:
    """Client for node registration and peer discovery via the central REST Signaling Server."""

    def __init__(
        self,
        base_url: str = "https://network.shardai.online",
        timeout_seconds: float = 8.0,
        connect_timeout_seconds: float = 4.0,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url: str = base_url.rstrip("/")
        self.timeout: httpx.Timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
        )
        self.max_retries: int = max_retries
        self.base_backoff_seconds: float = base_backoff_seconds
        self._transport: httpx.AsyncBaseTransport | None = transport

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request against the Cloudflare Tracker with timeout and retry."""
        url: str = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        async with httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        ) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response: httpx.Response = await client.request(
                        method=method,
                        url=url,
                        json=json_payload,
                        params=params,
                    )
                    if response.status_code in RETRYABLE_STATUS_CODES:
                        raise httpx.HTTPStatusError(
                            f"Transient tracker HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                    return data
                except (
                    httpx.TimeoutException,
                    httpx.ConnectError,
                    httpx.NetworkError,
                    httpx.HTTPStatusError,
                ) as exc:
                    last_error = exc
                    # Do not retry non-retryable client HTTP errors (e.g. 400, 401, 403, 422)
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code not in RETRYABLE_STATUS_CODES
                    ):
                        raise

                    if attempt < self.max_retries:
                        jitter: float = random.uniform(0.0, 0.25)
                        sleep_duration: float = (
                            self.base_backoff_seconds * (2 ** (attempt - 1)) + jitter
                        )
                        logger.warning(
                            "Tracker request %s %s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                            method,
                            url,
                            attempt,
                            self.max_retries,
                            exc,
                            sleep_duration,
                        )
                        await asyncio.sleep(sleep_duration)

        raise TrackerUnreachableError(
            f"Signaling tracker request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    async def ping(self, timeout_seconds: float = 3.0) -> bool:
        """Single-shot (no retry) reachability probe of the Tracker root `GET /`.

        Used by the UI Network Status Indicator; never raises.
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=timeout_seconds, connect=timeout_seconds),
                transport=self._transport,
            ) as client:
                response = await client.get(f"{self.base_url}/")
                return response.status_code < 500
        except (httpx.HTTPError, OSError) as exc:
            logger.debug("Tracker ping failed: %s", exc)
            return False

    async def fetch_model_catalog(self) -> list[dict[str, Any]]:
        """Fetch enabled models from Cloudflare Worker `GET /config/models`."""
        response_data: dict[str, Any] = await self._request_with_retry(
            method="GET",
            endpoint="/config/models",
        )
        inner_data: dict[str, Any] = (
            response_data["data"]
            if isinstance(response_data.get("data"), dict)
            else response_data
        )
        raw_models: Any = inner_data.get("models", response_data.get("models", []))
        if not isinstance(raw_models, list):
            return []
        return [
            dict(item)
            for item in raw_models
            if isinstance(item, dict) and item.get("enabled", True) is True
        ]

    async def register_node_with_assignment(
        self,
        config: StartNodeRequest,
        total_model_layers: int | None = None,
        max_layers_can_hold: int | None = None,
        model_fingerprint: str | None = None,
    ) -> AnnounceAssignment:
        """Announce node capacity to `POST /announce` and receive Cloud layer assignment & role."""
        resolved_total_layers: int = total_model_layers or config.total_layers
        resolved_max_layers: int = max_layers_can_hold or max(
            1, config.shard_layer_end - config.shard_layer_start + 1
        )
        room_key_hash: str | None = hash_room_key(config.room_key)
        is_private: bool = room_key_hash is not None

        payload: dict[str, Any] = {
            "node_id": config.node_id,
            "model": config.model_id,
            "model_id": config.model_id,
            "network_mode": config.network_mode.value,
            "room_id": config.room_id,
            "is_private": is_private,
            "room_key_hash": room_key_hash,
            "unlisted": config.is_unlisted,
            "is_unlisted": config.is_unlisted,
            "available_ram_gb": float(config.ram_gb),
            "ram_gb": config.ram_gb,
            "total_model_layers": resolved_total_layers,
            "max_layers_can_hold": resolved_max_layers,
            "n_layers": resolved_max_layers,
            "port": config.p2p_listen_port,
            "p2p_port": config.p2p_listen_port,
            "shard_layer_start": config.shard_layer_start,
            "shard_layer_end": config.shard_layer_end,
            "max_vram_mb": config.max_vram_mb,
            "source_type": config.source_type.value,
            **({"model_fingerprint": model_fingerprint} if model_fingerprint else {}),
        }
        response_data: dict[str, Any] = await self._request_with_retry(
            method="POST",
            endpoint="/announce",
            json_payload=payload,
        )
        inner_data: dict[str, Any] = (
            response_data["data"]
            if isinstance(response_data.get("data"), dict)
            else response_data
        )

        raw_start = inner_data.get(
            "assigned_start_layer",
            response_data.get("assigned_start_layer", config.shard_layer_start),
        )
        raw_end = inner_data.get(
            "assigned_end_layer",
            response_data.get("assigned_end_layer", config.shard_layer_end),
        )
        raw_role = str(
            inner_data.get("role", response_data.get("role", PeerRole.PRIMARY.value))
        ).lower()
        assigned_role = (
            PeerRole.BACKUP if raw_role == PeerRole.BACKUP.value else PeerRole.PRIMARY
        )
        assigned_start = max(0, min(int(raw_start), resolved_total_layers - 1))
        assigned_end = max(
            assigned_start, min(int(raw_end), resolved_total_layers - 1)
        )

        raw_peers: list[dict[str, Any]] = inner_data.get(
            "peers", response_data.get("peers", [])
        )
        normalized: list[PeerNodeInfo] = [
            _normalize_peer(raw, config.room_id, resolved_total_layers - 1)
            for raw in raw_peers
        ]

        # If /announce didn't embed peers, query GET /peers/:room_id/:model
        if not normalized and "peers" not in inner_data and "peers" not in response_data:
            try:
                normalized = await self.fetch_peers(
                    model_id=config.model_id,
                    room_id=config.room_id,
                    room_key=config.room_key,
                )
            except Exception as exc:
                logger.debug("Optional follow-up fetch_peers skipped: %s", exc)

        return AnnounceAssignment(
            assigned_start_layer=assigned_start,
            assigned_end_layer=assigned_end,
            role=assigned_role,
            peers=normalized,
        )

    async def register_node_and_fetch_peers(
        self,
        config: StartNodeRequest,
        total_model_layers: int | None = None,
        max_layers_can_hold: int | None = None,
    ) -> list[PeerNodeInfo]:
        """Announce this node via HTTP POST /announce and return active peers."""
        assignment = await self.register_node_with_assignment(
            config=config,
            total_model_layers=total_model_layers,
            max_layers_can_hold=max_layers_can_hold,
        )
        return assignment.peers

    async def fetch_available_rooms(self) -> list[RoomLobbyItem]:
        """Fetch active public & locked rooms from Cloudflare Tracker GET /rooms."""
        response_data: dict[str, Any] = await self._request_with_retry(
            method="GET",
            endpoint="/rooms",
        )
        inner_data: dict[str, Any] = (
            response_data["data"]
            if isinstance(response_data.get("data"), dict)
            else response_data
        )
        raw_rooms: list[dict[str, Any]] = inner_data.get(
            "rooms", response_data.get("rooms", [])
        )
        parsed: list[RoomLobbyItem] = []
        for room in raw_rooms:
            nodes_count = int(
                room.get("nodes_online") or room.get("node_count") or 1
            )
            shared_ram = float(
                room.get("total_shared_ram_gb") or room.get("total_ram_gb") or 0.0
            )
            coverage = float(
                room.get("coverage_percent")
                if room.get("coverage_percent") is not None
                else 100.0
            )
            redundancy = int(
                room.get("redundancy_factor")
                if room.get("redundancy_factor") is not None
                else 1
            )
            parsed.append(
                RoomLobbyItem(
                    room_id=str(room.get("room_id") or "public"),
                    model=str(room.get("model") or ""),
                    nodes_online=max(1, nodes_count),
                    total_shared_ram_gb=max(0.0, shared_ram),
                    is_private=bool(room.get("is_private", False)),
                    network_scope=room.get("network_scope") or "internet",
                    coverage_percent=max(0.0, min(100.0, coverage)),
                    redundancy_factor=max(1, redundancy),
                )
            )
        return parsed

    async def fetch_peers(
        self,
        model_id: str,
        room_id: str = "public",
        room_key: str | None = None,
        room_key_hash: str | None = None,
    ) -> list[PeerNodeInfo]:
        """Query the Signaling Server via HTTP GET for active peers in a room."""
        params: dict[str, Any] = {"room_id": room_id}
        effective_hash = room_key_hash or hash_room_key(room_key)
        if effective_hash is not None:
            params["room_key_hash"] = effective_hash
        safe_room = quote(room_id, safe="")
        safe_model = quote(model_id, safe="")
        response_data: dict[str, Any] = await self._request_with_retry(
            method="GET",
            endpoint=f"/peers/{safe_room}/{safe_model}",
            params=params,
        )
        raw_peers: list[dict[str, Any]] = response_data.get("peers", [])
        return [_normalize_peer(raw, room_id) for raw in raw_peers]

    async def check_room_join(
        self,
        model_id: str,
        room_id: str = "public",
        room_key: str | None = None,
        room_key_hash: str | None = None,
    ) -> list[PeerNodeInfo]:
        """Pre-flight check against `GET /peers/:room_id/:model` without mutating tracker state.

        Raises `httpx.HTTPStatusError` immediately (no retries) on deterministic 403/409
        rejections (invalid private room key, full room, or busy join queue).
        """
        return await self.fetch_peers(
            model_id=model_id,
            room_id=room_id,
            room_key=room_key,
            room_key_hash=room_key_hash,
        )

    async def unregister_node(
        self, node_id: str, room_id: str | None = None
    ) -> None:
        """Notify the Signaling Server that this node is going offline."""
        payload: dict[str, Any] = {"node_id": node_id}
        if room_id is not None:
            payload["room_id"] = room_id
        await self._request_with_retry(
            method="POST",
            endpoint="/nodes/leave",
            json_payload=payload,
        )
