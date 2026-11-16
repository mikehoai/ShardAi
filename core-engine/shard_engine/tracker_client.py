"""HTTP Signaling Client for communicating with the central Cloudflare Tracker REST API.

Complies with `.antigravity/rules.md`:
- 100% Python Type Hints.
- Mandatory explicit timeout on all external HTTP requests.
- Exponential backoff retry on transient network/server errors.
"""

import asyncio
import logging
import random
from typing import Any

import httpx

from shard_engine.schemas import PeerNodeInfo, StartNodeRequest

logger = logging.getLogger("shardai.tracker")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class SignalingTrackerClient:
    """Client for node registration and peer discovery via the central REST Signaling Server."""

    def __init__(
        self,
        base_url: str = "https://tracker.shardai.network/api/v1",
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

        raise RuntimeError(
            f"Signaling tracker request failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    async def register_node_and_fetch_peers(
        self, config: StartNodeRequest
    ) -> list[PeerNodeInfo]:
        """Announce this node via HTTP POST and retrieve active peer IPs for the model."""
        payload: dict[str, Any] = {
            "node_id": config.node_id,
            "model_id": config.model_id,
            "p2p_port": config.p2p_listen_port,
            "shard_layer_start": config.shard_layer_start,
            "shard_layer_end": config.shard_layer_end,
            "max_vram_mb": config.max_vram_mb,
        }
        response_data: dict[str, Any] = await self._request_with_retry(
            method="POST",
            endpoint="/nodes/announce",
            json_payload=payload,
        )
        raw_peers: list[dict[str, Any]] = response_data.get("peers", [])
        return [PeerNodeInfo.model_validate(peer) for peer in raw_peers]

    async def fetch_peers(self, model_id: str) -> list[PeerNodeInfo]:
        """Query the Signaling Server via HTTP GET for active peers hosting a specific model."""
        response_data: dict[str, Any] = await self._request_with_retry(
            method="GET",
            endpoint="/nodes/peers",
            params={"model_id": model_id},
        )
        raw_peers: list[dict[str, Any]] = response_data.get("peers", [])
        return [PeerNodeInfo.model_validate(peer) for peer in raw_peers]

    async def unregister_node(self, node_id: str) -> None:
        """Notify the Signaling Server that this node is going offline."""
        await self._request_with_retry(
            method="POST",
            endpoint="/nodes/leave",
            json_payload={"node_id": node_id},
        )
