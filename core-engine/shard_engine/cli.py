"""ShardAi CLI — Standard subcommand-based CLI and Tauri daemon entrypoint.

Provides:
- Standard `shardai <command> [options]` subcommands (`create`, `join`, `serve`,
  `status`, `stop`, `rooms`, `models`, `pull`, `peers`, `chat`, `network`,
  `doctor`, `service`, `completion`, `version`).
- Seamless normalization for single-dash long flags (`-model`, `-room`, `-ram`, etc.)
  alongside standard POSIX `--model` / `-m` flags.
- Backwards compatibility for the Tauri desktop sidecar (`--mode daemon --port 8000`)
  and legacy `--mode cli` headless invocations.
- Strict loopback-only control plane enforcement (`127.0.0.1`, `localhost`, `::1`),
  masked API key output by default, and explicit `httpx.Timeout` on all HTTP calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import importlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys
import time
import types
from typing import Literal, NoReturn
from urllib.parse import urlparse

import click
import httpx
import typer
from typer.core import TyperGroup
import uvicorn

from models.downloader import (
    DEFAULT_CATALOG_MODELS,
    SUPPORTED_MODELS,
    DownloadProgressEvent,
    ModelDownloader,
    is_offline_mode,
    normalize_hf_repo_id,
)
from shard_engine import __version__ as SHARDAI_VERSION
from shard_engine.linux_env import (
    SERVICE_ACTIONS,
    clear_daemon_pid,
    collect_environment_diagnostics,
    get_daemon_log_file,
    get_systemd_unit_path,
    is_process_alive,
    read_daemon_pid,
    render_shell_completion,
    render_systemd_unit,
    run_systemctl,
    terminate_daemon_process,
    write_daemon_pid,
)
from shard_engine.memory_manager import get_system_memory_info
from shard_engine.schemas import (
    ModelSourceType,
    NetworkMode,
    RoomLobbyItem,
    StartNodeRequest,
)
from shard_engine.tracker_client import SignalingTrackerClient

CLI_VERSION: str = SHARDAI_VERSION
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8000
DEFAULT_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

ALLOWED_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})

SINGLE_DASH_LONG_FLAGS: frozenset[str] = frozenset(
    {
        "api-key",
        "api-url",
        "backend",
        "background",
        "enable",
        "filename",
        "foreground",
        "help",
        "host",
        "json",
        "max-tokens",
        "mode",
        "model",
        "network",
        "no-enable",
        "no-ssd-offload",
        "no-stream",
        "no-unlisted",
        "node-id",
        "offline",
        "online",
        "p2p-port",
        "port",
        "prompt",
        "ram",
        "room",
        "room-key",
        "scope",
        "show-key",
        "shutdown-daemon",
        "source-type",
        "ssd-offload",
        "stream",
        "system",
        "temperature",
        "tracker-url",
        "unlisted",
        "version",
    }
)

_COMMON_CONTEXT_SETTINGS: dict[str, object] = {
    "help_option_names": ["-h", "--help"],
}

_registered_engine_module: types.ModuleType | None = None
_default_no_subcommand_action: Literal["help", "daemon"] = "help"
_http_transport_override: httpx.BaseTransport | None = None


def normalize_cli_argv(argv: Sequence[str]) -> list[str]:
    """Convert single-dash long flags like `-model` or `-model=x` into `--model`.

    Preserves short POSIX flags (`-m`, `-r`, `-p`, `-k`, `-H`, `-h`, `-V`) and stops
    normalizing after an explicit `--` end-of-options separator.
    """
    normalized: list[str] = []
    passthrough: bool = False
    for arg in argv:
        if passthrough:
            normalized.append(arg)
            continue
        if arg == "--":
            passthrough = True
            normalized.append(arg)
            continue
        if arg.startswith("-") and not arg.startswith("--") and len(arg) > 2:
            raw_body = arg[1:]
            flag_name, sep, remainder = raw_body.partition("=")
            if flag_name in SINGLE_DASH_LONG_FLAGS:
                normalized.append(f"--{flag_name}{sep}{remainder}")
                continue
        normalized.append(arg)
    return normalized


class _ShardAiTyperGroup(TyperGroup):
    """Typer group that normalizes single-dash long flags before Click parsing."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        normalized = normalize_cli_argv(args)
        return super().parse_args(ctx, normalized)


cli_app = typer.Typer(
    name="shardai",
    cls=_ShardAiTyperGroup,
    help="ShardAi CLI — P2P Distributed LLM Node & Local OpenAI-Compatible API Gateway.",
    add_completion=False,
    context_settings=_COMMON_CONTEXT_SETTINGS,
)


def validate_loopback_host(host: str) -> str:
    """Validate that the local HTTP control plane host is strictly loopback."""
    cleaned = host.strip().lower()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned not in ALLOWED_LOOPBACK_HOSTS:
        raise typer.BadParameter(
            f"Control plane host must be loopback-only (127.0.0.1, localhost, or ::1); got '{host}'."
        )
    return cleaned


def _validate_loopback_url(url: str, label: str = "URL") -> str:
    """Validate that an HTTP(S) URL targets a strictly loopback hostname."""
    cleaned = url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https"):
        raise typer.BadParameter(
            f"{label} must begin with http:// or https://; got '{url}'."
        )
    validate_loopback_host(parsed.hostname or "")
    return cleaned


def _is_loopback_url(url: str | None) -> bool:
    """Return True when `url` is an http(s) URL whose hostname is strictly loopback."""
    if not url or not url.strip():
        return False
    try:
        _validate_loopback_url(url)
        return True
    except typer.BadParameter:
        return False


def build_engine_base_url(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    api_url: str | None = None,
) -> str:
    """Resolve the local core-engine HTTP base URL from `--api-url` or `--host`/`--port`."""
    if api_url is not None and api_url.strip():
        cleaned = _validate_loopback_url(api_url, label="--api-url")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[:-3]
        return cleaned.rstrip("/")

    safe_host = validate_loopback_host(host)
    formatted_host = f"[{safe_host}]" if ":" in safe_host else safe_host
    return f"http://{formatted_host}:{port}"


def mask_api_key(token: str | None) -> str:
    """Redact a session API token for safe terminal display."""
    if not token or not token.strip():
        return "<none>"
    cleaned = token.strip()
    if len(cleaned) <= 10:
        return "****"
    prefix = "sk-shard-" if cleaned.startswith("sk-shard-") else cleaned[:6]
    return f"{prefix}****{cleaned[-4:]}"


def extract_http_error_detail(response: httpx.Response, fallback: str) -> str:
    """Extract a clean English error string from a FastAPI or OpenAI JSON error body."""
    try:
        payload_obj: object = response.json()
    except Exception:
        text = response.text.strip()
        return text or fallback

    if isinstance(payload_obj, dict):
        detail = payload_obj.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        error_obj = payload_obj.get("error")
        if isinstance(error_obj, dict):
            err_msg = error_obj.get("message")
            if isinstance(err_msg, str) and err_msg.strip():
                return err_msg.strip()
        message = payload_obj.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return fallback


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a clean aligned plain-text table without external dependencies."""
    widths: list[int] = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))

    def _fmt_row(values: Sequence[str]) -> str:
        padded = [
            (values[idx] if idx < len(values) else "").ljust(widths[idx])
            for idx in range(len(widths))
        ]
        return "  ".join(padded).rstrip()

    lines: list[str] = [
        _fmt_row(headers),
        _fmt_row(["-" * width for width in widths]),
    ]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _lookup_catalog_item(
    model_str: str,
    catalog_entries: Sequence[dict[str, object]] | None = None,
) -> dict[str, object] | None:
    """Match a catalog shorthand `id` or `repo_id` against provided or default entries."""
    cleaned = model_str.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    normalized = normalize_hf_repo_id(cleaned).lower()
    pool: list[dict[str, object]] = []
    if catalog_entries:
        pool.extend(dict(item) for item in catalog_entries)
    pool.extend(dict(item) for item in DEFAULT_CATALOG_MODELS)

    for entry in pool:
        entry_id = str(entry.get("id") or "").strip().lower()
        repo_id = str(entry.get("repo_id") or "").strip().lower()
        if lowered in (entry_id, repo_id) or normalized in (entry_id, repo_id):
            return entry
    return None


def resolve_model_spec(
    model: str,
    source_type: str = "auto",
    filename: str | None = None,
    catalog_entries: Sequence[dict[str, object]] | None = None,
) -> tuple[str, ModelSourceType, str | None]:
    """Resolve user-provided model input into `(model_id, ModelSourceType, filename)`."""
    cleaned = model.strip()
    if not cleaned:
        raise typer.BadParameter("Model identifier cannot be empty.")

    cleaned_filename = filename.strip() if filename and filename.strip() else None
    normalized_source = (source_type or "auto").strip().lower()

    if normalized_source != "auto":
        try:
            explicit_type = ModelSourceType(normalized_source)
        except ValueError as exc:
            valid = ", ".join(item.value for item in ModelSourceType)
            raise typer.BadParameter(
                f"Invalid --source-type '{source_type}'. Choose 'auto' or one of: {valid}."
            ) from exc

        if explicit_type == ModelSourceType.CATALOG:
            match = _lookup_catalog_item(cleaned, catalog_entries=catalog_entries)
            if match is not None:
                repo_id = str(match.get("repo_id") or cleaned).strip()
                cat_file = match.get("filename")
                resolved_file = cleaned_filename or (
                    str(cat_file).strip() if isinstance(cat_file, str) and cat_file.strip() else None
                )
                return repo_id, ModelSourceType.CATALOG, resolved_file
        return cleaned, explicit_type, cleaned_filename

    # 1. Exact catalog ID or repo_id match
    catalog_match = _lookup_catalog_item(cleaned, catalog_entries=catalog_entries)
    if catalog_match is not None:
        repo_id = str(catalog_match.get("repo_id") or cleaned).strip()
        cat_file = catalog_match.get("filename")
        resolved_file = cleaned_filename or (
            str(cat_file).strip() if isinstance(cat_file, str) and cat_file.strip() else None
        )
        return repo_id, ModelSourceType.CATALOG, resolved_file

    # 2. HTTP(S) URLs: distinguish direct .gguf / /resolve/ downloads from HF repo URLs
    lowered = cleaned.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        if (
            "huggingface.co/" not in lowered
            or "/resolve/" in lowered
            or "/blob/" in lowered
            or lowered.endswith(".gguf")
        ):
            return cleaned, ModelSourceType.DIRECT_URL, cleaned_filename
        return normalize_hf_repo_id(cleaned), ModelSourceType.HUGGINGFACE, cleaned_filename

    # 3. Local .gguf file path or local model directory containing config.json
    expanded = Path(cleaned).expanduser()
    if expanded.suffix.lower() == ".gguf" or (
        expanded.exists()
        and (expanded.is_file() or (expanded.is_dir() and (expanded / "config.json").is_file()))
    ):
        return str(expanded), ModelSourceType.LOCAL_FILE, cleaned_filename

    # 4. Built-in supported preset models vs custom Hugging Face repository IDs
    if cleaned in SUPPORTED_MODELS:
        return cleaned, ModelSourceType.CATALOG, cleaned_filename

    return cleaned, ModelSourceType.HUGGINGFACE, cleaned_filename


def _strip_local_token_on_non_loopback_request(request: httpx.Request) -> None:
    """Ensure `X-ShardAi-Local-Token` is never transmitted to a non-loopback target."""
    if "X-ShardAi-Local-Token" in request.headers and not _is_loopback_url(str(request.url)):
        del request.headers["X-ShardAi-Local-Token"]


def _make_http_client(
    timeout: httpx.Timeout,
    transport: httpx.BaseTransport | None = None,
    base_url: str | None = None,
) -> httpx.Client:
    """Create an `httpx.Client` with explicit timeout, loopback-scoped local token, and optional test transport."""
    effective_transport = transport if transport is not None else _http_transport_override
    local_token = os.getenv("SHARDAI_LOCAL_TOKEN", "").strip()
    headers: dict[str, str] | None = (
        {"X-ShardAi-Local-Token": local_token}
        if local_token and _is_loopback_url(base_url)
        else None
    )
    return httpx.Client(
        timeout=timeout,
        transport=effective_transport,
        headers=headers,
        event_hooks={"request": [_strip_local_token_on_non_loopback_request]},
    )


def _check_daemon_healthy(
    base_url: str,
    transport: httpx.BaseTransport | None = None,
) -> bool:
    """Return True when a local ShardAi daemon responds with HTTP 200 on `/health`."""
    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=1.5, connect=0.8),
            transport=transport,
            base_url=base_url,
        ) as client:
            resp = client.get(f"{base_url}/health")
            return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _load_engine_module() -> types.ModuleType:
    """Lazily import `main.py` (reusing an already-loaded main module when available)."""
    if _registered_engine_module is not None:
        return _registered_engine_module

    existing_main = sys.modules.get("__main__")
    if (
        existing_main is not None
        and hasattr(existing_main, "app")
        and hasattr(existing_main, "start_node")
    ):
        return existing_main

    existing_named = sys.modules.get("main")
    if (
        existing_named is not None
        and hasattr(existing_named, "app")
        and hasattr(existing_named, "start_node")
    ):
        return existing_named

    core_dir = str(Path(__file__).resolve().parent.parent)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    return importlib.import_module("main")


def _apply_offline_env(offline: bool | None) -> None:
    """Apply explicit `--offline` / `--online` override to process environment."""
    if offline is not None:
        os.environ["SHARDAI_OFFLINE_MODE"] = "true" if offline else "false"


def _emit_json(payload: object) -> None:
    """Print JSON payload deterministically to stdout."""
    typer.echo(json.dumps(payload, indent=2))


def _fail(message: str, code: int = 1) -> NoReturn:
    """Print an English error message to stderr and exit with non-zero code."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


def _print_node_banner(status_data: dict[str, object], fallback_network: str) -> None:
    """Print a clean English summary banner for an active node session."""
    is_ready = bool(status_data.get("is_swarm_ready"))
    coverage = float(status_data.get("coverage_percent") or 0.0)
    readiness = "API READY" if is_ready else f"STANDBY ({coverage:.0f}%)"
    role = str(status_data.get("role") or "primary").upper()
    layer_start = status_data.get("shard_layer_start")
    layer_end = status_data.get("shard_layer_end")
    total_layers = status_data.get("total_layers")
    net_status = status_data.get("network_status") or fallback_network

    typer.echo("\n" + "=" * 68)
    typer.echo(f" ShardAi Gateway Active — {readiness}")
    typer.echo(f" Model        : {status_data.get('model_id')}")
    typer.echo(f" Room         : {status_data.get('room_id')} ({net_status})")
    typer.echo(f" Role / Layers: {role} ({layer_start}..{layer_end}/{total_layers})")
    typer.echo(f" OpenAI URL   : {status_data.get('api_base_url')}")
    typer.echo(f" API Key      : {status_data.get('session_token')}")
    typer.echo(f" SSD Offload  : {bool(status_data.get('ssd_offload_active'))}")
    typer.echo("=" * 68 + "\n")


async def _serve_headless_node_async(
    host: str,
    port: int,
    request: StartNodeRequest,
    *,
    check_join_first: bool = False,
    output_json: bool = False,
) -> None:
    """Run pre-flight admission checks (optional), start Uvicorn, and join the ShardAi room."""
    engine = _load_engine_module()

    if check_join_first:
        try:
            await engine._verify_cloud_room_join_permission(
                room_id=request.room_id,
                model_id=request.model_id,
                network_mode=request.network_mode,
                room_key=request.room_key,
            )
        except Exception as exc:
            detail = getattr(exc, "detail", str(exc))
            raise RuntimeError(str(detail)) from exc

    config = uvicorn.Config(
        engine.app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    server = uvicorn.Server(config)
    startup_error: list[str] = []

    async def _bootstrap_cli_node() -> None:
        while not server.started:
            await asyncio.sleep(0.05)
        try:
            status_resp = await engine.start_node(request)
            payload = status_resp.model_dump(mode="json")
            if output_json:
                _emit_json(payload)
            else:
                _print_node_banner(payload, request.network_mode.value)
        except Exception as exc:
            detail = getattr(exc, "detail", str(exc))
            startup_error.append(str(detail))
            typer.echo(f"[ShardAi CLI] Failed to start node: {detail}", err=True)
            server.should_exit = True

    bootstrap_task = asyncio.create_task(_bootstrap_cli_node())
    try:
        await server.serve()
    finally:
        if not bootstrap_task.done():
            bootstrap_task.cancel()

    if startup_error:
        raise RuntimeError(startup_error[0])


def _run_standalone_node(
    host: str,
    port: int,
    request: StartNodeRequest,
    *,
    check_join_first: bool = False,
    output_json: bool = False,
) -> None:
    """Synchronous wrapper launching `_serve_headless_node_async` and surfacing exit code 1 on error."""
    try:
        asyncio.run(
            _serve_headless_node_async(
                host=host,
                port=port,
                request=request,
                check_join_first=check_join_first,
                output_json=output_json,
            )
        )
    except RuntimeError as exc:
        _fail(str(exc), code=1)


def _run_daemon_server(host: str, port: int) -> None:
    """Start the local FastAPI control plane and OpenAI gateway daemon."""
    safe_host = validate_loopback_host(host)
    os.environ["SHARDAI_ENGINE_PORT"] = str(port)
    engine = _load_engine_module()
    uvicorn.run(
        engine.app,
        host=safe_host,
        port=port,
        reload=False,
        log_level="info",
    )


def _daemon_start_node(
    base_url: str,
    request: StartNodeRequest,
    *,
    check_join_first: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    """Attach to a running daemon: run pre-flight check (when requested) and call `POST /start-node`."""
    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=120.0, connect=4.0),
            transport=transport,
            base_url=base_url,
        ) as client:
            if check_join_first or request.network_mode == NetworkMode.INTERNET:
                pre_resp = client.post(
                    f"{base_url}/rooms/check-join",
                    json={
                        "room_id": request.room_id,
                        "model_id": request.model_id,
                        "network_mode": request.network_mode.value,
                        "room_key": request.room_key,
                    },
                )
                if pre_resp.status_code >= 400:
                    _fail(
                        extract_http_error_detail(
                            pre_resp,
                            f"Room '{request.room_id}' rejected join (HTTP {pre_resp.status_code}).",
                        )
                    )

            start_resp = client.post(
                f"{base_url}/start-node",
                json=request.model_dump(mode="json"),
            )
            if start_resp.status_code >= 400:
                _fail(
                    extract_http_error_detail(
                        start_resp,
                        f"Failed to start node (HTTP {start_resp.status_code}).",
                    )
                )
            data_obj: object = start_resp.json()
            if not isinstance(data_obj, dict):
                _fail("Unexpected response format from local ShardAi daemon.")
            return dict(data_obj)
    except (httpx.HTTPError, OSError) as exc:
        _fail(f"Could not communicate with ShardAi daemon at {base_url}: {exc}")


def _fetch_catalog_entries_best_effort(
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> list[dict[str, object]]:
    """Fetch catalog entries from local daemon, cache file, or defaults without failing."""
    if base_url and _check_daemon_healthy(base_url, transport=transport):
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=5.0, connect=2.0),
                transport=transport,
                base_url=base_url,
            ) as client:
                resp = client.get(f"{base_url}/models/catalog")
                if resp.status_code == 200:
                    payload = resp.json()
                    models = payload.get("models") if isinstance(payload, dict) else None
                    if isinstance(models, list):
                        return [dict(item) for item in models if isinstance(item, dict)]
        except Exception:
            pass
    try:
        cached = ModelDownloader().load_catalog_cache()
        if cached:
            return [dict(item) for item in cached]
    except Exception:
        pass
    return [dict(item) for item in DEFAULT_CATALOG_MODELS]


def _resolve_room_model_for_join(
    room_id: str,
    base_url: str,
    daemon_running: bool,
    network_mode: NetworkMode,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """Resolve the model assigned to `room_id` when `--model` was omitted in `shardai join`."""
    if daemon_running:
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=6.0, connect=2.0),
                transport=transport,
                base_url=base_url,
            ) as client:
                resp = client.get(f"{base_url}/available-rooms")
                if resp.status_code == 200:
                    items = resp.json()
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and item.get("room_id") == room_id:
                                model_name = str(item.get("model") or "").strip()
                                if model_name:
                                    return model_name
        except Exception:
            pass
    elif network_mode == NetworkMode.INTERNET and not is_offline_mode():
        try:
            tracker = SignalingTrackerClient(
                base_url=os.getenv("SHARDAI_TRACKER_URL", "https://network.shardai.online"),
                timeout_seconds=5.0,
                connect_timeout_seconds=3.0,
                max_retries=1,
            )
            rooms = asyncio.run(tracker.fetch_available_rooms())
            for room in rooms:
                if room.room_id == room_id and room.model.strip():
                    return room.model.strip()
        except Exception:
            pass

    if room_id == "public":
        return DEFAULT_MODEL

    _fail(
        f"Could not auto-detect model for room '{room_id}'. Pass --model <model_id> explicitly."
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"shardai {CLI_VERSION}")
        raise typer.Exit(code=0)


@cli_app.callback(invoke_without_command=True)
def root_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show ShardAi CLI version and exit.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="Legacy entrypoint mode: 'daemon' (Tauri sidecar) or 'cli' (headless node).",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback HTTP API bind host (127.0.0.1, localhost, or ::1).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
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
        help="[legacy --mode cli] RAM contribution in GB.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="[legacy --mode cli] Model Repo ID, Direct .gguf URL, or Local .gguf path.",
    ),
    source_type: ModelSourceType = typer.Option(
        ModelSourceType.CATALOG,
        "--source-type",
        help="[legacy --mode cli] Model source: catalog, huggingface, direct_url, or local_file.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        help="[legacy --mode cli] Optional specific .gguf filename in HF repo.",
    ),
    room: str = typer.Option(
        "public",
        "--room",
        help="[legacy --mode cli] Room ID to create or join.",
    ),
    room_key: str | None = typer.Option(
        None,
        "--room-key",
        help="[legacy --mode cli] Optional secret passphrase for private rooms.",
    ),
    network: NetworkMode = typer.Option(
        NetworkMode.INTERNET,
        "--network",
        help="[legacy --mode cli] Network scope: internet or lan.",
    ),
    p2p_port: int = typer.Option(
        49152,
        "--p2p-port",
        min=1024,
        max=65535,
        help="[legacy --mode cli] P2P tensor listener port.",
    ),
    ssd_offload: bool = typer.Option(
        False,
        "--ssd-offload/--no-ssd-offload",
        help="[legacy --mode cli] Enable llama_cpp use_mmap=True SSD offloading in Solo Mode.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE environment setting.",
    ),
) -> None:
    """ShardAi CLI — run `shardai <command> --help` for subcommand details."""
    _ = version
    if ctx.invoked_subcommand is not None:
        return

    safe_host = validate_loopback_host(host)
    _apply_offline_env(offline)
    os.environ["SHARDAI_ENGINE_PORT"] = str(port)

    env_mode = os.getenv("SHARDAI_RUN_MODE", "").strip().lower()
    effective_mode = mode.strip().lower() if mode is not None else env_mode

    if effective_mode:
        if effective_mode not in ("daemon", "cli"):
            raise typer.BadParameter("mode must be either 'daemon' or 'cli'.")
        if effective_mode == "cli":
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
            _run_standalone_node(host=safe_host, port=port, request=cli_request)
            return

        _run_daemon_server(host=safe_host, port=port)
        return

    if _default_no_subcommand_action == "daemon":
        _run_daemon_server(host=safe_host, port=port)
        return

    typer.echo(ctx.get_help())
    raise typer.Exit(code=0)


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def create(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Catalog ID, HuggingFace repo/URL, direct .gguf URL, or local .gguf path.",
    ),
    room: str | None = typer.Option(
        None,
        "--room",
        "-r",
        help="Room ID to create (auto-generated if omitted).",
    ),
    room_key: str | None = typer.Option(
        None,
        "--room-key",
        "-k",
        help="Optional passphrase making the room private.",
    ),
    unlisted: bool = typer.Option(
        False,
        "--unlisted/--no-unlisted",
        help="Hide this room from the public Network Rooms Lobby.",
    ),
    ram: int = typer.Option(
        4,
        "--ram",
        min=2,
        max=256,
        help="RAM budget to share in GB (2-256).",
    ),
    network: NetworkMode = typer.Option(
        NetworkMode.INTERNET,
        "--network",
        "-n",
        help="Network scope: internet or lan.",
    ),
    source_type: str = typer.Option(
        "auto",
        "--source-type",
        help="Model source: auto, catalog, huggingface, direct_url, or local_file.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        "-f",
        help="Optional specific .gguf filename inside a HuggingFace repo.",
    ),
    p2p_port: int = typer.Option(
        49152,
        "--p2p-port",
        min=1024,
        max=65535,
        help="Local TCP port for direct P2P tensor streaming.",
    ),
    ssd_offload: bool = typer.Option(
        False,
        "--ssd-offload/--no-ssd-offload",
        help="Allow mmap SSD offloading when running solo below model RAM size.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for this command.",
    ),
    node_id: str = typer.Option(
        "shardai-cli-node",
        "--node-id",
        help="Custom node identifier advertised to peers.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host (127.0.0.1, localhost, or ::1).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Force running an embedded gateway in the foreground instead of attaching.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Create a new ShardAi room and start sharing RAM/VRAM for the selected model."""
    safe_host = validate_loopback_host(host)
    _apply_offline_env(offline)
    os.environ["SHARDAI_ENGINE_PORT"] = str(port)
    base_url = build_engine_base_url(host=safe_host, port=port)

    daemon_running = not foreground and _check_daemon_healthy(base_url)
    catalog_entries = _fetch_catalog_entries_best_effort(
        base_url if daemon_running else None
    )
    resolved_model, resolved_source, resolved_filename = resolve_model_spec(
        model=model,
        source_type=source_type,
        filename=filename,
        catalog_entries=catalog_entries,
    )

    target_room = (
        room.strip()
        if room is not None and room.strip()
        else f"room-{secrets.token_hex(3)}"
    )
    request = StartNodeRequest(
        node_id=node_id,
        model_id=resolved_model,
        source_type=resolved_source,
        filename=resolved_filename,
        network_mode=network,
        room_id=target_room,
        room_key=room_key,
        is_unlisted=unlisted,
        ram_gb=ram,
        shared_ram_gb=ram,
        max_ram_mb=ram * 1024,
        p2p_listen_port=p2p_port,
        allow_ssd_offload=ssd_offload,
    )

    if daemon_running:
        payload = _daemon_start_node(
            base_url=base_url,
            request=request,
            check_join_first=(network == NetworkMode.INTERNET),
        )
        if json_output:
            _emit_json(payload)
        else:
            _print_node_banner(payload, network.value)
        return

    _run_standalone_node(
        host=safe_host,
        port=port,
        request=request,
        check_join_first=(network == NetworkMode.INTERNET),
        output_json=json_output,
    )


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def join(
    room_arg: str | None = typer.Argument(
        None,
        help="Optional Room ID to join (defaults to 'public').",
    ),
    room: str | None = typer.Option(
        None,
        "--room",
        "-r",
        help="Target Room ID to join.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model ID (auto-detected from active rooms when omitted).",
    ),
    room_key: str | None = typer.Option(
        None,
        "--room-key",
        "-k",
        help="Passphrase for private rooms.",
    ),
    ram: int = typer.Option(
        4,
        "--ram",
        min=2,
        max=256,
        help="RAM budget to share in GB (2-256).",
    ),
    network: NetworkMode = typer.Option(
        NetworkMode.INTERNET,
        "--network",
        "-n",
        help="Network scope: internet or lan.",
    ),
    source_type: str = typer.Option(
        "auto",
        "--source-type",
        help="Model source: auto, catalog, huggingface, direct_url, or local_file.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        "-f",
        help="Optional specific .gguf filename inside a HuggingFace repo.",
    ),
    p2p_port: int = typer.Option(
        49152,
        "--p2p-port",
        min=1024,
        max=65535,
        help="Local TCP port for direct P2P tensor streaming.",
    ),
    ssd_offload: bool = typer.Option(
        False,
        "--ssd-offload/--no-ssd-offload",
        help="Allow mmap SSD offloading when running solo below model RAM size.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for this command.",
    ),
    node_id: str = typer.Option(
        "shardai-cli-node",
        "--node-id",
        help="Custom node identifier advertised to peers.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host (127.0.0.1, localhost, or ::1).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    foreground: bool = typer.Option(
        False,
        "--foreground",
        help="Force running an embedded gateway in the foreground instead of attaching.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Join an existing ShardAi room with pre-flight admission checks before download."""
    safe_host = validate_loopback_host(host)
    _apply_offline_env(offline)
    os.environ["SHARDAI_ENGINE_PORT"] = str(port)
    base_url = build_engine_base_url(host=safe_host, port=port)

    target_room = (room or room_arg or "public").strip() or "public"
    daemon_running = not foreground and _check_daemon_healthy(base_url)

    raw_model = model.strip() if model and model.strip() else _resolve_room_model_for_join(
        room_id=target_room,
        base_url=base_url,
        daemon_running=daemon_running,
        network_mode=network,
    )
    catalog_entries = _fetch_catalog_entries_best_effort(
        base_url if daemon_running else None
    )
    resolved_model, resolved_source, resolved_filename = resolve_model_spec(
        model=raw_model,
        source_type=source_type,
        filename=filename,
        catalog_entries=catalog_entries,
    )

    request = StartNodeRequest(
        node_id=node_id,
        model_id=resolved_model,
        source_type=resolved_source,
        filename=resolved_filename,
        network_mode=network,
        room_id=target_room,
        room_key=room_key,
        ram_gb=ram,
        shared_ram_gb=ram,
        max_ram_mb=ram * 1024,
        p2p_listen_port=p2p_port,
        allow_ssd_offload=ssd_offload,
    )

    if daemon_running:
        payload = _daemon_start_node(
            base_url=base_url,
            request=request,
            check_join_first=True,
        )
        if json_output:
            _emit_json(payload)
        else:
            _print_node_banner(payload, network.value)
        return

    _run_standalone_node(
        host=safe_host,
        port=port,
        request=request,
        check_join_first=True,
        output_json=json_output,
    )


def _start_daemon_background(
    *,
    host: str,
    port: int,
    offline: bool | None,
    json_output: bool,
    startup_timeout_seconds: float = 15.0,
) -> None:
    """Launch `shardai serve` detached in the background, track its PID, and await `/health`."""
    safe_host = validate_loopback_host(host)
    base_url = build_engine_base_url(host=safe_host, port=port)

    if _check_daemon_healthy(base_url):
        existing_pid = read_daemon_pid(port)
        if json_output:
            _emit_json(
                {
                    "status": "already_running",
                    "host": safe_host,
                    "port": port,
                    "pid": existing_pid,
                    "api_base_url": f"{base_url}/v1",
                }
            )
        else:
            typer.echo(
                f"ShardAi daemon is already running at {base_url}"
                + (f" (PID {existing_pid})." if existing_pid else ".")
            )
        return

    stale_pid = read_daemon_pid(port)
    if stale_pid is not None and not is_process_alive(stale_pid):
        clear_daemon_pid(port)

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "serve", "--host", safe_host, "--port", str(port)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "shard_engine",
            "serve",
            "--host",
            safe_host,
            "--port",
            str(port),
        ]
    if offline is True:
        cmd.append("--offline")
    elif offline is False:
        cmd.append("--online")

    child_env = os.environ.copy()
    child_env["SHARDAI_ENGINE_PORT"] = str(port)
    child_env["PYTHONUNBUFFERED"] = "1"
    if not getattr(sys, "frozen", False):
        core_dir = str(Path(__file__).resolve().parent.parent)
        existing_pp = child_env.get("PYTHONPATH", "").strip()
        child_env["PYTHONPATH"] = f"{core_dir}{os.pathsep}{existing_pp}" if existing_pp else core_dir

    log_file = get_daemon_log_file(port)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as log_handle:
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=child_env,
                creationflags=flags,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )

    write_daemon_pid(port, proc.pid)

    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if _check_daemon_healthy(base_url):
            if json_output:
                _emit_json(
                    {
                        "status": "running",
                        "host": safe_host,
                        "port": port,
                        "pid": proc.pid,
                        "log_file": str(log_file),
                        "api_base_url": f"{base_url}/v1",
                    }
                )
            else:
                typer.echo(f"ShardAi daemon started in background (PID {proc.pid}).")
                typer.echo(f"Gateway URL : {base_url}/v1")
                typer.echo(f"Log file    : {log_file}")
            return
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except Exception:
            proc.kill()
    clear_daemon_pid(port)
    _fail(
        f"Background ShardAi daemon failed to become healthy on {base_url}. "
        f"Check logs at {log_file}."
    )


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def serve(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Loopback host to bind the FastAPI daemon (127.0.0.1, localhost, or ::1).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local HTTP OpenAI Gateway port.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE environment setting.",
    ),
    background: bool = typer.Option(
        False,
        "--background",
        "-d",
        help="Run the daemon detached in the background and record its PID/log files.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON (with --background).",
    ),
) -> None:
    """Start the local ShardAi FastAPI daemon without immediately joining a room."""
    safe_host = validate_loopback_host(host)
    _apply_offline_env(offline)
    if background:
        _start_daemon_background(
            host=safe_host,
            port=port,
            offline=offline,
            json_output=json_output,
        )
        return
    _run_daemon_server(host=safe_host, port=port)


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def status(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL (e.g. http://127.0.0.1:8000).",
    ),
    show_key: bool = typer.Option(
        False,
        "--show-key",
        help="Display the full session API key instead of masking it.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Show active node status, allocated RAM/VRAM, layer range, peers, and readiness."""
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=5.0, connect=2.0),
            base_url=base_url,
        ) as client:
            status_resp = client.get(f"{base_url}/status")
            if status_resp.status_code >= 400:
                _fail(
                    extract_http_error_detail(
                        status_resp,
                        f"Failed to retrieve node status (HTTP {status_resp.status_code}).",
                    )
                )
            node_data = status_resp.json()
            if not isinstance(node_data, dict):
                _fail("Unexpected status payload from local daemon.")

            net_data: dict[str, object] | None = None
            try:
                net_resp = client.get(f"{base_url}/network-status")
                if net_resp.status_code == 200 and isinstance(net_resp.json(), dict):
                    net_data = dict(net_resp.json())
            except Exception:
                net_data = None
    except (httpx.HTTPError, OSError) as exc:
        _fail(
            f"Local ShardAi daemon is not reachable at {base_url}. "
            f"Run 'shardai serve' or 'shardai create --model ...' first. ({exc})"
        )

    raw_token = node_data.get("session_token")
    token_str = str(raw_token) if isinstance(raw_token, str) and raw_token else None
    displayed_token = token_str if show_key and token_str else mask_api_key(token_str)

    if json_output:
        sanitized = dict(node_data)
        if not show_key and token_str is not None:
            sanitized["session_token"] = displayed_token
        if net_data is not None:
            sanitized["network"] = net_data
        _emit_json(sanitized)
        return

    state = str(node_data.get("state") or "UNKNOWN")
    is_ready = bool(node_data.get("is_swarm_ready"))
    coverage = float(node_data.get("coverage_percent") or 0.0)
    typer.echo(f"State         : {state}")
    typer.echo(f"Readiness     : {'READY' if is_ready else f'STANDBY ({coverage:.1f}%)'}")
    typer.echo(f"Node ID       : {node_data.get('node_id') or '<none>'}")
    typer.echo(f"Model         : {node_data.get('model_id') or '<none>'}")
    typer.echo(f"Room          : {node_data.get('room_id') or '<none>'}")
    typer.echo(
        f"Layers        : {node_data.get('shard_layer_start')}..{node_data.get('shard_layer_end')} / {node_data.get('total_layers')}"
    )
    typer.echo(
        f"Memory        : RAM {node_data.get('allocated_ram_mb', 0)} MB | VRAM {node_data.get('allocated_vram_mb', 0)} MB"
    )
    typer.echo(f"ConnectedPeers: {node_data.get('connected_peers_count', 0)}")
    typer.echo(f"OpenAI URL    : {node_data.get('api_base_url') or '<inactive>'}")
    typer.echo(f"API Key       : {displayed_token}")
    if net_data is not None:
        typer.echo(f"Network Tier  : {net_data.get('network_status')}")


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def stop(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    shutdown_daemon: bool = typer.Option(
        False,
        "--shutdown-daemon",
        help="Terminate the background daemon process after stopping the active node.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Stop the active node, revoke the session API key, and release RAM/VRAM shards."""
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    payload: dict[str, object]
    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=10.0, connect=2.0),
            base_url=base_url,
        ) as client:
            resp = client.post(f"{base_url}/stop-node")
            if resp.status_code >= 400:
                _fail(
                    extract_http_error_detail(
                        resp,
                        f"Failed to stop node (HTTP {resp.status_code}).",
                    )
                )
            raw_payload = resp.json()
            if not isinstance(raw_payload, dict):
                _fail("Unexpected stop response from local daemon.")
            payload = dict(raw_payload)
    except (httpx.HTTPError, OSError) as exc:
        if not shutdown_daemon:
            _fail(f"Local ShardAi daemon is not reachable at {base_url}: {exc}")
        daemon_stopped, daemon_msg = terminate_daemon_process(port)
        if not daemon_stopped:
            _fail(f"Local ShardAi daemon is not reachable at {base_url} ({daemon_msg}).")
        payload = {
            "state": "STOPPED",
            "allocated_ram_mb": 0,
            "allocated_vram_mb": 0,
            "message": daemon_msg,
            "daemon_stopped": True,
            "daemon_message": daemon_msg,
        }
    else:
        if shutdown_daemon:
            daemon_stopped, daemon_msg = terminate_daemon_process(port)
            payload["daemon_stopped"] = daemon_stopped
            payload["daemon_message"] = daemon_msg

    if json_output:
        _emit_json(payload)
        return

    typer.echo(
        f"Node stopped (RAM: {payload.get('allocated_ram_mb', 0)} MB, "
        f"VRAM: {payload.get('allocated_vram_mb', 0)} MB). "
        f"{payload.get('message', '')}".strip()
    )
    if shutdown_daemon and payload.get("daemon_message") and payload.get("daemon_message") != payload.get("message"):
        typer.echo(str(payload["daemon_message"]))


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def rooms(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for standalone discovery.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """List available Internet and LAN ShardAi rooms."""
    _apply_offline_env(offline)
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    room_items: list[dict[str, object]] = []

    if _check_daemon_healthy(base_url):
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=8.0, connect=2.0),
                base_url=base_url,
            ) as client:
                resp = client.get(f"{base_url}/available-rooms")
                if resp.status_code >= 400:
                    _fail(
                        extract_http_error_detail(
                            resp,
                            f"Failed to list rooms (HTTP {resp.status_code}).",
                        )
                    )
                raw = resp.json()
                if isinstance(raw, list):
                    room_items = [dict(item) for item in raw if isinstance(item, dict)]
        except (httpx.HTTPError, OSError) as exc:
            _fail(f"Failed to fetch rooms from {base_url}: {exc}")
    elif not is_offline_mode():
        tracker = SignalingTrackerClient(
            base_url=os.getenv("SHARDAI_TRACKER_URL", "https://network.shardai.online"),
            timeout_seconds=6.0,
            connect_timeout_seconds=3.0,
            max_retries=2,
        )
        try:
            fetched: list[RoomLobbyItem] = asyncio.run(tracker.fetch_available_rooms())
            room_items = [item.model_dump(mode="json") for item in fetched]
        except Exception as exc:
            _fail(f"Could not fetch rooms from Signaling Tracker: {exc}")

    if json_output:
        _emit_json(room_items)
        return

    if not room_items:
        typer.echo("No active rooms found.")
        return

    rows: list[list[str]] = []
    for item in room_items:
        cov = float(item.get("coverage_percent") or 0.0)
        red = int(item.get("redundancy_factor") or 1)
        cov_label = f"Ready ({red}x)" if cov >= 100.0 else f"Pooling ({cov:.0f}%)"
        vis_label = "Private" if item.get("is_private") else "Public"
        rows.append(
            [
                str(item.get("room_id") or ""),
                str(item.get("model") or ""),
                str(item.get("nodes_online") or 1),
                f"{float(item.get('total_shared_ram_gb') or 0.0):.1f} GB",
                cov_label,
                vis_label,
                str(item.get("network_scope") or "internet").upper(),
            ]
        )

    typer.echo(
        format_table(
            ("ROOM", "MODEL", "NODES", "RAM", "COVERAGE", "VISIBILITY", "SCOPE"),
            rows,
        )
    )


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def models(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for catalog sync.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """List curated ShardAi catalog models and estimated RAM requirements."""
    _apply_offline_env(offline)
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    source_label = "default"
    model_items: list[dict[str, object]] = []

    if _check_daemon_healthy(base_url):
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=8.0, connect=2.0),
                base_url=base_url,
            ) as client:
                resp = client.get(f"{base_url}/models/catalog")
                if resp.status_code >= 400:
                    _fail(
                        extract_http_error_detail(
                            resp,
                            f"Failed to fetch model catalog (HTTP {resp.status_code}).",
                        )
                    )
                payload = resp.json()
                if isinstance(payload, dict):
                    source_label = str(payload.get("source") or "default")
                    raw_models = payload.get("models")
                    if isinstance(raw_models, list):
                        model_items = [
                            dict(item) for item in raw_models if isinstance(item, dict)
                        ]
        except (httpx.HTTPError, OSError) as exc:
            _fail(f"Failed to fetch models from {base_url}: {exc}")
    else:
        downloader = ModelDownloader()
        if not is_offline_mode():
            tracker = SignalingTrackerClient(
                base_url=os.getenv("SHARDAI_TRACKER_URL", "https://network.shardai.online"),
                timeout_seconds=5.0,
                connect_timeout_seconds=3.0,
                max_retries=1,
            )
            try:
                cloud_models = asyncio.run(tracker.fetch_model_catalog())
                if cloud_models:
                    model_items = [dict(item) for item in cloud_models]
                    source_label = "cloud"
            except Exception:
                pass
        if not model_items:
            cached = downloader.load_catalog_cache()
            if cached:
                model_items = [dict(item) for item in cached]
                source_label = "cache"
            else:
                model_items = [dict(item) for item in DEFAULT_CATALOG_MODELS]
                source_label = "default"

    if json_output:
        _emit_json(
            {
                "source": source_label,
                "count": len(model_items),
                "models": model_items,
            }
        )
        return

    typer.echo(f"Catalog source: {source_label} ({len(model_items)} models)\n")
    rows: list[list[str]] = [
        [
            str(item.get("id") or ""),
            str(item.get("name") or ""),
            str(item.get("repo_id") or ""),
            f"{float(item.get('estimated_ram_gb') or 5.0):.1f} GB",
            str(item.get("filename") or "-"),
        ]
        for item in model_items
    ]
    typer.echo(format_table(("ID", "NAME", "REPO ID", "EST. RAM", "FILENAME"), rows))


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def pull(
    model_arg: str | None = typer.Argument(
        None,
        help="Catalog ID, HuggingFace repo/URL, direct .gguf URL, or local .gguf path.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model identifier (alternative to positional MODEL_ARG).",
    ),
    source_type: str = typer.Option(
        "auto",
        "--source-type",
        help="Model source: auto, catalog, huggingface, direct_url, or local_file.",
    ),
    filename: str | None = typer.Option(
        None,
        "--filename",
        "-f",
        help="Optional specific .gguf filename in a HuggingFace repo.",
    ),
    ram: int | None = typer.Option(
        None,
        "--ram",
        min=2,
        max=256,
        help="Optional RAM budget (GB) for selective Safetensors layer download.",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for this download.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Download and cache a model (or assigned shard) before starting a node."""
    raw_model = (model or model_arg or "").strip()
    if not raw_model:
        _fail("Specify a model to pull (e.g. 'shardai pull llama-3.1-8b').")

    _apply_offline_env(offline)
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    daemon_running = _check_daemon_healthy(base_url)
    catalog_entries = _fetch_catalog_entries_best_effort(
        base_url if daemon_running else None
    )
    resolved_model, resolved_source, resolved_filename = resolve_model_spec(
        model=raw_model,
        source_type=source_type,
        filename=filename,
        catalog_entries=catalog_entries,
    )

    if daemon_running:
        params: dict[str, str | int] = {
            "model_id": resolved_model,
            "source_type": resolved_source.value,
        }
        if resolved_filename is not None:
            params["filename"] = resolved_filename
        if ram is not None:
            params["ram_gb"] = ram

        last_event: dict[str, object] = {}
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=300.0, connect=4.0),
                base_url=base_url,
            ) as client:
                with client.stream(
                    "GET", f"{base_url}/models/download-stream", params=params
                ) as resp:
                    if resp.status_code >= 400:
                        resp.read()
                        _fail(
                            extract_http_error_detail(
                                resp,
                                f"Model pull failed (HTTP {resp.status_code}).",
                            )
                        )
                    for raw_line in resp.iter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            break
                        evt_obj = json.loads(data_str)
                        if not isinstance(evt_obj, dict):
                            continue
                        last_event = dict(evt_obj)
                        if evt_obj.get("status") == "error":
                            _fail(str(evt_obj.get("message") or "Model download failed."))
                        if not json_output:
                            pct = float(evt_obj.get("progress") or 0.0)
                            msg = str(evt_obj.get("message") or evt_obj.get("status") or "")
                            typer.echo(f"[{pct:5.1f}%] {msg}")
        except (httpx.HTTPError, OSError) as exc:
            _fail(f"Failed to pull model via daemon at {base_url}: {exc}")

        if json_output:
            _emit_json(last_event)
        return

    events: list[DownloadProgressEvent] = []

    def _on_evt(evt: DownloadProgressEvent) -> None:
        events.append(evt)
        if not json_output:
            typer.echo(f"[{evt.progress_percent:5.1f}%] {evt.message or evt.status}")

    try:
        local_path = asyncio.run(
            ModelDownloader().ensure_model_downloaded(
                model_id=resolved_model,
                on_event=_on_evt,
                source_type=resolved_source.value,
                filename=resolved_filename,
            )
        )
    except Exception as exc:
        _fail(f"Failed to pull model '{resolved_model}': {exc}")

    if json_output:
        final_evt = events[-1] if events else None
        _emit_json(
            {
                "model_id": resolved_model,
                "source_type": resolved_source.value,
                "local_path": str(local_path),
                "status": final_evt.status if final_evt else "completed",
                "progress": final_evt.progress_percent if final_evt else 100.0,
            }
        )


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def peers(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """List direct P2P routing peers currently known to the local node."""
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=5.0, connect=2.0),
            base_url=base_url,
        ) as client:
            resp = client.get(f"{base_url}/peers")
            if resp.status_code >= 400:
                _fail(
                    extract_http_error_detail(
                        resp, f"Failed to fetch peers (HTTP {resp.status_code})."
                    )
                )
            raw = resp.json()
            peer_items = (
                [dict(item) for item in raw if isinstance(item, dict)]
                if isinstance(raw, list)
                else []
            )
    except (httpx.HTTPError, OSError) as exc:
        _fail(f"Local ShardAi daemon is not reachable at {base_url}: {exc}")

    if json_output:
        _emit_json(peer_items)
        return

    if not peer_items:
        typer.echo("No connected peers.")
        return

    rows: list[list[str]] = [
        [
            str(item.get("peer_id") or ""),
            f"{item.get('ip_address')}:{item.get('port')}",
            f"{item.get('shard_layer_start')}..{item.get('shard_layer_end')}",
            str(item.get("role") or "primary").upper(),
            str(item.get("room_id") or "public"),
        ]
        for item in peer_items
    ]
    typer.echo(format_table(("PEER ID", "ADDRESS", "LAYERS", "ROLE", "ROOM"), rows))


def _send_chat_request(
    client: httpx.Client,
    gateway_base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    print_tokens: bool,
) -> tuple[str, dict[str, object] | None]:
    """Send one `/v1/chat/completions` turn (SSE streaming or JSON) and return `(reply, raw_json)`."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    endpoint = f"{gateway_base_url.rstrip('/')}/chat/completions"

    if stream:
        collected: list[str] = []
        with client.stream("POST", endpoint, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                _fail(
                    extract_http_error_detail(
                        resp, f"Chat completion failed (HTTP {resp.status_code})."
                    )
                )
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if not isinstance(chunk, dict):
                    continue
                if isinstance(chunk.get("error"), dict):
                    err_msg = str(
                        chunk["error"].get("message") or "Streaming inference error."
                    )
                    _fail(err_msg)
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        delta = first.get("delta")
                        if isinstance(delta, dict):
                            token = delta.get("content")
                            if isinstance(token, str) and token:
                                collected.append(token)
                                if print_tokens:
                                    typer.echo(token, nl=False)
        if print_tokens:
            typer.echo("")
        return "".join(collected), None

    resp = client.post(endpoint, headers=headers, json=body)
    if resp.status_code >= 400:
        _fail(
            extract_http_error_detail(
                resp, f"Chat completion failed (HTTP {resp.status_code})."
            )
        )
    payload_obj = resp.json()
    if not isinstance(payload_obj, dict):
        _fail("Unexpected JSON payload from /v1/chat/completions.")
    content = ""
    choices = payload_obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            content = msg["content"]
    if print_tokens:
        typer.echo(content)
    return content, dict(payload_obj)


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def chat(
    prompt_arg: str | None = typer.Argument(
        None,
        help="One-shot prompt message (omit for interactive multi-turn chat).",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="One-shot prompt text.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model identifier (auto-detected from active node status when omitted).",
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        help="Optional system prompt prepended to the conversation.",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        min=1,
        max=4096,
        help="Maximum generated tokens per turn.",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        min=0.0,
        max=2.0,
        help="Sampling temperature (0.0-2.0).",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream assistant tokens over SSE (default enabled).",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Bearer session token (auto-discovered from local /status when omitted).",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON (one-shot prompt).",
    ),
) -> None:
    """Chat with the active ShardAi node via the local OpenAI-compatible gateway."""
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    effective_token = api_key.strip() if api_key and api_key.strip() else None
    effective_model = model.strip() if model and model.strip() else None
    gateway_v1_url = f"{base_url}/v1"

    try:
        with _make_http_client(
            timeout=httpx.Timeout(timeout=120.0, connect=3.0),
            base_url=base_url,
        ) as client:
            if effective_token is None or effective_model is None:
                try:
                    status_resp = client.get(f"{base_url}/status")
                except (httpx.HTTPError, OSError) as exc:
                    _fail(
                        f"Local ShardAi daemon is not reachable at {base_url}: {exc}"
                    )
                if status_resp.status_code >= 400:
                    _fail(
                        extract_http_error_detail(
                            status_resp,
                            f"Failed to query node status (HTTP {status_resp.status_code}).",
                        )
                    )
                status_data = status_resp.json()
                if not isinstance(status_data, dict):
                    _fail("Unexpected /status payload from local daemon.")
                if status_data.get("state") != "RUNNING":
                    _fail(
                        "ShardAi node is not RUNNING. Start one with 'shardai create --model ...' or 'shardai join'."
                    )
                if not status_data.get("is_swarm_ready"):
                    cov = float(status_data.get("coverage_percent") or 0.0)
                    _fail(
                        f"Model not ready (swarm coverage {cov:.0f}% < 100%). Wait for more peers or restart with --ssd-offload."
                    )
                if effective_token is None:
                    raw_tok = status_data.get("session_token")
                    if not isinstance(raw_tok, str) or not raw_tok.strip():
                        _fail("No active session API key found on running node.")
                    effective_token = raw_tok.strip()
                if effective_model is None:
                    raw_mod = status_data.get("model_id")
                    effective_model = (
                        raw_mod.strip()
                        if isinstance(raw_mod, str) and raw_mod.strip()
                        else DEFAULT_MODEL
                    )
                raw_api = status_data.get("api_base_url")
                if isinstance(raw_api, str) and raw_api.strip():
                    try:
                        gateway_v1_url = _validate_loopback_url(
                            raw_api, label="/status api_base_url"
                        )
                    except typer.BadParameter as exc:
                        _fail(f"Refusing untrusted /status api_base_url: {exc}")

            assert effective_token is not None
            assert effective_model is not None

            history: list[dict[str, str]] = []
            if system is not None and system.strip():
                history.append({"role": "system", "content": system.strip()})

            one_shot = (
                prompt.strip()
                if prompt is not None and prompt.strip()
                else (prompt_arg.strip() if prompt_arg is not None and prompt_arg.strip() else None)
            )
            if one_shot is not None:
                history.append({"role": "user", "content": one_shot})
                reply_text, raw_json = _send_chat_request(
                    client=client,
                    gateway_base_url=gateway_v1_url,
                    api_key=effective_token,
                    model_id=effective_model,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=(stream and not json_output),
                    print_tokens=(not json_output),
                )
                if json_output:
                    _emit_json(
                        raw_json
                        if raw_json is not None
                        else {
                            "model": effective_model,
                            "content": reply_text,
                        }
                    )
                return

            typer.echo(
                f"Connected to {effective_model} at {gateway_v1_url} "
                "(type /clear to reset, /exit or /quit to leave)."
            )
            while True:
                try:
                    user_line = input("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    typer.echo("")
                    break
                if not user_line:
                    continue
                if user_line.lower() in ("/exit", "/quit"):
                    break
                if user_line.lower() == "/clear":
                    history = (
                        [{"role": "system", "content": system.strip()}]
                        if system is not None and system.strip()
                        else []
                    )
                    typer.echo("Conversation cleared.")
                    continue

                history.append({"role": "user", "content": user_line})
                typer.echo("Assistant> ", nl=False)
                reply_text, _ = _send_chat_request(
                    client=client,
                    gateway_base_url=gateway_v1_url,
                    api_key=effective_token,
                    model_id=effective_model,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    print_tokens=True,
                )
                history.append({"role": "assistant", "content": reply_text})
    except (httpx.HTTPError, OSError) as exc:
        _fail(f"Chat request failed against {gateway_v1_url}: {exc}")


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def network(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Override SHARDAI_OFFLINE_MODE for connectivity probing.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Check Signaling Tracker reachability, LAN peer count, and host RAM headroom."""
    _apply_offline_env(offline)
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    net_payload: dict[str, object]
    mem_payload: dict[str, object]

    if _check_daemon_healthy(base_url):
        try:
            with _make_http_client(
                timeout=httpx.Timeout(timeout=6.0, connect=2.0),
                base_url=base_url,
            ) as client:
                net_resp = client.get(f"{base_url}/network-status")
                if net_resp.status_code >= 400:
                    _fail(
                        extract_http_error_detail(
                            net_resp,
                            f"Network status probe failed (HTTP {net_resp.status_code}).",
                        )
                    )
                net_payload = dict(net_resp.json())
                mem_resp = client.get(f"{base_url}/system/memory")
                if mem_resp.status_code == 200 and isinstance(mem_resp.json(), dict):
                    mem_payload = dict(mem_resp.json())
                else:
                    total_gb, avail_gb, max_share = get_system_memory_info()
                    mem_payload = {
                        "total_ram_gb": total_gb,
                        "available_ram_gb": avail_gb,
                        "max_shareable_ram_gb": max_share,
                    }
        except (httpx.HTTPError, OSError) as exc:
            _fail(f"Failed to probe network status from {base_url}: {exc}")
    else:
        tracker_url = os.getenv("SHARDAI_TRACKER_URL", "https://network.shardai.online")
        tracker_ok = False
        if not is_offline_mode():
            tracker = SignalingTrackerClient(base_url=tracker_url)
            tracker_ok = asyncio.run(tracker.ping(timeout_seconds=3.0))
        effective = "online" if tracker_ok else "offline_solo"
        net_payload = {
            "network_status": effective,
            "tracker_reachable": tracker_ok,
            "tracker_url": tracker_url,
            "lan_peers_count": 0,
            "zeroconf_available": False,
        }
        total_gb, avail_gb, max_share = get_system_memory_info()
        mem_payload = {
            "total_ram_gb": total_gb,
            "available_ram_gb": avail_gb,
            "max_shareable_ram_gb": max_share,
        }

    if json_output:
        _emit_json({"network": net_payload, "memory": mem_payload})
        return

    typer.echo(f"Network Tier     : {net_payload.get('network_status')}")
    typer.echo(f"Tracker Reachable: {net_payload.get('tracker_reachable')}")
    typer.echo(f"Tracker URL      : {net_payload.get('tracker_url')}")
    typer.echo(f"LAN Peers        : {net_payload.get('lan_peers_count', 0)}")
    typer.echo(
        f"System Memory    : {mem_payload.get('available_ram_gb')} GB free / "
        f"{mem_payload.get('total_ram_gb')} GB total "
        f"(max shareable: {mem_payload.get('max_shareable_ram_gb')} GB)"
    )


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def doctor(
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Local loopback daemon host.",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Local daemon HTTP port.",
    ),
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        help="Override daemon base URL.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON diagnostics.",
    ),
) -> None:
    """Inspect Linux/Ubuntu runtime readiness, memory/cgroup limits, GPU, and daemon status."""
    base_url = build_engine_base_url(host=host, port=port, api_url=api_url)
    parsed = urlparse(base_url)
    diag_host = parsed.hostname or validate_loopback_host(host)
    diag_port = parsed.port or port
    daemon_reachable = _check_daemon_healthy(base_url)
    diag = collect_environment_diagnostics(
        host=diag_host,
        port=diag_port,
        daemon_reachable=daemon_reachable,
    )

    if json_output:
        _emit_json(diag)
        if not diag.get("ok"):
            raise typer.Exit(code=1)
        return

    distro_obj = diag.get("distro")
    distro_dict = distro_obj if isinstance(distro_obj, dict) else {}
    accel_obj = diag.get("accelerator")
    accel_dict = accel_obj if isinstance(accel_obj, dict) else {}
    mem_obj = diag.get("memory")
    mem_dict = mem_obj if isinstance(mem_obj, dict) else {}

    typer.echo(
        f"Environment   : {distro_dict.get('pretty_name') or diag.get('platform')} "
        f"({diag.get('target_triple')})"
        + (" [WSL]" if distro_dict.get("is_wsl") else "")
    )
    typer.echo(
        f"Accelerator   : {str(accel_dict.get('backend') or 'cpu').upper()}"
        + (f" ({accel_dict.get('device_name')})" if accel_dict.get("device_name") else "")
    )
    if mem_dict.get("cgroup_limit_bytes") is not None:
        limit_gb = float(mem_dict["cgroup_limit_bytes"]) / float(1024**3)  # type: ignore[arg-type]
        typer.echo(f"Cgroup Limit  : {limit_gb:.2f} GB")

    checks_obj = diag.get("checks")
    rows: list[list[str]] = []
    if isinstance(checks_obj, list):
        for item in checks_obj:
            if isinstance(item, dict):
                rows.append(
                    [
                        str(item.get("name") or ""),
                        str(item.get("status") or "").upper(),
                        str(item.get("detail") or ""),
                    ]
                )
    if rows:
        typer.echo("")
        typer.echo(format_table(("CHECK", "STATUS", "DETAIL"), rows))

    if not diag.get("ok"):
        raise typer.Exit(code=1)


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def service(
    action: str = typer.Argument(
        ...,
        help="Systemd action: unit, install, uninstall, status, start, stop, restart, or logs.",
    ),
    scope: str = typer.Option(
        "user",
        "--scope",
        help="Systemd unit scope: user (~/.config/systemd/user) or system (/etc/systemd/system).",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host",
        "-H",
        help="Loopback daemon host for unit ExecStart (127.0.0.1, localhost, or ::1).",
    ),
    port: int = typer.Option(
        int(os.getenv("SHARDAI_ENGINE_PORT", str(DEFAULT_PORT))),
        "--port",
        "-p",
        min=1024,
        max=65535,
        help="Loopback daemon HTTP port.",
    ),
    tracker_url: str | None = typer.Option(
        None,
        "--tracker-url",
        help="Optional SHARDAI_TRACKER_URL environment variable for the service unit.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--online",
        help="Configure SHARDAI_OFFLINE_MODE=true inside the generated service unit.",
    ),
    enable: bool = typer.Option(
        True,
        "--enable/--no-enable",
        help="Enable and immediately start shardai.service on install.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Generate or manage a hardened loopback-only Linux systemd service unit."""
    safe_host = validate_loopback_host(host)
    normalized_action = action.strip().lower()
    if normalized_action not in SERVICE_ACTIONS:
        _fail(
            f"Unsupported service action '{action}'. "
            f"Choose one of: {', '.join(SERVICE_ACTIONS)}."
        )

    normalized_scope = scope.strip().lower()
    if normalized_scope not in ("user", "system"):
        _fail(f"Invalid --scope '{scope}'. Choose 'user' or 'system'.")
    typed_scope: Literal["user", "system"] = (
        "system" if normalized_scope == "system" else "user"
    )

    resolved_bin = shutil.which("shardai")
    if resolved_bin:
        exec_path = resolved_bin
    elif getattr(sys, "frozen", False):
        exec_path = sys.executable
    else:
        exec_path = "/usr/bin/shardai" if platform.system() == "Linux" else sys.executable

    unit_path = get_systemd_unit_path(typed_scope)
    if normalized_action == "unit":
        unit_text = render_systemd_unit(
            exec_path=exec_path,
            host=safe_host,
            port=port,
            scope=typed_scope,
            tracker_url=tracker_url,
            offline=bool(offline),
        )
        if json_output:
            _emit_json(
                {
                    "scope": typed_scope,
                    "unit_path": str(unit_path),
                    "content": unit_text,
                }
            )
            return
        typer.echo(unit_text, nl=False)
        return

    if platform.system() != "Linux":
        _fail(
            "systemd service management is only available on Linux. "
            "Run 'shardai service unit' to preview the unit template."
        )

    if shutil.which("systemctl") is None:
        _fail("systemctl was not found in PATH on this Linux host.")

    if normalized_action == "install":
        unit_text = render_systemd_unit(
            exec_path=exec_path,
            host=safe_host,
            port=port,
            scope=typed_scope,
            tracker_url=tracker_url,
            offline=bool(offline),
        )
        try:
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(unit_text, encoding="utf-8", newline="\n")
        except PermissionError as exc:
            _fail(f"Permission denied writing {unit_path}: {exc}")
        reload_res = run_systemctl(["daemon-reload"], scope=typed_scope)
        if reload_res.returncode != 0:
            _fail(reload_res.stderr.strip() or "systemctl daemon-reload failed.")
        if enable:
            enable_res = run_systemctl(["enable", "--now", "shardai.service"], scope=typed_scope)
            if enable_res.returncode != 0:
                _fail(enable_res.stderr.strip() or "systemctl enable --now failed.")
        if json_output:
            _emit_json({"action": "install", "scope": typed_scope, "unit_path": str(unit_path), "enabled": enable})
        else:
            typer.echo(f"Installed systemd unit at {unit_path}" + (" and started shardai.service." if enable else "."))
        return

    if normalized_action == "uninstall":
        run_systemctl(["disable", "--now", "shardai.service"], scope=typed_scope)
        try:
            unit_path.unlink(missing_ok=True)
        except PermissionError as exc:
            _fail(f"Permission denied removing {unit_path}: {exc}")
        run_systemctl(["daemon-reload"], scope=typed_scope)
        if json_output:
            _emit_json({"action": "uninstall", "scope": typed_scope, "unit_path": str(unit_path)})
        else:
            typer.echo(f"Removed systemd unit {unit_path}.")
        return

    if normalized_action == "logs":
        if shutil.which("journalctl") is None:
            _fail("journalctl was not found in PATH.")
        jcmd = ["journalctl"]
        if typed_scope == "user":
            jcmd.append("--user")
        jcmd.extend(["-u", "shardai.service", "-n", "100", "--no-pager"])
        proc = subprocess.run(jcmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            _fail(proc.stderr.strip() or "Failed to read shardai.service journal logs.")
        typer.echo(proc.stdout, nl=False)
        return

    res = run_systemctl([normalized_action, "shardai.service"], scope=typed_scope)
    if json_output:
        _emit_json(
            {
                "action": normalized_action,
                "scope": typed_scope,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        )
        if res.returncode != 0:
            raise typer.Exit(code=res.returncode)
        return

    if res.stdout:
        typer.echo(res.stdout, nl=False)
    if res.returncode != 0:
        _fail(res.stderr.strip() or f"systemctl {normalized_action} exited with code {res.returncode}.")


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def completion(
    shell: str = typer.Argument(
        ...,
        help="Target shell completion format: bash or zsh.",
    ),
) -> None:
    """Output Bash or Zsh shell completion script for the `shardai` CLI."""
    normalized_shell = shell.strip().lower()
    if normalized_shell not in ("bash", "zsh"):
        _fail(f"Unsupported shell '{shell}'. Choose 'bash' or 'zsh'.")
    typed_shell: Literal["bash", "zsh"] = "zsh" if normalized_shell == "zsh" else "bash"
    typer.echo(render_shell_completion(typed_shell), nl=False)


@cli_app.command(context_settings=_COMMON_CONTEXT_SETTINGS)
def version(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output machine-readable JSON.",
    ),
) -> None:
    """Print the installed ShardAi CLI version."""
    if json_output:
        _emit_json({"name": "shardai", "version": CLI_VERSION})
        return
    typer.echo(f"shardai {CLI_VERSION}")


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    default_no_subcommand: Literal["help", "daemon"] = "help",
    engine_module: types.ModuleType | None = None,
) -> None:
    """Invoke the ShardAi CLI with single-dash long-flag normalization."""
    global _default_no_subcommand_action, _registered_engine_module
    _default_no_subcommand_action = default_no_subcommand
    if engine_module is not None:
        _registered_engine_module = engine_module

    raw_args = list(sys.argv[1:] if argv is None else argv)
    normalized = normalize_cli_argv(raw_args)
    prog_name = "core-engine" if default_no_subcommand == "daemon" else "shardai"
    cli_app(args=normalized, prog_name=prog_name)
