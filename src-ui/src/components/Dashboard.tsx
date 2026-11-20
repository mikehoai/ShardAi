import React, { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open as openFileDialog } from "@tauri-apps/plugin-dialog";
import type {
  CatalogModelItem,
  ModelCatalogResponse,
  ModelSourceType,
  NetworkMode,
  NetworkStatus,
  NetworkStatusResponse,
  NodeStatusResponse,
  PeerRole,
  RoomLobbyItem,
  StartNodeRequest,
  SystemMemoryResponse,
} from "../types/node";
import {
  checkRoomJoin,
  extractApiErrorDetail,
  RoomJoinRejectedError,
} from "../services/engineApi";
import {
  inferCustomSourceType,
  ModelSelector,
  resolveModelSelection,
  type CatalogSource,
  type ModelTab,
} from "./ModelSelector";

const LOCAL_ENGINE_ROOT = "http://127.0.0.1:8000";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/v1";
const MIN_RAM_GB = 2;
const DEFAULT_MAX_RAM_GB = 16;
const NETWORK_STATUS_POLL_MS = 10000;
const GITHUB_REPO_URL = "https://github.com/mikehoai/ShardAi";

type StatusDotTone = "green" | "yellow" | "red" | "neutral";

const NETWORK_STATUS_META: Record<
  NetworkStatus,
  { tone: Exclude<StatusDotTone, "neutral">; label: string }
> = {
  online: {
    tone: "green",
    label: "Online (Cloud)",
  },
  lan_only: {
    tone: "yellow",
    label: "LAN Only",
  },
  offline_solo: {
    tone: "red",
    label: "Offline (Solo)",
  },
};

type ThemeMode = "light" | "dark";
type NavSection = "overview" | "node" | "rooms" | "credentials";

const MAX_COMPACT_ROOM_NAME_LEN = 18;
const MODEL_PREFIX_PATTERNS: readonly RegExp[] = [
  /^(?:deepseek[-_](?:r1|v\d+|llama|qwen)(?:[-_]\d+b)?[-_]+)/i,
  /^(?:qwen(?:\d+(?:\.\d+)?)?[-_](?:coder[-_])?(?:\d+b[-_]+)?)/i,
  /^(?:(?:meta[-_])?llama(?:\d+(?:\.\d+)?)?[-_](?:\d+b[-_]+)?)/i,
  /^(?:mistral|mixtral|gemma|phi)(?:\d+(?:\.\d+)?)?[-_](?:\d+b[-_]+)?/i,
];

export function createRoomSessionSeed(): number {
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    const buf = new Uint32Array(1);
    crypto.getRandomValues(buf);
    return buf[0] || 1;
  }
  return Math.max(1, Math.floor(Math.random() * 0xffffffff));
}

function hashRoomForSeed(roomId: string, seed: number): number {
  let hash = (seed ^ 0x811c9dc5) >>> 0;
  for (let i = 0; i < roomId.length; i += 1) {
    hash ^= roomId.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export function orderRoomsForSession(
  rooms: readonly RoomLobbyItem[],
  sessionSeed: number
): RoomLobbyItem[] {
  if (rooms.length <= 1) {
    return [...rooms];
  }
  return [...rooms].sort((a, b) => {
    const scoreA = hashRoomForSeed(a.room_id, sessionSeed);
    const scoreB = hashRoomForSeed(b.room_id, sessionSeed);
    if (scoreA !== scoreB) {
      return scoreA - scoreB;
    }
    return a.room_id.localeCompare(b.room_id);
  });
}

export function formatCompactRoomName(roomId: string): string {
  const trimmed = roomId.trim();
  if (!trimmed) {
    return trimmed;
  }
  let compact = trimmed;
  for (const pattern of MODEL_PREFIX_PATTERNS) {
    const stripped = compact.replace(pattern, "");
    if (stripped.length >= 2 && stripped !== compact) {
      compact = stripped;
      break;
    }
  }
  if (compact.length > MAX_COMPACT_ROOM_NAME_LEN) {
    return `${compact.slice(0, MAX_COMPACT_ROOM_NAME_LEN - 1)}…`;
  }
  return compact;
}

export function formatCompactModelName(model: string): string {
  const trimmed = model.trim();
  if (!trimmed) {
    return trimmed;
  }
  const withoutOrg = trimmed.includes("/") ? trimmed.slice(trimmed.lastIndexOf("/") + 1) : trimmed;
  const cleaned = withoutOrg
    .replace(/[-_]?gguf$/i, "")
    .replace(/[-_]?instruct$/i, "")
    .replace(/[-_]?distill(?=[-_])/i, "")
    .replace(/^meta[-_]/i, "");
  return cleaned || withoutOrg;
}

function resolveInitialTheme(): ThemeMode {
  try {
    const saved = window.localStorage.getItem("shardai_theme");
    if (saved === "light" || saved === "dark") {
      return saved;
    }
  } catch {
    // Ignore storage access errors in restricted environments
  }
  if (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }
  return "light";
}

interface SseDownloadEvent {
  model_id: string;
  status: string;
  progress: number;
  downloaded_bytes: number;
  total_bytes: number;
  local_path: string | null;
  message: string;
}

interface StartConfigOverride {
  model: string;
  sourceType?: ModelSourceType;
  filename?: string | null;
  room: string;
  key: string;
  unlisted: boolean;
  mode: NetworkMode;
}

const isTauriEnvironment = (): boolean => {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)
  );
};

async function invokeTauriCommand<T>(
  cmd: string,
  args?: Record<string, unknown>
): Promise<T> {
  if (isTauriEnvironment()) {
    return await invoke<T>(cmd, args);
  }
  if (cmd === "start_python_core") {
    return "[Browser Preview Mode] Core engine signal ready" as unknown as T;
  }
  if (cmd === "stop_python_core") {
    return "[Browser Preview Mode] Core engine signal stopped" as unknown as T;
  }
  return "" as unknown as T;
}

export const Dashboard: React.FC = () => {
  // Native Desktop Shell State (Navigation + OS/User Theme)
  const [activeSection, setActiveSection] = useState<NavSection>("overview");
  const [isSidebarHidden, setIsSidebarHidden] = useState<boolean>(() => {
    try {
      return window.localStorage.getItem("shardai_sidebar_hidden") === "true";
    } catch {
      return false;
    }
  });
  const [themeMode, setThemeMode] = useState<ThemeMode>(resolveInitialTheme);
  const [isMaximized, setIsMaximized] = useState<boolean>(false);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        "shardai_sidebar_hidden",
        isSidebarHidden ? "true" : "false"
      );
    } catch {
      // Ignore storage write errors
    }
  }, [isSidebarHidden]);

  const toggleSidebar = useCallback((): void => {
    setIsSidebarHidden((prev) => !prev);
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) {
        return;
      }
      if (event.key.toLowerCase() !== "b") {
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      event.preventDefault();
      toggleSidebar();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [toggleSidebar]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", themeMode);
    try {
      window.localStorage.setItem("shardai_theme", themeMode);
    } catch {
      // Ignore storage write errors
    }
  }, [themeMode]);

  const toggleThemeMode = (): void => {
    setThemeMode((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Network Rooms Lobby State
  const roomSessionSeedRef = useRef<number>(createRoomSessionSeed());
  const [availableRooms, setAvailableRooms] = useState<RoomLobbyItem[]>([]);
  const [isLoadingRooms, setIsLoadingRooms] = useState<boolean>(false);

  // Private Room Key Modal State (when clicking Join on a locked room)
  const [pendingPrivateRoom, setPendingPrivateRoom] = useState<RoomLobbyItem | null>(null);
  const [privateKeyInput, setPrivateKeyInput] = useState<string>("");

  // Create New Room / Join Unlisted Modal State
  const [showRoomModal, setShowRoomModal] = useState<boolean>(false);
  const [modalRoomId, setModalRoomId] = useState<string>("studio-room-1");
  const [modalModelTab, setModalModelTab] = useState<ModelTab>("catalog");
  const [modalSelectedCatalogId, setModalSelectedCatalogId] = useState<string>("");
  const [modalModel, setModalModel] = useState<string>("meta-llama/Llama-3.1-8B-Instruct");
  const [modalCustomFilename, setModalCustomFilename] = useState<string>("");
  const [modalLocalFilePath, setModalLocalFilePath] = useState<string>("");
  const [modalRoomKey, setModalRoomKey] = useState<string>("");
  const [modalUnlisted, setModalUnlisted] = useState<boolean>(false);
  const [invitePasteInput, setInvitePasteInput] = useState<string>("");

  // 3-Tab Model Selection State (Online Catalog / Custom HF or URL / Local File)
  const [modelTab, setModelTab] = useState<ModelTab>("catalog");
  const [catalogModels, setCatalogModels] = useState<CatalogModelItem[]>([]);
  const [catalogSource, setCatalogSource] = useState<CatalogSource | null>(null);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState<boolean>(false);
  const [selectedCatalogId, setSelectedCatalogId] = useState<string>("");
  const [modelInput, setModelInput] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
  const [customFilename, setCustomFilename] = useState<string>("");
  const [localFilePath, setLocalFilePath] = useState<string>("");
  const [networkMode, setNetworkMode] = useState<NetworkMode>("internet");
  const [roomId, setRoomId] = useState<string>("public");
  const [roomKey, setRoomKey] = useState<string>("");
  const [isUnlisted, setIsUnlisted] = useState<boolean>(false);
  const [availableRamGb, setAvailableRamGb] = useState<number | null>(null);
  const [maxRamGb, setMaxRamGb] = useState<number>(DEFAULT_MAX_RAM_GB);
  const [sharedRamGb, setSharedRamGb] = useState<number>(4);
  const [allowSsdOffload, setAllowSsdOffload] = useState<boolean>(false);

  // Offline / Solo Mode connectivity (null = engine not reachable yet)
  const [networkStatus, setNetworkStatus] = useState<NetworkStatus | null>(null);
  const [lanPeersCount, setLanPeersCount] = useState<number>(0);
  const [ssdOffloadActive, setSsdOffloadActive] = useState<boolean>(false);
  const [modelFingerprint, setModelFingerprint] = useState<string | null>(null);

  // Node Lifecycle & Gateway Credentials State
  const [isSharing, setIsSharing] = useState<boolean>(false);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [sessionToken, setSessionToken] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(DEFAULT_API_BASE_URL);
  const [activeModelId, setActiveModelId] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
  const [assignedRole, setAssignedRole] = useState<PeerRole>("primary");
  const [assignedLayerRange, setAssignedLayerRange] = useState<[number, number] | null>(
    null
  );
  const [coveragePercent, setCoveragePercent] = useState<number>(0);
  const [isSwarmReady, setIsSwarmReady] = useState<boolean>(false);
  const [connectedPeersCount, setConnectedPeersCount] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Idle — Select a room from the Lobby or configure your own node below."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Model Download Progress Bar State
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [downloadStatusMsg, setDownloadStatusMsg] = useState<string>("");

  // Clipboard copy feedback
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const fetchAvailableRooms = useCallback(
    async (options?: { reseed?: boolean }): Promise<void> => {
      setIsLoadingRooms(true);
      if (options?.reseed) {
        roomSessionSeedRef.current = createRoomSessionSeed();
      }
      try {
        const res = await fetch(`${LOCAL_ENGINE_ROOT}/available-rooms`);
        if (res.ok) {
          const rooms = (await res.json()) as unknown;
          if (Array.isArray(rooms)) {
            setAvailableRooms(
              orderRoomsForSession(
                rooms as RoomLobbyItem[],
                roomSessionSeedRef.current
              )
            );
          }
        }
      } catch {
        // Keep existing lobby state if local core engine is not yet running
      } finally {
        setIsLoadingRooms(false);
      }
    },
    []
  );

  const engineSyncedRef = useRef<boolean>(false);

  const fetchNetworkStatus = useCallback(async (): Promise<boolean> => {
    try {
      const res = await fetch(`${LOCAL_ENGINE_ROOT}/network-status`);
      if (res.ok) {
        const data = (await res.json()) as Partial<NetworkStatusResponse>;
        if (
          data.network_status === "online" ||
          data.network_status === "lan_only" ||
          data.network_status === "offline_solo"
        ) {
          setNetworkStatus(data.network_status);
          setLanPeersCount(
            typeof data.lan_peers_count === "number" ? data.lan_peers_count : 0
          );
          return true;
        }
      }
    } catch {
      // Engine still booting/unpacking; keep waiting silently
    }
    return false;
  }, []);

  const fetchModelCatalog = useCallback(async (): Promise<void> => {
    setIsLoadingCatalog(true);
    try {
      const res = await fetch(`${LOCAL_ENGINE_ROOT}/models/catalog`);
      if (res.ok) {
        const data = (await res.json()) as Partial<ModelCatalogResponse>;
        if (Array.isArray(data.models) && data.models.length > 0) {
          setCatalogModels(data.models);
          if (
            data.source === "cloud" ||
            data.source === "cache" ||
            data.source === "default"
          ) {
            setCatalogSource(data.source);
          }
          setSelectedCatalogId((prev) => {
            if (prev && data.models?.some((m) => m.id === prev)) {
              return prev;
            }
            const first = data.models?.[0];
            if (first) {
              setModelInput(first.repo_id);
              return first.id;
            }
            return prev;
          });
          setModalSelectedCatalogId((prev) => {
            if (prev && data.models?.some((m) => m.id === prev)) {
              return prev;
            }
            const first = data.models?.[0];
            if (first) {
              setModalModel(first.repo_id);
              return first.id;
            }
            return prev;
          });
        }
      }
    } catch {
      // Engine may not be reachable yet
    } finally {
      setIsLoadingCatalog(false);
    }
  }, []);

  // Network Status Indicator: fast 2s poll while engine is unpacking, 10s once connected
  useEffect(() => {
    let cancelled = false;
    const pollMs = networkStatus === null ? 2000 : NETWORK_STATUS_POLL_MS;

    const tick = async (): Promise<void> => {
      const connected = await fetchNetworkStatus();
      if (!cancelled && connected && !engineSyncedRef.current) {
        engineSyncedRef.current = true;
        void Promise.all([fetchAvailableRooms(), fetchModelCatalog()]);
      }
    };

    void tick();
    const timerId = window.setInterval(() => {
      void tick();
    }, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(timerId);
    };
  }, [networkStatus, fetchNetworkStatus, fetchAvailableRooms, fetchModelCatalog]);

  const applyNodeNetworkFields = (statusData: NodeStatusResponse): void => {
    if (statusData.network_status) {
      setNetworkStatus(statusData.network_status);
    }
    setSsdOffloadActive(Boolean(statusData.ssd_offload_active));
    setModelFingerprint(statusData.model_fingerprint ?? null);
    setConnectedPeersCount(statusData.connected_peers_count ?? 0);
  };

  const resetSharingState = useCallback((): void => {
    setIsSharing(false);
    setSessionToken(null);
    setAssignedRole("primary");
    setAssignedLayerRange(null);
    setCoveragePercent(0);
    setIsSwarmReady(false);
    setSsdOffloadActive(false);
    setModelFingerprint(null);
    setConnectedPeersCount(0);
    setDownloadProgress(null);
    setDownloadStatusMsg("");
  }, []);

  // Public/Cloud rooms cannot be fetched while offline: hide the Lobby list in LAN/Solo mode
  const isCloudOffline = networkStatus === "lan_only" || networkStatus === "offline_solo";

  // Inspect host RAM and fetch initial Room Lobby on mount
  useEffect(() => {
    let cancelled = false;

    const initDashboard = async (): Promise<void> => {
      let freeGb: number | null = null;
      if (isTauriEnvironment()) {
        try {
          freeGb = await invoke<number>("get_available_ram_gb");
        } catch {
          freeGb = null;
        }
      }
      if (freeGb === null) {
        try {
          const res = await fetch(`${LOCAL_ENGINE_ROOT}/system/memory`);
          if (res.ok) {
            const mem = (await res.json()) as SystemMemoryResponse;
            freeGb = mem.available_ram_gb;
          }
        } catch {
          freeGb = null;
        }
      }

      if (!cancelled && freeGb !== null && Number.isFinite(freeGb)) {
        const safeMax = Math.max(MIN_RAM_GB, Math.floor(freeGb - 1));
        setAvailableRamGb(freeGb);
        setMaxRamGb(safeMax);
        setSharedRamGb((prev) => Math.min(prev, safeMax));
      }

      if (!cancelled) {
        try {
          const statusRes = await fetch(`${LOCAL_ENGINE_ROOT}/status`);
          if (statusRes.ok) {
            const statusData = (await statusRes.json()) as NodeStatusResponse;
            if (!cancelled && statusData.state === "RUNNING" && statusData.session_token) {
              setIsSharing(true);
              setSessionToken(statusData.session_token);
              setApiBaseUrl(statusData.api_base_url ?? DEFAULT_API_BASE_URL);
              setActiveModelId((prev) => statusData.model_id ?? prev);
              setRoomId(statusData.room_id ?? "public");
              setAssignedRole(statusData.role === "backup" ? "backup" : "primary");
              if (
                typeof statusData.shard_layer_start === "number" &&
                typeof statusData.shard_layer_end === "number"
              ) {
                setAssignedLayerRange([
                  statusData.shard_layer_start,
                  statusData.shard_layer_end,
                ]);
              }
              const pct = statusData.coverage_percent ?? 100;
              setCoveragePercent(pct);
              setIsSwarmReady(Boolean(statusData.is_swarm_ready ?? pct >= 100));
              setStatusMessage(statusData.message || "Active & Connected");
              applyNodeNetworkFields(statusData);
            }
          }
        } catch {
          // Core engine may not be started yet
        }
        await Promise.all([fetchAvailableRooms(), fetchModelCatalog()]);
      }
    };

    void initDashboard();
    return () => {
      cancelled = true;
    };
  }, [fetchAvailableRooms, fetchModelCatalog]);

  // Poll local /status every 5s while sharing so API Standby -> API Ready flips live
  useEffect(() => {
    if (!isSharing) return;
    let cancelled = false;

    const pollStatus = async (): Promise<void> => {
      try {
        const res = await fetch(`${LOCAL_ENGINE_ROOT}/status`);
        if (!res.ok) return;
        const statusData = (await res.json()) as NodeStatusResponse;
        if (!cancelled && statusData.state === "RUNNING") {
          const pct = statusData.coverage_percent ?? 100;
          setCoveragePercent(pct);
          setIsSwarmReady(Boolean(statusData.is_swarm_ready ?? pct >= 100));
          setAssignedRole(statusData.role === "backup" ? "backup" : "primary");
          applyNodeNetworkFields(statusData);
          if (
            typeof statusData.shard_layer_start === "number" &&
            typeof statusData.shard_layer_end === "number"
          ) {
            setAssignedLayerRange([
              statusData.shard_layer_start,
              statusData.shard_layer_end,
            ]);
          }
        } else if (
          !cancelled &&
          (statusData.state === "STOPPED" || statusData.state === "ERROR")
        ) {
          resetSharingState();
          if (statusData.network_status) {
            setNetworkStatus(statusData.network_status);
          }
          if (statusData.state === "ERROR") {
            const errMsg =
              statusData.message?.trim() || "Core engine entered ERROR state.";
            setErrorMessage(errMsg);
            setStatusMessage(`Error — ${errMsg}`);
          } else {
            const stopMsg = statusData.message?.trim();
            setStatusMessage(
              stopMsg
                ? `Stopped — ${stopMsg}`
                : "Stopped — Node was stopped by core engine."
            );
          }
          void Promise.all([fetchAvailableRooms(), fetchNetworkStatus()]);
        }
      } catch {
        // Ignore transient poll errors while engine restarts
      }
    };

    const timerId = window.setInterval(() => {
      void pollStatus();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(timerId);
    };
  }, [isSharing, resetSharingState, fetchAvailableRooms, fetchNetworkStatus]);

  const handleCopy = async (field: string, text: string): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => {
        setCopiedField((prev) => (prev === field ? null : prev));
      }, 1800);
    } catch {
      setErrorMessage("Clipboard access denied by browser.");
    }
  };

  const streamModelDownload = async (
    modelId: string,
    sourceType: ModelSourceType = "catalog",
    filename?: string | null,
    roomContext?: {
      roomId: string;
      networkMode: NetworkMode;
      roomKey?: string | null;
    }
  ): Promise<void> => {
    setDownloadProgress(0);
    setDownloadStatusMsg(`Checking cache / preparing ${modelId}...`);

    let backendAvailable = true;
    try {
      const probe = await fetch(`${LOCAL_ENGINE_ROOT}/health`);
      if (!probe.ok) backendAvailable = false;
    } catch {
      backendAvailable = false;
    }

    if (!backendAvailable && !isTauriEnvironment()) {
      for (const pct of [25, 60, 90, 100]) {
        await new Promise((resolve) => setTimeout(resolve, 60));
        setDownloadProgress(pct);
        setDownloadStatusMsg(`Simulating model preparation (${pct}%)...`);
      }
      return;
    }

    const params = new URLSearchParams({
      model_id: modelId,
      ram_gb: String(sharedRamGb),
      source_type: sourceType,
    });
    if (filename) {
      params.set("filename", filename);
    }
    if (roomContext?.roomId) {
      params.set("room_id", roomContext.roomId);
      params.set("network_mode", roomContext.networkMode);
      if (roomContext.roomKey) {
        params.set("room_key", roomContext.roomKey);
      }
    }

    const response = await fetch(
      `${LOCAL_ENGINE_ROOT}/models/download-stream?${params.toString()}`
    );
    if (response.status === 403 || response.status === 409) {
      const detail = await extractApiErrorDetail(
        response,
        `Cannot join room '${roomContext?.roomId ?? roomId}' (HTTP ${response.status}).`
      );
      throw new RoomJoinRejectedError(response.status, detail);
    }
    if (!response.ok || !response.body) {
      const detail = !response.ok
        ? await extractApiErrorDetail(
            response,
            `Failed to stream model download: HTTP ${response.status}`
          )
        : `Failed to stream model download: HTTP ${response.status}`;
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const rawEvent of events) {
        const line = rawEvent.trim();
        if (!line.startsWith("data:")) continue;
        const dataStr = line.slice(5).trim();
        if (dataStr === "[DONE]") {
          await reader.cancel();
          return;
        }
        let payload: SseDownloadEvent;
        try {
          payload = JSON.parse(dataStr) as SseDownloadEvent;
        } catch {
          continue;
        }
        if (payload.status === "error") {
          await reader.cancel();
          throw new Error(payload.message || "Model download failed.");
        }
        setDownloadProgress(payload.progress);
        setDownloadStatusMsg(payload.message || `${payload.progress.toFixed(0)}%`);
      }
    }
  };

  const resolveActiveModelSelection = () =>
    resolveModelSelection({
      modelTab,
      catalogModels,
      selectedCatalogId,
      modelInput,
      customFilename,
      localFilePath,
    });

  const syncModelSelectionState = (
    targetModel: string
  ): {
    model: string;
    sourceType: ModelSourceType;
    filename: string | null;
  } => {
    const trimmed = targetModel.trim();
    const matchedCatalog = catalogModels.find(
      (m) => m.repo_id === trimmed || m.id === trimmed
    );
    if (matchedCatalog) {
      setModelTab("catalog");
      setSelectedCatalogId(matchedCatalog.id);
      setModelInput(matchedCatalog.repo_id);
      return {
        model: matchedCatalog.repo_id,
        sourceType: "catalog",
        filename: matchedCatalog.filename ?? null,
      };
    }
    setModelTab("custom");
    setModelInput(trimmed);
    setCustomFilename("");
    return {
      model: trimmed,
      sourceType: inferCustomSourceType(trimmed),
      filename: null,
    };
  };

  const handleBrowseGgufFile = async (
    onPicked: (filePath: string) => void,
    triggerFallbackInput: () => void
  ): Promise<void> => {
    setErrorMessage(null);
    if (isTauriEnvironment()) {
      try {
        const picked = await openFileDialog({
          multiple: false,
          directory: false,
          filters: [{ name: "GGUF Model", extensions: ["gguf"] }],
        });
        if (typeof picked === "string" && picked.trim()) {
          onPicked(picked.trim());
          return;
        }
        if (picked === null) {
          return;
        }
      } catch {
        // Fallback to native HTML file picker if dialog plugin is unavailable
      }
    }
    triggerFallbackInput();
  };

  const openCreateRoomModal = (): void => {
    setModalModelTab(modelTab);
    setModalSelectedCatalogId(selectedCatalogId);
    setModalModel(modelInput);
    setModalCustomFilename(customFilename);
    setModalLocalFilePath(localFilePath);
    setErrorMessage(null);
    setShowRoomModal(true);
  };

  const startSharingWithConfig = async (
    override?: StartConfigOverride
  ): Promise<void> => {
    const activeSelection = resolveActiveModelSelection();
    const targetModel = (override?.model ?? activeSelection.model).trim();
    const targetSourceType: ModelSourceType =
      override?.sourceType ??
      (override?.model ? inferCustomSourceType(override.model) : activeSelection.sourceType);
    const targetFilename: string | null =
      override?.filename !== undefined
        ? override.filename
        : override?.model
          ? null
          : activeSelection.filename;
    const targetRoom = (override?.room ?? roomId).trim() || "public";
    const targetKey = (override?.key ?? roomKey).trim();
    const targetUnlisted = override?.unlisted ?? isUnlisted;
    const targetMode = override?.mode ?? networkMode;

    if (!targetModel) {
      setErrorMessage(
        modelTab === "local"
          ? "Please browse or enter a valid local .gguf file path."
          : "Please select a catalog model or enter a HuggingFace Repo ID / Direct URL."
      );
      return;
    }

    setIsBusy(true);
    setErrorMessage(null);
    setDownloadProgress(null);
    setDownloadStatusMsg("");

    try {
      setStatusMessage("Starting local Python core engine...");
      await invokeTauriCommand<string>("start_python_core");

      // Pre-flight room admission check BEFORE downloading any model weights
      setStatusMessage(`Checking room '${targetRoom}' availability...`);
      try {
        await checkRoomJoin({
          room_id: targetRoom,
          model_id: targetModel,
          network_mode: targetMode,
          room_key: targetKey || null,
        });
      } catch (checkErr: unknown) {
        if (checkErr instanceof RoomJoinRejectedError) {
          throw checkErr;
        }
        if (isTauriEnvironment()) {
          throw checkErr;
        }
      }

      setStatusMessage("Checking and preparing model weights...");
      await streamModelDownload(targetModel, targetSourceType, targetFilename, {
        roomId: targetRoom,
        networkMode: targetMode,
        roomKey: targetKey || null,
      });

      setStatusMessage(
        `Joining ${targetMode.toUpperCase()} room '${targetRoom}' & allocating ${sharedRamGb} GB RAM...`
      );

      const payload: StartNodeRequest = {
        node_id: "shardai-client-01",
        model_id: targetModel,
        source_type: targetSourceType,
        filename: targetFilename,
        network_mode: targetMode,
        room_id: targetRoom,
        room_key: targetKey || null,
        is_unlisted: targetUnlisted,
        backend: "llama_cpp",
        ram_gb: sharedRamGb,
        shared_ram_gb: sharedRamGb,
        max_ram_mb: sharedRamGb * 1024,
        max_vram_mb: 2048,
        p2p_listen_port: 49152,
        shard_layer_start: 0,
        shard_layer_end: 15,
        allow_ssd_offload: allowSsdOffload,
      };

      let response: Response | null = null;
      try {
        response = await fetch(`${LOCAL_ENGINE_ROOT}/start-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (fetchErr: unknown) {
        if (isTauriEnvironment()) {
          throw fetchErr;
        }
      }

      let data: NodeStatusResponse;
      if (response === null) {
        data = {
          state: "RUNNING",
          node_id: payload.node_id,
          model_id: payload.model_id,
          network_mode: payload.network_mode,
          room_id: payload.room_id,
          is_private: Boolean(targetKey),
          is_unlisted: targetUnlisted,
          backend: payload.backend,
          role: "primary",
          n_layers: 20,
          total_layers: 32,
          shard_layer_start: 0,
          shard_layer_end: 19,
          allocated_ram_mb: payload.max_ram_mb,
          allocated_vram_mb: payload.max_vram_mb,
          p2p_listen_port: payload.p2p_listen_port,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-shard-browser-preview-token",
          api_base_url: DEFAULT_API_BASE_URL,
          coverage_percent: 100,
          is_swarm_ready: true,
          message: "Gateway simulated in browser preview mode",
        };
      } else {
        if (response.status === 403 || response.status === 409) {
          const detail = await extractApiErrorDetail(
            response,
            `Cannot join room '${targetRoom}' (HTTP ${response.status}).`
          );
          throw new RoomJoinRejectedError(response.status, detail);
        }
        if (!response.ok) {
          const errText = await extractApiErrorDetail(
            response,
            `Python API /start-node failed (${response.status})`
          );
          throw new Error(`Python API /start-node failed (${response.status}): ${errText}`);
        }
        data = (await response.json()) as NodeStatusResponse;
      }

      setIsSharing(true);
      setSessionToken(data.session_token ?? null);
      setApiBaseUrl(data.api_base_url ?? DEFAULT_API_BASE_URL);
      setActiveModelId(data.model_id ?? targetModel);
      const nextRole: PeerRole = data.role === "backup" ? "backup" : "primary";
      setAssignedRole(nextRole);
      const initialCoverage = data.coverage_percent ?? 100;
      setCoveragePercent(initialCoverage);
      setIsSwarmReady(Boolean(data.is_swarm_ready ?? initialCoverage >= 100));
      if (
        typeof data.shard_layer_start === "number" &&
        typeof data.shard_layer_end === "number"
      ) {
        setAssignedLayerRange([data.shard_layer_start, data.shard_layer_end]);
      } else {
        setAssignedLayerRange([0, Math.max(0, (data.n_layers ?? 1) - 1)]);
      }

      const layersLabel =
        data.n_layers && data.total_layers
          ? `${data.n_layers}/${data.total_layers} layers`
          : `${sharedRamGb} GB RAM`;
      applyNodeNetworkFields(data);
      const statusLabel = data.network_status
        ? NETWORK_STATUS_META[data.network_status].label
        : (data.network_mode ?? targetMode).toUpperCase();
      const offloadLabel = data.ssd_offload_active ? " | 💾 SSD Offload (mmap)" : "";
      setStatusMessage(
        `Active & Connected (${statusLabel} / Room: ${data.room_id ?? targetRoom}) — ${layersLabel} | Peers: ${data.connected_peers_count}${offloadLabel}`
      );
      await fetchAvailableRooms();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setErrorMessage(message);
      setDownloadProgress(null);
      setDownloadStatusMsg("");
      if (err instanceof RoomJoinRejectedError) {
        setStatusMessage("Room join rejected — model download cancelled.");
      } else {
        setStatusMessage("Failed to start P2P AI Gateway.");
      }
    } finally {
      setIsBusy(false);
    }
  };

  const handleStopSharing = async (): Promise<void> => {
    setIsBusy(true);
    setErrorMessage(null);
    try {
      setStatusMessage("Revoking session token & releasing RAM/VRAM shards...");
      let stoppedViaApi = false;
      try {
        const stopRes = await fetch(`${LOCAL_ENGINE_ROOT}/stop-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });
        stoppedViaApi = stopRes.ok;
      } catch {
        // Proceed to terminate Python process if HTTP endpoint is unreachable
      }

      const stopMsg = stoppedViaApi
        ? "Node stopped & shards released (Core Engine ready)."
        : await invokeTauriCommand<string>("stop_python_core");
      resetSharingState();
      setStatusMessage(`Stopped — ${stopMsg}`);
      await Promise.all([fetchAvailableRooms(), fetchNetworkStatus()]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setErrorMessage(message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleJoinLobbyRoom = (room: RoomLobbyItem): void => {
    if (isSharing || isBusy) return;
    if (room.is_private) {
      setPendingPrivateRoom(room);
      setPrivateKeyInput("");
      return;
    }

    const mode: NetworkMode = room.network_scope === "lan" ? "lan" : "internet";
    const syncedModel = syncModelSelectionState(room.model);
    setRoomId(room.room_id);
    setRoomKey("");
    setIsUnlisted(false);
    setNetworkMode(mode);
    void startSharingWithConfig({
      model: syncedModel.model,
      sourceType: syncedModel.sourceType,
      filename: syncedModel.filename,
      room: room.room_id,
      key: "",
      unlisted: false,
      mode,
    });
  };

  const handleConfirmPrivateJoin = (event: React.FormEvent): void => {
    event.preventDefault();
    if (!pendingPrivateRoom || !privateKeyInput.trim()) {
      return;
    }
    const target = pendingPrivateRoom;
    const secret = privateKeyInput.trim();
    const mode: NetworkMode = target.network_scope === "lan" ? "lan" : "internet";
    const syncedModel = syncModelSelectionState(target.model);

    setPendingPrivateRoom(null);
    setRoomId(target.room_id);
    setRoomKey(secret);
    setIsUnlisted(false);
    setNetworkMode(mode);
    void startSharingWithConfig({
      model: syncedModel.model,
      sourceType: syncedModel.sourceType,
      filename: syncedModel.filename,
      room: target.room_id,
      key: secret,
      unlisted: false,
      mode,
    });
  };

  const handleCreateOrJoinUnlisted = (event: React.FormEvent): void => {
    event.preventDefault();
    const cleanRoom = modalRoomId.trim() || "private-room";
    const cleanKey = modalRoomKey.trim();
    const resolvedModel = resolveModelSelection({
      modelTab: modalModelTab,
      catalogModels,
      selectedCatalogId: modalSelectedCatalogId,
      modelInput: modalModel,
      customFilename: modalCustomFilename,
      localFilePath: modalLocalFilePath,
    });

    if (!resolvedModel.model) {
      setErrorMessage(
        modalModelTab === "local"
          ? "Please browse or enter a valid local .gguf file path."
          : "Please select a catalog model or enter a HuggingFace Repo ID / Direct URL."
      );
      return;
    }

    // Sync chosen room model configuration back to main Node & Model Configuration panel
    setModelTab(modalModelTab);
    if (modalModelTab === "catalog") {
      const matchedCatalog =
        catalogModels.find((m) => m.id === modalSelectedCatalogId) ??
        catalogModels.find((m) => m.repo_id === resolvedModel.model);
      if (matchedCatalog) {
        setSelectedCatalogId(matchedCatalog.id);
        setModelInput(matchedCatalog.repo_id);
      } else {
        setModelInput(resolvedModel.model);
      }
    } else if (modalModelTab === "local") {
      setLocalFilePath(resolvedModel.model);
    } else {
      setModelInput(resolvedModel.model);
      setCustomFilename(modalCustomFilename.trim());
    }

    setErrorMessage(null);
    setShowRoomModal(false);
    setRoomId(cleanRoom);
    setRoomKey(cleanKey);
    setIsUnlisted(modalUnlisted);
    void startSharingWithConfig({
      model: resolvedModel.model,
      sourceType: resolvedModel.sourceType,
      filename: resolvedModel.filename,
      room: cleanRoom,
      key: cleanKey,
      unlisted: modalUnlisted,
      mode: networkMode,
    });
  };

  const handleJoinByInviteString = (): void => {
    const raw = invitePasteInput.trim();
    if (!raw) return;
    const [parsedRoom, parsedKey] = raw.split("|").map((part) => part.trim());
    if (!parsedRoom) return;

    const activeSelection = resolveActiveModelSelection();
    setShowRoomModal(false);
    setRoomId(parsedRoom);
    setRoomKey(parsedKey ?? "");
    setIsUnlisted(true);
    void startSharingWithConfig({
      model: activeSelection.model,
      sourceType: activeSelection.sourceType,
      filename: activeSelection.filename,
      room: parsedRoom,
      key: parsedKey ?? "",
      unlisted: true,
      mode: networkMode,
    });
  };

  const generatedInviteString: string = modalRoomKey.trim()
    ? `${modalRoomId.trim() || "room"} | ${modalRoomKey.trim()}`
    : modalRoomId.trim() || "public";

  const curlExample: string = [
    `curl ${apiBaseUrl}/chat/completions \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer ${sessionToken ?? "sk-shard-..."}" \\`,
    `  -d '{`,
    `    "model": "${activeModelId}",`,
    `    "messages": [{"role": "user", "content": "Hello from IDE!"}],`,
    `    "stream": true`,
    `  }'`,
  ].join("\n");

  const showNodeConfig = activeSection === "overview" || activeSection === "node";
  const showRooms = activeSection === "overview" || activeSection === "rooms";
  const showCredentials =
    activeSection === "overview" || activeSection === "credentials";

  const inTauri = isTauriEnvironment();
  const netTone: StatusDotTone = networkStatus
    ? NETWORK_STATUS_META[networkStatus].tone
    : "neutral";
  const netTooltip: string =
    networkStatus === "lan_only"
      ? `Network: LAN Only (${lanPeersCount} LAN peer(s) discovered via mDNS/zeroconf)`
      : networkStatus
        ? `Network: ${NETWORK_STATUS_META[networkStatus].label}`
        : "Network: Starting Engine...";
  const gatewayTone: Exclude<StatusDotTone, "red" | "neutral"> = isSharing
    ? "green"
    : "yellow";
  const gatewayTooltip: string = isSharing ? "Gateway: Active" : "Gateway: Standby";
  const themeTooltip: string =
    themeMode === "dark"
      ? "Dark mode — Switch to Light"
      : "Light mode — Switch to Dark";

  return (
    <div className="native-shell" aria-label="ShardAi P2P AI Gateway">
      {/* =================================================================== */}
      {/* Custom Native OS Window Titlebar                                    */}
      {/* =================================================================== */}
      <header
        className="native-titlebar"
        data-tauri-drag-region
        onMouseDown={(e) => {
          if (
            inTauri &&
            e.button === 0 &&
            !(e.target as HTMLElement).closest("button")
          ) {
            void invokeTauriCommand("window_start_drag");
          }
        }}
        onDoubleClick={(e) => {
          if (
            inTauri &&
            !(e.target as HTMLElement).closest("button")
          ) {
            void invokeTauriCommand<boolean>("window_toggle_maximize").then((max) =>
              setIsMaximized(Boolean(max))
            );
          }
        }}
      >
        <div className="native-titlebar-left" data-tauri-drag-region>
          <button
            type="button"
            data-testid="sidebar-toggle-btn"
            className="sidebar-toggle-btn"
            onClick={toggleSidebar}
            aria-expanded={!isSidebarHidden}
            aria-controls="primary-navigation-sidebar"
            aria-label={isSidebarHidden ? "Show sidebar menu" : "Hide sidebar menu"}
            title={isSidebarHidden ? "Show Sidebar (Ctrl+B)" : "Hide Sidebar (Ctrl+B)"}
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <rect x="1.75" y="2.25" width="12.5" height="11.5" rx="2" />
              <line x1="6" y1="2.25" x2="6" y2="13.75" />
              {!isSidebarHidden && (
                <rect
                  x="2.25"
                  y="2.75"
                  width="3.25"
                  height="10.5"
                  rx="1"
                  fill="currentColor"
                  stroke="none"
                  opacity="0.28"
                />
              )}
            </svg>
          </button>
          <img
            src="/logo.png"
            alt="ShardAi Logo"
            data-testid="shardai-logo"
            className="native-titlebar-logo"
          />
          <span className="native-titlebar-title" data-tauri-drag-region>
            ShardAi Gateway
          </span>
          <span className="native-titlebar-version">v0.1.0</span>
        </div>

        <div className={`native-titlebar-right${inTauri ? " has-window-controls" : ""}`}>
          <div className="titlebar-status-group">
            <span
              role="status"
              tabIndex={0}
              data-testid="network-status-indicator"
              data-status={networkStatus ?? "starting"}
              data-tone={netTone}
              data-tooltip={netTooltip}
              aria-label={netTooltip}
              title={netTooltip}
              className="titlebar-status-dot native-tooltip-trigger"
            >
              <span
                className={`status-dot status-dot--${netTone}${networkStatus === null ? " status-dot--pulse" : ""}`}
                aria-hidden="true"
              />
            </span>

            <span
              role="status"
              tabIndex={0}
              data-testid="gateway-badge"
              data-status={isSharing ? "active" : "standby"}
              data-tone={gatewayTone}
              data-tooltip={gatewayTooltip}
              aria-label={gatewayTooltip}
              title={gatewayTooltip}
              className="titlebar-status-dot native-tooltip-trigger"
            >
              <span
                className={`status-dot status-dot--${gatewayTone}${isSharing ? " status-dot--active" : ""}`}
                aria-hidden="true"
              />
            </span>
          </div>

          <span className="titlebar-divider" aria-hidden="true" />

          <button
            type="button"
            data-testid="theme-toggle-btn"
            data-theme-mode={themeMode}
            data-theme-icon={themeMode === "dark" ? "moon" : "sun"}
            data-tooltip={themeTooltip}
            className="titlebar-icon-btn native-tooltip-trigger"
            onClick={toggleThemeMode}
            aria-label={themeTooltip}
            title={themeTooltip}
          >
            {themeMode === "dark" ? (
              <svg
                data-testid="theme-icon-moon"
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            ) : (
              <svg
                data-testid="theme-icon-sun"
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="4.5" />
                <line x1="12" y1="2.5" x2="12" y2="5" />
                <line x1="12" y1="19" x2="12" y2="21.5" />
                <line x1="4.93" y1="4.93" x2="6.7" y2="6.7" />
                <line x1="17.3" y1="17.3" x2="19.07" y2="19.07" />
                <line x1="2.5" y1="12" x2="5" y2="12" />
                <line x1="19" y1="12" x2="21.5" y2="12" />
                <line x1="4.93" y1="19.07" x2="6.7" y2="17.3" />
                <line x1="17.3" y1="6.7" x2="19.07" y2="4.93" />
              </svg>
            )}
          </button>

          {inTauri && (
            <div className="window-controls">
              <button
                type="button"
                aria-label="Minimize window"
                className="window-caption-btn"
                onClick={() => void invokeTauriCommand("window_minimize")}
              >
                <svg
                  width="11"
                  height="11"
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.2"
                  aria-hidden="true"
                >
                  <line x1="2" y1="6" x2="10" y2="6" />
                </svg>
              </button>
              <button
                type="button"
                aria-label="Maximize window"
                className="window-caption-btn"
                onClick={() =>
                  void invokeTauriCommand<boolean>("window_toggle_maximize").then((max) =>
                    setIsMaximized(Boolean(max))
                  )
                }
              >
                {isMaximized ? (
                  <svg
                    width="11"
                    height="11"
                    viewBox="0 0 12 12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.1"
                    aria-hidden="true"
                  >
                    <rect x="2" y="3.5" width="6.5" height="6.5" rx="0.5" />
                    <path d="M4 3.5V2h6v6H8.5" />
                  </svg>
                ) : (
                  <svg
                    width="11"
                    height="11"
                    viewBox="0 0 12 12"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.1"
                    aria-hidden="true"
                  >
                    <rect x="2" y="2" width="8" height="8" rx="0.5" />
                  </svg>
                )}
              </button>
              <button
                type="button"
                aria-label="Close application"
                className="window-caption-btn close"
                onClick={() => void invokeTauriCommand("window_close")}
              >
                <svg
                  width="11"
                  height="11"
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <line x1="2.5" y1="2.5" x2="9.5" y2="9.5" />
                  <line x1="9.5" y1="2.5" x2="2.5" y2="9.5" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* =================================================================== */}
      {/* Desktop Split Body: Left Sidebar + Main Workspace                   */}
      {/* =================================================================== */}
      <div className="native-body">
        <aside
          id="primary-navigation-sidebar"
          data-testid="native-sidebar"
          className={`native-sidebar${isSidebarHidden ? " collapsed" : ""}`}
          aria-label="Primary Navigation"
          aria-hidden={isSidebarHidden}
          hidden={isSidebarHidden}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <nav>
              <div className="sidebar-section-label">Workspace</div>
              {(
                [
                  { id: "overview", label: "Overview" },
                  { id: "node", label: "Node & Model" },
                  { id: "rooms", label: "Rooms Lobby" },
                  { id: "credentials", label: "API Gateway" },
                ] as const
              ).map((item) => (
                <button
                  key={item.id}
                  type="button"
                  data-testid={`nav-${item.id}`}
                  className={`sidebar-nav-btn${activeSection === item.id ? " active" : ""}`}
                  onClick={() => setActiveSection(item.id)}
                >
                  {item.id === "overview" && (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <rect x="3" y="3" width="7" height="7" rx="1.5" />
                      <rect x="14" y="3" width="7" height="7" rx="1.5" />
                      <rect x="3" y="14" width="7" height="7" rx="1.5" />
                      <rect x="14" y="14" width="7" height="7" rx="1.5" />
                    </svg>
                  )}
                  {item.id === "node" && (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <rect x="4" y="4" width="16" height="16" rx="2" />
                      <rect x="9" y="9" width="6" height="6" rx="1" />
                      <line x1="9" y1="1.5" x2="9" y2="4" />
                      <line x1="15" y1="1.5" x2="15" y2="4" />
                      <line x1="9" y1="20" x2="9" y2="22.5" />
                      <line x1="15" y1="20" x2="15" y2="22.5" />
                      <line x1="1.5" y1="9" x2="4" y2="9" />
                      <line x1="1.5" y1="15" x2="4" y2="15" />
                      <line x1="20" y1="9" x2="22.5" y2="9" />
                      <line x1="20" y1="15" x2="22.5" y2="15" />
                    </svg>
                  )}
                  {item.id === "rooms" && (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <circle cx="12" cy="12" r="9" />
                      <line x1="3" y1="12" x2="21" y2="12" />
                      <path d="M12 3a14.5 14.5 0 0 1 0 18 14.5 14.5 0 0 1 0-18Z" />
                    </svg>
                  )}
                  {item.id === "credentials" && (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <circle cx="8" cy="15" r="4" />
                      <path d="M11 12l9-9" />
                      <path d="M16 7l2.5 2.5" />
                      <path d="M19 4l2.5 2.5" />
                    </svg>
                  )}
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>

            <div className="sidebar-telemetry" data-testid="sidebar-telemetry">
              <div className="sidebar-section-label" style={{ padding: 0, marginBottom: "2px" }}>
                Node Telemetry
              </div>
              <div className="sidebar-telemetry-row">
                <span style={{ color: "var(--text-muted)" }}>Allocated</span>
                <strong>{sharedRamGb} GB RAM</strong>
              </div>
              <div className="sidebar-telemetry-row">
                <span style={{ color: "var(--text-muted)" }}>Layers</span>
                <strong>
                  {isSharing && assignedLayerRange
                    ? `${assignedLayerRange[0]}..${assignedLayerRange[1]}`
                    : "Idle"}
                </strong>
              </div>
              <div className="sidebar-telemetry-row">
                <span style={{ color: "var(--text-muted)" }}>Peers</span>
                <strong>{connectedPeersCount}</strong>
              </div>
              <div className="sidebar-telemetry-row">
                <span style={{ color: "var(--text-muted)" }}>Swarm</span>
                <strong style={{ color: isSwarmReady ? "var(--success-fg)" : "var(--text-secondary)" }}>
                  {isSharing ? `${Math.round(coveragePercent)}%` : "Standby"}
                </strong>
              </div>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="github-repo-link"
              aria-label="ShardAi on GitHub"
              className="native-btn"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
                textDecoration: "none",
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
              <span>ShardAi GitHub</span>
            </a>
          </div>
        </aside>

        <main className="native-main">
          {errorMessage !== null && (
            <div
              role="alert"
              style={{
                padding: "9px 12px",
                borderRadius: "8px",
                backgroundColor: "var(--danger-soft)",
                border: "1px solid var(--danger)",
                color: "var(--danger)",
                fontSize: "12.5px",
                fontWeight: 500,
              }}
            >
              {errorMessage}
            </div>
          )}

          <div className={activeSection === "overview" ? "native-grid" : "native-grid single"}>
            {/* =============================================================== */}
            {/* Left Workspace Column: Node & Model Configuration               */}
            {/* =============================================================== */}
            {showNodeConfig && (
              <section className="native-panel" aria-label="Node Configuration">
                <div className="native-panel-header">
                  <h2 className="native-panel-title">Node &amp; Model Configuration</h2>
                  <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                    Local Gateway :8000
                  </span>
                </div>

                {/* 3-Tab Model Selector */}
                <ModelSelector
                  label="Model Source"
                  testIdPrefix=""
                  modelTab={modelTab}
                  catalogModels={catalogModels}
                  catalogSource={catalogSource}
                  isLoadingCatalog={isLoadingCatalog}
                  selectedCatalogId={selectedCatalogId}
                  modelInput={modelInput}
                  customFilename={customFilename}
                  localFilePath={localFilePath}
                  disabled={isSharing || isBusy}
                  onTabChange={(tab) => {
                    setModelTab(tab);
                    setDownloadProgress(null);
                    setDownloadStatusMsg("");
                  }}
                  onSelectCatalogModel={(chosenId, found) => {
                    setSelectedCatalogId(chosenId);
                    if (found) {
                      setModelInput(found.repo_id);
                    }
                    setDownloadProgress(null);
                    setDownloadStatusMsg("");
                  }}
                  onRefreshCatalog={() => void fetchModelCatalog()}
                  onModelInputChange={(value) => {
                    setModelInput(value);
                    setDownloadProgress(null);
                    setDownloadStatusMsg("");
                  }}
                  onCustomFilenameChange={setCustomFilename}
                  onLocalFilePathChange={(value) => {
                    setLocalFilePath(value);
                    setDownloadProgress(null);
                    setDownloadStatusMsg("");
                  }}
                  onBrowseLocalFile={(triggerFallback) =>
                    void handleBrowseGgufFile((pickedPath) => {
                      setLocalFilePath(pickedPath);
                      setDownloadProgress(null);
                      setDownloadStatusMsg("");
                    }, triggerFallback)
                  }
                />

                {/* RAM Slider */}
                <div style={{ marginBottom: "16px" }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "6px",
                    }}
                  >
                    <label
                      htmlFor="ram-slider"
                      className="native-label"
                      style={{ marginBottom: 0 }}
                    >
                      Shared RAM Contribution
                    </label>
                    <span
                      data-testid="selected-ram-label"
                      style={{ fontSize: "13px", fontWeight: 700, color: "var(--accent-text)" }}
                    >
                      {sharedRamGb} GB
                    </span>
                  </div>

                  <input
                    id="ram-slider"
                    type="range"
                    min={MIN_RAM_GB}
                    max={maxRamGb}
                    step={1}
                    value={sharedRamGb}
                    disabled={isSharing || isBusy}
                    onChange={(e) => setSharedRamGb(Number(e.target.value))}
                    style={{
                      width: "100%",
                      accentColor: "var(--accent)",
                      cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                    }}
                  />

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: "11px",
                      color: "var(--text-muted)",
                      marginTop: "4px",
                    }}
                  >
                    <span>{MIN_RAM_GB} GB</span>
                    {availableRamGb !== null && (
                      <span data-testid="free-ram-note">
                        Free: {availableRamGb.toFixed(1)} GB (1 GB reserved for OS)
                      </span>
                    )}
                    <span>{maxRamGb} GB</span>
                  </div>

                  <label
                    htmlFor="ssd-offload-checkbox"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      marginTop: "10px",
                      padding: "8px 10px",
                      borderRadius: "6px",
                      backgroundColor: allowSsdOffload
                        ? "var(--info-bg)"
                        : "var(--bg-surface-subtle)",
                      border: `1px solid ${
                        allowSsdOffload ? "var(--info-border)" : "var(--border-hairline)"
                      }`,
                      fontSize: "12px",
                      color: "var(--text-secondary)",
                      cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                    }}
                  >
                    <input
                      id="ssd-offload-checkbox"
                      data-testid="ssd-offload-checkbox"
                      type="checkbox"
                      checked={allowSsdOffload}
                      disabled={isSharing || isBusy}
                      onChange={(e) => setAllowSsdOffload(e.target.checked)}
                    />
                    <span>
                      Allow SSD Offloading (Slow but guarantees 100% run in Solo Mode)
                    </span>
                  </label>
                </div>

                {/* Network Mode, Room ID & Optional Room Key */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "10px",
                    marginBottom: "16px",
                  }}
                >
                  <div style={{ gridColumn: "1 / -1" }}>
                    <span className="native-label">Network Scope</span>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
                      {(["lan", "internet"] as const).map((mode) => (
                        <button
                          key={mode}
                          type="button"
                          data-testid={`network-mode-${mode}`}
                          disabled={isSharing || isBusy}
                          onClick={() => setNetworkMode(mode)}
                          className={`native-btn${networkMode === mode ? " native-btn-primary" : ""}`}
                        >
                          {mode === "lan" ? "LAN (Wi-Fi / mDNS)" : "Internet (Cloud)"}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label htmlFor="room-id-input" className="native-label">
                      Room ID
                    </label>
                    <input
                      id="room-id-input"
                      data-testid="room-id-input"
                      type="text"
                      className="native-input"
                      value={roomId}
                      disabled={isSharing || isBusy}
                      onChange={(e) => setRoomId(e.target.value)}
                      placeholder="public"
                    />
                  </div>

                  <div>
                    <label htmlFor="room-key-input" className="native-label">
                      Room Key (Optional)
                    </label>
                    <input
                      id="room-key-input"
                      data-testid="room-key-input"
                      type="password"
                      className="native-input"
                      value={roomKey}
                      disabled={isSharing || isBusy}
                      onChange={(e) => setRoomKey(e.target.value)}
                      placeholder="Public if blank"
                    />
                  </div>
                </div>

                {/* Model Download Progress Bar */}
                {downloadProgress !== null && (
                  <div style={{ marginBottom: "14px" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "11.5px",
                        marginBottom: "5px",
                      }}
                    >
                      <span
                        data-testid="model-download-status-msg"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {downloadStatusMsg || "Preparing model..."}
                      </span>
                      <span
                        data-testid="model-download-percent"
                        style={{ fontWeight: 700, color: "var(--accent-text)" }}
                      >
                        {downloadProgress.toFixed(0)}%
                      </span>
                    </div>
                    <div
                      style={{
                        width: "100%",
                        height: "6px",
                        backgroundColor: "var(--bg-app)",
                        borderRadius: "999px",
                        overflow: "hidden",
                        border: "1px solid var(--border-hairline)",
                      }}
                    >
                      <div
                        data-testid="model-download-progress-bar"
                        style={{
                          width: `${downloadProgress}%`,
                          height: "100%",
                          backgroundColor:
                            downloadProgress >= 100 ? "#16a34a" : "var(--accent)",
                          transition: "width 0.2s ease-in-out",
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Primary Action Button */}
                <button
                  type="button"
                  data-testid="toggle-sharing-btn"
                  onClick={() => {
                    if (isSharing) {
                      void handleStopSharing();
                    } else {
                      void startSharingWithConfig();
                    }
                  }}
                  disabled={isBusy}
                  className={`native-btn ${isSharing ? "native-btn-danger" : "native-btn-primary"}`}
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    fontSize: "13px",
                  }}
                >
                  {isBusy
                    ? "Connecting..."
                    : isSharing
                      ? "Stop Sharing"
                      : "Start Sharing & Join Room"}
                </button>
              </section>
            )}

            {/* =============================================================== */}
            {/* Right Workspace Column: Rooms Lobby + Modals + API Credentials  */}
            {/* =============================================================== */}
            {(showRooms || showCredentials) && (
              <div style={{ display: "flex", flexDirection: "column", gap: "14px", minWidth: 0 }}>
                {showRooms && (
                  <section
                    data-testid="network-rooms-lobby"
                    className="native-panel"
                    aria-label="Network Rooms Lobby"
                  >
                    <div className="native-panel-header">
                      <h2 className="native-panel-title">Network Rooms Lobby</h2>
                      <div style={{ display: "flex", gap: "6px" }}>
                        <button
                          type="button"
                          data-testid="refresh-rooms-btn"
                          className="native-btn"
                          onClick={() => {
                            void fetchAvailableRooms({ reseed: true });
                            void fetchNetworkStatus();
                          }}
                          disabled={isLoadingRooms}
                        >
                          {isLoadingRooms ? "Scanning..." : "Refresh"}
                        </button>
                        <button
                          type="button"
                          data-testid="create-room-modal-btn"
                          className="native-btn native-btn-primary"
                          disabled={isSharing || isBusy}
                          onClick={openCreateRoomModal}
                        >
                          Create New Room / Join Unlisted
                        </button>
                      </div>
                    </div>

                    {isCloudOffline ? (
                      <div
                        data-testid="offline-rooms-hidden-notice"
                        style={{
                          padding: "12px",
                          borderRadius: "8px",
                          backgroundColor: "var(--bg-surface-subtle)",
                          border: "1px dashed var(--border-strong)",
                          fontSize: "12px",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {networkStatus === "lan_only" ? (
                          <>
                            <div
                              style={{
                                fontWeight: 600,
                                color: "var(--warn-fg)",
                                marginBottom: "6px",
                              }}
                            >
                              🟡 Public Cloud Rooms hidden in LAN Only mode (Cloudflare Tracker
                              unreachable).
                            </div>
                            {availableRooms.filter((r) => r.network_scope === "lan").length > 0 ? (
                              <div
                                style={{
                                  display: "flex",
                                  flexDirection: "column",
                                  gap: "6px",
                                  marginTop: "8px",
                                }}
                              >
                                {availableRooms
                                  .filter((r) => r.network_scope === "lan")
                                  .map((room) => (
                                    <div
                                      key={room.room_id}
                                      style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center",
                                        padding: "8px 10px",
                                        borderRadius: "6px",
                                        backgroundColor: "var(--warn-bg)",
                                        border: "1px solid var(--warn-border)",
                                      }}
                                    >
                                      <span>
                                        <strong>{room.room_id}</strong> ({room.model}) —{" "}
                                        {room.nodes_online} LAN node(s), {room.total_shared_ram_gb} GB
                                      </span>
                                      <button
                                        type="button"
                                        data-testid={`join-room-${room.room_id}`}
                                        className="native-btn native-btn-primary"
                                        disabled={isSharing || isBusy}
                                        onClick={() => handleJoinLobbyRoom(room)}
                                      >
                                        Join LAN
                                      </button>
                                    </div>
                                  ))}
                              </div>
                            ) : (
                              <span>
                                Configure a room on the left or wait for mDNS/Zeroconf LAN peers.
                              </span>
                            )}
                          </>
                        ) : (
                          <span>
                            🔴 <strong>Offline (Solo Mode)</strong> — Public Rooms list is hidden
                            while offline. Click <strong>Start Sharing &amp; Join Room</strong> to
                            run standalone (with optional SSD Offloading).
                          </span>
                        )}
                      </div>
                    ) : (
                      <div className="native-table-wrap">
                        <table data-testid="rooms-lobby-table" className="native-table">
                          <thead>
                            <tr>
                              <th>Room Name</th>
                              <th>Model</th>
                              <th>Nodes</th>
                              <th>Shared RAM</th>
                              <th>Resources</th>
                              <th>Status</th>
                              <th style={{ textAlign: "right" }}>Action</th>
                            </tr>
                          </thead>
                          <tbody>
                            {availableRooms.length === 0 ? (
                              <tr>
                                <td
                                  colSpan={7}
                                  style={{
                                    padding: "18px",
                                    textAlign: "center",
                                    color: "var(--text-muted)",
                                  }}
                                >
                                  No active rooms discovered yet. Click &quot;Refresh&quot; or start sharing.
                                </td>
                              </tr>
                            ) : (
                              availableRooms.map((room) => {
                                const coverage = room.coverage_percent ?? 100;
                                const redundancy = room.redundancy_factor ?? 1;
                                const isReady = coverage >= 100;
                                return (
                                  <tr key={room.room_id}>
                                    <td
                                      data-testid={`room-name-${room.room_id}`}
                                      className="room-name-cell"
                                      title={room.room_id}
                                    >
                                      {formatCompactRoomName(room.room_id)}
                                    </td>
                                    <td
                                      data-testid={`room-model-${room.room_id}`}
                                      className="room-model-cell"
                                      title={room.model}
                                    >
                                      {formatCompactModelName(room.model)}
                                    </td>
                                    <td>{room.nodes_online}</td>
                                    <td>{room.total_shared_ram_gb} GB</td>
                                    <td>
                                      <span
                                        data-testid={`room-coverage-${room.room_id}`}
                                        className={`room-resource-text ${
                                          isReady
                                            ? "room-resource-text--ready"
                                            : "room-resource-text--pooling"
                                        }`}
                                      >
                                        {isReady
                                          ? `Ready (Redundancy: ${redundancy}x)`
                                          : `Pooling RAM: ${Math.round(coverage)}%`}
                                      </span>
                                    </td>
                                    <td>
                                      <span
                                        data-testid={`room-status-${room.room_id}`}
                                        className={`room-status-text ${
                                          room.is_private
                                            ? "room-status-text--private"
                                            : "room-status-text--public"
                                        }`}
                                      >
                                        {room.is_private ? "Private" : "Public"}
                                      </span>
                                    </td>
                                    <td style={{ textAlign: "right" }}>
                                      <button
                                        type="button"
                                        data-testid={`join-room-${room.room_id}`}
                                        className="native-btn native-btn-primary"
                                        disabled={isSharing || isBusy}
                                        onClick={() => handleJoinLobbyRoom(room)}
                                        style={{ padding: "4px 10px", fontSize: "11.5px" }}
                                      >
                                        Join
                                      </button>
                                    </td>
                                  </tr>
                                );
                              })
                            )}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </section>
                )}

                {/* Private Room Key Prompt Modal */}
                {pendingPrivateRoom !== null && (
                  <div
                    role="dialog"
                    aria-label="Enter Private Room Key"
                    data-testid="private-room-key-modal"
                    className="native-panel"
                    style={{
                      backgroundColor: "var(--warn-bg)",
                      borderColor: "var(--warn-border)",
                    }}
                  >
                    <h3
                      style={{
                        margin: "0 0 8px 0",
                        fontSize: "13px",
                        fontWeight: 700,
                        color: "var(--warn-fg)",
                      }}
                    >
                      🔒 Enter Room Key for &quot;{pendingPrivateRoom.room_id}&quot;
                    </h3>
                    <form onSubmit={handleConfirmPrivateJoin}>
                      <div style={{ display: "flex", gap: "8px" }}>
                        <input
                          type="password"
                          data-testid="private-room-key-input"
                          className="native-input"
                          value={privateKeyInput}
                          onChange={(e) => setPrivateKeyInput(e.target.value)}
                          placeholder="Room passphrase / invite key"
                          autoFocus
                          style={{ flex: 1 }}
                        />
                        <button
                          type="submit"
                          data-testid="confirm-private-join-btn"
                          className="native-btn native-btn-primary"
                        >
                          Unlock &amp; Join
                        </button>
                        <button
                          type="button"
                          className="native-btn"
                          onClick={() => setPendingPrivateRoom(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    </form>
                  </div>
                )}

                {/* Create New Room / Join Unlisted Modal */}
                {showRoomModal && (
                  <div
                    role="dialog"
                    aria-label="Create New Room or Join Unlisted"
                    data-testid="create-room-modal"
                    className="native-panel"
                  >
                    <div className="native-panel-header">
                      <h3 className="native-panel-title">
                        Create New Room / Join Unlisted
                      </h3>
                      <button
                        type="button"
                        className="native-btn"
                        style={{ padding: "2px 8px" }}
                        onClick={() => setShowRoomModal(false)}
                      >
                        ✕
                      </button>
                    </div>

                    <form onSubmit={handleCreateOrJoinUnlisted}>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "1fr 1fr",
                          gap: "10px",
                          marginBottom: "10px",
                        }}
                      >
                        <div>
                          <label className="native-label">New Room Name</label>
                          <input
                            type="text"
                            data-testid="modal-room-name-input"
                            className="native-input"
                            value={modalRoomId}
                            onChange={(e) => setModalRoomId(e.target.value)}
                            placeholder="my-private-cluster"
                          />
                        </div>
                        <div>
                          <label className="native-label">
                            Room Key (Optional — 🔒 Private)
                          </label>
                          <input
                            type="text"
                            data-testid="modal-room-key-input"
                            className="native-input"
                            value={modalRoomKey}
                            onChange={(e) => setModalRoomKey(e.target.value)}
                            placeholder="Leave blank for Public"
                          />
                        </div>
                      </div>

                      <ModelSelector
                        label="AI Model"
                        testIdPrefix="modal-"
                        marginBottom="12px"
                        modelTab={modalModelTab}
                        catalogModels={catalogModels}
                        catalogSource={catalogSource}
                        isLoadingCatalog={isLoadingCatalog}
                        selectedCatalogId={modalSelectedCatalogId}
                        modelInput={modalModel}
                        customFilename={modalCustomFilename}
                        localFilePath={modalLocalFilePath}
                        disabled={isSharing || isBusy}
                        onTabChange={(tab) => {
                          setModalModelTab(tab);
                          setErrorMessage(null);
                        }}
                        onSelectCatalogModel={(chosenId, found) => {
                          setModalSelectedCatalogId(chosenId);
                          if (found) {
                            setModalModel(found.repo_id);
                          }
                          setErrorMessage(null);
                        }}
                        onRefreshCatalog={() => void fetchModelCatalog()}
                        onModelInputChange={(value) => {
                          setModalModel(value);
                          setErrorMessage(null);
                        }}
                        onCustomFilenameChange={setModalCustomFilename}
                        onLocalFilePathChange={(value) => {
                          setModalLocalFilePath(value);
                          setErrorMessage(null);
                        }}
                        onBrowseLocalFile={(triggerFallback) =>
                          void handleBrowseGgufFile((pickedPath) => {
                            setModalLocalFilePath(pickedPath);
                            setErrorMessage(null);
                          }, triggerFallback)
                        }
                      />

                      <label
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          fontSize: "12px",
                          marginBottom: "12px",
                          cursor: "pointer",
                        }}
                      >
                        <input
                          type="checkbox"
                          data-testid="modal-unlisted-checkbox"
                          checked={modalUnlisted}
                          onChange={(e) => setModalUnlisted(e.target.checked)}
                        />
                        Unlisted Room (hide from public Network Rooms Lobby)
                      </label>

                      <div style={{ marginBottom: "12px" }}>
                        <span className="native-label">
                          Invite String (Share with teammates)
                        </span>
                        <div style={{ display: "flex", gap: "8px" }}>
                          <code
                            data-testid="invite-string-preview"
                            className="native-input"
                            style={{ flex: 1, fontFamily: "monospace" }}
                          >
                            {generatedInviteString}
                          </code>
                          <button
                            type="button"
                            className="native-btn"
                            onClick={() => void handleCopy("invite", generatedInviteString)}
                          >
                            {copiedField === "invite" ? "Copied!" : "Copy Invite"}
                          </button>
                        </div>
                      </div>

                      <button
                        type="submit"
                        data-testid="modal-create-join-btn"
                        className="native-btn native-btn-primary"
                        style={{ width: "100%", padding: "8px 14px" }}
                      >
                        Create &amp; Start Sharing
                      </button>
                    </form>

                    <div
                      style={{
                        marginTop: "12px",
                        paddingTop: "12px",
                        borderTop: "1px solid var(--border-hairline)",
                      }}
                    >
                      <span className="native-label">
                        Or Paste Invite String (RoomID | Key)
                      </span>
                      <div style={{ display: "flex", gap: "8px" }}>
                        <input
                          type="text"
                          data-testid="paste-invite-input"
                          className="native-input"
                          value={invitePasteInput}
                          onChange={(e) => setInvitePasteInput(e.target.value)}
                          placeholder="e.g. studio-room-1 | secret-pass"
                          style={{ flex: 1 }}
                        />
                        <button
                          type="button"
                          data-testid="join-by-invite-btn"
                          className="native-btn"
                          onClick={handleJoinByInviteString}
                        >
                          Join Unlisted
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Active Node Role & Session API Credentials Panel */}
                {showCredentials && isSharing && sessionToken ? (
                  <section
                    data-testid="session-credentials-panel"
                    className="native-panel"
                    aria-label="Session API Credentials"
                  >
                    <div
                      data-testid="active-node-role-card"
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        flexWrap: "wrap",
                        gap: "8px",
                        padding: "9px 12px",
                        marginBottom: "12px",
                        borderRadius: "8px",
                        backgroundColor:
                          assignedRole === "primary" ? "var(--info-bg)" : "var(--warn-bg)",
                        border: `1px solid ${
                          assignedRole === "primary"
                            ? "var(--info-border)"
                            : "var(--warn-border)"
                        }`,
                      }}
                    >
                      <span
                        data-testid="active-node-role-label"
                        style={{
                          fontSize: "12.5px",
                          fontWeight: 700,
                          color:
                            assignedRole === "primary" ? "var(--info-fg)" : "var(--warn-fg)",
                        }}
                      >
                        {assignedRole === "primary"
                          ? `Role: Primary Seeder (Layers ${assignedLayerRange?.[0] ?? 0} - ${assignedLayerRange?.[1] ?? 0})`
                          : `Role: Backup Node (Layers ${assignedLayerRange?.[0] ?? 0} - ${assignedLayerRange?.[1] ?? 0})`}
                      </span>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "6px",
                          flexWrap: "wrap",
                        }}
                      >
                        {modelFingerprint && (
                          <span
                            data-testid="model-fingerprint-badge"
                            title={`SHA-256 (10MB + size): ${modelFingerprint}`}
                            style={{
                              fontSize: "11px",
                              fontWeight: 600,
                              padding: "2px 8px",
                              borderRadius: "999px",
                              backgroundColor: "var(--bg-surface-subtle)",
                              color: "var(--text-secondary)",
                              border: "1px solid var(--border-strong)",
                              fontFamily: "monospace",
                            }}
                          >
                            FP: {modelFingerprint.slice(0, 10)}
                          </span>
                        )}
                        {ssdOffloadActive && (
                          <span
                            data-testid="ssd-offload-active-badge"
                            style={{
                              fontSize: "11px",
                              fontWeight: 600,
                              padding: "2px 8px",
                              borderRadius: "999px",
                              backgroundColor: "var(--info-bg)",
                              color: "var(--info-fg)",
                              border: "1px solid var(--info-border)",
                            }}
                          >
                            💾 SSD Offload (mmap)
                          </span>
                        )}
                        <span
                          style={{
                            fontSize: "11px",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: "999px",
                            backgroundColor:
                              assignedRole === "primary" ? "var(--info-bg)" : "var(--warn-bg)",
                            color:
                              assignedRole === "primary" ? "var(--info-fg)" : "var(--warn-fg)",
                          }}
                        >
                          {assignedRole === "primary" ? "PRIMARY" : "BACKUP"}
                        </span>
                      </div>
                    </div>

                    <div className="native-panel-header">
                      <h2 className="native-panel-title">Session API Credentials</h2>
                      <span
                        data-testid="api-readiness-badge"
                        style={{
                          fontSize: "11px",
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: "999px",
                          backgroundColor:
                            isSwarmReady && coveragePercent >= 100
                              ? "var(--success-bg)"
                              : "var(--warn-bg)",
                          color:
                            isSwarmReady && coveragePercent >= 100
                              ? "var(--success-fg)"
                              : "var(--warn-fg)",
                          border: `1px solid ${
                            isSwarmReady && coveragePercent >= 100
                              ? "var(--success-border)"
                              : "var(--warn-border)"
                          }`,
                        }}
                      >
                        {isSwarmReady && coveragePercent >= 100
                          ? "🟢 API Ready (Swarm is ready to accept requests)"
                          : `🟡 API Standby (Waiting for sufficient RAM: ${Math.round(coveragePercent)}% - Model not ready)`}
                      </span>
                    </div>

                    <div style={{ marginBottom: "10px" }}>
                      <span className="native-label">OpenAI Base URL</span>
                      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                        <code
                          data-testid="gateway-base-url"
                          className="native-input"
                          style={{ flex: 1, fontFamily: "monospace" }}
                        >
                          {apiBaseUrl}
                        </code>
                        <button
                          type="button"
                          className="native-btn"
                          onClick={() => void handleCopy("baseUrl", apiBaseUrl)}
                        >
                          {copiedField === "baseUrl" ? "Copied!" : "Copy"}
                        </button>
                      </div>
                    </div>

                    <div style={{ marginBottom: "12px" }}>
                      <span className="native-label">API Key (Session Token)</span>
                      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                        <code
                          data-testid="session-token-value"
                          className="native-input"
                          style={{
                            flex: 1,
                            fontFamily: "monospace",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {sessionToken}
                        </code>
                        <button
                          type="button"
                          className="native-btn"
                          onClick={() => void handleCopy("token", sessionToken)}
                        >
                          {copiedField === "token" ? "Copied!" : "Copy"}
                        </button>
                        {(!isSwarmReady || coveragePercent < 100) && (
                          <button
                            type="button"
                            data-testid="copy-invite-code-btn"
                            className="native-btn"
                            style={{
                              borderColor: "var(--warn-border)",
                              backgroundColor: "var(--warn-bg)",
                              color: "var(--warn-fg)",
                            }}
                            onClick={() =>
                              void handleCopy(
                                "activeInvite",
                                roomKey.trim()
                                  ? `${roomId.trim() || "public"} | ${roomKey.trim()}`
                                  : roomId.trim() || "public"
                              )
                            }
                          >
                            {copiedField === "activeInvite"
                              ? "Copied Invite!"
                              : "Copy Invite Code"}
                          </button>
                        )}
                      </div>
                    </div>

                    <div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: "6px",
                        }}
                      >
                        <span className="native-label" style={{ marginBottom: 0 }}>
                          Quick Test (curl / IDE Config)
                        </span>
                        <button
                          type="button"
                          className="native-btn"
                          style={{ padding: "3px 8px", fontSize: "11px" }}
                          onClick={() => void handleCopy("curl", curlExample)}
                        >
                          {copiedField === "curl" ? "Copied!" : "Copy curl"}
                        </button>
                      </div>
                      <pre
                        data-testid="curl-snippet"
                        style={{
                          margin: 0,
                          padding: "10px 12px",
                          borderRadius: "8px",
                          backgroundColor: "var(--bg-code)",
                          color: "#e2e8f0",
                          fontSize: "11.5px",
                          lineHeight: 1.45,
                          overflowX: "auto",
                        }}
                      >
                        <code>{curlExample}</code>
                      </pre>
                    </div>
                  </section>
                ) : (
                  activeSection === "credentials" && (
                    <section className="native-panel">
                      <div className="native-panel-header">
                        <h2 className="native-panel-title">Session API Credentials</h2>
                      </div>
                      <p style={{ margin: 0, fontSize: "12.5px", color: "var(--text-secondary)" }}>
                        Start sharing RAM in <strong>Node &amp; Model</strong> (or click{" "}
                        <strong>Overview</strong>) to issue your local OpenAI Base URL and
                        Session API Token.
                      </p>
                    </section>
                  )
                )}
              </div>
            )}
          </div>
        </main>
      </div>

      {/* =================================================================== */}
      {/* Bottom Native OS Statusbar                                          */}
      {/* =================================================================== */}
      <footer className="native-statusbar">
        <span
          data-testid="status-message"
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {statusMessage}
        </span>
        <span
          data-testid="footer-copyright"
          style={{ whiteSpace: "nowrap", color: "var(--text-muted)" }}
        >
          © {new Date().getFullYear()} ShardAi (MIT)
        </span>
      </footer>
    </div>
  );
};

export default Dashboard;
