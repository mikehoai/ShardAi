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

const LOCAL_ENGINE_ROOT = "http://127.0.0.1:8000";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/v1";
const MIN_RAM_GB = 2;
const DEFAULT_MAX_RAM_GB = 16;
const NETWORK_STATUS_POLL_MS = 10000;
const GITHUB_REPO_URL = "https://github.com/mikehoai/ShardAi";

const NETWORK_STATUS_META: Record<
  NetworkStatus,
  { icon: string; label: string; color: string; background: string; border: string }
> = {
  online: {
    icon: "🟢",
    label: "Online (Cloud)",
    color: "#166534",
    background: "#f0fdf4",
    border: "#86efac",
  },
  lan_only: {
    icon: "🟡",
    label: "LAN Only",
    color: "#854d0e",
    background: "#fefce8",
    border: "#fde047",
  },
  offline_solo: {
    icon: "🔴",
    label: "Offline (Solo)",
    color: "#991b1b",
    background: "#fef2f2",
    border: "#fca5a5",
  },
};

type ModelTab = "catalog" | "custom" | "local";

function inferCustomSourceType(rawInput: string): ModelSourceType {
  const lowered = rawInput.trim().toLowerCase();
  if (
    (lowered.startsWith("http://") || lowered.startsWith("https://")) &&
    (!lowered.includes("huggingface.co/") ||
      lowered.includes("/resolve/") ||
      lowered.endsWith(".gguf"))
  ) {
    return "direct_url";
  }
  return "huggingface";
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
  // Network Rooms Lobby State
  const [availableRooms, setAvailableRooms] = useState<RoomLobbyItem[]>([]);
  const [isLoadingRooms, setIsLoadingRooms] = useState<boolean>(false);

  // Private Room Key Modal State (when clicking Join on a locked room)
  const [pendingPrivateRoom, setPendingPrivateRoom] = useState<RoomLobbyItem | null>(null);
  const [privateKeyInput, setPrivateKeyInput] = useState<string>("");

  // Create New Room / Join Unlisted Modal State
  const [showRoomModal, setShowRoomModal] = useState<boolean>(false);
  const [modalRoomId, setModalRoomId] = useState<string>("studio-room-1");
  const [modalModel, setModalModel] = useState<string>("meta-llama/Llama-3.1-8B-Instruct");
  const [modalRoomKey, setModalRoomKey] = useState<string>("");
  const [modalUnlisted, setModalUnlisted] = useState<boolean>(false);
  const [invitePasteInput, setInvitePasteInput] = useState<string>("");

  // 3-Tab Model Selection State (Online Catalog / Custom HF or URL / Local File)
  const [modelTab, setModelTab] = useState<ModelTab>("catalog");
  const [catalogModels, setCatalogModels] = useState<CatalogModelItem[]>([]);
  const [catalogSource, setCatalogSource] = useState<"cloud" | "cache" | "default" | null>(
    null
  );
  const [isLoadingCatalog, setIsLoadingCatalog] = useState<boolean>(false);
  const [selectedCatalogId, setSelectedCatalogId] = useState<string>("");
  const [modelInput, setModelInput] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
  const [customFilename, setCustomFilename] = useState<string>("");
  const [localFilePath, setLocalFilePath] = useState<string>("");
  const localFileInputRef = useRef<HTMLInputElement | null>(null);
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
  const [statusMessage, setStatusMessage] = useState<string>(
    "Idle — Select a room from the Lobby or configure your own node below."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Model Download Progress Bar State
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [downloadStatusMsg, setDownloadStatusMsg] = useState<string>("");

  // Clipboard copy feedback
  const [copiedField, setCopiedField] = useState<string | null>(null);

  const fetchAvailableRooms = useCallback(async (): Promise<void> => {
    setIsLoadingRooms(true);
    try {
      const res = await fetch(`${LOCAL_ENGINE_ROOT}/available-rooms`);
      if (res.ok) {
        const rooms = (await res.json()) as unknown;
        if (Array.isArray(rooms)) {
          setAvailableRooms(rooms as RoomLobbyItem[]);
        }
      }
    } catch {
      // Keep existing lobby state if local core engine is not yet running
    } finally {
      setIsLoadingRooms(false);
    }
  }, []);

  const fetchNetworkStatus = useCallback(async (): Promise<void> => {
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
        }
      }
    } catch {
      // Engine not reachable yet; keep the last known indicator
    }
  }, []);

  // Network Status Indicator: probe on mount and every 10s (Cloud -> LAN -> Solo)
  useEffect(() => {
    void fetchNetworkStatus();
    const timerId = window.setInterval(() => {
      void fetchNetworkStatus();
    }, NETWORK_STATUS_POLL_MS);
    return () => {
      window.clearInterval(timerId);
    };
  }, [fetchNetworkStatus]);

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
        }
      }
    } catch {
      // Engine may not be reachable yet
    } finally {
      setIsLoadingCatalog(false);
    }
  }, []);

  const applyNodeNetworkFields = (statusData: NodeStatusResponse): void => {
    if (statusData.network_status) {
      setNetworkStatus(statusData.network_status);
    }
    setSsdOffloadActive(Boolean(statusData.ssd_offload_active));
    setModelFingerprint(statusData.model_fingerprint ?? null);
  };

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
  }, [isSharing]);

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
    filename?: string | null
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

    const response = await fetch(
      `${LOCAL_ENGINE_ROOT}/models/download-stream?${params.toString()}`
    );
    if (!response.ok || !response.body) {
      throw new Error(`Failed to stream model download: HTTP ${response.status}`);
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

  const resolveActiveModelSelection = (): {
    model: string;
    sourceType: ModelSourceType;
    filename: string | null;
  } => {
    if (modelTab === "catalog") {
      const selected = catalogModels.find((m) => m.id === selectedCatalogId) ?? catalogModels[0];
      if (selected) {
        return {
          model: selected.repo_id,
          sourceType: "catalog",
          filename: selected.filename ?? null,
        };
      }
      return {
        model: modelInput.trim(),
        sourceType: "catalog",
        filename: null,
      };
    }
    if (modelTab === "local") {
      return {
        model: localFilePath.trim(),
        sourceType: "local_file",
        filename: null,
      };
    }
    const customModel = modelInput.trim();
    const detectedType = inferCustomSourceType(customModel);
    return {
      model: customModel,
      sourceType: detectedType,
      filename: detectedType === "huggingface" ? customFilename.trim() || null : null,
    };
  };

  const handleBrowseGgufFile = async (): Promise<void> => {
    setErrorMessage(null);
    if (isTauriEnvironment()) {
      try {
        const picked = await openFileDialog({
          multiple: false,
          directory: false,
          filters: [{ name: "GGUF Model", extensions: ["gguf"] }],
        });
        if (typeof picked === "string" && picked.trim()) {
          setLocalFilePath(picked.trim());
          setDownloadProgress(null);
          setDownloadStatusMsg("");
          return;
        }
        if (picked === null) {
          return;
        }
      } catch {
        // Fallback to native HTML file picker if dialog plugin is unavailable
      }
    }
    localFileInputRef.current?.click();
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

    try {
      setStatusMessage("Starting local Python core engine...");
      await invokeTauriCommand<string>("start_python_core");

      setStatusMessage("Checking and preparing model weights...");
      await streamModelDownload(targetModel, targetSourceType, targetFilename);

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
        if (!response.ok) {
          const errText = await response.text();
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
      setStatusMessage("Failed to start P2P AI Gateway.");
      try {
        await invokeTauriCommand<string>("stop_python_core");
      } catch {
        // Ignore secondary cleanup error
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
      try {
        await fetch(`${LOCAL_ENGINE_ROOT}/stop-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });
      } catch {
        // Proceed to terminate Python process even if HTTP endpoint is unreachable
      }

      const stopMsg = await invokeTauriCommand<string>("stop_python_core");
      setIsSharing(false);
      setSessionToken(null);
      setAssignedRole("primary");
      setAssignedLayerRange(null);
      setCoveragePercent(0);
      setIsSwarmReady(false);
      setSsdOffloadActive(false);
      setModelFingerprint(null);
      setDownloadProgress(null);
      setDownloadStatusMsg("");
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
    setModelInput(room.model);
    setRoomId(room.room_id);
    setRoomKey("");
    setIsUnlisted(false);
    setNetworkMode(mode);
    void startSharingWithConfig({
      model: room.model,
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

    setPendingPrivateRoom(null);
    setModelInput(target.model);
    setRoomId(target.room_id);
    setRoomKey(secret);
    setIsUnlisted(false);
    setNetworkMode(mode);
    void startSharingWithConfig({
      model: target.model,
      room: target.room_id,
      key: secret,
      unlisted: false,
      mode,
    });
  };

  const handleCreateOrJoinUnlisted = (event: React.FormEvent): void => {
    event.preventDefault();
    const cleanRoom = modalRoomId.trim() || "private-room";
    const cleanModel = modalModel.trim() || "meta-llama/Llama-3.1-8B-Instruct";
    const cleanKey = modalRoomKey.trim();

    setShowRoomModal(false);
    setModelInput(cleanModel);
    setRoomId(cleanRoom);
    setRoomKey(cleanKey);
    setIsUnlisted(modalUnlisted);
    void startSharingWithConfig({
      model: cleanModel,
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

    setShowRoomModal(false);
    setRoomId(parsedRoom);
    setRoomKey(parsedKey ?? "");
    setIsUnlisted(true);
    void startSharingWithConfig({
      model: modelInput,
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

  return (
    <section
      aria-label="ShardAi P2P AI Gateway"
      style={{
        maxWidth: "720px",
        margin: "32px auto",
        padding: "28px",
        borderRadius: "14px",
        border: "1px solid #e2e8f0",
        fontFamily: "system-ui, -apple-system, sans-serif",
        backgroundColor: "#ffffff",
        color: "#0f172a",
        boxShadow: "0 6px 20px rgba(15, 23, 42, 0.06)",
      }}
    >
      <header style={{ marginBottom: "22px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "8px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <img
              src="/logo.png"
              alt="ShardAi Logo"
              data-testid="shardai-logo"
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "8px",
                border: "1px solid #e2e8f0",
                objectFit: "contain",
                backgroundColor: "#ffffff",
                padding: "2px",
              }}
            />
            <h1 style={{ fontSize: "1.4rem", margin: 0, fontWeight: 700 }}>
              ShardAi Gateway
            </h1>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            {(() => {
              const meta = networkStatus
                ? NETWORK_STATUS_META[networkStatus]
                : {
                    icon: "⚪",
                    label: "Checking Network...",
                    color: "#475569",
                    background: "#f8fafc",
                    border: "#cbd5e1",
                  };
              return (
                <span
                  data-testid="network-status-indicator"
                  title={
                    networkStatus === "lan_only"
                      ? `${lanPeersCount} LAN peer(s) discovered via mDNS/zeroconf`
                      : meta.label
                  }
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    padding: "4px 10px",
                    borderRadius: "999px",
                    backgroundColor: meta.background,
                    color: meta.color,
                    border: `1px solid ${meta.border}`,
                    whiteSpace: "nowrap",
                  }}
                >
                  {meta.icon} {meta.label}
                  {networkStatus === "lan_only" && lanPeersCount > 0
                    ? ` (${lanPeersCount})`
                    : ""}
                </span>
              );
            })()}
            <span
              data-testid="gateway-badge"
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                padding: "4px 10px",
                borderRadius: "999px",
                backgroundColor: isSharing ? "#dcfce7" : "#f1f5f9",
                color: isSharing ? "#15803d" : "#475569",
              }}
            >
              {isSharing ? "ACTIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
        <p style={{ margin: "6px 0 0 0", fontSize: "0.85rem", color: "#64748b" }}>
          Discover active compute rooms or launch your own P2P OpenAI-compatible gateway.
        </p>
      </header>

      {/* ===================================================================== */}
      {/* 1. Network Rooms Lobby                                                */}
      {/* ===================================================================== */}
      <div
        data-testid="network-rooms-lobby"
        style={{
          marginBottom: "24px",
          padding: "16px",
          borderRadius: "10px",
          backgroundColor: "#f8fafc",
          border: "1px solid #e2e8f0",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "8px",
            marginBottom: "12px",
          }}
        >
          <h2 style={{ fontSize: "0.95rem", margin: 0, fontWeight: 700 }}>
            Network Rooms Lobby
          </h2>
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              type="button"
              data-testid="refresh-rooms-btn"
              onClick={() => {
                void fetchAvailableRooms();
                void fetchNetworkStatus();
              }}
              disabled={isLoadingRooms}
              style={{
                padding: "6px 12px",
                fontSize: "0.78rem",
                fontWeight: 600,
                borderRadius: "6px",
                border: "1px solid #cbd5e1",
                backgroundColor: "#ffffff",
                color: "#334155",
                cursor: isLoadingRooms ? "wait" : "pointer",
              }}
            >
              {isLoadingRooms ? "Scanning..." : "Refresh"}
            </button>
            <button
              type="button"
              data-testid="create-room-modal-btn"
              disabled={isSharing || isBusy}
              onClick={() => setShowRoomModal(true)}
              style={{
                padding: "6px 12px",
                fontSize: "0.78rem",
                fontWeight: 600,
                borderRadius: "6px",
                border: "none",
                backgroundColor: "#0f172a",
                color: "#ffffff",
                cursor: isSharing || isBusy ? "not-allowed" : "pointer",
              }}
            >
              Create New Room / Join Unlisted
            </button>
          </div>
        </div>

        {isCloudOffline ? (
          <div
            data-testid="offline-rooms-hidden-notice"
            style={{
              padding: "14px",
              borderRadius: "8px",
              backgroundColor: "#ffffff",
              border: "1px dashed #cbd5e1",
              fontSize: "0.82rem",
              color: "#475569",
            }}
          >
            {networkStatus === "lan_only" ? (
              <>
                <div style={{ fontWeight: 600, color: "#854d0e", marginBottom: "6px" }}>
                  🟡 Public Cloud Rooms hidden in LAN Only mode (Cloudflare Tracker unreachable).
                </div>
                {availableRooms.filter((r) => r.network_scope === "lan").length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "8px" }}>
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
                            backgroundColor: "#fefce8",
                            border: "1px solid #fde047",
                          }}
                        >
                          <span>
                            <strong>{room.room_id}</strong> ({room.model}) — {room.nodes_online} LAN node(s), {room.total_shared_ram_gb} GB
                          </span>
                          <button
                            type="button"
                            data-testid={`join-room-${room.room_id}`}
                            disabled={isSharing || isBusy}
                            onClick={() => handleJoinLobbyRoom(room)}
                            style={{
                              padding: "4px 10px",
                              fontSize: "0.75rem",
                              fontWeight: 600,
                              borderRadius: "6px",
                              border: "none",
                              backgroundColor: "#2563eb",
                              color: "#ffffff",
                              cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                            }}
                          >
                            Join LAN
                          </button>
                        </div>
                      ))}
                  </div>
                ) : (
                  <span>Configure a room below or wait for mDNS/Zeroconf LAN peers.</span>
                )}
              </>
            ) : (
              <span>
                🔴 <strong>Offline (Solo Mode)</strong> — Public Rooms list is hidden while offline. Click{" "}
                <strong>Start Sharing &amp; Join Room</strong> below to run standalone (with optional SSD Offloading).
              </span>
            )}
          </div>
        ) : (
        <div style={{ overflowX: "auto" }}>
          <table
            data-testid="rooms-lobby-table"
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "0.82rem",
              backgroundColor: "#ffffff",
              borderRadius: "8px",
              overflow: "hidden",
              border: "1px solid #e2e8f0",
            }}
          >
            <thead>
              <tr style={{ backgroundColor: "#f1f5f9", textAlign: "left", color: "#475569" }}>
                <th style={{ padding: "8px 10px" }}>Room Name</th>
                <th style={{ padding: "8px 10px" }}>Model</th>
                <th style={{ padding: "8px 10px" }}>Nodes Online</th>
                <th style={{ padding: "8px 10px" }}>Total Shared RAM</th>
                <th style={{ padding: "8px 10px" }}>Resources</th>
                <th style={{ padding: "8px 10px" }}>Status</th>
                <th style={{ padding: "8px 10px", textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {availableRooms.length === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    style={{
                      padding: "16px",
                      textAlign: "center",
                      color: "#64748b",
                    }}
                  >
                    No active rooms discovered yet. Click "Refresh" or start sharing below.
                  </td>
                </tr>
              ) : (
                availableRooms.map((room) => {
                  const coverage = room.coverage_percent ?? 100;
                  const redundancy = room.redundancy_factor ?? 1;
                  const isReady = coverage >= 100;
                  return (
                  <tr key={room.room_id} style={{ borderTop: "1px solid #f1f5f9" }}>
                    <td style={{ padding: "9px 10px", fontWeight: 600 }}>
                      {room.room_id}{" "}
                      <span
                        style={{
                          fontSize: "0.68rem",
                          padding: "2px 6px",
                          borderRadius: "4px",
                          backgroundColor:
                            room.network_scope === "lan" ? "#dbeafe" : "#e2e8f0",
                          color: "#1e40af",
                          marginLeft: "4px",
                        }}
                      >
                        {room.network_scope.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: "9px 10px", color: "#334155" }}>{room.model}</td>
                    <td style={{ padding: "9px 10px" }}>{room.nodes_online}</td>
                    <td style={{ padding: "9px 10px" }}>{room.total_shared_ram_gb} GB</td>
                    <td style={{ padding: "9px 10px" }}>
                      <span
                        data-testid={`room-coverage-${room.room_id}`}
                        style={{
                          display: "inline-block",
                          padding: "3px 8px",
                          borderRadius: "999px",
                          fontSize: "0.73rem",
                          fontWeight: 600,
                          backgroundColor: isReady ? "#dcfce7" : "#fef3c7",
                          color: isReady ? "#166534" : "#92400e",
                          border: isReady ? "1px solid #86efac" : "1px solid #fcd34d",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {isReady
                          ? `✅ Sẵn sàng (Dự phòng: ${redundancy}x)`
                          : `⚠️ Đang gom RAM: ${Math.round(coverage)}%`}
                      </span>
                    </td>
                    <td style={{ padding: "9px 10px" }}>
                      {room.is_private ? "🔒 Private" : "Public"}
                    </td>
                    <td style={{ padding: "9px 10px", textAlign: "right" }}>
                      <button
                        type="button"
                        data-testid={`join-room-${room.room_id}`}
                        disabled={isSharing || isBusy}
                        onClick={() => handleJoinLobbyRoom(room)}
                        style={{
                          padding: "5px 12px",
                          fontSize: "0.76rem",
                          fontWeight: 600,
                          borderRadius: "6px",
                          border: "none",
                          backgroundColor: "#2563eb",
                          color: "#ffffff",
                          cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                          opacity: isSharing || isBusy ? 0.5 : 1,
                        }}
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
      </div>

      {/* ===================================================================== */}
      {/* Private Room Key Prompt Modal                                         */}
      {/* ===================================================================== */}
      {pendingPrivateRoom !== null && (
        <div
          role="dialog"
          aria-label="Enter Private Room Key"
          data-testid="private-room-key-modal"
          style={{
            marginBottom: "22px",
            padding: "16px",
            borderRadius: "10px",
            backgroundColor: "#fffbeb",
            border: "1px solid #fcd34d",
          }}
        >
          <form onSubmit={handleConfirmPrivateJoin}>
            <h3 style={{ margin: "0 0 8px 0", fontSize: "0.95rem" }}>
              🔒 Enter Room Key for "{pendingPrivateRoom.room_id}"
            </h3>
            <p style={{ margin: "0 0 10px 0", fontSize: "0.8rem", color: "#78350f" }}>
              Key is SHA-256 hashed locally before verifying with the Signaling Tracker.
            </p>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                type="password"
                data-testid="private-key-modal-input"
                value={privateKeyInput}
                onChange={(e) => setPrivateKeyInput(e.target.value)}
                placeholder="Enter secret room key..."
                autoFocus
                style={{
                  flex: 1,
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "0.85rem",
                }}
              />
              <button
                type="submit"
                data-testid="confirm-private-join-btn"
                style={{
                  padding: "8px 14px",
                  borderRadius: "6px",
                  border: "none",
                  backgroundColor: "#2563eb",
                  color: "#ffffff",
                  fontWeight: 600,
                  fontSize: "0.82rem",
                  cursor: "pointer",
                }}
              >
                Unlock & Join
              </button>
              <button
                type="button"
                onClick={() => setPendingPrivateRoom(null)}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  backgroundColor: "#ffffff",
                  fontSize: "0.82rem",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* ===================================================================== */}
      {/* Create New Room / Join Unlisted Modal                                 */}
      {/* ===================================================================== */}
      {showRoomModal && (
        <div
          role="dialog"
          aria-label="Create New Room or Join Unlisted"
          data-testid="create-room-modal"
          style={{
            marginBottom: "22px",
            padding: "18px",
            borderRadius: "10px",
            backgroundColor: "#f8fafc",
            border: "1px solid #94a3b8",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px" }}>
            <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 700 }}>
              Create New Room / Join Unlisted
            </h3>
            <button
              type="button"
              onClick={() => setShowRoomModal(false)}
              style={{
                border: "none",
                background: "transparent",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              ✕
            </button>
          </div>

          <form onSubmit={handleCreateOrJoinUnlisted}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "10px" }}>
              <div>
                <label style={{ display: "block", fontSize: "0.78rem", fontWeight: 600, marginBottom: "4px" }}>
                  New Room Name
                </label>
                <input
                  type="text"
                  data-testid="modal-room-name-input"
                  value={modalRoomId}
                  onChange={(e) => setModalRoomId(e.target.value)}
                  placeholder="my-private-cluster"
                  style={{
                    width: "100%",
                    boxSizing: "border-box",
                    padding: "8px 10px",
                    borderRadius: "6px",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.84rem",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: "0.78rem", fontWeight: 600, marginBottom: "4px" }}>
                  Room Key (Optional — 🔒 Private)
                </label>
                <input
                  type="text"
                  data-testid="modal-room-key-input"
                  value={modalRoomKey}
                  onChange={(e) => setModalRoomKey(e.target.value)}
                  placeholder="Leave blank for Public"
                  style={{
                    width: "100%",
                    boxSizing: "border-box",
                    padding: "8px 10px",
                    borderRadius: "6px",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.84rem",
                  }}
                />
              </div>
            </div>

            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", fontSize: "0.78rem", fontWeight: 600, marginBottom: "4px" }}>
                Model
              </label>
              <input
                type="text"
                list="suggested-models"
                value={modalModel}
                onChange={(e) => setModalModel(e.target.value)}
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "0.84rem",
                }}
              />
            </div>

            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "0.82rem",
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

            {/* Invite String Copy Box */}
            <div style={{ marginBottom: "12px" }}>
              <span style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, color: "#64748b", marginBottom: "4px" }}>
                Invite String (Share with teammates)
              </span>
              <div style={{ display: "flex", gap: "8px" }}>
                <code
                  data-testid="invite-string-preview"
                  style={{
                    flex: 1,
                    padding: "7px 10px",
                    borderRadius: "6px",
                    backgroundColor: "#ffffff",
                    border: "1px solid #cbd5e1",
                    fontSize: "0.8rem",
                  }}
                >
                  {generatedInviteString}
                </code>
                <button
                  type="button"
                  onClick={() => void handleCopy("invite", generatedInviteString)}
                  style={{
                    padding: "7px 12px",
                    borderRadius: "6px",
                    border: "1px solid #cbd5e1",
                    backgroundColor: "#ffffff",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {copiedField === "invite" ? "Copied!" : "Copy Invite"}
                </button>
              </div>
            </div>

            <button
              type="submit"
              data-testid="modal-create-join-btn"
              style={{
                width: "100%",
                padding: "9px 14px",
                borderRadius: "6px",
                border: "none",
                backgroundColor: "#2563eb",
                color: "#ffffff",
                fontWeight: 600,
                fontSize: "0.85rem",
                cursor: "pointer",
              }}
            >
              Create & Start Sharing
            </button>
          </form>

          {/* Paste Invite String Quick Join */}
          <div style={{ marginTop: "14px", paddingTop: "12px", borderTop: "1px solid #e2e8f0" }}>
            <span style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, color: "#64748b", marginBottom: "6px" }}>
              Or Paste Invite String (RoomID | Key)
            </span>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                type="text"
                data-testid="paste-invite-input"
                value={invitePasteInput}
                onChange={(e) => setInvitePasteInput(e.target.value)}
                placeholder="e.g. studio-room-1 | secret-pass"
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  fontSize: "0.82rem",
                }}
              />
              <button
                type="button"
                onClick={handleJoinByInviteString}
                style={{
                  padding: "7px 14px",
                  borderRadius: "6px",
                  border: "none",
                  backgroundColor: "#0f172a",
                  color: "#ffffff",
                  fontWeight: 600,
                  fontSize: "0.8rem",
                  cursor: "pointer",
                }}
              >
                Join Unlisted
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* 2. Node Configuration Form (3-Tab Model Selector)                     */}
      {/* ===================================================================== */}
      <div style={{ marginBottom: "18px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "8px",
            marginBottom: "8px",
          }}
        >
          <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>Model Source</span>
          {catalogSource !== null && modelTab === "catalog" && (
            <span
              data-testid="catalog-source-badge"
              style={{
                fontSize: "0.72rem",
                fontWeight: 600,
                padding: "2px 8px",
                borderRadius: "999px",
                backgroundColor: catalogSource === "cloud" ? "#dcfce7" : "#fefce8",
                color: catalogSource === "cloud" ? "#166534" : "#854d0e",
                border:
                  catalogSource === "cloud" ? "1px solid #86efac" : "1px solid #fde047",
              }}
            >
              {catalogSource === "cloud"
                ? "Synced from Cloud"
                : catalogSource === "cache"
                  ? "Loaded from ~/.shardai/models_cache.json"
                  : "Default Offline Catalog"}
            </span>
          )}
        </div>

        {/* 3 Model Selection Tabs */}
        <div
          role="tablist"
          aria-label="Model Source Tabs"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "6px",
            marginBottom: "10px",
          }}
        >
          {(
            [
              { id: "catalog", label: "Online Catalog" },
              { id: "custom", label: "Custom HF / URL" },
              { id: "local", label: "Local File" },
            ] as const
          ).map((tab) => {
            const active = modelTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={active}
                data-testid={`tab-${tab.id}`}
                disabled={isSharing || isBusy}
                onClick={() => {
                  setModelTab(tab.id);
                  setDownloadProgress(null);
                  setDownloadStatusMsg("");
                }}
                style={{
                  padding: "8px 10px",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  borderRadius: "8px",
                  border: active ? "1px solid #2563eb" : "1px solid #cbd5e1",
                  backgroundColor: active ? "#eff6ff" : "#f8fafc",
                  color: active ? "#1d4ed8" : "#475569",
                  cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                }}
              >
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab 1: Online Catalog */}
        {modelTab === "catalog" && (
          <div data-testid="panel-catalog" style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <select
              aria-label="Online Model Catalog"
              data-testid="catalog-model-select"
              value={selectedCatalogId}
              disabled={isSharing || isBusy || catalogModels.length === 0}
              onChange={(e) => {
                const chosenId = e.target.value;
                setSelectedCatalogId(chosenId);
                const found = catalogModels.find((m) => m.id === chosenId);
                if (found) {
                  setModelInput(found.repo_id);
                }
                setDownloadProgress(null);
                setDownloadStatusMsg("");
              }}
              style={{
                flex: 1,
                padding: "10px 12px",
                borderRadius: "8px",
                border: "1px solid #cbd5e1",
                fontSize: "0.875rem",
                backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
                color: "#0f172a",
              }}
            >
              {catalogModels.length === 0 ? (
                <option value="">
                  {isLoadingCatalog ? "Loading model catalog..." : "No catalog models loaded yet — click Refresh"}
                </option>
              ) : (
                catalogModels.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} (~{item.estimated_ram_gb} GB)
                  </option>
                ))
              )}
            </select>
            <button
              type="button"
              data-testid="refresh-catalog-btn"
              onClick={() => void fetchModelCatalog()}
              disabled={isLoadingCatalog || isSharing || isBusy}
              style={{
                padding: "10px 14px",
                fontSize: "0.8rem",
                fontWeight: 600,
                borderRadius: "8px",
                border: "1px solid #cbd5e1",
                backgroundColor: "#ffffff",
                color: "#334155",
                cursor: isLoadingCatalog || isSharing || isBusy ? "wait" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {isLoadingCatalog ? "Syncing..." : "Refresh"}
            </button>
          </div>
        )}

        {/* Tab 2: Custom HF Repo / External Direct URL */}
        {modelTab === "custom" && (
          <div data-testid="panel-custom" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <input
              id="model-input"
              data-testid="model-input"
              type="text"
              value={modelInput}
              disabled={isSharing || isBusy}
              onChange={(e) => {
                setModelInput(e.target.value);
                setDownloadProgress(null);
                setDownloadStatusMsg("");
              }}
              placeholder="HuggingFace Repo (e.g. bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) or Direct .gguf URL"
              style={{
                width: "100%",
                boxSizing: "border-box",
                padding: "10px 12px",
                borderRadius: "8px",
                border: "1px solid #cbd5e1",
                fontSize: "0.875rem",
                backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
                color: "#0f172a",
              }}
            />
            {inferCustomSourceType(modelInput) === "huggingface" && (
              <input
                type="text"
                data-testid="custom-filename-input"
                value={customFilename}
                disabled={isSharing || isBusy}
                onChange={(e) => setCustomFilename(e.target.value)}
                placeholder="Optional .gguf filename (leave blank to auto-pick Q4_K_M)"
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  border: "1px solid #e2e8f0",
                  fontSize: "0.8rem",
                  backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
                  color: "#334155",
                }}
              />
            )}
          </div>
        )}

        {/* Tab 3: Local File (.gguf via Tauri File Dialog) */}
        {modelTab === "local" && (
          <div data-testid="panel-local" style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <input
              type="text"
              data-testid="local-file-input"
              value={localFilePath}
              disabled={isSharing || isBusy}
              onChange={(e) => {
                setLocalFilePath(e.target.value);
                setDownloadProgress(null);
                setDownloadStatusMsg("");
              }}
              placeholder="Select or paste absolute path to local .gguf file..."
              style={{
                flex: 1,
                padding: "10px 12px",
                borderRadius: "8px",
                border: "1px solid #cbd5e1",
                fontSize: "0.85rem",
                backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
                color: "#0f172a",
              }}
            />
            <input
              ref={localFileInputRef}
              type="file"
              accept=".gguf"
              aria-hidden="true"
              tabIndex={-1}
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  const fileWithPath = file as File & { path?: string };
                  setLocalFilePath(fileWithPath.path ?? file.name);
                  setDownloadProgress(null);
                  setDownloadStatusMsg("");
                }
              }}
            />
            <button
              type="button"
              data-testid="browse-gguf-btn"
              disabled={isSharing || isBusy}
              onClick={() => void handleBrowseGgufFile()}
              style={{
                padding: "10px 14px",
                fontSize: "0.8rem",
                fontWeight: 600,
                borderRadius: "8px",
                border: "none",
                backgroundColor: "#0f172a",
                color: "#ffffff",
                cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              Browse (.gguf)
            </button>
          </div>
        )}
      </div>

      {/* RAM Slider */}
      <div style={{ marginBottom: "20px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "6px",
          }}
        >
          <label htmlFor="ram-slider" style={{ fontSize: "0.875rem", fontWeight: 600 }}>
            Shared RAM Contribution
          </label>
          <span
            data-testid="selected-ram-label"
            style={{ fontSize: "0.95rem", fontWeight: 700, color: "#2563eb" }}
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
            cursor: isSharing || isBusy ? "not-allowed" : "pointer",
          }}
        />

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.75rem",
            color: "#64748b",
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
            borderRadius: "8px",
            backgroundColor: allowSsdOffload ? "#eff6ff" : "#f8fafc",
            border: allowSsdOffload ? "1px solid #93c5fd" : "1px solid #e2e8f0",
            fontSize: "0.8rem",
            color: "#334155",
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

      {/* Network Mode (LAN / Internet), Room ID & Optional Room Key */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "10px",
          marginBottom: "20px",
        }}
      >
        <div>
          <span
            style={{
              display: "block",
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: "6px",
            }}
          >
            Network
          </span>
          <div style={{ display: "flex", gap: "6px" }}>
            {(["lan", "internet"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                data-testid={`network-mode-${mode}`}
                disabled={isSharing || isBusy}
                onClick={() => setNetworkMode(mode)}
                style={{
                  flex: 1,
                  padding: "9px 6px",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  borderRadius: "8px",
                  border:
                    networkMode === mode ? "1px solid #2563eb" : "1px solid #cbd5e1",
                  backgroundColor: networkMode === mode ? "#eff6ff" : "#ffffff",
                  color: networkMode === mode ? "#1d4ed8" : "#475569",
                  cursor: isSharing || isBusy ? "not-allowed" : "pointer",
                }}
              >
                {mode === "lan" ? "LAN" : "Internet"}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label
            htmlFor="room-id-input"
            style={{
              display: "block",
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: "6px",
            }}
          >
            Room ID
          </label>
          <input
            id="room-id-input"
            data-testid="room-id-input"
            type="text"
            value={roomId}
            disabled={isSharing || isBusy}
            onChange={(e) => setRoomId(e.target.value)}
            placeholder="public"
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "9px 10px",
              borderRadius: "8px",
              border: "1px solid #cbd5e1",
              fontSize: "0.85rem",
              backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
              color: "#0f172a",
            }}
          />
        </div>

        <div>
          <label
            htmlFor="room-key-input"
            style={{
              display: "block",
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: "6px",
            }}
          >
            Room Key (Optional)
          </label>
          <input
            id="room-key-input"
            data-testid="room-key-input"
            type="password"
            value={roomKey}
            disabled={isSharing || isBusy}
            onChange={(e) => setRoomKey(e.target.value)}
            placeholder="Public if blank"
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "9px 10px",
              borderRadius: "8px",
              border: "1px solid #cbd5e1",
              fontSize: "0.85rem",
              backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
              color: "#0f172a",
            }}
          />
        </div>
      </div>

      {/* Model Download Progress Bar */}
      {downloadProgress !== null && (
        <div style={{ marginBottom: "18px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.8rem",
              marginBottom: "6px",
            }}
          >
            <span data-testid="model-download-status-msg" style={{ color: "#475569" }}>
              {downloadStatusMsg || "Preparing model..."}
            </span>
            <span
              data-testid="model-download-percent"
              style={{ fontWeight: 600, color: "#2563eb" }}
            >
              {downloadProgress.toFixed(0)}%
            </span>
          </div>
          <div
            style={{
              width: "100%",
              height: "8px",
              backgroundColor: "#e2e8f0",
              borderRadius: "4px",
              overflow: "hidden",
            }}
          >
            <div
              data-testid="model-download-progress-bar"
              style={{
                width: `${downloadProgress}%`,
                height: "100%",
                backgroundColor: downloadProgress >= 100 ? "#16a34a" : "#2563eb",
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
        style={{
          width: "100%",
          padding: "12px 16px",
          fontSize: "0.95rem",
          fontWeight: 600,
          color: "#ffffff",
          backgroundColor: isSharing ? "#dc2626" : "#2563eb",
          border: "none",
          borderRadius: "8px",
          cursor: isBusy ? "wait" : "pointer",
          opacity: isBusy ? 0.7 : 1,
        }}
      >
        {isBusy
          ? "Connecting..."
          : isSharing
            ? "Stop Sharing"
            : "Start Sharing & Join Room"}
      </button>

      {/* ===================================================================== */}
      {/* 3. Active Node Role & Session API Credentials Panel                   */}
      {/* ===================================================================== */}
      {isSharing && sessionToken && (
        <div
          data-testid="session-credentials-panel"
          style={{
            marginTop: "22px",
            padding: "18px",
            borderRadius: "10px",
            backgroundColor: "#f8fafc",
            border: "1px solid #cbd5e1",
          }}
        >
          <div
            data-testid="active-node-role-card"
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "10px 12px",
              marginBottom: "14px",
              borderRadius: "8px",
              backgroundColor: assignedRole === "primary" ? "#eff6ff" : "#fffbeb",
              border:
                assignedRole === "primary" ? "1px solid #93c5fd" : "1px solid #fcd34d",
            }}
          >
            <span
              data-testid="active-node-role-label"
              style={{
                fontSize: "0.85rem",
                fontWeight: 700,
                color: assignedRole === "primary" ? "#1d4ed8" : "#92400e",
              }}
            >
              {assignedRole === "primary"
                ? `Role: Primary Seeder (Layers ${assignedLayerRange?.[0] ?? 0} - ${assignedLayerRange?.[1] ?? 0})`
                : `Role: Backup Node (Layers ${assignedLayerRange?.[0] ?? 0} - ${assignedLayerRange?.[1] ?? 0})`}
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
              {modelFingerprint && (
                <span
                  data-testid="model-fingerprint-badge"
                  title={`SHA-256 (10MB + size): ${modelFingerprint}`}
                  style={{
                    fontSize: "0.7rem",
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: "999px",
                    backgroundColor: "#f1f5f9",
                    color: "#334155",
                    border: "1px solid #cbd5e1",
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
                    fontSize: "0.72rem",
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: "999px",
                    backgroundColor: "#f3e8ff",
                    color: "#6b21a8",
                    border: "1px solid #d8b4fe",
                  }}
                >
                  💾 SSD Offload (mmap)
                </span>
              )}
              <span
                style={{
                  fontSize: "0.72rem",
                  fontWeight: 600,
                  padding: "2px 8px",
                  borderRadius: "999px",
                  backgroundColor: assignedRole === "primary" ? "#dbeafe" : "#fef3c7",
                  color: assignedRole === "primary" ? "#1e40af" : "#78350f",
                }}
              >
                {assignedRole === "primary" ? "PRIMARY" : "BACKUP"}
              </span>
            </div>
          </div>

          <h2 style={{ fontSize: "0.95rem", margin: "0 0 12px 0", fontWeight: 700, color: "#0f172a" }}>
            Session API Credentials
          </h2>

          <div style={{ marginBottom: "12px" }}>
            <span style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, color: "#64748b", marginBottom: "4px" }}>
              OpenAI Base URL
            </span>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <code
                data-testid="gateway-base-url"
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  borderRadius: "6px",
                  backgroundColor: "#ffffff",
                  border: "1px solid #e2e8f0",
                  fontSize: "0.825rem",
                  color: "#0f172a",
                }}
              >
                {apiBaseUrl}
              </code>
              <button
                type="button"
                onClick={() => void handleCopy("baseUrl", apiBaseUrl)}
                style={{
                  padding: "7px 12px",
                  fontSize: "0.78rem",
                  fontWeight: 600,
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  backgroundColor: "#ffffff",
                  color: "#0f172a",
                  cursor: "pointer",
                }}
              >
                {copiedField === "baseUrl" ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          <div style={{ marginBottom: "14px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "8px",
                marginBottom: "6px",
              }}
            >
              <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#64748b" }}>
                API Key (Session Token)
              </span>
              <span
                data-testid="api-readiness-badge"
                style={{
                  fontSize: "0.74rem",
                  fontWeight: 600,
                  padding: "3px 9px",
                  borderRadius: "999px",
                  backgroundColor:
                    isSwarmReady && coveragePercent >= 100 ? "#dcfce7" : "#fef3c7",
                  color: isSwarmReady && coveragePercent >= 100 ? "#166534" : "#92400e",
                  border:
                    isSwarmReady && coveragePercent >= 100
                      ? "1px solid #86efac"
                      : "1px solid #fcd34d",
                }}
              >
                {isSwarmReady && coveragePercent >= 100
                  ? "🟢 API Ready (Mạng lưới đã sẵn sàng nhận request)"
                  : `🟡 API Standby (Đang chờ đủ RAM: ${Math.round(coveragePercent)}% - Chưa thể gọi model)`}
              </span>
            </div>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <code
                data-testid="session-token-value"
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  borderRadius: "6px",
                  backgroundColor: "#ffffff",
                  border: "1px solid #e2e8f0",
                  fontSize: "0.8rem",
                  color: "#0f172a",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {sessionToken}
              </code>
              <button
                type="button"
                onClick={() => void handleCopy("token", sessionToken)}
                style={{
                  padding: "7px 12px",
                  fontSize: "0.78rem",
                  fontWeight: 600,
                  borderRadius: "6px",
                  border: "1px solid #cbd5e1",
                  backgroundColor: "#ffffff",
                  color: "#0f172a",
                  cursor: "pointer",
                }}
              >
                {copiedField === "token" ? "Copied!" : "Copy"}
              </button>
              {(!isSwarmReady || coveragePercent < 100) && (
                <button
                  type="button"
                  data-testid="copy-invite-code-btn"
                  onClick={() =>
                    void handleCopy(
                      "activeInvite",
                      roomKey.trim()
                        ? `${roomId.trim() || "public"} | ${roomKey.trim()}`
                        : roomId.trim() || "public"
                    )
                  }
                  style={{
                    padding: "7px 12px",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                    borderRadius: "6px",
                    border: "1px solid #f59e0b",
                    backgroundColor: "#fffbeb",
                    color: "#92400e",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  {copiedField === "activeInvite" ? "Copied Invite!" : "Copy Invite Code"}
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
              <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#64748b" }}>
                Quick Test (curl / IDE Config)
              </span>
              <button
                type="button"
                onClick={() => void handleCopy("curl", curlExample)}
                style={{
                  padding: "3px 9px",
                  fontSize: "0.72rem",
                  fontWeight: 600,
                  borderRadius: "5px",
                  border: "1px solid #cbd5e1",
                  backgroundColor: "#ffffff",
                  color: "#334155",
                  cursor: "pointer",
                }}
              >
                {copiedField === "curl" ? "Copied!" : "Copy curl"}
              </button>
            </div>
            <pre
              data-testid="curl-snippet"
              style={{
                margin: 0,
                padding: "12px",
                borderRadius: "8px",
                backgroundColor: "#0f172a",
                color: "#e2e8f0",
                fontSize: "0.75rem",
                lineHeight: 1.45,
                overflowX: "auto",
              }}
            >
              <code>{curlExample}</code>
            </pre>
          </div>
        </div>
      )}

      <footer style={{ marginTop: "18px", fontSize: "0.825rem", color: "#475569" }}>
        <p data-testid="status-message" style={{ margin: "0 0 6px 0" }}>
          {statusMessage}
        </p>
        {errorMessage !== null && (
          <p
            role="alert"
            style={{
              margin: 0,
              padding: "8px 10px",
              borderRadius: "6px",
              backgroundColor: "#fef2f2",
              color: "#b91c1c",
            }}
          >
            {errorMessage}
          </p>
        )}

        <div
          style={{
            marginTop: "14px",
            paddingTop: "12px",
            borderTop: "1px solid #e2e8f0",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
            fontSize: "0.75rem",
            color: "#64748b",
          }}
        >
          <span data-testid="footer-copyright">
            © {new Date().getFullYear()} ShardAi — Open-Source P2P Distributed LLM Engine (MIT)
          </span>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="github-repo-link"
            aria-label="ShardAi on GitHub"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              padding: "5px 10px",
              borderRadius: "6px",
              border: "1px solid #cbd5e1",
              backgroundColor: "#f8fafc",
              color: "#0f172a",
              textDecoration: "none",
              fontWeight: 600,
              fontSize: "0.75rem",
            }}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </footer>
    </section>
  );
};

export default Dashboard;
