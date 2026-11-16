import React, { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
  NetworkMode,
  NodeStatusResponse,
  RoomLobbyItem,
  StartNodeRequest,
  SystemMemoryResponse,
} from "../types/node";

const LOCAL_ENGINE_ROOT = "http://127.0.0.1:8000";
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/v1";
const MIN_RAM_GB = 2;
const DEFAULT_MAX_RAM_GB = 16;

const SUGGESTED_MODELS = [
  {
    id: "Qwen/Qwen2.5-0.5B-Instruct",
    label: "Qwen 2.5 0.5B (Fast CPU / Test)",
  },
  {
    id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    label: "SmolLM2 360M (Ultra-Light Test)",
  },
  {
    id: "meta-llama/Llama-3.1-8B-Instruct",
    label: "Llama 3.1 8B",
  },
  {
    id: "Qwen/Qwen2.5-7B-Instruct",
    label: "Qwen 2.5 7B",
  },
  {
    id: "mistralai/Mistral-7B-Instruct-v0.3",
    label: "Mistral 7B v0.3",
  },
] as const;

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

  // Configuration Form State
  const [modelInput, setModelInput] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
  const [networkMode, setNetworkMode] = useState<NetworkMode>("internet");
  const [roomId, setRoomId] = useState<string>("public");
  const [roomKey, setRoomKey] = useState<string>("");
  const [isUnlisted, setIsUnlisted] = useState<boolean>(false);
  const [availableRamGb, setAvailableRamGb] = useState<number | null>(null);
  const [maxRamGb, setMaxRamGb] = useState<number>(DEFAULT_MAX_RAM_GB);
  const [sharedRamGb, setSharedRamGb] = useState<number>(4);

  // Node Lifecycle & Gateway Credentials State
  const [isSharing, setIsSharing] = useState<boolean>(false);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [sessionToken, setSessionToken] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(DEFAULT_API_BASE_URL);
  const [activeModelId, setActiveModelId] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
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
        const rooms = (await res.json()) as RoomLobbyItem[];
        setAvailableRooms(rooms);
      }
    } catch {
      // Keep existing lobby state if local core engine is not yet running
    } finally {
      setIsLoadingRooms(false);
    }
  }, []);

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
        await fetchAvailableRooms();
      }
    };

    void initDashboard();
    return () => {
      cancelled = true;
    };
  }, [fetchAvailableRooms]);

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

  const streamModelDownload = async (modelId: string): Promise<void> => {
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

    const response = await fetch(
      `${LOCAL_ENGINE_ROOT}/models/download-stream?model_id=${encodeURIComponent(modelId)}&ram_gb=${sharedRamGb}`
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
        if (dataStr === "[DONE]") break;
        try {
          const payload = JSON.parse(dataStr) as SseDownloadEvent;
          setDownloadProgress(payload.progress);
          setDownloadStatusMsg(payload.message || `${payload.progress.toFixed(0)}%`);
        } catch {
          // ignore parsing error
        }
      }
    }
  };

  const startSharingWithConfig = async (
    override?: StartConfigOverride
  ): Promise<void> => {
    const targetModel = (override?.model ?? modelInput).trim();
    const targetRoom = (override?.room ?? roomId).trim() || "public";
    const targetKey = (override?.key ?? roomKey).trim();
    const targetUnlisted = override?.unlisted ?? isUnlisted;
    const targetMode = override?.mode ?? networkMode;

    if (!targetModel) {
      setErrorMessage("Please enter or select a valid HuggingFace Repo ID or local .gguf path.");
      return;
    }

    setIsBusy(true);
    setErrorMessage(null);

    try {
      setStatusMessage("Starting local Python core engine...");
      await invokeTauriCommand<string>("start_python_core");

      setStatusMessage("Checking and preparing model weights...");
      await streamModelDownload(targetModel);

      setStatusMessage(
        `Joining ${targetMode.toUpperCase()} room '${targetRoom}' & allocating ${sharedRamGb} GB RAM...`
      );

      const payload: StartNodeRequest = {
        node_id: "shardai-client-01",
        model_id: targetModel,
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
      };

      let data: NodeStatusResponse;
      try {
        const response = await fetch(`${LOCAL_ENGINE_ROOT}/start-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(`Python API /start-node failed (${response.status}): ${errText}`);
        }
        data = (await response.json()) as NodeStatusResponse;
      } catch (fetchErr: unknown) {
        if (!isTauriEnvironment()) {
          data = {
            state: "RUNNING",
            node_id: payload.node_id,
            model_id: payload.model_id,
            network_mode: payload.network_mode,
            room_id: payload.room_id,
            is_private: Boolean(targetKey),
            is_unlisted: targetUnlisted,
            backend: payload.backend,
            n_layers: 20,
            total_layers: 32,
            allocated_ram_mb: payload.max_ram_mb,
            allocated_vram_mb: payload.max_vram_mb,
            p2p_listen_port: payload.p2p_listen_port,
            connected_peers_count: 1,
            peers: [],
            session_token: "sk-shard-browser-preview-token",
            api_base_url: DEFAULT_API_BASE_URL,
            message: "Gateway simulated in browser preview mode",
          };
        } else {
          throw fetchErr;
        }
      }

      setIsSharing(true);
      setSessionToken(data.session_token ?? null);
      setApiBaseUrl(data.api_base_url ?? DEFAULT_API_BASE_URL);
      setActiveModelId(data.model_id ?? targetModel);

      const layersLabel =
        data.n_layers && data.total_layers
          ? `${data.n_layers}/${data.total_layers} layers`
          : `${sharedRamGb} GB RAM`;
      setStatusMessage(
        `Active & Connected (${(data.network_mode ?? targetMode).toUpperCase()} / Room: ${data.room_id ?? targetRoom}) — ${layersLabel} | Peers: ${data.connected_peers_count}`
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
      setDownloadProgress(null);
      setDownloadStatusMsg("");
      setStatusMessage(`Stopped — ${stopMsg}`);
      await fetchAvailableRooms();
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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 style={{ fontSize: "1.4rem", margin: 0, fontWeight: 700 }}>
            ShardAi Gateway
          </h1>
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
              onClick={() => void fetchAvailableRooms()}
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
                <th style={{ padding: "8px 10px" }}>Status</th>
                <th style={{ padding: "8px 10px", textAlign: "right" }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {availableRooms.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
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
                availableRooms.map((room) => (
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
                ))
              )}
            </tbody>
          </table>
        </div>
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
      {/* 2. Node Configuration Form                                            */}
      {/* ===================================================================== */}
      <div style={{ marginBottom: "18px" }}>
        <label
          htmlFor="model-input"
          style={{
            display: "block",
            fontSize: "0.875rem",
            fontWeight: 600,
            marginBottom: "6px",
          }}
        >
          Model (HuggingFace Repo / URL / Local .gguf Path)
        </label>
        <input
          id="model-input"
          data-testid="model-input"
          type="text"
          list="suggested-models"
          value={modelInput}
          disabled={isSharing || isBusy}
          onChange={(e) => {
            setModelInput(e.target.value);
            setDownloadProgress(null);
            setDownloadStatusMsg("");
          }}
          placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct or D:\models\model.gguf"
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
        <datalist id="suggested-models">
          {SUGGESTED_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </datalist>

        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginTop: "8px" }}>
          {SUGGESTED_MODELS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              disabled={isSharing || isBusy}
              onClick={() => setModelInput(preset.id)}
              style={{
                padding: "4px 10px",
                fontSize: "0.75rem",
                borderRadius: "6px",
                border:
                  modelInput === preset.id ? "1px solid #2563eb" : "1px solid #e2e8f0",
                backgroundColor: modelInput === preset.id ? "#eff6ff" : "#f8fafc",
                color: modelInput === preset.id ? "#1d4ed8" : "#475569",
                cursor: isSharing || isBusy ? "not-allowed" : "pointer",
              }}
            >
              {preset.label}
            </button>
          ))}
        </div>
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
      {/* 3. Session API Credentials Panel (Active & Connected Only)            */}
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
            <span style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, color: "#64748b", marginBottom: "4px" }}>
              API Key (Session Token)
            </span>
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
      </footer>
    </section>
  );
};

export default Dashboard;
