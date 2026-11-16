import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { NodeStatusResponse, StartNodeRequest } from "../types/node";

const LOCAL_PYTHON_API = "http://127.0.0.1:8000";
const MIN_RAM_GB = 2;
const MAX_RAM_GB = 16;

const SUPPORTED_MODELS = [
  {
    id: "meta-llama/Llama-3.1-8B-Instruct",
    label: "Meta Llama 3.1 8B Instruct (Q4_K_M GGUF)",
  },
  {
    id: "Qwen/Qwen2.5-7B-Instruct",
    label: "Qwen 2.5 7B Instruct (Q4_K_M GGUF)",
  },
] as const;

interface SseTokenEvent {
  request_id: string;
  step: number;
  token_id: number;
  token: string;
  is_eos: boolean;
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
  const [selectedModel, setSelectedModel] = useState<string>(
    "meta-llama/Llama-3.1-8B-Instruct"
  );
  const [sharedRamGb, setSharedRamGb] = useState<number>(4);
  const [isSharing, setIsSharing] = useState<boolean>(false);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Idle — Select a model and allocate RAM to join ShardAi."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Model download progress bar state
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [downloadStatusMsg, setDownloadStatusMsg] = useState<string>("");

  // Prompt & SSE token streaming state
  const [promptText, setPromptText] = useState<string>(
    "Explain how ShardAi streams hidden state tensors across P2P nodes."
  );
  const [streamedOutput, setStreamedOutput] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const handleRamChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setSharedRamGb(Number(event.target.value));
  };

  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    setSelectedModel(event.target.value);
    setDownloadProgress(null);
    setDownloadStatusMsg("");
  };

  const streamModelDownload = async (modelId: string): Promise<void> => {
    setDownloadProgress(0);
    setDownloadStatusMsg(`Checking cache / downloading ${modelId}...`);

    let backendAvailable = true;
    try {
      const probe = await fetch(`${LOCAL_PYTHON_API}/models/supported`);
      if (!probe.ok) backendAvailable = false;
    } catch {
      backendAvailable = false;
    }

    if (!backendAvailable && !isTauriEnvironment()) {
      // Simulate live progress bar if testing UI in browser without backend running
      const steps = [20, 50, 75, 95, 100];
      for (const pct of steps) {
        await new Promise((resolve) => setTimeout(resolve, 80));
        setDownloadProgress(pct);
        setDownloadStatusMsg(`Simulating GGUF download (${pct}%)...`);
      }
      return;
    }

    const response = await fetch(
      `${LOCAL_PYTHON_API}/models/download-stream?model_id=${encodeURIComponent(modelId)}`
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

  const handleToggleSharing = async (): Promise<void> => {
    setIsBusy(true);
    setErrorMessage(null);

    if (!isSharing) {
      try {
        // Step 1: Ensure Python core engine is active
        setStatusMessage("Starting local Python core engine...");
        const coreStartupMsg = await invokeTauriCommand<string>("start_python_core");

        // Step 2: Download / verify GGUF model shard with live Progress Bar
        setStatusMessage("Checking and preparing model weights...");
        await streamModelDownload(selectedModel);

        // Step 3: Register node with both ram_gb and model_id
        setStatusMessage(
          `${coreStartupMsg} Initializing model and announcing to Cloudflare Tracker...`
        );
        const payload: StartNodeRequest = {
          node_id: "shardai-client-01",
          model_id: selectedModel,
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
          const response = await fetch(`${LOCAL_PYTHON_API}/start-node`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          if (!response.ok) {
            throw new Error(`Python API /start-node failed with HTTP ${response.status}`);
          }

          data = (await response.json()) as NodeStatusResponse;
        } catch (fetchErr: unknown) {
          if (!isTauriEnvironment()) {
            data = {
              state: "RUNNING",
              node_id: payload.node_id,
              model_id: payload.model_id,
              backend: payload.backend,
              allocated_ram_mb: payload.max_ram_mb,
              allocated_vram_mb: payload.max_vram_mb,
              p2p_listen_port: payload.p2p_listen_port,
              connected_peers_count: 2,
              peers: [],
              message: "Node simulated in browser preview mode",
            };
          } else {
            throw fetchErr;
          }
        }

        setIsSharing(true);
        setStatusMessage(
          `Active (${data.state}) — Model: ${selectedModel.split("/")[1]} | ${sharedRamGb} GB RAM | Peers: ${data.connected_peers_count}`
        );
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        setErrorMessage(message);
        setStatusMessage("Failed to start sharing.");
        try {
          await invokeTauriCommand<string>("stop_python_core");
        } catch {
          // Ignore secondary cleanup error
        }
      } finally {
        setIsBusy(false);
      }
    } else {
      try {
        setStatusMessage("Releasing RAM/VRAM shards and disconnecting peers...");
        try {
          await fetch(`${LOCAL_PYTHON_API}/stop-node`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          });
        } catch {
          // Proceed to kill the process even if the HTTP endpoint is already unreachable
        }

        const stopMsg = await invokeTauriCommand<string>("stop_python_core");
        setIsSharing(false);
        setStreamedOutput("");
        setDownloadProgress(null);
        setDownloadStatusMsg("");
        setStatusMessage(`Stopped — ${stopMsg}`);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        setErrorMessage(message);
      } finally {
        setIsBusy(false);
      }
    }
  };

  const handleGenerateStream = async (): Promise<void> => {
    if (!promptText.trim() || isGenerating) {
      return;
    }
    setIsGenerating(true);
    setErrorMessage(null);
    setStreamedOutput("");

    try {
      const response = await fetch(`${LOCAL_PYTHON_API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: promptText,
          max_tokens: 19,
          compress_activations: true,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Pipeline /generate failed with HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith("data:")) {
            continue;
          }
          const dataStr = line.slice(5).trim();
          if (dataStr === "[DONE]") {
            break;
          }
          const parsed = JSON.parse(dataStr) as SseTokenEvent;
          setStreamedOutput((prev) => prev + parsed.token);
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setErrorMessage(message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <section
      aria-label="ShardAi Node Dashboard"
      style={{
        maxWidth: "500px",
        margin: "48px auto",
        padding: "28px",
        borderRadius: "12px",
        border: "1px solid #e2e8f0",
        fontFamily: "system-ui, -apple-system, sans-serif",
        backgroundColor: "#ffffff",
        color: "#0f172a",
        boxShadow: "0 4px 12px rgba(15, 23, 42, 0.05)",
      }}
    >
      <header style={{ marginBottom: "24px" }}>
        <h1 style={{ fontSize: "1.35rem", margin: "0 0 6px 0", fontWeight: 600 }}>
          ShardAi Node
        </h1>
        <p style={{ margin: 0, fontSize: "0.875rem", color: "#64748b" }}>
          Distributed P2P LLM Inference Client
        </p>
      </header>

      {/* Model Selection Dropdown */}
      <div style={{ marginBottom: "20px" }}>
        <label
          htmlFor="model-select"
          style={{
            display: "block",
            fontSize: "0.9rem",
            fontWeight: 500,
            marginBottom: "8px",
          }}
        >
          Select AI Model
        </label>
        <select
          id="model-select"
          data-testid="model-select"
          value={selectedModel}
          disabled={isSharing || isBusy}
          onChange={handleModelChange}
          style={{
            width: "100%",
            padding: "10px 12px",
            borderRadius: "8px",
            border: "1px solid #cbd5e1",
            fontSize: "0.875rem",
            backgroundColor: isSharing || isBusy ? "#f8fafc" : "#ffffff",
            cursor: isSharing || isBusy ? "not-allowed" : "pointer",
            color: "#0f172a",
          }}
        >
          {SUPPORTED_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Shared RAM Slider */}
      <div style={{ marginBottom: "24px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <label htmlFor="ram-slider" style={{ fontSize: "0.9rem", fontWeight: 500 }}>
            Shared RAM Allocation
          </label>
          <span
            data-testid="selected-ram-label"
            style={{ fontSize: "0.95rem", fontWeight: 600, color: "#2563eb" }}
          >
            {sharedRamGb} GB
          </span>
        </div>

        <input
          id="ram-slider"
          type="range"
          min={MIN_RAM_GB}
          max={MAX_RAM_GB}
          step={1}
          value={sharedRamGb}
          disabled={isSharing || isBusy}
          onChange={handleRamChange}
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
            color: "#94a3b8",
            marginTop: "4px",
          }}
        >
          <span>{MIN_RAM_GB} GB</span>
          <span>{MAX_RAM_GB} GB</span>
        </div>
      </div>

      {/* Real-Time Model Download Progress Bar */}
      {downloadProgress !== null && (
        <div style={{ marginBottom: "20px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.825rem",
              marginBottom: "6px",
            }}
          >
            <span
              data-testid="model-download-status-msg"
              style={{ color: "#475569", fontWeight: 500 }}
            >
              {downloadStatusMsg || "Downloading Model Weights..."}
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

      {/* Toggle Sharing Button */}
      <button
        type="button"
        onClick={() => void handleToggleSharing()}
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
        {isBusy ? "Processing..." : isSharing ? "Stop Sharing" : "Start Sharing"}
      </button>

      {/* Distributed Inference Section */}
      {isSharing && (
        <div
          style={{
            marginTop: "24px",
            paddingTop: "20px",
            borderTop: "1px solid #f1f5f9",
          }}
        >
          <label
            htmlFor="prompt-input"
            style={{
              display: "block",
              fontSize: "0.85rem",
              fontWeight: 600,
              marginBottom: "8px",
            }}
          >
            P2P Pipeline Prompt (SSE Token Stream)
          </label>
          <textarea
            id="prompt-input"
            rows={2}
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            disabled={isGenerating}
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "8px 10px",
              borderRadius: "6px",
              border: "1px solid #cbd5e1",
              fontSize: "0.85rem",
              marginBottom: "10px",
              fontFamily: "inherit",
            }}
          />
          <button
            type="button"
            onClick={() => void handleGenerateStream()}
            disabled={isGenerating || !promptText.trim()}
            style={{
              width: "100%",
              padding: "9px 14px",
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "#ffffff",
              backgroundColor: "#0f172a",
              border: "none",
              borderRadius: "6px",
              cursor: isGenerating ? "wait" : "pointer",
            }}
          >
            {isGenerating ? "Streaming Tokens..." : "Run Distributed Inference"}
          </button>

          {streamedOutput && (
            <div
              data-testid="sse-token-output"
              style={{
                marginTop: "12px",
                padding: "10px 12px",
                borderRadius: "6px",
                backgroundColor: "#f8fafc",
                border: "1px solid #e2e8f0",
                fontSize: "0.85rem",
                lineHeight: 1.5,
                color: "#1e293b",
              }}
            >
              {streamedOutput}
            </div>
          )}
        </div>
      )}

      <footer style={{ marginTop: "20px", fontSize: "0.825rem", color: "#475569" }}>
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
