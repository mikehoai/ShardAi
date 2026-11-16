import React, { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { NodeStatusResponse, StartNodeRequest } from "../types/node";

const LOCAL_PYTHON_API = "http://127.0.0.1:8000";
const MIN_RAM_GB = 2;
const MAX_RAM_GB = 16;

export const Dashboard: React.FC = () => {
  const [sharedRamGb, setSharedRamGb] = useState<number>(4);
  const [isSharing, setIsSharing] = useState<boolean>(false);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>(
    "Idle — Ready to share RAM with the ShardAi network."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleRamChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setSharedRamGb(Number(event.target.value));
  };

  const handleToggleSharing = async (): Promise<void> => {
    setIsBusy(true);
    setErrorMessage(null);

    if (!isSharing) {
      try {
        // Step 1: Ask Rust Tauri shell to spawn `core-engine/main.py` in the background
        setStatusMessage("Starting local Python core engine...");
        const coreStartupMsg = await invoke<string>("start_python_core");

        // Step 2: Once Rust confirms the Python core is listening, call local FastAPI `/start-node`
        setStatusMessage(`${coreStartupMsg} Registering node & allocating ${sharedRamGb} GB RAM...`);
        const payload: StartNodeRequest = {
          node_id: "shardai-client-01",
          model_id: "meta-llama/Llama-3.1-8B-Instruct",
          backend: "llama_cpp",
          shared_ram_gb: sharedRamGb,
          max_ram_mb: sharedRamGb * 1024,
          max_vram_mb: 2048,
          p2p_listen_port: 49152,
          shard_layer_start: 0,
          shard_layer_end: 15,
        };

        const response = await fetch(`${LOCAL_PYTHON_API}/start-node`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Python API /start-node failed with HTTP ${response.status}`);
        }

        const data = (await response.json()) as NodeStatusResponse;
        setIsSharing(true);
        setStatusMessage(
          `Active (${data.state}) — Sharing ${sharedRamGb} GB RAM | Connected Peers: ${data.connected_peers_count}`
        );
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        setErrorMessage(message);
        setStatusMessage("Failed to start sharing.");
        // Ensure background Python process is cleaned up if `/start-node` fails
        try {
          await invoke<string>("stop_python_core");
        } catch {
          // Ignore secondary cleanup error
        }
      } finally {
        setIsBusy(false);
      }
    } else {
      try {
        // Step 1: Instruct FastAPI to release model shards & unregister from Cloudflare Tracker
        setStatusMessage("Releasing RAM/VRAM shards and disconnecting peers...");
        try {
          await fetch(`${LOCAL_PYTHON_API}/stop-node`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          });
        } catch {
          // Proceed to kill the process even if the HTTP endpoint is already unreachable
        }

        // Step 2: Ask Rust Tauri shell to terminate the background Python process
        const stopMsg = await invoke<string>("stop_python_core");
        setIsSharing(false);
        setStatusMessage(`Stopped — ${stopMsg}`);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        setErrorMessage(message);
      } finally {
        setIsBusy(false);
      }
    }
  };

  return (
    <section
      aria-label="ShardAi Node Dashboard"
      style={{
        maxWidth: "460px",
        margin: "64px auto",
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
          style={{ width: "100%", cursor: isSharing || isBusy ? "not-allowed" : "pointer" }}
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
        {isBusy
          ? "Processing..."
          : isSharing
            ? "Stop Sharing"
            : "Start Sharing"}
      </button>

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
