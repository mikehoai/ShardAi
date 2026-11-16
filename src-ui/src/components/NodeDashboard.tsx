import React, { useState } from "react";
import type { NodeStatusResponse, StartNodeRequest } from "../types/node";
import { startLocalNode, stopLocalNode } from "../services/engineApi";

const INITIAL_STATUS: NodeStatusResponse = {
  state: "STOPPED",
  node_id: null,
  model_id: null,
  backend: null,
  allocated_ram_mb: 0,
  allocated_vram_mb: 0,
  p2p_listen_port: null,
  connected_peers_count: 0,
  peers: [],
  message: "Node is ready to share idle RAM/VRAM.",
};

export const NodeDashboard: React.FC = () => {
  const [status, setStatus] = useState<NodeStatusResponse>(INITIAL_STATUS);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async (): Promise<void> => {
    setError(null);
    const request: StartNodeRequest = {
      node_id: "shardai-client-01",
      model_id: "meta-llama/Llama-3.1-8B-Instruct",
      backend: "llama_cpp",
      shared_ram_gb: 4,
      max_ram_mb: 4096,
      max_vram_mb: 2048,
      p2p_listen_port: 49152,
      shard_layer_start: 0,
      shard_layer_end: 15,
    };
    try {
      const updated = await startLocalNode(request);
      setStatus(updated);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown engine error");
    }
  };

  const handleStop = async (): Promise<void> => {
    setError(null);
    try {
      const updated = await stopLocalNode();
      setStatus(updated);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown engine error");
    }
  };

  return (
    <section aria-label="ShardAi Node Control Panel">
      <h1>ShardAi P2P Node</h1>
      <p data-testid="node-state">Status: {status.state}</p>
      <p>
        Allocated Memory: {status.allocated_ram_mb} MB RAM / {status.allocated_vram_mb} MB VRAM
      </p>
      <p>Connected P2P Peers: {status.connected_peers_count}</p>
      {error !== null && <p role="alert">{error}</p>}
      <div>
        <button
          type="button"
          onClick={() => void handleStart()}
          disabled={status.state === "RUNNING"}
        >
          Start Node
        </button>
        <button
          type="button"
          onClick={() => void handleStop()}
          disabled={status.state === "STOPPED"}
        >
          Stop Node
        </button>
      </div>
    </section>
  );
};
