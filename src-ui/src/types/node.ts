/**
 * Strictly typed contracts matching `core-engine/shard_engine/schemas.py`.
 * Complies with `.antigravity/rules.md` (TypeScript strict mode, zero `any`).
 */

export type NodeState = "STOPPED" | "STARTING" | "RUNNING" | "STOPPING" | "ERROR";

export type ComputeBackend = "llama_cpp" | "pytorch";

export interface PeerNodeInfo {
  peer_id: string;
  ip_address: string;
  port: number;
  shard_layer_start: number;
  shard_layer_end: number;
  available_vram_mb: number;
}

export interface StartNodeRequest {
  node_id: string;
  model_id: string;
  backend: ComputeBackend;
  shared_ram_gb: number;
  max_ram_mb: number;
  max_vram_mb: number;
  p2p_listen_port: number;
  shard_layer_start: number;
  shard_layer_end: number;
}

export interface NodeStatusResponse {
  state: NodeState;
  node_id: string | null;
  model_id: string | null;
  backend: ComputeBackend | null;
  allocated_ram_mb: number;
  allocated_vram_mb: number;
  p2p_listen_port: number | null;
  connected_peers_count: number;
  peers: PeerNodeInfo[];
  message: string;
}
