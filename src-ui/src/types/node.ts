/**
 * Strictly typed contracts matching `core-engine/shard_engine/schemas.py`.
 * Complies with `.antigravity/rules.md` (TypeScript strict mode, zero `any`).
 */

export type NodeState = "STOPPED" | "STARTING" | "RUNNING" | "STOPPING" | "ERROR";

export type ComputeBackend = "llama_cpp" | "pytorch";

export type NetworkMode = "lan" | "internet";

export type PeerRole = "primary" | "backup";

export type ModelSourceType = "catalog" | "huggingface" | "direct_url" | "local_file";

export interface CatalogModelItem {
  id: string;
  name: string;
  repo_id: string;
  filename?: string | null;
  estimated_ram_gb: number;
  enabled: boolean;
}

export interface ModelCatalogResponse {
  source: "cloud" | "cache" | "default";
  count: number;
  models: CatalogModelItem[];
}

/** Auto-fallback connectivity tier: Cloud -> LAN -> Solo (matches `NetworkStatus` in schemas.py). */
export type NetworkStatus = "online" | "lan_only" | "offline_solo";

export interface NetworkStatusResponse {
  network_status: NetworkStatus;
  tracker_reachable: boolean;
  tracker_url: string;
  lan_peers_count: number;
  zeroconf_available: boolean;
}

export interface PeerNodeInfo {
  peer_id: string;
  ip_address: string;
  port: number;
  shard_layer_start: number;
  shard_layer_end: number;
  role?: PeerRole;
  available_vram_mb: number;
  room_id: string;
}

export interface RoomLobbyItem {
  room_id: string;
  model: string;
  nodes_online: number;
  total_shared_ram_gb: number;
  is_private: boolean;
  network_scope: "internet" | "lan" | "hybrid";
  coverage_percent?: number;
  redundancy_factor?: number;
}

export interface SystemMemoryResponse {
  total_ram_gb: number;
  available_ram_gb: number;
  max_shareable_ram_gb: number;
}

export interface StartNodeRequest {
  node_id: string;
  model_id: string;
  source_type?: ModelSourceType;
  filename?: string | null;
  network_mode: NetworkMode;
  room_id: string;
  room_key?: string | null;
  is_unlisted?: boolean;
  backend: ComputeBackend;
  ram_gb: number;
  shared_ram_gb: number;
  max_ram_mb: number;
  max_vram_mb: number;
  p2p_listen_port: number;
  shard_layer_start: number;
  shard_layer_end: number;
  allow_ssd_offload?: boolean;
}

export interface NodeStatusResponse {
  state: NodeState;
  node_id: string | null;
  model_id: string | null;
  model_path?: string | null;
  network_mode?: NetworkMode | null;
  room_id?: string | null;
  is_private?: boolean;
  is_unlisted?: boolean;
  role?: PeerRole;
  backend: ComputeBackend | null;
  n_layers?: number;
  total_layers?: number;
  shard_layer_start?: number | null;
  shard_layer_end?: number | null;
  allocated_ram_mb: number;
  allocated_vram_mb: number;
  p2p_listen_port: number | null;
  connected_peers_count: number;
  peers: PeerNodeInfo[];
  session_token?: string | null;
  api_base_url?: string | null;
  coverage_percent?: number;
  is_swarm_ready?: boolean;
  network_status?: NetworkStatus | null;
  is_solo_mode?: boolean;
  use_mmap?: boolean;
  ssd_offload_active?: boolean;
  source_type?: ModelSourceType | null;
  model_fingerprint?: string | null;
  message: string;
}
