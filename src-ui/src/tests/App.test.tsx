import { describe, expect, it } from "vitest";
import type { NodeStatusResponse, StartNodeRequest } from "../types/node";

describe("ShardAi UI Contracts & Dashboard State", () => {
  it("constructs a valid StartNodeRequest with model_id and ram_gb (2GB - 16GB)", () => {
    const sharedRamGb = 8;
    const modelId = "Qwen/Qwen2.5-7B-Instruct";

    const request: StartNodeRequest = {
      node_id: "test-ui-node",
      model_id: modelId,
      backend: "llama_cpp",
      ram_gb: sharedRamGb,
      shared_ram_gb: sharedRamGb,
      max_ram_mb: sharedRamGb * 1024,
      max_vram_mb: 2048,
      p2p_listen_port: 49152,
      shard_layer_start: 0,
      shard_layer_end: 15,
    };

    expect(["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]).toContain(
      request.model_id
    );
    expect(request.ram_gb).toBe(8);
    expect(request.shared_ram_gb).toBeGreaterThanOrEqual(2);
    expect(request.shared_ram_gb).toBeLessThanOrEqual(16);
    expect(request.max_ram_mb).toBe(8192);
  });

  it("validates zero memory footprint when node is STOPPED", () => {
    const stoppedStatus: NodeStatusResponse = {
      state: "STOPPED",
      node_id: null,
      model_id: null,
      backend: null,
      allocated_ram_mb: 0,
      allocated_vram_mb: 0,
      p2p_listen_port: null,
      connected_peers_count: 0,
      peers: [],
      message: "Stopped",
    };

    expect(stoppedStatus.state).toBe("STOPPED");
    expect(stoppedStatus.allocated_ram_mb).toBe(0);
    expect(stoppedStatus.allocated_vram_mb).toBe(0);
  });
});
