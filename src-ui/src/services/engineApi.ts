import type { NodeStatusResponse, StartNodeRequest } from "../types/node";

export const LOCAL_ENGINE_BASE_URL = "http://127.0.0.1:8000";

export async function fetchNodeStatus(): Promise<NodeStatusResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/status`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch node status: HTTP ${response.status}`);
  }
  return (await response.json()) as NodeStatusResponse;
}

export async function startLocalNode(
  payload: StartNodeRequest
): Promise<NodeStatusResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/start-node`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Failed to start local node: HTTP ${response.status}`);
  }
  return (await response.json()) as NodeStatusResponse;
}

export async function stopLocalNode(): Promise<NodeStatusResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/stop-node`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Failed to stop local node: HTTP ${response.status}`);
  }
  return (await response.json()) as NodeStatusResponse;
}
