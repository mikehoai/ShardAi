import type {
  CheckRoomJoinRequest,
  CheckRoomJoinResponse,
  NodeStatusResponse,
  StartNodeRequest,
} from "../types/node";

export const LOCAL_ENGINE_BASE_URL = "http://127.0.0.1:8000";

export class RoomJoinRejectedError extends Error {
  public readonly statusCode: number;

  constructor(statusCode: number, message: string) {
    super(message);
    this.name = "RoomJoinRejectedError";
    this.statusCode = statusCode;
  }
}

export async function extractApiErrorDetail(
  response: Response,
  fallback: string
): Promise<string> {
  try {
    const rawText = await response.text();
    if (!rawText.trim()) {
      return fallback;
    }
    const parsed: unknown = JSON.parse(rawText);
    if (typeof parsed === "object" && parsed !== null) {
      const record = parsed as Record<string, unknown>;
      if (typeof record.detail === "string" && record.detail.trim()) {
        return record.detail.trim();
      }
      if (typeof record.message === "string" && record.message.trim()) {
        return record.message.trim();
      }
      if (typeof record.error === "object" && record.error !== null) {
        const errObj = record.error as Record<string, unknown>;
        if (typeof errObj.message === "string" && errObj.message.trim()) {
          return errObj.message.trim();
        }
      }
    }
    return rawText.trim();
  } catch {
    return fallback;
  }
}

export async function checkRoomJoin(
  payload: CheckRoomJoinRequest
): Promise<CheckRoomJoinResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/rooms/check-join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.status === 403 || response.status === 409) {
    const detail = await extractApiErrorDetail(
      response,
      `Cannot join room '${payload.room_id}' (HTTP ${response.status}).`
    );
    throw new RoomJoinRejectedError(response.status, detail);
  }
  if (!response.ok) {
    const detail = await extractApiErrorDetail(
      response,
      `Failed to verify room availability: HTTP ${response.status}`
    );
    throw new Error(detail);
  }
  return (await response.json()) as CheckRoomJoinResponse;
}

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
