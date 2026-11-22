import type {
  CheckRoomJoinRequest,
  CheckRoomJoinResponse,
  NetworkStatusResponse,
  NodeStatusResponse,
  OpenAIChatMessagePayload,
  StartNodeRequest,
} from "../types/node";

export const LOCAL_ENGINE_BASE_URL = "http://127.0.0.1:8000";

export function isAbortError(err: unknown): boolean {
  return (
    (typeof DOMException !== "undefined" &&
      err instanceof DOMException &&
      err.name === "AbortError") ||
    (err instanceof Error && err.name === "AbortError")
  );
}

export class RoomJoinRejectedError extends Error {
  public readonly statusCode: number;

  constructor(statusCode: number, message: string) {
    super(message);
    this.name = "RoomJoinRejectedError";
    this.statusCode = statusCode;
  }
}

const VIETNAMESE_DIACRITICS_RE =
  /[\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D2-\u00D5\u00D9-\u00DA\u00DD\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01B0\u1EA0-\u1EF9]/;

export function normalizeNoticeToEnglish(
  message: string,
  fallback: string,
  errorCode?: string
): string {
  const cleaned = message.trim();
  if (!cleaned) {
    return fallback;
  }
  if (!VIETNAMESE_DIACRITICS_RE.test(cleaned)) {
    return cleaned;
  }

  const roomMatch = cleaned.match(/Room\s+"([^"]+)"/i);
  const roomLabel = roomMatch ? `Room "${roomMatch[1]}"` : "This room";
  const pctMatch = cleaned.match(/(\d+(?:\.\d+)?)%\s*RAM/i);

  const normalizedCode = (errorCode ?? "").trim().toUpperCase();
  if (normalizedCode === "ROOM_FULL" || cleaned.includes("100% RAM")) {
    return `${roomLabel} has reached 100% RAM/Layer capacity and is locked for active inference. Please select another room or create a new room.`;
  }
  if (normalizedCode === "ROOM_QUEUE_BUSY") {
    const pctPart = pctMatch ? ` (${pctMatch[1]}% RAM)` : "";
    return `${roomLabel}${pctPart} currently has another client joining and syncing layers. Please wait or create a new room.`;
  }
  if (normalizedCode === "FORBIDDEN") {
    return "Invalid room key for this private room.";
  }

  return fallback;
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
      let errorCode: string | undefined;
      if (typeof record.error === "object" && record.error !== null) {
        const errObj = record.error as Record<string, unknown>;
        if (typeof errObj.code === "string" && errObj.code.trim()) {
          errorCode = errObj.code.trim();
        }
      }
      if (typeof record.detail === "string" && record.detail.trim()) {
        return normalizeNoticeToEnglish(record.detail, fallback, errorCode);
      }
      if (typeof record.message === "string" && record.message.trim()) {
        return normalizeNoticeToEnglish(record.message, fallback, errorCode);
      }
      if (typeof record.error === "object" && record.error !== null) {
        const errObj = record.error as Record<string, unknown>;
        if (typeof errObj.message === "string" && errObj.message.trim()) {
          return normalizeNoticeToEnglish(errObj.message, fallback, errorCode);
        }
      }
    }
    return normalizeNoticeToEnglish(rawText, fallback);
  } catch {
    return fallback;
  }
}

export async function checkRoomJoin(
  payload: CheckRoomJoinRequest,
  options?: { signal?: AbortSignal }
): Promise<CheckRoomJoinResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/rooms/check-join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    ...(options?.signal ? { signal: options.signal } : {}),
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

export async function fetchNodeStatus(options?: {
  signal?: AbortSignal;
}): Promise<NodeStatusResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/status`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    ...(options?.signal ? { signal: options.signal } : {}),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch node status: HTTP ${response.status}`);
  }
  return (await response.json()) as NodeStatusResponse;
}

export async function fetchNetworkStatus(options?: {
  signal?: AbortSignal;
}): Promise<NetworkStatusResponse> {
  const response = await fetch(`${LOCAL_ENGINE_BASE_URL}/network-status`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    ...(options?.signal ? { signal: options.signal } : {}),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch network status: HTTP ${response.status}`);
  }
  return (await response.json()) as NetworkStatusResponse;
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

export interface StreamChatCompletionOptions {
  apiBaseUrl: string;
  apiKey: string;
  model: string;
  messages: OpenAIChatMessagePayload[];
  onToken: (token: string) => void;
  signal?: AbortSignal;
  temperature?: number;
  maxTokens?: number;
}

function extractDeltaContent(parsed: unknown): {
  token: string | null;
  errorMessage: string | null;
} {
  if (typeof parsed !== "object" || parsed === null) {
    return { token: null, errorMessage: null };
  }
  const record = parsed as Record<string, unknown>;
  if (typeof record.error === "object" && record.error !== null) {
    const errObj = record.error as Record<string, unknown>;
    if (typeof errObj.message === "string" && errObj.message.trim()) {
      const code = typeof errObj.code === "string" ? errObj.code : undefined;
      return {
        token: null,
        errorMessage: normalizeNoticeToEnglish(
          errObj.message,
          "Chat inference failed on the local gateway.",
          code
        ),
      };
    }
  }

  if (Array.isArray(record.choices) && record.choices.length > 0) {
    const firstChoice = record.choices[0];
    if (typeof firstChoice === "object" && firstChoice !== null) {
      const choiceRec = firstChoice as Record<string, unknown>;
      if (typeof choiceRec.delta === "object" && choiceRec.delta !== null) {
        const deltaRec = choiceRec.delta as Record<string, unknown>;
        if (typeof deltaRec.content === "string" && deltaRec.content.length > 0) {
          return { token: deltaRec.content, errorMessage: null };
        }
      }
      if (typeof choiceRec.message === "object" && choiceRec.message !== null) {
        const msgRec = choiceRec.message as Record<string, unknown>;
        if (typeof msgRec.content === "string" && msgRec.content.length > 0) {
          return { token: msgRec.content, errorMessage: null };
        }
      }
    }
  }
  return { token: null, errorMessage: null };
}

export async function streamChatCompletion(
  options: StreamChatCompletionOptions
): Promise<string> {
  const normalizedBase = options.apiBaseUrl.replace(/\/+$/, "");
  const endpoint = `${normalizedBase}/chat/completions`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${options.apiKey}`,
    },
    body: JSON.stringify({
      model: options.model,
      messages: options.messages,
      stream: true,
      ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
      ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
    }),
    signal: options.signal,
  });

  if (!response.ok) {
    const fallback =
      response.status === 503
        ? "Model not ready — swarm does not have 100% RAM coverage yet."
        : response.status === 401
          ? "Unauthorized — session API key is invalid or expired."
          : `Chat request failed: HTTP ${response.status}`;
    const detail = await extractApiErrorDetail(response, fallback);
    throw new Error(detail);
  }

  const contentType = (response.headers.get("content-type") ?? "").toLowerCase();
  if (contentType.includes("application/json") || !response.body) {
    const data: unknown = await response.json();
    const { token, errorMessage } = extractDeltaContent(data);
    if (errorMessage) {
      throw new Error(errorMessage);
    }
    if (token) {
      options.onToken(token);
      return token;
    }
    return "";
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let fullContent = "";

  const processEventBlock = (block: string): boolean => {
    const lines = block.split(/\r?\n/);
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) {
        continue;
      }
      const payload = line.slice(5).trim();
      if (!payload) {
        continue;
      }
      if (payload === "[DONE]") {
        return true;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        continue;
      }
      const { token, errorMessage } = extractDeltaContent(parsed);
      if (errorMessage) {
        throw new Error(errorMessage);
      }
      if (token) {
        fullContent += token;
        options.onToken(token);
      }
    }
    return false;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        if (buffer.trim()) {
          processEventBlock(buffer);
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        if (processEventBlock(block)) {
          return fullContent;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return fullContent;
}
