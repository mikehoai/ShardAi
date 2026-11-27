/**
 * Formatting helpers for reasoning/thinking blocks, token counting, and live chat header telemetry.
 */
import type { NetworkMode, NetworkStatus, P2PTransportKind } from "../types/node";

export interface ParsedThinkingContent {
  hasThinking: boolean;
  thinking: string;
  answer: string;
  isThinkingInProgress: boolean;
}

const OPEN_THINK_TAG_RE = /<(think|thought|thinking)\s*>/gi;

/**
 * Separate `<think>...</think>` / `<thought>...</thought>` / `<thinking>...</thinking>` blocks
 * from the final answer so DeepSeek and other reasoning models show only the final answer inline,
 * with reasoning tucked behind a collapsible header toggle.
 */
export function parseThinkingContent(
  rawContent: string,
  isStreaming: boolean = false
): ParsedThinkingContent {
  if (!rawContent) {
    return {
      hasThinking: false,
      thinking: "",
      answer: "",
      isThinkingInProgress: false,
    };
  }

  const thinkingParts: string[] = [];
  const answerParts: string[] = [];
  let cursor = 0;
  let isThinkingInProgress = false;

  OPEN_THINK_TAG_RE.lastIndex = 0;
  let openMatch = OPEN_THINK_TAG_RE.exec(rawContent);

  while (openMatch !== null) {
    const tagName = openMatch[1];
    const openStart = openMatch.index;
    const openEnd = OPEN_THINK_TAG_RE.lastIndex;

    if (openStart > cursor) {
      answerParts.push(rawContent.slice(cursor, openStart));
    }

    const closeRe = new RegExp(`</${tagName}\\s*>`, "i");
    const remainder = rawContent.slice(openEnd);
    const closeMatch = closeRe.exec(remainder);

    if (closeMatch === null) {
      const partialThinking = remainder.trim();
      if (partialThinking) {
        thinkingParts.push(partialThinking);
      }
      if (isStreaming) {
        isThinkingInProgress = true;
      }
      cursor = rawContent.length;
      break;
    }

    const innerThinking = remainder.slice(0, closeMatch.index).trim();
    if (innerThinking) {
      thinkingParts.push(innerThinking);
    }
    cursor = openEnd + closeMatch.index + closeMatch[0].length;
    OPEN_THINK_TAG_RE.lastIndex = cursor;
    openMatch = OPEN_THINK_TAG_RE.exec(rawContent);
  }

  if (cursor < rawContent.length) {
    answerParts.push(rawContent.slice(cursor));
  }

  const hasAnyTag = thinkingParts.length > 0 || isThinkingInProgress;
  if (!hasAnyTag) {
    return {
      hasThinking: false,
      thinking: "",
      answer: rawContent,
      isThinkingInProgress: false,
    };
  }

  return {
    hasThinking: true,
    thinking: thinkingParts.join("\n\n").trim(),
    answer: answerParts.join("").replace(/^\s*\n{2,}/, "").trim(),
    isThinkingInProgress,
  };
}

/** Deterministic fallback token estimate when exact SSE usage metadata is absent. */
export function estimateTokenCount(text: string): number {
  const cleaned = text.trim();
  if (!cleaned) {
    return 0;
  }
  return Math.max(1, Math.ceil(cleaned.length / 4));
}

/** Format token count compactly for message badges and session telemetry. */
export function formatTokenCount(count: number): string {
  const safe = Math.max(0, Math.round(count));
  return `${safe.toLocaleString("en-US")} tok`;
}

export interface TransportLabelParams {
  activeTransport?: P2PTransportKind | null;
  networkStatus?: NetworkStatus | null;
  networkMode?: NetworkMode | null;
  isSoloMode?: boolean;
  connectedPeersCount?: number;
  p2pQuicEnabled?: boolean;
  p2pDirectPeersCount?: number;
  relayConnected?: boolean;
}

/** Resolve a crisp human-readable label describing the active peer transport path. */
export function resolveConnectionTransportLabel(params: TransportLabelParams): string {
  const {
    activeTransport,
    networkStatus,
    networkMode,
    isSoloMode,
    connectedPeersCount = 0,
    p2pQuicEnabled = false,
    p2pDirectPeersCount = 0,
    relayConnected = false,
  } = params;

  switch (activeTransport) {
    case "solo":
      return "Local (Solo)";
    case "lan_direct":
      return "LAN Direct (TCP)";
    case "quic_direct":
      return "QUIC P2P (Direct)";
    case "quic_relay":
      return "QUIC P2P (Relay)";
    case "ws_relay":
      return "WebSocket Relay";
    case "tcp_direct":
      return "TCP Direct";
    default:
      break;
  }

  if (isSoloMode || networkStatus === "offline_solo" || connectedPeersCount === 0) {
    return "Local (Solo)";
  }
  if (p2pQuicEnabled) {
    return p2pDirectPeersCount > 0 ? "QUIC P2P (Direct)" : "QUIC P2P (Relay)";
  }
  if (networkStatus === "lan_only" || networkMode === "lan") {
    return "LAN Direct (TCP)";
  }
  if (relayConnected) {
    return "WebSocket Relay";
  }
  return "TCP Direct";
}

export interface NetworkSpeedParams {
  throughputMbps?: number | null;
  rttMs?: number | null;
  tokensPerSec?: number | null;
  isSolo?: boolean;
}

/** Format live network link throughput/RTT and inference speed cleanly for the chat header. */
export function formatNetworkSpeedSummary(params: NetworkSpeedParams): string {
  const parts: string[] = [];
  const { throughputMbps, rttMs, tokensPerSec, isSolo = false } = params;

  if (typeof throughputMbps === "number" && Number.isFinite(throughputMbps) && throughputMbps > 0) {
    parts.push(`${throughputMbps >= 100 ? throughputMbps.toFixed(0) : throughputMbps.toFixed(1)} Mbps`);
  }
  if (typeof rttMs === "number" && Number.isFinite(rttMs) && rttMs > 0) {
    parts.push(`${Math.round(rttMs)} ms`);
  }
  if (typeof tokensPerSec === "number" && Number.isFinite(tokensPerSec) && tokensPerSec > 0) {
    parts.push(`${tokensPerSec.toFixed(1)} tok/s`);
  }

  if (parts.length > 0) {
    return parts.join(" • ");
  }
  return isSolo ? "Local memory" : "Standby";
}
