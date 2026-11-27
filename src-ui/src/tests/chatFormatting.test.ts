import { describe, expect, it } from "vitest";
import {
  estimateTokenCount,
  formatNetworkSpeedSummary,
  formatTokenCount,
  parseThinkingContent,
  resolveConnectionTransportLabel,
} from "../utils/chatFormatting";

describe("chatFormatting helpers", () => {
  it("parses completed and streaming DeepSeek <think> blocks cleanly", () => {
    const plain = parseThinkingContent("Hello **world**!");
    expect(plain).toEqual({
      hasThinking: false,
      thinking: "",
      answer: "Hello **world**!",
      isThinkingInProgress: false,
    });

    const completed = parseThinkingContent(
      "<think>\nStep 1: analyze\nStep 2: verify\n</think>\n\nThe final answer is **42**."
    );
    expect(completed.hasThinking).toBe(true);
    expect(completed.thinking).toBe("Step 1: analyze\nStep 2: verify");
    expect(completed.answer).toBe("The final answer is **42**.");
    expect(completed.isThinkingInProgress).toBe(false);

    const streaming = parseThinkingContent("<thought>Working on partial thought", true);
    expect(streaming.hasThinking).toBe(true);
    expect(streaming.thinking).toBe("Working on partial thought");
    expect(streaming.answer).toBe("");
    expect(streaming.isThinkingInProgress).toBe(true);
  });

  it("estimates and formats token counts", () => {
    expect(estimateTokenCount("")).toBe(0);
    expect(estimateTokenCount("Hi")).toBe(1);
    expect(estimateTokenCount("12345678")).toBe(2);
    expect(formatTokenCount(1250)).toBe("1,250 tok");
  });

  it("resolves connection transport labels and network speed summaries", () => {
    expect(
      resolveConnectionTransportLabel({
        activeTransport: "quic_direct",
        connectedPeersCount: 2,
      })
    ).toBe("QUIC P2P (Direct)");

    expect(
      resolveConnectionTransportLabel({
        networkStatus: "offline_solo",
        connectedPeersCount: 0,
      })
    ).toBe("Local (Solo)");

    expect(
      resolveConnectionTransportLabel({
        networkStatus: "online",
        connectedPeersCount: 1,
        p2pQuicEnabled: true,
        p2pDirectPeersCount: 0,
      })
    ).toBe("QUIC P2P (Relay)");

    expect(
      formatNetworkSpeedSummary({
        throughputMbps: 24.5,
        rttMs: 19,
        tokensPerSec: 18.2,
      })
    ).toBe("24.5 Mbps • 19 ms • 18.2 tok/s");

    expect(formatNetworkSpeedSummary({ isSolo: true })).toBe("Local memory");
  });
});
