import React, { StrictMode } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const nestedSetterCallsDuringUpdater: unknown[] = [];
let updaterExecutionDepth = 0;
let functionalUpdaterCallCount = 0;

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();

  function wrappedUseState<S>(
    initialState?: S | (() => S)
  ): [S | undefined, React.Dispatch<React.SetStateAction<S | undefined>>] {
    const [state, rawSetState] = actual.useState(initialState);
    const wrappedSetState = actual.useCallback(
      (action: React.SetStateAction<S | undefined>) => {
        if (updaterExecutionDepth > 0) {
          nestedSetterCallsDuringUpdater.push(action);
        }
        if (typeof action === "function") {
          const updaterFn = action as (prevState: S | undefined) => S | undefined;
          rawSetState((prevState) => {
            functionalUpdaterCallCount += 1;
            updaterExecutionDepth += 1;
            try {
              return updaterFn(prevState);
            } finally {
              updaterExecutionDepth -= 1;
            }
          });
          return;
        }
        rawSetState(action);
      },
      [rawSetState]
    );
    return [state, wrappedSetState];
  }

  const typedUseState = wrappedUseState as typeof actual.useState;
  const patchedReact = {
    ...actual,
    useState: typedUseState,
  };
  return {
    ...patchedReact,
    default: patchedReact,
  };
});

import { ChatWindow } from "../components/ChatWindow";

describe("ChatWindow StrictMode Pure State Updaters (Issue #74)", () => {
  beforeEach(() => {
    nestedSetterCallsDuringUpdater.length = 0;
    updaterExecutionDepth = 0;
    functionalUpdaterCallCount = 0;
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  it("updates message tokenCount/tokensPerSec and latestTokensPerSec on stream completion in StrictMode without side effects inside setMessages", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    let nowMs = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);

    let sseController: ReadableStreamDefaultController<Uint8Array> | null = null;
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementationOnce(async () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          sseController = controller;
        },
      });
      return new Response(stream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    });

    render(
      <StrictMode>
        <ChatWindow
          isOpen={true}
          isModelReady={true}
          coveragePercent={100}
          apiBaseUrl="http://127.0.0.1:8000/v1"
          sessionToken="sk-strict-pure-test"
          modelId="Qwen/Qwen2.5-7B-Instruct"
          isSoloMode={true}
          onClose={vi.fn()}
        />
      </StrictMode>
    );

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Verify pure updater behavior" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(sseController).not.toBeNull();
    });

    // Chunk 1 at +200ms (0.2s): 1 chunk / 0.2s = 5.0 tok/s
    await act(async () => {
      nowMs = 1200;
      sseController?.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: "<think>\nReason step\n</think>\n\nHello " } }],
          })}\n\n`
        )
      );
    });

    await waitFor(() => {
      expect(screen.getByTestId("chat-header-net-speed").textContent).toBe("5.0 tok/s");
    });

    // Chunk 2 + usage + [DONE] finishing at +500ms (0.5s): 20 completion tokens / 0.5s = 40.0 tok/s
    await act(async () => {
      nowMs = 1400;
      sseController?.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: "StrictMode!" } }],
          })}\n\n`
        )
      );
      sseController?.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            choices: [{ delta: {}, finish_reason: "stop" }],
            usage: { prompt_tokens: 7, completion_tokens: 20, total_tokens: 27 },
          })}\n\n`
        )
      );
      nowMs = 1500;
      sseController?.enqueue(encoder.encode("data: [DONE]\n\n"));
      sseController?.close();
    });

    await waitFor(() => {
      expect(screen.queryByTestId("chat-stop-btn")).toBeNull();
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "Hello StrictMode!"
      );
    });

    // Verify final tokenCount, tokensPerSec on message badge, and latestTokensPerSec in header
    const tokenBadges = screen.getAllByTestId("chat-msg-tokens");
    expect(tokenBadges).toHaveLength(2);
    expect(tokenBadges[1].textContent).toBe("20 tok • 40.0 tok/s");
    expect(screen.getByTestId("chat-header-net-speed").textContent).toBe("40.0 tok/s");
    expect(screen.getByTestId("chat-thinking-toggle")).toBeTruthy();

    // Verify StrictMode exercised functional updaters and zero nested setters or React warnings fired
    expect(functionalUpdaterCallCount).toBeGreaterThanOrEqual(6);
    expect(nestedSetterCallsDuringUpdater).toEqual([]);
    expect(consoleErrorSpy).not.toHaveBeenCalled();
    expect(consoleWarnSpy).not.toHaveBeenCalled();
  });

  it("preserves prior latestTokensPerSec without calling setter when stream completes with zero token chunks", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    let nowMs = 2000;
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);

    const encoder = new TextEncoder();
    let callIndex = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      callIndex += 1;
      if (callIndex === 1) {
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            nowMs = 2500;
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  choices: [{ delta: { content: "First reply" } }],
                  usage: { prompt_tokens: 4, completion_tokens: 15, total_tokens: 19 },
                })}\n\n`
              )
            );
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });
        return new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }

      // Second response completes immediately with 0 content chunks
      const emptyStream = new ReadableStream<Uint8Array>({
        start(controller) {
          nowMs = 3000;
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        },
      });
      return new Response(emptyStream, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    });

    render(
      <StrictMode>
        <ChatWindow
          isOpen={true}
          isModelReady={true}
          coveragePercent={100}
          apiBaseUrl="http://127.0.0.1:8000/v1"
          sessionToken="sk-strict-zero-chunk"
          modelId="Qwen/Qwen2.5-7B-Instruct"
          isSoloMode={true}
          onClose={vi.fn()}
        />
      </StrictMode>
    );

    // First prompt establishes 15 tok / 0.5s = 30.0 tok/s
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "First turn" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-header-net-speed").textContent).toBe("30.0 tok/s");
    });

    // Second prompt streams 0 chunks; latestTokensPerSec stays 30.0 tok/s and empty placeholder is removed
    nowMs = 2800;
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Zero chunk turn" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getAllByTestId("chat-message-user")).toHaveLength(2);
      expect(screen.queryByTestId("chat-stop-btn")).toBeNull();
    });

    expect(screen.getAllByTestId("chat-message-assistant")).toHaveLength(1);
    expect(screen.getByTestId("chat-header-net-speed").textContent).toBe("30.0 tok/s");
    expect(nestedSetterCallsDuringUpdater).toEqual([]);
    expect(consoleErrorSpy).not.toHaveBeenCalled();
  });
});
