import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatWindow } from "../components/ChatWindow";
import { MarkdownMessage } from "../components/MarkdownMessage";

interface ChatRequestBody {
  model: string;
  messages: Array<{
    role: string;
    content: string;
  }>;
}

function createSseResponse(
  chunks: string[],
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              choices: [{ delta: { content: chunk } }],
            })}\n\n`
          )
        );
      }
      if (usage) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              choices: [{ delta: {}, finish_reason: "stop" }],
              usage,
            })}\n\n`
          )
        );
      }
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("ChatWindow Markdown Formatting, Copy & Inline Edit", () => {
  let writeTextMock: ReturnType<typeof vi.fn<(text: string) => Promise<void>>>;

  beforeEach(() => {
    writeTextMock = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  it("renders rich Markdown in assistant replies and rejects unsafe XSS links/tags", async () => {
    const markdownReply = [
      "# Main Heading",
      "## Sub Heading",
      "",
      "Paragraph with **bold**, *italic*, ~~struck~~, `inline_code`, and snake_case_words.",
      "",
      "> Quoted insight here",
      "",
      "- Bullet one",
      "- Bullet two",
      "",
      "1. First step",
      "2. Second step",
      "",
      "---",
      "",
      "| Name | Role |",
      "| :--- | ---: |",
      "| Alice | Admin |",
      "",
      "```python",
      "def greet(name):",
      "    return f'hi {name}'",
      "```",
      "",
      "Safe: [Docs](https://shardai.online/docs) and [Email](mailto:team@shardai.online)",
      "Unsafe: [Click](javascript:alert(1)) and [Data](data:text/html,hi)",
      "Raw script: <script>alert(1)</script>",
    ].join("\n");

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(createSseResponse([markdownReply]));

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-md-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={vi.fn()}
      />
    );

    const input = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "Format demo" } });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant")).toBeTruthy();
      expect(screen.queryByTestId("chat-streaming-cursor")).toBeNull();
    });

    const assistantMsg = screen.getByTestId("chat-message-assistant");

    // Headings
    const h1 = assistantMsg.querySelector("h1.chat-md-heading");
    const h2 = assistantMsg.querySelector("h2.chat-md-heading");
    expect(h1?.textContent).toBe("Main Heading");
    expect(h2?.textContent).toBe("Sub Heading");

    // Inline formatting & snake_case preservation
    expect(assistantMsg.querySelector("strong")?.textContent).toBe("bold");
    const emTags = Array.from(assistantMsg.querySelectorAll("em")).map((el) => el.textContent);
    expect(emTags).toEqual(["italic"]);
    expect(assistantMsg.textContent).toContain("snake_case_words");
    expect(assistantMsg.querySelector("del")?.textContent).toBe("struck");
    expect(assistantMsg.querySelector("code.chat-md-inline-code")?.textContent).toBe("inline_code");

    // Blockquote, lists, HR, and GFM table
    expect(assistantMsg.querySelector("blockquote.chat-md-blockquote")?.textContent).toContain(
      "Quoted insight here"
    );
    expect(assistantMsg.querySelectorAll("ul li")).toHaveLength(2);
    expect(assistantMsg.querySelectorAll("ol li")).toHaveLength(2);
    expect(assistantMsg.querySelector("hr.chat-md-hr")).toBeTruthy();
    const tableHeaders = Array.from(assistantMsg.querySelectorAll("table.chat-md-table th")).map(
      (th) => th.textContent
    );
    expect(tableHeaders).toEqual(["Name", "Role"]);
    const tableCells = Array.from(assistantMsg.querySelectorAll("table.chat-md-table td")).map(
      (td) => td.textContent
    );
    expect(tableCells).toEqual(["Alice", "Admin"]);

    // Fenced code block header and Copy code feedback
    expect(assistantMsg.querySelector(".chat-md-code-lang")?.textContent).toBe("python");
    expect(assistantMsg.querySelector(".chat-md-pre code")?.textContent).toBe(
      "def greet(name):\n    return f'hi {name}'"
    );

    const copyCodeBtn = screen.getByTestId("chat-code-copy-btn");
    expect(copyCodeBtn.textContent).toBe("Copy code");
    await act(async () => {
      fireEvent.click(copyCodeBtn);
    });
    expect(writeTextMock).toHaveBeenCalledWith("def greet(name):\n    return f'hi {name}'");
    expect(copyCodeBtn.textContent).toBe("Copied");

    // Safe links rendered as anchors; unsafe links and <script> rendered harmlessly
    const links = Array.from(assistantMsg.querySelectorAll("a.chat-md-link"));
    expect(links).toHaveLength(2);
    expect(links[0].getAttribute("href")).toBe("https://shardai.online/docs");
    expect(links[0].getAttribute("target")).toBe("_blank");
    expect(links[0].getAttribute("rel")).toBe("noopener noreferrer");
    expect(links[1].getAttribute("href")).toBe("mailto:team@shardai.online");
    expect(assistantMsg.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(assistantMsg.querySelector('a[href^="data:"]')).toBeNull();
    expect(assistantMsg.querySelector("script")).toBeNull();
    expect(assistantMsg.textContent).toContain("<script>alert(1)</script>");
  });

  it("renders unclosed fenced code blocks gracefully during streaming with cursor", async () => {
    const partialMarkdown = "Streaming example:\n\n```typescript\nconst answer = 42;";
    const onCopyCode = vi.fn();
    render(
      <MarkdownMessage
        content={partialMarkdown}
        isStreaming={true}
        onCopyCode={onCopyCode}
      />
    );

    expect(screen.getByText("typescript")).toBeTruthy();
    expect(screen.getByText("const answer = 42;")).toBeTruthy();
    expect(screen.getByTestId("chat-streaming-cursor").textContent).toBe("▍");

    const copyBtn = screen.getByTestId("chat-code-copy-btn");
    await act(async () => {
      fireEvent.click(copyBtn);
      await Promise.resolve();
    });
    expect(onCopyCode).toHaveBeenCalledWith("const answer = 42;");
    expect(copyBtn.textContent).toBe("Copied");
  });

  it("renders Copy action below both user and assistant messages and resets feedback after 1800ms", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      createSseResponse(["Assistant **markdown** reply"], {
        prompt_tokens: 6,
        completion_tokens: 14,
        total_tokens: 20,
      })
    );

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-copy-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={vi.fn()}
      />
    );

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Hello user prompt" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getAllByTestId("chat-copy-msg-btn")).toHaveLength(2);
    });

    vi.useFakeTimers();

    const copyButtons = screen.getAllByTestId("chat-copy-msg-btn");
    // Copy user message
    await act(async () => {
      fireEvent.click(copyButtons[0]);
      await Promise.resolve();
    });
    expect(writeTextMock).toHaveBeenCalledWith("Hello user prompt");
    expect(copyButtons[0].getAttribute("data-copied")).toBe("true");
    expect(copyButtons[0].getAttribute("aria-label")).toBe("Copied message");
    expect(copyButtons[1].getAttribute("data-copied")).toBe("false");

    // Copy assistant message (raw markdown content)
    await act(async () => {
      fireEvent.click(copyButtons[1]);
      await Promise.resolve();
    });
    expect(writeTextMock).toHaveBeenCalledWith("Assistant **markdown** reply");
    expect(copyButtons[1].getAttribute("data-copied")).toBe("true");
    expect(copyButtons[0].getAttribute("data-copied")).toBe("false");

    act(() => {
      vi.advanceTimersByTime(1800);
    });
    expect(copyButtons[1].getAttribute("data-copied")).toBe("false");
    expect(copyButtons[1].getAttribute("aria-label")).toBe("Copy message");
  });

  it("opens inline edit textarea and cancels edit on Escape without closing ChatWindow", async () => {
    const onClose = vi.fn();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(createSseResponse(["First reply"]));

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-esc-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={onClose}
      />
    );

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Initial prompt" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-edit-msg-btn")).toBeTruthy();
    });

    // Enter inline edit mode
    fireEvent.click(screen.getByTestId("chat-edit-msg-btn"));
    const editTextarea = screen.getByTestId("chat-edit-textarea") as HTMLTextAreaElement;
    expect(editTextarea.value).toBe("Initial prompt");

    fireEvent.change(editTextarea, { target: { value: "Modified draft" } });

    // Escape on textarea cancels edit WITHOUT calling onClose
    fireEvent.keyDown(editTextarea, { key: "Escape" });
    expect(screen.queryByTestId("chat-edit-textarea")).toBeNull();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByTestId("chat-message-user").textContent).toContain("Initial prompt");

    // Enter edit mode again and press Escape on window (e.g. while focus is on Cancel button)
    fireEvent.click(screen.getByTestId("chat-edit-msg-btn"));
    expect(screen.getByTestId("chat-edit-textarea")).toBeTruthy();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("chat-edit-textarea")).toBeNull();
    expect(onClose).not.toHaveBeenCalled();

    // When NOT editing, Escape closes ChatWindow
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("truncates subsequent turns on Save & Send and preserves edited prompt if resend fails pre-token", async () => {
    const sentBodies: ChatRequestBody[] = [];
    let callCount = 0;

    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      callCount += 1;
      if (typeof init?.body === "string") {
        sentBodies.push(JSON.parse(init.body) as ChatRequestBody);
      }
      if (callCount === 1) {
        return createSseResponse(["Reply 1"]);
      }
      if (callCount === 2) {
        return createSseResponse(["Reply 2"]);
      }
      if (callCount === 3) {
        return createSseResponse(["Edited Reply 1"]);
      }
      return new Response(JSON.stringify({ error: { message: "Gateway busy" } }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    });

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-edit-save-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={vi.fn()}
      />
    );

    const input = screen.getByTestId("chat-input") as HTMLTextAreaElement;

    // Turn 1
    fireEvent.change(input, { target: { value: "Prompt 1" } });
    fireEvent.click(screen.getByTestId("chat-send-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain("Reply 1");
    });

    // Turn 2
    fireEvent.change(input, { target: { value: "Prompt 2" } });
    fireEvent.click(screen.getByTestId("chat-send-btn"));
    await waitFor(() => {
      expect(screen.getAllByTestId("chat-message-assistant")).toHaveLength(2);
    });

    // Edit Turn 1 and Save & Send
    const editButtons = screen.getAllByTestId("chat-edit-msg-btn");
    fireEvent.click(editButtons[0]);
    const editTextarea = screen.getByTestId("chat-edit-textarea") as HTMLTextAreaElement;
    fireEvent.change(editTextarea, { target: { value: "Edited Prompt 1" } });
    fireEvent.click(screen.getByTestId("chat-edit-save-btn"));

    await waitFor(() => {
      const userMessages = screen.getAllByTestId("chat-message-user");
      const assistantMessages = screen.getAllByTestId("chat-message-assistant");
      expect(userMessages).toHaveLength(1);
      expect(assistantMessages).toHaveLength(1);
      expect(userMessages[0].textContent).toContain("Edited Prompt 1");
      expect(assistantMessages[0].textContent).toContain("Edited Reply 1");
    });

    // Verify request sent only the edited prompt (prior history was empty)
    expect(sentBodies[2].messages).toEqual([{ role: "user", content: "Edited Prompt 1" }]);

    // Now edit again when gateway returns 503 before any token: keep edited user prompt visible
    fireEvent.click(screen.getByTestId("chat-edit-msg-btn"));
    fireEvent.change(screen.getByTestId("chat-edit-textarea"), {
      target: { value: "Retried Prompt 1" },
    });
    fireEvent.click(screen.getByTestId("chat-edit-save-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-error-banner").textContent).toContain("Gateway busy");
    });
    expect(screen.getAllByTestId("chat-message-user")).toHaveLength(1);
    expect(screen.getByTestId("chat-message-user").textContent).toContain("Retried Prompt 1");
    expect(screen.queryByTestId("chat-message-assistant")).toBeNull();
  });

  it("disables Edit button while streaming or when model is not ready", async () => {
    let sseController: ReadableStreamDefaultController<Uint8Array> | null = null;
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
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

    const { rerender } = render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-disabled-edit-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={vi.fn()}
      />
    );

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Check disabled edit" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    // While streaming, Edit button is disabled
    await waitFor(() => {
      const editBtn = screen.getByTestId("chat-edit-msg-btn") as HTMLButtonElement;
      expect(editBtn.disabled).toBe(true);
    });

    // Finish streaming
    act(() => {
      if (sseController) {
        sseController.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ choices: [{ delta: { content: "Done" } }] })}\n\n`
          )
        );
        sseController.enqueue(encoder.encode("data: [DONE]\n\n"));
        sseController.close();
      }
    });

    await waitFor(() => {
      const editBtn = screen.getByTestId("chat-edit-msg-btn") as HTMLButtonElement;
      expect(editBtn.disabled).toBe(false);
    });

    // When model becomes not ready, Edit button is disabled again
    rerender(
      <ChatWindow
        isOpen={true}
        isModelReady={false}
        coveragePercent={60}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-disabled-edit-test"
        modelId="Qwen/Qwen2.5-7B-Instruct"
        onClose={vi.fn()}
      />
    );

    expect((screen.getByTestId("chat-edit-msg-btn") as HTMLButtonElement).disabled).toBe(true);
  });

  it("renders user Markdown, collapsible DeepSeek thinking header, Retry icon, token counts, and header telemetry", async () => {
    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      callCount += 1;
      if (callCount === 1) {
        return createSseResponse(
          [
            "<think>\nCheck **constraints** carefully.\n</think>\n\n",
            "Final **answer** is ready.",
          ],
          { prompt_tokens: 10, completion_tokens: 18, total_tokens: 28 }
        );
      }
      return createSseResponse(["Regenerated **clean** response."], {
        prompt_tokens: 10,
        completion_tokens: 9,
        total_tokens: 19,
      });
    });

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-think-test"
        modelId="bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF"
        roomId="team-deepseek"
        sharedRamGb={4}
        roomTotalRamGb={8}
        connectedPeersCount={1}
        activeP2pTransport="quic_direct"
        p2pRttMs={22}
        p2pThroughputMbps={45.2}
        onClose={vi.fn()}
      />
    );

    // Single-row header telemetry icons (aligned with Clear and Close; no Maximize button)
    expect(screen.queryByTestId("chat-maximize-btn")).toBeNull();
    const headerRight = screen.getByTestId("chat-header-telemetry").parentElement;
    expect(headerRight?.querySelector("[data-testid='chat-clear-btn']")).toBeTruthy();
    expect(headerRight?.querySelector("[data-testid='chat-close-btn']")).toBeTruthy();

    expect(screen.getByTestId("chat-header-ram").querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("chat-header-ram").textContent).toBe("4 / 8 GB");
    expect(screen.getByTestId("chat-header-room-status").querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("chat-header-room-status").textContent).toContain(
      "team-deepseek"
    );
    expect(screen.getByTestId("chat-header-transport").querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("chat-header-transport").textContent).toBe(
      "QUIC P2P (Direct)"
    );
    expect(screen.getByTestId("chat-header-net-speed").querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("chat-header-net-speed").textContent).toBe(
      "45.2 Mbps • 22 ms"
    );
    expect(screen.getByTestId("chat-header-session-tokens").querySelector("svg")).toBeTruthy();
    expect(screen.getByTestId("chat-header-session-tokens").textContent).toBe("0 tok");

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Explain **quarks** and `gluons`" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-thinking-toggle")).toBeTruthy();
    });

    // User message rendered as Markdown (<strong> and <code>)
    const userBubble = screen.getByTestId("chat-message-user");
    expect(userBubble.querySelector("strong")?.textContent).toBe("quarks");
    expect(userBubble.querySelector("code.chat-md-inline-code")?.textContent).toBe("gluons");

    // Thinking block is collapsed by default; answer renders directly
    const assistantBubble = screen.getByTestId("chat-message-assistant");
    expect(assistantBubble.textContent).toContain("Final answer is ready.");
    expect(screen.queryByTestId("chat-thinking-content")).toBeNull();
    expect(screen.getByTestId("chat-thinking-toggle").getAttribute("aria-expanded")).toBe(
      "false"
    );

    // Clicking the thinking toggle expands the reasoning block
    fireEvent.click(screen.getByTestId("chat-thinking-toggle"));
    expect(screen.getByTestId("chat-thinking-toggle").getAttribute("aria-expanded")).toBe(
      "true"
    );
    expect(screen.getByTestId("chat-thinking-content").textContent).toContain(
      "Check constraints carefully."
    );

    // Per-message token badges & session token counter updated from SSE usage (8 + 18 = 26)
    const tokenBadges = screen.getAllByTestId("chat-msg-tokens");
    expect(tokenBadges).toHaveLength(2);
    expect(tokenBadges[1].textContent).toContain("18 tok");
    expect(screen.getByTestId("chat-header-session-tokens").textContent).toContain("26 tok");

    // Retry icon button regenerates assistant reply cleanly
    fireEvent.click(screen.getByTestId("chat-retry-msg-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "Regenerated clean response."
      );
    });
    expect(screen.getAllByTestId("chat-msg-tokens")[1].textContent).toContain("9 tok");
  });
});
