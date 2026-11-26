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

function createSseResponse(chunks: string[]): Response {
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
      createSseResponse(["Assistant **markdown** reply"])
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
    expect(copyButtons[0].textContent).toBe("Copied");
    expect(copyButtons[1].textContent).toBe("Copy");

    // Copy assistant message (raw markdown content)
    await act(async () => {
      fireEvent.click(copyButtons[1]);
      await Promise.resolve();
    });
    expect(writeTextMock).toHaveBeenCalledWith("Assistant **markdown** reply");
    expect(copyButtons[1].textContent).toBe("Copied");
    expect(copyButtons[0].textContent).toBe("Copy");

    act(() => {
      vi.advanceTimersByTime(1800);
    });
    expect(copyButtons[1].textContent).toBe("Copy");
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
});
