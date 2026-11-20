import React, { useEffect, useRef, useState } from "react";
import { streamChatCompletion } from "../services/engineApi";
import type { ChatMessageItem, OpenAIChatMessagePayload } from "../types/node";

export interface ChatWindowProps {
  isOpen: boolean;
  isModelReady: boolean;
  coveragePercent: number;
  apiBaseUrl: string;
  sessionToken: string;
  modelId: string;
  onClose: () => void;
}

function createMessageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  isOpen,
  isModelReady,
  coveragePercent,
  apiBaseUrl,
  sessionToken,
  modelId,
  onClose,
}) => {
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [inputValue, setInputValue] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Abort any in-flight stream when closing or unmounting
  useEffect(() => {
    if (!isOpen && abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [isOpen]);

  // Focus input when opened and listen for Escape to close
  useEffect(() => {
    if (!isOpen) return;
    textareaRef.current?.focus();

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Auto-scroll message list on updates
  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isStreaming]);

  if (!isOpen) {
    return null;
  }

  const handleStopGeneration = (): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  };

  const handleClearChat = (): void => {
    handleStopGeneration();
    setMessages([]);
    setChatError(null);
  };

  const handleSendMessage = async (): Promise<void> => {
    const trimmed = inputValue.trim();
    if (!trimmed || isStreaming || !isModelReady) {
      return;
    }

    setChatError(null);
    setInputValue("");

    const userMsg: ChatMessageItem = {
      id: createMessageId("user"),
      role: "user",
      content: trimmed,
      createdAt: Date.now(),
    };
    const assistantMsgId = createMessageId("assistant");
    const assistantPlaceholder: ChatMessageItem = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
    };

    const requestHistory: OpenAIChatMessagePayload[] = [
      ...messages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: trimmed },
    ];

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChatCompletion({
        apiBaseUrl,
        apiKey: sessionToken,
        model: modelId,
        messages: requestHistory,
        signal: controller.signal,
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((item) =>
              item.id === assistantMsgId
                ? { ...item, content: `${item.content}${token}` }
                : item
            )
          );
        },
      });
    } catch (err) {
      const isAbortError =
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError");
      if (!isAbortError) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to complete chat request against local gateway.";
        setChatError(message);
        // Remove empty assistant placeholder if no tokens arrived before error
        setMessages((prev) =>
          prev.filter((item) => !(item.id === assistantMsgId && item.content.trim() === ""))
        );
      }
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
    }
  };

  const handleSubmitForm = (event: React.FormEvent): void => {
    event.preventDefault();
    void handleSendMessage();
  };

  const handleTextareaKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSendMessage();
    }
  };

  return (
    <div
      className="chat-modal-backdrop"
      data-testid="ai-chat-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="AI Chat Window"
        data-testid="ai-chat-window"
        className="chat-modal-dialog"
      >
        <div className="chat-modal-header">
          <div className="chat-modal-title-group">
            <div className="chat-modal-title-row">
              <h2 className="chat-modal-title">AI Chat</h2>
              <span
                data-testid="chat-window-readiness"
                className={`chat-readiness-pill${isModelReady ? " ready" : " standby"}`}
              >
                {isModelReady
                  ? "100% RAM Ready"
                  : `Model not ready (${Math.round(coveragePercent)}%)`}
              </span>
            </div>
            <div className="chat-modal-meta">
              <span data-testid="chat-window-model" title={modelId}>
                Model: <strong>{modelId}</strong>
              </span>
              <span className="chat-meta-sep" aria-hidden="true">
                •
              </span>
              <span data-testid="chat-window-endpoint" title={apiBaseUrl}>
                API: <code>{apiBaseUrl}</code>
              </span>
              <span className="chat-meta-sep" aria-hidden="true">
                •
              </span>
              <span data-testid="chat-window-key-status">API Key: Active</span>
            </div>
          </div>

          <div className="chat-modal-actions">
            <button
              type="button"
              data-testid="chat-clear-btn"
              className="native-btn"
              onClick={handleClearChat}
              disabled={messages.length === 0 && !isStreaming}
            >
              Clear
            </button>
            <button
              type="button"
              data-testid="chat-close-btn"
              aria-label="Close chat window"
              className="native-btn"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>

        {!isModelReady && (
          <div
            role="status"
            data-testid="chat-window-not-ready-banner"
            className="chat-standby-banner"
          >
            Model not ready — Waiting for 100% RAM coverage ({Math.round(coveragePercent)}%)
            before sending messages.
          </div>
        )}

        {chatError !== null && (
          <div role="alert" data-testid="chat-error-banner" className="chat-error-banner">
            {chatError}
          </div>
        )}

        <div className="chat-messages-viewport" data-testid="chat-messages-list">
          {messages.length === 0 ? (
            <div className="chat-empty-state" data-testid="chat-empty-state">
              <div className="chat-empty-title">Start a conversation with {modelId}</div>
              <div className="chat-empty-subtitle">
                Connected to <code>{apiBaseUrl}/chat/completions</code> with your active
                session API key.
              </div>
            </div>
          ) : (
            messages.map((msg) => {
              const isLastStreamingAssistant =
                isStreaming &&
                msg.role === "assistant" &&
                msg.id === messages[messages.length - 1]?.id;
              return (
                <div
                  key={msg.id}
                  data-testid={`chat-message-${msg.role}`}
                  data-role={msg.role}
                  className={`chat-bubble-row ${msg.role}`}
                >
                  <div className={`chat-bubble ${msg.role}`}>
                    <div className="chat-bubble-content">
                      {msg.content || (isLastStreamingAssistant ? "Thinking..." : "")}
                    </div>
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="chat-composer" onSubmit={handleSubmitForm}>
          <textarea
            ref={textareaRef}
            data-testid="chat-input"
            className="native-input chat-textarea"
            rows={2}
            value={inputValue}
            disabled={!isModelReady || isStreaming}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleTextareaKeyDown}
            placeholder={
              isModelReady
                ? `Message ${modelId}... (Enter to send, Shift+Enter for new line)`
                : "Model not ready — waiting for 100% RAM coverage..."
            }
          />
          <div className="chat-composer-actions">
            {isStreaming ? (
              <button
                type="button"
                data-testid="chat-stop-btn"
                className="native-btn native-btn-danger"
                onClick={handleStopGeneration}
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                data-testid="chat-send-btn"
                className="native-btn native-btn-primary"
                disabled={!isModelReady || !inputValue.trim()}
              >
                Send
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
