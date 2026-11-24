import React, { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { streamChatCompletion } from "../services/engineApi";
import { extractDocumentText, MAX_DOC_TEXT_CHARS } from "../utils/documentExtractor";
import type {
  ChatAttachmentItem,
  ChatAttachmentKind,
  ChatMessageItem,
  OpenAIChatContentPart,
  OpenAIChatMessagePayload,
} from "../types/node";

export interface ChatWindowProps {
  isOpen: boolean;
  isModelReady: boolean;
  coveragePercent: number;
  apiBaseUrl: string;
  sessionToken: string;
  modelId: string;
  onClose: () => void;
}

const MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024;

const IMAGE_EXTENSIONS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
  ".bmp",
]);

const VIDEO_EXTENSIONS = new Set([
  ".mp4",
  ".webm",
  ".mov",
  ".mkv",
  ".avi",
]);

const TEXT_DOCUMENT_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".markdown",
  ".csv",
  ".json",
  ".xml",
  ".yaml",
  ".yml",
  ".log",
  ".py",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".html",
  ".css",
  ".sql",
  ".sh",
  ".rs",
  ".go",
  ".java",
  ".c",
  ".cpp",
  ".h",
  ".hpp",
]);

const BINARY_DOCUMENT_EXTENSIONS = new Set([".pdf", ".doc", ".docx"]);

const isTauriEnvironment = (): boolean => {
  return (
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window)
  );
};

function getFileExtension(name: string): string {
  const dotIdx = name.lastIndexOf(".");
  return dotIdx >= 0 ? name.slice(dotIdx).toLowerCase() : "";
}

export function classifyAttachmentKind(file: File): ChatAttachmentKind | null {
  const mime = (file.type || "").toLowerCase();
  const ext = getFileExtension(file.name);

  if (mime.startsWith("image/") || IMAGE_EXTENSIONS.has(ext)) {
    return "image";
  }
  if (mime.startsWith("video/") || VIDEO_EXTENSIONS.has(ext)) {
    return "video";
  }
  if (
    mime.startsWith("text/") ||
    mime === "application/json" ||
    mime === "application/xml" ||
    mime === "application/pdf" ||
    mime === "application/msword" ||
    mime.startsWith("application/vnd.openxmlformats-officedocument.") ||
    TEXT_DOCUMENT_EXTENSIONS.has(ext) ||
    BINARY_DOCUMENT_EXTENSIONS.has(ext)
  ) {
    return "document";
  }
  return null;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error(`Unable to read "${file.name}" as data URL.`));
      }
    };
    reader.onerror = () => {
      reject(new Error(`Failed to read "${file.name}".`));
    };
    reader.readAsDataURL(file);
  });
}

async function extractVideoKeyframe(file: File): Promise<string | undefined> {
  if (
    typeof window === "undefined" ||
    typeof document === "undefined" ||
    typeof URL === "undefined" ||
    typeof URL.createObjectURL !== "function" ||
    (typeof navigator !== "undefined" &&
      navigator.userAgent.toLowerCase().includes("jsdom"))
  ) {
    return undefined;
  }

  const objectUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";

  try {
    return await new Promise<string | undefined>((resolve) => {
      let settled = false;
      const finish = (result: string | undefined): void => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timerId);
        video.removeAttribute("src");
        video.load();
        resolve(result);
      };

      const timerId = window.setTimeout(() => finish(undefined), 2000);

      const captureFrame = (): void => {
        try {
          const srcW = video.videoWidth || 640;
          const srcH = video.videoHeight || 360;
          const scale = Math.min(1, 768 / Math.max(srcW, srcH));
          const canvas = document.createElement("canvas");
          canvas.width = Math.max(1, Math.round(srcW * scale));
          canvas.height = Math.max(1, Math.round(srcH * scale));
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            finish(undefined);
            return;
          }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          finish(canvas.toDataURL("image/jpeg", 0.85));
        } catch {
          finish(undefined);
        }
      };

      video.addEventListener("loadeddata", captureFrame, { once: true });
      video.addEventListener("error", () => finish(undefined), { once: true });
      video.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export function buildOpenAIMessagePayload(
  role: "user" | "assistant",
  text: string,
  attachments?: ChatAttachmentItem[]
): OpenAIChatMessagePayload {
  const sections: string[] = [];
  if (text.trim()) {
    sections.push(text.trim());
  }

  const imageParts: OpenAIChatContentPart[] = [];
  for (const att of attachments ?? []) {
    if (att.kind === "document") {
      if (att.textContent !== undefined) {
        sections.push(`[Attached document: ${att.name}]\n\`\`\`\n${att.textContent}\n\`\`\``);
      } else {
        sections.push(
          `[Attached document: ${att.name} (${formatFileSize(att.sizeBytes)})]`
        );
      }
    } else if (att.kind === "video") {
      sections.push(`[Attached video: ${att.name} (${formatFileSize(att.sizeBytes)})]`);
      if (att.dataUrl) {
        imageParts.push({
          type: "image_url",
          image_url: { url: att.dataUrl, detail: "auto" },
        });
      }
    } else if (att.kind === "image" && att.dataUrl) {
      imageParts.push({
        type: "image_url",
        image_url: { url: att.dataUrl, detail: "auto" },
      });
    }
  }

  const combinedText = sections.join("\n\n").trim();
  if (imageParts.length > 0) {
    const contentParts: OpenAIChatContentPart[] = [];
    if (combinedText) {
      contentParts.push({ type: "text", text: combinedText });
    }
    contentParts.push(...imageParts);
    return { role, content: contentParts };
  }

  return { role, content: combinedText };
}

let messageSeq = 0;

function createMessageId(prefix: string): string {
  messageSeq += 1;
  return `${prefix}-${Date.now()}-${messageSeq}-${Math.random().toString(36).slice(2, 8)}`;
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
  const [attachments, setAttachments] = useState<ChatAttachmentItem[]>([]);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [isMaximized, setIsMaximized] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const isMountedRef = useRef<boolean>(true);
  const abortControllerRef = useRef<AbortController | null>(null);
  const onCloseRef = useRef<() => void>(onClose);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragCounterRef = useRef<number>(0);
  const maximizedByChatRef = useRef<boolean>(false);
  const prevModelSessionRef = useRef<{ modelId: string; sessionToken: string }>({
    modelId,
    sessionToken,
  });

  // Track mount lifecycle (StrictMode-safe) and abort any in-flight stream on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  // Keep latest onClose callback without re-triggering open-focus or keydown effects
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  // Reset chat history when switching active model or gateway session token
  useEffect(() => {
    const prev = prevModelSessionRef.current;
    if (prev.modelId === modelId && prev.sessionToken === sessionToken) {
      return;
    }
    prevModelSessionRef.current = { modelId, sessionToken };
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setMessages([]);
    setAttachments([]);
    setIsDragOver(false);
    dragCounterRef.current = 0;
    setChatError(null);
    setInputValue("");
  }, [modelId, sessionToken]);

  // Abort any in-flight stream and restore window maximize state when closing
  useEffect(() => {
    if (!isOpen) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      setIsStreaming(false);
      setIsDragOver(false);
      dragCounterRef.current = 0;
      setIsMaximized(false);
      if (maximizedByChatRef.current) {
        maximizedByChatRef.current = false;
        if (isTauriEnvironment()) {
          void invoke<boolean>("window_set_maximized", { maximized: false }).catch(
            () => undefined
          );
        }
      }
    }
  }, [isOpen]);

  // Focus composer textarea only when ChatWindow opens
  useEffect(() => {
    if (isOpen) {
      textareaRef.current?.focus();
    }
  }, [isOpen]);

  // Listen for Escape to close without tearing down/re-attaching on parent re-renders
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  // Auto-scroll message list on updates
  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isStreaming]);

  if (!isOpen) {
    return null;
  }

  const canAttachFiles = isModelReady && !isStreaming;

  const handleToggleMaximize = (): void => {
    const nextMax = !isMaximized;
    setIsMaximized(nextMax);
    if (isTauriEnvironment()) {
      maximizedByChatRef.current = nextMax;
      void invoke<boolean>("window_set_maximized", { maximized: nextMax }).catch(
        () => undefined
      );
    }
  };

  const processIncomingFiles = async (fileList: FileList | File[]): Promise<void> => {
    if (!canAttachFiles) {
      return;
    }
    const files = Array.from(fileList);
    if (files.length === 0) {
      return;
    }

    const nextItems: ChatAttachmentItem[] = [];
    for (const file of files) {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        setChatError(`File "${file.name}" exceeds the 50 MB attachment limit.`);
        continue;
      }
      const kind = classifyAttachmentKind(file);
      if (kind === null) {
        setChatError(
          `Unsupported file type for "${file.name}". Supported: documents, images, and videos.`
        );
        continue;
      }

      try {
        if (kind === "image") {
          const dataUrl = await readFileAsDataUrl(file);
          nextItems.push({
            id: createMessageId("att"),
            name: file.name,
            mimeType: file.type || "image/png",
            sizeBytes: file.size,
            kind,
            dataUrl,
          });
        } else if (kind === "video") {
          const keyframeUrl = await extractVideoKeyframe(file);
          nextItems.push({
            id: createMessageId("att"),
            name: file.name,
            mimeType: file.type || "video/mp4",
            sizeBytes: file.size,
            kind,
            ...(keyframeUrl ? { dataUrl: keyframeUrl } : {}),
          });
        } else {
          const textContent = await extractDocumentText(file, MAX_DOC_TEXT_CHARS);
          nextItems.push({
            id: createMessageId("att"),
            name: file.name,
            mimeType: file.type || "application/octet-stream",
            sizeBytes: file.size,
            kind,
            textContent,
          });
        }
      } catch (err: unknown) {
        const detail = err instanceof Error ? err.message : String(err);
        setChatError(detail);
      }
    }

    if (nextItems.length > 0 && isMountedRef.current) {
      setChatError(null);
      setAttachments((prev) => [...prev, ...nextItems]);
    }
  };

  const handleRemoveAttachment = (attachmentId: string): void => {
    setAttachments((prev) => prev.filter((item) => item.id !== attachmentId));
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    if (!canAttachFiles) return;
    dragCounterRef.current += 1;
    setIsDragOver(true);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    if (!canAttachFiles) {
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "none";
      }
      return;
    }
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    if (!canAttachFiles) return;
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
    if (dragCounterRef.current === 0) {
      setIsDragOver(false);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    dragCounterRef.current = 0;
    setIsDragOver(false);
    if (!canAttachFiles) {
      return;
    }
    const droppedFiles = event.dataTransfer?.files;
    if (droppedFiles && droppedFiles.length > 0) {
      void processIncomingFiles(droppedFiles);
    }
  };

  const handleStopGeneration = (): void => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
    setMessages((prev) =>
      prev.filter((item) => !(item.role === "assistant" && item.content.trim() === ""))
    );
  };

  const handleClearChat = (): void => {
    handleStopGeneration();
    setMessages([]);
    setAttachments([]);
    setChatError(null);
  };

  const handleSendMessage = async (): Promise<void> => {
    const trimmed = inputValue.trim();
    const pendingAttachments = [...attachments];
    if ((!trimmed && pendingAttachments.length === 0) || isStreaming || !isModelReady) {
      return;
    }

    setChatError(null);
    setInputValue("");
    setAttachments([]);

    const userMsg: ChatMessageItem = {
      id: createMessageId("user"),
      role: "user",
      content: trimmed,
      ...(pendingAttachments.length > 0 ? { attachments: pendingAttachments } : {}),
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
      ...messages.map((m) => buildOpenAIMessagePayload(m.role, m.content, m.attachments)),
      buildOpenAIMessagePayload("user", trimmed, pendingAttachments),
    ];

    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let receivedAnyToken = false;

    try {
      await streamChatCompletion({
        apiBaseUrl,
        apiKey: sessionToken,
        model: modelId,
        messages: requestHistory,
        signal: controller.signal,
        onToken: (token) => {
          if (!isMountedRef.current || abortControllerRef.current !== controller) {
            return;
          }
          receivedAnyToken = true;
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
      if (!isMountedRef.current || abortControllerRef.current !== controller) {
        return;
      }
      const isAbortError =
        controller.signal.aborted ||
        (err instanceof DOMException && err.name === "AbortError") ||
        (err instanceof Error && err.name === "AbortError");
      if (!isAbortError) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to complete chat request against local gateway.";
        setChatError(message);
        // Remove empty assistant placeholder and, if no tokens arrived, drop the failed user message
        // so rejected attachments are not resent on subsequent prompts.
        setMessages((prev) =>
          prev.filter((item) => {
            if (item.id === assistantMsgId && item.content.trim() === "") {
              return false;
            }
            if (!receivedAnyToken && item.id === userMsg.id) {
              return false;
            }
            return true;
          })
        );
      }
    } finally {
      const isLatestController = abortControllerRef.current === controller;
      if (isLatestController) {
        abortControllerRef.current = null;
        if (isMountedRef.current) {
          setIsStreaming(false);
        }
      }
      if (isMountedRef.current && (isLatestController || controller.signal.aborted)) {
        setMessages((prev) => {
          const hasEmptyPlaceholder = prev.some(
            (item) => item.id === assistantMsgId && item.content.trim() === ""
          );
          return hasEmptyPlaceholder
            ? prev.filter((item) => !(item.id === assistantMsgId && item.content.trim() === ""))
            : prev;
        });
      }
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
      className={`chat-modal-backdrop${isMaximized ? " chat-modal-backdrop--maximized" : ""}`}
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
        data-maximized={isMaximized ? "true" : "false"}
        className={`chat-modal-dialog${isMaximized ? " chat-modal-dialog--maximized" : ""}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragOver && canAttachFiles && (
          <div className="chat-drop-overlay" data-testid="chat-drop-overlay">
            <div className="chat-drop-overlay-card">
              <div className="chat-drop-overlay-title">Drop files to attach</div>
              <div className="chat-drop-overlay-subtitle">
                Documents, images, and videos will be added to your prompt
              </div>
            </div>
          </div>
        )}

        <div
          className="chat-modal-header"
          onDoubleClick={(e) => {
            if (!(e.target as HTMLElement).closest("button")) {
              handleToggleMaximize();
            }
          }}
        >
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
              disabled={messages.length === 0 && attachments.length === 0 && !isStreaming}
            >
              Clear
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
                session API key. Drag and drop documents, images, or videos here when ready.
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
                    {msg.attachments && msg.attachments.length > 0 && (
                      <div
                        className="chat-bubble-attachments"
                        data-testid="chat-message-attachments"
                      >
                        {msg.attachments.map((att) =>
                          att.kind === "image" && att.dataUrl ? (
                            <img
                              key={att.id}
                              src={att.dataUrl}
                              alt={att.name}
                              className="chat-bubble-attachment-img"
                            />
                          ) : (
                            <span
                              key={att.id}
                              className="chat-bubble-attachment-pill"
                              title={att.name}
                            >
                              {att.kind === "video" ? "🎬" : "📄"} {att.name} (
                              {formatFileSize(att.sizeBytes)})
                            </span>
                          )
                        )}
                      </div>
                    )}
                    <div className="chat-bubble-content">
                      {isLastStreamingAssistant && !msg.content ? (
                        <span
                          className="chat-thinking-indicator"
                          data-testid="chat-thinking-indicator"
                          aria-label="AI is thinking"
                        >
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                          <span className="chat-thinking-dot" />
                        </span>
                      ) : (
                        <>
                          {msg.content}
                          {isLastStreamingAssistant && msg.content.length > 0 && (
                            <span
                              className="chat-streaming-cursor"
                              data-testid="chat-streaming-cursor"
                              aria-hidden="true"
                            >
                              ▍
                            </span>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>

        {attachments.length > 0 && (
          <div className="chat-attachments-bar" data-testid="chat-attachments-bar">
            {attachments.map((att, idx) => (
              <div key={att.id} className="chat-attachment-chip">
                {att.dataUrl ? (
                  <img
                    src={att.dataUrl}
                    alt={att.name}
                    className="chat-attachment-thumb"
                  />
                ) : (
                  <span className="chat-attachment-icon" aria-hidden="true">
                    {att.kind === "video" ? "🎬" : "📄"}
                  </span>
                )}
                <div className="chat-attachment-meta">
                  <span className="chat-attachment-name" title={att.name}>
                    {att.name}
                  </span>
                  <span className="chat-attachment-size">{formatFileSize(att.sizeBytes)}</span>
                </div>
                <button
                  type="button"
                  data-testid={`chat-remove-attachment-${idx}`}
                  aria-label={`Remove ${att.name}`}
                  className="chat-attachment-remove"
                  onClick={() => handleRemoveAttachment(att.id)}
                  disabled={isStreaming}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <form className="chat-composer" onSubmit={handleSubmitForm}>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            data-testid="chat-file-input"
            accept="image/*,video/*,.pdf,.doc,.docx,.txt,.md,.csv,.json,.xml,.yaml,.yml,.log,.py,.ts,.tsx,.js,.jsx,.html,.css,.sql,.sh,.rs,.go,.java,.c,.cpp"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                void processIncomingFiles(e.target.files);
              }
              e.target.value = "";
            }}
          />
          <button
            type="button"
            data-testid="chat-attach-btn"
            className="native-btn chat-attach-btn"
            aria-label="Attach document, image, or video"
            title="Attach document, image, or video"
            disabled={!canAttachFiles}
            onClick={() => fileInputRef.current?.click()}
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
          </button>
          <textarea
            ref={textareaRef}
            data-testid="chat-input"
            className="native-input chat-textarea"
            rows={2}
            value={inputValue}
            disabled={!isModelReady || isStreaming}
            style={{ opacity: isStreaming ? 0.6 : 1 }}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleTextareaKeyDown}
            placeholder={
              isStreaming
                ? "AI is generating a response..."
                : isModelReady
                  ? `Message ${modelId} or drop documents, images, videos... (Enter to send)`
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
                style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <rect x="4" y="4" width="16" height="16" rx="2" />
                </svg>
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                data-testid="chat-send-btn"
                className="native-btn native-btn-primary"
                disabled={!isModelReady || (!inputValue.trim() && attachments.length === 0)}
                style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <line x1="12" y1="19" x2="12" y2="5" />
                  <polyline points="5 12 12 5 19 12" />
                </svg>
                <span>Send</span>
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatWindow;
