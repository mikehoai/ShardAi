import React, { useRef } from "react";
import type { CatalogModelItem, ModelSourceType } from "../types/node";

export type ModelTab = "catalog" | "custom" | "local";
export type CatalogSource = "cloud" | "cache" | "default";

export interface ResolvedModelSelection {
  model: string;
  sourceType: ModelSourceType;
  filename: string | null;
}

export function inferCustomSourceType(rawInput: string): ModelSourceType {
  const lowered = rawInput.trim().toLowerCase();
  if (
    (lowered.startsWith("http://") || lowered.startsWith("https://")) &&
    (!lowered.includes("huggingface.co/") ||
      lowered.includes("/resolve/") ||
      lowered.endsWith(".gguf"))
  ) {
    return "direct_url";
  }
  return "huggingface";
}

export function resolveModelSelection(params: {
  modelTab: ModelTab;
  catalogModels: CatalogModelItem[];
  selectedCatalogId: string;
  modelInput: string;
  customFilename: string;
  localFilePath: string;
}): ResolvedModelSelection {
  const {
    modelTab,
    catalogModels,
    selectedCatalogId,
    modelInput,
    customFilename,
    localFilePath,
  } = params;

  if (modelTab === "catalog") {
    const selected =
      catalogModels.find((m) => m.id === selectedCatalogId) ?? catalogModels[0];
    if (selected) {
      return {
        model: selected.repo_id,
        sourceType: "catalog",
        filename: selected.filename ?? null,
      };
    }
    return {
      model: modelInput.trim(),
      sourceType: "catalog",
      filename: null,
    };
  }

  if (modelTab === "local") {
    return {
      model: localFilePath.trim(),
      sourceType: "local_file",
      filename: null,
    };
  }

  const customModel = modelInput.trim();
  const detectedType = inferCustomSourceType(customModel);
  return {
    model: customModel,
    sourceType: detectedType,
    filename: detectedType === "huggingface" ? customFilename.trim() || null : null,
  };
}

export interface ModelSelectorProps {
  label?: string;
  testIdPrefix?: string;
  modelTab: ModelTab;
  catalogModels: CatalogModelItem[];
  catalogSource: CatalogSource | null;
  isLoadingCatalog: boolean;
  selectedCatalogId: string;
  modelInput: string;
  customFilename: string;
  localFilePath: string;
  disabled: boolean;
  isDesktopShell?: boolean;
  marginBottom?: string;
  onTabChange: (tab: ModelTab) => void;
  onSelectCatalogModel: (catalogId: string, item: CatalogModelItem | undefined) => void;
  onRefreshCatalog: () => void;
  onModelInputChange: (value: string) => void;
  onCustomFilenameChange: (value: string) => void;
  onLocalFilePathChange: (value: string) => void;
  onBrowseLocalFile: (triggerFallbackInput: () => void) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  label = "Model Source",
  testIdPrefix = "",
  modelTab,
  catalogModels,
  catalogSource,
  isLoadingCatalog,
  selectedCatalogId,
  modelInput,
  customFilename,
  localFilePath,
  disabled,
  isDesktopShell,
  marginBottom = "16px",
  onTabChange,
  onSelectCatalogModel,
  onRefreshCatalog,
  onModelInputChange,
  onCustomFilenameChange,
  onLocalFilePathChange,
  onBrowseLocalFile,
}) => {
  const localFileInputRef = useRef<HTMLInputElement | null>(null);
  const tid = (name: string): string => `${testIdPrefix}${name}`;
  const resolvedIsDesktopShell =
    isDesktopShell ??
    (typeof window !== "undefined" &&
      ("__TAURI_INTERNALS__" in window || "__TAURI__" in window));

  return (
    <div style={{ marginBottom }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "6px",
          marginBottom: "6px",
        }}
      >
        <span className="native-label" style={{ marginBottom: 0 }}>
          {label}
        </span>
        {catalogSource !== null && modelTab === "catalog" && (() => {
          const catalogTooltip =
            catalogSource === "cloud"
              ? "Synced from Cloud"
              : catalogSource === "cache"
                ? "Loaded from ~/.shardai/models_cache.json"
                : "Default Offline Catalog";
          return (
            <span
              role="status"
              tabIndex={0}
              data-testid={tid("catalog-source-badge")}
              data-source={catalogSource}
              data-tooltip={catalogTooltip}
              aria-label={catalogTooltip}
              title={catalogTooltip}
              className={`catalog-cloud-indicator catalog-cloud-indicator--${catalogSource} native-tooltip-trigger`}
            >
              <svg
                data-testid={tid("catalog-cloud-icon")}
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M17.5 19H7a4.5 4.5 0 0 1-.42-8.98A6 6 0 0 1 18.2 8.2 4 4 0 0 1 17.5 19Z" />
                {catalogSource === "default" && <line x1="3" y1="3" x2="21" y2="21" />}
              </svg>
            </span>
          );
        })()}
      </div>

      <div
        role="tablist"
        aria-label={`${label} Tabs`}
        className="native-segmented"
      >
        {(
          [
            { id: "catalog", label: "Online Catalog" },
            { id: "custom", label: "Custom HF / URL" },
            { id: "local", label: "Local File" },
          ] as const
        ).map((tab) => {
          const active = modelTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active}
              data-testid={tid(`tab-${tab.id}`)}
              disabled={disabled}
              className={`native-seg-btn${active ? " active" : ""}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab 1: Online Catalog */}
      {modelTab === "catalog" && (
        <div
          data-testid={tid("panel-catalog")}
          style={{ display: "flex", gap: "6px", alignItems: "center" }}
        >
          <select
            aria-label={testIdPrefix ? "Room Online Model Catalog" : "Online Model Catalog"}
            data-testid={tid("catalog-model-select")}
            className="native-select"
            value={selectedCatalogId}
            disabled={disabled || catalogModels.length === 0}
            onChange={(e) => {
              const chosenId = e.target.value;
              const found = catalogModels.find((m) => m.id === chosenId);
              onSelectCatalogModel(chosenId, found);
            }}
            style={{ flex: 1 }}
          >
            {catalogModels.length === 0 ? (
              <option value="">
                {isLoadingCatalog
                  ? "Loading model catalog..."
                  : "No catalog models loaded yet — click Refresh"}
              </option>
            ) : (
              catalogModels.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} (~{item.estimated_ram_gb} GB)
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            data-testid={tid("refresh-catalog-btn")}
            className="native-btn"
            onClick={onRefreshCatalog}
            disabled={isLoadingCatalog || disabled}
          >
            {isLoadingCatalog ? "Syncing..." : "Refresh"}
          </button>
        </div>
      )}

      {/* Tab 2: Custom HF Repo / External Direct URL */}
      {modelTab === "custom" && (
        <div
          data-testid={tid("panel-custom")}
          style={{ display: "flex", flexDirection: "column", gap: "6px" }}
        >
          <input
            id={tid("model-input")}
            data-testid={tid("model-input")}
            type="text"
            className="native-input"
            value={modelInput}
            disabled={disabled}
            onChange={(e) => onModelInputChange(e.target.value)}
            placeholder="HuggingFace Repo (e.g. bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) or Direct .gguf URL"
          />
          {inferCustomSourceType(modelInput) === "huggingface" && (
            <input
              type="text"
              data-testid={tid("custom-filename-input")}
              className="native-input"
              value={customFilename}
              disabled={disabled}
              onChange={(e) => onCustomFilenameChange(e.target.value)}
              placeholder="Optional .gguf filename (leave blank to auto-pick Q4_K_M)"
              style={{ fontSize: "12px" }}
            />
          )}
        </div>
      )}

      {/* Tab 3: Local File (.gguf via Tauri File Dialog) */}
      {modelTab === "local" && (
        <div
          data-testid={tid("panel-local")}
          style={{ display: "flex", flexDirection: "column", gap: "6px" }}
        >
          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            <input
              type="text"
              data-testid={tid("local-file-input")}
              className="native-input"
              value={localFilePath}
              disabled={disabled}
              onChange={(e) => onLocalFilePathChange(e.target.value)}
              placeholder="Select or paste absolute path to local .gguf file..."
              style={{ flex: 1 }}
            />
            <input
              ref={localFileInputRef}
              type="file"
              accept=".gguf"
              aria-hidden="true"
              tabIndex={-1}
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  const fileWithPath = file as File & { path?: string };
                  onLocalFilePathChange(fileWithPath.path ?? file.name);
                }
              }}
            />
            <button
              type="button"
              data-testid={tid("browse-gguf-btn")}
              className="native-btn"
              disabled={disabled}
              onClick={() => onBrowseLocalFile(() => localFileInputRef.current?.click())}
            >
              Browse (.gguf)
            </button>
          </div>
          {!resolvedIsDesktopShell && (
            <div
              role="note"
              data-testid={tid("browser-preview-local-path-notice")}
              style={{
                padding: "7px 10px",
                borderRadius: "6px",
                backgroundColor: "var(--bg-surface-subtle)",
                border: "1px solid var(--border-hairline)",
                fontSize: "11.5px",
                color: "var(--text-secondary)",
                lineHeight: 1.4,
              }}
            >
              Browser preview cannot access full local filesystem paths. Paste
              the absolute path to your <code>.gguf</code> file above, or use
              the native desktop app to browse files.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
