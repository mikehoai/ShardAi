import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "../components/Dashboard";
import {
  extractApiErrorDetail,
  normalizeNoticeToEnglish,
} from "../services/engineApi";
import type {
  CatalogModelItem,
  NodeStatusResponse,
  RoomLobbyItem,
  StartNodeRequest,
} from "../types/node";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

describe("ShardAi Gateway & Network Rooms Lobby UI", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    vi.restoreAllMocks();
    window.localStorage.clear();
    delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
  });

  it("renders Network Rooms Lobby table, Refresh button, and Create/Unlisted Room modal", async () => {
    const mockRooms: RoomLobbyItem[] = [
      {
        room_id: "public",
        model: "meta-llama/Llama-3.1-8B-Instruct",
        nodes_online: 3,
        total_shared_ram_gb: 18,
        is_private: false,
        network_scope: "internet",
      },
      {
        room_id: "vault-alpha",
        model: "Qwen/Qwen2.5-7B-Instruct",
        nodes_online: 2,
        total_shared_ram_gb: 12,
        is_private: true,
        network_scope: "lan",
      },
    ];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify(mockRooms), { status: 200 });
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);

    expect(screen.getByTestId("network-rooms-lobby")).toBeTruthy();
    expect(screen.getByTestId("refresh-rooms-btn")).toBeTruthy();
    expect(screen.getByTestId("create-room-modal-btn")).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByTestId("join-room-public")).toBeTruthy();
      expect(screen.getByTestId("join-room-vault-alpha")).toBeTruthy();
    });

    // Clicking Join on a locked room opens the Private Room Key prompt
    fireEvent.click(screen.getByTestId("join-room-vault-alpha"));
    expect(screen.getByTestId("private-room-key-modal")).toBeTruthy();

    // Opening Create New Room / Join Unlisted modal displays Invite String preview
    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    expect(screen.getByTestId("create-room-modal")).toBeTruthy();
    fireEvent.change(screen.getByTestId("modal-room-name-input"), {
      target: { value: "team-secret" },
    });
    fireEvent.change(screen.getByTestId("modal-room-key-input"), {
      target: { value: "pass123" },
    });
    expect(screen.getByTestId("invite-string-preview").textContent).toBe(
      "team-secret | pass123"
    );
  });

  it("constructs a valid StartNodeRequest with room_key and is_unlisted", () => {
    const sharedRamGb = 8;
    const request: StartNodeRequest = {
      node_id: "test-ui-node",
      model_id: "Qwen/Qwen2.5-7B-Instruct",
      network_mode: "lan",
      room_id: "private-lab-room",
      room_key: "secret-key",
      is_unlisted: true,
      backend: "llama_cpp",
      ram_gb: sharedRamGb,
      shared_ram_gb: sharedRamGb,
      max_ram_mb: sharedRamGb * 1024,
      max_vram_mb: 2048,
      p2p_listen_port: 49152,
      shard_layer_start: 0,
      shard_layer_end: 15,
    };

    expect(request.room_key).toBe("secret-key");
    expect(request.is_unlisted).toBe(true);
    expect(request.max_ram_mb).toBe(8192);
  });

  it("validates zero memory footprint and no session_token when node is STOPPED", () => {
    const stoppedStatus: NodeStatusResponse = {
      state: "STOPPED",
      node_id: null,
      model_id: null,
      room_id: null,
      is_private: false,
      is_unlisted: false,
      backend: null,
      n_layers: 0,
      allocated_ram_mb: 0,
      allocated_vram_mb: 0,
      p2p_listen_port: null,
      connected_peers_count: 0,
      peers: [],
      session_token: null,
      api_base_url: null,
      message: "Stopped",
    };

    expect(stoppedStatus.state).toBe("STOPPED");
    expect(stoppedStatus.session_token).toBeNull();
    expect(stoppedStatus.allocated_ram_mb).toBe(0);
  });

  it("hides and unhides the navigation sidebar menu via toggle button and persists state", () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));

    const { unmount } = render(<Dashboard />);

    const toggleBtn = screen.getByTestId("sidebar-toggle-btn");
    const sidebar = screen.getByTestId("native-sidebar");

    // Visible by default
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("true");
    expect(sidebar.classList.contains("collapsed")).toBe(false);
    expect(sidebar.hidden).toBe(false);
    expect(sidebar.getAttribute("aria-hidden")).toBe("false");

    // Click to hide
    fireEvent.click(toggleBtn);
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(sidebar.classList.contains("collapsed")).toBe(true);
    expect(sidebar.hidden).toBe(true);
    expect(sidebar.getAttribute("aria-hidden")).toBe("true");
    expect(window.localStorage.getItem("shardai_sidebar_hidden")).toBe("true");

    // Persists hidden state across remount
    unmount();
    render(<Dashboard />);
    const remountedToggle = screen.getByTestId("sidebar-toggle-btn");
    const remountedSidebar = screen.getByTestId("native-sidebar");
    expect(remountedToggle.getAttribute("aria-expanded")).toBe("false");
    expect(remountedSidebar.hidden).toBe(true);

    // Click to unhide
    fireEvent.click(remountedToggle);
    expect(remountedToggle.getAttribute("aria-expanded")).toBe("true");
    expect(remountedSidebar.classList.contains("collapsed")).toBe(false);
    expect(remountedSidebar.hidden).toBe(false);
    expect(window.localStorage.getItem("shardai_sidebar_hidden")).toBe("false");
  });

  it("toggles sidebar with Ctrl+B shortcut unless focus is inside an input field", () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));

    render(<Dashboard />);

    const toggleBtn = screen.getByTestId("sidebar-toggle-btn");
    const sidebar = screen.getByTestId("native-sidebar");

    expect(toggleBtn.getAttribute("aria-expanded")).toBe("true");

    // Ctrl+B on window hides sidebar
    fireEvent.keyDown(window, { key: "b", ctrlKey: true });
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(sidebar.hidden).toBe(true);

    // Ctrl+B inside an input does NOT toggle sidebar
    const roomInput = screen.getByTestId("room-id-input");
    fireEvent.keyDown(roomInput, { key: "b", ctrlKey: true });
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("false");
    expect(sidebar.hidden).toBe(true);

    // Meta+B on window restores sidebar
    fireEvent.keyDown(window, { key: "B", metaKey: true });
    expect(toggleBtn.getAttribute("aria-expanded")).toBe("true");
    expect(sidebar.hidden).toBe(false);
  });

  it("blocks model download and room join immediately when /rooms/check-join returns 409", async () => {
    const mockRooms: RoomLobbyItem[] = [
      {
        room_id: "deepseek-r1-vn-01",
        model: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        nodes_online: 3,
        total_shared_ram_gb: 12,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 100,
        redundancy_factor: 1,
      },
    ];

    const rejectionDetail =
      'Room "deepseek-r1-vn-01" is at 100% RAM/Layer capacity and cannot accept new nodes.';
    const calledUrls: string[] = [];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      calledUrls.push(url);
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify(mockRooms), { status: 200 });
      }
      if (url.includes("/network-status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://network.shardai.online",
            lan_peers_count: 0,
            zeroconf_available: false,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/rooms/check-join")) {
        return new Response(JSON.stringify({ detail: rejectionDetail }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId("join-room-deepseek-r1-vn-01")).toBeTruthy();
    });

    fireEvent.click(screen.getByTestId("join-room-deepseek-r1-vn-01"));

    await waitFor(() => {
      expect(screen.getByText(rejectionDetail)).toBeTruthy();
    });

    expect(calledUrls.some((u) => u.includes("/rooms/check-join"))).toBe(true);
    expect(calledUrls.some((u) => u.includes("/models/download-stream"))).toBe(false);
    expect(calledUrls.some((u) => u.includes("/start-node"))).toBe(false);
    expect(screen.queryByTestId("model-download-progress-bar")).toBeNull();
  });

  it("renders English readiness badges in Rooms Lobby and API Gateway with zero Vietnamese text", async () => {
    const vietnameseDiacriticsRe =
      /[\u00C0-\u00C3\u00C8-\u00CA\u00CC-\u00CD\u00D2-\u00D5\u00D9-\u00DA\u00DD\u00E0-\u00E3\u00E8-\u00EA\u00EC-\u00ED\u00F2-\u00F5\u00F9-\u00FA\u00FD\u0102-\u0103\u0110-\u0111\u0128-\u0129\u0168-\u0169\u01A0-\u01B0\u1EA0-\u1EF9]/;

    const mockRooms: RoomLobbyItem[] = [
      {
        room_id: "ready-room",
        model: "meta-llama/Llama-3.1-8B-Instruct",
        nodes_online: 4,
        total_shared_ram_gb: 24,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 100,
        redundancy_factor: 2,
      },
      {
        room_id: "pooling-room",
        model: "Qwen/Qwen2.5-7B-Instruct",
        nodes_online: 1,
        total_shared_ram_gb: 6,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 65,
        redundancy_factor: 1,
      },
    ];

    const standbyStatus: NodeStatusResponse = {
      state: "RUNNING",
      node_id: "node-en-01",
      model_id: "Qwen/Qwen2.5-7B-Instruct",
      room_id: "pooling-room",
      is_private: false,
      is_unlisted: false,
      backend: "llama_cpp",
      n_layers: 16,
      allocated_ram_mb: 4096,
      allocated_vram_mb: 0,
      p2p_listen_port: 49152,
      connected_peers_count: 1,
      peers: [],
      session_token: "sk-shardai-test-token",
      api_base_url: "http://127.0.0.1:8000/v1",
      coverage_percent: 50,
      is_swarm_ready: false,
      message: "Gateway Active",
    };

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify(standbyStatus), { status: 200 });
      }
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify(mockRooms), { status: 200 });
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      return new Response("{}", { status: 200 });
    });

    const { container, unmount } = render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId("room-coverage-ready-room").textContent).toContain(
        "Ready (Redundancy: 2x)"
      );
      expect(screen.getByTestId("room-coverage-pooling-room").textContent).toContain(
        "Pooling RAM: 65%"
      );
      expect(screen.getByTestId("api-readiness-badge").textContent).toContain(
        "API Standby (Waiting for sufficient RAM: 50% - Model not ready)"
      );
    });

    expect(vietnameseDiacriticsRe.test(container.textContent ?? "")).toBe(false);
    unmount();

    const readyStatus: NodeStatusResponse = {
      ...standbyStatus,
      coverage_percent: 100,
      is_swarm_ready: true,
    };

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify(readyStatus), { status: 200 });
      }
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify(mockRooms), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    const readyRender = render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("api-readiness-badge").textContent).toContain(
        "API Ready (Swarm is ready to accept requests)"
      );
    });
    expect(vietnameseDiacriticsRe.test(readyRender.container.textContent ?? "")).toBe(false);
  });

  it("normalizes Vietnamese API error notices to English while preserving English notices verbatim", async () => {
    const englishNotice = 'Room "alpha" is at 100% RAM/Layer capacity and locked.';
    expect(normalizeNoticeToEnglish(englishNotice, "Fallback notice", "ROOM_FULL")).toBe(
      englishNotice
    );

    const vnFullNotice =
      'Room "deepseek-r1-01" \u0111\u00E3 \u0111\u1EA1t 100% RAM/Layer v\u00E0 \u0111ang kh\u00F3a phi\u00EAn ch\u1EA1y model.';
    const fullRes = new Response(
      JSON.stringify({
        error: { code: "ROOM_FULL", message: vnFullNotice },
        message: vnFullNotice,
      }),
      { status: 409, headers: { "Content-Type": "application/json" } }
    );
    const extractedFull = await extractApiErrorDetail(fullRes, "Fallback full");
    expect(extractedFull).toBe(
      'Room "deepseek-r1-01" has reached 100% RAM/Layer capacity and is locked for active inference. Please select another room or create a new room.'
    );

    const vnBusyNotice =
      'Room "qwen-7b" (65% RAM) hi\u1EC7n \u0111\u00E3 c\u00F3 l\u01B0\u1EE3t ch\u1EDD k\u1EBFt n\u1ED1i.';
    const busyRes = new Response(
      JSON.stringify({
        error: { code: "ROOM_QUEUE_BUSY", message: vnBusyNotice },
      }),
      { status: 409, headers: { "Content-Type": "application/json" } }
    );
    const extractedBusy = await extractApiErrorDetail(busyRes, "Fallback busy");
    expect(extractedBusy).toBe(
      'Room "qwen-7b" (65% RAM) currently has another client joining and syncing layers. Please wait or create a new room.'
    );

    const vnGenericNotice = "L\u1ED7i kh\u00F4ng x\u00E1c \u0111\u1ECBnh t\u1EEB m\u00E1y ch\u1EE7.";
    expect(normalizeNoticeToEnglish(vnGenericNotice, "Unknown server error.")).toBe(
      "Unknown server error."
    );
  });

  it("renders frameless color dots with hover tooltips for network status and gateway active/standby", async () => {
    const scenarios: Array<{
      network_status: "online" | "lan_only" | "offline_solo";
      lan_peers_count: number;
      expectedTone: "green" | "yellow" | "red";
      expectedTooltipFragment: string;
    }> = [
      {
        network_status: "online",
        lan_peers_count: 0,
        expectedTone: "green",
        expectedTooltipFragment: "Network: Online (Cloud)",
      },
      {
        network_status: "lan_only",
        lan_peers_count: 3,
        expectedTone: "yellow",
        expectedTooltipFragment: "Network: LAN Only (3 LAN peer(s)",
      },
      {
        network_status: "offline_solo",
        lan_peers_count: 0,
        expectedTone: "red",
        expectedTooltipFragment: "Network: Offline (Solo)",
      },
    ];

    for (const scenario of scenarios) {
      vi.restoreAllMocks();
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/network-status")) {
          return new Response(
            JSON.stringify({
              network_status: scenario.network_status,
              tracker_reachable: scenario.network_status === "online",
              tracker_url: "https://network.shardai.online",
              lan_peers_count: scenario.lan_peers_count,
              zeroconf_available: true,
            }),
            { status: 200 }
          );
        }
        return new Response("{}", { status: 200 });
      });

      const { unmount } = render(<Dashboard />);

      await waitFor(() => {
        const netDot = screen.getByTestId("network-status-indicator");
        expect(netDot.getAttribute("data-tone")).toBe(scenario.expectedTone);
        expect(netDot.getAttribute("data-status")).toBe(scenario.network_status);
        expect(netDot.getAttribute("title")).toContain(scenario.expectedTooltipFragment);
        expect(netDot.getAttribute("data-tooltip")).toContain(scenario.expectedTooltipFragment);
        expect(netDot.getAttribute("aria-label")).toContain(scenario.expectedTooltipFragment);
        expect(netDot.classList.contains("titlebar-status-dot")).toBe(true);
        expect(netDot.textContent?.trim()).toBe("");
        expect(netDot.querySelector(`.status-dot--${scenario.expectedTone}`)).toBeTruthy();
      });

      // Gateway starts in frameless yellow Standby dot with tooltip
      const gatewayStandbyDot = screen.getByTestId("gateway-badge");
      expect(gatewayStandbyDot.getAttribute("data-status")).toBe("standby");
      expect(gatewayStandbyDot.getAttribute("data-tone")).toBe("yellow");
      expect(gatewayStandbyDot.getAttribute("title")).toBe("Gateway: Standby");
      expect(gatewayStandbyDot.getAttribute("data-tooltip")).toBe("Gateway: Standby");
      expect(gatewayStandbyDot.textContent?.trim()).toBe("");
      expect(gatewayStandbyDot.querySelector(".status-dot--yellow")).toBeTruthy();

      unmount();
    }

    // Active gateway state renders green dot with Gateway: Active tooltip
    vi.restoreAllMocks();
    const activeStatus: NodeStatusResponse = {
      state: "RUNNING",
      node_id: "node-dot-01",
      model_id: "meta-llama/Llama-3.1-8B-Instruct",
      room_id: "public",
      is_private: false,
      is_unlisted: false,
      backend: "llama_cpp",
      n_layers: 16,
      allocated_ram_mb: 4096,
      allocated_vram_mb: 0,
      p2p_listen_port: 49152,
      connected_peers_count: 2,
      peers: [],
      session_token: "sk-shardai-active-dot",
      api_base_url: "http://127.0.0.1:8000/v1",
      coverage_percent: 100,
      is_swarm_ready: true,
      message: "Active",
    };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify(activeStatus), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);
    await waitFor(() => {
      const gatewayActiveDot = screen.getByTestId("gateway-badge");
      expect(gatewayActiveDot.getAttribute("data-status")).toBe("active");
      expect(gatewayActiveDot.getAttribute("data-tone")).toBe("green");
      expect(gatewayActiveDot.getAttribute("title")).toBe("Gateway: Active");
      expect(gatewayActiveDot.getAttribute("data-tooltip")).toBe("Gateway: Active");
      expect(gatewayActiveDot.textContent?.trim()).toBe("");
      expect(gatewayActiveDot.querySelector(".status-dot--green.status-dot--active")).toBeTruthy();
    });
  });

  it("toggles frameless Sun/Moon theme icon strictly between light and dark without Auto mode", () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("{}", { status: 200 }));

    // Legacy "system" value in localStorage must migrate cleanly to a concrete theme
    window.localStorage.setItem("shardai_theme", "system");

    const { unmount } = render(<Dashboard />);

    const themeBtn = screen.getByTestId("theme-toggle-btn");
    expect(themeBtn.classList.contains("titlebar-icon-btn")).toBe(true);
    expect(themeBtn.classList.contains("native-btn")).toBe(false);
    expect(themeBtn.textContent?.trim()).toBe("");

    const initialMode = themeBtn.getAttribute("data-theme-mode");
    expect(initialMode === "light" || initialMode === "dark").toBe(true);
    expect(window.localStorage.getItem("shardai_theme")).toBe(initialMode);
    expect(document.documentElement.getAttribute("data-theme")).toBe(initialMode);
    expect(
      screen.getByTestId(initialMode === "dark" ? "theme-icon-moon" : "theme-icon-sun")
    ).toBeTruthy();

    // First click switches directly to the opposite mode (never "system" / "Auto")
    const secondMode = initialMode === "dark" ? "light" : "dark";
    fireEvent.click(themeBtn);
    expect(themeBtn.getAttribute("data-theme-mode")).toBe(secondMode);
    expect(document.documentElement.getAttribute("data-theme")).toBe(secondMode);
    expect(window.localStorage.getItem("shardai_theme")).toBe(secondMode);
    expect(
      screen.getByTestId(secondMode === "dark" ? "theme-icon-moon" : "theme-icon-sun")
    ).toBeTruthy();
    expect(themeBtn.getAttribute("title")).not.toContain("Auto");
    expect(themeBtn.getAttribute("title")).not.toContain("System");

    // Second click switches directly back to initialMode (pure 2-state cycle)
    fireEvent.click(themeBtn);
    expect(themeBtn.getAttribute("data-theme-mode")).toBe(initialMode);
    expect(window.localStorage.getItem("shardai_theme")).toBe(initialMode);

    unmount();
  });

  it("renders a frameless cloud icon with tooltip for catalog sync source instead of a text pill", async () => {
    const sources: Array<{ source: "cloud" | "cache" | "default"; expectedTooltip: string }> = [
      { source: "cloud", expectedTooltip: "Synced from Cloud" },
      { source: "cache", expectedTooltip: "Loaded from ~/.shardai/models_cache.json" },
      { source: "default", expectedTooltip: "Default Offline Catalog" },
    ];

    for (const entry of sources) {
      vi.restoreAllMocks();
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/models/catalog")) {
          return new Response(
            JSON.stringify({
              source: entry.source,
              count: 1,
              models: [
                {
                  id: "llama-3.1-8b",
                  name: "Llama 3.1 8B",
                  repo_id: "meta-llama/Llama-3.1-8B-Instruct",
                  estimated_ram_gb: 6,
                },
              ],
            }),
            { status: 200 }
          );
        }
        return new Response("{}", { status: 200 });
      });

      const { unmount } = render(<Dashboard />);

      await waitFor(() => {
        const cloudBadge = screen.getByTestId("catalog-source-badge");
        expect(cloudBadge.classList.contains("catalog-cloud-indicator")).toBe(true);
        expect(cloudBadge.classList.contains(`catalog-cloud-indicator--${entry.source}`)).toBe(
          true
        );
        expect(cloudBadge.getAttribute("data-source")).toBe(entry.source);
        expect(cloudBadge.getAttribute("title")).toBe(entry.expectedTooltip);
        expect(cloudBadge.getAttribute("data-tooltip")).toBe(entry.expectedTooltip);
        expect(cloudBadge.getAttribute("aria-label")).toBe(entry.expectedTooltip);
        expect(cloudBadge.textContent?.trim()).toBe("");
        expect(screen.getByTestId("catalog-cloud-icon")).toBeTruthy();
      });

      unmount();
    }
  });


  it("never kills Python sidecar and clears download progress on HTTP 404 and SSE download errors", async () => {
    (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
    const mockedInvoke = vi.mocked(invoke);
    mockedInvoke.mockImplementation(async (cmd: string) => {
      if (cmd === "get_available_ram_gb") return 12;
      if (cmd === "start_python_core") return "ready";
      if (cmd === "stop_python_core") return "stopped";
      return "";
    });

    let downloadScenario: "http_404" | "sse_error" = "http_404";
    let sseController: ReadableStreamDefaultController<Uint8Array> | null = null;
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response("[]", { status: 200 });
      }
      if (url.includes("/network-status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://network.shardai.online",
            lan_peers_count: 0,
            zeroconf_available: false,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: 1,
            models: [
              {
                id: "llama3",
                name: "Llama 3.1 8B",
                repo_id: "meta-llama/Llama-3.1-8B-Instruct",
                filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                estimated_ram_gb: 6,
                enabled: true,
              },
            ],
          }),
          { status: 200 }
        );
      }
      if (url.includes("/rooms/check-join")) {
        return new Response(
          JSON.stringify({ allowed: true, room_id: "public", message: "OK" }),
          { status: 200 }
        );
      }
      if (url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      }
      if (url.includes("/models/download-stream")) {
        if (downloadScenario === "http_404") {
          return new Response(
            JSON.stringify({ detail: "Local GGUF file not found." }),
            { status: 404, headers: { "Content-Type": "application/json" } }
          );
        }
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            sseController = controller;
            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  model_id: "meta-llama/Llama-3.1-8B-Instruct",
                  status: "downloading",
                  progress: 40,
                  downloaded_bytes: 40,
                  total_bytes: 100,
                  local_path: null,
                  message: "Downloading 40%",
                })}\n\n`
              )
            );
          },
        });
        return new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    // Sub-case A: HTTP 404 from /models/download-stream
    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain(
        "Local GGUF file not found."
      );
    });
    expect(screen.queryByTestId("model-download-progress-bar")).toBeNull();
    expect(
      (screen.getByTestId("toggle-sharing-btn") as HTMLButtonElement).disabled
    ).toBe(false);
    expect(mockedInvoke).toHaveBeenCalledWith("start_python_core", undefined);
    expect(mockedInvoke).not.toHaveBeenCalledWith("stop_python_core", undefined);

    // Sub-case B: SSE stream emits progress (40%) followed by an error event
    downloadScenario = "sse_error";
    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("model-download-progress-bar")).toBeTruthy();
      expect(screen.getByTestId("model-download-percent").textContent).toBe("40%");
    });

    const activeController = sseController as ReadableStreamDefaultController<Uint8Array> | null;
    if (activeController) {
      activeController.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            model_id: "meta-llama/Llama-3.1-8B-Instruct",
            status: "error",
            progress: 40,
            downloaded_bytes: 40,
            total_bytes: 100,
            local_path: null,
            message: "Network download interrupted",
          })}\n\n`
        )
      );
      activeController.close();
    }

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain(
        "Network download interrupted"
      );
      expect(screen.queryByTestId("model-download-progress-bar")).toBeNull();
    });
    expect(screen.getByTestId("gateway-badge").getAttribute("data-status")).toBe("standby");
    expect(mockedInvoke).not.toHaveBeenCalledWith("stop_python_core", undefined);
  });

  it("syncs UI to standby when 5-second /status poll observes STOPPED or ERROR transitions", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });

    const runningStatus: NodeStatusResponse = {
      state: "RUNNING",
      node_id: "node-sync-01",
      model_id: "meta-llama/Llama-3.1-8B-Instruct",
      room_id: "public",
      is_private: false,
      is_unlisted: false,
      backend: "llama_cpp",
      n_layers: 32,
      total_layers: 32,
      allocated_ram_mb: 4096,
      allocated_vram_mb: 0,
      p2p_listen_port: 49152,
      connected_peers_count: 1,
      peers: [],
      session_token: "sk-live-token",
      api_base_url: "http://127.0.0.1:8000/v1",
      coverage_percent: 100,
      is_swarm_ready: true,
      network_status: "online",
      message: "Gateway Active",
    };

    let currentStatus: NodeStatusResponse = runningStatus;

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/status")) {
        return new Response(JSON.stringify(currentStatus), { status: 200 });
      }
      if (url.includes("/available-rooms")) {
        return new Response("[]", { status: 200 });
      }
      if (url.includes("/network-status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://network.shardai.online",
            lan_peers_count: 0,
            zeroconf_available: false,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      return new Response("{}", { status: 200 });
    });

    // Sub-case A: RUNNING -> STOPPED
    const firstRender = render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId("session-credentials-panel")).toBeTruthy();
      expect(screen.getByTestId("gateway-badge").getAttribute("data-status")).toBe("active");
      expect(
        (screen.getByTestId("room-id-input") as HTMLInputElement).disabled
      ).toBe(true);
    });

    currentStatus = {
      ...runningStatus,
      state: "STOPPED",
      session_token: null,
      allocated_ram_mb: 0,
      allocated_vram_mb: 0,
      connected_peers_count: 0,
      coverage_percent: 0,
      is_swarm_ready: false,
      message: "Stopped via CLI",
    };

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    await waitFor(() => {
      expect(screen.queryByTestId("session-credentials-panel")).toBeNull();
      expect(screen.getByTestId("gateway-badge").getAttribute("data-status")).toBe("standby");
      expect(
        (screen.getByTestId("room-id-input") as HTMLInputElement).disabled
      ).toBe(false);
      expect(screen.getByTestId("status-message").textContent).toContain(
        "Stopped — Stopped via CLI"
      );
    });

    firstRender.unmount();

    // Sub-case B: RUNNING -> ERROR
    currentStatus = runningStatus;
    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId("session-credentials-panel")).toBeTruthy();
    });

    currentStatus = {
      ...runningStatus,
      state: "ERROR",
      session_token: null,
      allocated_ram_mb: 0,
      allocated_vram_mb: 0,
      connected_peers_count: 0,
      coverage_percent: 0,
      is_swarm_ready: false,
      message: "Worker crashed",
    };

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    await waitFor(() => {
      expect(screen.queryByTestId("session-credentials-panel")).toBeNull();
      expect(screen.getByTestId("gateway-badge").getAttribute("data-status")).toBe("standby");
      expect(
        (screen.getByTestId("room-id-input") as HTMLInputElement).disabled
      ).toBe(false);
      expect(screen.getByRole("alert").textContent).toContain("Worker crashed");
      expect(screen.getByTestId("status-message").textContent).toContain(
        "Error — Worker crashed"
      );
    });
  });

  it("uses resolveActiveModelSelection for handleJoinByInviteString across Local File and Online Catalog tabs", async () => {
    const catalogItems: CatalogModelItem[] = [
      {
        id: "llama3",
        name: "Llama 3.1 8B",
        repo_id: "meta-llama/Llama-3.1-8B-Instruct",
        filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        estimated_ram_gb: 6,
        enabled: true,
      },
      {
        id: "qwen7b",
        name: "Qwen 2.5 7B",
        repo_id: "Qwen/Qwen2.5-7B-Instruct",
        filename: "qwen2.5-7b-q4_k_m.gguf",
        estimated_ram_gb: 5,
        enabled: true,
      },
    ];

    const startPayloads: StartNodeRequest[] = [];
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: catalogItems.length,
            models: catalogItems,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/available-rooms")) {
        return new Response("[]", { status: 200 });
      }
      if (url.includes("/network-status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://network.shardai.online",
            lan_peers_count: 0,
            zeroconf_available: false,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/rooms/check-join")) {
        return new Response(
          JSON.stringify({ allowed: true, room_id: "secret-lab", message: "OK" }),
          { status: 200 }
        );
      }
      if (url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      }
      if (url.includes("/models/download-stream")) {
        const doneStream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });
        return new Response(doneStream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url.endsWith("/start-node") && init?.body) {
        const parsed = JSON.parse(String(init.body)) as StartNodeRequest;
        startPayloads.push(parsed);
        const statusResp: NodeStatusResponse = {
          state: "RUNNING",
          node_id: parsed.node_id,
          model_id: parsed.model_id,
          network_mode: parsed.network_mode,
          room_id: parsed.room_id,
          is_private: Boolean(parsed.room_key),
          is_unlisted: Boolean(parsed.is_unlisted),
          backend: parsed.backend,
          n_layers: 32,
          total_layers: 32,
          allocated_ram_mb: parsed.max_ram_mb,
          allocated_vram_mb: parsed.max_vram_mb,
          p2p_listen_port: parsed.p2p_listen_port,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-invite-test",
          api_base_url: "http://127.0.0.1:8000/v1",
          coverage_percent: 100,
          is_swarm_ready: true,
          network_status: "online",
          message: "Active",
        };
        return new Response(JSON.stringify(statusResp), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    // Sub-case A: Local File tab active when joining via invite string
    const localRender = render(<Dashboard />);
    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("tab-local"));
    fireEvent.change(screen.getByTestId("local-file-input"), {
      target: { value: "C:/models/qwen-local.gguf" },
    });

    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    fireEvent.change(screen.getByTestId("paste-invite-input"), {
      target: { value: "secret-lab | key-42" },
    });
    fireEvent.click(screen.getByTestId("join-by-invite-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(1);
    });
    expect(startPayloads[0]).toMatchObject({
      model_id: "C:/models/qwen-local.gguf",
      source_type: "local_file",
      filename: null,
      room_id: "secret-lab",
      room_key: "key-42",
      is_unlisted: true,
    });
    localRender.unmount();

    // Sub-case B: Online Catalog tab with second catalog item selected
    render(<Dashboard />);
    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.change(screen.getByTestId("catalog-model-select"), {
      target: { value: "qwen7b" },
    });
    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    fireEvent.change(screen.getByTestId("paste-invite-input"), {
      target: { value: "team-alpha | pass99" },
    });
    fireEvent.click(screen.getByTestId("join-by-invite-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(2);
    });
    expect(startPayloads[1]).toMatchObject({
      model_id: "Qwen/Qwen2.5-7B-Instruct",
      source_type: "catalog",
      filename: "qwen2.5-7b-q4_k_m.gguf",
      room_id: "team-alpha",
      room_key: "pass99",
      is_unlisted: true,
    });
  });

  it("synchronizes modelTab and selectedCatalogId when joining a lobby room so retries after 409 use room.model", async () => {
    const catalogItems: CatalogModelItem[] = [
      {
        id: "llama3",
        name: "Llama 3.1 8B",
        repo_id: "meta-llama/Llama-3.1-8B-Instruct",
        filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        estimated_ram_gb: 6,
        enabled: true,
      },
      {
        id: "qwen7b",
        name: "Qwen 2.5 7B",
        repo_id: "Qwen/Qwen2.5-7B-Instruct",
        filename: "qwen2.5-7b-q4_k_m.gguf",
        estimated_ram_gb: 5,
        enabled: true,
      },
    ];

    const mockRooms: RoomLobbyItem[] = [
      {
        room_id: "custom-room",
        model: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        nodes_online: 1,
        total_shared_ram_gb: 8,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 50,
        redundancy_factor: 1,
      },
      {
        room_id: "qwen-room",
        model: "Qwen/Qwen2.5-7B-Instruct",
        nodes_online: 1,
        total_shared_ram_gb: 6,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 75,
        redundancy_factor: 1,
      },
    ];

    let rejectRoomJoin = true;
    const startPayloads: StartNodeRequest[] = [];
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: catalogItems.length,
            models: catalogItems,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify(mockRooms), { status: 200 });
      }
      if (url.includes("/network-status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://network.shardai.online",
            lan_peers_count: 0,
            zeroconf_available: false,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 11,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/rooms/check-join")) {
        if (rejectRoomJoin) {
          return new Response(
            JSON.stringify({ detail: "Room is busy right now." }),
            { status: 409, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response(
          JSON.stringify({ allowed: true, room_id: "ok", message: "OK" }),
          { status: 200 }
        );
      }
      if (url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      }
      if (url.includes("/models/download-stream")) {
        const doneStream = new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });
        return new Response(doneStream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      if (url.endsWith("/start-node") && init?.body) {
        const parsed = JSON.parse(String(init.body)) as StartNodeRequest;
        startPayloads.push(parsed);
        const statusResp: NodeStatusResponse = {
          state: "RUNNING",
          node_id: parsed.node_id,
          model_id: parsed.model_id,
          network_mode: parsed.network_mode,
          room_id: parsed.room_id,
          is_private: false,
          is_unlisted: false,
          backend: parsed.backend,
          n_layers: 32,
          total_layers: 32,
          allocated_ram_mb: parsed.max_ram_mb,
          allocated_vram_mb: parsed.max_vram_mb,
          p2p_listen_port: parsed.p2p_listen_port,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-retry-test",
          api_base_url: "http://127.0.0.1:8000/v1",
          coverage_percent: 100,
          is_swarm_ready: true,
          network_status: "online",
          message: "Active",
        };
        return new Response(JSON.stringify(statusResp), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    // Sub-case A: Custom model room (not in catalog)
    const firstRender = render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("join-room-custom-room")).toBeTruthy();
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("join-room-custom-room"));

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain(
        "Room is busy right now."
      );
      expect(screen.getByTestId("panel-custom")).toBeTruthy();
      expect((screen.getByTestId("model-input") as HTMLInputElement).value).toBe(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
      );
    });

    rejectRoomJoin = false;
    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(1);
    });
    expect(startPayloads[0]).toMatchObject({
      model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
      source_type: "huggingface",
      room_id: "custom-room",
    });
    firstRender.unmount();

    // Sub-case B: Catalog model room (second catalog item)
    rejectRoomJoin = true;
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("join-room-qwen-room")).toBeTruthy();
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("join-room-qwen-room"));

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain(
        "Room is busy right now."
      );
      expect(screen.getByTestId("panel-catalog")).toBeTruthy();
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).value
      ).toBe("qwen7b");
    });

    rejectRoomJoin = false;
    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(2);
    });
    expect(startPayloads[1]).toMatchObject({
      model_id: "Qwen/Qwen2.5-7B-Instruct",
      source_type: "catalog",
      filename: "qwen2.5-7b-q4_k_m.gguf",
      room_id: "qwen-room",
    });
  });
});
