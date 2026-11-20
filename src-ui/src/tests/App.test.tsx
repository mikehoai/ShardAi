import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { invoke } from "@tauri-apps/api/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Dashboard,
  formatCompactModelName,
  formatCompactRoomName,
  orderRoomsForSession,
} from "../components/Dashboard";
import { ChatWindow } from "../components/ChatWindow";
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

  it("randomizes lobby rooms per session, hides network scope tags, compacts names, and renders frameless colored status text", async () => {
    // 1. Compact formatting helpers
    expect(formatCompactRoomName("deepseek-r1-vn-01")).toBe("vn-01");
    expect(formatCompactRoomName("qwen-coder-32b-pro")).toBe("pro");
    expect(formatCompactRoomName("llama3-8b-asia")).toBe("asia");
    expect(formatCompactRoomName("vn-01")).toBe("vn-01");
    expect(formatCompactRoomName("public")).toBe("public");
    expect(formatCompactRoomName("very-long-custom-room-identifier-99").length).toBeLessThanOrEqual(
      18
    );

    expect(formatCompactModelName("bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF")).toBe(
      "DeepSeek-R1-Qwen-14B"
    );
    expect(formatCompactModelName("Qwen/Qwen2.5-Coder-32B-Instruct-GGUF")).toBe(
      "Qwen2.5-Coder-32B"
    );
    expect(formatCompactModelName("meta-llama/Llama-3.1-8B-Instruct")).toBe("Llama-3.1-8B");

    // 2. Session-scoped randomization varies across seeds and is deterministic per seed
    const mockRooms: RoomLobbyItem[] = [
      {
        room_id: "deepseek-r1-vn-01",
        model: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        nodes_online: 3,
        total_shared_ram_gb: 12,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 100,
        redundancy_factor: 2,
      },
      {
        room_id: "qwen-coder-7b-dev",
        model: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        nodes_online: 2,
        total_shared_ram_gb: 6,
        is_private: true,
        network_scope: "lan",
        coverage_percent: 72,
        redundancy_factor: 1,
      },
      {
        room_id: "sg-02",
        model: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        nodes_online: 4,
        total_shared_ram_gb: 24,
        is_private: false,
        network_scope: "hybrid",
        coverage_percent: 100,
        redundancy_factor: 1,
      },
      {
        room_id: "jp-03",
        model: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        nodes_online: 2,
        total_shared_ram_gb: 8,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 85,
        redundancy_factor: 1,
      },
      {
        room_id: "eu-05",
        model: "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        nodes_online: 3,
        total_shared_ram_gb: 10,
        is_private: false,
        network_scope: "internet",
        coverage_percent: 100,
        redundancy_factor: 1,
      },
    ];

    const orders = new Set<string>();
    for (let seed = 1; seed <= 10; seed += 1) {
      orders.add(
        orderRoomsForSession(mockRooms, seed)
          .map((r) => r.room_id)
          .join(",")
      );
    }
    expect(orders.size).toBeGreaterThan(1);
    expect(orderRoomsForSession(mockRooms, 42).map((r) => r.room_id)).toEqual(
      orderRoomsForSession(mockRooms, 42).map((r) => r.room_id)
    );

    // 3. Rendered table checks (no INTERNET/LAN/HYBRID tag, compact names, frameless colored text, no icons)
    const checkJoinPayloads: Array<{ room_id?: string; model_id?: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
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
      if (url.includes("/rooms/check-join")) {
        if (init?.body && typeof init.body === "string") {
          checkJoinPayloads.push(JSON.parse(init.body) as { room_id?: string; model_id?: string });
        }
        return new Response(JSON.stringify({ detail: "Room is full." }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);

    await waitFor(() => {
      expect(screen.getByTestId("room-name-deepseek-r1-vn-01")).toBeTruthy();
      expect(screen.getByTestId("room-name-qwen-coder-7b-dev")).toBeTruthy();
    });

    const readyNameCell = screen.getByTestId("room-name-deepseek-r1-vn-01");
    expect(readyNameCell.textContent?.trim()).toBe("vn-01");
    expect(readyNameCell.getAttribute("title")).toBe("deepseek-r1-vn-01");
    expect(readyNameCell.textContent).not.toContain("INTERNET");
    expect(readyNameCell.textContent).not.toContain("LAN");
    expect(readyNameCell.textContent).not.toContain("HYBRID");

    const lanNameCell = screen.getByTestId("room-name-qwen-coder-7b-dev");
    expect(lanNameCell.textContent?.trim()).toBe("dev");
    expect(lanNameCell.getAttribute("title")).toBe("qwen-coder-7b-dev");
    expect(lanNameCell.textContent).not.toContain("LAN");

    const modelCell = screen.getByTestId("room-model-deepseek-r1-vn-01");
    expect(modelCell.textContent?.trim()).toBe("DeepSeek-R1-Qwen-14B");
    expect(modelCell.getAttribute("title")).toBe("bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF");

    // Resources column: frameless colored text, zero border/box inline style, zero emojis
    const readyResource = screen.getByTestId("room-coverage-deepseek-r1-vn-01");
    expect(readyResource.classList.contains("room-resource-text")).toBe(true);
    expect(readyResource.classList.contains("room-resource-text--ready")).toBe(true);
    expect(readyResource.textContent?.trim()).toBe("Ready (Redundancy: 2x)");
    expect(readyResource.textContent).not.toContain("✅");
    expect(readyResource.getAttribute("style")).toBeNull();

    const poolingResource = screen.getByTestId("room-coverage-qwen-coder-7b-dev");
    expect(poolingResource.classList.contains("room-resource-text")).toBe(true);
    expect(poolingResource.classList.contains("room-resource-text--pooling")).toBe(true);
    expect(poolingResource.textContent?.trim()).toBe("Pooling RAM: 72%");
    expect(poolingResource.textContent).not.toContain("⚠️");
    expect(poolingResource.getAttribute("style")).toBeNull();

    // Status column: frameless colored text, zero lock emoji
    const publicStatus = screen.getByTestId("room-status-deepseek-r1-vn-01");
    expect(publicStatus.classList.contains("room-status-text--public")).toBe(true);
    expect(publicStatus.textContent?.trim()).toBe("Public");

    const privateStatus = screen.getByTestId("room-status-qwen-coder-7b-dev");
    expect(privateStatus.classList.contains("room-status-text--private")).toBe(true);
    expect(privateStatus.textContent?.trim()).toBe("Private");
    expect(privateStatus.textContent).not.toContain("🔒");

    // Clicking Join still sends the exact underlying room_id and full model_id
    fireEvent.click(screen.getByTestId("join-room-deepseek-r1-vn-01"));
    await waitFor(() => {
      expect(checkJoinPayloads).toHaveLength(1);
    });
    expect(checkJoinPayloads[0]).toMatchObject({
      room_id: "deepseek-r1-vn-01",
      model_id: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    });
  });

  it("allows selecting and refreshing an Online Catalog model inside Create New Room modal", async () => {
    const encoder = new TextEncoder();
    const mockCatalog: CatalogModelItem[] = [
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

    let catalogFetchCount = 0;
    const downloadUrls: string[] = [];
    const startPayloads: StartNodeRequest[] = [];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/models/catalog")) {
        catalogFetchCount += 1;
        return new Response(
          JSON.stringify({ source: "cloud", count: mockCatalog.length, models: mockCatalog }),
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
      if (url.includes("/rooms/check-join")) {
        return new Response(
          JSON.stringify({ allowed: true, room_id: "qwen-studio", message: "OK" }),
          { status: 200 }
        );
      }
      if (url.endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
      }
      if (url.includes("/models/download-stream")) {
        downloadUrls.push(url);
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
          session_token: "sk-room-modal-catalog",
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

    render(<Dashboard />);

    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    expect(screen.getByTestId("create-room-modal")).toBeTruthy();
    expect(screen.getByTestId("modal-tab-catalog").getAttribute("aria-selected")).toBe("true");
    expect(
      (screen.getByTestId("modal-catalog-model-select") as HTMLSelectElement).value
    ).toBe("llama3");

    // Refresh catalog from inside the modal
    const countBeforeRefresh = catalogFetchCount;
    fireEvent.click(screen.getByTestId("modal-refresh-catalog-btn"));
    await waitFor(() => {
      expect(catalogFetchCount).toBeGreaterThan(countBeforeRefresh);
    });

    // Select second catalog model and create room
    fireEvent.change(screen.getByTestId("modal-room-name-input"), {
      target: { value: "qwen-studio" },
    });
    fireEvent.change(screen.getByTestId("modal-room-key-input"), {
      target: { value: "secret42" },
    });
    fireEvent.change(screen.getByTestId("modal-catalog-model-select"), {
      target: { value: "qwen7b" },
    });
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(1);
    });

    expect(startPayloads[0]).toMatchObject({
      model_id: "Qwen/Qwen2.5-7B-Instruct",
      source_type: "catalog",
      filename: "qwen2.5-7b-q4_k_m.gguf",
      room_id: "qwen-studio",
      room_key: "secret42",
    });
    expect(downloadUrls[0]).toContain("source_type=catalog");
    expect(downloadUrls[0]).toContain("filename=qwen2.5-7b-q4_k_m.gguf");
    // Main panel synced to the modal's chosen catalog model
    expect((screen.getByTestId("catalog-model-select") as HTMLSelectElement).value).toBe(
      "qwen7b"
    );
  });

  it("supports Custom HF Repo (+ filename) and Direct URL selection inside Create New Room modal", async () => {
    const encoder = new TextEncoder();
    const startPayloads: StartNodeRequest[] = [];

    const mockFetch = () =>
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/available-rooms")) {
          return new Response("[]", { status: 200 });
        }
        if (url.includes("/rooms/check-join")) {
          return new Response(
            JSON.stringify({ allowed: true, room_id: "custom-room", message: "OK" }),
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
            session_token: "sk-room-modal-custom",
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

    // Sub-case A: Custom HuggingFace Repo + custom .gguf filename
    mockFetch();
    const hfRender = render(<Dashboard />);

    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    fireEvent.click(screen.getByTestId("modal-tab-custom"));
    fireEvent.change(screen.getByTestId("modal-room-name-input"), {
      target: { value: "hf-room" },
    });
    fireEvent.change(screen.getByTestId("modal-model-input"), {
      target: { value: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" },
    });
    fireEvent.change(screen.getByTestId("modal-custom-filename-input"), {
      target: { value: "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" },
    });
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(1);
    });
    expect(startPayloads[0]).toMatchObject({
      model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
      source_type: "huggingface",
      filename: "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
      room_id: "hf-room",
    });
    hfRender.unmount();

    // Sub-case B: Direct .gguf URL
    vi.restoreAllMocks();
    mockFetch();
    render(<Dashboard />);

    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    fireEvent.click(screen.getByTestId("modal-tab-custom"));
    fireEvent.change(screen.getByTestId("modal-room-name-input"), {
      target: { value: "url-room" },
    });
    fireEvent.change(screen.getByTestId("modal-model-input"), {
      target: { value: "https://cdn.example.com/models/phi-3-mini.gguf" },
    });
    expect(screen.queryByTestId("modal-custom-filename-input")).toBeNull();
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(2);
    });
    expect(startPayloads[1]).toMatchObject({
      model_id: "https://cdn.example.com/models/phi-3-mini.gguf",
      source_type: "direct_url",
      filename: null,
      room_id: "url-room",
    });
  });

  it("supports Local File (.gguf) selection and validates empty model paths in Create New Room modal", async () => {
    const encoder = new TextEncoder();
    const startPayloads: StartNodeRequest[] = [];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response("[]", { status: 200 });
      }
      if (url.includes("/rooms/check-join")) {
        return new Response(
          JSON.stringify({ allowed: true, room_id: "local-gguf-room", message: "OK" }),
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
          session_token: "sk-room-modal-local",
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

    render(<Dashboard />);

    fireEvent.click(screen.getByTestId("create-room-modal-btn"));
    fireEvent.change(screen.getByTestId("modal-room-name-input"), {
      target: { value: "local-gguf-room" },
    });

    // Empty Custom model blocks submit and keeps modal open
    fireEvent.click(screen.getByTestId("modal-tab-custom"));
    fireEvent.change(screen.getByTestId("modal-model-input"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));
    expect(screen.getByRole("alert").textContent).toContain(
      "Please select a catalog model or enter a HuggingFace Repo ID / Direct URL."
    );
    expect(screen.getByTestId("create-room-modal")).toBeTruthy();
    expect(startPayloads).toHaveLength(0);

    // Empty Local File path blocks submit and keeps modal open
    fireEvent.click(screen.getByTestId("modal-tab-local"));
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));
    expect(screen.getByRole("alert").textContent).toContain(
      "Please browse or enter a valid local .gguf file path."
    );
    expect(screen.getByTestId("create-room-modal")).toBeTruthy();
    expect(startPayloads).toHaveLength(0);

    // Valid Local .gguf file submits with source_type: "local_file" and syncs main panel
    fireEvent.change(screen.getByTestId("modal-local-file-input"), {
      target: { value: "D:/models/llama-3.1-8b-q4_k_m.gguf" },
    });
    fireEvent.click(screen.getByTestId("modal-create-join-btn"));

    await waitFor(() => {
      expect(startPayloads).toHaveLength(1);
    });
    expect(startPayloads[0]).toMatchObject({
      model_id: "D:/models/llama-3.1-8b-q4_k_m.gguf",
      source_type: "local_file",
      filename: null,
      room_id: "local-gguf-room",
    });
    expect(screen.queryByTestId("create-room-modal")).toBeNull();
    expect((screen.getByTestId("local-file-input") as HTMLInputElement).value).toBe(
      "D:/models/llama-3.1-8b-q4_k_m.gguf"
    );
  });

  it("hides Open AI Chat button and shows Model not ready notice under Quick Test when RAM coverage is below 100%", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: 1,
            models: [
              {
                id: "llama8b",
                name: "Llama 3.1 8B",
                repo_id: "meta-llama/Llama-3.1-8B-Instruct",
                filename: "llama-3.1-8b.gguf",
                estimated_ram_gb: 6,
                enabled: true,
              },
            ],
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 10,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/network/status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://tracker.example.com",
            lan_peers_count: 0,
            zeroconf_available: true,
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
      if (url.includes("/models/download-stream")) {
        return new Response(
          `data: ${JSON.stringify({
            model_id: "meta-llama/Llama-3.1-8B-Instruct",
            status: "completed",
            progress: 100,
            downloaded_bytes: 1000,
            total_bytes: 1000,
            local_path: "/tmp/model.gguf",
            message: "Ready",
          })}\n\n`,
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.includes("/start-node")) {
        const standbyStatus: NodeStatusResponse = {
          state: "RUNNING",
          node_id: "node-standby",
          model_id: "meta-llama/Llama-3.1-8B-Instruct",
          room_id: "public",
          is_private: false,
          is_unlisted: false,
          role: "primary",
          backend: "llama_cpp",
          n_layers: 16,
          total_layers: 32,
          shard_layer_start: 0,
          shard_layer_end: 15,
          allocated_ram_mb: 4096,
          allocated_vram_mb: 0,
          p2p_listen_port: 49152,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-shard-standby-key",
          api_base_url: "http://127.0.0.1:8000/v1",
          coverage_percent: 50,
          is_swarm_ready: false,
          network_status: "online",
          message: "Waiting for more RAM",
        };
        return new Response(JSON.stringify(standbyStatus), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);
    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("session-credentials-panel")).toBeTruthy();
      expect(screen.getByTestId("chat-launcher-section")).toBeTruthy();
      expect(screen.getByTestId("chat-not-ready-notice")).toBeTruthy();
    });

    expect(screen.queryByTestId("open-chat-window-btn")).toBeNull();
    expect(screen.getByTestId("chat-not-ready-notice").textContent).toContain(
      "Model not ready"
    );
    expect(screen.getByTestId("chat-not-ready-notice").textContent).toContain("50%");
  });

  it("shows Open AI Chat button under Quick Test when 100% RAM ready, streams SSE responses with Bearer token, supports multi-turn, clear, and Escape close", async () => {
    const chatRequests: Array<{
      url: string;
      authHeader: string | null;
      body: { model: string; messages: Array<{ role: string; content: string }>; stream: boolean };
    }> = [];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: 1,
            models: [
              {
                id: "qwen7b",
                name: "Qwen 2.5 7B",
                repo_id: "Qwen/Qwen2.5-7B-Instruct",
                filename: "qwen2.5-7b.gguf",
                estimated_ram_gb: 5,
                enabled: true,
              },
            ],
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 10,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/network/status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://tracker.example.com",
            lan_peers_count: 0,
            zeroconf_available: true,
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
      if (url.includes("/models/download-stream")) {
        return new Response(
          `data: ${JSON.stringify({
            model_id: "Qwen/Qwen2.5-7B-Instruct",
            status: "completed",
            progress: 100,
            downloaded_bytes: 1000,
            total_bytes: 1000,
            local_path: "/tmp/qwen.gguf",
            message: "Ready",
          })}\n\n`,
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.includes("/start-node")) {
        const readyStatus: NodeStatusResponse = {
          state: "RUNNING",
          node_id: "node-ready",
          model_id: "Qwen/Qwen2.5-7B-Instruct",
          room_id: "public",
          is_private: false,
          is_unlisted: false,
          role: "primary",
          backend: "llama_cpp",
          n_layers: 32,
          total_layers: 32,
          shard_layer_start: 0,
          shard_layer_end: 31,
          allocated_ram_mb: 6144,
          allocated_vram_mb: 0,
          p2p_listen_port: 49152,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-shard-ready-secret-123",
          api_base_url: "http://127.0.0.1:8000/v1",
          coverage_percent: 100,
          is_swarm_ready: true,
          network_status: "online",
          message: "Swarm ready",
        };
        return new Response(JSON.stringify(readyStatus), { status: 200 });
      }
      if (url.includes("/v1/chat/completions")) {
        const headers = new Headers(init?.headers);
        const parsedBody = JSON.parse(String(init?.body ?? "{}")) as {
          model: string;
          messages: Array<{ role: string; content: string }>;
          stream: boolean;
        };
        chatRequests.push({
          url,
          authHeader: headers.get("Authorization"),
          body: parsedBody,
        });

        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            const chunk1 = `data: ${JSON.stringify({
              choices: [{ delta: { role: "assistant", content: "Hello " } }],
            })}\n\n`;
            const chunk2 = `data: ${JSON.stringify({
              choices: [{ delta: { content: "from ShardAi!" } }],
            })}\n\n`;
            controller.enqueue(encoder.encode(chunk1));
            controller.enqueue(encoder.encode(chunk2));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
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

    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("open-chat-window-btn")).toBeTruthy();
    });
    expect(screen.queryByTestId("chat-not-ready-notice")).toBeNull();

    // Open AI Chat window
    fireEvent.click(screen.getByTestId("open-chat-window-btn"));
    expect(screen.getByTestId("ai-chat-window")).toBeTruthy();
    expect(screen.getByTestId("chat-window-model").textContent).toContain(
      "Qwen/Qwen2.5-7B-Instruct"
    );
    expect(screen.getByTestId("chat-window-endpoint").textContent).toContain(
      "http://127.0.0.1:8000/v1"
    );

    // Send first message
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Ping model" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "Hello from ShardAi!"
      );
    });

    expect(chatRequests).toHaveLength(1);
    expect(chatRequests[0].url).toBe("http://127.0.0.1:8000/v1/chat/completions");
    expect(chatRequests[0].authHeader).toBe("Bearer sk-shard-ready-secret-123");
    expect(chatRequests[0].body).toEqual({
      model: "Qwen/Qwen2.5-7B-Instruct",
      messages: [{ role: "user", content: "Ping model" }],
      stream: true,
    });

    // Clear conversation
    fireEvent.click(screen.getByTestId("chat-clear-btn"));
    expect(screen.getByTestId("chat-empty-state")).toBeTruthy();

    // Close via Escape
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("ai-chat-window")).toBeNull();
  });

  it("handles non-streaming JSON fallback and HTTP 503 error cleanly inside AI Chat window", async () => {
    let callCount = 0;

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/available-rooms")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.includes("/models/catalog")) {
        return new Response(
          JSON.stringify({
            source: "cloud",
            count: 1,
            models: [
              {
                id: "qwen7b",
                name: "Qwen 2.5 7B",
                repo_id: "Qwen/Qwen2.5-7B-Instruct",
                filename: "qwen2.5-7b.gguf",
                estimated_ram_gb: 5,
                enabled: true,
              },
            ],
          }),
          { status: 200 }
        );
      }
      if (url.includes("/system/memory")) {
        return new Response(
          JSON.stringify({
            total_ram_gb: 16,
            available_ram_gb: 12,
            max_shareable_ram_gb: 10,
          }),
          { status: 200 }
        );
      }
      if (url.includes("/network/status")) {
        return new Response(
          JSON.stringify({
            network_status: "online",
            tracker_reachable: true,
            tracker_url: "https://tracker.example.com",
            lan_peers_count: 0,
            zeroconf_available: true,
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
      if (url.includes("/models/download-stream")) {
        return new Response(
          `data: ${JSON.stringify({
            model_id: "Qwen/Qwen2.5-7B-Instruct",
            status: "completed",
            progress: 100,
            downloaded_bytes: 1000,
            total_bytes: 1000,
            local_path: "/tmp/qwen.gguf",
            message: "Ready",
          })}\n\n`,
          { status: 200, headers: { "Content-Type": "text/event-stream" } }
        );
      }
      if (url.includes("/start-node")) {
        const readyStatus: NodeStatusResponse = {
          state: "RUNNING",
          node_id: "node-ready-json",
          model_id: "Qwen/Qwen2.5-7B-Instruct",
          room_id: "public",
          is_private: false,
          is_unlisted: false,
          role: "primary",
          backend: "llama_cpp",
          n_layers: 32,
          total_layers: 32,
          shard_layer_start: 0,
          shard_layer_end: 31,
          allocated_ram_mb: 6144,
          allocated_vram_mb: 0,
          p2p_listen_port: 49152,
          connected_peers_count: 1,
          peers: [],
          session_token: "sk-shard-json-key",
          api_base_url: "http://127.0.0.1:8000/v1",
          coverage_percent: 100,
          is_swarm_ready: true,
          network_status: "online",
          message: "Swarm ready",
        };
        return new Response(JSON.stringify(readyStatus), { status: 200 });
      }
      if (url.includes("/v1/chat/completions")) {
        callCount += 1;
        if (callCount === 1) {
          return new Response(
            JSON.stringify({
              choices: [{ message: { role: "assistant", content: "JSON reply OK" } }],
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }
        return new Response(
          JSON.stringify({
            error: { message: "Swarm is not ready (RAM coverage < 100%)." },
          }),
          { status: 503, headers: { "Content-Type": "application/json" } }
        );
      }
      return new Response("{}", { status: 200 });
    });

    render(<Dashboard />);
    await waitFor(() => {
      expect(
        (screen.getByTestId("catalog-model-select") as HTMLSelectElement).disabled
      ).toBe(false);
    });

    fireEvent.click(screen.getByTestId("toggle-sharing-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("open-chat-window-btn")).toBeTruthy();
    });

    fireEvent.click(screen.getByTestId("open-chat-window-btn"));

    // 1st call: JSON response fallback
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Test JSON mode" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "JSON reply OK"
      );
    });

    // 2nd call: HTTP 503 error surfaces alert inside chat window
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "Trigger 503" },
    });
    fireEvent.click(screen.getByTestId("chat-send-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-error-banner").textContent).toContain(
        "Swarm is not ready"
      );
    });
  });

  it("ChatWindow displays animated thinking indicator, toggles Stop button with icons, and streams cursor", async () => {
    let sseController: ReadableStreamDefaultController<Uint8Array> | null = null;
    const encoder = new TextEncoder();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/chat/completions")) {
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            sseController = controller;
          },
        });
        return new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        });
      }
      return new Response("{}", { status: 200 });
    });

    render(
      <ChatWindow
        isOpen={true}
        isModelReady={true}
        coveragePercent={100}
        apiBaseUrl="http://127.0.0.1:8000/v1"
        sessionToken="sk-test-token"
        modelId="meta-llama/Llama-3.1-8B-Instruct"
        onClose={vi.fn()}
      />
    );

    // Initial state: Send button has SVG icon and Send text
    const sendBtn = screen.getByTestId("chat-send-btn");
    expect(sendBtn.textContent).toContain("Send");
    expect(sendBtn.querySelector("svg")).toBeTruthy();
    expect(screen.queryByTestId("chat-stop-btn")).toBeNull();

    const input = screen.getByTestId("chat-input") as HTMLTextAreaElement;
    expect(input.disabled).toBe(false);

    // Type a prompt and submit
    fireEvent.change(input, { target: { value: "Hello, model!" } });
    fireEvent.click(sendBtn);

    // During streaming: Send button transforms to Stop button with SVG square icon
    await waitFor(() => {
      const stopBtn = screen.getByTestId("chat-stop-btn");
      expect(stopBtn).toBeTruthy();
      expect(stopBtn.textContent).toContain("Stop");
      expect(stopBtn.querySelector("svg")).toBeTruthy();
    });

    // Textarea is disabled with streaming placeholder and reduced opacity
    expect(input.disabled).toBe(true);
    expect(input.placeholder).toBe("AI is generating a response...");
    expect(input.style.opacity).toBe("0.6");

    // Before tokens arrive, assistant bubble shows thinking indicator with 3 dots
    expect(screen.getByTestId("chat-thinking-indicator")).toBeTruthy();
    expect(screen.getByTestId("chat-thinking-indicator").getAttribute("aria-label")).toBe(
      "AI is thinking"
    );
    const dots = screen
      .getByTestId("chat-thinking-indicator")
      .querySelectorAll(".chat-thinking-dot");
    expect(dots.length).toBe(3);
    expect(screen.queryByTestId("chat-streaming-cursor")).toBeNull();

    // Enqueue first token chunk
    act(() => {
      if (sseController) {
        sseController.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              choices: [{ delta: { content: "Hello " } }],
            })}\n\n`
          )
        );
      }
    });

    // Thinking indicator disappears; streaming cursor appears
    await waitFor(() => {
      expect(screen.queryByTestId("chat-thinking-indicator")).toBeNull();
      expect(screen.getByTestId("chat-streaming-cursor")).toBeTruthy();
      expect(screen.getByTestId("chat-streaming-cursor").textContent).toBe("▍");
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain("Hello ");
    });

    // Enqueue second token chunk
    act(() => {
      if (sseController) {
        sseController.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              choices: [{ delta: { content: "world!" } }],
            })}\n\n`
          )
        );
      }
    });

    await waitFor(() => {
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "Hello world!"
      );
    });

    // Click Stop button to abort generation
    const stopBtn = screen.getByTestId("chat-stop-btn");
    fireEvent.click(stopBtn);

    // Stop button turns back into Send button, cursor disappears, content preserved
    await waitFor(() => {
      expect(screen.queryByTestId("chat-stop-btn")).toBeNull();
      expect(screen.getByTestId("chat-send-btn")).toBeTruthy();
      expect(screen.queryByTestId("chat-streaming-cursor")).toBeNull();
      expect(screen.getByTestId("chat-message-assistant").textContent).toContain(
        "Hello world!"
      );
      expect(input.disabled).toBe(false);
      expect(input.style.opacity).toBe("1");
    });
  });
});
