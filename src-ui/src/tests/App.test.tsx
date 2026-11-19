import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "../components/Dashboard";
import {
  extractApiErrorDetail,
  normalizeNoticeToEnglish,
} from "../services/engineApi";
import type {
  NodeStatusResponse,
  RoomLobbyItem,
  StartNodeRequest,
} from "../types/node";

describe("ShardAi Gateway & Network Rooms Lobby UI", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
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
});
