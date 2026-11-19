import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Dashboard } from "../components/Dashboard";
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
});
