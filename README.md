<div align="center">
  <img src="assets/logo.png" alt="ShardAi Logo" width="140" />
  <h1>ShardAi</h1>
  <p><strong>Peer-to-Peer Distributed LLM Inference Engine — BitTorrent for AI</strong></p>

  <p>
    <a href="https://github.com/mikehoai/ShardAi/stargazers"><img src="https://img.shields.io/github/stars/mikehoai/ShardAi?style=flat-square&color=2563eb" alt="Stars" /></a>
    <a href="https://github.com/mikehoai/ShardAi/network/members"><img src="https://img.shields.io/github/forks/mikehoai/ShardAi?style=flat-square" alt="Forks" /></a>
    <a href="https://github.com/mikehoai/ShardAi/issues"><img src="https://img.shields.io/github/issues/mikehoai/ShardAi?style=flat-square" alt="Issues" /></a>
    <a href="https://github.com/mikehoai/ShardAi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mikehoai/ShardAi?style=flat-square" alt="License" /></a>
  </p>
</div>

---

## 📖 Overview

**ShardAi** is an open-source, peer-to-peer (P2P) distributed inference engine that allows everyday consumer hardware (low-RAM laptops, desktops, and edge devices) to pool their idle RAM/VRAM together to run massive Large Language Models (LLMs) — much like BitTorrent, but for AI inference.

Instead of requiring a single \$10,000+ GPU rig, ShardAi splits (shards) an AI model across multiple peers using **Pipeline Parallelism**. Each node holds only a few layers of the model and passes intermediate activations (tensors) to the next peer over LAN or the Internet.

> **Note:** This repository (`ShardAi`) contains the open-source client application (Desktop GUI + Core AI/P2P Engine). For transparency and security, all local memory allocation and P2P tensor routing logic are 100% open for community audit.

## ✨ Key Features

* **🧠 Dynamic Model Sharding:** Automatically slices GGUF models (e.g., Llama-3.1-8B, Qwen-2.5) based on the exact RAM/VRAM limit you allocate via the UI slider.
* **🌐 Hybrid P2P Networking:** Seamlessly discovers peers on Local Area Networks (mDNS/UDP) for ultra-low latency or connects globally over WAN via lightweight signaling trackers.
* **⚡ Zero-Bloat Desktop App:** Built with **Tauri (Rust)** instead of Electron, ensuring the client itself consumes minimal RAM (<50MB), leaving maximum memory for AI workloads.
* **📦 Automatic Layer Fetching:** Streams and caches only the specific model weights required for your assigned layers directly from Hugging Face.
* **🔒 Privacy-First & Auditable:** End-to-end serialized binary tensor streaming (`msgpack`/`lz4`). Peers only process mathematical hidden states, never raw user prompts (except entry/exit nodes).

## 🏗 Architecture

ShardAi separates the **Control Plane (Signaling/Tracker)** from the **Data Plane (P2P Tensor Pipeline)**:

```text
+-------------------------------------------------------------------+
|                        ShardAi Desktop Client                     |
|  +-----------------------------+   +---------------------------+  |
|  |   Tauri UI (React + Rust)   |   |   Core Engine (Python)    |  |
|  |  - RAM Allocation Slider    |-->|  - FastAPI Local IPC      |  |
|  |  - Model Selector           |   |  - llama.cpp Sharding     |  |
|  |  - System Tray Daemon       |   |  - Binary Tensor Streamer|  |
|  +-----------------------------+   +-------------+-------------+  |
+--------------------------------------------------|----------------+
                                                   |
                   1. Announce / Discover Peers    | 2. Direct P2P Tensor Pipeline
                      (HTTP/REST Heartbeat)        |    (TCP/WebSockets + LZ4)
                                                   v
                +--------------------+      +------+------+
                |  Signaling Server  |      |   Peer B    |
                | (Cloudflare Edge)  |      | Layers 16-31|
                +--------------------+      +-------------+
```

1. **Discovery (Signaling):** When you click *Start Sharing*, your node registers its available RAM, assigned model, and endpoint with the lightweight Signaling Tracker.
2. **Allocation:** The Core Engine downloads the required model shards from Hugging Face and loads specific layers (e.g., `Layers 0..15`) into memory via `llama.cpp`.
3. **Pipeline Inference:** When an inference request arrives, Peer A computes its layers, serializes the output tensors into compressed binary frames, and streams them directly to Peer B (`Layers 16..31`) to generate the final tokens.

## 🛠 Tech Stack

* **Desktop & Native OS Integration:** [Tauri](https://tauri.app/) (Rust), React, TypeScript, TailwindCSS
* **Core AI & Inference Engine:** Python 3.10+, `llama-cpp-python`, PyTorch, Hugging Face Hub
* **Local IPC & P2P Transport:** FastAPI, `asyncio`, WebSockets / Raw TCP Sockets, `msgpack` + `lz4` compression
* **Signaling & Discovery (Cloud):** Cloudflare Workers, Hono, Cloudflare KV (TTL-based peer registry)

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your development machine:

* **Node.js** (v18+) & `pnpm`
* **Rust & Cargo** (latest stable via `rustup`)
* **Python** (3.10 or 3.11) & `uv` (or `poetry`)
* **C++ Build Tools** (Required for compiling `llama-cpp-python` with hardware acceleration like Metal, CUDA, or AVX2)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mikehoai/ShardAi.git
   cd ShardAi
   ```

2. **Setup the Python Core Engine:**
   ```bash
   cd core-engine
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

3. **Install Frontend & Tauri Dependencies:**
   ```bash
   pnpm install
   ```

### Running in Development Mode

Start the Tauri desktop application (which will automatically spawn the Python background engine):

```bash
pnpm tauri dev
```

1. Select a supported model from the dropdown (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
2. Adjust the **RAM Allocation Slider** (e.g., `4 GB`).
3. Click **Start Sharing** to download shards and join the swarm.

## 📂 Project Structure

```text
ShardAi/
├── .antigravity/        # AI Agent coding rules, review checklists, and test guidelines
├── assets/              # Project branding, logos, and media files
├── core-engine/         # Python AI & P2P background daemon
│   ├── api/             # FastAPI local IPC endpoints (/start-node, /generate)
│   ├── models/          # HuggingFace downloader & llama.cpp layer sharding
│   ├── p2p/             # Binary serialization, compression, and socket routing
│   └── tests/           # Pytest unit and integration tests
├── src-tauri/           # Rust backend (System tray, child process lifecycle)
├── src-ui/              # React + TypeScript frontend dashboard
└── docs/                # Architecture specs and protocol documentation
```

## 🗺 Roadmap

* [x] **Phase 1: Core Architecture & Signaling** — Lightweight Edge Tracker with TTL peer discovery.
* [x] **Phase 2: Desktop Wrapper** — Tauri + React UI communicating with local Python FastAPI daemon.
* [x] **Phase 3: Model Sharding & Pipeline** — `llama.cpp` layer splitting and binary tensor streaming.
* [ ] **Phase 4: Fault Tolerance & Auto-Rerouting** — Automatic pipeline recovery when a mid-chain peer disconnects mid-generation.
* [ ] **Phase 5: End-to-End Encryption (E2EE)** — Cryptographic blinding of activations so edge nodes cannot reconstruct prompt data.
* [ ] **Phase 6: Standalone CLI & Docker Image** — Headless mode for Linux servers and pro-users.

## 🤝 Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project (`https://github.com/mikehoai/ShardAi`)
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Ensure all tests pass (`pytest` for Python, `pnpm test` for UI, `cargo test` for Rust)
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
