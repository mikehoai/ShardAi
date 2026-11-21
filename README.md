<div align="center">
  <img src="assets/logo.png" alt="ShardAi Logo" width="140" />
  <h1>ShardAi Node</h1>
  <p><strong>Open-Source P2P Distributed LLM Desktop Client &amp; Local OpenAI-Compatible API Gateway</strong></p>

  <p>
    <a href="https://shardai.online"><img src="https://img.shields.io/badge/Website-shardai.online-2563eb?style=flat-square" alt="Official Website" /></a>
    <a href="https://github.com/mikehoai/ShardAi/stargazers"><img src="https://img.shields.io/github/stars/mikehoai/ShardAi?style=flat-square&color=2563eb" alt="Stars" /></a>
    <a href="https://github.com/mikehoai/ShardAi/network/members"><img src="https://img.shields.io/github/forks/mikehoai/ShardAi?style=flat-square" alt="Forks" /></a>
    <a href="https://github.com/mikehoai/ShardAi/issues"><img src="https://img.shields.io/github/issues/mikehoai/ShardAi?style=flat-square" alt="Issues" /></a>
    <a href="https://github.com/mikehoai/ShardAi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/mikehoai/ShardAi?style=flat-square" alt="License" /></a>
  </p>

  <p>
    🌐 <strong>Official Homepage:</strong> <a href="https://shardai.online">https://shardai.online</a>
  </p>
</div>

---

## 📖 Overview

**ShardAi Node** (`shardai-node`) is an open-source, peer-to-peer (P2P) distributed LLM inference desktop client and **Local OpenAI-Compatible API Gateway** (`/v1/chat/completions`). It enables everyday consumer hardware (laptops, desktops, and local workstations) to pool idle RAM and GPU VRAM together to load and run Large Language Models collaboratively.

Instead of requiring a single high-VRAM workstation, ShardAi splits (shards) model layers across participating peers using **Pipeline Parallelism**. Each node loads only its assigned slice of model layers into memory and streams compressed intermediate activation tensors directly to the next peer in the pipeline.

> **Client-Only Open-Source Repository:** This repository (`shardai-node`) contains **100% client-side code** — the native **Tauri v2 Desktop App** (`src-ui/` + `src-tauri/`) and the local **Python AI & P2P Core Engine** (`core-engine/`). It operates across three network modes: **Internet Swarm** (via a configurable Signaling REST API endpoint), **LAN Only** (zero-config local mDNS discovery), and **Offline Solo** (100% local execution).

---

## ✨ Key Features

* **🔌 Local OpenAI-Compatible API Gateway:** Exposes `POST http://127.0.0.1:8000/v1/chat/completions` (streaming SSE & non-streaming) and `GET /v1/models`, allowing drop-in integration with Cursor, Continue, OpenWebUI, LangChain, and the official OpenAI SDKs.
* **💬 Built-in AI Chat Window:** Interactive chat interface directly inside the desktop client once swarm RAM coverage reaches 100%, featuring real-time SSE token streaming, markdown rendering, multi-turn history, stop generation, and error surface alerts.
* **🎛 3-Tab AI Model Selector:** Unified model selection across the main panel and Create Room modal with Online Catalog (~5-8 GB RAM estimates & cloud sync), Custom HuggingFace Repos / Direct URLs, and Local `.gguf` file browsing.
* **🧠 Dynamic Layer Sharding:** Automatically slices GGUF (`llama-cpp-python` with `mmap` fallback) and Hugging Face `safetensors` (`PyTorch`) models based on the RAM/VRAM budget you allocate in the desktop UI.
* **🌐 Hybrid Network Modes & Smart Lobby:**
  * **Internet Swarm:** Connects to public or private rooms via Signaling REST API (`SHARDAI_TRACKER_URL`) with session-scoped room randomization and clean frameless colored status indicators.
  * **LAN Only (`zeroconf` mDNS):** Discovers peers automatically on your local network (`_shardai._tcp.local.`) with zero external internet dependency.
  * **Offline Solo Fallback:** Runs all model layers locally on a single machine when offline or unassisted.
* **⚡ Standard CLI & Lightweight Tauri Shell:**
  * **`shardai` CLI:** Rich cross-platform terminal commands (`create`, `join`, `chat`, `status`, `stop`, `pull`, `models`, `rooms`, `serve`, `network`, `doctor`, `service`, `completion`, `version`) with automated Windows, macOS, and Ubuntu/Linux PATH & `systemd` installers.
  * **Tauri v2 Desktop App:** Lightweight native window with system tray, collapsible sidebar, sun/moon dark theme, hardware telemetry, and automated Python sidecar supervisor (zero zombie processes).
* **🔒 Binary Tensor Streaming & Loopback Security:** Uses `lz4`-compressed binary tensor framing over P2P sockets (never `pickle` or raw JSON tensors) and binds the local control plane strictly to `127.0.0.1` with IPC token authentication (`X-ShardAi-Local-Token`).

---

## 🏗 Client Architecture

`shardai-node` is structured into three decoupled client layers:

```text
+-----------------------------------------------------------------------+
|                     ShardAi Node (Desktop Client)                     |
|                                                                       |
|  +------------------------------+   +------------------------------+  |
|  |  Tauri v2 Shell (UI + Rust)  |   |    Core Engine (Python)      |  |
|  |  - React 18 + TypeScript UI  |-->|  - FastAPI (127.0.0.1:8000)  |  |
|  |  - RAM/VRAM Budget Controls  |   |  - /v1/chat/completions      |  |
|  |  - Rooms Lobby & Telemetry   |   |  - PyTorch / llama.cpp Shards|  |
|  |  - Rust Sidecar Supervisor   |   |  - LAN mDNS & P2P Transport  |  |
|  +------------------------------+   +--------------+---------------+  |
+----------------------------------------------------|------------------+
                                                     |
          +------------------------------------------+------------------------------------------+
          | 1a. Optional Room Discovery              | 1b. Local LAN Discovery                  | 2. Direct P2P Tensor Pipeline
          |     (Configurable Signaling REST API)    |     (Zeroconf mDNS: _shardai._tcp.local) |    (Binary Frame + LZ4 Stream)
          v                                          v                                          v
+-----------------------+                  +-------------------+                      +-------------------+
|   Signaling Tracker   |                  |     LAN Peers     |                      |   Peer Node B     |
| (SHARDAI_TRACKER_URL) |                  |  (Local Subnet)   |                      |  (Layers 16..31)  |
+-----------------------+                  +-------------------+                      +-------------------+
```

1. **Pre-Flight & Room Admission:** When joining a room, `core-engine` verifies room availability, downloads only the required model weights/shards from Hugging Face into `core-engine/cache_models/`, and loads the assigned layer range into RAM/VRAM.
2. **Peer Discovery:** Nodes discover pipeline neighbors either via the configurable Signaling REST API (`POST /announce`, `GET /rooms`, `GET /peers/:room_id/:model`, `GET /config/models`) or locally on LAN via `zeroconf` mDNS.
3. **Distributed Pipeline Inference:** Requests sent to `http://127.0.0.1:8000/v1/chat/completions` traverse the layer pipeline across connected peers using compressed binary tensor streams and return standard OpenAI chat completion payloads.

---

## 🛠 Tech Stack (`shardai-node`)

* **Desktop UI (`src-ui/`):** React 18, TypeScript (strict mode), Vite, Vitest, Testing Library
* **Native Desktop Core (`src-tauri/`):** Rust (Edition 2021), Tauri v2 (`tray-icon`, native dialog plugin, thread-safe Python sidecar process supervisor)
* **AI & P2P Engine (`core-engine/`):** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, PyTorch, `transformers`, `safetensors`, `accelerate`, `llama-cpp-python` (optional GGUF extra), `huggingface-hub`, `typer`
* **Networking & Serialization:** `httpx` (async REST client with exponential backoff), `zeroconf` (LAN mDNS discovery), `lz4` binary tensor compression, raw TCP / HTTP tensor transport

---

## 🚀 Getting Started

### Prerequisites

* **Node.js** `>= 20` & **npm**
* **Rust & Cargo** (latest stable toolchain via `rustup`)
* **Python** `>= 3.11` (with `venv` or `uv`)
* **C++ Build Tools** (optional, required only if compiling `llama-cpp-python` from source with CUDA / Metal / AVX2 acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mikehoai/ShardAi.git
   cd ShardAi
   ```

2. **Set up the Python Core Engine (`core-engine/`):**
   ```bash
   cd core-engine
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux:
   # source .venv/bin/activate

   pip install -r requirements.txt
   cd ..
   ```

3. **Install Desktop & Frontend Dependencies:**
   ```bash
   npm install
   npm --prefix src-ui install
   ```

### Running in Development Mode

1. **Start the Python Core Engine (Daemon Mode on `127.0.0.1:8000`):**
   ```bash
   cd core-engine
   python main.py --mode daemon --host 127.0.0.1 --port 8000
   ```

2. **Launch the Tauri Desktop Client (or Vite UI Dev Server):**
   ```bash
   # Full native Tauri desktop app:
   npm run tauri dev

   # Or frontend UI dev server only (http://localhost:1420):
   npm run dev:ui
   ```

3. **Using the Local OpenAI-Compatible API Gateway:**
   Once your node is active, point any OpenAI-compatible client to `http://127.0.0.1:8000/v1`:
   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-1.5B-Instruct",
       "messages": [{"role": "user", "content": "Hello from ShardAi P2P pipeline!"}],
       "stream": false
     }'
   ```

---

## 💻 Command-Line Interface (`shardai` CLI)

ShardAi includes a developer-friendly command-line interface that allows you to operate nodes, manage rooms, pre-download models, and chat directly from any terminal.

### CLI Installation

Install the `shardai` command onto your system PATH:

* **Windows:**
  ```powershell
  .\scripts\install_cli.ps1
  ```
* **Linux / Ubuntu (headless client, PEP 668 safe, optional `systemd` & completions):**
  ```bash
  chmod +x ./scripts/install_linux.sh
  ./scripts/install_linux.sh --source --shell-completion
  ```
  See [`docs/LINUX.md`](docs/LINUX.md) for `.deb` installation, background daemonization, Docker, and Ubuntu firewall setup.
* **macOS / Generic Unix:**
  ```bash
  chmod +x ./scripts/install_cli.sh
  ./scripts/install_cli.sh
  ```
* **Or install as an editable package inside your Python virtual environment:**
  ```bash
  pip install -e ./core-engine
  ```

### CLI Commands Overview

| Command | Description | Example |
| :--- | :--- | :--- |
| `shardai create` | Host a room & share RAM/model | `shardai create --model llama-3.1-8b --ram 4` |
| `shardai join` | Join an existing room | `shardai join my-room` |
| `shardai serve` | Run local daemon (foreground or `--background`) | `shardai serve --background` |
| `shardai status` | Show node state & masked API key | `shardai status --show-key` |
| `shardai stop` | Stop active node & free memory (`--shutdown-daemon`) | `shardai stop --shutdown-daemon` |
| `shardai rooms` | List available rooms | `shardai rooms` |
| `shardai models` | List catalog models & RAM estimates | `shardai models` |
| `shardai pull` | Pre-download model weights | `shardai pull llama-3.1-8b` |
| `shardai peers` | List direct P2P connections | `shardai peers` |
| `shardai chat` | Interactive REPL or one-shot prompt | `shardai chat "Why is distributed inference fast?"` |
| `shardai network` | Telemetry & host RAM headroom | `shardai network` |
| `shardai doctor` | Inspect Linux/Ubuntu, RAM/cgroup & daemon readiness | `shardai doctor` |
| `shardai service` | Generate or manage Linux `systemd` service unit | `shardai service unit` |
| `shardai completion` | Output Bash or Zsh shell completion script | `shardai completion bash` |
| `shardai version` | Display installed CLI version | `shardai version` |

> **Flag Ergonomics:** All commands support standard POSIX double-dash flags (`--model`, `--room`), short flags (`-m`, `-r`, `-k`), and single-dash flags (`-model`, `-room`, `-ram`). Use `--json` for machine-readable output.

---

## ⚙️ Environment Variables

`shardai-node` never hardcodes private keys or non-configurable endpoints. You can customize client behavior via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SHARDAI_TRACKER_URL` | `https://network.shardai.online` | Base URL of the public/custom Signaling REST API for Internet Swarm discovery |
| `SHARDAI_OFFLINE_MODE` | `false` | Set to `true` (or `1`) to disable Internet Tracker calls and operate strictly in LAN / Offline Solo mode |
| `SHARDAI_LOCAL_TOKEN` | Auto-generated by Tauri | Internal IPC authentication token verified via `X-ShardAi-Local-Token` on local control routes |
| `SHARDAI_P2P_PORT` | `9100` | Local TCP port used for peer-to-peer binary tensor streaming |

---

## 🧪 Running Automated Tests

```bash
# 1. Python Core Engine unit & integration tests (pytest)
cd core-engine
pytest -v

# 2. Frontend React/TypeScript unit tests (Vitest) & typecheck/build
cd ../src-ui
npm run test
npm run build

# 3. Rust Tauri shell unit tests (Cargo)
cd ../src-tauri
cargo test
```

For production packaging instructions (Standalone CLI executable, Windows `.msi`/`.exe` installers, and macOS Universal `.dmg`), see [`docs/BUILD.md`](docs/BUILD.md).

---

## 📂 Project Structure

```text
ShardAi/
├── .github/             # GitHub Actions release workflows & declarative repo metadata
├── assets/              # Project branding and application logos
├── core-engine/         # Python AI sharding & P2P background engine (FastAPI)
│   ├── models/          # Hugging Face downloader, GGUF reader & shard runner
│   ├── p2p/             # Peer discovery, mDNS (zeroconf), LZ4 binary serializer & transport
│   ├── shard_engine/    # CLI entrypoint, memory manager, Pydantic schemas & signaling client
│   ├── tests/           # Pytest unit and integration test suites
│   ├── main.py          # FastAPI control plane, /v1/chat/completions gateway & sidecar entry
│   ├── pyproject.toml   # Python project metadata, [project.scripts], & build config
│   └── requirements.txt # Pinned runtime and test dependencies
├── docs/                # Build, Linux/Ubuntu, and cross-platform packaging documentation
├── packaging/           # Linux systemd units, shell completions, and container assets
├── scripts/             # PyInstaller sidecar, CLI builders, .deb packager & PATH installers
├── src-tauri/           # Rust native shell (windowing, system tray, Python process guard)
├── src-ui/              # React 18 + TypeScript desktop UI dashboard
├── package.json         # Root Tauri CLI & UI workspace scripts
└── README.md            # Client documentation
```

---

## 🤝 Contributing

Contributions to `shardai-node` are welcome!

1. Fork the repository ([https://github.com/mikehoai/ShardAi](https://github.com/mikehoai/ShardAi))
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Run the automated test suites (`pytest`, `npm --prefix src-ui run test`, `cargo test`)
4. Commit your changes using Conventional Commits (`git commit -m "feat(node): add your feature"`)
5. Open a Pull Request

---

## 🔗 Links

* **Official Website:** [https://shardai.online](https://shardai.online)
* **GitHub Repository:** [https://github.com/mikehoai/ShardAi](https://github.com/mikehoai/ShardAi)
* **Issue Tracker:** [https://github.com/mikehoai/ShardAi/issues](https://github.com/mikehoai/ShardAi/issues)

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
