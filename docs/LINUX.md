# ShardAi Linux & Ubuntu Headless Client Guide

ShardAi supports Ubuntu (22.04 LTS, 24.04 LTS, WSL2) and other modern Linux distributions with full `shardai` CLI command parity alongside dedicated daemon lifecycle, `systemd`, and environment diagnostic commands.

---

## 1. Quick Installation (Ubuntu / Debian / Linux)

### Option A: Automated Source Installer (PEP 668 Safe)

Ubuntu 24.04+ enforces PEP 668 (`externally-managed-environment`) on system Python. `scripts/install_linux.sh` automatically creates an isolated virtualenv, installs `core-engine`, and adds `~/.local/bin/shardai` to your PATH:

```bash
# Ubuntu / Debian prerequisites
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip curl

# Check host readiness without modifying files
./scripts/install_linux.sh --check

# Install CLI + Bash/Zsh completions + user systemd service
chmod +x ./scripts/install_linux.sh
./scripts/install_linux.sh --source --shell-completion --with-systemd
```

For a system-wide installation under `/usr/local/bin` and `/opt/shardai/venv`:

```bash
sudo ./scripts/install_linux.sh --source --system --shell-completion --with-systemd
```

### Option B: Ubuntu / Debian `.deb` Package

If you built or downloaded `shardai_<version>_amd64.deb`:

```bash
sudo dpkg -i shardai_0.1.0_amd64.deb
```

The `.deb` package installs `/usr/bin/shardai`, `systemd` unit files (user and system), and Bash/Zsh shell completions.

### Option C: Prebuilt Release Tarball or Standalone Binary

```bash
# Install latest release archive with SHA256 verification
./scripts/install_linux.sh --release latest

# Or install a locally built binary from dist/cli/shardai
./scripts/install_linux.sh --binary ./dist/cli/shardai
```

To uninstall later:

```bash
./scripts/install_linux.sh --uninstall
# Add --purge to also remove ~/.shardai cached models and logs
```

---

## 2. Full `shardai` CLI Command Parity

All standard `shardai` subcommands behave identically to Windows and macOS, supporting double-dash (`--model`), short (`-m`), and single-dash (`-model`) flags as well as `--json` output:

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `shardai doctor` | Inspect Ubuntu/WSL, RAM/cgroup, GPU & daemon readiness | `shardai doctor` |
| `shardai create` | Create a room & share RAM/VRAM | `shardai create --model llama-3.1-8b --ram 4` |
| `shardai join` | Join an existing room (pre-flight guard) | `shardai join my-room --ram 4` |
| `shardai serve` | Start local FastAPI daemon (`127.0.0.1:8000`) | `shardai serve --background` |
| `shardai status` | Show node state, layers, peers & masked API key | `shardai status --show-key` |
| `shardai stop` | Stop active node (and optional background daemon) | `shardai stop --shutdown-daemon` |
| `shardai rooms` | List available Internet and LAN rooms | `shardai rooms` |
| `shardai models` | List curated catalog models & RAM estimates | `shardai models` |
| `shardai pull` | Pre-download model weights into local cache | `shardai pull llama-3.1-8b` |
| `shardai peers` | List direct P2P routing peers | `shardai peers` |
| `shardai chat` | Interactive REPL or one-shot OpenAI chat | `shardai chat "Hello from Ubuntu!"` |
| `shardai network` | Check Signaling Tracker & host memory headroom | `shardai network` |
| `shardai service` | Generate or manage Linux `systemd` service unit | `shardai service unit` |
| `shardai completion` | Print Bash or Zsh completion script | `shardai completion bash` |
| `shardai version` | Print installed CLI version | `shardai version` |

---

## 3. Running as a Background Daemon or `systemd` Service

### Detached Background Daemon

```bash
# Start daemon detached in the background (writes PID and log files)
shardai serve --background

# Join or host a room through the running daemon
shardai create --model llama-3.1-8b --room ubuntu-room --ram 4

# Stop the active node and shut down the background daemon cleanly
shardai stop --shutdown-daemon
```

### Hardened `systemd` Service (User or System Scope)

All generated `systemd` units strictly bind the HTTP control plane to loopback (`127.0.0.1`) and enforce clean `SIGTERM` shutdown:

```bash
# Preview the generated unit
shardai service unit --scope user

# Install and start user-level service (~/.config/systemd/user/shardai.service)
shardai service install --scope user
shardai service status --scope user
shardai service logs --scope user

# Or install system-wide (/etc/systemd/system/shardai.service)
sudo shardai service install --scope system
```

---

## 4. Shell Autocompletion (Bash & Zsh)

```bash
# Bash
mkdir -p ~/.local/share/bash-completion/completions
shardai completion bash > ~/.local/share/bash-completion/completions/shardai

# Zsh
mkdir -p ~/.zsh/completions
shardai completion zsh > ~/.zsh/completions/_shardai
```

---

## 5. Docker (Headless Ubuntu 24.04 Container)

Build and run the containerized headless client from the repository root:

```bash
docker build -f packaging/linux/Dockerfile -t shardai-node:linux .
docker run --rm --network host -v ~/.shardai:/home/shardai/.shardai shardai-node:linux doctor
```

Memory allocation inside Docker/VPS containers automatically respects cgroup v1/v2 limits (`memory.max` / `memory.limit_in_bytes`).

---

## 6. Ubuntu LAN Firewall & mDNS Troubleshooting

When running in **LAN Only** mode (`--network lan` or `--offline`), peers discover each other using UDP multicast and connect over P2P TCP sockets:

- Allow direct P2P tensor streaming TCP port (default `49152`) and LAN discovery UDP ports (`49535` and `5353`):
  ```bash
  sudo ufw allow 49152/tcp
  sudo ufw allow 49535/udp
  sudo ufw allow 5353/udp
  ```
- Verify environment health, cgroup RAM caps, GPU acceleration (CUDA/ROCm/CPU), and cache directory writability anytime with:
  ```bash
  shardai doctor --json
  ```
