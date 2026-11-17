# Building & Releasing ShardAi Node

ShardAi produces three release deliverables using a **Tauri Sidecar** architecture:

1. **Standalone CLI Binary** (`shardai-core-<target>`): Headless OpenAI Gateway & P2P Node built with PyInstaller + Typer.
2. **Windows Desktop Installer** (`.exe` NSIS / `.msi` WiX): Tauri desktop shell with bundled Python sidecar.
3. **macOS Universal Desktop App** (`.dmg`): Tauri `universal-apple-darwin` bundle supporting both **Apple Silicon (`arm64`)** and **Intel (`x86_64`)**.

---

## Prerequisites

- **Python** `>= 3.11` (`core-engine/requirements.txt`, including `pyinstaller>=6.10.0`)
- **Node.js** `>= 20` & **npm**
- **Rust** stable toolchain (`rustup`) + Tauri CLI (`npm install` at repo root)
- On macOS, install both Rust targets for Universal builds:
  ```bash
  rustup target add aarch64-apple-darwin x86_64-apple-darwin
  ```

---

## 1. Building the Python Core (CLI + Tauri Sidecar)

The build scripts compile `core-engine/main.py` as a single executable (`--onefile`) and copy it to:
- `src-tauri/binaries/core-engine-<target-triple>` (for Tauri `bundle.externalBin`)
- `dist/cli/shardai-core-<target-triple>` (standalone CLI distribution)

### Windows
```powershell
.\scripts\build_core.ps1
# Output:
#   src-tauri\binaries\core-engine-x86_64-pc-windows-msvc.exe
#   dist\cli\shardai-core-x86_64-pc-windows-msvc.exe
```

### Linux / Single-Arch macOS
```bash
chmod +x scripts/build_core.sh
./scripts/build_core.sh
```

Running the CLI standalone:
```bash
# Headless node joining public/private room with 4 GB RAM
./dist/cli/shardai-core-x86_64-pc-windows-msvc.exe --mode cli --ram 4

# Tauri sidecar daemon mode (default when no flags passed)
./src-tauri/binaries/core-engine-x86_64-pc-windows-msvc.exe --mode daemon --port 8000
```

---

## 2. macOS Universal Binary (`universal-apple-darwin`) & `lipo`

Tauri's `--target universal-apple-darwin` packages both Intel (`x86_64`) and Apple Silicon (`arm64`) Rust binaries into a single `.app` / `.dmg` bundle.

### Why `lipo` is Important for Python PyInstaller Sidecars

PyInstaller `--target-arch universal2` only succeeds if **every native C extension wheel** installed in Python (`torch`, `lz4`, `pydantic-core`, `cryptography`, etc.) contains fat `universal2` Mach-O slices. Because several ML wheels publish separate `arm64` and `x86_64` wheels instead of `universal2`, the reliable approach is:

1. **Build the Apple Silicon (`arm64`) sidecar** inside an `arm64` Python venv:
   ```bash
   python3 -m venv .venv-arm64
   source .venv-arm64/bin/activate
   pip install -r core-engine/requirements.txt
   PYTHON_BIN="$(pwd)/.venv-arm64/bin/python" ./scripts/build_core.sh aarch64-apple-darwin
   ```

2. **Build the Intel (`x86_64`) sidecar** under Rosetta (`arch -x86_64`) with an `x86_64` Python binary:
   ```bash
   arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv .venv-x64
   source .venv-x64/bin/activate
   arch -x86_64 pip install -r core-engine/requirements.txt
   PYTHON_BIN="$(pwd)/.venv-x64/bin/python" ./scripts/build_core.sh x86_64-apple-darwin
   ```

3. **Merge both single-arch binaries with `lipo` into the universal Tauri sidecar** (or keep both triples alongside it):
   ```bash
   lipo -create \
     src-tauri/binaries/core-engine-x86_64-apple-darwin \
     src-tauri/binaries/core-engine-aarch64-apple-darwin \
     -output src-tauri/binaries/core-engine-universal-apple-darwin

   chmod +x src-tauri/binaries/core-engine-universal-apple-darwin
   lipo -info src-tauri/binaries/core-engine-universal-apple-darwin
   # Architectures in the fat file: ... are: x86_64 arm64
   ```

> **Note:** If every Python wheel in your environment is `universal2` (for example in CI or slim builds without arch-specific wheels), you can directly run `TARGET_ARCH=universal2 ./scripts/build_core.sh universal-apple-darwin`.

4. **Build the Universal `.dmg`**:
   ```bash
   npm --prefix src-ui ci
   npm ci
   npx tauri icon assets/logo.png
   npm run tauri build -- --target universal-apple-darwin --bundles dmg
   ```

The resulting Universal DMG is written to:
`src-tauri/target/universal-apple-darwin/release/bundle/dmg/*.dmg`

---

## 3. Windows Desktop Packaging

```powershell
.\scripts\build_core.ps1
npm --prefix src-ui ci
npm ci
npx tauri icon assets/logo.png
npm run tauri build -- --bundles msi,nsis
```

Outputs:
- NSIS Installer (`.exe`): `src-tauri/target/release/bundle/nsis/*.exe`
- MSI Installer (`.msi`): `src-tauri/target/release/bundle/msi/*.msi`
