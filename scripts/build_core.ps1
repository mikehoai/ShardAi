<#
.SYNOPSIS
    Build `core-engine/main.py` into a standalone PyInstaller Onefile binary
    and rename it to Tauri's externalBin target-triple convention on Windows.
.EXAMPLE
    .\scripts\build_core.ps1
    .\scripts\build_core.ps1 -TargetTriple x86_64-pc-windows-msvc -PythonExe core-engine\.venv\Scripts\python.exe
#>
[CmdletBinding()]
param(
    [string]$TargetTriple = "",
    [string]$PythonExe = "",
    [switch]$InstallPyInstaller
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CoreDir = Join-Path $RepoRoot "core-engine"
$BinariesDir = Join-Path $RepoRoot "src-tauri\binaries"
$CliDistDir = Join-Path $RepoRoot "dist\cli"
$PyInstallerDistDir = Join-Path $CoreDir "dist"
$PyInstallerWorkDir = Join-Path $CoreDir "build\pyinstaller"

if (-not $PythonExe) {
    $VenvPython = Join-Path $CoreDir ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonExe = $VenvPython
    } else {
        $PythonExe = "python"
    }
}

if (-not $TargetTriple) {
    $Arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) {
        $Arch = $env:PROCESSOR_ARCHITEW6432
    }
    switch -Regex ($Arch) {
        "ARM64" { $TargetTriple = "aarch64-pc-windows-msvc" }
        default { $TargetTriple = "x86_64-pc-windows-msvc" }
    }
}

Write-Host "==> Target Triple : $TargetTriple"
Write-Host "==> Python        : $PythonExe"

if ($InstallPyInstaller) {
    & $PythonExe -m pip install "pyinstaller>=6.10.0"
}

& $PythonExe -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in $PythonExe. Run 'pip install pyinstaller' or pass -InstallPyInstaller."
}

New-Item -ItemType Directory -Force -Path $BinariesDir, $CliDistDir, $PyInstallerWorkDir | Out-Null

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", "core-engine",
    "--distpath", $PyInstallerDistDir,
    "--workpath", $PyInstallerWorkDir,
    "--specpath", $PyInstallerWorkDir,
    "--paths", $CoreDir,
    "--hidden-import", "uvicorn.logging",
    "--hidden-import", "uvicorn.loops",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols",
    "--hidden-import", "uvicorn.protocols.http",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.websockets",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.lifespan",
    "--hidden-import", "uvicorn.lifespan.on",
    "--hidden-import", "typer",
    "--hidden-import", "click",
    "--hidden-import", "zeroconf",
    "--hidden-import", "zeroconf._utils.ipaddress",
    "--hidden-import", "zeroconf._handlers.answers",
    "--collect-submodules", "zeroconf",
    "--collect-submodules", "shard_engine",
    "--collect-submodules", "models",
    "--collect-submodules", "p2p",
    (Join-Path $CoreDir "main.py")
)

Push-Location $CoreDir
try {
    & $PythonExe @PyInstallerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$BuiltBinary = Join-Path $PyInstallerDistDir "core-engine.exe"
if (-not (Test-Path $BuiltBinary)) {
    throw "Expected PyInstaller output not found at $BuiltBinary"
}

$SidecarTarget = Join-Path $BinariesDir "core-engine-$TargetTriple.exe"
$CliTarget = Join-Path $CliDistDir "shardai-core-$TargetTriple.exe"

Copy-Item -Force -Path $BuiltBinary -Destination $SidecarTarget
Copy-Item -Force -Path $BuiltBinary -Destination $CliTarget

Write-Host "==> Tauri Sidecar binary ready : $SidecarTarget"
Write-Host "==> Standalone CLI binary ready: $CliTarget"
