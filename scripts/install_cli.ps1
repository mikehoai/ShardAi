<#
.SYNOPSIS
    Install the `shardai` CLI command into the local user PATH on Windows.
.DESCRIPTION
    By default, installs `core-engine` in editable mode into `core-engine\.venv` (or the
    specified `-PythonExe`) and writes a lightweight `shardai.cmd` shim in
    `%LOCALAPPDATA%\ShardAi\bin`. Pass `-UseBinary` to copy a built PyInstaller executable
    (`dist\cli\shardai.exe`) instead.
.EXAMPLE
    .\scripts\install_cli.ps1
    .\scripts\install_cli.ps1 -UseBinary
    .\scripts\install_cli.ps1 -NoPath
#>
[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$InstallDir = "",
    [switch]$UseBinary,
    [switch]$NoPath
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CoreDir = Join-Path $RepoRoot "core-engine"
if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "ShardAi\bin"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if ($UseBinary) {
    $BinaryPath = Join-Path $RepoRoot "dist\cli\shardai.exe"
    if (-not (Test-Path $BinaryPath)) {
        throw "Standalone CLI binary not found at $BinaryPath. Run '.\scripts\build_core.ps1' first."
    }
    $TargetExe = Join-Path $InstallDir "shardai.exe"
    Copy-Item -Force -Path $BinaryPath -Destination $TargetExe
    Write-Host "==> Installed standalone CLI binary: $TargetExe"
} else {
    if (-not $PythonExe) {
        $VenvPython = Join-Path $CoreDir ".venv\Scripts\python.exe"
        if (Test-Path $VenvPython) {
            $PythonExe = $VenvPython
        } else {
            $PythonExe = "python"
        }
    }

    Write-Host "==> Installing shardai CLI package from $CoreDir using $PythonExe..."
    & $PythonExe -m pip install --no-build-isolation --no-deps -e $CoreDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install shardai-core-engine editable package (exit code $LASTEXITCODE)."
    }

    $ResolvedPython = (& $PythonExe -c "import sys; print(sys.executable)").Trim()
    $ShimPath = Join-Path $InstallDir "shardai.cmd"
    $ShimContent = @"
@echo off
"$ResolvedPython" -m shard_engine %*
"@
    Set-Content -Path $ShimPath -Value $ShimContent -Encoding ASCII
    Write-Host "==> Installed CLI shim: $ShimPath"
}

if (-not $NoPath) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $Parts = @()
    if ($UserPath) {
        $Parts = $UserPath.Split(";") | Where-Object { $_ -and $_.Trim() -ne "" }
    }
    $NormalizedTarget = $InstallDir.TrimEnd("\").ToLowerInvariant()
    $AlreadyPresent = $false
    foreach ($Entry in $Parts) {
        if ($Entry.Trim().TrimEnd("\").ToLowerInvariant() -eq $NormalizedTarget) {
            $AlreadyPresent = $true
            break
        }
    }
    if (-not $AlreadyPresent) {
        $NewUserPath = (@($Parts + $InstallDir) -join ";")
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
        if ($env:Path -notlike "*$InstallDir*") {
            $env:Path = "$InstallDir;$env:Path"
        }
        Write-Host "==> Added '$InstallDir' to User PATH. Open a new terminal if needed."
    } else {
        Write-Host "==> '$InstallDir' is already on User PATH."
    }
}

Write-Host ""
Write-Host "ShardAi CLI quickstart:"
Write-Host "  shardai --help"
Write-Host "  shardai create --model llama-3.1-8b --room my-room --ram 4"
Write-Host "  shardai join my-room"
Write-Host "  shardai status"
Write-Host "  shardai chat `"Hello from ShardAi!`""
