param (
    [Parameter(Mandatory = $true)]
    [int]$Port # Local port to run this app on.
)

$ErrorActionPreference = "Stop"

# Is debug mode enabled?
$debug = (Test-Path "DEBUG") -or (Test-Path "DEBUG.txt")
$DebugPreference = if ($debug) { "Continue" } else { "SilentlyContinue" }

$title = (Get-Content metadata\NAME, metadata\VERSION) -join ' '
# Console is hidden by its title once webview is displayed (see console.cpp)
# unless debug mode is enabled:
if ($debug) { $Host.UI.RawUI.WindowTitle = "Debug $title" }
else { $Host.UI.RawUI.WindowTitle = $title }

# Project homepage URL.
$homeUrl = Get-Content "metadata\HOME_URL"

trap {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Please report issue at $homeUrl/issues" -ForegroundColor Cyan
    Write-Host "You can now close this window..." -NoNewline
    $null = Read-Host
    exit 1
}

Write-Host "Detecting GPUs..." -ForegroundColor Blue

. "source\ps\gpu_detection.ps1"

try {
    $gpus = Get-Gpus
    if ($gpus.Count -eq 0) { throw "No GPUs found" }

    Write-Debug "Well detected $($gpus.Count) GPUs:"
    foreach ($gpu in $gpus) {
        Write-Debug "- $($gpu.Name) ($($gpu.Memory / 1GB)GB)"
    }
    $gpu = $gpus | Sort-Object Memory -Descending | Select-Object -First 1
    Write-Host "$($gpu.Name) selected."
}
catch {
    Write-Warning "GPU detection failed, venv may not be optimized."
    $gpu = [Gpu]@{
        Name   = "None"
        Memory = 0
        Vendor = "Unknown"
    }
}

# Source model to load from Hugging Face.
# TODO: This could be dynamic, depending on selected GPU VRAM and system RAM?
$model = "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32"

# Backup model to load from Hugging Face
# if source model fails to load.
$backupModel = "SamuelTallet/Z-Image-Turbo-SDNQ-uint4-svd-r32"

. "source\ps\app_invoking.ps1"

# Path to uv executable distributed with this app.
# So we don't rely on a global uv that maybe uninstalled outside of this app.
$uv = "tools\astral\uv.exe"

if (-not (Test-Path $uv)) {
    throw "uv executable not found at $uv"
}

# Python venv was previously optimized?
if (Test-Path ".venv\optimized") {
    Write-Host "Optimized Python venv found. Skipping install." -ForegroundColor Green
    try {
        Write-Host "Loading model, please wait..." -ForegroundColor Blue
        Invoke-App -Model $model -BackupModel $backupModel -Port $Port -Uv $uv
        exit # to not go to install since app ran successfully if we reach this stage.
    }
    catch {
        Write-Warning "Failed to run app, let's try reinstall."
    }
}

Write-Host "Installing..." -ForegroundColor Blue

. "source\ps\venv_creation.ps1"
. "source\ps\package_utils.ps1"

# Python venv is currently optimized?
$optimized = $false

if ($gpu.Vendor -eq "NVIDIA") {
    New-VirtualEnv -Python "3.13.11" -Uv $uv
    try {
        Write-Host "Trying optimized setup for your NVIDIA GPU..."
        Install-Torch -Version "2.9.1+cu128" -IndexUrl "cu128" -Uv $Uv
        Install-Package -Id "triton-windows" -Version "3.5.1.post24" -Uv $Uv
        # Since it's a pre-release wheel and it's pretty light (around 11MB), I prefer to bundle it with the app.
        Install-Wheel -Source "wheels\sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl" -Uv $Uv
        Install-Package -Id "diffusers" -Version "0.36.0" -Uv $Uv
        Install-Package -Id "peft" -Version "0.18.1" -Uv $Uv
        Install-Package -Id "sdnq" -Version "0.1.3" -Uv $Uv
        Install-Package -Id "gradio" -Version "6.2.0" -Uv $Uv
        $optimized = $true
    }
    catch {
        Write-Warning "NVIDIA setup failed, falling back to default setup."
    }
}

if ($optimized) {
    # We leave a marker in venv for next start to skip installation.
    New-Item -ItemType File -Path ".venv\optimized" -Force | Out-Null
}
else {
    # This marker is removed by `uv venv --clear`, that's consistent.
    New-VirtualEnv -Python "3.13.11" -Uv $uv
    Write-Host "Trying default setup..."
    Install-Torch -Version "2.9.1" -Backend "auto" -Uv $Uv
    Install-Package -Id "diffusers" -Version "0.36.0" -Uv $Uv
    Install-Package -Id "peft" -Version "0.18.1" -Uv $Uv
    Install-Package -Id "sdnq" -Version "0.1.3" -Uv $Uv
    Install-Package -Id "gradio" -Version "6.2.0" -Uv $Uv
}

Write-Host "Installation complete." -ForegroundColor Green

Write-Host "Loading model... We are nearly there!" -ForegroundColor Blue
Invoke-App -Model $model -BackupModel $backupModel -Port $Port -Uv $uv
