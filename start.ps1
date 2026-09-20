# ZPix starting script for Windows.

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
        Invoke-App -Uv $uv
        exit # to not go to install since app ran successfully if we reach this stage.
    }
    catch {
        Write-Warning "Failed to run app, let's try reinstall."
    }
}

Write-Host "Installing..." -ForegroundColor Blue

. "source\ps\venv_creation.ps1"
. "source\ps\package_utils.ps1"
. "source\ps\pytorch_cuda.ps1"

# The optimized marker is removed by `uv venv --clear`, that's consistent.
New-VirtualEnv -Python "3.14" -Uv $uv

# Python venv is currently optimized?
$optimized = $false

if ($gpu.Vendor -eq "NVIDIA") {
    # uv selects the PyTorch backend matching the installed CUDA driver.
    Install-Dependency -Spec "torch==2.13.0" -Backend "auto" -Uv $uv
    Install-Dependency -Spec "torchvision==0.28.0" -Backend "auto" -Uv $uv

    try {
        Write-Host "Trying optimized setup for your NVIDIA GPU..."

        $cuda = Get-CudaVersion -Uv $uv
        if (-not $cuda) {
            throw "PyTorch was installed without CUDA support"
        }

        Install-Dependency -Spec "triton-windows==3.7.1.post27" -Uv $uv

        # Each FlashAttention wheel targets one CUDA build; we follow uv's pick.
        $cudaTag = switch ($cuda) {
            "12.6" { "cu126" }
            "13.0" { "cu130" }
            "13.2" { "cu132" }
            Default { "" }
        }
        if (-not $cudaTag) {
            throw "No FlashAttention wheel mapped to CUDA $cuda"
        }

        Install-Dependency -Spec "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.52/flash_attn-2.8.3+${cudaTag}torch2.13-cp314-cp314-win_amd64.whl" -Uv $uv
        $optimized = $true
    }
    catch {
        Write-Warning "Can't optimize NVIDIA setup: $($_.Exception.Message)"
    }
}
elseif ($gpu.Vendor -eq "AMD") {
    $rocmIndex = "https://stable.repo.amd.com/rocm/whl-next/"

    Install-Dependency -Spec "rocm[libraries,device-all]==10.0.0" -IndexUrl $rocmIndex -Uv $uv
    Install-Dependency -Spec "torch[device-all]==2.13.0+rocm10.0.0" -IndexUrl $rocmIndex -Uv $uv
    Install-Dependency -Spec "torchvision[device-all]==0.28.0+rocm10.0.0" -IndexUrl $rocmIndex -Uv $uv

    try {
        Write-Host "Trying optimized setup for your AMD GPU..."
        Install-Dependency -Spec "triton-windows==3.7.1.post27" -Uv $uv
        # TODO Install FlashAttention-2.
        $optimized = $true
    }
    catch {
        Write-Warning "Can't optimize AMD setup: $($_.Exception.Message)"
    }
}
else {
    # Intel or unknown vendor.
    Install-Dependency -Spec "torch==2.13.0" -Backend "auto" -Uv $uv
    Install-Dependency -Spec "torchvision==0.28.0" -Backend "auto" -Uv $uv
}

Install-Requirements -File "requirements.txt" -Uv $uv

if ($optimized) {
    # We leave a marker in venv for next start to skip installation.
    New-Item -ItemType File -Path ".venv\optimized" -Force | Out-Null
}

Write-Host "Installation complete." -ForegroundColor Green

Write-Host "Loading model... We are nearly there!" -ForegroundColor Blue
Invoke-App -Uv $uv
