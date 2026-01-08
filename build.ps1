$ErrorActionPreference = 'Stop'

# Load VS environment.
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vsWhere)) {
    Write-Host "Visual Studio is required. Get it here: https://aka.ms/vs/download"
    exit 1
}

$vsPath = & $vsWhere -latest -property installationPath
Import-Module "$vsPath\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation

# Install dependencies.
vcpkg install

# Configure and build.
cmake -B build -S .
cmake --build build --config Release
