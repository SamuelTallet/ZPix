# ZPix packaging script for Windows.
$ErrorActionPreference = "Stop"

$include = @(
    "assets",
    "data",
    "metadata",
    "source\css",
    "source\ps",
    "source\py\*.py",
    "source\app.js",
    "tools\astral\LICENSE.txt",
    "tools\astral\uv.exe",
    "translations",
    "app.py",
    "clean.cmd",
    "LICENSE",
    "requirements.txt",
    "start.ps1",
    "WebView2Loader.dll",
    "ZPix.exe"
)
$appName = Get-Content "metadata\NAME"
$version = Get-Content "metadata\VERSION"
$archive = "$appName-v$version.zip"
$distDir = "dist"

# Copy the listed items into a temp folder to keep their paths in the archive
# (Compress-Archive otherwise flattens nested files to their leaf name).
$stage = "$env:TEMP\ZPix-Package"
Remove-Item $stage -Recurse -Force -ErrorAction Ignore
try {
    foreach ($item in $include) {
        $targetDir = Split-Path (Join-Path $stage $item)
        New-Item $targetDir -ItemType Directory -Force | Out-Null
        Copy-Item $item $targetDir -Recurse
    }

    New-Item $distDir -ItemType Directory -Force | Out-Null
    Compress-Archive -Path "$stage\*" -DestinationPath "$distDir\$archive" -Force
}
finally {
    Remove-Item $stage -Recurse -Force -ErrorAction Ignore
}
