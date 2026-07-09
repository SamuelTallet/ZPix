#!/usr/bin/env bash
# ZPix packaging script for Linux & macOS.

set -euo pipefail
shopt -s failglob

target=${1:-}

if [ "$target" != "Linux" ] && [ "$target" != "macOS" ]; then
    echo "Usage: $0 Linux or $0 macOS" >&2
    exit 1
fi

include=(
    "assets"
    "data"
    "metadata"
    "source/css"
    source/py/*.py
    "source/app.js"
    "tools/astral/LICENSE.txt"
    "translations"
    "app.py"
    "LICENSE"
    "requirements.txt"
    "start.sh"
)
executables=("start.sh")

if [ "$target" = "macOS" ]; then
    include+=("ZPix.command")
    executables+=("ZPix.command")
fi

appName=$(cat metadata/NAME)
version=$(cat metadata/VERSION)
archive="$appName-v$version-$target.tar.gz"
distDir="dist"

# tar keeps permission bit.
chmod +x "${executables[@]}"

mkdir -p "$distDir"

# Prevent macOS tar from adding AppleDouble
# "._*" metadata files.
export COPYFILE_DISABLE=1

tar -czf "$distDir/$archive" "${include[@]}"

echo "Archive $distDir/$archive created."
