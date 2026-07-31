#!/usr/bin/env bash
# ZPix launcher for macOS.
# package.sh copies it to ZPix.app/Contents/MacOS/ZPix.

set -euo pipefail

resources="$(dirname "$0")/../Resources"

if [ ! -w "$resources" ]; then
    osascript -e 'display alert "ZPix" message "Please drag ZPix to your Applications folder, then open it from there."' > /dev/null
    exit 1
fi

exec open -a Terminal "$resources/start.sh"
