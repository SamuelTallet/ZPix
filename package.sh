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

appName=$(cat metadata/NAME)
version=$(cat metadata/VERSION)
distDir="dist"

# tar keeps permission bit.
chmod +x start.sh

mkdir -p "$distDir"

if [ "$target" = "macOS" ]; then
    # The app is shipped as a bundle, inside a disk image.
    imageDir="$distDir/image"
    bundle="$imageDir/$appName.app"
    rm -rf "$imageDir"
    mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"

    for item in "${include[@]}"; do
        itemDir="$bundle/Contents/Resources/$(dirname "$item")"
        mkdir -p "$itemDir"
        cp -R "$item" "$itemDir/"
    done

    cp resources/macos/launcher.sh "$bundle/Contents/MacOS/$appName"
    chmod +x "$bundle/Contents/MacOS/$appName"

    cp resources/macos/icon.icns "$bundle/Contents/Resources/"

    # Info.plist only accepts period-separated integers, whereas our version
    # may carry a "-beta.n" suffix. Any increasing triplet works as a build
    # version: Launch Services merely compares it to the one it has cached.
    shortVersion=${version%%-*}
    build=$(date -u +%y.%m%d.%H%M)

    sed -e "s/@NAME@/$appName/g" \
        -e "s/@SHORT_VERSION@/$shortVersion/g" \
        -e "s/@BUILD@/$build/g" \
        resources/macos/Info.plist.in > "$bundle/Contents/Info.plist"

    # Legacy but still expected by Launch Services.
    printf 'APPL????' > "$bundle/Contents/PkgInfo"

    ln -s /Applications "$imageDir/Applications"
    # Dragging the app there makes its directory writable for the venv.

    archive="$appName-v$version-$target.dmg"
    hdiutil create -volname "$appName $version" -srcfolder "$imageDir" \
        -ov -format UDZO "$distDir/$archive" > /dev/null

    rm -rf "$imageDir"
else
    # The app is shipped as a tarball.
    # Prevent macOS tar from adding AppleDouble "._*" metadata files.
    export COPYFILE_DISABLE=1

    archive="$appName-v$version-$target.tar.gz"
    tar -czf "$distDir/$archive" "${include[@]}"
fi

echo "Archive $distDir/$archive created."
