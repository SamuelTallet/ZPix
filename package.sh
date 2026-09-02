#!/usr/bin/env bash
# ZPix packaging script for Linux & macOS.

set -euo pipefail
shopt -s failglob

target=${1:-}

if [ "$target" != "deb" ] && [ "$target" != "dmg" ]; then
    echo "Usage: $0 deb or $0 dmg" >&2
    exit 1
fi

include=(
    "assets"
    "data"
    "metadata"
    "source/css"
    source/py/*.py
    "source/app.js"
    "translations"
    ".uv-version"
    "app.py"
    "LICENSE"
    "requirements.txt"
    "start.sh"
)

appName=$(cat metadata/NAME)
version=$(cat metadata/VERSION)
description=$(cat metadata/DESCRIPTION)
homeUrl=$(cat metadata/HOME_URL)
distDir="dist"

# tar & dpkg-deb keep permission bit.
chmod +x start.sh

mkdir -p "$distDir"

if [ "$target" = "dmg" ]; then
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

    archive="${appName}_${version}_AppleSilicon.dmg"
    hdiutil create -volname "$appName $version" -srcfolder "$imageDir" \
        -ov -format UDZO "$distDir/$archive" > /dev/null

    rm -rf "$imageDir"
else
    # The app is shipped as a Debian package.
    if ! command -v dpkg-deb > /dev/null 2>&1; then
        echo "Error: dpkg-deb is required to build a Debian package" >&2
        exit 1
    fi

    # Package names are lowercase and versions use "~" to mark a pre-release,
    # "-" being reserved as the separator of the Debian revision.
    package=$(echo "$appName" | tr '[:upper:]' '[:lower:]')
    debVersion="${version//-/\~}-1"

    # curl or wget bootstraps uv, zenity or kdialog backs the file pickers.
    depends="ca-certificates, curl | wget, zenity | kdialog"

    rootDir="$distDir/deb"
    appDir="$rootDir/opt/$appName"
    iconDir="$rootDir/usr/share/icons/hicolor/256x256/apps"
    rm -rf "$rootDir"
    mkdir -p "$rootDir/DEBIAN" "$appDir" "$iconDir" \
        "$rootDir/usr/share/applications"

    for item in "${include[@]}"; do
        itemDir="$appDir/$(dirname "$item")"
        mkdir -p "$itemDir"
        cp -R "$item" "$itemDir/"
    done

    cp resources/neutral/icon_256.png "$iconDir/$package.png"

    sed -e "s|@NAME@|$appName|g" \
        -e "s|@PACKAGE@|$package|g" \
        -e "s|@DESCRIPTION@|$description|g" \
        resources/debian/app.desktop.in \
        > "$rootDir/usr/share/applications/$package.desktop"

    # Installed-Size is an estimate of the disk usage, in kibibytes.
    installedSize=$(du -sk --exclude=DEBIAN "$rootDir" | cut -f1)

    # "#" delimits below because the dependencies contain a "|" alternative.
    sed -e "s#@PACKAGE@#$package#g" \
        -e "s#@VERSION@#$debVersion#g" \
        -e "s#@INSTALLED_SIZE@#$installedSize#g" \
        -e "s#@DEPENDS@#$depends#g" \
        -e "s#@DESCRIPTION@#$description#g" \
        -e "s#@HOME_URL@#$homeUrl#g" \
        resources/debian/control.in > "$rootDir/DEBIAN/control"

    # md5sums lets dpkg -V and debsums check the installed files.
    (cd "$rootDir" && find * -type f ! -path 'DEBIAN/*' -exec md5sum {} +) \
        > "$rootDir/DEBIAN/md5sums"

    archive="${package}_${debVersion}_all.deb"
    dpkg-deb --build --root-owner-group "$rootDir" "$distDir/$archive" > /dev/null

    rm -rf "$rootDir"
fi

echo "Archive $distDir/$archive created."
