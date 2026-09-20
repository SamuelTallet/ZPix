#!/usr/bin/env bash
# ZPix starting script for Linux & macOS.

set -euo pipefail

# Relative paths below resolve from this script.
cd "$(dirname "$0")"

version=$(cat ./metadata/VERSION)
echo "Starting ZPix $version..."
printf '\033]0;ZPix %s\007' "$version"

os=$(uname -s); arch=$(uname -m)
echo "Detected platform: $os ($arch)"

# Cache directory, can be overridden by XDG_CACHE_HOME.
if [ -n "${XDG_CACHE_HOME:-}" ]; then
    user_cache_dir="$XDG_CACHE_HOME/ZPix"
elif [ "$os" = "Darwin" ]; then
    user_cache_dir="$HOME/Library/Caches/ZPix"
else
    user_cache_dir="$HOME/.cache/ZPix"
fi

# By default, Python writes bytecode next to sources.
# But the app directory must be read-only once packaged.
export PYTHONPYCACHEPREFIX="$user_cache_dir/python"

# Same for the Python environment, otherwise created in the app directory.
# Outside of a project, uv reads VIRTUAL_ENV for its pip and run commands.
export VIRTUAL_ENV="$user_cache_dir/venv"

# Installs a version of uv in a given directory.
# Exits with code 1 if neither curl nor wget are available.
#
# Parameters:
#   $1: uv version to install. Example: "0.11.32"
#   $2: Path to installation directory
#
install_uv() {
    local uv_version=$1
    local uv_dir=$2
    local uv_installer=https://astral.sh/uv/${uv_version}/install.sh

    if command -v curl > /dev/null 2>&1; then
        curl -LsSf "$uv_installer" | env UV_UNMANAGED_INSTALL="$uv_dir" sh
    elif command -v wget > /dev/null 2>&1; then
        wget -qO- "$uv_installer" | env UV_UNMANAGED_INSTALL="$uv_dir" sh
    else
        echo "Error: curl or wget is required to install uv"
        exit 1
    fi
}

# Installs an optional Python package with uv.
# Prints a warning if installation fails.
#
# Parameters:
#   $1: Package or wheel URL. Example: "flash_attn==2.8.3"
#   $2: Path to uv executable
#
install_optional_py() {
    local package=$1
    local uv_exe=$2

    if ! "$uv_exe" pip install "$package"; then
        echo "Warning: Failed to install optional package $package"
    fi
}

# Prints the current locale in RFC 4646 format. Example: "fr-FR"
# Defaults to "en-US" if the locale cannot be determined.
#
get_locale() {
    # POSIX priority order: LC_ALL overrides LC_MESSAGES which overrides LANG.
    local raw=${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}

    # On macOS, apps launched from Finder don't inherit shell locale variables.
    if { [ -z "$raw" ] || [ "$raw" = "C" ] || [ "$raw" = "POSIX" ]; } \
        && [ "$os" = "Darwin" ]; then
        raw=$(defaults read -g AppleLocale 2> /dev/null || true)
    fi

    # Strip encoding and modifier ("fr_FR.UTF-8@euro" -> "fr_FR")
    # then replace "_" with "-" as required by RFC 4646.
    local locale=${raw%%[.@]*}
    locale=${locale//_/-}

    # Default locale ("C" or "POSIX") and unset locale are not language tags.
    case "$locale" in
        C|POSIX|"") locale="en-US" ;;
    esac

    echo "$locale"
}

# Prints the vendors of the GPUs found on this computer.
#
detect_gpus() {
    local device_class vendor_file
    local nvidia=false amd=false intel=false

    for vendor_file in /sys/bus/pci/devices/*/vendor; do
        [ -r "$vendor_file" ] || continue

        # 0x8086 tags every Intel chip, so we keep the classes a GPU uses:
        # display controllers (0x03) and compute accelerators (0x12).
        device_class=$(cat "${vendor_file%/vendor}/class" 2> /dev/null || true)
        case "$device_class" in
            0x03*|0x12*) ;;
            *) continue ;;
        esac

        case "$(cat "$vendor_file")" in
            0x10de) nvidia=true ;;
            0x1002) amd=true ;;
            0x8086) intel=true ;;
        esac
    done

    # A laptop pairing an integrated GPU with a discrete one reports both.
    local vendors=""
    if $nvidia; then vendors="$vendors NVIDIA"; fi
    if $amd; then vendors="$vendors AMD"; fi
    if $intel; then vendors="$vendors INTEL"; fi

    echo "${vendors# }"
}

# Path to uv executable.
# We use the project's uv to avoid conflicts with a possibly installed global uv.
# Bumping .uv-version in the project installs a new uv version in the cache.
uv_dir="$user_cache_dir/uv"
uv_version=$(cat ./.uv-version)
uv_exe="$uv_dir/$uv_version/uv"

# Check uv availability.
if ! "$uv_exe" --version > /dev/null 2>&1; then
    echo "uv $uv_version is not available, let's install it..."

    install_uv "$uv_version" "$uv_dir/$uv_version"

    if ! "$uv_exe" --version > /dev/null 2>&1; then
        echo "uv is still not available, please run start.sh again"
        exit 1
    fi

    # Drop previously installed uv versions.
    for stale_uv in "$uv_dir"/*; do
        [ "$stale_uv" = "$uv_dir/$uv_version" ] || rm -rf "$stale_uv"
    done
fi

# Python environment is maybe broken so it's safer to set it up again every time.
# Wasted time is not so important thanks to uv cache.
"$uv_exe" venv --python-preference only-managed --python 3.14 --clear --force "$VIRTUAL_ENV"

echo "Installing dependencies in $VIRTUAL_ENV..."

"$uv_exe" pip install "numpy==2.5.1"

# GPU detection only matters on Linux, where vendor may vary.
gpu_vendors=""
if [ "$os" = "Linux" ]; then
    gpu_vendors=$(detect_gpus)
    echo "GPU vendor(s) detected: ${gpu_vendors:-none}"
fi

# About packages: on Linux, Triton comes along as a dependency of PyTorch.
case "$gpu_vendors" in

    *NVIDIA*)
        # uv selects the PyTorch backend matching the installed CUDA driver.
        "$uv_exe" pip install "torch==2.13.0" --torch-backend=auto
        "$uv_exe" pip install "torchvision==0.28.0" --torch-backend=auto

        # Each FlashAttention wheel targets one CUDA build; we follow uv's pick.
        cuda=$("$uv_exe" run python -c "import torch; print(torch.version.cuda)")

        case "$cuda" in
            12.6) cuda_tag=cu126 ;;
            13.0) cuda_tag=cu130 ;;
            13.2) cuda_tag=cu132 ;;
            *)    cuda_tag="" ;;
        esac

        case "$arch" in
            x86_64)  flash_tag=v0.9.47 ;;
            aarch64) flash_tag=v0.9.49 ;;
            *)       flash_tag="" ;;
        esac

        if [ -n "$cuda_tag" ] && [ -n "$flash_tag" ]; then
            install_optional_py "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${flash_tag}/flash_attn-2.8.3+${cuda_tag}torch2.13-cp314-cp314-linux_${arch}.whl" "$uv_exe"
        fi
        ;;

    *AMD*)
        rocm_index=https://stable.repo.amd.com/rocm/whl-next/
        # AMD hasn't shipped ROCm 10.0.0 wheels for ARM yet, so a failure is expected on this combo.

        "$uv_exe" pip install --default-index "$rocm_index" "rocm[libraries,device-all]==10.0.0"
        "$uv_exe" pip install --default-index "$rocm_index" "torch[device-all]==2.13.0+rocm10.0.0"
        "$uv_exe" pip install --default-index "$rocm_index" "torchvision[device-all]==0.28.0+rocm10.0.0"
        # TODO Install FlashAttention-2.
        ;;

    *)
        # Apple or unknown vendor on Linux.
        "$uv_exe" pip install "torch==2.13.0" --torch-backend=auto
        "$uv_exe" pip install "torchvision==0.28.0" --torch-backend=auto
        ;;

esac

"$uv_exe" pip install -r requirements.txt

echo "Loading model... We are nearly there!"
"$uv_exe" run app.py --in-browser --locale "$(get_locale)"
