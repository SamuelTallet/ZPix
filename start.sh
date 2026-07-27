#!/usr/bin/env bash
# ZPix starting script for Linux & macOS.

set -euo pipefail

# Paths below are relative to this script.
cd "$(dirname "$0")"

# Installs uv in a given directory.
# Exits with code 1 if neither curl nor wget are available.
#
# Parameters:
#   $1: Path to installation directory
#
install_uv_in() {
    local uv_dir=$1

    if command -v curl > /dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$uv_dir" sh
    elif command -v wget > /dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$uv_dir" sh
    else
        echo "Error: curl or wget is required to install uv"
        exit 1
    fi
}

# Installs an optional Python package with uv.
# Prints a warning if installation fails.
#
# Parameters:
#   $1: Package. Example: "triton==3.7.1"
#   $2: Path to uv executable
#
install_optional_py() {
    local package=$1
    local uv_exe=$2

    if ! $uv_exe pip install "$package"; then
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
        && [ "$(uname -s)" = "Darwin" ]; then
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

echo "Starting ZPix $(cat ./metadata/VERSION)..."

os=$(uname -s); arch=$(uname -m)
echo "Detected platform: $os ($arch)"

# Path to uv executable.
# We use a local uv to avoid conflicts with a possibly installed global uv.
local_uv_dir=./tools/astral
uv_exe=${local_uv_dir}/uv

# Check uv availability.
if ! $uv_exe --version > /dev/null 2>&1; then
    echo "local uv is not available, let's install it..."
    install_uv_in $local_uv_dir

    if ! $uv_exe --version > /dev/null 2>&1; then
        echo "uv is still not available, please run again start.sh"
        exit 1
    fi
fi

# Python environment is maybe broken so it's safer to resetup it everytime.
# Wasted time is not so important thanks to uv cache.
$uv_exe venv --python 3.14 --clear

echo "Installing dependencies in .venv..."

$uv_exe pip install "numpy==2.5.1"
$uv_exe pip install "torch==2.13.0" --torch-backend=auto
$uv_exe pip install "torchvision==0.28.0" --torch-backend=auto

if [ "$os" = "Linux" ]; then

    # Detect a NVIDIA GPU heuristically.
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
        is_nvidia=true
    else
        is_nvidia=false
    fi
    echo "NVIDIA GPU detected: $is_nvidia"

    cuda=$($uv_exe run python -c "import torch; print(torch.version.cuda)")
    echo "PyTorch CUDA version installed: $cuda"

    install_optional_py "triton==3.7.1" $uv_exe

    # We explicitely check NVIDIA because ROCm (AMD) emulates CUDA availability.
    if $is_nvidia && [ "$cuda" = "13.0" ]; then
        case "$arch" in
            x86_64)
                install_optional_py "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp314-cp314-linux_x86_64.whl" $uv_exe
                ;;
            aarch64)
                install_optional_py "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.49/flash_attn-2.8.3+cu130torch2.13-cp314-cp314-linux_aarch64.whl" $uv_exe
                ;;
        esac
    fi

fi

$uv_exe pip install -r requirements.txt

echo "Loading model... We are nearly there!"
$uv_exe run app.py --port 26000 --in-browser --locale "$(get_locale)"
