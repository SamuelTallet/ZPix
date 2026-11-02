# ZPix

<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/screens/ZPix-v1.1.1.apng">

Generate and edit images easily, for free, using solely the power of your GPU. 
Drop an image, select an area with familiar tools like the lasso, inpaint it.
ZPix aims to be simple yet powerful. See [all features](#features).

## Quick start

Windows:
1. Download and extract [ZPix_1.1.1_Windows.zip](https://github.com/SamuelTallet/ZPix/releases/download/v1.1.1/ZPix_1.1.1_Windows.zip)
2. Run `ZPix.exe`
    - If SmartScreen pops, click on "More info", "Run anyway"
    - If a DLL is missing, install [Visual C++ Redist](https://aka.ms/vc14/vc_redist.x64.exe) and re-run

Debian/Ubuntu:
1. Download and install [zpix_1.1.1-1_all.deb](https://github.com/SamuelTallet/ZPix/releases/download/v1.1.1/zpix_1.1.1-1_all.deb)
2. Run ZPix from app launcher

macOS:
1. Download and open [ZPix_1.1.1_AppleSilicon.dmg](https://github.com/SamuelTallet/ZPix/releases/download/v1.1.1/ZPix_1.1.1_AppleSilicon.dmg)
2. Drag ZPix to Applications
3. Run ZPix from Applications
    - If the app doesn't launch, go to system "Settings" > "Privacy & Security", click on "Open Anyway"

Once automatic setup is done:
- Write a prompt and click on "Generate Image"
- Images are autosaved in your Pictures folder, under ZPix

## Supported hardware

- Recent NVIDIA graphics cards
- Apple Silicon GPUs

## Recommended configuration

- OS: Windows or Linux
- GPU: NVIDIA RTX 30/40/50 series with 8GB VRAM or more
- RAM: 32GB (16GB also works but slower)

## Features

### One-click install

ZPix sets up the Python environment and packages for you.

### Nine models

- FLUX.1 [dev]
- Z-Image Turbo
- FLUX.2 [klein] 4B
- Anima Turbo
- Anima Base
- Krea 2 Turbo
- Qwen-Image 2.1 Turbo
- Pruna Qwen-Image 2.1
- Qwen-Image 2.1

### Lazy download

Models are progressively downloaded from Hugging Face.<br>
Initial model at start, then other models when requested.

### Custom models folder

By default models land in the Hugging Face cache of your user profile.<br>
A folder icon in the settings bar lets you pick another location — handy to
keep them on another drive, or to share a folder you already use with
ComfyUI. Takes effect at next ZPix start.

### Text-to-Image

Create an image from a prompt.

### Image-to-Image

- Create or edit an image based on a prompt and/or reference images.
- Add or drag references from the in-app gallery or anywhere.

### Inpainting

- Drop an image, select an area with tools, then prompt for precise changes.<br>
Tools available: Lasso, Polygonal Lasso, Rectangle, Ellipse.
- Enhance patch quality with the bundled hi-res fix (1.5x, 2x).
- Pixels outside your selection stay untouched.

### Prompt history

- Drop an image made by ZPix in the prompt textarea to retrieve its prompt.<br>
You can also drag the image from the in-app gallery.
- Reuse or copy the prompt of the displayed image with dedicated buttons.

### Custom styles

- Load a LoRA safetensors file on the fly and adjust its strength.
- LoRA trigger words are auto-inserted if found in LoRA metadata.

### Fourteen ratios

1:1, 16:9, 9:16, 4:3, 3:4, 5:4, 4:5, 3:2, 2:3, 16:10, 10:16, 21:9, 9:20, 2:1

### Keyboard shortcuts

Photoshop-like shortcuts for inpainting selection tools, fullscreen, and more.

### Advanced settings

- Adjust steps, and CFG on base models.
- Specify a seed or leave it random.

## Gallery

All images below were generated using ZPix:

<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image0.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image2.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image7.webp">

See more examples [here](https://github.com/SamuelTallet/ZPix/tree/main/docs/gallery).

## Frequently asked questions

Please get to [FAQ page](https://github.com/SamuelTallet/ZPix/blob/main/docs/FAQ.md).

## Credits

Base models:
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) by Black Forest Labs
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs
- [Anima](https://huggingface.co/circlestone-labs/Anima) by CircleStone Labs, built on [NVIDIA Cosmos](https://github.com/nvidia-cosmos)
- [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) by Krea
- [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) by Alibaba
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) by Alibaba

Adapters:
- [Pruna-Qwen-Image-2.1](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1) by Pruna AI

Quantized models:
- [FLUX.1-dev-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/FLUX.1-dev-SDNQ-uint4-svd-r32) by Disty0
- [Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0
- [FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) by Disty0
- [Anima-base-v1.0_sdnq_dynamic_8bit](https://huggingface.co/OzzyGT/Anima-base-v1.0_sdnq_dynamic_8bit) by OzzyGT

Python packages:
- [Torch](https://pytorch.org/)
- [Triton Windows](https://pypi.org/project/triton-windows/)
- [FlashAttention Prebuild Wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
- [Diffusers](https://pypi.org/project/diffusers/)
- [PEFT](https://pypi.org/project/peft/)
- [SDNQ](https://pypi.org/project/sdnq/)
- [PlatformDirs](https://pypi.org/project/platformdirs/)
- [CrossFileDialog](https://pypi.org/project/crossfiledialog/)
- [Gradio](https://pypi.org/project/gradio/)

Python tools: [uv](https://github.com/astral-sh/uv), [ty](https://github.com/astral-sh/ty) and [Ruff](https://github.com/astral-sh/ruff) by Astral

JavaScript libraries:
- [Polygon Clipping](https://github.com/mfogel/polygon-clipping) by Mike Fogel

C++ libraries: [WebView2](https://learn.microsoft.com/microsoft-edge/webview2/) by Microsoft<br>
C++ tools: [CMake](https://cmake.org/) by Kitware, and [vcpkg](https://github.com/microsoft/vcpkg) by Microsoft

Icons:
- [Control](https://www.flaticon.com/free-icon/control_10472397) by Ian Anandara
- [Dice](https://www.flaticon.com/free-icon/dice_1714307) by Juicy Fish
- [Folder](https://www.flaticon.com/free-icon/folder_2577144) by Freepik
- [FAQ](https://www.freepik.com/icon/technology_13631866) by Kerismaker
- [Refresh](https://lucide.dev/icons/refresh-ccw), [Lasso](https://lucide.dev/icons/lasso), [Square Dashed](https://lucide.dev/icons/square-dashed) and [Circle Dashed](https://lucide.dev/icons/circle-dashed) by Lucide

## Thanks

Thanks to M1000, Kellzea, JDD, Design8400, Jskup, TGS and Nomis for beta testing this program.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl.html) version 3 or later.

## Terms of Use

You engage to use this program legally and ethically.

## Developer

Samuel Tallet