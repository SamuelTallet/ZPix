# ZPix

<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/screens/ZPix-v1.0.5.webp">

Generate and edit images easily, for free, using solely the power of your GPU. 
Hotswap LoRAs. Drag reference images directly from output gallery, which is always visible. Everything is made for your comfort. See [all features](#features).

## Quick start

1. Download and extract [ZPix-v1.0.5.zip](https://github.com/SamuelTallet/ZPix/releases/download/v1.0.5/ZPix-v1.0.5.zip)
2. Run `ZPix.exe`
    - If SmartScreen pops, click on "More info", "Run anyway"
    - If a DLL is missing, install [Visual C++ Redist](https://aka.ms/vc14/vc_redist.x64.exe) and re-run
3. Write a prompt
4. Click on "Generate Image"
5. Images are autosaved in your Pictures \ ZPix

## Recommended configuration

- GPU: NVIDIA RTX 30/40/50 series with 8GB VRAM or more
- 32GB RAM (16GB also works but slower)

## Features

### One-click install

ZPix setups Python environment and packages for you.

### Lazy download

Models are progressively¹ downloaded from Hugging Face.

### Text-to-Image

Create image from prompt.

### Image-to-Image

- Add or drag reference images from output gallery or anywhere.
- Edit image based on prompt and optional reference images.

### Prompts history

- Extract prompt from image², dragged from output gallery or anywhere.
- Reuse or copy used prompt in one click with dedicated buttons.

### Custom styles

- Load LoRA safetensors on-the-fly; change LoRA strength.
- LoRA trigger words are auto-inserted, if found in LoRA metadata.

### Eleven ratios

1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 16:10, 10:16, 21:9, 9:20

### Seed control

Specify a seed value for reproducibility or leave it random.

#### Remarks

1. Initial model at start, then other models when requested.
2. If image contains prompt in its metadata.

## Gallery

All images below were generated using ZPix:

<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image0.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image2.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/ZPix/refs/heads/main/docs/gallery/image7.webp">

See more examples [here](https://github.com/SamuelTallet/ZPix/tree/main/docs/gallery).

## Frequently asked questions

Please get to [FAQ page](https://github.com/SamuelTallet/ZPix/blob/main/docs/FAQ.md).

## Credits

Quantized models:
- [Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0
- [FLUX.2-klein-4B-SDNQ-4bit-dynamic](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) by Disty0

Base models:
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs

Python packages:
- [Torch](https://pytorch.org/)
- [Triton Windows](https://pypi.org/project/triton-windows/)
- [FlashAttention Prebuild Wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
- [Diffusers](https://pypi.org/project/diffusers/)
- [PEFT](https://pypi.org/project/peft/)
- [SDNQ](https://pypi.org/project/sdnq/)
- [PlatformDirs](https://pypi.org/project/platformdirs/)
- [CrossFileDialog](https://pypi.org/project/crossfiledialog) and [Python for Win32](https://pypi.org/project/pywin32)
- [Gradio](https://pypi.org/project/gradio/)

Python tools: [uv](https://github.com/astral-sh/uv), [ty](https://github.com/astral-sh/ty) and [Ruff](https://github.com/astral-sh/ruff) by Astral.

C++ libraries: [WebView2](https://learn.microsoft.com/microsoft-edge/webview2/) by Microsoft.<br>
C++ tools: [CMake](https://cmake.org/) by Kitware, and [vcpkg](https://github.com/microsoft/vcpkg) by Microsoft.

Icons: [High voltage](https://github.com/googlefonts/noto-emoji/blob/main/svg/emoji_u26a1.svg) by Google, [Dice](https://www.flaticon.com/free-icon/dice_1714307) by Juicy Fish, [Folder](https://www.flaticon.com/free-icon/folder_2577144) by Freepik, [FAQ](https://www.freepik.com/icon/technology_13631866) by Kerismaker, and [Refresh](https://lucide.dev/icons/refresh-ccw) by Lucide.

Reference image: [A Woman Posing in a Field of Yellow Flowers](https://www.pexels.com/photo/a-woman-posing-in-a-field-of-yellow-flowers-16465981/) by Josh Hild.

## Thanks

Thanks to M1000, TGS and Nomis for beta testing this program.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl.html) version 3 or later.

## Terms of Use

You engage to use this program legally and ethically.

## Developer

Samuel Tallet