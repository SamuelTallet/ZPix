# Z-Image Turbo for Windows

<img src="https://raw.githubusercontent.com/SamuelTallet/Z-Image-Turbo-Windows/refs/heads/main/docs/screens/Z-Image-Turbo-v1.0.0.webp">

Generate images easily, for free, using solely the power of your GPU.<br>
No watermarks, no limits. Images are computed on your computer.

## Quick start

1. Download and extract [Z-Image-Turbo-v1.0.1.zip](https://github.com/SamuelTallet/Z-Image-Turbo-Windows/releases/download/v1.0.1/Z-Image-Turbo-v1.0.1.zip)
2. Run `Z-Image-Turbo.exe`
    - If SmartScreen pops, click on "More info", "Run anyway"
    - If a DLL is missing, install [Visual C++ Redist](https://aka.ms/vc14/vc_redist.x64.exe) and re-run
3. Write a prompt
4. Click on "Generate Image"
5. Download (export) generated image if you want

## Recommended configuration

- GPU: NVIDIA RTX 30/40/50 series with 8GB VRAM or more
- 32GB RAM (16GB also works but slower)

## Gallery

All images below were generated using Z-Image Turbo for Windows:

<img src="https://raw.githubusercontent.com/SamuelTallet/Z-Image-Turbo-Windows/refs/heads/main/docs/gallery/image0.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/Z-Image-Turbo-Windows/refs/heads/main/docs/gallery/image2.webp">
<img src="https://raw.githubusercontent.com/SamuelTallet/Z-Image-Turbo-Windows/refs/heads/main/docs/gallery/image7.webp">

See more examples [here](https://github.com/SamuelTallet/Z-Image-Turbo-Windows/tree/main/docs/gallery).

## Frequently asked questions

Please get to [FAQ page](https://github.com/SamuelTallet/Z-Image-Turbo-Windows/blob/main/docs/FAQ.md).

## Credits

Quantized model: [Z-Image-Turbo-SDNQ](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0.<br>
Base model: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Tongyi-MAI.

Python packages: [Torch](https://pytorch.org/), [Triton Windows](https://pypi.org/project/triton-windows/), [Diffusers](https://pypi.org/project/diffusers/), [SDNQ](https://pypi.org/project/sdnq/), and [Gradio](https://pypi.org/project/gradio/).<br>
Python tools: [uv](https://github.com/astral-sh/uv) and [Ruff](https://github.com/astral-sh/ruff) by Astral.

C++ libraries: [WebView2](https://learn.microsoft.com/microsoft-edge/webview2/) by Microsoft.<br>
C++ tools: [CMake](https://cmake.org/) by Kitware, and [vcpkg](https://github.com/microsoft/vcpkg) by Microsoft.

Icons: [High voltage](https://github.com/googlefonts/noto-emoji/blob/main/svg/emoji_u26a1.svg) by Google, and [FAQ](https://www.freepik.com/icon/technology_13631866) by Kerismaker.

## Thanks

Thanks to TGS for beta testing this program.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl.html) version 3 or later.

## Copyright

© 2026 Samuel Tallet