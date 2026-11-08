# ZPix FAQ

## Inpainting doesn't work well or at all...

Things you can do:

- Use a `FLUX.2 [klein]` model, `FLUX.1 Fill [dev]` or a `Qwen-Image 2.1` model, as they usually inpaint better.
- Slightly adjust the selection; for example, enlarge it to include more context.
- Generate an image with a different seed.

## Is it possible to generate images with a custom style or a specific character?

Yes, using LoRA models:

1. Download a LoRA model from [CivitAI](https://civitai.com/models) for example, or create your own. Ensure this LoRA is based on an image model available in ZPix, for example: `SDXL 1.0`, `Flux.1 D`, `ZImageBase`, `ZImageTurbo`, `Flux.2 Klein 4B`, `Flux.2 Klein 9B`, `Anima`, `Krea 2` or `Qwen 2.1`.
2. Back to ZPix, click on LoRA button in sidebar.
3. Select LoRA file (extension is *.safetensors*).
4. Generate a new image.

Note you don't need to restart ZPix to unload or load a new LoRA.

## LoRA has loaded but has no effect...

Things you can do:

- Return to LoRA source page, authors often give usage tips (e.g. a lower LoRA strength, a trigger word) and prompts examples.
- Generate an image with a different seed.

## Where do discarded images go?

If they were generated during the current session, to the *Discarded* subfolder in the output folder.

## How to update to the latest version?

1. Close ZPix.
2. Go to [latest release](https://github.com/SamuelTallet/ZPix/releases/latest).
3. Follow instructions in "How to update" section, at the bottom.

## Can I use ZPix for commercial purposes?

It depends on model used to generate image:

- `SDXL Base 1.0` allows commercial use. See [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
- `FLUX.1 [dev]` and `FLUX.1 Fill [dev]` prohibit commercial use. See [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
- `Z-Image Turbo` and `FLUX.2 [klein] 4B` allow commercial use. See [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- `FLUX.2 [klein] 9B` prohibits commercial use. See [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md).
- `Anima 1.0 Turbo 0.2` and `Anima Base 1.0` prohibit commercial use. See [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md) and [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
- `Krea 2 Turbo` requires a commercial licence once your revenue reaches a certain amount. See [Krea 2 Community License](https://www.krea.ai/krea-2-licensing).
- `Pruna Qwen-Image 2.1` and `Qwen-Image 2.1` may apply some commercial restrictions. See [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).

## Does this application works offline?

Once model is downloaded, yes.

## I got an error without details, how to know more?

### Windows
1. Close application.
2. Create a file named *DEBUG* next to *ZPix.exe*.
3. Restart application; notice that a console stays open in background.
4. Repeat actions that previously triggered this error.
5. Look at console output.

### Linux and macOS
Look at console output.

## How to uninstall this application?

### Windows
Close it, then delete its folder.

### Linux and macOS
Close it, then uninstall it like any other applications.
