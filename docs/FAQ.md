# ZPix FAQ

## Is it possible to generate images with a custom style or a specific character?

Yes, using LoRA models:

1. Search for a LoRA model on [CivitAI](https://civitai.com/search/models?baseModel=ZImageTurbo&baseModel=ZImageBase&modelType=LORA&modelType=Checkpoint) for example.
Ensure LoRA is compatible with Z-Image Turbo or Z-Image Base.
2. Download the LoRA.
3. Back to ZPix, click on "LoRA" button in sidebar.
4. Select downloaded LoRA file (extension is *.safetensors*).
5. Generate a new image.

Note you don't need to restart ZPix to unload or load a new LoRA.

## LoRA has loaded but has no effect...

Things you can do:

- Return to LoRA source page, authors often give usage tips (e.g. a lower LoRA strength, a trigger word) and prompts examples.
- Generate an image with a different seed.
- Ensure LoRA is compatible with Z-Image Turbo or Z-Image Base.

## Can I use ZPix for commercial purposes?

Yes, in accordance with [GNU General Public License 3.0](https://www.gnu.org/licenses/gpl-3.0.html) and [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Does this application works offline?

Once model is downloaded, yes.

## I got an error without details, how to know more?

1. Close application.
2. Create a file named `DEBUG` next to `ZPix.exe`.
3. Restart application; notice that a console stays open in background.
4. Repeat actions that previously triggered this error.
5. Look at console output.

## How to uninstall this application?

Since it's a no-installer application, close it and just delete its folder.<br>
For a deep uninstall, run `clean.cmd` before deleting application folder.
