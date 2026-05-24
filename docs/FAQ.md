# ZPix FAQ

## Is it possible to generate images with a custom style or a specific character?

Yes, using LoRA models:

1. Download a LoRA model from [CivitAI](https://civitai.com/models) for example, or create your own. Ensure this LoRA is based on an image model available in ZPix, for example: `ZImageBase`, `ZImageTurbo`, or `Flux.2 Klein 4B`.
2. Back to ZPix, click on "LoRA" button in sidebar.
3. Select LoRA file (extension is *.safetensors*).
4. Generate a new image.

Note you don't need to restart ZPix to unload or load a new LoRA.

## LoRA has loaded but has no effect...

Things you can do:

- Return to LoRA source page, authors often give usage tips (e.g. a lower LoRA strength, a trigger word) and prompts examples.
- Generate an image with a different seed.
- Ensure LoRA is compatible with image model currently loaded.

## How can I upgrade to latest version?

1. Close ZPix.
2. Remove folder containing `ZPix.exe`.
3. Download [latest](https://github.com/SamuelTallet/ZPix/releases/latest) `ZPix.zip` and extract it to any location.
4. Run `ZPix.exe` from that location.

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
