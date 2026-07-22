"""Image generation."""

from collections.abc import Callable
from pathlib import Path
from random import randint
from shutil import rmtree
from time import time_ns

import gradio as gr
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.image_pipe import ImagePipeline
from source.py.image_utilities import to_rgb
from source.py.resolutions import parse_resolution


def generate(
    image_pipe: ImagePipeline,
    output_dir: Path,
    t: Callable[[str], str],
    model: ImageModel,
    mm_prompt: dict | None,
    reference_images: dict | None,
    ref_image_strength: float,
    resolution: str,
    seed: int,
    random_seed: bool,
    steps: int,
    cfg: float,
    gallery_images: list[tuple] | None,
    images_paths: dict[str, str],
    lora_name: str | None,
) -> tuple[list[tuple], int, dict[str, str], int]:
    """Generate an image and possibly a seed, and update gallery.

    Args:
        image_pipe: Image pipeline holding the loaded model.
        output_dir: The folder where generated images are saved.
        t: Translation function.
        model: Loaded image model.
        mm_prompt: Multimodal dictionary containing possibly a text prompt.
        reference_images: List of reference images.
        ref_image_strength: How much of the reference image to keep.
        resolution: Resolution string (e.g. "1024x1024").
        seed: Seed value for reproducibility.
        random_seed: Ignore seed argument and generate a seed?
        steps: Number of inference (denoising) steps.
        cfg: Classifier-free guidance scale.
        gallery_images: Existing gallery images to append to.
        images_paths: Dictionary mapping images IDs to output paths.
        lora_name: Name of loaded LoRA (e.g. "Anime_20").
    Returns:
        Tuple of (updated gallery, last image index, output paths, used seed).

    Raises:
        gr.Error
    """
    pipe = image_pipe.instance

    prompt: str = (mm_prompt or {}).get("text", "").strip()

    if model.family == "Anima" and not prompt:
        # Anima models can produce NSFW images even if not asked for.
        raise gr.Error(
            t("Please enter a prompt to generate an image."),
            duration=4,
        )

    width, height = parse_resolution(resolution)
    used_seed = randint(1, 1000000) if random_seed else int(seed)

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "generator": torch.manual_seed(used_seed),
    }

    if model.has_modular_pipeline():
        if "guider" in pipe.component_names:
            guider_spec = pipe.get_component_spec("guider")
            pipe.update_components(
                guider=guider_spec.create(guidance_scale=max(float(cfg), 1.0))
            )
    else:
        # Standard pipelines take CFG as a call argument.
        pipe_kwargs["guidance_scale"] = float(cfg)

    if (
        reference_images
        and reference_images.get("files")
        and "image-to-image" in model.features
    ):
        ref_images_files = reference_images["files"]

        def normalize_ref_image(file):
            """Normalize a reference image file."""
            image = Image.open(file)
            ImageOps.exif_transpose(image, in_place=True)
            return to_rgb(image)

        if image_pipe.supports_strength():
            # Strength-based pipelines (e.g. Anima, Z-Image) condition on a
            # single reference image via the batch dimension.
            if len(ref_images_files) >= 2:
                logger.warning("This pipeline doesn't support multiple ref images.")

            pipe_kwargs["image"] = normalize_ref_image(ref_images_files[0])
            pipe_kwargs["strength"] = 1 - ref_image_strength
        else:
            pipe_kwargs["image"] = [normalize_ref_image(f) for f in ref_images_files]

    with BlockingTask.run(t("Please try again shortly, an image is being generated.")):
        try:
            image = pipe(**pipe_kwargs).images[0]  # ty: ignore
        except UnicodeDecodeError:
            # A corrupted Triton cache can cause an UnicodeDecodeError.
            rmtree(Path.home() / ".triton", ignore_errors=True)
            gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

            gr.Info(t("Regenerating same image..."), duration=8)
            image = pipe(**pipe_kwargs).images[0]  # ty: ignore

    # Prepare metadata to be saved in PNG text chunks.
    image_metadata = PngInfo()
    image_metadata.add_text("model", model.id)
    image_metadata.add_itxt("prompt", prompt)
    image_metadata.add_text("seed", str(used_seed))
    image_metadata.add_text("steps", str(steps))
    image_metadata.add_text("cfg", str(cfg))

    # Milliseconds precision is more than enough to avoid filename collision.
    image_id = str(time_ns() // 1_000_000)
    image_basename = f"image_{image_id}.png"

    # LoRA name (if provided) is included in output path.
    if lora_name:
        image_file = output_dir / lora_name / image_basename
    else:
        image_file = output_dir / image_basename

    # Ensure output directory exists.
    image_file.parent.mkdir(parents=True, exist_ok=True)

    image.save(image_file, pnginfo=image_metadata)

    # Output path is recorded for a possible later deletion.
    images_paths[image_id] = str(image_file)

    if gallery_images is None:
        gallery_images = []

    # Prompt is added as image caption.
    gallery_images.append((image_file, prompt))

    return gallery_images, len(gallery_images) - 1, images_paths, used_seed
