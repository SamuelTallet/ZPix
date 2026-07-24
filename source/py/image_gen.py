"""Image generation."""

from collections.abc import Callable
from pathlib import Path
from random import randint
from shutil import rmtree

import gradio as gr
import torch

from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.image_pipe import ImagePipeline
from source.py.output_image import OutputImage
from source.py.ref_images import normalize_ref_image
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
        lora_name: Name of loaded LoRA (e.g. "Retro_Anime").
    Returns:
        Tuple of (updated gallery, last image index, output paths, used seed).

    Raises:
        gr.Error
    """
    pipe = image_pipe.instance

    if pipe is None:
        raise gr.Error(
            t("Please wait, a model is being loaded."),
            duration=4,
        )

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

    output_image = OutputImage(image, output_dir, lora_name)
    output_image.embed_settings(model, prompt, used_seed, steps, cfg)
    output_image.save()

    # Output path is recorded for a possible later deletion.
    images_paths[output_image.id] = str(output_image.path)

    if gallery_images is None:
        gallery_images = []

    # Prompt is added as image caption.
    gallery_images.append((output_image.path, prompt))

    return gallery_images, len(gallery_images) - 1, images_paths, used_seed
