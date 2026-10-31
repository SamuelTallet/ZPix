"""Image generation."""

from collections.abc import Callable
from inspect import signature
from pathlib import Path
from random import randint
from shutil import rmtree

import gradio as gr
import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from gradio_canvas import CanvasValue, Doc, Layer
from gradio_select_tools import selection_to_mask
from PIL import Image

from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.high_res_fix import (
    HIGH_RES_FIX_STRENGTH,
    high_res_fix,
    high_res_fix_steps,
)
from source.py.image_model import ImageModel
from source.py.image_pipe import ImagePipeline
from source.py.output_image import OutputImage
from source.py.ref_images import normalize_ref_image
from source.py.regional_edits import (
    create_inpaint_pipe,
    get_inpaint_kwargs,
    get_inpaint_size,
    get_inpaint_source,
    get_inpaint_steps,
    keep_outside_region,
)
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
    canvas: CanvasValue,
    lora_name: str | None,
    lasso: list[list[tuple[int, int]]] | None,
    inpaint_steps: int,
    high_res_fix_factor: float,
) -> tuple[CanvasValue, int]:
    """Generate an image and possibly a seed, and add it to canvas.

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
        canvas: Canvas documents, the generated image is added to.
        lora_name: Name of loaded LoRA (e.g. "Retro_Anime").
        lasso: Lasso drawn over displayed document around region to edit, if any.
        inpaint_steps: Steps to regenerate the region to edit with, out of steps.
        high_res_fix_factor: How much to upscale the region to edit, to denoise
            it a second time, or 0 not to.
    Returns:
        Tuple of (updated canvas, used seed).

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

    ref_images = []

    if reference_images and reference_images.get("files"):
        ref_images = [normalize_ref_image(file) for file in reference_images["files"]]

    # Regenerate drawn region of displayed image.
    inpaint_source = get_inpaint_source(model, lasso, canvas.doc)

    # References inpaint pipeline appends to the stream, if any.
    inpaint_references = 0

    if inpaint_source:
        pipe, inpaint_references = create_inpaint_pipe(pipe, model.family)
        width, height = get_inpaint_size(inpaint_source, model.family)
        ref_images = []

    # Strength-based pipelines (e.g. Anima, Z-Image) condition on a single
    # reference image via the batch dimension.
    uses_strength = image_pipe.supports_strength()

    if uses_strength and len(ref_images) >= 2:
        logger.warning("This pipeline doesn't support multiple ref images.")
        ref_images = ref_images[:1]

    # A reference costs the run only where the denoiser reads it: the pipelines
    # above pour it into the latents it starts from and denoise the picture
    # alone, while the others append it to the stream, at a size the image
    # pipeline reads as they encode it. Inpainting takes no reference image,
    # but some inpaint pipelines append the source image to the stream.
    references = (0 if uses_strength else len(ref_images)) + inpaint_references

    image_pipe.fit_to_resolution(
        width,
        height,
        references,
        input_name="inpaint source" if inpaint_source else "reference",
    )

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "generator": torch.manual_seed(used_seed),
    }

    # Asked of the pipeline in hand rather than of the model, which answers for
    # the family and not for what the load ended up building.
    if isinstance(pipe, ModularPipeline):
        if "guider" in pipe.component_names:
            guider_spec = pipe.get_component_spec("guider")
            pipe.update_components(
                guider=guider_spec.create(guidance_scale=max(float(cfg), 1.0))
            )
    else:
        # Standard pipelines take CFG as a call argument, named after what the
        # family scales, and refuse the one they don't declare.
        # Some models declare both, but were distilled to scale `guidance_scale`.
        call_params = signature(type(pipe).__call__).parameters
        cfg_argument = (
            "guidance_scale" if "guidance_scale" in call_params else "true_cfg_scale"
        )
        pipe_kwargs[cfg_argument] = float(cfg)

    if ref_images:
        if uses_strength:
            pipe_kwargs["image"] = ref_images[0]
            pipe_kwargs["strength"] = 1 - ref_image_strength
        else:
            pipe_kwargs["image"] = ref_images

    if inpaint_source:
        mask = selection_to_mask(lasso, inpaint_source.size)  # ty: ignore
        pipe_kwargs.update(
            get_inpaint_kwargs(
                inpaint_source, mask, (width, height), int(inpaint_steps), int(steps)
            )
        )

    with BlockingTask.run(t("Please try again shortly, an image is being generated.")):
        try:
            image = pipe(**pipe_kwargs).images[0]  # ty: ignore
        except UnicodeDecodeError:
            # A corrupted Triton cache can cause an UnicodeDecodeError.
            rmtree(Path.home() / ".triton", ignore_errors=True)
            image_pipe.forget_compiled()
            gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

            gr.Info(t("Regenerating same image..."), duration=8)
            image = pipe(**pipe_kwargs).images[0]  # ty: ignore

        image_pipe.remember_compiled()

        if inpaint_source:
            image = keep_outside_region(inpaint_source, image, lasso)  # ty: ignore

        if inpaint_source and high_res_fix_factor:

            def denoise(region: Image.Image, region_mask: Image.Image) -> Image.Image:
                size = get_inpaint_size(region, model.family)
                image_pipe.fit_to_resolution(
                    *size, inpaint_references, input_name="inpaint source"
                )
                region_steps = high_res_fix_steps(int(steps))
                region_kwargs = get_inpaint_kwargs(
                    region,
                    region_mask,
                    size,
                    get_inpaint_steps(HIGH_RES_FIX_STRENGTH, region_steps),
                    region_steps,
                )
                region_kwargs["num_inference_steps"] = region_steps
                region_image = pipe(**pipe_kwargs | region_kwargs).images[0]  # ty: ignore
                image_pipe.remember_compiled()

                return region_image

            image = high_res_fix(image, mask, high_res_fix_factor, denoise)

    output_image = OutputImage(image, output_dir, lora_name)
    output_image.embed_settings(model, prompt, used_seed, steps, cfg)
    output_image.save()

    # Served from output file, whose prompt is shown as caption.
    canvas.add(
        Doc([Layer(Image.open(output_image.path), source_file=output_image.path)])
    )

    return canvas, used_seed
