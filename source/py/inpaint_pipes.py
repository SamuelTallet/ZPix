"""Inpaint pipelines, sharing the components of the loaded pipeline."""

from inspect import signature

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import (
    Flux2KleinInpaintPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
    StableDiffusionXLInpaintPipeline,
)
from diffusers.pipelines.z_image.pipeline_z_image_inpaint import ZImageInpaintPipeline

from source.py.anima_inpaint import create_anima_inpaint_pipe
from source.py.custom_logger import logger
from source.py.krea2_inpaint import create_krea2_inpaint_pipe
from source.py.qwen21_inpaint import create_qwen21_inpaint_pipe

INPAINT_PIPELINES: dict[str, tuple[type[DiffusionPipeline], int]] = {
    "Z-Image": (ZImageInpaintPipeline, 0),
    "FLUX": (FluxInpaintPipeline, 0),
    "SDXL": (StableDiffusionXLInpaintPipeline, 0),
    # Appends the source image to the stream, as a reference.
    "FLUX.2": (Flux2KleinInpaintPipeline, 1),
}
"""Inpaint pipeline of each family whose models support inpainting, except
Anima, Krea 2 and Qwen-Image 2.1, which Diffusers ships no inpaint pipeline for,
and the references it appends to the denoised stream."""


def create_inpaint_pipe(
    pipe: DiffusionPipeline | ModularPipeline, family: str
) -> tuple[DiffusionPipeline | ModularPipeline | None, int]:
    """Create the inpaint pipeline of a family from the loaded pipeline.

    It shares the loaded components (LoRA included), so no weights load twice.

    Returns:
        Inpaint pipeline, `None` for a family without one, and the references it
        appends to the denoised stream: some read the source image there, rather
        than only starting from it.
    """
    # Inpaints by itself, its denoiser reading the mask, e.g. FLUX.1 Fill.
    if "mask_image" in signature(type(pipe).__call__).parameters:
        return pipe, 0

    # These only start from the source image.
    if family == "Anima":
        return create_anima_inpaint_pipe(pipe), 0

    if family == "Krea 2":
        return create_krea2_inpaint_pipe(pipe), 0

    # Appends the source image to the stream, as a condition image.
    if family == "Qwen-Image 2.1":
        return create_qwen21_inpaint_pipe(pipe), 1

    if family not in INPAINT_PIPELINES:
        logger.warning(f"Inpainting is not available for {family}.")

        return None, 0

    pipeline_class, references = INPAINT_PIPELINES[family]
    # Only those the pipeline takes: a modular one holds others, e.g. a guider.
    accepted = signature(pipeline_class.__init__).parameters
    components = {
        name: component
        for name, component in pipe.components.items()
        if name in accepted
    }

    return pipeline_class(**components), references
