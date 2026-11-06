"""MOSS-Image image-to-image, on top of the MOSS-Image modular blocks.

The denoising loop starts from the source image, noised to the level strength
spares.
"""

from typing import cast

import torch
from diffusers.models import AutoencoderKLFlux2
from diffusers.modular_pipelines.modular_pipeline import (
    AutoPipelineBlocks,
    BlockState,
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.modular_pipelines.qwenimage.before_denoise import get_timesteps
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from source.py.moss_blocks import (
    IMAGE_PROCESSOR_SPEC,
    SPATIAL_COMPRESSION,
    create_moss_blocks,
)


class MOSSImageImg2ImgLatentsStep(ModularPipelineBlocks):
    """Encode the source image at the latents size, skip the first steps, as
    much as strength spares, and noise the source image to the level of the
    first remaining one.

    The noise drawn is kept, so inpainting can noise the source image with it
    again.
    """

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLFlux2),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler),
            IMAGE_PROCESSOR_SPEC,
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image"),
            InputParam.template("strength"),
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
            OutputParam("image_latents", type_hint=torch.Tensor),
            OutputParam("initial_noise", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        vae = components.vae
        scheduler = components.scheduler
        noise = block_state["latents"]

        image = components.image_processor.preprocess(
            block_state["image"],
            height=noise.shape[2] * SPATIAL_COMPRESSION,
            width=noise.shape[3] * SPATIAL_COMPRESSION,
        )
        latents = vae.encode(
            image.to(components._execution_device, vae.dtype)
        ).latent_dist.mode()

        # What the decode step undoes: pack 2x2 patches into channels, then
        # normalize with the VAE batch norm.
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(
            batch_size, channels * 4, height // 2, width // 2
        )
        bn_std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt()
        latents = latents - vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
        block_state["image_latents"] = (
            latents / bn_std.to(latents).view(1, -1, 1, 1)
        ).float()

        block_state["timesteps"], block_state["num_inference_steps"] = get_timesteps(
            scheduler, block_state["num_inference_steps"], block_state["strength"]
        )
        block_state["initial_noise"] = noise
        block_state["latents"] = scheduler.scale_noise(
            block_state["image_latents"], block_state["timesteps"][:1], noise
        )

        self.set_block_state(state, block_state)
        return components, state


class MOSSImageAutoImg2ImgStep(AutoPipelineBlocks):
    """Start from the source image when given, from noise otherwise."""

    model_name = "moss-image"

    def __init__(self) -> None:
        self.block_classes = [MOSSImageImg2ImgLatentsStep]
        self.block_names = ["img2img"]
        self.block_trigger_inputs = ["image"]
        super().__init__()


def create_moss_img2img_blocks() -> SequentialPipelineBlocks:
    """Create the MOSS-Image text-to-image blocks, plus image-to-image."""
    blocks = create_moss_blocks()
    # Once the noise is drawn and the schedule set, before denoising.
    blocks.sub_blocks.insert("img2img", MOSSImageAutoImg2ImgStep(), 3)

    return blocks
