"""MOSS-Image inpainting, on top of MOSS-Image image-to-image.

Inpainting here is image-to-image whose latents outside the mask are reset,
after each step, to the source image noised to the next step's level. The
transformer never sees the mask.
"""

from typing import cast

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.modular_pipeline import (
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
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from source.py.moss_blocks import (
    SPATIAL_COMPRESSION,
    MOSSImageDecodeStep,
    MOSSImageDenoiseLoopWrapper,
    MOSSImageLoopAfterDenoiser,
    MOSSImageLoopDenoiser,
    MOSSImageModularPipeline,
    MOSSImagePrepareLatentsStep,
    MOSSImageSetTimestepsStep,
    MOSSImageTextEncoderStep,
)
from source.py.moss_img2img import MOSSImageImg2ImgLatentsStep


class MOSSImageInpaintPrepareMaskStep(ModularPipelineBlocks):
    """Bring the mask image down to the latent cells: 1 for each cell the mask
    touches, to regenerate."""

    model_name = "moss-image"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("mask_image"),
            InputParam("latents", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("mask", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        latents = block_state["latents"]

        mask = VaeImageProcessor(
            vae_scale_factor=SPATIAL_COMPRESSION,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        ).preprocess(
            block_state["mask_image"],
            height=latents.shape[2] * SPATIAL_COMPRESSION,
            width=latents.shape[3] * SPATIAL_COMPRESSION,
        )
        block_state["mask"] = F.max_pool2d(mask, SPATIAL_COMPRESSION).to(latents)

        self.set_block_state(state, block_state)
        return components, state


class MOSSImageLoopAfterDenoiserInpaint(ModularPipelineBlocks):
    """Reset the latents outside the mask to the source image, noised to the
    level of the next step."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("mask", required=True, type_hint=torch.Tensor),
            InputParam("image_latents", required=True, type_hint=torch.Tensor),
            InputParam("initial_noise", required=True, type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[ModularPipeline, BlockState]:
        kept = block_state["image_latents"]

        # Past the last step, latents are clean: so is the source image.
        if i < len(block_state["timesteps"]) - 1:
            kept = components.scheduler.scale_noise(
                kept,
                block_state["timesteps"][i + 1 : i + 2],
                block_state["initial_noise"],
            )

        mask = block_state["mask"]
        block_state["latents"] = (1 - mask) * kept + mask * block_state["latents"]

        return components, block_state


def create_moss_inpaint_blocks() -> SequentialPipelineBlocks:
    """Create the MOSS-Image image-to-image blocks, plus the mask."""
    # The loop keeps this dict as is: it takes instances, not classes.
    denoise = MOSSImageDenoiseLoopWrapper.from_blocks_dict(
        {
            "denoiser": MOSSImageLoopDenoiser(),
            "after_denoiser": MOSSImageLoopAfterDenoiser(),
            "inpaint": MOSSImageLoopAfterDenoiserInpaint(),
        }
    )

    return SequentialPipelineBlocks.from_blocks_dict(
        {
            "text_encoder": MOSSImageTextEncoderStep,
            "prepare_latents": MOSSImagePrepareLatentsStep,
            "set_timesteps": MOSSImageSetTimestepsStep,
            "img2img": MOSSImageImg2ImgLatentsStep,
            "prepare_mask": MOSSImageInpaintPrepareMaskStep,
            "denoise": denoise,
            "decode": MOSSImageDecodeStep,
        }
    )


def create_moss_inpaint_pipe(
    pipe: DiffusionPipeline | ModularPipeline,
) -> ModularPipeline:
    """Create the MOSS-Image inpaint pipeline from the loaded MOSS-Image pipeline.

    It shares the loaded components (LoRA and attention processors included),
    so no weights load twice.
    """
    inpaint_pipe = MOSSImageModularPipeline(blocks=create_moss_inpaint_blocks())
    inpaint_pipe.update_components(**pipe.components)

    return inpaint_pipe
