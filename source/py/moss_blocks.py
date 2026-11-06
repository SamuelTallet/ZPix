"""MOSS-Image modular blocks, from OzzyGT/moss_image_custom_blocks.

Copyright 2026 OpenMOSS and The HuggingFace Team, licensed under the Apache
License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Modified: reference
images dropped, components loaded from the model repo without remote code,
inputs ZPix never passes removed.
"""

from typing import cast

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.models import AutoencoderKLFlux2
from diffusers.modular_pipelines.modular_pipeline import (
    BlockState,
    LoopSequentialPipelineBlocks,
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
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2Tokenizer, Qwen3_5TextModel

from source.py.custom_logger import logger
from source.py.moss_transformer import MOSSImageTransformer2DModel

SPATIAL_COMPRESSION = 16
"""VAE downsampling (8) times the latent patch size (2)."""

LATENT_CHANNELS = 128
"""32 VAE channels packed in 2x2 patches."""

GUIDER_SPEC = ComponentSpec(
    "guider",
    ClassifierFreeGuidance,
    config=FrozenDict({"guidance_scale": 3.0}),
    default_creation_method="from_config",
)

IMAGE_PROCESSOR_SPEC = ComponentSpec(
    "image_processor",
    VaeImageProcessor,
    config=FrozenDict({"vae_scale_factor": SPATIAL_COMPRESSION}),
    default_creation_method="from_config",
)


def truncate(tokenizer: Qwen2Tokenizer, input_ids: torch.Tensor, max_length: int):
    """Cut the token ids to `max_length`, warning about the text cut."""
    if input_ids.shape[1] <= max_length:
        return input_ids

    removed_text = tokenizer.decode(input_ids[0, max_length:])
    logger.warning(f"The prompt was truncated to {max_length} tokens: {removed_text}")

    return input_ids[:, :max_length]


class MOSSImageTextEncoderStep(ModularPipelineBlocks):
    """Encode each prompt alone with Qwen3.5: padding would change its
    linear-attention states."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3_5TextModel),
            ComponentSpec("tokenizer", Qwen2Tokenizer),
            GUIDER_SPEC,
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt"),
            InputParam.template("negative_prompt", default=""),
            InputParam.template("max_sequence_length", default=4096),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
            ),
            OutputParam(
                "negative_prompt_embeds",
                type_hint=list[torch.Tensor],
                kwargs_type="denoiser_input_fields",
            ),
        ]

    def encode(
        self, components: ModularPipeline, block_state: BlockState, prompts: list[str]
    ) -> list[torch.Tensor]:
        tokenizer = components.tokenizer
        embeds = []

        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            input_ids = truncate(
                tokenizer, input_ids, block_state["max_sequence_length"]
            )
            input_ids = input_ids.to(components._execution_device)
            output = components.text_encoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            embeds.append(output.last_hidden_state[0])

        return embeds

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        prompts = block_state["prompt"]
        prompts = [prompts] if isinstance(prompts, str) else prompts

        block_state["prompt_embeds"] = self.encode(components, block_state, prompts)
        block_state["negative_prompt_embeds"] = None

        guider = components.guider
        if guider._enabled and guider.num_conditions > 1:
            negative_prompts = block_state["negative_prompt"]
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts] * len(prompts)
            block_state["negative_prompt_embeds"] = self.encode(
                components, block_state, negative_prompts
            )

        self.set_block_state(state, block_state)
        return components, state


class MOSSImagePrepareLatentsStep(ModularPipelineBlocks):
    """Draw the initial noise, in float32: noise cast from bfloat16 would have
    lost its precision."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("transformer", MOSSImageTransformer2DModel)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam.template("generator"),
            InputParam("prompt_embeds", required=True, type_hint=list[torch.Tensor]),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("latents", type_hint=torch.Tensor),
            OutputParam("dtype", type_hint=torch.dtype),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        shape = (
            len(block_state["prompt_embeds"]),
            LATENT_CHANNELS,
            block_state["height"] // SPATIAL_COMPRESSION,
            block_state["width"] // SPATIAL_COMPRESSION,
        )
        block_state["latents"] = randn_tensor(
            shape,
            generator=block_state["generator"],
            device=components._execution_device,
            dtype=torch.float32,
        )
        block_state["dtype"] = components.transformer.dtype

        self.set_block_state(state, block_state)
        return components, state


class MOSSImageSetTimestepsStep(ModularPipelineBlocks):
    """Run the sigmas linearly from 1 down to `shift / (num_train_timesteps +
    shift - 1)`, before the scheduler applies its static shift."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam.template("num_inference_steps", default=28)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        scheduler = components.scheduler

        shift = scheduler.config.shift
        lowest = shift / (scheduler.config.num_train_timesteps + shift - 1)
        sigmas = np.linspace(1.0, lowest, block_state["num_inference_steps"])
        scheduler.set_timesteps(sigmas=sigmas, device=components._execution_device)
        scheduler.set_begin_index(0)
        block_state["timesteps"] = scheduler.timesteps
        block_state["num_inference_steps"] = len(block_state["timesteps"])

        self.set_block_state(state, block_state)
        return components, state


class MOSSImageLoopDenoiser(ModularPipelineBlocks):
    """Run the transformer for each guidance batch, the unconditional one
    skipping the blocks in `uncond_skip_layers`, and combine the predictions."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            GUIDER_SPEC,
            ComponentSpec("transformer", MOSSImageTransformer2DModel),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam("dtype", required=True, type_hint=torch.dtype),
            InputParam("num_inference_steps", required=True, type_hint=int),
            InputParam("uncond_skip_layers", type_hint=list[int], default=[22]),
            InputParam.template("attention_kwargs"),
            InputParam("prompt_embeds", required=True, type_hint=list[torch.Tensor]),
            InputParam("negative_prompt_embeds", type_hint=list[torch.Tensor]),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[ModularPipeline, BlockState]:
        guider = components.guider
        guider.set_state(
            step=i, num_inference_steps=block_state["num_inference_steps"], timestep=t
        )
        guider_state = guider.prepare_inputs_from_block_state(
            block_state,
            {"encoder_hidden_states": ("prompt_embeds", "negative_prompt_embeds")},
        )
        latents = block_state["latents"].to(block_state["dtype"])

        for guider_state_batch in guider_state:
            guider.prepare_models(components.transformer)
            is_uncond = (
                getattr(guider_state_batch, guider._identifier_key) == "pred_uncond"
            )
            model_out = components.transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=[
                    embeds.to(block_state["dtype"])
                    for embeds in guider_state_batch.encoder_hidden_states
                ],
                skip_layers=block_state["uncond_skip_layers"] if is_uncond else None,
                attention_kwargs=block_state["attention_kwargs"],
                return_dict=False,
            )[0]
            # Guidance and the Euler update run in float32.
            guider_state_batch.noise_pred = model_out.float()
            guider.cleanup_models(components.transformer)

        block_state["noise_pred"] = guider(guider_state)[0]
        return components, block_state


class MOSSImageLoopAfterDenoiser(ModularPipelineBlocks):
    """Take the Euler step from the combined prediction."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @torch.no_grad()
    def __call__(
        self,
        components: ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[ModularPipeline, BlockState]:
        block_state["latents"] = components.scheduler.step(
            block_state["noise_pred"], t, block_state["latents"], return_dict=False
        )[0]
        return components, block_state


class MOSSImageDenoiseLoopWrapper(LoopSequentialPipelineBlocks):
    """Denoise the latents over the timesteps, with the blocks it wraps."""

    model_name = "moss-image"

    @property
    def loop_expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def loop_inputs(self) -> list[InputParam]:
        return [
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
            InputParam("num_inference_steps", required=True, type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))

        with self.progress_bar(
            total=block_state["num_inference_steps"]
        ) as progress_bar:
            for i, t in enumerate(block_state["timesteps"]):
                for block in self.sub_blocks.values():
                    components, block_state = block(components, block_state, i=i, t=t)
                progress_bar.update()

        self.set_block_state(state, block_state)
        return components, state


class MOSSImageDecodeStep(ModularPipelineBlocks):
    """Undo the VAE batch-norm normalization, unpack the 2x2 patches, and decode
    with the FLUX.2 VAE."""

    model_name = "moss-image"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLFlux2), IMAGE_PROCESSOR_SPEC]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("latents", required=True, type_hint=torch.Tensor),
            InputParam.template("output_type"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("images")]

    @torch.no_grad()
    def __call__(
        self, components: ModularPipeline, state: PipelineState
    ) -> tuple[ModularPipeline, PipelineState]:
        block_state = cast(BlockState, self.get_block_state(state))
        vae = components.vae

        latents = block_state["latents"].to(vae.dtype)
        bn_std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt()
        latents = latents * bn_std.to(latents).view(1, -1, 1, 1)
        latents = latents + vae.bn.running_mean.to(latents).view(1, -1, 1, 1)

        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(
            batch_size, channels // 4, height * 2, width * 2
        )

        images = vae.decode(latents, return_dict=False)[0].clamp(-1, 1)
        block_state["images"] = components.image_processor.postprocess(
            images, output_type=block_state["output_type"]
        )

        self.set_block_state(state, block_state)
        return components, state


def create_moss_blocks() -> SequentialPipelineBlocks:
    """Create the MOSS-Image text-to-image blocks."""
    # The loop keeps this dict as is: it takes instances, not classes.
    denoise = MOSSImageDenoiseLoopWrapper.from_blocks_dict(
        {
            "denoiser": MOSSImageLoopDenoiser(),
            "after_denoiser": MOSSImageLoopAfterDenoiser(),
        }
    )

    return SequentialPipelineBlocks.from_blocks_dict(
        {
            "text_encoder": MOSSImageTextEncoderStep,
            "prepare_latents": MOSSImagePrepareLatentsStep,
            "set_timesteps": MOSSImageSetTimestepsStep,
            "denoise": denoise,
            "decode": MOSSImageDecodeStep,
        }
    )


def bind_conv_weights(text_encoder: torch.nn.Module) -> None:
    """Hand the weights of each linear attention's convolution to the attention.

    It reads them rather than calling the convolution, which streamed leaf by
    leaf would then never load them: owned by the attention, they load with it.
    """
    for attention in list(text_encoder.modules()):
        conv = attention._modules.pop("conv1d", None)
        if conv is None:
            continue

        for name, parameter in conv.named_parameters():
            attention.register_parameter(f"conv1d_{name}", parameter)
        object.__setattr__(attention, "conv1d", conv)


class MOSSImageModularPipeline(ModularPipeline, LoraBaseMixin):
    """A MOSS-Image pipeline taking LoRAs into its transformer, keys converted
    beforehand."""

    _lora_loadable_modules = ("transformer",)

    def load_lora_weights(self, *args, **kwargs) -> None:
        """Load a LoRA into the transformer, as its `load_lora_adapter` does."""
        self.transformer.load_lora_adapter(*args, prefix="transformer", **kwargs)


def create_moss_pipe(
    model_id: str, blocks: SequentialPipelineBlocks
) -> ModularPipeline:
    """Create a MOSS-Image pipeline from its blocks, each component they expect
    loaded from its subfolder of the model repo."""
    pipe = MOSSImageModularPipeline(blocks=blocks)
    pipe.update_components(
        **{
            name: pipe.get_component_spec(name).load(
                pretrained_model_name_or_path=model_id,
                subfolder=name,
                dtype=torch.bfloat16,
            )
            for name in pipe.pretrained_component_names
        }
    )
    bind_conv_weights(pipe.text_encoder)

    return pipe
