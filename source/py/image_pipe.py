"""Diffusion pipeline wrapper."""

import gc
from collections.abc import Callable

import gradio as gr
import torch
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module
from diffusers.modular_pipelines.components_manager import custom_offload_with_hook
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import get_device
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.krea2_flash import install_krea2_flash_attn
from source.py.offload_strat import FreeMemoryOffloadStrategy

MEMORY_RESERVE_MARGIN = 3 * 1024**3
"""GPU memory kept free, in bytes, for the activations of the running component."""

VAE_MEGAPIXELS_PER_GB = 0.23
"""Megapixels a VAE can handle in one pass, per GB of GPU memory.

Its peak is about 1.75GB per megapixel: 3.7GB at 1920x1088, 7.2GB at 2048x2048.
Measured on the Z-Image VAE, whose decoder holds 128 channels at full resolution,
as the FLUX.2 one does; the Qwen-Image VAE of Anima and Krea 2 has another
architecture and may peak elsewhere.
"""


def get_total_memory() -> int | None:
    """Get the GPU memory, in bytes, or `None` if it can't be measured."""
    if torch.backends.mps.is_available():
        return getattr(torch.mps, "recommended_max_memory", lambda: None)()

    if not (torch.cuda.is_available() or torch.xpu.is_available()):
        return None

    device = torch.device(get_device())

    if device.index is None:
        device = torch.device(f"{device.type}:0")

    device_module = getattr(torch, device.type, torch.cuda)

    return device_module.mem_get_info(device.index)[1]


class ImagePipeline:
    """An image pipeline."""

    def __init__(self):
        self.instance: ModularPipeline | DiffusionPipeline | None = None
        """Loaded pipeline, `None` during a swap."""

    def load(self, model: ImageModel) -> ImageModel:
        """Load an image model pipeline."""

        def create_pipe(model_id: str) -> DiffusionPipeline | ModularPipeline:
            """Create a standard pipeline or a modular one."""
            if not model.has_modular_pipeline():
                return DiffusionPipeline.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                )

            modular_pipeline = ModularPipeline.from_pretrained(model_id)
            modular_pipeline.load_components(dtype=torch.bfloat16)

            return modular_pipeline

        try:
            self.instance = create_pipe(model.id)
        except Exception:
            if model.backup_id:
                logger.warning(
                    f"Can't load {model.id}, falling back to {model.backup_id}."
                )
                self.instance = create_pipe(model.backup_id)
            else:
                raise

        # On NVIDIA, AMD & Intel ARC GPUs:
        if triton_is_available and (
            torch.cuda.is_available() or torch.xpu.is_available()
        ):
            for component_name, component in self.instance.components.items():
                quantization_config = getattr(component, "quantization_config", None)
                quant_method = getattr(quantization_config, "quant_method", None)

                if quant_method == "sdnq":
                    apply_sdnq_options_to_model(component, use_quantized_matmul=True)
                    logger.info(f"SDNQ Quantized MatMul enabled for {component_name}.")

            if model.family in ("Z-Image", "FLUX", "FLUX.2"):
                try:
                    self.instance.transformer.set_attention_backend("flash")
                except Exception as e:  # noqa: BLE001
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available: {e}")
            elif model.family == "Krea 2":
                try:
                    install_krea2_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available for Krea 2: {e}")
            else:
                self.instance.transformer.set_attention_backend("native")

        try:
            self.instance.vae.to(memory_format=torch.channels_last)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Can't apply memory format optimization: {e}")

        # On NVIDIA, AMD & Intel ARC GPUs:
        if torch.cuda.is_available() or torch.xpu.is_available():
            self.offload_to_cpu()

        # On Mac GPUs:
        elif torch.backends.mps.is_available():
            self.instance.to("mps")

            # To prevent swap and performance degradation...
            if hasattr(self.instance, "enable_attention_slicing"):
                self.instance.enable_attention_slicing()

        return model

    def tile_vae_if_needed(self, width: int, height: int) -> None:
        """Tile the VAE work only for the pictures this GPU can't handle in one pass.

        The VAE encodes and decodes the picture as a whole, so its peak memory
        grows with the resolution and, past a point, it alone fills the GPU. Tiling
        bounds that peak but can leave faint seams, so it's a trade only worth
        making when the picture wouldn't go through otherwise.

        Args:
            width: Width of the picture to generate, in pixels.
            height: Height of the picture to generate, in pixels.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None:
            return

        total_memory = get_total_memory()

        # An unmeasurable GPU gets the safe path rather than an optimistic one.
        needs_tiling = total_memory is None or width * height / 1e6 > (
            VAE_MEGAPIXELS_PER_GB * total_memory / 1024**3
        )

        toggle = getattr(
            vae, "enable_tiling" if needs_tiling else "disable_tiling", None
        )

        if toggle is not None:
            toggle()

    def offload_to_cpu(self) -> None:
        """Offload the pipeline weights to CPU, keeping on GPU what fits."""
        if self.instance is None:
            return

        if isinstance(self.instance, DiffusionPipeline):
            self.instance.enable_sequential_cpu_offload()
            return

        # A modular pipeline has no offload helper: replicate the auto CPU offload
        # of its components manager, which can't be used as is because it hooks
        # every component, including those too large to stay on the GPU.
        device = torch.device(get_device())

        if device.index is None:
            device = torch.device(f"{device.type}:0")

        device_module = getattr(torch, device.type, torch.cuda)
        total_memory = device_module.mem_get_info(device.index)[1]

        offload_strategy = FreeMemoryOffloadStrategy(
            memory_reserve_margin=MEMORY_RESERVE_MARGIN
        )

        hooks = []

        for name, component in self.instance.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            footprint = getattr(component, "get_memory_footprint", None)
            fits_on_gpu = (
                footprint is not None
                and footprint() + MEMORY_RESERVE_MARGIN <= total_memory
            )

            # Moving such a component as a whole would saturate the GPU: stream its
            # weights one submodule at a time, and never elect it for eviction.
            if not fits_on_gpu:
                cpu_offload(
                    component,
                    device,
                    offload_buffers=len(component._parameters) > 0,
                )
                logger.warning(
                    f"{name} ({footprint() / 1024**3:.1f}GB) is too large "
                    "for this GPU: streaming it."
                    if footprint
                    else f"Can't size {name}: streaming it."
                )
                continue

            hooks.append(
                custom_offload_with_hook(
                    name, component, device, offload_strategy=offload_strategy
                )
            )

        # Let each component evict its siblings when the GPU runs short.
        for hook in hooks:
            for other_hook in hooks:
                if other_hook is not hook:
                    hook.add_other_hook(other_hook)

    def remove_hooks(self) -> None:
        """Break the reference cycles left by CPU offload."""
        if self.instance is None:
            return

        for component in self.instance.components.values():
            if isinstance(component, torch.nn.Module):
                remove_hook_from_module(component, recurse=True)

    def swap(self, model: ImageModel, t: Callable[[str], str]) -> ImageModel:
        """Swap an image model pipeline, blocking other critical tasks.

        Args:
            model: The image model to load.
            t: Translation function.
        """
        with BlockingTask.run(t("Please wait, a model is being loaded.")):
            self.remove_hooks()

            # Drop the old pipeline before collecting, otherwise it stays alive
            # and its VRAM makes the next auto CPU offload overly aggressive.
            self.instance = None
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()

            return self.load(model)

    def supports_strength(self) -> bool:
        """Does this pipeline accept a `strength` argument?

        Returns:
            `True` for strength-based image-to-image pipelines (e.g. Anima, Z-Image),
            `False` for conditioning-based edit models (e.g. FLUX.2 [klein]),
            for text-to-image only pipelines, and during a swap.
        """
        blocks = getattr(self.instance, "blocks", None)
        return blocks is not None and "strength" in blocks.input_names

    @staticmethod
    def warn_if_not_optimizable(t: Callable[[str], str]) -> None:
        """Warn the user if the pipeline can't be optimized.

        Args:
            t: Translation function.
        """
        if torch.backends.mps.is_available():
            return  # The checks below do not apply to Mac.

        try:
            from flash_attn import flash_attn_func  # noqa: F401

            flash_is_available = True
        except Exception:  # noqa: BLE001
            flash_is_available = False

        if not triton_is_available or not flash_is_available:
            gr.Warning(
                t(
                    "Image generation may be slow because the diffusion pipeline can't be optimized."
                )
                + "<br>"
                + t(
                    "Try upgrading your graphics card drivers, then reboot your PC and restart ZPix."
                ),
                duration=None,  # Until user closes it.
            )
