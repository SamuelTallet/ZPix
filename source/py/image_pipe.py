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

from source.py.anima_flash import install_anima_flash_attn
from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.krea2_flash import install_krea2_flash_attn
from source.py.offload_strat import FreeMemoryOffloadStrategy

MEMORY_RESERVE_RATIO = 0.35
"""Share of the GPU kept free for the activations of the running component."""

MIN_MEMORY_RESERVE = int(2.5 * 1024**3)
"""Smallest reserve, in bytes: what a tiled VAE decode peaks at, with a margin."""

MAX_MEMORY_RESERVE = 4 * 1024**3
"""Largest reserve, in bytes: past that, a roomy GPU would evict for nothing."""

BYTES_PER_MEGAPIXEL = int(1.75 * 1024**3)
"""What one megapixel costs a VAE pass, in bytes.

Measured on a decoder holding 128 channels at full resolution. Other
architectures may peak elsewhere.
"""


def get_memory_reserve(total_memory: int) -> int:
    """Get the GPU memory to keep free, in bytes, for the component running.

    A flat reserve doesn't travel: comfortable on a large GPU, it eats half of a
    small one and leaves the transformer streamed submodule by submodule at every
    step. A share of the card puts that trade-off at the same place everywhere.

    Args:
        total_memory: The GPU memory, in bytes.
    """
    reserve = int(MEMORY_RESERVE_RATIO * total_memory)

    return min(max(reserve, MIN_MEMORY_RESERVE), MAX_MEMORY_RESERVE)


def get_execution_device() -> torch.device:
    """Get the GPU the pipeline runs on, index included."""
    device = torch.device(get_device())

    if device.index is None:
        device = torch.device(f"{device.type}:0")

    return device


def get_memory_info() -> tuple[int, int] | None:
    """Get the GPU memory free and total, in bytes, or `None` if unmeasurable.

    Only the free figure nets out what the desktop and the other applications
    hold; a budget read off the total would promise memory this process never gets.
    """
    if torch.backends.mps.is_available():
        # Mac memory is shared with the CPU: its budget stands for both figures.
        budget = getattr(torch.mps, "recommended_max_memory", lambda: None)()

        return (budget, budget) if budget else None

    if not (torch.cuda.is_available() or torch.xpu.is_available()):
        return None

    device = get_execution_device()
    device_module = getattr(torch, device.type, torch.cuda)

    return device_module.mem_get_info(device.index)


class ImagePipeline:
    """An image pipeline."""

    def __init__(self):
        self.instance: ModularPipeline | DiffusionPipeline | None = None
        """Loaded pipeline, `None` during a swap."""

        self.memory_reserve = 0
        """GPU memory the offload strategy keeps free, in bytes; 0 without one."""

        self.memory_budget: int | None = None
        """GPU memory a single VAE pass can count on, in bytes; `None` if unknown."""

        self.streams_weights = False
        """Are the weights streamed submodule by submodule, leaving the GPU free?"""

        self.offload_strategy: FreeMemoryOffloadStrategy | None = None
        """Strategy evicting the components, `None` when none is in play."""

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

            # The backend is picked process-wide, not per pipeline, so a swap
            # would otherwise inherit whatever the family before it chose.
            self.instance.transformer.set_attention_backend("native")

            if model.family in ("Z-Image", "FLUX", "FLUX.2"):
                try:
                    self.instance.transformer.set_attention_backend("flash")
                except Exception as e:  # noqa: BLE001
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available: {e}")

            # These two carry attention the backend refuses, Anima because its
            # text conditioner masks it, Krea 2 because of its grouped queries.
            # Only their transformer takes the kernel, one processor at a time.
            elif model.family == "Anima":
                try:
                    install_anima_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"FlashAttention is not available for Anima: {e}")
            elif model.family == "Krea 2":
                try:
                    install_krea2_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"FlashAttention is not available for Krea 2: {e}")

        try:
            self.instance.vae.to(memory_format=torch.channels_last)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Can't apply memory format optimization: {e}")

        # Read before anything reaches the GPU: this is what the weights and the
        # activations will share.
        memory_info = get_memory_info()
        self.memory_reserve = 0
        self.memory_budget = None
        self.streams_weights = False
        self.offload_strategy = None

        # On NVIDIA, AMD & Intel ARC GPUs:
        if memory_info and (torch.cuda.is_available() or torch.xpu.is_available()):
            self.offload_to_cpu(memory_info[1])

        # On Mac GPUs:
        elif torch.backends.mps.is_available():
            self.instance.to("mps")

            # To prevent swap and performance degradation...
            if hasattr(self.instance, "enable_attention_slicing"):
                self.instance.enable_attention_slicing()

        self.memory_budget = self.measure_memory_budget(memory_info)

        return model

    def measure_memory_budget(self, memory_info: tuple[int, int] | None) -> int | None:
        """Measure the GPU memory a single VAE pass can count on, in bytes.

        Only the weights that can't leave the GPU are deducted. Under the offload
        strategy that's the VAE itself, every sibling being evictable, so the
        decode gets nearly the whole card rather than the leftovers of whatever
        ran before it. It starts from the free memory, never the total: the
        desktop takes its cut first, and a share of the card would ignore it.

        Args:
            memory_info: GPU memory free and total, in bytes, read with the
                pipeline still on CPU, or `None` if it couldn't be measured.

        Returns:
            The budget, or `None` if it can't be measured.
        """
        if memory_info is None or self.instance is None:
            return None

        free_memory = memory_info[0]

        # Streamed weights never claim the card, leaving all of it to the pass.
        if self.streams_weights:
            return free_memory

        def footprint_of(component) -> int:
            get_footprint = getattr(component, "get_memory_footprint", None)

            return get_footprint() if get_footprint is not None else 0

        if self.offload_strategy is not None:
            resident = footprint_of(getattr(self.instance, "vae", None))
        else:
            # Nothing evicts on Mac: the whole pipeline stays on the GPU.
            resident = sum(map(footprint_of, self.instance.components.values()))

        budget = max(self.memory_reserve, free_memory - resident)

        logger.info(
            f"VAE budget: {budget / BYTES_PER_MEGAPIXEL:.1f}MP in one pass, "
            f"{free_memory / 1024**3:.1f}GB free."
        )

        return budget

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

        megapixels = width * height / 1e6
        peak = int(megapixels * BYTES_PER_MEGAPIXEL)

        # An unmeasurable GPU gets the safe path rather than an optimistic one.
        needs_tiling = self.memory_budget is None or peak > self.memory_budget

        # The strategy sizes its evictions on the weights it moves, not on the
        # activations behind them: the VAE always looks small enough to fit, so
        # whatever ran before keeps the room the decode needs. Hence its peak.
        if self.offload_strategy is not None:
            margin = max(
                self.memory_reserve, MIN_MEMORY_RESERVE if needs_tiling else peak
            )
            self.offload_strategy.memory_reserve_margin = margin

            logger.info(
                f"Decoding {megapixels:.1f}MP "
                f"{'tiled' if needs_tiling else 'in one pass'}, "
                f"evicting down to {margin / 1024**3:.1f}GB free."
            )

        toggle = getattr(
            vae, "enable_tiling" if needs_tiling else "disable_tiling", None
        )

        if toggle is not None:
            toggle()

    def offload_to_cpu(self, total_memory: int) -> None:
        """Offload the pipeline weights to CPU, keeping on GPU what fits.

        Args:
            total_memory: The GPU memory, in bytes.
        """
        if self.instance is None:
            return

        if isinstance(self.instance, DiffusionPipeline):
            self.instance.enable_sequential_cpu_offload()
            self.streams_weights = True
            return

        # A modular pipeline has no offload helper: replicate the auto CPU offload
        # of its components manager, which can't be used as is because it hooks
        # every component, including those too large to stay on the GPU.
        device = get_execution_device()
        memory_reserve = get_memory_reserve(total_memory)
        self.memory_reserve = memory_reserve

        offload_strategy = FreeMemoryOffloadStrategy(
            memory_reserve_margin=memory_reserve
        )
        self.offload_strategy = offload_strategy

        hooks = []

        for name, component in self.instance.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            footprint = getattr(component, "get_memory_footprint", None)
            fits_on_gpu = (
                footprint is not None and footprint() + memory_reserve <= total_memory
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
