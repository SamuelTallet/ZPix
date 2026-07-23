"""Diffusion pipeline wrapper."""

import gc
from collections.abc import Callable

import gradio as gr
import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.krea2_flash import install_krea2_flash_attn


class ImagePipeline:
    """An image pipeline."""

    def __init__(self):
        self.instance: ModularPipeline | DiffusionPipeline

    def load(self, model: ImageModel) -> ImageModel:
        """Load an image model pipeline."""

        def create_pipe(
            model_id: str,
        ) -> tuple[DiffusionPipeline | ModularPipeline, ComponentsManager | None]:
            """Create a standard pipeline or a modular one."""
            if not model.has_modular_pipeline():
                return DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                ), None

            components_manager = ComponentsManager()
            modular_pipeline = ModularPipeline.from_pretrained(
                model_id,
                components_manager=components_manager,
            )
            modular_pipeline.load_components(torch_dtype=torch.bfloat16)

            return modular_pipeline, components_manager

        try:
            self.instance, manager = create_pipe(model.id)
        except Exception:
            if model.backup_id:
                logger.warning(
                    f"Can't load {model.id}, falling back to {model.backup_id}."
                )
                self.instance, manager = create_pipe(model.backup_id)
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
                except Exception as e:
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available: {e}")
            elif model.family == "Krea 2":
                try:
                    install_krea2_flash_attn(self.instance)
                except Exception as e:
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available for Krea 2: {e}")
            else:
                self.instance.transformer.set_attention_backend("native")

        try:
            self.instance.vae.to(memory_format=torch.channels_last)
        except RuntimeError as e:
            logger.warning(f"Can't apply memory format optimization: {e}")

        # On NVIDIA, AMD & Intel ARC GPUs:
        if torch.cuda.is_available() or torch.xpu.is_available():
            if manager:
                manager.enable_auto_cpu_offload()
            else:
                self.instance.enable_sequential_cpu_offload()

        # On Mac GPUs:
        elif torch.backends.mps.is_available():
            self.instance.to("mps")

            # To prevent swap and performance degradation...
            if hasattr(self.instance, "enable_attention_slicing"):
                self.instance.enable_attention_slicing()

        return model

    def swap(self, model: ImageModel, t: Callable[[str], str]) -> ImageModel:
        """Swap an image model pipeline, blocking other critical tasks.

        Args:
            model: The image model to load.
            t: Translation function.
        """
        with BlockingTask.run(t("Please wait, a model is being loaded.")):
            # Break the hook <-> module reference cycles left by CPU offload.
            if hasattr(self.instance, "remove_all_hooks"):
                self.instance.remove_all_hooks()

            # Drop the old pipeline before collecting, otherwise it stays alive
            # and its VRAM makes the next auto CPU offload overly aggressive.
            del self.instance
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
            `False` for conditioning-based edit models (e.g. FLUX.2 [klein])
            and for text-to-image only pipelines.
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
        except Exception:
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
