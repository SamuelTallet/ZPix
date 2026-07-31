"""LoRA models management."""

from collections.abc import Callable
from pathlib import Path

import gradio as gr
from crossfiledialog import open_file
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from source.py.blocking_task import BlockingTask
from source.py.custom_errors import EventAbort
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.image_pipe import ImagePipeline
from source.py.lora_model import LoraModel


def select_lora_file(t: Callable[[str], str]) -> Path | None:
    """Prompt user to select a LoRA file.

    Args:
        t: Translation function.

    Returns:
        Path to selected LoRA file, or None if user cancels.
    """
    selected_file: str | None = open_file(
        title=t("Select a LoRA file to load in ZPix"),
        filter={t("LoRA file"): ["*.safetensors"]},
    )

    if not selected_file:
        return None

    return Path(selected_file)


def validate_lora_swap(path: Path | None, t: Callable[[str], str]):
    """Validate a LoRA selection before swapping.

    Args:
        path: Path to selected LoRA file, or None if selection was cancelled.
        t: Translation function.

    Raises:
        EventAbort: If the user cancelled the file dialog.
        gr.Error: If the file is not a *.safetensors, or a blocking task runs.
    """
    if path is None:
        raise EventAbort("LoRA file selection cancelled.")

    if path.suffix != ".safetensors":
        raise gr.Error(
            t("LoRA file extension must be .safetensors"),
            duration=10,
        )

    if BlockingTask.is_running:
        raise gr.Error(str(BlockingTask.message), duration=4)


def swap_lora(
    path: Path,
    image_model: ImageModel,
    t: Callable[[str], str],
    image_pipe: ImagePipeline,
) -> str | None:
    """Swap or load a new LoRA model.

    Args:
        path: Path to a LoRA file.
        image_model: Loaded image model.
        t: Translation function.
        image_pipe: Loaded image model pipeline.

    Returns:
        Trigger word of LoRA model.

    Raises:
        gr.Error: If the pipeline does not support LoRA.
    """
    pipe = image_pipe.instance

    if not isinstance(pipe, LoraBaseMixin):
        raise gr.Error("Pipeline doesn't support LoRA.")

    lora = LoraModel(path)

    try:
        if lora.base_model() not in image_model.base_ids:
            gr.Warning(
                f"{t('This LoRA seems incompatible with')} {image_model.name}.<br>"
                f"{t('It might not work.')}",
                duration=5,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Can't check LoRA compatibility: {e}")

    bfloat16_lora = lora.to_bf16()

    # Workaround: Diffusers FLUX.2 LoRA converter doesn't handle .alpha keys.
    if image_model.family == "FLUX.2":
        bfloat16_lora = {
            k: v for k, v in bfloat16_lora.items() if not k.endswith(".alpha")
        }

    try:
        with (
            BlockingTask.run(t("Please try again, a LoRA was loading.")),
            image_pipe.unhooked(),
        ):
            pipe.unload_lora_weights()
            pipe.load_lora_weights(
                bfloat16_lora,
                adapter_name="lora_1",
            )
    except Exception as error:
        raise gr.Error(t("Failed to load LoRA."), duration=4) from error

    trigger_word = lora.trigger_word()

    return trigger_word


def set_lora_strength(
    strength: float,
    pipe: DiffusionPipeline | ModularPipeline | None,
):
    """Set the strength of the loaded LoRA adapter.

    Args:
        strength: Scale applied to the LoRA weights.
        pipe: Pipeline whose LoRA strength is updated.

    Raises:
        gr.Error: If the pipeline doesn't support LoRA
            or no LoRA is loaded on its transformer.
    """
    if not isinstance(pipe, LoraBaseMixin):
        raise gr.Error("Pipeline doesn't support LoRA.")

    adapters = pipe.get_list_adapters()

    if "transformer" not in adapters or "lora_1" not in adapters["transformer"]:
        raise gr.Error("No LoRA loaded.")

    pipe.set_adapters("lora_1", strength)


def unload_lora(pipe: DiffusionPipeline | ModularPipeline | None):
    """Unload all LoRA weights from the pipeline.

    Args:
        pipe: Pipeline to remove the LoRA weights from.

    Raises:
        gr.Error: If the pipeline doesn't support LoRA.
    """
    if not isinstance(pipe, LoraBaseMixin):
        raise gr.Error("Pipeline doesn't support LoRA.")

    pipe.unload_lora_weights()


def extract_lora_name(lora_path: Path) -> str:
    """Extract the display name of a LoRA from its file path.

    Args:
        lora_path: Path to the LoRA file.

    Returns:
        The file name without its extension.
    """
    return lora_path.stem
