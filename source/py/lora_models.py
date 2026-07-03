"""LoRA models management."""

from collections.abc import Callable
from pathlib import Path

from crossfiledialog import open_file


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


def extract_lora_name(lora_path: Path) -> str:
    """Extract the display name of a LoRA from its file path.

    Args:
        lora_path: Path to the LoRA file.

    Returns:
        The file name without its extension.
    """
    return lora_path.stem
