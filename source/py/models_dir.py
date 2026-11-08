"""Models directory management."""

from collections.abc import Callable
from os import environ
from pathlib import Path

import gradio as gr
from crossfiledialog import choose_folder

from source.py.custom_logger import logger
from source.py.user_data_dir import get_user_preferences_dir


def _models_dir_cfg() -> Path:
    """Path to config file storing models directory set by user."""
    return get_user_preferences_dir() / "models_dir.cfg"


def get_models_dir() -> Path | None:
    """Get directory where ZPix stores downloaded image models, in this order:
    - Custom directory possibly set by user
    - None, meaning Hugging Face's default cache location applies
    """
    try:
        models_dir_cfg = _models_dir_cfg()

        if models_dir_cfg.exists():
            return Path(models_dir_cfg.read_text(encoding="utf-8"))

    except Exception as error:  # noqa: BLE001
        logger.warning(f"We'll use the default models directory because: {error}")

    return None


def configure_hf_cache() -> None:
    """Point the Hugging Face cache at the models directory set by user.

    Does nothing when no custom directory is set. Must run before anything
    imports huggingface_hub: the library resolves its cache location from
    the environment at import time.
    """
    models_dir = get_models_dir()

    if models_dir is None:
        return

    environ["HF_HUB_CACHE"] = str(models_dir)


def change_models_dir(t: Callable[[str], str]):
    """Prompt user to select a models directory
    then write it in a file, later read by configure_hf_cache().

    Args:
        t: Translation function.
    """
    selected_dir: str | None = choose_folder(
        title=t("Where should ZPix store downloaded image models?"),
    )

    if not selected_dir:
        return

    models_dir_cfg = _models_dir_cfg()
    models_dir_cfg.parent.mkdir(parents=True, exist_ok=True)
    models_dir_cfg.write_text(selected_dir, encoding="utf-8")

    gr.Info(
        t("New models folder will take effect at next ZPix start."),
        duration=None,
    )
