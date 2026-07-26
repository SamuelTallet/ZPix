"""Image models management."""

from collections.abc import Callable
from pathlib import Path

import gradio as gr
from huggingface_hub import snapshot_download
from pydantic import TypeAdapter

from source.py.blocking_task import BlockingTask
from source.py.image_model import ImageModel


def get_models(json_file: Path) -> list[ImageModel]:
    """Get image models from a JSON file."""
    adapter = TypeAdapter(list[ImageModel])
    models_json = json_file.read_text(encoding="utf-8")

    return adapter.validate_json(models_json)


def find_model(model_id: str, models: list[ImageModel]) -> ImageModel:
    """Find an image model by its ID among a list.
    Raises: StopIteration if not found.
    """
    return next(m for m in models if m.id == model_id)


def download_model(model: ImageModel, t: Callable[[str], str]) -> None:
    """Download an image model.
    Raises: gr.Error if download definitively fails.
    """
    try:
        snapshot_download(model.id)
    except Exception:  # noqa: BLE001
        if model.backup_id:
            gr.Warning(
                t("Can't download {id}, let's use backup model...").format(id=model.id)
            )
            try:
                snapshot_download(model.backup_id)
            except Exception:  # noqa: BLE001
                raise gr.Error(
                    t("Can't download backup model {id}.").format(id=model.backup_id)
                )
        else:
            raise gr.Error(
                t("Can't download model {id}, no backup available.").format(id=model.id)
            )


def fetch_model(model: ImageModel, t: Callable[[str], str]) -> None:
    """Fetch an image model, blocking other critical tasks."""
    with BlockingTask.run(t("Please wait, a model is being downloaded.")):
        download_model(model, t)
