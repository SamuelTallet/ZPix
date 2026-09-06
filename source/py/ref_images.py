"""Reference images."""

from collections.abc import Callable
from typing import Literal

import gradio as gr
from PIL import Image, ImageOps

from source.py.image_model import ImageModel
from source.py.image_pipe import ImagePipeline
from source.py.image_utilities import to_rgb


def _takes_and_has_one_ref(
    pipe: ImagePipeline,
    ref_images: dict | None,
) -> bool:
    """Does the loaded model take a single reference image, and was one added?

    Args:
        pipe: Image pipeline in use.
        ref_images: Multimodal dictionary possibly containing files.

    Returns:
        `True` when a strength-based pipeline is loaded
        and at least one reference image was added.
    """
    return bool(pipe.supports_strength() and ref_images and ref_images.get("files"))


def normalize_ref_image(file: str) -> Image.Image:
    """Normalize a reference image file.

    Args:
        file: Path to the reference image.

    Returns:
        An upright RGB image.
    """
    image = Image.open(file)
    ImageOps.exif_transpose(image, in_place=True)

    return to_rgb(image)


def show_ref_image_strength(
    pipe: ImagePipeline,
    ref_images: dict | None,
) -> dict:
    """Show the reference strength row only when the loaded model
    uses it and at least one reference image was added.

    Args:
        pipe: Image pipeline in use.
        ref_images: Multimodal dictionary possibly containing files.

    Returns:
        Reference strength row visibility update.
    """
    return gr.update(visible=_takes_and_has_one_ref(pipe, ref_images))


def get_ref_images_label(
    pipe: ImagePipeline,
    t: Callable[[str], str],
) -> str:
    """Get the reference images block label for the loaded model.

    Args:
        pipe: Image pipeline in use.
        t: Translation function.

    Returns:
        A singular label for strength-based pipelines, which condition
        on a single reference image, a plural one otherwise.
    """
    return t("Reference Image") if pipe.supports_strength() else t("Reference Images")


def get_ref_images_file_count(pipe: ImagePipeline) -> Literal["single", "multiple"]:
    """Get the reference images block file count for the loaded model.

    Args:
        pipe: Image pipeline in use.

    Returns:
        `"single"` for strength-based pipelines, which condition on a single
        reference image, `"multiple"` otherwise.
    """
    return "single" if pipe.supports_strength() else "multiple"


def update_ref_images(
    image_model: ImageModel,
    pipe: ImagePipeline,
    ref_images: dict | None,
    t: Callable[[str], str],
) -> dict:
    """Update the reference images block according to the loaded model.

    Args:
        image_model: Image model in use.
        pipe: Image pipeline in use.
        ref_images: Multimodal dictionary possibly containing files.
        t: Translation function.

    Returns:
        Reference images block label and file count update, along with a value
        update when reference images the model can't take were added.
    """
    files: list[str] = (ref_images or {}).get("files", [])

    block_update = gr.update(
        label=get_ref_images_label(pipe, t),
        # Let the block itself turn its upload zone off once it holds the single
        # reference image a strength-based pipeline conditions on.
        file_count=get_ref_images_file_count(pipe),
    )

    # A strength-based pipeline can be loaded while several reference images are
    # already added, e.g. coming from an edit model, so drop the extra ones. The
    # label would otherwise turn singular over a block still holding them all.
    if pipe.supports_strength() and len(files) > 1:
        block_update["value"] = {"files": files[:1]}

        gr.Warning(
            t("{model} supports only one reference image.").format(
                model=image_model.name
            ),
            duration=5,
        )

    return block_update
