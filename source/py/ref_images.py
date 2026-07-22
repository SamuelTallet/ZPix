"""Reference images."""

import gradio as gr
from PIL import Image, ImageOps

from source.py.image_pipe import ImagePipeline
from source.py.image_utilities import to_rgb


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
    return gr.update(
        visible=bool(
            pipe.supports_strength() and ref_images and ref_images.get("files")
        )
    )
