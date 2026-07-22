"""Reference images."""

import gradio as gr

from source.py.image_pipe import ImagePipeline


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
