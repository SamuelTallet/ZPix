"""Gallery images helpers."""

from logging import warning
from pathlib import Path
from re import fullmatch

import gradio as gr


def delete_image(
    images: list[tuple[str, str]],
    index: int | None,
    output_paths: dict[str, str],
) -> tuple[dict, int | None, dict]:
    """Delete an image from gallery, including its file in output directory.

    Returns:
        Tuple of (updated gallery, new selected index, updated output paths).
    """
    if not images or index is None:
        raise Exception("Gallery is empty or no image was selected")

    temp_file, _caption = images.pop(index)
    new_index = min(index, len(images) - 1) if images else None

    try:
        # Image filename follows a pattern, see generate()
        path_match = fullmatch(r"image_(\d+)", Path(temp_file).stem)

        if not path_match:
            raise Exception("Path mismatch")

        image_id = path_match.group(1)
        output_file = output_paths.pop(image_id)
        Path(output_file).unlink()
    except Exception as error:
        # Maybe image was deleted outside ZPix meanwhile.
        warning(f"Can't delete image file: {error}")

    return gr.update(value=images), new_index, output_paths
