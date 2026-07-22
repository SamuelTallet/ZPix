"""Image utilities."""

from PIL import Image

from source.py.custom_logger import logger


def to_rgb(
    image: Image.Image,
    bg_color: str = "white",
) -> Image.Image:
    """Convert an image to RGB, flattening any transparency.

    Only pixels are carried over: metadata may be dropped along the way.

    Args:
        image: Image to process.
        bg_color: Opaque background color to flatten transparency onto.

    Raises:
        TypeError: If image is not a PIL image or background color is not a string.

    Returns:
        A new RGB image or same image if already in RGB without transparency.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Image must be a PIL Image, got {type(image)}")

    if not isinstance(bg_color, str):
        raise TypeError(f"Background color must be a str, got {type(bg_color)}")

    has_transparency = image.has_transparency_data

    if image.mode == "RGB" and not has_transparency:
        # Detach from source file, as conversion paths implicitly do.
        image.load()
        return image

    original_mode = image.mode

    # TODO Rescale "I", "I;16*" and "F" modes, as PIL clips them to 0-255.

    if has_transparency:
        if image.mode == "La":
            # Premultiplied grayscale only converts to its straight alpha form.
            image = image.convert("LA")

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        bg_image = Image.new("RGBA", image.size, bg_color)
        image = Image.alpha_composite(bg_image, image)

    rgb_image = image.convert("RGB")

    if has_transparency:
        logger.info(
            f"{original_mode} image transparency flattened onto {bg_color} background.",
        )
    else:
        logger.info(f"{original_mode} image converted to RGB.")

    return rgb_image
