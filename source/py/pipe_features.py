"""Feature detection helpers for diffusion pipelines."""


def pipe_supports_strength(pipe) -> bool:
    """Does the given pipeline accept a `strength` argument?

    Returns:
        `True` for strength-based image-to-image pipelines (e.g. Anima, Z-Image),
        `False` for conditioning-based edit models (e.g. FLUX.2 [klein])
        and for text-to-image only pipelines.
    """
    blocks = getattr(pipe, "blocks", None)
    return blocks is not None and "strength" in blocks.input_names
