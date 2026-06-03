from typing import Literal

from pydantic import BaseModel


class Settings(BaseModel):
    """Settings for an image model."""

    steps: int
    """Inference steps."""

    cfg: float
    """Classifier-free guidance scale."""


class ImageModel(BaseModel):
    """An image model."""

    id: str
    """Model ID at Hugging Face.
    Example: Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32
    """

    backup_id: str | None
    """Backup model ID at Hugging Face."""

    name: str
    """Model name.
    Example: Z-Image Turbo
    """

    family: str
    """Family of this model.
    Example: Z-Image
    """

    pipeline: str
    """Diffusers pipeline class of this model."""

    default: Settings
    """Default settings of this model."""

    base_ids: list[str]
    """Known IDs of base image model.
    Example: zimage
    """

    features: list[
        Literal[
            "text-to-image",
            "image-to-image",
        ]
    ]
    """Features supported by this model."""
