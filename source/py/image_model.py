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

    default: Settings
    """Default settings of this model."""

    def supports_inpainting(self) -> bool:
        """Does this model's family support inpainting?"""
        return self.family in (
            "Anima",
            "FLUX",
            "FLUX.2",
            "Krea 2",
            "Qwen-Image 2.1",
            "SDXL",
            "Z-Image",
        )

    def has_modular_pipeline(self) -> bool:
        """Does this model's family has a Diffusers modular pipeline?"""
        return self.family in (
            "Anima",
            "ERNIE-Image",
            "FLUX",
            "FLUX.2",
            "Krea 2",
            "Qwen-Image",
            "Qwen-Image 2.1",
            "SD3",
            "SDXL",
            "Z-Image",
        )
