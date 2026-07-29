"""User data directory management."""

from pathlib import Path


def get_user_data_dir() -> Path:
    """Get directory where ZPix stores user data.

    User data includes configuration files, Terms acceptance, etc.
    Output images are stored elsewhere. See `get_output_dir()`.
    """
    return Path.home() / ".zpix"
