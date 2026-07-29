"""User temporary directory management."""

import os
from pathlib import Path
from sys import platform
from tempfile import gettempdir


def get_user_temp_dir() -> Path:
    """Get directory where ZPix stores temporary files of current user.

    Each consumer is expected to append its own subfolder.
    """
    if platform == "win32":
        # The Windows temp directory already belongs to the current user.
        return Path(gettempdir()) / "ZPix"

    # On Unix, the temp directory is shared between users, hence the UID.
    return Path(gettempdir()) / f"ZPix-{os.getuid()}"
