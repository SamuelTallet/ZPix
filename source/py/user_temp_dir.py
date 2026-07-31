"""User temporary directory management."""

import os
from pathlib import Path
from sys import platform
from tempfile import gettempdir


def get_user_temp_dir() -> Path:
    """Get directory where ZPix stores temporary files of current user.

    Each consumer is expected to append its own subfolder.
    """
    # On Windows and macOS, the temp directory belongs to the current user.
    if platform in ("win32", "darwin"):
        return Path(gettempdir()) / "ZPix"

    # On Linux, the temp directory is shared between users, hence the UID.
    return Path(gettempdir()) / f"ZPix-{os.getuid()}"
