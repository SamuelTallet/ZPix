"""OS abstractions."""

import os
from pathlib import Path
from subprocess import run
from sys import platform
from typing import Literal

MAC_KEY_NAMES = {"Alt": "Option", "Ctrl": "Cmd"}
"""Key names labeled differently on Mac keyboards."""


def open_with_default_app(target: Path | str):
    """Open a directory, file or URL using system's default app.

    Args:
        target: Directory, file or URL to open.

    May raise an exception.
    """
    match platform:
        case "linux":
            run(["xdg-open", target], check=True)
        case "darwin":  # macOS
            run(["open", target], check=True)
        case "win32":
            os.startfile(target)
        case _:
            raise OSError(f"Unsupported platform: {platform}")


def get_size_base() -> Literal[1000, 1024]:
    """Get bytes in a kilobyte as system's file manager counts them.

    Windows Explorer counts in powers of 1024,
    macOS Finder and GNOME Files in powers of 1000.
    """
    return 1024 if platform == "win32" else 1000


def get_key_name(key: str) -> str:
    """Get name of a key as labeled on system's keyboards."""
    match platform:
        case "darwin":
            return MAC_KEY_NAMES.get(key, key)
        case _:
            return key
