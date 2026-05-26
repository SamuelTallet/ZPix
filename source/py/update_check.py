"""Update check helpers."""

from collections.abc import Callable

import gradio as gr
import requests
from semantic_version import Version

UPDATE_AVAILABLE = "A new ZPix version is available! Check out what's new and download it <a href='{url}'>here</a>."
"""Message displayed when an update is available."""


def _fetch_version(url: str) -> str:
    """Fetch the remote version string from a given URL."""
    response = requests.get(url, timeout=1)  # second
    response.raise_for_status()

    return response.text.strip()


def check_for_updates(
    local_version: str,
    version_url: str,
    releases_url: str,
    t: Callable[[str], str],
):
    """Compare local version against the remote one.
    Notify the user if a newer release is available.

    Args:
        local_version: Installed version.
        version_url: URL returning the latest version.
        releases_url: Base URL of the releases page.
        t: Translation function.
    """
    # on_app_load() executes this request in a thread,
    # so it seems safe to let any errors propagate.
    remote_version = _fetch_version(version_url)

    if Version(remote_version) > Version(local_version):
        # Tags are kept in sync with VERSION files.
        release_url = f"{releases_url}/tag/v{remote_version}"

        gr.Info(
            t(UPDATE_AVAILABLE).format(url=release_url),
            duration=None,
        )
