"""User data directories management."""

from pathlib import Path

from platformdirs import user_cache_path


def get_user_preferences_dir() -> Path:
    """Get directory where ZPix stores user preferences.

    This includes configuration, terms acceptance, etc.
    """
    return Path.home() / ".zpix"


def get_user_cache_dir() -> Path:
    """Get directory where ZPix stores cache for current user.

    This includes Python bytecode, virtual env, uv, etc.
    """
    return user_cache_path("ZPix", appauthor=False)
