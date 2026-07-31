"""User data directories management."""

from pathlib import Path
from shutil import move

from platformdirs import user_cache_path, user_config_path

from source.py.custom_logger import logger


def get_user_preferences_dir() -> Path:
    """Get directory where ZPix stores user preferences.

    This includes configuration, terms acceptance, etc.
    """
    return user_config_path("ZPix", appauthor=False)


def migrate_user_preferences() -> None:
    """Move output directory config from the legacy "~/.zpix" directory,
    used by ZPix v1.0.6 and v1.0.7, to the current preferences directory.
    """
    legacy_output_dir_cfg = Path.home() / ".zpix" / "output_dir.cfg"
    output_dir_cfg = get_user_preferences_dir() / "output_dir.cfg"

    if not legacy_output_dir_cfg.exists() or output_dir_cfg.exists():
        return

    try:
        output_dir_cfg.parent.mkdir(parents=True, exist_ok=True)
        move(legacy_output_dir_cfg, output_dir_cfg)

    except Exception as error:  # noqa: BLE001
        logger.warning(f"Output directory config migration failed: {error}")


def get_user_cache_dir() -> Path:
    """Get directory where ZPix stores cache for current user.

    This includes Python bytecode, virtual env, uv, etc.
    """
    return user_cache_path("ZPix", appauthor=False)
