from collections.abc import Callable
from json import load as load_json
from pathlib import Path

from source.py.custom_logger import logger


def get_translate_func(
    translations_dir: Path,
    locale: str,
) -> Callable[[str], str]:
    """Get a function translating a string for a given locale.

    Falls back to the untranslated string if the locale
    has no translation file or the string is not translated.

    Args:
        translations_dir: Directory containing translation files.
        locale: Locale (e.g. "fr-FR").
    """
    translation: dict[str, str] = {}

    # Skip default locale.
    if locale != "en-US":
        translation_file = translations_dir / f"{locale}.json"

        if translation_file.exists():
            with open(translation_file, "r", encoding="utf-8") as file:
                translation = load_json(file)
        else:
            logger.warning(f"Translation for {locale} not found.")

    def translate(string: str) -> str:
        """Translate a string."""
        return translation.get(string, string)

    return translate
