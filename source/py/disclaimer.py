"""Disclaimer helper."""

from logging import warning
from pathlib import Path


class TermsOfUse:
    """Terms of Use helper."""

    accepted_file: Path
    """Path to file created when Terms of Use are accepted."""

    def __init__(
        self,
        accepted_file: Path,
    ):
        """Initialize Terms of Use helper."""
        if not isinstance(accepted_file, Path):
            raise TypeError("accepted_file must be a Path")

        self.accepted_file = accepted_file

    def accept(self):
        """Accept Terms of Use."""
        try:
            self.accepted_file.write_text("by user")
        except Exception as e:
            warning(f"Can't write {self.accepted_file}: {e}")
            # Even on file write error, it's crucial to continue,
            # otherwise user won't be able to use app.

    def accepted(self) -> bool:
        """Terms of Use were accepted?"""
        return self.accepted_file.exists()
