"""Filter for unwanted stdout output."""
# Applied as a side effect on import.

import sys

_stdout_write = sys.stdout.write
"""The original stdout writer."""


_SILENCED = ("not documented",)  # "ERROR ... not documented" from transformers
"""Substrings that, when found in a write, cause it to be dropped."""


def _filtered_write(s: str) -> int:
    return 0 if any(pattern in s for pattern in _SILENCED) else _stdout_write(s)


sys.stdout.write = _filtered_write  # ty: ignore[invalid-assignment]
