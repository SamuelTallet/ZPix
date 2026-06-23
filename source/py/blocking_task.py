"""Blocking task management."""

# To guarantee some stability, ZPix runs one critical task at a time:
# generating an image, (un)loading a LoRA, (down)loading a model.

from contextlib import contextmanager


class BlockingTask:
    is_running: bool = False
    """Is a blocking task in progress?"""

    message: str | None = None
    """Message of the blocking task, if it is in progress."""

    @classmethod
    @contextmanager
    def run(cls, message: str):
        """Mark a blocking task as running.

        Args:
            message: Message describing the running task.
        """
        cls.is_running = True
        cls.message = message
        try:
            yield
        finally:
            cls.is_running = False
            cls.message = None
