"""Gradio server closing."""

import gradio as gr

from source.py.custom_logger import logger


def close_server(app: gr.Blocks) -> None:
    """Close Gradio server.

    Args:
        app: Launched app.
    """
    logger.info("Closing server.")
    app.close()
