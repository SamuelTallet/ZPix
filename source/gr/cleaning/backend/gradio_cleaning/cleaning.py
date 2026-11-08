"""Cleaning Gradio custom component."""

from __future__ import annotations

from typing import Any, Literal

from gradio.components.base import Component
from gradio.data_classes import GradioModel


class CleaningItem(GradioModel):
    """Something user may clean, e.g. a category of files."""

    id: str
    label: str
    size: int | None = None
    """Bytes freed by cleaning it, or None while being measured."""


class CleaningValue(GradioModel):
    """Items user may clean, and IDs of the selected ones."""

    items: list[CleaningItem]
    selected: list[str]


class Cleaning(Component):
    """Modal letting user select items to clean, e.g. to free up disk space.

    Hidden by default: show it with `gr.update(visible=True, value=...)`.
    """

    EVENTS = ["clean", "close"]  # noqa: RUF012 (Gradio API)

    data_model = CleaningValue

    def __init__(
        self,
        value: CleaningValue | None = None,
        *,
        title: str | None = None,
        clean_label: str = "Clean",
        cancel_label: str = "Cancel",
        size_base: Literal[1000, 1024] = 1000,
        visible: bool = False,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
    ):
        """
        Parameters:
            value: items user may clean, and IDs of the selected ones.
            title: shown at the top of the modal.
            clean_label: label of the button triggering the `clean` event.
            cancel_label: label of the button triggering the `close` event.
            size_base: bytes in a kilobyte when showing sizes, e.g. 1024 to match Windows Explorer, which counts in powers of 1024 while writing kB, MB, GB.
            visible: if True, the modal is shown.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
        """
        self.title = title
        self.clean_label = clean_label
        self.cancel_label = cancel_label
        self.size_base = size_base

        super().__init__(
            value=value,
            show_label=False,
            container=False,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
        )

    def preprocess(self, payload: CleaningValue | None) -> CleaningValue | None:
        """
        Parameters:
            payload: items, and IDs of those selected by user.
        Returns:
            Passes them as is.
        """
        return payload

    def postprocess(self, value: CleaningValue | None) -> CleaningValue | None:
        """
        Parameters:
            value: Expects items, and IDs of the selected ones.
        Returns:
            Them as is.
        """
        return value

    def example_payload(self) -> Any:
        return {
            "items": [{"id": "temp", "label": "Temporary files", "size": 1024}],
            "selected": ["temp"],
        }

    def example_value(self) -> Any:
        return CleaningValue(
            items=[CleaningItem(id="temp", label="Temporary files", size=1024)],
            selected=[],
        )
