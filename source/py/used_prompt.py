import gradio as gr


def sync_used_prompt(
    gallery: list[tuple[str, str]],
    selected_image_index: int | None,
) -> dict:
    """Sync "Used prompt" value and visibility
    according to image selected in gallery.
    """
    if not gallery or selected_image_index is None:
        return gr.update(value=None, visible=False)

    # Prompt is stored as image caption.
    prompt = gallery[selected_image_index][1]

    return gr.update(value=prompt, visible=bool(prompt))
