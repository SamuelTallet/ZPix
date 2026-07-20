"""ZPix Gradio app."""

# Based on https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo
from argparse import ArgumentParser
from functools import partial
from os import environ
from pathlib import Path
from shutil import rmtree

import gradio as gr
import torch

import source.py.stdout_filter  # noqa: F401

# Force tensorwise FP8 matmul kernels as a fallback on hardware that lacks
# native row-wise FP8 support, such as consumer NVIDIA Blackwell cards.
# This must run before importing SDNQ, which reads the variable at import time.
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0):
    environ.setdefault("SDNQ_USE_TENSORWISE_FP8_MM", "1")

from sdnq import SDNQConfig  # noqa: F401

from source.py.blocking_task import BlockingTask
from source.py.custom_theme import get_theme
from source.py.disclaimer import TermsOfUse
from source.py.ex_prompts import get_example_prompts
from source.py.gallery_images import delete_image
from source.py.image_gen import generate
from source.py.image_model import ImageModel
from source.py.image_models import fetch_model, find_model, get_models
from source.py.image_pipe import ImagePipeline
from source.py.lora_models import (
    extract_lora_name,
    select_lora_file,
    set_lora_strength,
    swap_lora,
    unload_lora,
    validate_lora_swap,
)
from source.py.os_abstract import open_with_default_app
from source.py.output_dir import change_output_dir, get_output_dir
from source.py.prompt_extract import extract_update_prompt
from source.py.resolutions import get_aspects_and_resolutions
from source.py.translations import get_translate_func
from source.py.trigger_word import remove_trigger_word, update_trigger_word
from source.py.update_check import check_for_updates
from source.py.used_prompt import sync_used_prompt

# Path to Triton cache directory
# shortened by good measure to avoid too long path errors on Windows
# even if this has been fixed recently.
environ["TRITON_CACHE_DIR"] = str(Path.home() / ".triton")

app_dir = Path(__file__).parent
"""App directory."""

# As we store temp files created by Gradio in this app' subfolder
# we can remove them without worry about impacting other Gradio apps.
gradio_temp_dir = app_dir / "temp" / "GradioApp"
environ["GRADIO_TEMP_DIR"] = str(gradio_temp_dir)

# This temp directory may hold files existing also in output directory
# so we clear it on each app run to save space.
rmtree(gradio_temp_dir, ignore_errors=True)

assets_dir = app_dir / "assets"
"""Assets directory."""

# Let's serve assets directly.
gr.set_static_paths(paths=[assets_dir])

metadata: dict[str, str] = {}
"""App metadata."""

models: list[ImageModel] = []
"""Available image models."""

output_dir = get_output_dir()
"""The folder where ZPix saves generated images."""


def get_metadata(filename: str) -> str:
    """Get metadata."""
    if filename not in metadata:
        file = app_dir / "metadata" / filename
        metadata[filename] = file.read_text()

    return metadata[filename]


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--port", type=int, required=True)
    arg_parser.add_argument("--in-browser", action="store_true", default=False)
    arg_parser.add_argument("--locale", type=str, required=False, default="en-US")
    args, _ = arg_parser.parse_known_args()

    if not (
        torch.cuda.is_available()
        or torch.xpu.is_available()
        or torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "PyTorch couldn't find an accelerator; try updating your GPU drivers."
        )

    t = get_translate_func(app_dir / "translations", args.locale)
    """Translation function."""

    image_pipe = ImagePipeline()
    """Image pipeline."""

    with gr.Blocks(
        title=f"{get_metadata('NAME')} {get_metadata('VERSION')}",
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        models = get_models(app_dir / "data" / "curated_models.json")
        initial_model = image_pipe.load(models[0])

        model = gr.State(value=initial_model)
        """Loaded image model."""

        (
            resolutions_by_aspect,
            default_resolution_choices,
            aspect_ratio_choices,
            default_aspect_ratio,
        ) = get_aspects_and_resolutions()

        tou = TermsOfUse(app_dir / ".tou_accepted")

        with gr.Row(elem_classes=[] if tou.accepted() else ["blurred"]) as ui_row:
            with gr.Column(min_width=48, elem_classes=["sidebar"]):
                swap_lora_btn = gr.Button(
                    "",
                    icon=assets_dir / "lora_grad.svg",
                    elem_id="swap-lora-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("swap-lora-btn")
                        btn.title = "{t("Load a LoRA file to apply a new style")}"
                    """
                )

                lora_path = gr.State(value=None)
                """Path of LoRA to load or loaded."""

                lora_name = gr.State(value=None)
                """Name of loaded LoRA."""

                show_adv_params_btn = gr.Button(
                    "",
                    icon=assets_dir / "ian-anandara" / "control.svg",
                    elem_id="show-adv-params-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("show-adv-params-btn")
                        btn.title = "{t("Control more params (CFG, Steps...)")}"
                    """
                )

                show_seed_btn = gr.Button(
                    "",
                    icon=assets_dir / "juicy-fish" / "dice.svg",
                    elem_id="show-seed-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("show-seed-btn")
                        btn.title = "{t("Use a specific or random seed")}"
                    """
                )

                change_out_folder_btn = gr.Button(
                    "",
                    icon=assets_dir / "freepik" / "folder.svg",
                    elem_id="change-output-folder-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("change-output-folder-btn")
                        btn.title = "{t("Change output folder")}"
                    """
                )
                change_out_folder_btn.click(
                    lambda: change_output_dir(t),
                )

                access_faq_btn = gr.Button(
                    "",
                    icon=assets_dir / "kerismaker" / "tech_13631866.svg",
                    elem_id="access-faq-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("access-faq-btn")
                        btn.title = "{t("Access the FAQ of this application")}"
                    """
                )
                access_faq_btn.click(
                    lambda: open_with_default_app(
                        f"{get_metadata('HOME_URL')}/blob/main/docs/FAQ.md"
                    ),
                )

                donate_btn = gr.Button(
                    "",
                    icon=assets_dir / "kofi_symbol.svg",
                    elem_id="donate-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        const btn = document.getElementById("donate-btn")
                        btn.title = "{t("Keep project developer awake with a coffee")} 😄"
                    """
                )
                donate_btn.click(
                    lambda: open_with_default_app(get_metadata("DONATE_URL")),
                )

            with gr.Column():
                with gr.Row():
                    model_select = gr.Dropdown(
                        container=False,
                        scale=2,
                        choices=[(t(m.name), m.id) for m in models],
                        value=initial_model.id,
                        filterable=False,
                        elem_id="model-select",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let select = document.getElementById("model-select")
                            select.title = "{t("To edit photos, select [klein] 4B")}"
                        """,
                    )
                    model_status = gr.HTML(
                        t("Model loaded"),
                        elem_id="model-status",
                    )

                trigger_words = gr.State(value=[None, None])
                """Trigger words (previous, current)."""

                with gr.Row():
                    mm_prompt = gr.MultimodalTextbox(
                        label=t("Prompt"),
                        lines=3,
                        max_plain_text_length=4096,
                        placeholder=t("Enter your prompt here..."),
                        html_attributes=gr.InputHTMLAttributes(spellcheck=False),
                        file_types=["image"],
                        submit_btn=False,
                        elem_id="prompt",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let zone = document.getElementById("prompt")
                            zone.title = "{t("Drag an image here to recover its prompt")}"
                        """,
                    )
                    mm_prompt.change(
                        extract_update_prompt,
                        inputs=mm_prompt,
                        outputs=mm_prompt,
                        show_progress="hidden",
                    )

                with gr.Row(
                    # A row hidden on startup is not mounted by Gradio,
                    # that's why we use the "hidden" class instead of the `visible` property.
                    elem_classes=(
                        [] if "image-to-image" in initial_model.features else ["hidden"]
                    )
                ) as reference_images_row:
                    reference_images = gr.MultimodalTextbox(
                        label=t("Reference Images"),
                        sources=["upload"],
                        file_count="multiple",
                        file_types=["image"],
                        max_plain_text_length=0,
                        submit_btn=False,
                        elem_id="reference-images",
                    )
                    gr.HTML(
                        visible="hidden",
                        elem_classes="hidden",
                        js_on_load=f"""
                            let zone = document.getElementById("reference-images")
                            zone.title = "{t("Drag an image here to add it as a reference")}"
                        """,
                    )

                with gr.Row() as ref_image_strength_row:
                    ref_image_strength = gr.Slider(
                        label=t("Reference Strength"),
                        minimum=0.1,
                        maximum=0.9,
                        step=0.1,
                        value=0.5,
                    )

                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        value=default_aspect_ratio,
                        choices=aspect_ratio_choices,
                        container=False,
                        elem_id="aspect-ratio",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let select = document.getElementById("aspect-ratio")
                            select.title = "{t("Aspect Ratio")}"
                        """,
                    )
                    resolution = gr.Dropdown(
                        value=default_resolution_choices[0],
                        choices=default_resolution_choices,
                        container=False,
                        elem_id="resolution",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            let select = document.getElementById("resolution")
                            select.title = "{t("Resolution")}"
                        """,
                    )
                    generate_btn = gr.Button(
                        t("Generate Image"),
                        variant="primary",
                    )

                # Start visible so Gradio mounts and lays out the slider at
                # load time, then gets collapsed on app load (see app.load
                # below). Otherwise the slider, first mounted while hidden,
                # stays invisible the first time the row is revealed.
                with gr.Row() as lora_row:
                    lora_strength = gr.Slider(
                        scale=2,
                        label=t("LoRA Strength"),
                        minimum=-2.5,
                        maximum=2.5,
                        step=0.1,
                        value=1.0,
                    )
                    lora_strength.change(
                        lambda strength: set_lora_strength(strength, image_pipe.instance),
                        inputs=lora_strength,
                    )
                    unload_lora_btn = gr.Button(t("Unload LoRA"))

                    # On "Unload LoRA" button click:
                    # - lock image model dropdown,
                    # - unload LoRA model,
                    # - release image model dropdown, unless it's owned,
                    # - remove trigger word from prompt,
                    # - empty trigger words history,
                    # - make LoRA row invisible,
                    # - forget name of loaded LoRA.
                    unload_lora_btn.click(
                        lambda: gr.update(interactive=False),
                        outputs=model_select,
                    ).then(
                        lambda: unload_lora(image_pipe.instance),
                    ).then(
                        lambda: gr.update(interactive=not BlockingTask.is_running),
                        outputs=model_select,
                    ).then(
                        remove_trigger_word,
                        inputs=[trigger_words, mm_prompt],
                        outputs=[trigger_words, mm_prompt],
                    ).then(
                        lambda: gr.update(visible=False),
                        outputs=lora_row,
                    ).then(
                        lambda: None,
                        outputs=lora_name,
                    )

                # When the LoRA button is clicked:
                # - lock image model dropdown,
                # - prompt for a LoRA file,
                # - validate the selection.
                lora_swap_validated = (
                    swap_lora_btn.click(
                        lambda: gr.update(interactive=False),
                        outputs=model_select,
                    )
                    .then(
                        lambda: select_lora_file(t),
                        outputs=lora_path,
                    )
                    .then(
                        lambda p: validate_lora_swap(p, t),
                        inputs=lora_path,
                    )
                )

                # If the LoRA selection was validated, shift trigger words
                # history, unload any LoRA model then load selected LoRA model.
                lora_swapped = lora_swap_validated.success(
                    lambda p, tw, m: [tw[1], swap_lora(p, m, t, image_pipe.instance)],
                    inputs=[lora_path, trigger_words, model],
                    outputs=trigger_words,
                )

                # On LoRA swap success:
                # - release image model dropdown, unless it's owned,
                # - update trigger word in prompt,
                # - make LoRA row visible,
                # - remember name of loaded LoRA.
                lora_swapped.success(
                    lambda: gr.update(interactive=not BlockingTask.is_running),
                    outputs=model_select,
                ).then(
                    update_trigger_word,
                    inputs=[trigger_words, mm_prompt],
                    outputs=mm_prompt,
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=lora_row,
                ).then(
                    extract_lora_name,
                    inputs=lora_path,
                    outputs=lora_name,
                ).then(
                    lambda: gr.Info(t("LoRA loaded"), duration=2),
                )

                # On cancelled/invalid LoRA selection:
                # - release image model dropdown, unless it's owned,
                # - forget selected path.
                lora_swap_validated.failure(
                    lambda: gr.update(interactive=not BlockingTask.is_running),
                    outputs=model_select,
                ).then(
                    lambda: None,
                    outputs=lora_path,
                )

                # On failed LoRA swap:
                # - unload any LoRA model,
                # - remove trigger word from prompt,
                # - empty trigger words history,
                # - release image model dropdown, unless it's owned,
                # - make LoRA row invisible,
                # - forget path and name of loaded LoRA.
                lora_swapped.failure(
                    lambda: unload_lora(image_pipe.instance),
                ).then(
                    remove_trigger_word,
                    inputs=[trigger_words, mm_prompt],
                    outputs=[trigger_words, mm_prompt],
                ).then(
                    lambda: gr.update(interactive=not BlockingTask.is_running),
                    outputs=model_select,
                ).then(
                    lambda: gr.update(visible=False),
                    outputs=lora_row,
                ).then(
                    lambda: None,
                    outputs=lora_path,
                ).then(
                    lambda: None,
                    outputs=lora_name,
                )

                with gr.Row(visible=False) as adv_params_row:
                    with gr.Column(min_width=160):
                        cfg = gr.Slider(
                            label=t("CFG"),
                            minimum=0.0,
                            maximum=10.0,
                            value=initial_model.default.cfg,
                            step=0.1,
                        )
                    with gr.Column(min_width=160):
                        steps = gr.Slider(
                            label=t("Steps"),
                            minimum=1,
                            maximum=50,
                            value=initial_model.default.steps,
                            step=1,
                        )

                show_adv_params_state = gr.State(value=False)

                def toggle_row(visibility):
                    visibility = not visibility
                    return visibility, gr.update(visible=visibility)

                show_adv_params_btn.click(
                    toggle_row,
                    inputs=show_adv_params_state,
                    outputs=[show_adv_params_state, adv_params_row],
                )

                with gr.Row(visible=False) as seed_row:
                    seed = gr.Number(label=t("Seed"), value=42, precision=0)
                    random_seed = gr.Checkbox(label=t("Random"), value=True)

                show_seed_state = gr.State(value=False)

                show_seed_btn.click(
                    toggle_row,
                    inputs=show_seed_state,
                    outputs=[show_seed_state, seed_row],
                )

                def validate_model_swap():
                    """Validate that a model swap can start, blocking otherwise.

                    Raises:
                        gr.Error: If a blocking task (e.g. a LoRA load) is running.
                    """
                    if BlockingTask.is_running:
                        raise gr.Error(str(BlockingTask.message), duration=6)

                # When a new image model is selected, validate the swap.
                model_swap_validated = model_select.change(
                    validate_model_swap,
                    show_progress="hidden",
                )

                # On rejected model swap, resync the dropdown with the model
                # actually loaded, otherwise it keeps showing the rejected one.
                model_swap_validated.failure(
                    lambda loaded_model: gr.update(value=loaded_model.id),
                    inputs=model,
                    outputs=model_select,
                    show_progress="hidden",
                )

                # If the swap was validated:
                # - lock model dropdown,
                # - unload LoRA model,
                # - remove trigger word from prompt,
                # - empty trigger words history,
                # - make LoRA row invisible,
                # - forget name of loaded LoRA,
                # - download selected model...
                model_download = (
                    model_swap_validated.success(
                        lambda: gr.update(interactive=False),
                        outputs=model_select,
                        show_progress="hidden",
                    )
                    .then(lambda: unload_lora(image_pipe.instance))
                    .then(
                        remove_trigger_word,
                        inputs=[trigger_words, mm_prompt],
                        outputs=[trigger_words, mm_prompt],
                        show_progress="hidden",
                    )
                    .then(
                        lambda: gr.update(visible=False),
                        outputs=lora_row,
                    )
                    .then(
                        lambda: None,
                        outputs=lora_name,
                    )
                    .then(
                        lambda: gr.update(value=t("Downloading...")),
                        outputs=model_status,
                        show_progress="hidden",
                    )
                    .then(
                        lambda model_id: fetch_model(find_model(model_id, models), t),
                        inputs=model_select,
                    )
                )

                # On model download success: load model...
                model_load = model_download.success(
                    lambda: gr.update(
                        value=f"""
                            <span class='text'>{t("Loading")}</span>
                            <span class='pac-loader'/>
                        """
                    ),
                    outputs=model_status,
                    show_progress="hidden",
                ).then(
                    lambda model_id: image_pipe.swap(find_model(model_id, models), t),
                    inputs=model_select,
                    outputs=model,
                    show_progress="hidden",
                )

                # On failed model download or load:
                # - reselect initial model,
                # - release model dropdown.
                for model_phase in (model_download, model_load):
                    model_phase.failure(
                        lambda: gr.Info(f"{t('Fallback to')} {initial_model.name}.")
                    ).then(
                        lambda: gr.update(
                            value=initial_model.id,  # This triggers a change.
                            interactive=True,
                        ),
                        outputs=model_select,
                    )

                # On model load success:
                # - display reference images block and settings if model supports them,
                # - update other settings according to model,
                # - release model dropdown.
                model_load.success(
                    lambda image_model: (
                        gr.update(
                            elem_classes=(
                                []
                                if "image-to-image" in image_model.features
                                else ["hidden"]
                            )
                        ),
                        gr.update(visible=image_pipe.supports_strength()),
                        gr.update(value=image_model.default.steps),
                        gr.update(value=image_model.default.cfg),
                    ),
                    inputs=model,
                    outputs=[reference_images_row, ref_image_strength_row, steps, cfg],
                    show_progress="hidden",
                ).then(
                    lambda: gr.update(interactive=True),
                    outputs=model_select,
                    show_progress="hidden",
                ).then(
                    lambda: gr.update(value=t("Model loaded")),
                    outputs=model_status,
                    show_progress="hidden",
                )

                with gr.Column() as examples_column:
                    gr.Examples(
                        examples=get_example_prompts(
                            app_dir / "data" / "example_prompts.json"
                        ),
                        inputs=mm_prompt,
                        label=t("Example Prompts"),
                        elem_id="example-prompts",
                    )

                reuse_prompt_btn = gr.Button(value=t("Reuse"))

                with gr.Row():
                    used_prompt = gr.Textbox(
                        visible="hidden",
                        label=t("Displayed Image Prompt"),
                        max_lines=6,
                        buttons=["copy", reuse_prompt_btn],
                        interactive=False,
                        elem_id="used-prompt",
                    )

                reuse_prompt_btn.click(
                    lambda p: gr.update(value=p),
                    inputs=used_prompt,
                    outputs=mm_prompt,
                    show_progress="hidden",
                )

            with gr.Column(scale=2):
                gallery_images = gr.Gallery(
                    label=t("Generated Images"),
                    object_fit="contain",
                    format="png",
                    type="filepath",
                    buttons=["fullscreen"],
                    interactive=False,
                    elem_id="gallery",
                )
                output_images_paths = gr.State(value={})
                """Output images paths indexed by image ID."""

                selected_image_index = gr.State(value=None)
                """Index of image to select or selected in gallery."""

                def get_selected_image_index(gallery_image: gr.SelectData) -> int:
                    return gallery_image.index

                gallery_images.select(
                    get_selected_image_index, outputs=selected_image_index
                ).success(
                    sync_used_prompt,
                    inputs=[gallery_images, selected_image_index],
                    outputs=used_prompt,
                    show_progress="hidden",
                )

                # Prevent grid display.
                gallery_images.preview_close(
                    lambda idx: gr.update(selected_index=idx),
                    inputs=selected_image_index,
                    outputs=gallery_images,
                    show_progress="hidden",
                )

                with gr.Row():
                    delete_output_image_btn = gr.Button(
                        t("Delete Image"),
                        visible="hidden",
                        elem_id="delete-output-image-btn",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            const btn = document.getElementById("delete-output-image-btn")
                            btn.title = "{t("file included")}"
                        """,
                    )

                    # Rapid clicks can cause desync with gallery.
                    # To prevent this, we disable the button during image deletion.
                    image_deletion = delete_output_image_btn.click(
                        lambda: gr.update(interactive=False),
                        outputs=delete_output_image_btn,
                    ).then(
                        delete_image,
                        inputs=[
                            gallery_images,
                            selected_image_index,
                            output_images_paths,
                        ],
                        outputs=[
                            gallery_images,
                            selected_image_index,
                            output_images_paths,
                        ],
                    )

                    image_deletion.success(
                        sync_used_prompt,
                        inputs=[gallery_images, selected_image_index],
                        outputs=used_prompt,
                        show_progress="hidden",
                    )

                    image_deletion.then(
                        lambda: gr.update(interactive=True),
                        outputs=delete_output_image_btn,
                    )

                    gallery_images.change(
                        lambda images: gr.update(visible=bool(images)),
                        inputs=gallery_images,
                        outputs=delete_output_image_btn,
                    )

                    open_output_folder_btn = gr.Button(
                        t("Open Output Folder"),
                        variant="primary",
                        elem_id="open-output-folder-btn",
                    )
                    gr.HTML(
                        visible="hidden",
                        js_on_load=f"""
                            const btn = document.getElementById("open-output-folder-btn")
                            btn.title = "{t("of generated images")}"
                        """,
                    )

                    def create_open_output_dir():
                        output_dir.mkdir(parents=True, exist_ok=True)
                        open_with_default_app(output_dir)

                    open_output_folder_btn.click(create_open_output_dir)

        with gr.Row():
            # Add a credits link to footer, after Gradio credit.
            gr.HTML(
                visible="hidden",
                js_on_load=f"""
                    document.querySelector("footer").insertAdjacentHTML(
                        "beforeend",
                        `<a
                            href="{get_metadata("HOME_URL")}#credits"
                            target="_blank"
                        >
                            {t("See all credits")}
                        </a>`
                    )
                """,
            )

        with gr.Row(
            visible=not tou.accepted(),
            elem_id="tou-row",
        ) as tou_row:
            with gr.Column(elem_id="tou-card"):
                gr.Markdown(f"### {t('Terms of Use')}")
                gr.Markdown(t(get_metadata("TERMS_OF_USE")))
                agree_tou_btn = gr.Button(
                    t("I agree"),
                    variant="primary",
                )

                agree_tou_btn.click(tou.accept).then(
                    lambda: (gr.update(visible=False), gr.update(elem_classes=[])),
                    outputs=[tou_row, ui_row],
                )

        def update_resolution_choices(_aspect_ratio):
            resolution_choices = resolutions_by_aspect.get(
                _aspect_ratio, default_resolution_choices
            )
            return gr.update(value=resolution_choices[0], choices=resolution_choices)

        aspect_ratio.change(
            update_resolution_choices,
            inputs=aspect_ratio,
            outputs=resolution,
            show_progress="hidden",
        )

        def validate_generation():
            """Validate that generation can start, blocking otherwise.

            Raises:
                gr.Error: If a blocking task (e.g. a model load) is running.
            """
            if BlockingTask.is_running:
                raise gr.Error(str(BlockingTask.message), duration=6)

        # On "Generate Image" button click:
        generation_validated = generate_btn.click(validate_generation)

        # If generation was validated:
        # - lock model dropdown,
        # - hide "Used Prompt" a.k.a "Displayed Image Prompt" block,
        # - hide "Delete Image" button (it's shown later, see gallery_images.change).
        ui_setup_for_generation = generation_validated.success(
            lambda: (
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[model_select, used_prompt, delete_output_image_btn],
        )

        # Once UI is setup: generate image.
        generation = ui_setup_for_generation.success(
            partial(generate, image_pipe, output_dir, t),
            inputs=[
                model,
                mm_prompt,
                reference_images,
                ref_image_strength,
                resolution,
                seed,
                random_seed,
                steps,
                cfg,
                gallery_images,
                output_images_paths,
                lora_name,
            ],
            outputs=[
                gallery_images,
                selected_image_index,
                output_images_paths,
                seed,
            ],
            show_progress_on=gallery_images,
        )

        # On generation success:
        # - release model dropdown,
        # - select (preview) generated image in gallery,
        # - sync used prompt,
        # - make example prompts invisible.
        generation.success(
            lambda: gr.update(interactive=True),
            outputs=model_select,
        ).then(
            lambda idx: gr.update(selected_index=idx),  # See gallery_images.select
            inputs=selected_image_index,
            outputs=gallery_images,
        ).then(
            # Sync explicitly rather than relying on the gallery .select event.
            # Reselecting the same index after a deletion wouldn't fire .select.
            sync_used_prompt,
            inputs=[gallery_images, selected_image_index],
            outputs=used_prompt,
            show_progress="hidden",
        ).then(
            lambda: gr.update(visible=False),
            outputs=examples_column,
        )

        # On failed generation: release the model dropdown, unless another
        # blocking task (e.g. a model download) is still running and "owns" it.
        generation.failure(
            lambda: gr.update(interactive=not BlockingTask.is_running),
            outputs=model_select,
        )

        app.load(
            lambda: ImagePipeline.warn_if_not_optimized(t, get_metadata("NAME"))
        )

        app.load(
            lambda: check_for_updates(
                get_metadata("VERSION"),
                get_metadata("VERSION_URL"),
                f"{get_metadata('HOME_URL')}/releases",
                t,
            )
        )

        # Collapse the LoRA settings row, which starts visible only so its
        # slider mounts and lays out at load time (see lora_row above).
        app.load(lambda: gr.update(visible=False), outputs=lora_row)

        # Same for the reference image strength slider.
        app.load(
            lambda: gr.update(visible=image_pipe.supports_strength()),
            outputs=ref_image_strength_row,
        )

    app.launch(
        server_port=args.port,
        inbrowser=args.in_browser,
        favicon_path=assets_dir / "favicon_180.png",
        theme=get_theme(),
        footer_links=["gradio"],  # Credit
        css_paths=[
            app_dir / "source" / "css" / "pac-loader.css",
            app_dir / "source" / "css" / "app.css",
        ],
        js=(app_dir / "source" / "app.js").read_text(),
        allowed_paths=[output_dir],
    )
