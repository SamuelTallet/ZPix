"""ZPix Gradio app."""

# Based on https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo
import logging
from argparse import ArgumentParser
from json import load as load_json
from os import environ
from pathlib import Path
from random import randint
from shutil import rmtree
from time import time_ns

import gradio as gr
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.guiders import ClassifierFreeGuidance
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Force tensorwise FP8 matmul kernels as a fallback on hardware that lacks
# native row-wise FP8 support, such as consumer NVIDIA Blackwell cards.
# This must run before importing SDNQ, which reads the variable at import time.
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0):
    environ.setdefault("SDNQ_USE_TENSORWISE_FP8_MM", "1")

from sdnq import SDNQConfig  # noqa: E402, F401
from sdnq.common import use_torch_compile as triton_is_available  # noqa: E402
from sdnq.loader import apply_sdnq_options_to_model  # noqa: E402

from source.py.disclaimer import TERMS_OF_USE, TermsOfUse
from source.py.ex_prompts import get_example_prompts
from source.py.gallery_images import delete_image
from source.py.image_model import ImageModel
from source.py.image_models import download_model, find_model, get_models
from source.py.lora_model import LoraModel
from source.py.os_abstract import open_with_default_app
from source.py.output_dir import change_output_dir, get_output_dir
from source.py.prompt_extract import extract_update_prompt
from source.py.resolutions import get_aspects_and_resolutions, parse_resolution
from source.py.trigger_word import remove_trigger_word, update_trigger_word
from source.py.update_check import check_for_updates
from source.py.used_prompt import sync_used_prompt

logging.basicConfig(format="%(levelname)s: %(message)s")

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

translation: dict[str, str] = {}
"""Translation."""

metadata: dict[str, str] = {}
"""App metadata."""

models: list[ImageModel] = []
"""Available image models."""

pipe: ModularPipeline | None = None
"""Pipeline."""

pipe_is_optimized: bool = False
"""Pipeline is optimized?"""

pipe_is_busy: bool = False
"""Pipeline is busy? e.g. loading a LoRA."""

output_dir = get_output_dir()
"""The folder where ZPix saves generated images."""


def load_translation(locale: str) -> None:
    """Load translation for a given locale, if available."""
    global translation

    translation_file = app_dir / "translations" / f"{locale}.json"
    if not translation_file.exists():
        logging.warning(f"Translation for {locale} not found.")
        return

    with open(translation_file, "r", encoding="utf-8") as file:
        translation = load_json(file)


def t(string: str) -> str:
    """Translate a string."""
    return translation.get(string, string)


def get_metadata(filename: str) -> str:
    """Get metadata."""
    if filename not in metadata:
        file = app_dir / "metadata" / filename
        metadata[filename] = file.read_text()

    return metadata[filename]


def get_theme():
    """Get customized theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f7f6ff",
            c100="#efedff",
            c200="#d8d2ff",
            c300="#c0b7ff",
            c400="#a192ff",
            c500="#624aff",
            c600="#5843e6",
            c700="#4534b3",
            c800="#312580",
            c900="#1d164d",
            c950="#0a071a",
        )
    )


def on_app_load():
    """On app load."""
    if not pipe_is_optimized:
        gr.Warning(
            t(
                "Image generation may be slow because diffusion pipeline is not optimized."
            )
            + "<br>"
            + t(
                "Try upgrading your graphics card drivers, then reboot your PC and restart"
            )
            + f" {get_metadata('NAME')}.",
            duration=None,  # Until user closes it.
        )

    check_for_updates(
        get_metadata("VERSION"),
        get_metadata("VERSION_URL"),
        f"{get_metadata('HOME_URL')}/releases",
        t,
    )


def load_model(model: ImageModel) -> ImageModel:
    """Load an image model pipeline."""
    global pipe
    global pipe_is_optimized

    # A fresh manager per load avoids accumulating components across model swaps.
    manager = ComponentsManager()

    try:
        pipe = ModularPipeline.from_pretrained(
            model.id,
            components_manager=manager,
        )
        pipe.load_components(torch_dtype=torch.bfloat16)
    except Exception:
        if model.backup_id:
            logging.warning(
                f"Can't load {model.id}, falling back to {model.backup_id}."
            )
            pipe = ModularPipeline.from_pretrained(
                model.backup_id,
                components_manager=manager,
            )
            pipe.load_components(torch_dtype=torch.bfloat16)
        else:
            raise

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, use_quantized_matmul=True
        )
        pipe.text_encoder = apply_sdnq_options_to_model(
            pipe.text_encoder, use_quantized_matmul=True
        )
        # Not all models are compatible with FlashAttention:
        if model.family in ("Z-Image", "FLUX.2"):
            try:
                pipe.transformer.set_attention_backend("flash")
                pipe_is_optimized = True
            except Exception as e:
                pipe.transformer.reset_attention_backend()
                logging.warning(f"FlashAttention is not available: {e}")
        else:
            pipe.transformer.set_attention_backend("native")

    try:
        pipe.vae.to(memory_format=torch.channels_last)
    except RuntimeError as e:
        logging.warning(f"Can't apply memory format optimization: {e}")

    manager.enable_auto_cpu_offload()

    return model


def swap_model(model: ImageModel) -> ImageModel:
    """Swap an image model pipeline."""
    global pipe_is_busy

    pipe_is_busy = True
    try:
        return load_model(model)
    finally:
        pipe_is_busy = False


def swap_lora(path: str, image_model: ImageModel) -> str | None:
    """Swap or load a new LoRA model.

    Args:
        path: Path to a LoRA file.
        image_model: Loaded image model.

    Returns:
        Trigger word of LoRA model.
    """
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    if not path.endswith(".safetensors"):
        raise gr.Error(
            t("LoRA file extension must be .safetensors"),
            duration=20,
        )

    lora = LoraModel(path)

    try:
        if lora.base_model() not in image_model.base_ids:
            gr.Warning(
                f"{t('This LoRA seems incompatible with')} {image_model.name}.<br>"
                f"{t('It might not work.')}",
                duration=5,
            )
    except Exception as e:
        logging.warning(f"Can't check LoRA compatibility: {e}")

    bfloat16_lora = lora.to_bf16()

    # Workaround: Diffusers FLUX.2 LoRA converter doesn't handle .alpha keys.
    if image_model.family == "FLUX.2":
        bfloat16_lora = {
            k: v for k, v in bfloat16_lora.items() if not k.endswith(".alpha")
        }

    pipe_is_busy = True

    try:
        pipe.unload_lora_weights()
        pipe.load_lora_weights(bfloat16_lora, adapter_name="lora_1")
    except Exception as error:
        raise gr.Error(
            t("Failed to load LoRA, you may need to restart application."),
            duration=None,
        ) from error
    finally:
        pipe_is_busy = False

    trigger_word = lora.trigger_word()

    return trigger_word


def set_lora_strength(strength: float):
    """Set LoRA strength."""
    adapters = pipe.get_list_adapters()

    if "transformer" not in adapters or "lora_1" not in adapters["transformer"]:
        raise gr.Error("No LoRA loaded.")

    pipe.set_adapters("lora_1", strength)


def unload_lora():
    """Unload LoRA model."""
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    try:
        pipe_is_busy = True
        pipe.unload_lora_weights()
    finally:
        pipe_is_busy = False


def generate(
    model: ImageModel,
    mm_prompt: dict | None,
    reference_images: dict | None,
    resolution: str,
    seed: int,
    random_seed: bool,
    steps: int,
    cfg: float,
    gallery_images: list[tuple] | None,
    images_paths: dict[str, str],
    lora_name: str | None,
) -> tuple[list[tuple], int, dict[str, str], int]:
    """Generate an image and possibly a seed, and update gallery.

    Args:
        model: Loaded image model.
        mm_prompt: Multimodal dictionary containing possibly a text prompt.
        reference_images: List of reference images.
        resolution: Resolution string (e.g. "1024x1024").
        seed: Seed value for reproducibility.
        random_seed: Ignore seed argument and generate a seed?
        steps: Number of inference (denoising) steps.
        cfg: Classifier-free guidance scale.
        gallery_images: Existing gallery images to append to.
        images_paths: Dictionary mapping images IDs to output paths.
        lora_name: Name of loaded LoRA (e.g. "Anime_20").
    Returns:
        Tuple of (updated gallery, last image index, output paths, used seed).

    Raises:
        gr.Error: If the pipeline is not loaded or busy.
    """
    global pipe_is_busy

    if pipe is None:
        raise gr.Error("Pipeline not loaded.")

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    prompt: str = (mm_prompt or {}).get("text", "").strip()

    if not prompt:
        raise gr.Error(
            t("Please enter a prompt to generate an image."),
            duration=4,
        )

    width, height = parse_resolution(resolution)
    used_seed = randint(1, 1000000) if random_seed else int(seed)

    if "guider" in pipe.component_names:
        pipe.update_components(
            # Clamp CFG to >= 1.0: below 1.0 the guider pulls predictions toward
            # the unconditional, weakening prompt adherence (ignored at 0.0).
            guider=ClassifierFreeGuidance(guidance_scale=max(float(cfg), 1.0))
        )

    pipe_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": int(steps),
        "generator": torch.manual_seed(used_seed),
    }

    if (
        reference_images
        and reference_images.get("files")
        and "image-to-image" in model.features
    ):
        pipe_kwargs["image"] = [Image.open(f) for f in reference_images["files"]]

    try:
        pipe_is_busy = True
        image = pipe(**pipe_kwargs).images[0]
    except UnicodeDecodeError:
        # A corrupted Triton cache can cause an UnicodeDecodeError.
        rmtree(Path.home() / ".triton", ignore_errors=True)
        gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

        gr.Info(t("Regenerating same image..."), duration=8)
        image = pipe(**pipe_kwargs).images[0]
    finally:
        pipe_is_busy = False

    # Prepare metadata to be saved in PNG text chunks.
    image_metadata = PngInfo()
    image_metadata.add_text("model", model.id)
    image_metadata.add_itxt("prompt", prompt)
    image_metadata.add_text("seed", str(used_seed))
    image_metadata.add_text("steps", str(steps))
    image_metadata.add_text("cfg", str(cfg))

    # Milliseconds precision is more than enough to avoid filename collision.
    image_id = str(time_ns() // 1_000_000)
    image_basename = f"image_{image_id}.png"

    # LoRA name (if provided) is included in output path.
    if lora_name:
        image_file = output_dir / lora_name / image_basename
    else:
        image_file = output_dir / image_basename

    # Ensure output directory exists.
    image_file.parent.mkdir(parents=True, exist_ok=True)

    image.save(image_file, pnginfo=image_metadata)

    # Output path is recorded for a possible later deletion.
    images_paths[image_id] = str(image_file)

    if gallery_images is None:
        gallery_images = []

    # Prompt is added as image caption.
    gallery_images.append((image_file, prompt))

    return gallery_images, len(gallery_images) - 1, images_paths, used_seed


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--port", type=int, required=True)
    arg_parser.add_argument("--locale", type=str, required=False, default="en-US")
    args, _ = arg_parser.parse_known_args()

    with gr.Blocks(
        fill_width=True,
        analytics_enabled=False,
    ) as app:
        models = get_models(app_dir / "data" / "curated_models.json")
        initial_model = load_model(models[0])

        model = gr.State(value=initial_model)
        """Loaded image model."""

        (
            resolutions_by_aspect,
            default_resolution_choices,
            aspect_ratio_choices,
            default_aspect_ratio,
        ) = get_aspects_and_resolutions()

        if args.locale != "en-US":
            load_translation(args.locale)

        tou = TermsOfUse(app_dir / ".tou_accepted")

        with gr.Row(elem_classes=[] if tou.accepted() else ["blurred"]) as ui_row:
            with gr.Column(min_width=48, elem_classes=["sidebar"]):
                gr.Button(
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
                lora_path = gr.Textbox(
                    visible="hidden",  # See "portal" in app.js
                    elem_id="lora-path",
                )
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
                    visible="image-to-image" in initial_model.features
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
                        elem_classes="always-hidden",
                        js_on_load=f"""
                            let zone = document.getElementById("reference-images")
                            zone.title = "{t("Drag an image here to add it as a reference")}"
                        """,
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

                with gr.Row(visible=False) as lora_row:
                    lora_strength = gr.Slider(
                        scale=2,
                        label=t("LoRA Strength"),
                        minimum=-2.5,
                        maximum=2.5,
                        step=0.1,
                        value=1.0,
                    )
                    lora_strength.change(
                        set_lora_strength,
                        inputs=lora_strength,
                    )
                    unload_lora_btn = gr.Button(t("Unload LoRA"))

                    # On "Unload LoRA" button click:
                    # - unload LoRA model,
                    # - remove trigger word from prompt,
                    # - empty trigger words history,
                    # - make LoRA row invisible,
                    # - forget name of loaded LoRA.
                    unload_lora_btn.click(
                        lambda: gr.update(interactive=False),
                        outputs=model_select,
                    ).then(
                        unload_lora,
                    ).then(
                        lambda: gr.update(interactive=True),
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

                # When a LoRA path is selected:
                # - lock image model dropdown,
                # - discard appended timestamp,
                # - shift trigger words history,
                # - unload any LoRA model,
                # - load selected LoRA model.
                lora_swap = lora_path.change(
                    lambda: gr.update(interactive=False),
                    outputs=model_select,
                ).then(
                    lambda p, tw, m: [tw[1], swap_lora(p, m)],
                    inputs=[lora_path, trigger_words, model],
                    outputs=trigger_words,
                    js="(p, tw, m) => [p.split('|')[0], tw, m]",
                )

                # On LoRA swap success:
                # - release image model dropdown,
                # - update trigger word in prompt,
                # - make LoRA row visible,
                # - remember name of loaded LoRA.
                lora_swap.success(
                    lambda: gr.update(interactive=True),
                    outputs=model_select,
                ).then(
                    update_trigger_word,
                    inputs=[trigger_words, mm_prompt],
                    outputs=mm_prompt,
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=lora_row,
                ).then(
                    lambda p: Path(p).stem,
                    inputs=lora_path,
                    outputs=lora_name,
                ).then(
                    lambda: gr.Info(t("LoRA loaded"), duration=2),
                )

                # On LoRA swap failure:
                # - release image model dropdown,
                # - unload any LoRA model,
                # - remove trigger word from prompt,
                # - empty trigger words history,
                # - make LoRA row invisible,
                # - forget name of loaded LoRA.
                lora_swap.failure(
                    lambda: gr.update(interactive=True),
                    outputs=model_select,
                ).then(unload_lora).then(
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

                # When a new image model is selected:
                # - lock model dropdown,
                # - unload LoRA model,
                # - remove trigger word from prompt,
                # - empty trigger words history,
                # - make LoRA row invisible,
                # - forget name of loaded LoRA,
                # - download selected model...
                model_download = (
                    model_select.change(
                        lambda: gr.update(interactive=False),
                        outputs=model_select,
                        show_progress="hidden",
                    )
                    .then(unload_lora)
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
                        lambda model_id: download_model(
                            find_model(model_id, models), t
                        ),
                        inputs=model_select,
                    )
                )

                # On model download failure:
                # - reselect initial model,
                # - release model dropdown.
                model_download.failure(
                    lambda: gr.Info(f"{t('Fallback to')} {initial_model.name}.")
                ).then(
                    lambda: gr.update(
                        value=initial_model.id,  # This triggers a change.
                        interactive=True,
                    ),
                    outputs=model_select,
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
                    lambda model_id: swap_model(find_model(model_id, models)),
                    inputs=model_select,
                    outputs=model,
                    show_progress="hidden",
                )

                # On model load success:
                # - display reference images if model supports them,
                # - update settings according to model,
                # - release model dropdown.
                model_load.success(
                    lambda image_model: (
                        gr.update(visible="image-to-image" in image_model.features),
                        gr.update(value=image_model.default.steps),
                        gr.update(value=image_model.default.cfg),
                    ),
                    inputs=model,
                    outputs=[reference_images_row, steps, cfg],
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
                gr.Markdown(t(TERMS_OF_USE))
                agree_tou_btn = gr.Button(t("I agree"), variant="primary")

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

        # Before generation starts:
        # - lock model dropdown,
        # - hide "Used Prompt" a.k.a "Displayed Image Prompt" block,
        # - hide "Delete Image" button (it's shown later, see gallery_images.change).
        generation = generate_btn.click(
            lambda: (
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            outputs=[
                model_select,
                used_prompt,
                delete_output_image_btn,
            ],
        ).then(
            generate,
            inputs=[
                model,
                mm_prompt,
                reference_images,
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
        # - make example prompts invisible.
        generation.success(
            lambda: gr.update(interactive=True),
            outputs=model_select,
        ).then(
            lambda idx: gr.update(selected_index=idx),  # See gallery_images.select
            inputs=selected_image_index,
            outputs=gallery_images,
        ).then(
            lambda: gr.update(visible=False),
            outputs=examples_column,
        )

        # On generation failure: release model dropdown.
        generation.failure(
            lambda: gr.update(interactive=True),
            outputs=model_select,
        )

        app.load(on_app_load)

    app.launch(
        server_port=args.port,
        footer_links=["gradio"],  # Credit
        theme=get_theme(),
        css_paths=[
            app_dir / "source" / "css" / "pac-loader.css",
            app_dir / "source" / "css" / "app.css",
        ],
        js=(app_dir / "source" / "app.js").read_text(),
        allowed_paths=[output_dir],
    )
