"""Z-Image Turbo Gradio app."""
# Based on https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo

import logging
from argparse import ArgumentParser
from json import load as load_json
from os import environ
from pathlib import Path
from random import randint
from re import search
from shutil import rmtree

import gradio as gr
from diffusers import ZImagePipeline
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from torch import bfloat16, cuda, manual_seed, xpu

logging.basicConfig(format="%(levelname)s: %(message)s")

# Path to Triton cache directory
# shortened by good measure to avoid too long path errors on Windows
# even if this has been fixed recently.
environ["TRITON_CACHE_DIR"] = str(Path.home() / ".triton")

app_dir = Path(__file__).parent
"""App directory."""

# We store temp files created by Gradio in app directory
# to ease visualization of disk space used by this app.
environ["GRADIO_TEMP_DIR"] = str(app_dir / "temp" / "GradioApp")

translation: dict[str, str] = {}
"""Translation."""

metadata: dict[str, str] = {}
"""Metadata."""

pipe: ZImagePipeline | None = None
"""Pipeline."""

optimized: bool = False
"""Pipeline is optimized?"""


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


def get_example_prompts() -> list[str]:
    """Get example prompts."""
    prompts_file = app_dir / "examples" / "prompts.json"

    with open(prompts_file, "r", encoding="utf-8") as file:
        prompts = load_json(file)

    return [prompt["text"] for prompt in prompts]


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
    if not optimized:
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


def get_aspects_and_resolutions():
    """Get aspect ratios and resolutions,
    possibly translated.

    Returns:
        Tuple of (
            resolutions by aspect,
            default resolution choices,
            aspect ratio choices,
            default aspect ratio
        )
    """
    default_aspect_ratio = "{} (16:9)".format(t("Landscape"))

    resolutions_by_aspect = {
        "{} (1:1)".format(t("Square")): [
            "1024x1024",
            "1280x1280",
            "1440x1440",
        ],
        "{} (16:9)".format(t("Landscape")): [
            "1280x720",
            "1920x1088",
        ],
        "{} (9:16)".format(t("Portrait")): [
            "720x1280",
            "1088x1920",
        ],
        "{} (4:3)".format(t("Landscape")): [
            "1152x864",
            "1440x1088",
            "1920x1440",
        ],
        "{} (3:4)".format(t("Portrait")): [
            "864x1152",
            "1088x1440",
            "1440x1920",
        ],
        "{} (16:10)".format(t("Landscape")): [
            "1280x800",
            "1440x912",
            "1920x1200",
        ],
        "{} (10:16)".format(t("Portrait")): [
            "800x1280",
            "912x1440",
            "1200x1920",
        ],
        "{} (21:9)".format(t("Ultra Wide")): [
            "1344x576",
        ],
    }

    default_resolution_choices = resolutions_by_aspect[default_aspect_ratio]
    aspect_ratio_choices = list(resolutions_by_aspect.keys())

    return (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    )


def parse_resolution(resolution):
    """Parse resolution string into width and height.

    Args:
        resolution: Resolution string in format "WIDTHxHEIGHT" or "WIDTH×HEIGHT".

    Returns:
        Tuple of (width, height) as integers. Defaults to (1024, 1024) if parsing fails.
    """
    match = search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


def load_model(model: str, backup_model: str):
    """Load and configure the Z-Image pipeline.

    Args:
        model: Hugging Face (HF) model name.
        backup_model: HF backup model name.
    """
    global pipe
    global optimized

    try:
        pipe = ZImagePipeline.from_pretrained(
            model,
            torch_dtype=bfloat16,
        )
    except Exception:
        logging.warning(f"Can't load {model}, falling back to {backup_model}.")
        pipe = ZImagePipeline.from_pretrained(
            backup_model,
            torch_dtype=bfloat16,
        )

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (cuda.is_available() or xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, use_quantized_matmul=True
        )
        pipe.text_encoder = apply_sdnq_options_to_model(
            pipe.text_encoder, use_quantized_matmul=True
        )
        try:
            pipe.transformer.set_attention_backend("_sage_qk_int8_pv_fp16_triton")
            optimized = True
        except Exception as e:
            logging.warning(f"SageAttention is not available: {e}")

    pipe.enable_model_cpu_offload()


def generate_image(
    pipe,
    prompt,
    resolution="1024x1024",
    seed=42,
    num_inference_steps=8,
):
    """Generate an image using the Z-Image pipeline.

    Args:
        pipe: The loaded ZImagePipeline instance.
        prompt: Text prompt describing the desired image.
        resolution: Output resolution as "WIDTHxHEIGHT" string.
        seed: Random seed for reproducible generation.
        num_inference_steps: Number of denoising steps.

    Returns:
        Generated PIL Image.
    """
    width, height = parse_resolution(resolution)

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        generator=manual_seed(seed),
    ).images[0]

    return image


def generate(
    prompt,
    resolution="1024x1024",
    seed=42,
    steps=8,
    random_seed=True,
    gallery_images=None,
):
    """Gradio callback to generate an image and update the gallery.

    Args:
        prompt: Text prompt for image generation.
        resolution: Resolution string (e.g. "1024x1024").
        seed: Seed value for reproducibility.
        steps: Number of inference (denoising) steps.
        random_seed: If True, generate a random seed ignoring the seed parameter.
        gallery_images: Existing gallery images to append to.

    Returns:
        Tuple of (updated gallery, last image index, seed as str, seed as int).

    Raises:
        gr.Error: If the pipeline is not loaded.
    """

    if pipe is None:
        raise gr.Error("Pipeline not loaded.")

    if random_seed:
        new_seed = randint(1, 1000000)
    else:
        new_seed = int(seed) if seed != -1 else randint(1, 1000000)

    generation_args = {
        "pipe": pipe,
        "prompt": prompt,
        "resolution": resolution,
        "seed": new_seed,
        "num_inference_steps": int(steps + 1),
    }
    try:
        image = generate_image(**generation_args)
    except UnicodeDecodeError:
        # A corrupted Triton cache can cause an UnicodeDecodeError.
        rmtree(Path.home() / ".triton", ignore_errors=True)
        gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

        gr.Info(t("Regenerating same image..."), duration=8)
        image = generate_image(**generation_args)

    if gallery_images is None:
        gallery_images = []

    # Prompt is added as image caption.
    gallery_images.append((image, prompt))

    return gallery_images, len(gallery_images) - 1, str(new_seed), int(new_seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backup-model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--locale", type=str, required=False, default="en-US")
    args, _ = parser.parse_known_args()

    load_model(args.model, args.backup_model)

    if args.locale != "en-US":
        load_translation(args.locale)

    (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    ) = get_aspects_and_resolutions()

    with gr.Blocks(
        analytics_enabled=False,
    ) as app:
        with gr.Row():
            with gr.Column(min_width=48, elem_classes=["sidebar"]):
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "noto-emoji" / "emoji_u26a1.svg",
                    link=get_metadata("HOME_URL"),
                    link_target="_blank",  # Opens default browser. See app.js
                    elem_id="home-button",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("home-button")
                        btn.title = "{t("Visit project homepage to check updates")}"
                    """
                )
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "kerismaker" / "tech_13631866.png",
                    link=f"{get_metadata('HOME_URL')}/blob/main/docs/FAQ.md",
                    link_target="_blank",
                    elem_id="faq-button",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("faq-button")
                        btn.title = "{t("Access the FAQ of this application")}"
                    """
                )
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "kofi_symbol.svg",
                    link=get_metadata("DONATE_URL"),
                    link_target="_blank",
                    elem_id="donate-button",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("donate-button")
                        btn.title = "{t("Keep project developer awake with a coffee")} 😄"
                    """
                )

            with gr.Column():
                prompt = gr.Textbox(
                    label=t("Prompt"),
                    lines=3,
                    placeholder=t("Enter your prompt here..."),
                    html_attributes={"spellcheck": "false"},
                )

                with gr.Row():
                    with gr.Column():
                        aspect_ratio = gr.Dropdown(
                            value=default_aspect_ratio,
                            choices=aspect_ratio_choices,
                            label=t("Aspect Ratio"),
                        )
                    with gr.Column():
                        resolution = gr.Dropdown(
                            value=default_resolution_choices[0],
                            choices=default_resolution_choices,
                            label=t("Resolution"),
                        )

                with gr.Row():
                    with gr.Column():
                        advanced_checkbox = gr.Checkbox(
                            label=t("Advanced Settings"), value=False
                        )
                    with gr.Column():
                        generate_btn = gr.Button(t("Generate Image"), variant="primary")

                with gr.Row(visible=False) as seed_random_row:
                    seed = gr.Number(label=t("Seed"), value=42, precision=0)
                    random_seed = gr.Checkbox(label=t("Random"), value=True)

                with gr.Row(visible=False) as steps_row:
                    steps = gr.Slider(
                        label=t("Denoising Steps"),
                        minimum=4,
                        maximum=9,
                        value=8,
                        step=1,
                    )

                def advanced_rows_visibility(v):
                    return gr.update(visible=v), gr.update(visible=v)

                advanced_checkbox.change(
                    advanced_rows_visibility,
                    inputs=advanced_checkbox,
                    outputs=[seed_random_row, steps_row],
                )

                gr.Examples(
                    examples=get_example_prompts(),
                    inputs=prompt,
                    label=t("Example Prompts"),
                )

            with gr.Column():
                gallery_images = gr.Gallery(
                    label=t("Generated Images"),
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                    interactive=False,
                )
                last_image_index = gr.State(value=None)
                used_seed = gr.Textbox(
                    label=t("Seed Used"), interactive=False, visible=False
                )

        with gr.Row():
            # Add source model link to footer, after Gradio credit.
            gr.HTML(
                js_on_load=f"""
                    document.querySelector("footer").insertAdjacentHTML(
                        "beforeend",
                        `<a
                            href="https://huggingface.co/{args.model}"
                            target="_blank"
                        >
                            {t("Source model")} 🤗
                        </a>`
                    )
                """
            )

        def update_resolution_choices(_aspect_ratio):
            resolution_choices = resolutions_by_aspect.get(
                _aspect_ratio, default_resolution_choices
            )
            return gr.update(value=resolution_choices[0], choices=resolution_choices)

        aspect_ratio.change(
            update_resolution_choices, inputs=aspect_ratio, outputs=resolution
        )
        generate_btn.click(
            generate,
            inputs=[prompt, resolution, seed, steps, random_seed, gallery_images],
            outputs=[gallery_images, last_image_index, used_seed, seed],
        ).then(
            # Select generated image in gallery:
            lambda imgs, idx: gr.Gallery(value=imgs, selected_index=idx),
            inputs=[gallery_images, last_image_index],
            outputs=gallery_images,
        )

        app.load(on_app_load)

    app.launch(
        server_port=args.port,
        footer_links=["gradio"],  # Credit
        theme=get_theme(),
        css_paths=[app_dir / "source" / "app.css"],
        js=(app_dir / "source" / "app.js").read_text(),
    )
