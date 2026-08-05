"""Diffusion pipeline wrapper."""

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from os import environ
from pathlib import Path

import gradio as gr
import torch
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module
from accelerate.utils.memory import clear_device_cache
from diffusers.modular_pipelines.components_manager import custom_offload_with_hook
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import get_device
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

from source.py.anima_flash import install_anima_flash_attn
from source.py.blocking_task import BlockingTask
from source.py.custom_logger import logger
from source.py.image_model import ImageModel
from source.py.krea2_flash import install_krea2_flash_attn
from source.py.offload_strat import (
    DENOISER_RANK,
    ENCODER_RANK,
    ReuseDistanceOffloadStrategy,
    eviction_rank,
)

KERNEL_CACHES = {
    "Triton": "TRITON_CACHE_DIR",
    "PyTorch Inductor": "TORCHINDUCTOR_CACHE_DIR",
}
"""Compilation caches, by the environment variable naming each one's directory.

Triton compiles the kernels, Inductor the graphs calling them: layered rather
than parallel, they are told apart by where they write. `app.py` sets both.
"""

MEMORY_RESERVE_RATIO = 0.35
"""Share of the GPU kept free for the activations of the running component."""

MIN_MEMORY_RESERVE = int(2.5 * 1024**3)
"""Smallest reserve, in bytes: enough that a small GPU is never filled to the
brim, where the driver starts backing it with host memory."""

MAX_MEMORY_RESERVE = 4 * 1024**3
"""Largest reserve, in bytes: past that, a roomy GPU would evict for nothing."""

VAE_BYTES_PER_MEGAPIXEL = int(0.9 * 1024**3)
"""What one megapixel costs a VAE pass, in bytes.

Measured on one decoder and carried above it, since another architecture may peak
elsewhere. It sizes the tiling decision and the room freed before a decode, a
pass costing a fraction of the run it ends: sizing either on the run's figure
would seam pictures that go through in one pass, and evict the denoiser to free
several times what a decode takes.
"""

PIXELS_PER_MEGAPIXEL = 1e6
"""Pixels in a megapixel, the unit pictures are sized in here.

Decimal, as the word is used of pictures, where memory stays binary. The bases
don't meet, which costs nothing as long as the same conversion sizes every
picture and the figures per megapixel were measured with it.
"""

STREAM_HYSTERESIS_RATIO = 0.10
"""Share of a run's own need the GPU has to fall short of before a denoiser
already seen seated is taken off it.

What the GPU offers swings between two identical generations, the allocator
keeping workspaces no component owns, and a threshold read bare inside that swing
flips every generation, re-hooking the pipeline each time. A band, not an
override: what a browser or a game takes from the GPU dwarfs it and carries the
decision.
"""

SEATED_DENOISERS: set[tuple[int, float]] = set()
"""Footprint and picture size of every denoiser that has held its seat.

Kept for the life of the process, not the pipeline's: a size come back to is not
a question reopened. Keyed on the footprint, that being what the seat was weighed
against, so a model swapped or a LoRA loaded files its own.

Only the seats are filed, never the streamings: one forced by a passing shortage
would otherwise hold for the whole session.
"""

RUN_COSTS_PER_MEGAPIXEL: dict[int, float] = {}
"""What a run was seen to cost per megapixel, in bytes, by denoiser footprint.

Kept for the life of the process, like the seats above and keyed the same way: a
model come back to has already been measured, and starting from the seed again
sizes its first residency on a figure that isn't its own.
"""

COST_REVISION_SHARE = 0.25
"""How far a run's measured cost may pull the figure in force towards it.

The highest figure was once kept outright, too much room costing speed where too
little fills the GPU. But a run can reserve more than its successors, being the
second of a model or following a resolution it had to stream, and it then holds
for the session: the room it asks for is room a residency is refused.
"""

SEED_BYTES_PER_MEGAPIXEL = int(2.5 * 1024**3)
"""What one megapixel is assumed to cost the GPU, in bytes, weights excluded,
until a run has been watched.

What a run costs is not what it holds but what the allocator reserves to serve
it, and the reserve is what has to fit, so the reserve is what gets watched.

This stands for the first generation of a freshly loaded model only, the figure
being that model's own from the second on. It is architecture bound and doesn't
travel, a model packing more pixels into a token holding a fraction of the
tensors of one packing fewer.

Set just above the cheapest run measured. It erred high before, an under-estimate
filling the GPU meaning a crawl on Windows rather than an error. But a seed high
enough to refuse a seat that would have held costs the same one generation as one
too low, so that caution was buying nothing. What bounds it is resolution: it
would have to fall well under anything measured for a high-resolution denoiser to
be seated, so those stay streamed whatever it is set to.
"""


def count_cached_kernels() -> dict[str, int]:
    """Count the entries each compilation cache holds, by cache name.

    A count, not a size: what a generation adds is what it had to compile, and
    the figure only has to be comparable with itself.
    """
    counts = {}

    for name, variable in KERNEL_CACHES.items():
        directory = environ.get(variable)

        try:
            counts[name] = (
                sum(1 for _ in Path(directory).rglob("*")) if directory else 0
            )
        except OSError:
            counts[name] = 0

    return counts


def get_memory_reserve(total_memory: int) -> int:
    """Get the GPU memory to keep free, in bytes, for the component running.

    A flat reserve doesn't travel: comfortable on a large GPU, it eats half of a
    small one and leaves the transformer streamed submodule by submodule at every
    step. A share of the GPU puts that trade-off at the same place everywhere.

    Args:
        total_memory: The GPU memory, in bytes.
    """
    reserve = int(MEMORY_RESERVE_RATIO * total_memory)

    return min(max(reserve, MIN_MEMORY_RESERVE), MAX_MEMORY_RESERVE)


def get_execution_device() -> torch.device:
    """Get the GPU the pipeline runs on, index included."""
    device = torch.device(get_device())

    if device.index is None:
        device = torch.device(f"{device.type}:0")

    return device


def get_memory_info() -> tuple[int, int] | None:
    """Get the GPU memory free and total, in bytes, or `None` if unmeasurable.

    Only the free figure nets out what the desktop and the other applications
    hold; a budget read off the total would promise memory this process never gets.
    """
    if torch.backends.mps.is_available():
        # Mac memory is shared with the CPU: its budget stands for both figures.
        budget = getattr(torch.mps, "recommended_max_memory", lambda: None)()

        return (budget, budget) if budget else None

    if not (torch.cuda.is_available() or torch.xpu.is_available()):
        return None

    device = get_execution_device()
    device_module = getattr(torch, device.type, torch.cuda)

    return device_module.mem_get_info(device.index)


class ImagePipeline:
    """An image pipeline."""

    def __init__(self):
        self.instance: ModularPipeline | DiffusionPipeline | None = None
        """Loaded pipeline, `None` during a swap."""

        self.memory_reserve = 0
        """GPU memory the offload strategy keeps free, in bytes; 0 without one."""

        self.memory_budget: int | None = None
        """GPU memory a single VAE pass can count on, in bytes; `None` if unknown."""

        self.free_memory = 0
        """GPU memory free before the pipeline reached it, in bytes.

        The one reading no weight of ours has distorted, kept as the fallback of
        `free_memory_without_ours`.
        """

        self.total_memory = 0
        """The GPU memory, in bytes, as read when the pipeline was loaded."""

        self.streams_weights = False
        """Are the weights streamed submodule by submodule, leaving the GPU free?"""

        self.streams_denoiser = False
        """Is the denoiser left on CPU, the resolution leaving it no seat?"""

        self.run_bytes_per_megapixel = 0.0
        """What a run of this model was seen to cost per megapixel, in bytes; 0
        until one has been watched, the seed standing in until then."""

        self.watched_run: tuple[float, int, bool] | None = None
        """Megapixels, resident weights and warming-up state of the generation
        being watched."""

        self.warms_up_next_run = True
        """Is the coming generation the one a pipeline warms up on? The first of a
        freshly loaded model is, and so is the first after a weight edit."""

        self.kernel_cache_counts: dict[str, int] = {}
        """What each compilation cache held before the warming-up generation."""

        self.streamed_reasons: dict[str, str] = {}
        """Why each component was last left on CPU, by component name."""

        self.run_margin = 0
        """GPU memory the coming run needs beyond the weights, in bytes; 0 until
        a resolution is known."""

        self.decode_margin = 0
        """GPU memory a VAE pass needs beyond its weights, in bytes; 0 until a
        resolution is known."""

        self.offload_strategy: ReuseDistanceOffloadStrategy | None = None
        """Strategy evicting the components, `None` when none is in play."""

        self.offload_hooks: list = []
        """Hooks of the evictable components, empty without an offload strategy."""

    def load(self, model: ImageModel) -> ImageModel:
        """Load an image model pipeline."""

        def create_pipe(model_id: str) -> DiffusionPipeline | ModularPipeline:
            """Create a standard pipeline or a modular one."""
            if not model.has_modular_pipeline():
                return DiffusionPipeline.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                )

            modular_pipeline = ModularPipeline.from_pretrained(model_id)
            modular_pipeline.load_components(dtype=torch.bfloat16)

            return modular_pipeline

        try:
            self.instance = create_pipe(model.id)
        except Exception:
            if model.backup_id:
                logger.warning(
                    f"Can't load {model.id}, falling back to {model.backup_id}."
                )
                self.instance = create_pipe(model.backup_id)
            else:
                raise

        # On NVIDIA, AMD & Intel ARC GPUs:
        if triton_is_available and (
            torch.cuda.is_available() or torch.xpu.is_available()
        ):
            for component_name, component in self.instance.components.items():
                quantization_config = getattr(component, "quantization_config", None)
                quant_method = getattr(quantization_config, "quant_method", None)

                if quant_method == "sdnq":
                    apply_sdnq_options_to_model(component, use_quantized_matmul=True)
                    logger.info(f"SDNQ Quantized MatMul enabled for {component_name}.")

            # The backend is picked process-wide, not per pipeline, so a swap
            # would otherwise inherit whatever the family before it chose.
            self.instance.transformer.set_attention_backend("native")

            if model.family in ("Z-Image", "FLUX", "FLUX.2"):
                try:
                    self.instance.transformer.set_attention_backend("flash")
                except Exception as e:  # noqa: BLE001
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available: {e}")

            # These two carry attention the backend refuses, Anima because its
            # text conditioner masks it, Krea 2 because of its grouped queries.
            # Only their transformer takes the kernel, one processor at a time.
            elif model.family == "Anima":
                try:
                    install_anima_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"FlashAttention is not available for Anima: {e}")
            elif model.family == "Krea 2":
                try:
                    install_krea2_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"FlashAttention is not available for Krea 2: {e}")

        try:
            self.instance.vae.to(memory_format=torch.channels_last)
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Can't apply memory format optimization: {e}")

        # Dropped before the reclaim below, and before anything is measured: each
        # hook holds the component it offloads, and they hold one another, so the
        # weights of the pipeline swapped out stay on the GPU while this list
        # names them. A budget read over them is the one the model before left.
        self.offload_strategy = None
        self.offload_hooks = []

        clear_device_cache(garbage_collection=True)

        # Read before anything reaches the GPU: this is what the weights and the
        # activations will share.
        memory_info = get_memory_info()
        self.memory_reserve = 0
        self.memory_budget = None
        self.free_memory, self.total_memory = memory_info or (0, 0)
        self.streams_weights = False
        self.streams_denoiser = False
        # A model already measured in this session keeps its figure; what another
        # one cost says nothing about it, and the seed stands in.
        self.run_bytes_per_megapixel = RUN_COSTS_PER_MEGAPIXEL.get(
            self.denoiser_footprint(), 0.0
        )
        self.watched_run = None
        self.warms_up_next_run = True
        self.kernel_cache_counts = {}
        self.streamed_reasons = {}
        self.run_margin = 0
        self.decode_margin = 0

        self.log_gpu_state()

        # On NVIDIA, AMD & Intel ARC GPUs:
        if memory_info and (torch.cuda.is_available() or torch.xpu.is_available()):
            self.offload_to_cpu(self.total_memory)

        # On Mac GPUs:
        elif torch.backends.mps.is_available():
            self.instance.to("mps")

            # To prevent swap and performance degradation...
            if hasattr(self.instance, "enable_attention_slicing"):
                self.instance.enable_attention_slicing()

        self.memory_budget = self.measure_memory_budget(memory_info)

        return model

    def measure_memory_budget(self, memory_info: tuple[int, int] | None) -> int | None:
        """Measure the GPU memory a single VAE pass can count on, in bytes.

        Only the weights that can't leave the GPU are deducted. Under the offload
        strategy that's the VAE itself, every sibling being evictable, so the
        decode gets nearly the whole GPU rather than the leftovers of whatever
        ran before it. It starts from the free memory, never the total: the
        desktop takes its cut first, and a share of the GPU would ignore it.

        Args:
            memory_info: GPU memory free and total, in bytes, read with the
                pipeline still on CPU, or `None` if it couldn't be measured.

        Returns:
            The budget, or `None` if it can't be measured.
        """
        if memory_info is None or self.instance is None:
            return None

        free_memory = memory_info[0]

        # Streamed weights never claim the GPU, leaving all of it to the pass.
        if self.streams_weights:
            return free_memory

        def footprint_of(component) -> int:
            get_footprint = getattr(component, "get_memory_footprint", None)

            return get_footprint() if get_footprint is not None else 0

        if self.offload_strategy is not None:
            resident = footprint_of(getattr(self.instance, "vae", None))
        else:
            # Nothing evicts on Mac: the whole pipeline stays on the GPU.
            resident = sum(map(footprint_of, self.instance.components.values()))

        budget = max(self.memory_reserve, free_memory - resident)

        logger.info(
            f"VAE budget: {budget / VAE_BYTES_PER_MEGAPIXEL:.1f}MP in one pass, "
            f"{free_memory / 1024**3:.1f}GB free."
        )

        return budget

    def log_gpu_state(self) -> None:
        """Say what a freshly loaded pipeline finds already taken on the GPU.

        Read after the reclaim, so what shows up is what the reclaim couldn't get
        back. Told in three because the figures rule things out in turn: past the
        allocator's reserve lies the context, the driver and the pools the compiled
        kernels keep, none of it ours; inside it, what is still allocated says
        whether a reference holds the weights or the segments are merely split too
        fine to return.
        """
        memory_info = get_memory_info()

        if memory_info is None:
            return

        device = get_execution_device()
        device_module = getattr(torch, device.type, torch.cuda)
        free_memory, total_memory = memory_info

        logger.info(
            f"GPU has {(total_memory - free_memory) / 1024**3:.1f}GB taken, "
            f"{device_module.memory_reserved(device.index) / 1024**3:.1f}GB of it "
            f"reserved by the allocator and "
            f"{device_module.memory_allocated(device.index) / 1024**3:.1f}GB "
            f"still allocated."
        )

    def free_memory_without_ours(self) -> int:
        """Memory the GPU would offer with none of our weights on it.

        What is free counts our own resident weights as taken, so a residency
        decided on it would depend on what the last generation left behind.
        Adding back only what is ours removes that, while a browser or a game
        taking its share of the GPU still shows up in the figure.
        """
        memory_info = get_memory_info()

        if memory_info is None or self.instance is None:
            return self.free_memory

        free_memory = memory_info[0]
        device = get_execution_device()

        for component in self.instance.components.values():
            if not isinstance(component, torch.nn.Module):
                continue

            footprint = getattr(component, "get_memory_footprint", None)

            if footprint is not None and getattr(component, "device", None) == device:
                free_memory += footprint()

        return free_memory

    def denoiser_footprint(self) -> int:
        """Weight of the component a run calls at every step, in bytes.

        Zero when there is none to size, which is also what keys the figures a
        model leaves behind: two pipelines of the same denoiser want the same room.
        """
        if self.instance is None:
            return 0

        denoiser = getattr(self.instance, "transformer", None) or getattr(
            self.instance, "unet", None
        )
        footprint = getattr(denoiser, "get_memory_footprint", None)

        return footprint() if footprint is not None else 0

    def bytes_per_megapixel(self) -> float:
        """What a run of the loaded model costs the GPU per megapixel, in bytes.

        Read off the generations already made rather than assumed. No
        configuration gives it reliably: it follows the token count, hence the
        patch size and the VAE scale under names that differ by family, and it
        follows the quantization, the attention backend and whether a reference
        image is encoded on the way in. The GPU answers all of it at once.
        """
        return self.run_bytes_per_megapixel or SEED_BYTES_PER_MEGAPIXEL

    def measure_run(self) -> None:
        """Take the measure of the generation that just ran, to size the next.

        What is read is the reserve the allocator peaked at, not the tensors it
        held, the reserve being what the GPU had to give. Taking out the weights
        meant to sit there leaves what the picture cost. The figure rises to a
        costlier run at once and eases towards a cheaper one, since leaving too
        much room costs speed where too little fills the GPU.

        The gap between the two is not slack to be reclaimed: the allocator gives
        ground gracefully as the GPU is taken from it, then falls off a cliff,
        and sizing on the tensors alone lands a run at its edge.

        The generation a pipeline warms up on is not measured, compiled or not:
        autotuning reserves a workspace per kernel variant it benchmarks, and the
        weights climb onto the GPU inside that same run. Either peak is the
        warming-up's rather than the picture's.

        Read before the coming residency is settled, so the figure sizing it is
        the one the generation before left, not one a generation stale.
        """
        watched, self.watched_run = self.watched_run, None

        if watched is None:
            return

        device = get_execution_device()
        device_module = getattr(torch, device.type, torch.cuda)
        watched_megapixels, resident, warmed_up = watched
        cost = device_module.max_memory_reserved(device.index) - resident

        if warmed_up:
            self.log_compiled_kernels()
        elif cost > 0 and watched_megapixels > 0:
            measured = cost / watched_megapixels
            in_force = self.bytes_per_megapixel()

            # Up to a costlier run at once, the GPU not being open to argument;
            # down towards a cheaper one by a share, so that a generation which
            # reserved more than those after it stops holding the whole session.
            self.run_bytes_per_megapixel = max(
                measured, in_force - COST_REVISION_SHARE * (in_force - measured)
            )
            RUN_COSTS_PER_MEGAPIXEL[self.denoiser_footprint()] = (
                self.run_bytes_per_megapixel
            )

            logger.info(
                f"A run of this model costs "
                f"{self.run_bytes_per_megapixel / 1024**3:.1f}GB per megapixel, "
                f"{measured / 1024**3:.1f}GB as just measured."
            )

    def watch_run(self, megapixels: float) -> None:
        """Note what the coming generation is to be measured against.

        Args:
            megapixels: Size of the picture the next generation will make.
        """
        device = get_execution_device()
        device_module = getattr(torch, device.type, torch.cuda)

        # Reset last, so the weights this fit shuttled land outside the window the
        # coming generation is measured over.
        device_module.reset_peak_memory_stats(device.index)

        # A streamed denoiser holds no seat: the peak its submodules raise
        # crossing the bus stands against no residency.
        self.watched_run = (
            None
            if self.streams_denoiser
            else (
                megapixels,
                sum(hook.model.get_memory_footprint() for hook in self.offload_hooks),
                self.warms_up_next_run,
            )
        )

        if self.warms_up_next_run:
            self.kernel_cache_counts = count_cached_kernels()
            logger.info(
                "Warming up Triton and PyTorch Inductor caches... "
                "Next generation will be longer."
            )

        self.warms_up_next_run = False

    def log_compiled_kernels(self) -> None:
        """Say what the generation a pipeline warmed up on wrote to each cache.

        Nothing written means the caches already held it all, and the generation
        was the longer one for putting the weights on the GPU alone.
        """
        counts = count_cached_kernels()

        # Read off the snapshot, never off the caches: without one there is no
        # growth to report.
        grown = [
            (name, counts[name] - held)
            for name, held in self.kernel_cache_counts.items()
            if counts.get(name, 0) > held
        ]

        # The unit spelled out once, shortened for the caches after it.
        written = [
            f"{count} cache entries for {name}" if index == 0 else f"{count} for {name}"
            for index, (name, count) in enumerate(grown)
        ]

        logger.info(
            f"Compilation wrote {', '.join(written)}."
            if written
            else "Compilation wrote nothing: the caches already held these kernels."
        )

    def fit_to_resolution(self, width: int, height: int) -> None:
        """Set the pipeline up for the size of the picture about to be made.

        Args:
            width: Width of the picture to generate, in pixels.
            height: Height of the picture to generate, in pixels.
        """
        # Blocks kept from the generation before are sized for another picture,
        # and until handed back they read as taken by the measures below.
        clear_device_cache(garbage_collection=True)

        megapixels = width * height / PIXELS_PER_MEGAPIXEL

        # Measured before anything is settled, its figure being what settles it.
        if self.offload_strategy is not None:
            self.measure_run()

        # What the run wants beyond the weights, and the one figure the denoiser's
        # seat is weighed against.
        self.run_margin = int(megapixels * self.bytes_per_megapixel())

        self.stream_denoiser_if_needed(megapixels)
        self.tile_vae_if_needed(megapixels)

        # Last, so that what it records is the residency just settled on.
        if self.offload_strategy is not None:
            self.watch_run(megapixels)

    def stream_denoiser_if_needed(self, megapixels: float) -> None:
        """Take the denoiser off the GPU when the picture leaves it no seat.

        A run calls the denoiser at every step, so it earns its place while there
        is one: streamed, its weights pay a trip over the bus each time, and every
        sibling follows it off the GPU. Past a resolution those weights and what
        the loop keeps alive no longer fit together, and where the allocator can't
        be set to `expandable_segments`, as on Windows, the driver backs the GPU
        with host memory rather than fail.

        The trip over the bus costs less than that overflow, and by a wide margin.
        The whole of the run's margin is what the seat is weighed against for that
        reason: weighing it against a fraction was measured several times slower
        at high resolution. Do not trade this threshold for room that isn't there.

        Args:
            megapixels: Size of the picture to generate.
        """
        if self.instance is None or self.offload_strategy is None:
            return

        weights = self.denoiser_footprint()

        if not weights:
            return

        # The whole of the run's margin, its caller having settled the figure.
        needed = weights + self.run_margin
        available = self.free_memory_without_ours()

        # A swing in what the GPU offers is noise, not an answer that changed: a
        # size seen to hold its seat keeps it until the GPU falls short by more
        # than that swing. One never seated is answered on the estimate alone.
        seat = (weights, megapixels)
        band = int(STREAM_HYSTERESIS_RATIO * needed) if seat in SEATED_DENOISERS else 0
        streams = needed > available + band

        if not streams:
            SEATED_DENOISERS.add(seat)

        if streams == self.streams_denoiser:
            return

        logger.info(
            f"{'Streaming' if streams else 'Seating'} the denoiser for "
            f"{megapixels:.1f}MP: {needed / 1024**3:.1f}GB needed with the run, "
            f"{available / 1024**3:.1f}GB to be had."
        )

        self.remove_hooks()
        self.offload_to_cpu(self.total_memory, stream_denoiser=streams)

    def tile_vae_if_needed(self, megapixels: float) -> None:
        """Tile the VAE work only for the pictures this GPU can't handle in one pass.

        The VAE encodes and decodes the picture as a whole, so its peak memory
        grows with the resolution and, past a point, it alone fills the GPU. Tiling
        bounds that peak but can leave faint seams, so it's a trade only worth
        making when the picture wouldn't go through otherwise.

        Args:
            megapixels: Size of the picture to generate.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None:
            return

        peak = int(megapixels * VAE_BYTES_PER_MEGAPIXEL)

        # An unmeasurable GPU gets the safe path rather than an optimistic one.
        needs_tiling = self.memory_budget is None or peak > self.memory_budget

        # What the pass itself needs, the reserve standing in where it can't be
        # sized: tiling bounds a pass unmeasurably, and a streamed denoiser leaves
        # its siblings little to fit around.
        streams = self.streams_denoiser or needs_tiling
        self.decode_margin = self.memory_reserve if streams else peak

        # The room a run's tensors want beyond the weights the strategy moves is
        # not asked for here: a margin sized on the run has it evict every sibling
        # on each arrival, where the allocator would have given ground instead.
        # Only the seat of the denoiser is weighed against that figure.
        if self.offload_strategy is not None:
            self.offload_strategy.memory_reserve_margin = self.memory_reserve

            logger.info(
                f"Decoding {megapixels:.1f}MP "
                f"{'tiled' if needs_tiling else 'in one pass'}, "
                f"evicting down to {self.decode_margin / 1024**3:.1f}GB free."
            )

        toggle = getattr(
            vae, "enable_tiling" if needs_tiling else "disable_tiling", None
        )

        if toggle is not None:
            toggle()

    def offload_to_cpu(self, total_memory: int, stream_denoiser: bool = False) -> None:
        """Offload the pipeline weights to CPU, keeping on GPU what fits.

        Args:
            total_memory: The GPU memory, in bytes.
            stream_denoiser: Leave the denoiser on CPU too, the resolution
                leaving it no room next to the tensors of the run.
        """
        if self.instance is None:
            return

        if isinstance(self.instance, DiffusionPipeline):
            self.instance.enable_sequential_cpu_offload()
            self.streams_weights = True
            return

        self.streams_denoiser = stream_denoiser

        # A modular pipeline has no offload helper: replicate the auto CPU offload
        # of its components manager, which can't be used as is because it hooks
        # every component, including those too large to stay on the GPU.
        device = get_execution_device()
        memory_reserve = get_memory_reserve(total_memory)
        self.memory_reserve = memory_reserve

        offload_strategy = ReuseDistanceOffloadStrategy(
            memory_reserve_margin=memory_reserve
        )
        self.offload_strategy = offload_strategy

        streamed: dict[str, str] = {}

        def stream(
            name: str, component: torch.nn.Module, reason: str, chosen: bool = True
        ) -> None:
            """Leave a component on CPU, feeding the GPU one submodule at a time.

            Being picked for it is an optimisation; being forced into it by a
            component the GPU can't hold is a limit worth warning about.

            A seat given up settles every component again, nearly all of them the
            same way as before, so only a reason that changed earns a line.
            """
            cpu_offload(
                component, device, offload_buffers=len(component._parameters) > 0
            )
            streamed[name] = reason

            if self.streamed_reasons.get(name) != reason:
                log = logger.info if chosen else logger.warning
                log(f"Streaming {name}: {reason}.")

        candidates = []

        for name, component in self.instance.components.items():
            if not isinstance(component, torch.nn.Module):
                continue

            footprint = getattr(component, "get_memory_footprint", None)

            # Saturates the GPU on its own, so it skips the arbitration below.
            if footprint is None:
                stream(name, component, "can't be sized", chosen=False)
            elif footprint() + memory_reserve > total_memory:
                stream(
                    name,
                    component,
                    f"its {footprint() / 1024**3:.1f}GB is too large for this GPU",
                    chosen=False,
                )
            elif stream_denoiser and eviction_rank(name) >= DENOISER_RANK:
                stream(name, component, "this resolution wants the whole GPU")
            else:
                candidates.append((name, component, footprint()))

        candidates = self.stream_least_called_until_resident(
            candidates, memory_reserve, stream
        )

        # Set once the pass is over, so a component streamed twice within it is
        # weighed against the pass before rather than against itself.
        self.streamed_reasons = streamed

        hooks = [
            custom_offload_with_hook(
                name, component, device, offload_strategy=offload_strategy
            )
            for name, component, _ in candidates
        ]

        # Let each component evict its siblings when the GPU runs short.
        for hook in hooks:
            for other_hook in hooks:
                if other_hook is not hook:
                    hook.add_other_hook(other_hook)

        self.offload_hooks = hooks
        self.free_gpu_before_decode()

    def stream_least_called_until_resident(
        self,
        candidates: list[tuple[str, torch.nn.Module, int]],
        memory_reserve: int,
        stream: Callable[[str, torch.nn.Module, str], None],
    ) -> list[tuple[str, torch.nn.Module, int]]:
        """Stream the components a run calls once, so the denoiser can stay put.

        Sizing components one by one says nothing about them sharing the GPU:
        when the set doesn't fit, someone leaves at every generation. Picking by
        size elects the denoiser, whose streaming is paid once per step, over an
        encoder paying once per generation.

        The criterion is thus the call count, not the reuse distance
        `eviction_rank` measures. That ranking is borrowed because the two
        coincide here: what runs once is also what isn't wanted again until the
        next generation. An encoder called at every step would need its own.

        The denoiser is never streamed here; it only is when it can't fit the
        GPU at all, or when the resolution leaves it no seat, both settled
        before this runs. Where it was, the encoders follow it off the GPU
        whether they would have fit or not: their room is room its flow never
        gets, and being streamed it never asks the strategy for any.

        The GPU is measured with our own weights added back, never as it stands.
        This also runs after a LoRA load, and read raw there the free memory
        counts those weights as taken: the guards below give up early, and a
        topology settled at load turns into one that shuttles components in and
        out at every generation.

        Args:
            candidates: Name, component and footprint of what could stay on GPU.
            memory_reserve: GPU memory to leave free for the run, in bytes.
            stream: Callback leaving a component on CPU.

        Returns:
            The candidates still meant to stay on the GPU.
        """
        # The components streamed just above handed their blocks to the allocator,
        # not to the driver, and their parameters sit on `meta` where the reading
        # no longer credits them: unreclaimed, that memory is counted by nobody
        # and the guards below give up on a GPU that has the room.
        clear_device_cache(garbage_collection=True)

        free_memory = self.free_memory_without_ours()

        if not free_memory:
            return candidates

        # A denoiser absent from the candidates is already on the bus, put there by
        # this GPU's size or by the resolution.
        denoiser_streams = not any(
            eviction_rank(candidate[0]) >= DENOISER_RANK for candidate in candidates
        )

        # Least called first, largest of those: most room bought per crossing.
        for name, component, footprint in sorted(
            candidates, key=lambda c: (eviction_rank(c[0]), -c[2])
        ):
            resident = sum(size for _, _, size in candidates)

            # A streamed denoiser asks the strategy for nothing: its submodules
            # arrive by a hook of their own, where the eviction it would have set
            # off never fires. So the room an encoder holds is room its flow never
            # gets, and the encoders leave whether they would have fit or not.
            if resident + memory_reserve <= free_memory and not (
                denoiser_streams and eviction_rank(name) == ENCODER_RANK
            ):
                break

            if eviction_rank(name) >= DENOISER_RANK:
                break

            # Even streaming all of them leaves the denoiser no room to compute:
            # it has to go instead, which the caller's own sizing covers.
            remaining = resident - footprint

            if remaining + MIN_MEMORY_RESERVE > free_memory:
                break

            stream(
                name,
                component,
                f"{footprint / 1024**3:.1f}GB freed, and it is called once per run",
            )
            candidates = [c for c in candidates if c[0] != name]

        return candidates

    def free_gpu_before_decode(self) -> None:
        """Make room for a decode, evicting the furthest used siblings first.

        A component only evicts others when it *arrives* on the GPU, and the VAE
        is already there when the picture is decoded: nothing ever makes room for
        the pass that needs it most, and the driver backs it with host memory.

        Only what the pass is short of gets freed. Emptying the GPU wholesale
        would cost every weight its return trip next generation, reallocated
        host-side each way by `module.to()`: that churn is what makes a session
        slower as it goes.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None or getattr(vae.decode, "frees_gpu", False):
            return

        decode = vae.decode

        def decode_with_room(*args, **kwargs):
            device = get_execution_device()
            # The pass's own margin, never the run's: the siblings freed here are
            # wanted again at the next generation.
            margin = self.decode_margin or self.memory_reserve

            # Cached activation blocks read as used until handed back: reclaim
            # before measuring, or the figure is stale.
            clear_device_cache(garbage_collection=True)
            memory_info = get_memory_info()
            free_memory = memory_info[0] if memory_info else 0

            siblings = sorted(
                (
                    hook
                    for hook in self.offload_hooks
                    if hook.model is not vae and hook.model.device == device
                ),
                key=lambda hook: eviction_rank(hook.model_id),
            )
            evicted = False

            for hook in siblings:
                # An unmeasurable GPU gets the safe path: evict everything.
                if memory_info is not None and free_memory >= margin:
                    break

                logger.info(f"Evicting {hook.model_id} before the decode.")
                hook.offload()
                free_memory += hook.model.get_memory_footprint()
                evicted = True

            if evicted:
                clear_device_cache(garbage_collection=True)

            return decode(*args, **kwargs)

        setattr(decode_with_room, "frees_gpu", True)  # noqa: B010
        vae.decode = decode_with_room

    def remove_hooks(self) -> None:
        """Break the reference cycles left by CPU offload."""
        if self.instance is None:
            return

        for component in self.instance.components.values():
            if isinstance(component, torch.nn.Module):
                remove_hook_from_module(component, recurse=True)

    @contextmanager
    def unhooked(self) -> Iterator[None]:
        """Drop the CPU offload hooks for the time of a weight edit.

        Loading a LoRA, Diffusers removes them itself, then restores them with
        `enable_sequential_cpu_offload()`, which a modular pipeline lacks: it
        raises, leaving every component unhooked on CPU. Only the Accelerate
        hooks are visible to it, hence a crash reserved to the components too
        large for the GPU.
        """
        if (
            self.instance is None
            or isinstance(self.instance, DiffusionPipeline)
            or self.offload_strategy is None
            or not self.total_memory
        ):
            yield
            return

        streams_denoiser = self.streams_denoiser
        self.remove_hooks()

        try:
            yield
        finally:
            # The new weights come in unhooked: re-offload covers them too, and
            # the denoiser keeps the seat, or the CPU, the resolution gave it.
            self.offload_to_cpu(self.total_memory, stream_denoiser=streams_denoiser)

            # Shuttling the weights raised a peak no generation held, so the run
            # being watched is dropped rather than measured wrong.
            self.watched_run = None

            # The new weights want a warming-up run. The seat found for the
            # denoiser needs no undoing, filed under a footprint they move.
            self.warms_up_next_run = True

    def swap(self, model: ImageModel, t: Callable[[str], str]) -> ImageModel:
        """Swap an image model pipeline, blocking other critical tasks.

        Args:
            model: The image model to load.
            t: Translation function.
        """
        with BlockingTask.run(t("Please wait, a model is being loaded.")):
            self.remove_hooks()

            # Drop the old pipeline before collecting, otherwise it stays alive
            # and its VRAM makes the next auto CPU offload overly aggressive.
            self.instance = None

            # Its compiled graphs hold CUDA graph pools of their own, which no
            # `empty_cache()` reaches: kept, the GPU carries them for a model that
            # will never call them again, and the residency of the one arriving is
            # settled on a smaller GPU than it has. The kernels stay on disk, so
            # what this costs is a cache read on the warming-up generation.
            try:
                torch.compiler.reset()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Can't reset the compiler between models: {e}")

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()

            loaded = self.load(model)
            logger.info(f"Switched to {model.name}.")

            return loaded

    def supports_strength(self) -> bool:
        """Does this pipeline accept a `strength` argument?

        Returns:
            `True` for strength-based image-to-image pipelines (e.g. Anima, Z-Image),
            `False` for conditioning-based edit models (e.g. FLUX.2 [klein]),
            for text-to-image only pipelines, and during a swap.
        """
        blocks = getattr(self.instance, "blocks", None)
        return blocks is not None and "strength" in blocks.input_names

    @staticmethod
    def warn_if_not_optimizable(t: Callable[[str], str]) -> None:
        """Warn the user if the pipeline can't be optimized.

        Args:
            t: Translation function.
        """
        if torch.backends.mps.is_available():
            return  # The checks below do not apply to Mac.

        try:
            from flash_attn import flash_attn_func  # noqa: F401

            flash_is_available = True
        except Exception:  # noqa: BLE001
            flash_is_available = False

        if not triton_is_available or not flash_is_available:
            gr.Warning(
                t(
                    "Image generation may be slow because the diffusion pipeline can't be optimized."
                )
                + "<br>"
                + t(
                    "Try upgrading your graphics card drivers, then reboot your PC and restart ZPix."
                ),
                duration=None,  # Until user closes it.
            )
