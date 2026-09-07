"""Diffusion pipeline wrapper."""

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import chain
from os import environ
from pathlib import Path

import gradio as gr
import torch
from accelerate.hooks import remove_hook_from_module
from accelerate.utils.memory import clear_device_cache
from diffusers.hooks.group_offloading import (
    _GROUP_OFFLOADING,
    _LAYER_EXECUTION_TRACKER,
    _LAZY_PREFETCH_GROUP_OFFLOADING,
    apply_group_offloading,
)
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
    DenoiserFirstOffloadStrategy,
    eviction_rank,
)
from source.py.resolutions import smallest_picture_pixels
from source.py.run_memory import RunMemory, read_architecture

KERNEL_CACHES = {
    "Triton": "TRITON_CACHE_DIR",
    "PyTorch Inductor": "TORCHINDUCTOR_CACHE_DIR",
}
"""Compilation caches, by the environment variable naming each one's directory.

Triton compiles the kernels, Inductor the graphs calling them: layered rather
than parallel, they are told apart by where they write. `app.py` sets both.
"""

RESERVED_MEMORY = int(1.5 * 1024**3)
"""GPU memory no arbitration here may hand out, in bytes.

One pool, for everything that will want room on the card without ever appearing
in a reading of it:

- what the desktop and its windows take *beyond* what they were holding when the
  figures were read, a window opened mid-session helping itself whether or not a
  generation is under way;
- a LoRA the denoiser hasn't been given yet, whose layers land on the card while
  it is already sitting there, before anything has had the chance to arbitrate;
- the room the allocator needs to place the blocks of a run, which it can't find
  on the last slice of a card and can't rearrange where `expandable_segments` is
  unavailable. A seat granted with a twentieth of the card to spare was watched
  to spend minutes on a step, at a resolution every figure said fitted.

Sized for a desktop, its windows, the image editor a picture is likely to be
opened in next, and an adapter on top. Not for a game running beside them: one
takes several times this on its own, and a figure large enough to survive that
would refuse the denoiser its seat on every card, for a case nobody generates
pictures in.

Flat, and deliberately so. What the right figure is depends on the card, on the
driver and on what else the machine is running, none of which the architecture
of a model says anything about: it answers for what the reading can't see rather
than for a shortcoming of it. Above about twelve gigabytes of card it costs
nothing, every model here sitting whole with room to spare.
"""

ADAPTED_RUN = 3
"""What a run with an adapter is weighed as asking against one without, per run.

An adapter wraps a layer rather than replacing it: the input stays alive for two
paths, and two outputs are added where one was written. That is a cost per
activation, following the resolution and the layers wrapped rather than the
adapter's own weight, so a small adapter over the same layers asks nearly what a
large one does. Weighed on the footprint instead, this changed no decision.

It stands for more than the adapter, though, and the figure says so: a 75MB one
was watched to take a 1080p denoiser seated with two gigabytes to spare into
paging, from 4.5 to 27.9 seconds a step from the third generation on and for the
rest of the session. The same denoiser without it held five generations. What
the reserve doesn't cover, at that resolution, is thin enough that an adapter
crosses it, so this is charged where the crossing happens rather than to every
picture: read as the adapter's own cost it would be several times too large.

Conservative on purpose, the two errors costing nothing alike: refusing a seat
that would have held costs half a second a step, granting one that doesn't has
the driver page the card at twenty-three.
"""

WARMING_UP_RUN = 3
"""What a generation that compiles asks against a settled one, per run.

The compiler runs each shape before keeping a kernel for it, holding what a step
holds and a copy beside, in blocks whose sizes the cached ones can't serve. One
generation in a resolution's life pays this, and the run is counted several
times over for it.

Measured, not reasoned: a denoiser settling at 4.2GB was watched to hold 5.9GB
while compiling, and to have the allocator take 7.8GB of an 8GB card to place
them. Three times the run covers both.

Charged to the seat alone, never to the eviction reserve: this says what a
generation might reach, where the reserve is what every arrival has to leave
behind, and one nothing can reach empties the GPU on each of them.

Sizing the reserve on it was tried and doesn't hold: what a compiling run reaches
isn't a multiple of a settled one at every resolution, so the multiple covering
one leaves the arrival short at another. `ENCODER_RANK` frees that room without a
figure at all.
"""

PIXELS_PER_MEGAPIXEL = 1e6
"""Pixels in a megapixel, the unit pictures are sized in here.

Decimal, as the word is used of pictures, where memory stays binary. The bases
don't meet, which costs nothing as long as the same conversion sizes every
picture, no figure here being per megapixel to begin with.
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


def tensor_bytes(
    module: torch.nn.Module, deep: bool = True, device: torch.device | None = None
) -> int:
    """What a module's own tensors weigh, in bytes, optionally on one device only.

    Summed over the tensors themselves rather than read off `get_memory_footprint`,
    which answers for a whole module and can't answer for part of one. A module
    holding some of its blocks on the card and the rest on the bus weighs, on the
    card, only what is really there, and an offloaded tensor left behind on the
    meta device weighs nothing anywhere.

    Args:
        module: The module to weigh.
        deep: Weigh what its children hold as well as what it holds itself.
        device: Count only the tensors sitting on this device, or all of them.
    """
    tensors = chain(
        module.parameters(recurse=deep),
        module.buffers(recurse=deep),
    )

    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if device is None or tensor.device == device
    )


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

        self.residency_reserve = 0
        """GPU memory the resident set has to leave free, in bytes; 0 without one.

        The widest phase of a generation, where `memory_reserve` is the loop's
        alone. It says who may sit beside the denoiser, never whether the
        denoiser sits: a decode too wide to share the card sends the encoders
        onto the bus, the seat not being theirs to take.
        """

        self.settled = 0
        """The reserve the resident set was last arbitrated against, in bytes."""

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

        self.run_memory: RunMemory | None = None
        """What this model's architecture was read to ask of the GPU per pixel,
        `None` while none is loaded or where it couldn't be walked."""

        self.warmed_up_streams: set[int] = set()
        """Stream sizes this model has already compiled for, in pixels.

        The kernels are compiled per shape, so a resolution never generated is a
        resolution still to compile, whatever the ones before it cost. Emptied by
        a load and by a weight edit, both of which the compiled graphs don't
        survive.
        """

        self.warms_up_next_run = True
        """Is the coming generation the one that compiles for its resolution?"""

        self.warmed_up_run = False
        """Was the generation that just ran the one the pipeline warmed up on?"""

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

        self.offload_strategy: DenoiserFirstOffloadStrategy | None = None
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

            # Picked process-wide, not per pipeline: a swap would otherwise
            # inherit whatever the family before it chose.
            self.instance.transformer.set_attention_backend("native")

            if model.family in ("Z-Image", "FLUX", "FLUX.2"):
                try:
                    self.instance.transformer.set_attention_backend("flash")
                except Exception as e:  # noqa: BLE001
                    self.instance.transformer.reset_attention_backend()
                    logger.warning(f"FlashAttention is not available: {e}")

            # These two families mask their attention and the backend refuses any
            # attn_mask, while the flash_attn package unpads around it.
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
        # hook holds the component it offloads, so the weights of the pipeline
        # swapped out stay on the GPU while this list names them.
        self.offload_strategy = None
        self.offload_hooks = []

        clear_device_cache(garbage_collection=True)

        # Read before anything reaches the GPU: the weights and the activations
        # will share this.
        memory_info = get_memory_info()
        self.memory_reserve = 0
        self.residency_reserve = 0
        self.memory_budget = None
        self.free_memory, self.total_memory = memory_info or (0, 0)
        self.streams_weights = False
        self.streams_denoiser = False
        self.settled = 0
        self.warmed_up_streams = set()
        self.warms_up_next_run = True
        self.warmed_up_run = False
        self.kernel_cache_counts = {}
        self.streamed_reasons = {}
        self.run_margin = 0
        self.decode_margin = 0

        self.log_gpu_state()
        self.read_run_memory()

        # No resolution is in hand here, so the smallest picture stands in. The
        # residency it settles is provisional, the fit before each generation
        # taking it again on the picture actually asked for.
        #
        # The run's margin and nothing else, here as before every generation: the
        # reserve answers to the loop. Reserving for a whole pass was watched to
        # declare a denoiser too large for a card it fits.
        self.memory_reserve = self.run_bytes(smallest_picture_pixels())
        self.residency_reserve = self.memory_reserve

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

        Only the weights that can't leave the GPU are deducted, which under the
        offload strategy is the VAE itself: by the time a pass runs, the encoders
        are done and the denoiser has taken its last step, so every sibling is
        evictable. It starts from the free memory, never the total: the desktop
        takes its cut first, and a share of the GPU would ignore it.

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

        # What is left once the weights that can't leave have taken their cut, and
        # nothing else. Floored at the reserve, which is read off a picture and can
        # want more than the card holds, it would promise a pass memory the GPU was
        # never going to give it.
        budget = max(free_memory - resident, 0)

        # Told as a picture rather than as a size, that being the question it
        # answers, and against this decoder's own cost per pixel: the same budget
        # buys a larger picture through a decoder holding fewer channels at the
        # resolution it ends on.
        per_pixel = self.run_memory.decode_bytes_per_pixel if self.run_memory else 0

        logger.info(
            f"VAE budget: {budget / per_pixel / PIXELS_PER_MEGAPIXEL:.1f}MP in one "
            f"pass, {free_memory / 1024**3:.1f}GB free."
            if per_pixel
            else f"VAE budget: {free_memory / 1024**3:.1f}GB free."
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
        # Mac neither reclaims nor offloads, and reads one shared budget for the
        # free and the total alike: no reserve of ours to split, and no remainder
        # to stand for what the rest of the machine took.
        if torch.backends.mps.is_available():
            return

        memory_info = get_memory_info()

        if memory_info is None:
            return

        device = get_execution_device()
        device_module = getattr(torch, device.type, torch.cuda)
        free_memory, total_memory = memory_info

        reserved = device_module.memory_reserved(device.index)
        allocated = device_module.memory_allocated(device.index)

        # Nothing of ours left needs no figures to be told: only a reserve that
        # survived the reclaim earns the split.
        ours = (
            f"{reserved / 1024**3:.1f}GB of it held by our allocator, "
            f"{allocated / 1024**3:.1f}GB of that still allocated"
            if reserved
            else "none of it ours"
        )

        logger.info(
            f"GPU has {(total_memory - free_memory) / 1024**3:.1f}GB taken, {ours}."
        )

    def free_memory_without_ours(self) -> int:
        """Memory the GPU would offer with none of our weights on it.

        What is free counts our own resident weights as taken, so a residency
        decided on it would depend on what the last generation left behind.
        Adding back only what is ours removes that, while a browser or a game
        taking its share of the GPU still shows up in the figure.

        Added back tensor by tensor, and only those actually on the card. A
        component is no longer all on one side of the bus: a denoiser seats the
        blocks that fit and streams the others, and asking it for one device, as
        this used to, reads whichever side its first weight happens to be on and
        then adds back the whole of it either way.
        """
        memory_info = get_memory_info()

        if memory_info is None or self.instance is None:
            return self.free_memory

        free_memory = memory_info[0]
        device = get_execution_device()

        for component in self.instance.components.values():
            if isinstance(component, torch.nn.Module):
                free_memory += tensor_bytes(component, device=device)

        return free_memory

    def denoiser(self) -> torch.nn.Module | None:
        """The component a run calls at every step, `None` where there is none."""
        if self.instance is None:
            return None

        return getattr(self.instance, "transformer", None) or getattr(
            self.instance, "unet", None
        )

    def adapter_footprint(self) -> int:
        """Weight of the adapters given to the denoiser, in bytes, 0 without any.

        Read off the tensor names rather than asked of the pipeline, which
        answers for adapters it loaded and not for those merged into the weights.
        """
        denoiser = self.denoiser()

        if denoiser is None:
            return 0

        tensors = chain(denoiser.named_parameters(), denoiser.named_buffers())

        return sum(
            tensor.numel() * tensor.element_size()
            for name, tensor in tensors
            if "lora" in name.lower()
        )

    def denoiser_footprint(self) -> int:
        """Weight of the component a run calls at every step, in bytes.

        Zero when there is none to size, which is also what keys the figures a
        model leaves behind: two pipelines of the same denoiser want the same room.
        """
        footprint = getattr(self.denoiser(), "get_memory_footprint", None)

        return footprint() if footprint is not None else 0

    def read_run_memory(self) -> None:
        """Read what the loaded model asks of the GPU, off its own architecture.

        Walked once, at load, and kept for the session: what a block writes and
        what a decode holds are fixed by the shapes the model was built with, so
        one reading answers every resolution, including those never generated.

        This used to be measured instead, off the reserve a generation peaked at,
        divided by the picture it made and carried into the next size up. Two
        things break that. The reserve is not the run: an allocator under no
        pressure keeps the blocks it cached, so on a GPU with room to spare the
        figure approaches the whole GPU whatever the run really held. And the
        division assumes a cost proportional to the picture, where a good part of
        what a generation reserves doesn't move with it, so the fixed part gets
        multiplied along with the rest. Both err the same way, upwards, and the
        larger the GPU the further: the model is then refused a seat it had the
        room for, at every resolution above the one that was measured, and it
        crosses the bus at every step for nothing.
        """
        self.run_memory = read_architecture(
            self.denoiser(), getattr(self.instance, "vae", None)
        )

        if self.run_memory is None:
            logger.warning("Can't read what this model asks of the GPU: sizing blind.")

    def run_bytes(self, pixels: int) -> int:
        """What a generation of this picture asks beyond the weights, in bytes.

        Falls back on the whole GPU where the architecture couldn't be read, which
        is the safe answer: the denoiser then streams, and the pass gets tiled.

        Args:
            pixels: Pixels of the picture to generate.
        """
        if self.run_memory is None:
            return self.total_memory

        return self.run_memory.run_bytes(pixels)

    def decode_bytes(self, pixels: int) -> int:
        """What decoding this picture asks beyond the weights, in bytes.

        Args:
            pixels: Pixels of the picture to generate.
        """
        if self.run_memory is None:
            return self.total_memory

        return self.run_memory.decode_bytes(pixels)

    def watch_run(self, stream_pixels: int) -> None:
        """Note what the coming generation is to be read against.

        Args:
            stream_pixels: Pixels the denoiser reads at every step.
        """
        self.warmed_up_run = self.warms_up_next_run

        if self.warms_up_next_run:
            self.kernel_cache_counts = count_cached_kernels()
            logger.info(
                "Warming up Triton and PyTorch Inductor caches... "
                "Next generation will be longer."
            )

        self.warmed_up_streams.add(stream_pixels)
        self.warms_up_next_run = False

    def log_compiled_kernels(self) -> None:
        """Say what the generation a pipeline warmed up on wrote to each cache.

        Nothing written means the caches already held it all, and the generation
        was the longer one for putting the weights on the GPU alone.
        """
        counts = count_cached_kernels()

        # Read off the snapshot, never off the caches: without one there is no
        # growth to report. Named, never counted: how many entries a cache took
        # answers nothing, where one growing alone points at the other's
        # directory.
        grown = [
            name
            for name, held in self.kernel_cache_counts.items()
            if counts.get(name, 0) > held
        ]

        logger.info(
            f"Compilation wrote to {' and '.join(grown)} "
            f"cache{'s' if len(grown) > 1 else ''}."
            if grown
            else "Compilation wrote nothing: these kernels were previously cached."
        )

    def fit_to_resolution(
        self, width: int, height: int, reference_pixels: int = 0
    ) -> None:
        """Set the pipeline up for the size of the picture about to be made.

        Args:
            width: Width of the picture to generate, in pixels.
            height: Height of the picture to generate, in pixels.
            reference_pixels: Pixels of the references the denoiser reads beside
                the picture, none of them where a family conditions otherwise.
        """
        # Blocks kept from the generation before are sized for another picture,
        # and until handed back they read as taken by the measures below.
        clear_device_cache(garbage_collection=True)

        pixels = width * height

        # A reference joins the picture in one stream, at its own size, and every
        # step carries both. The decode stays the picture's, never reaching them.
        stream_pixels = pixels + reference_pixels

        # The figures below are the stream's, so say what the stream is made of.
        if reference_pixels:
            logger.info(
                f"Denoising {stream_pixels / PIXELS_PER_MEGAPIXEL:.1f}MP: the "
                f"{pixels / PIXELS_PER_MEGAPIXEL:.1f}MP output and "
                f"{reference_pixels / PIXELS_PER_MEGAPIXEL:.1f}MP of reference."
            )

        # All the generation before still owes, nothing being measured off it.
        if self.warmed_up_run:
            self.log_compiled_kernels()
            self.warmed_up_run = False

        # Compiled per shape, so a size never generated is a size still to
        # compile. Settled before the seat below, which is weighed on it.
        self.warms_up_next_run = stream_pixels not in self.warmed_up_streams

        # What the run wants beyond the weights, and all that the denoiser's seat
        # is weighed against.
        self.run_margin = self.run_bytes(stream_pixels)

        # What every arrival leaves free behind it: the loop's margin alone. Read
        # on the decode's instead, the shortest phase set the room every other
        # one has to leave, and the strategy emptied the GPU on each arrival.
        self.memory_reserve = self.run_margin

        # Who may sit beside the denoiser is another question, and the widest
        # phase answers it: an encoder resident through a decode that wants the
        # rest of the card leaves the pass nowhere to go, and was watched to be
        # evicted for the denoiser and back, twice a generation. Called once, it
        # pays the bus; neither figure here can take the denoiser's seat.
        #
        # The whole pass is that phase only where the pass runs whole. A tiled one
        # is bounded by its tile, and asked for the picture's figure it names a
        # reserve as large as the card, which reads downstream as unreachable and
        # leaves the residency untouched. Told apart here as `tile_vae_if_needed`
        # tells `decode_margin` apart, and off the same `tiles_decode`, so the two
        # can't disagree.
        self.residency_reserve = (
            self.run_margin
            if self.tiles_decode(pixels)
            else max(self.run_margin, self.decode_bytes(pixels))
        )

        # A reserve larger than the GPU is not an instruction anything can carry
        # out: it asks for a figure no eviction ever reaches. The seat is weighed
        # on the run's margin, never on these, so capping them frees no one.
        offered = self.free_memory_without_ours()

        if offered:
            self.memory_reserve = min(self.memory_reserve, offered)
            self.residency_reserve = min(self.residency_reserve, offered)

        self.stream_denoiser_if_needed(stream_pixels)
        self.tile_vae_if_needed(pixels)

        # Last, so that what it records is the residency just settled on.
        if self.offload_strategy is not None:
            self.watch_run(stream_pixels)

    def stream_denoiser_if_needed(self, pixels: int) -> None:
        """Take the denoiser off the GPU when the picture leaves it no seat.

        Where the seat is granted or refused. Every other arbitration here runs
        on what this leaves. `DENOISER_RANK` says why it gets to go first.

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

        That margin is the run's, though, not the GPU's. Sized on anything that
        grows with the card rather than with the picture, this refuses the seat
        exactly where the room to hold it is largest. `RESERVED_MEMORY` is the one
        share of the card kept out of the arbitration, and says what for.

        A generation that compiles is weighed on what it really asks, which is
        several times a settled run. That is the whole of what a card's size
        decides here: a large one absorbs the difference and never notices, where
        a small one lends the seat back for the one generation that pays for the
        rest. Weighed on a settled run instead, a denoiser that fits by every
        figure saturated the card while compiling and spent twenty seconds on a
        step that costs six.

        Args:
            pixels: Pixels of the picture to generate.
        """
        if self.instance is None or self.offload_strategy is None:
            return

        weights = self.denoiser_footprint()

        if not weights:
            return

        needed = weights + self.projected_run()
        available = self.free_memory_without_ours()
        streams = self.denoiser_streams()

        # The seat is not the only thing a resolution settles: who sits beside the
        # denoiser was arbitrated against the reserve of the picture before, and a
        # set that fitted a small one overflows on a larger.
        if streams == self.streams_denoiser and self.residency_reserve == self.settled:
            return

        if streams != self.streams_denoiser:
            logger.info(
                f"{pixels / PIXELS_PER_MEGAPIXEL:.1f}MP needs "
                f"{needed / 1024**3:.1f}GB, "
                f"{'compiling ' if self.warms_up_next_run else ''}"
                f"{'adapted ' if self.adapter_footprint() else ''}run included, "
                f"GPU offers {available / 1024**3:.1f}GB "
                f"less {RESERVED_MEMORY / 1024**3:.1f}GB "
                f"reserved, so we {'stream' if streams else 'seat'} the denoiser."
            )

        self.remove_hooks()
        self.offload_to_cpu(self.total_memory, stream_denoiser=streams)

    def denoiser_streams(self) -> bool:
        """Will the denoiser cross the bus rather than sit on the GPU?

        The one question that frees the seat while the loop still wants it, and
        it is asked once, before the loop starts: too short here and the denoiser
        streams for the whole resolution. Only the decode takes the seat back,
        and only once the last step is over.

        Takes no resolution: the picture reaches this through `run_margin`, which
        the caller sizes on it beforehand. Passing it as well invites the two to
        disagree.
        """
        weights = self.denoiser_footprint()

        if not weights:
            return self.streams_denoiser

        return (
            weights + self.projected_run()
            > self.free_memory_without_ours() - RESERVED_MEMORY
        )

    def projected_run(self) -> int:
        """What the coming run is weighed as asking beyond the weights, in bytes.

        The margin its resolution settled, several times over while the graphs
        are still to compile, and doubled again where an adapter wraps the
        layers. `WARMING_UP_RUN` and `ADAPTED_RUN` say what each is measured on.

        Charged to the seat alone: the reserve every arrival leaves behind
        answers another question, and one nothing can reach empties the GPU on
        each of them.
        """
        run = self.run_margin * (WARMING_UP_RUN if self.warms_up_next_run else 1)

        return run * ADAPTED_RUN if self.adapter_footprint() else run

    def tiles_decode(self, pixels: int) -> bool:
        """Will this picture be decoded in tiles rather than whole?

        Against the budget alone, every weight the pass finds in its way being
        evictable by then: the encoders are done conditioning and the denoiser
        has run its last step, so what they hold is room the pass can have for
        the price of a trip. A pass too wide for the whole card is another
        matter, and tiling bounds it by the tile at the price of faint seams.

        This used to answer `True` for every streamed denoiser whatever the
        budget said, tiling a 0.9MP pass beside a budget of 1.7MP. Streaming and
        tiling are not one question, and joining them let each rearrange the
        other's answer.

        Args:
            pixels: Pixels of the picture to generate.
        """
        # An unmeasurable GPU gets the safe path rather than an optimistic one.
        if self.memory_budget is None:
            return True

        return self.decode_bytes(pixels) > self.memory_budget

    def tile_vae_if_needed(self, pixels: int) -> None:
        """Tile the VAE work only for the pictures this GPU can't handle in one pass.

        The VAE encodes and decodes the picture as a whole, so its peak memory
        grows with the resolution and, past a point, it alone fills the GPU. Tiling
        bounds that peak but can leave faint seams, so it's a trade only worth
        making when the picture wouldn't go through otherwise.

        What the pass holds is the decoder's own, read off its stages: they end at
        the resolution of the picture, where the widest of them sets the peak, and
        two families put a different number of channels through that last stage.
        A figure carried from one decoder to another seams pictures that had the
        room to go through whole.

        Args:
            pixels: Pixels of the picture to generate.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None:
            return

        needs_tiling = self.tiles_decode(pixels)

        # What the pass itself needs, the reserve standing in where it can't be
        # sized: tiling bounds a pass unmeasurably.
        self.decode_margin = (
            self.memory_reserve if needs_tiling else self.decode_bytes(pixels)
        )

        # The run's own margin is not asked for here: sized on it, the strategy
        # would evict every sibling on each arrival, where the allocator gives
        # ground instead. Only the denoiser's seat is weighed against it.
        if self.offload_strategy is not None:
            self.offload_strategy.memory_reserve_margin = self.memory_reserve

            logger.info(
                f"Decoding {pixels / PIXELS_PER_MEGAPIXEL:.1f}MP "
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
        # of its components manager, which hooks every component, including those
        # too large to stay on the GPU.
        device = get_execution_device()

        # Two figures, two questions: what each arrival leaves behind is the
        # loop's margin, who may sit at all is the widest phase. A set that fits
        # the loop and not the decode evicts itself all session.
        memory_reserve = self.memory_reserve
        residency_reserve = self.residency_reserve
        self.settled = residency_reserve

        offload_strategy = DenoiserFirstOffloadStrategy(
            memory_reserve_margin=memory_reserve
        )
        self.offload_strategy = offload_strategy

        streamed: dict[str, str] = {}

        def stream(
            name: str, component: torch.nn.Module, reason: str, chosen: bool = True
        ) -> None:
            """Leave a component on CPU, feeding the GPU one leaf at a time.

            Being picked for it is an optimisation; being forced into it by a
            component the GPU can't hold is a limit worth warning about.

            A seat given up settles every component again, nearly all of them the
            same way as before, so only a reason that changed earns a line.

            Diffusers' offload rather than Accelerate's `cpu_offload`, for the
            stream it fetches on: the next leaf crosses the bus while the current
            one computes, where `cpu_offload` waits for each. Measured on a Krea 2
            block, 13ms of bus a step against 321ms. `leaf_level` and no other:
            `block_level` on that stream measured slower than `cpu_offload`.

            Host memory is left pageable. Pinning buys a tenth of an already small
            figure and can't be paged back out, which a machine holding a large
            model in little memory pays for everywhere else.
            """
            apply_group_offloading(
                component,
                onload_device=device,
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
                use_stream=True,
                record_stream=True,
                non_blocking=True,
                low_cpu_mem_usage=True,
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
            candidates, residency_reserve, stream
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
        counts those weights as taken: the loop below sends off components the
        GPU had the room to seat, and a reload that changed nothing costs them a
        crossing at every generation after it.

        Nothing here weighs what would be left against the room a run wants. The
        set not fitting is the whole of the question, and a component whose next
        use is furthest off is the one to send off whether or not the denoiser
        ends up holding its seat: that seat is weighed against the run's own
        margin by the caller, at every resolution, where this also runs at load
        with no resolution in hand. A test on the reserve here answers a different
        question, and answering it early leaves the set that doesn't fit intact,
        its components evicting one another for the whole session.

        Args:
            candidates: Name, component and footprint of what could stay on GPU.
            memory_reserve: GPU memory the resident set has to leave free, in
                bytes: the widest phase of a generation, not the loop's alone.
            stream: Callback leaving a component on CPU.

        Returns:
            The candidates still meant to stay on the GPU.
        """
        # The components streamed just above handed their blocks to the allocator,
        # not to the driver: unreclaimed, that memory is counted by nobody and the
        # loop below streams components the GPU has the room to seat.
        clear_device_cache(garbage_collection=True)

        free_memory = self.free_memory_without_ours()

        if not free_memory:
            return candidates

        # A reserve no eviction can reach, an empty GPU being still short of it,
        # buys nothing below: every crossing is paid for room the run never gets.
        if memory_reserve >= free_memory:
            return candidates

        # A denoiser absent from the candidates is already on the bus, put there by
        # this GPU's size or by the resolution.
        denoiser_on_bus = not any(
            eviction_rank(candidate[0]) >= DENOISER_RANK for candidate in candidates
        )

        # Least called first, largest of those: most room bought per crossing.
        for name, component, footprint in sorted(
            candidates, key=lambda c: (eviction_rank(c[0]), -c[2])
        ):
            resident = sum(size for _, _, size in candidates)
            fits = resident + memory_reserve <= free_memory

            # A streamed denoiser asks the strategy for nothing, its submodules
            # arriving by a hook of their own: the room an encoder holds is room
            # its flow never gets, so the encoders leave whether they fit or not.
            if fits and not (denoiser_on_bus and eviction_rank(name) == ENCODER_RANK):
                break

            if eviction_rank(name) >= DENOISER_RANK:
                break

            # Fitting and streamed anyway is the case just above, told apart so
            # the line doesn't blame a shortage of room there was none of.
            reason = (
                "room the streamed denoiser would never use"
                if fits
                else "the least called of what's resident"
            )
            stream(name, component, f"{footprint / 1024**3:.1f}GB freed, {reason}")
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

        The denoiser is among them, and this is the one place it ever is: the
        last step is over by the time this runs, so its seat is worth the trip
        back and no more. Kept seated through a 1080p decode, the pass had to be
        tiled instead and the generation went from 4.2 to 23.3 seconds a step.
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

                # Diffusers keeps its own registry, out of Accelerate's reach.
                # All three names, not the offload alone: the two beside it are
                # dropped on the first forward, so a component streamed twice
                # before generating anything still carries them, and registering
                # them again raises.
                registry = getattr(component, "_diffusers_hook", None)

                for name in (
                    _GROUP_OFFLOADING,
                    _LAZY_PREFETCH_GROUP_OFFLOADING,
                    _LAYER_EXECUTION_TRACKER,
                ):
                    if registry is not None:
                        registry.remove_hook(name, recurse=True)

        # Each hook names its component and, through the eviction ring, every
        # other, so this list holds the set before until it is cleared. Cleared
        # here rather than overwritten later, the arbitration that follows
        # measuring the GPU in between.
        self.offload_hooks = []

    @contextmanager
    def unhooked(self) -> Iterator[None]:
        """Drop the CPU offload hooks for the time of a weight edit.

        Loading a LoRA, Diffusers removes them itself, then restores them with
        `enable_sequential_cpu_offload()`, which a modular pipeline lacks: it
        raises, leaving every component unhooked on CPU. Only the Accelerate
        hooks are visible to it, hence a crash reserved to a pipeline with a
        component streamed, whatever put it on CPU.
        """
        if (
            self.instance is None
            or isinstance(self.instance, DiffusionPipeline)
            or self.offload_strategy is None
            or not self.total_memory
        ):
            yield
            return

        self.remove_hooks()

        try:
            yield
        finally:
            # The edit outlives every compiled graph, so the coming run warms up
            # again at every size. Settled first: the projection below weighs it.
            self.warmed_up_streams.clear()
            self.warms_up_next_run = True

            # Read again under the footprint the edit moved. The architecture is
            # the one it was, adapters changing no shape the walk reads, but that
            # footprint is what files the reading and what a pass was seen to be
            # handed, so neither is found under the weights that came out of this.
            self.read_run_memory()

            # The new weights come in unhooked, and heavier. Projected again
            # rather than handed back the seat the weights before them earned: an
            # adapter landing on a seated denoiser was watched to take the card
            # past what it holds, where the driver pages instead of failing and a
            # step went from 4.5 to 27.9 seconds for the rest of the session.
            self.offload_to_cpu(
                self.total_memory, stream_denoiser=self.denoiser_streams()
            )

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

            # Its compiled graphs hold CUDA graph pools no `empty_cache()`
            # reaches: kept, they settle the residency of the model arriving on a
            # smaller GPU than it has. The kernels stay on disk, so what this
            # costs is a cache read on the warming-up generation.
            try:
                torch.compiler.reset()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Can't reset the compiler between models: {e}")

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

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
