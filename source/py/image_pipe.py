"""Diffusion pipeline wrapper."""

import gc
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from inspect import signature
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
from torch._dynamo.utils import counters

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
from source.py.qwen21_flash import install_qwen21_flash_attn
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

KEPT_KEYS_ARGUMENT = "kv_cache"
"""Argument a denoiser keeping keys and values across steps is handed them in.

Named alike by the Diffusers families keeping them. The one returning them
instead clears them itself once its loop is over.
"""

PIXELS_PER_MEGAPIXEL = 1e6
"""Pixels in a megapixel, the unit pictures are sized in here.

Decimal, as the word is used of pictures, where memory stays binary. The bases
don't meet, which costs nothing as long as the same conversion sizes every
picture, no figure here being per megapixel to begin with.
"""

LATENT_TILE_SIDE = 32
"""Latent cells along a side of the smallest tile a decode is given.

What Diffusers' defaults come to against a VAE compressing by eight, their 256
pixels over that ratio. Held in latent, which is what the blend between two
tiles works with.
"""

LATENT_TILE_STEP = 8
"""Latent cells a tile grows by where the budget holds a larger one.

A rung the budget can't be spent on is a join paid for nothing. Climbed too
coarsely, the ladder steps from a band well under budget straight onto the
picture's own height, where it is refused for tiling nothing, and the decode
settles on the rung below with room to spare on the card. The shorter the band,
the further into the picture its join falls, and the taller one would have cost
the card nothing it had.

Finer is not better by construction, the join not weakening with the band at
every size. It is better for reaching the sizes the budget already held.
"""

LATENT_TILE_OVERLAP = 2
"""The part of a tile two neighbours share and are blended across, inverted.

A half, where Diffusers' own figures come to a quarter. What the join carries
is decided just inside a tile's trailing edge, where the decode has invented
the context it could not see and both neighbours are wrong: sharing half a tile
leaves that edge almost no weight where it lands.

It takes the colour out of the join, not the join. What survives is a step of
brightness, which no size and no overlap measured here removes, a whole pass
alone being exact. Widening further is not a direction: past a half the blend
spans so much of the tile that the join returns, paid for twice over.
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


def compiled_graphs() -> int:
    """Graphs Dynamo has compiled so far in this process.

    Read before and after a call to tell one that compiled from one that ran
    kernels it already had: the count doesn't move on a cache hit. The count of
    frames beside it would say the same, and stays empty under SDNQ's compile.
    """
    return counters["stats"]["unique_graphs"]


def release_tensors(holder: object, seen: set[int] | None = None) -> None:
    """Let go of every tensor an object holds, through its attributes and lists.

    For containers a pipeline builds and keeps as a local of its own call, which
    nothing outside can drop, and whose families lay them out each their own way.

    Args:
        holder: The object to empty.
        seen: Identities of the objects already emptied, against cycles.
    """
    seen = seen if seen is not None else set()

    if id(holder) in seen or isinstance(holder, torch.nn.Module):
        return

    seen.add(id(holder))

    if isinstance(holder, list):
        for index, value in enumerate(holder):
            if torch.is_tensor(value):
                holder[index] = None
            else:
                release_tensors(value, seen)
    elif isinstance(holder, dict):
        for key, value in holder.items():
            if torch.is_tensor(value):
                holder[key] = None
            else:
                release_tensors(value, seen)
    elif isinstance(holder, tuple):
        for value in holder:
            release_tensors(value, seen)
    elif hasattr(holder, "__dict__"):
        for name, value in vars(holder).items():
            if torch.is_tensor(value):
                setattr(holder, name, None)
            else:
                release_tensors(value, seen)


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

        self.cache_margin = 0
        """GPU memory the keys and values the denoiser keeps take, in bytes; 0
        where the pipeline keeps none, or until its references are encoded."""

        self.kept_keys: weakref.WeakSet = weakref.WeakSet()
        """What the denoiser was handed this run to keep its keys and values in,
        one per guidance branch. Weak: the pipeline owns them."""

        self.picture: tuple[int, int] = (0, 0)
        """Width and height of the picture the coming run makes, in pixels."""

        self.references_to_encode = 0
        """References the coming run has yet to hand its VAE; the seat waits
        for the last."""

        self.reference_pixels = 0
        """Pixels of the references encoded so far this run, at their encoded
        size."""

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

            # These families mask their attention and the backend refuses any
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
            elif model.family == "Qwen-Image 2.1":
                try:
                    install_qwen21_flash_attn(self.instance)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"FlashAttention is not available for Qwen-Image 2.1: {e}"
                    )

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
        self.streams_denoiser = False
        self.settled = 0
        self.warmed_up_streams = set()
        self.warms_up_next_run = True
        self.warmed_up_run = False
        self.kernel_cache_counts = {}
        self.streamed_reasons = {}
        self.run_margin = 0
        self.decode_margin = 0
        self.cache_margin = 0
        self.kept_keys = weakref.WeakSet()
        self.picture = (0, 0)
        self.references_to_encode = 0
        self.reference_pixels = 0

        self.log_gpu_state()
        self.read_run_memory()

        # Inner first: the tiling an encode is given has to be the last thing
        # settled before it runs, after the fit the count may trigger.
        self.tile_encodes_on_their_size()
        self.count_references_when_encoded()
        self.watch_kept_keys()
        self.collect_after_compiling()

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

        self.memory_budget = self.measure_memory_budget(
            memory_info[0] if memory_info else None
        )

        return model

    def measure_memory_budget(self, free_memory: int | None) -> int | None:
        """Measure the GPU memory a single VAE pass can count on, in bytes.

        With offload, only the VAE's weights are deducted; otherwise, all
        components are counted. The budget starts from free memory: the desktop
        takes its cut first, and a share of the GPU would ignore it.

        What it decides is not how a seam falls but whether there is one: no
        tiling of this decoder was measured seamless, every size and overlap
        leaving a step of brightness at the join where the whole pass is exact.

        Args:
            free_memory: GPU memory free, in bytes, with none of our weights
                counted as taken, or `None` if it couldn't be measured.

        Returns:
            The budget, or `None` if it can't be measured.
        """
        if free_memory is None or self.instance is None:
            return None

        def footprint_of(component) -> int:
            get_footprint = getattr(component, "get_memory_footprint", None)

            return get_footprint() if get_footprint is not None else 0

        # Model offload leaves the decoder on the GPU for the whole of its pass,
        # and a budget blind to it tiles nothing and overflows instead.
        if self.offload_strategy is not None:
            resident = footprint_of(getattr(self.instance, "vae", None))
        else:
            # Nothing evicts on Mac: the whole pipeline stays on the GPU.
            resident = sum(map(footprint_of, self.instance.components.values()))

        # Growing a band to all remaining VRAM leaves no room for allocator
        # overhead or changes in desktop usage. Keep the same headroom as the
        # denoiser: a decode can otherwise spill into host memory on Windows.
        budget = max(free_memory - resident - RESERVED_MEMORY, 0)

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

    def cache_bytes(self, pixels: int) -> int:
        """What the keys and values the denoiser keeps of these pixels take, in bytes.

        A pipeline keeping them computes the references once, on the first step,
        and reads them back after. That pass is counted with the run; the cache
        it leaves is not. Every attention layer keeps its own, and they stay on
        the card to the last step: `release_kept_keys` hands them back before the
        decode, which would otherwise find them in its way.

        The prompt's tokens are kept too and go uncounted, being known only once
        encoded, and far fewer than a reference's.

        Args:
            pixels: Pixels of the references. The picture is never kept.
        """
        if not pixels or not self.keeps_keys():
            return 0

        if self.run_memory is None:
            return self.total_memory

        return self.run_memory.cache_bytes(pixels)

    def keeps_keys(self) -> bool:
        """Does the pipeline keep the keys and values of what precedes the picture?

        Asked of the pipeline, not the denoiser: an architecture able to keep them
        doesn't say the loop driving it does. Read at the argument's default,
        which nothing here overrides.
        """
        return self.call_default("use_kv_cache") is True

    def call_default(self, name: str) -> object:
        """What the pipeline takes for a call argument left out, `None` without one.

        Args:
            name: Name of the argument.
        """
        if self.instance is None:
            return None

        blocks = getattr(self.instance, "blocks", None)

        if blocks is not None:
            return next(
                (param.default for param in blocks.inputs if param.name == name),
                None,
            )

        parameter = signature(type(self.instance).__call__).parameters.get(name)

        if parameter is None or parameter.default is parameter.empty:
            return None

        return parameter.default

    def encode_bytes(self, pixels: int) -> int:
        """What encoding this picture asks beyond the weights, in bytes.

        Args:
            pixels: Pixels of the picture to encode.
        """
        if self.run_memory is None:
            return self.total_memory

        return self.run_memory.encode_bytes(pixels)

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

    def fit_to_resolution(self, width: int, height: int, references: int = 0) -> None:
        """Set the pipeline up for the size of the picture about to be made.

        Settles what the phases before the loop need. The loop's own arrangement
        waits for the references, if any, to be encoded.

        Args:
            width: Width of the picture to generate, in pixels.
            height: Height of the picture to generate, in pixels.
            references: Reference images the denoiser reads beside the picture,
                none of them where a family conditions otherwise.
        """
        # All the generation before still owes, nothing being measured off it.
        if self.warmed_up_run:
            self.log_compiled_kernels()
            self.warmed_up_run = False

        # A reference joins the stream at a size the pipeline chooses and says
        # nowhere beforehand, so each is counted as it is encoded.
        self.picture = (width, height)
        self.references_to_encode = references
        self.reference_pixels = 0
        self.kept_keys.clear()

        self.fit_stream()

    def fit_stream(self) -> None:
        """Arbitrate the GPU for the stream the denoiser is known to read so far.

        Runs before the call on the picture alone, then again once the last
        reference is encoded. The seat waits for that second pass: settled twice,
        every hook attached in between would send its component back to the CPU.
        """
        width, height = self.picture

        # Blocks kept by what ran before are sized for something else, and until
        # handed back they read as taken by the measures below.
        clear_device_cache(garbage_collection=True)

        pixels = width * height
        waits_for_references = self.references_to_encode > 0

        # A reference joins the picture in one stream, and every step carries
        # both. The decode stays the picture's, never reaching them.
        stream_pixels = pixels + self.reference_pixels

        # The figures below are the stream's, so say what the stream is made of.
        if self.reference_pixels:
            logger.info(
                f"Denoising {stream_pixels / PIXELS_PER_MEGAPIXEL:.1f}MP: the "
                f"{pixels / PIXELS_PER_MEGAPIXEL:.1f}MP output and "
                f"{self.reference_pixels / PIXELS_PER_MEGAPIXEL:.1f}MP of reference."
            )

        # Compiled per shape, so a size never generated is a size still to
        # compile. Settled before the seat below, which is weighed on it.
        self.warms_up_next_run = stream_pixels not in self.warmed_up_streams

        # What the run wants beyond the weights, and what the pipeline keeps from
        # it: all that the denoiser's seat is weighed against.
        self.run_margin = self.run_bytes(stream_pixels)
        self.cache_margin = self.cache_bytes(self.reference_pixels)

        # What every arrival leaves free behind it: the loop's margin alone. Read
        # on the decode's instead, the shortest phase set the room every other
        # one has to leave, and the strategy emptied the GPU on each arrival.
        # The cache is the loop's too, and the loop's alone.
        loop = self.run_margin + self.cache_margin
        self.memory_reserve = loop

        # Who may sit beside the denoiser is another question, and the widest
        # phase answers it: an encoder resident through a decode that wants the
        # rest of the card leaves the pass nowhere to go, and was watched to be
        # evicted for the denoiser and back, twice a generation. Called once, it
        # pays the bus; neither figure here can take the denoiser's seat.
        #
        # The whole pass is that phase only where the pass runs whole. A tiled one
        # is bounded by its tile, and asked for the picture's figure it names a
        # reserve as large as the card, which reads downstream as unreachable and
        # leaves the residency untouched. Keep the loop's reserve for a tiled
        # pass here; `decode_margin` frees the selected band's room on demand,
        # after the denoiser has finished.
        self.residency_reserve = (
            loop if self.tiles_decode(pixels) else max(loop, self.decode_bytes(pixels))
        )

        # A reserve larger than the GPU is not an instruction anything can carry
        # out: it asks for a figure no eviction ever reaches. The seat is weighed
        # on the run's margin, never on these, so capping them frees no one.
        offered = self.free_memory_without_ours()

        if offered:
            self.memory_reserve = min(self.memory_reserve, offered)
            self.residency_reserve = min(self.residency_reserve, offered)

        if not waits_for_references:
            self.stream_denoiser_if_needed(stream_pixels)

        self.tile_vae_if_needed(pixels, width, height, final=not waits_for_references)

        # Last, so that what it records is the residency just settled on. With
        # references, it runs after the encoders, whose compilation then goes
        # unreported rather than walk both caches before every edit.
        if self.offload_strategy is not None and not waits_for_references:
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
                f"{'adapted ' if self.adapter_footprint() else ''}run "
                f"{'and cache ' if self.cache_margin else ''}included, "
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

        The cache is added after both: compiling doesn't grow it, and an adapter
        leaves the width of a key or a value as it was.

        Charged to the seat alone: the reserve every arrival leaves behind
        answers another question, and one nothing can reach empties the GPU on
        each of them.
        """
        run = self.run_margin * (WARMING_UP_RUN if self.warms_up_next_run else 1)
        run = run * ADAPTED_RUN if self.adapter_footprint() else run

        return run + self.cache_margin

    def tiles_decode(self, pixels: int) -> bool:
        """Will this picture be decoded in tiles rather than whole?

        Against the budget alone, every weight the pass finds in its way being
        evictable by then: the encoders are done conditioning and the denoiser
        has run its last step, so what they hold is room the pass can have for
        the price of a trip. A pass too wide for the whole card is another
        matter, and tiling bounds it by the band at the price of a join, no cut
        of this decoder having been measured exact.

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

    def tile_vae_if_needed(
        self, pixels: int, width: int, height: int, final: bool = True
    ) -> None:
        """Tile the VAE work only for the pictures this GPU can't handle in one pass.

        The VAE encodes and decodes the picture as a whole, so its peak memory
        grows with the resolution and, past a point, it alone fills the GPU. Tiling
        bounds that peak and leaves a join across the picture, one no size and no
        overlap was measured to remove, so it's a trade only worth making when the
        picture wouldn't go through otherwise.

        What the pass holds is the decoder's own, read off its stages: they end at
        the resolution of the picture, where the widest of them sets the peak, and
        two families put a different number of channels through that last stage.
        A figure carried from one decoder to another seams pictures that had the
        room to go through whole.

        Args:
            pixels: Pixels of the picture to generate.
            width: Width of the picture to generate, in pixels.
            height: Height of the picture to generate, in pixels.
            final: Is this the arrangement the decode runs on? One settled
                before the references are encoded is applied but not logged.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None:
            return

        needs_tiling = self.tiles_decode(pixels)
        sides = (
            self.tile_sides(
                vae, width, height, self.decode_bytes, self.memory_budget or 0
            )
            if needs_tiling
            else None
        )

        # A grown band can ask for much more than the denoising loop. Free room
        # for that actual band, including the headroom withheld from its budget.
        # Families whose tiles can't be sized retain the existing fallback.
        if needs_tiling and sides is None:
            self.decode_margin = self.memory_reserve
        else:
            decode_pixels = min(sides[0], height) * width if sides else pixels
            self.decode_margin = self.decode_bytes(decode_pixels) + RESERVED_MEMORY

        # The run's own margin is not asked for here: sized on it, the strategy
        # would evict every sibling on each arrival, where the allocator gives
        # ground instead. Only the denoiser's seat is weighed against it.
        if self.offload_strategy is not None:
            self.offload_strategy.memory_reserve_margin = self.memory_reserve

        if self.offload_strategy is not None and final:
            logger.info(
                f"Decoding {pixels / PIXELS_PER_MEGAPIXEL:.1f}MP "
                f"{'tiled' if needs_tiling else 'in one pass'}, "
                f"evicting down to {self.decode_margin / 1024**3:.1f}GB free."
            )

        toggle = getattr(
            vae, "enable_tiling" if needs_tiling else "disable_tiling", None
        )

        if toggle is None:
            return

        if sides is None:
            toggle()

            return

        band, full_width, stride, full_stride = sides
        toggle(band, full_width, stride, full_stride)

        if not final:
            return

        logger.info(f"Decoding in {band}px bands, {stride}px apart.")

        # The ladder only climbs what the budget holds, so a band over budget is
        # the narrowest one, with nowhere lower to go. Said, as a slow decode
        # shows up nowhere else.
        asked = self.decode_bytes(band * width)
        budget = self.memory_budget

        if budget is not None and asked > budget:
            logger.warning(
                f"A {band}px band asks {asked / 1024**3:.1f}GB of the "
                f"{budget / 1024**3:.1f}GB left to the decode: it may spill into "
                f"host memory."
            )

    def tile_sides(
        self,
        vae: torch.nn.Module,
        width: int,
        height: int,
        cost: Callable[[int], int],
        budget: int,
    ) -> tuple[int, int, int, int] | None:
        """A band of the picture and the stride between two, in pixels, or `None`.

        Diffusers sizes a tile in pixels, and every family's figures were written
        against a VAE compressing by eight. One compressing twice as hard is
        handed half the latent for the same tile, and the blend then runs on too
        few cells to hide the seam: on a 16x decoder, the defaults streaked a
        picture a hundredfold against the same latent decoded whole. Sized here
        in latent, multiplied back out, then grown to the largest the budget
        holds.

        Cut across the height alone, the band keeping the whole width. A grid
        joins its neighbours twice over and one of those joins runs down the
        picture, which on a portrait is down a face; bands leave a single join,
        lying across, and cost the same memory for the same area decoded at
        once. No blend was measured exact, so what is chosen here is how much of
        the picture a join is allowed to cross, never whether there is one.

        The caller also uses this area to free enough room before decoding.

        Args:
            vae: The VAE whose pass is to be tiled.
            width: Width of the picture the pass reads or writes, in pixels.
            height: Height of that picture, in pixels.
            cost: What the pass asks for a given number of pixels, in bytes.
            budget: GPU memory the pass can count on, in bytes.
        """
        ratio = getattr(vae, "spatial_compression_ratio", 0)
        enable = getattr(vae, "enable_tiling", None)

        if not ratio or enable is None:
            return None

        try:
            sized = "tile_sample_min_height" in signature(enable).parameters
        except (TypeError, ValueError):
            return None

        # Several families take no sizes, and handing them one raises.
        if not sized:
            return None

        side = LATENT_TILE_SIDE

        while True:
            grown = (side + LATENT_TILE_STEP) * ratio

            # A band as tall as the picture tiles nothing, the budget answering
            # for the rest. What it costs is its own area, the width being the
            # picture's whole.
            if grown >= height or cost(grown * width) > budget:
                break

            side += LATENT_TILE_STEP

        band = side * ratio

        # Across the width, a stride as wide as the picture leaves one column,
        # so nothing is blended that way and no join runs down the picture.
        return band, width, band - band // LATENT_TILE_OVERLAP, width

    def offload_to_cpu(self, total_memory: int, stream_denoiser: bool = False) -> None:
        """Offload the pipeline weights to CPU, keeping on GPU what fits.

        Args:
            total_memory: The GPU memory, in bytes.
            stream_denoiser: Leave the denoiser on CPU too, the resolution
                leaving it no room next to the tensors of the run.
        """
        if self.instance is None:
            return

        self.streams_denoiser = stream_denoiser

        # Both pipeline types expose their components. Place them with the same
        # strategy, independently of the pipeline's own CPU offload helpers.
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
                # Only a component too large on its own is the GPU's limit: the
                # others are sent off by the room the run needs beside them.
                alone = footprint() > total_memory
                reason = (
                    "is too large for this GPU"
                    if alone
                    else f"doesn't fit beside the {memory_reserve / 1024**3:.1f}GB "
                    "this run needs"
                )
                stream(
                    name,
                    component,
                    f"its {footprint() / 1024**3:.1f}GB {reason}",
                    chosen=not alone,
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

        # Using Diffusers' _all_hooks would reactivate its standard offload at run end.
        self.offload_hooks = hooks
        self.free_gpu_before_decode()

        # Attaching or streaming a component hands its weights' blocks to the
        # allocator, not the driver, and an arrival with no sibling reclaims
        # nothing. Unreclaimed, the loop allocates around blocks shaped for
        # weights, and the driver pages the card for room the allocator holds.
        clear_device_cache(garbage_collection=True)

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
            # This wrapper runs before the VAE's onload hook. Its absent weights
            # will also need room after the siblings have been evicted.
            margin += tensor_bytes(vae) - tensor_bytes(vae, device=device)

            # A reference still waiting here never reached the VAE, so the loop
            # ran on the picture's arrangement alone.
            if self.references_to_encode:
                logger.warning(
                    f"{self.references_to_encode} reference(s) never reached the "
                    "VAE: the denoiser ran on a stream counted without them."
                )
                self.references_to_encode = 0

            # The loop is over, and what it kept is read by no step after it.
            self.release_kept_keys()

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

    def count_references_when_encoded(self) -> None:
        """Count each reference as the pipeline encodes it, then fit the stream.

        A pipeline hands each reference to its VAE, resized, after its encoders
        and before its first step: the first moment the size is known, and the
        last one the seat can still be settled.

        A strength-based pipeline's source image crosses it uncounted, being
        denoised in place of the picture rather than beside it.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None or getattr(vae.encode, "counts_references", False):
            return

        encode = vae.encode

        def encode_counting_references(*args, **kwargs):
            images = args[0] if args else kwargs.get("x")

            if self.references_to_encode and torch.is_tensor(images):
                # One call can hand over a batch of them, along its first axis.
                count = images.shape[0]

                self.reference_pixels += count * images.shape[-2] * images.shape[-1]
                self.references_to_encode = max(self.references_to_encode - count, 0)

                if not self.references_to_encode:
                    self.fit_stream()

            return encode(*args, **kwargs)

        setattr(encode_counting_references, "counts_references", True)  # noqa: B010
        vae.encode = encode_counting_references

    def tile_encodes_on_their_size(self) -> None:
        """Tile each encode on the picture it is handed, never on the one to make.

        Diffusers keeps one tiling per VAE for its encodes and decodes alike, and
        the decode's is cut to the picture to make: bands as wide as it is. A
        reference comes in at a size of its own, and one wider than the picture
        was cut into a band and the sliver left over, encoded alone with nothing
        to blend it against. The denoiser copies what it reads, so the edit came
        out with a strip down its edge.

        Each encode is then weighed on its own picture and its own encoder, and
        the decode's arrangement put back once it is done.
        """
        vae = getattr(self.instance, "vae", None)

        if vae is None or getattr(vae.encode, "tiles_on_its_size", False):
            return

        encode = vae.encode

        def encode_on_its_size(*args, **kwargs):
            images = args[0] if args else kwargs.get("x")

            if not torch.is_tensor(images):
                return encode(*args, **kwargs)

            # Whatever a family names its figures, Diffusers prefixes them alike.
            decode_tiling = {
                name: value
                for name, value in vars(vae).items()
                if name == "use_tiling" or name.startswith("tile_")
            }

            self.tile_encode(vae, images.shape[-1], images.shape[-2])

            try:
                return encode(*args, **kwargs)
            finally:
                for name, value in decode_tiling.items():
                    setattr(vae, name, value)

        setattr(encode_on_its_size, "tiles_on_its_size", True)  # noqa: B010
        vae.encode = encode_on_its_size

    def tile_encode(self, vae: torch.nn.Module, width: int, height: int) -> None:
        """Tile an encode of this picture only where it can't go through whole.

        A join in a reference is one the denoiser reads and reproduces, so the
        whole pass is worth more here than anywhere: it is refused only for want
        of room, against the encoder's own cost, which can be a third of what the
        decoder holds over the same picture.

        Args:
            vae: The VAE about to encode.
            width: Width of the picture to encode, in pixels.
            height: Height of the picture to encode, in pixels.
        """
        pixels = width * height
        budget = self.encode_budget(vae)
        asked = self.encode_bytes(pixels)
        needs_tiling = budget is None or asked > budget

        toggle = getattr(
            vae, "enable_tiling" if needs_tiling else "disable_tiling", None
        )

        if toggle is None:
            return

        sides = (
            self.tile_sides(vae, width, height, self.encode_bytes, budget or 0)
            if needs_tiling
            else None
        )

        if sides is None:
            toggle()
        else:
            toggle(*sides)

        if sides is not None:
            how = f"in {sides[0]}px bands"
        else:
            how = "tiled" if needs_tiling else "in one pass"

        logger.info(
            f"Encoding {pixels / PIXELS_PER_MEGAPIXEL:.1f}MP {how}, "
            f"asking {asked / 1024**3:.1f}GB"
            + (f" of {budget / 1024**3:.1f}GB." if budget is not None else ".")
        )

    def encode_budget(self, vae: torch.nn.Module) -> int | None:
        """GPU memory an encode can count on now, in bytes, or `None` if unknown.

        Read on the spot rather than taken from the budget read at load: an encode
        runs between the encoders and the loop, beside whatever the arbitration
        seated, and nothing is evicted for it. On a GPU without offload the load's
        figure stands, every component staying where it is.

        Args:
            vae: The VAE about to encode.
        """
        if self.offload_strategy is None:
            return self.memory_budget

        clear_device_cache(garbage_collection=True)
        memory_info = get_memory_info()

        if memory_info is None:
            return None

        # The encode's own hook brings in whatever of the weights is still away.
        absent = tensor_bytes(vae) - tensor_bytes(vae, device=get_execution_device())

        return max(memory_info[0] - absent - RESERVED_MEMORY, 0)

    def watch_kept_keys(self) -> None:
        """Note what the denoiser is handed to keep its keys and values in.

        Held weakly: the pipeline owns it, and a run that never reaches its
        decode must not leave it on the card.
        """
        denoiser = self.denoiser()

        if denoiser is None:
            return

        def note(module, args, kwargs):
            kept = kwargs.get(KEPT_KEYS_ARGUMENT)

            if kept is None:
                return

            try:
                self.kept_keys.add(kept)
            except TypeError:
                logger.warning(
                    f"Can't watch the {type(kept).__name__} the denoiser keeps its "
                    "keys in: the decode will find it in its way."
                )

        denoiser.register_forward_pre_hook(note, with_kwargs=True)

    def release_kept_keys(self) -> None:
        """Hand back the keys and values the denoiser kept, its loop being over.

        The pipeline holds them as a local of its call until the call returns, so
        they sit through the decode, which on a small card they leave short of the
        band it was sized for. Emptied in place: the container is the one thing
        nothing outside the call can drop.

        Called by the decode, which every pipeline here runs once, after its last
        step. A pipeline reading them after that would raise rather than go on
        without them.
        """
        for kept in list(self.kept_keys):
            release_tensors(kept)

        self.kept_keys.clear()

    def collect_after_compiling(self) -> None:
        """Hand back what compiling a call left behind, as soon as the call ends.

        Tracing a new shape leaves the tensors of the traced call in reference
        cycles, which only a collection reaches, and Python runs one when it sees
        fit. Meanwhile they stay on the card: the two steps compiling a Qwen-Image
        2.1 edit were watched to leave 2.3GB there, and a streamed loop spilling
        on them into host memory took 27 seconds a step where it took 2.3 with
        them collected.

        Only after a call that compiled, which Dynamo counts: a settled run leaves
        nothing to collect, and pays nothing to be asked.
        """
        if self.instance is None:
            return

        for component in self.instance.components.values():
            if not isinstance(component, torch.nn.Module):
                continue

            compiled_before = [0]

            def note(module, args, compiled_before=compiled_before):
                compiled_before[0] = compiled_graphs()

            def collect(module, args, output, compiled_before=compiled_before):
                if compiled_graphs() > compiled_before[0]:
                    clear_device_cache(garbage_collection=True)

            component.register_forward_pre_hook(note)
            component.register_forward_hook(collect)

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

        Diffusers' LoRA helpers manage their own offload modes, not our mix of
        resident and streamed components. Remove that mix before an edit and
        restore our strategy afterwards, for standard and modular pipelines.
        """
        if (
            self.instance is None
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
