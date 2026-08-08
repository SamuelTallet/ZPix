"""What a generation asks of the GPU, read off the architecture that runs it."""

import weakref
from dataclasses import dataclass
from itertools import chain

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary

from source.py.custom_logger import logger

READ_ARCHITECTURES: dict[tuple[int, str], "RunMemory"] = {}
"""What each architecture was read to cost, by denoiser footprint and VAE class.

The reading walks a model rather than a generation, so it holds for the life of
the process and for every resolution: a model swapped out and come back to is
already answered, and only a weight edit, which moves the footprint keying this,
asks the question again.
"""

PASS_THROUGH = (torch.nn.Dropout, torch.nn.Identity)
"""Layers that hand on the tensor they were given rather than write one.

Named here because every other layer writes something, and a walk that assumed
otherwise would drop whatever a family happens not to declare. These two are the
identity once a model is in evaluation mode, which is the only mode it runs in.
"""

LATENT_PROBE_SIDE = 8
"""Side of the latent square the decoder is walked on, in latent pixels.

Only a shape travels through that walk, so the side buys nothing by growing. It
has to clear the downsamplings a decoder inverts, an odd size rounding its way up
the stages and reporting widths no real pass writes.
"""


def stream_dtype_size(module: torch.nn.Module) -> int:
    """Bytes one element of a tensor flowing through a module takes.

    Read off the floating-point parameters, the integer ones being weights packed
    for storage where what flows between the layers stays in the dtype the module
    computes in. Taken by weight rather than from the first one found: a handful
    of layers are held at a wider dtype than the model runs in, and whichever of
    them a walk happens to reach first would double every figure resting on this.
    """
    sizes: dict[int, int] = {}

    for parameter in module.parameters():
        if parameter.is_floating_point():
            size = parameter.element_size()
            sizes[size] = sizes.get(size, 0) + parameter.numel()

    if not sizes:
        return torch.get_default_dtype().itemsize

    return max(sizes, key=lambda size: sizes[size])


def write_dtype_size(module: torch.nn.Module, stream_size: int) -> int:
    """Bytes one element of what a layer writes takes at the moment it writes it.

    Most layers write in the dtype the model computes in, which is also what
    flows between them. A layer whose weights are quantized doesn't: it
    accumulates a product of integers, rescales that by the scale it keeps
    beside those weights, and only then narrows the result down to the stream.
    The accumulator and the rescaled copy are both the shape of what the layer
    writes and both as wide as that scale, and they are alive together, so the
    widest layer of a block asks several times what its output alone would.

    Missing this read a first walk of these models at half what they allocate,
    and half a reading is worse than none here: the denoiser is seated where it
    doesn't fit, the driver backs the overflow with host memory, and the steps
    then cost far more than the crossing the seat was there to save.

    Args:
        module: The layer whose write is being sized.
        stream_size: Bytes an element flowing between the layers takes.
    """
    dequantizer = getattr(module, "sdnq_dequantizer", None)
    scale = getattr(module, "scale", None)

    if dequantizer is None or not getattr(dequantizer, "use_quantized_matmul", False):
        return stream_size

    if not torch.is_tensor(scale):
        return stream_size

    return 2 * scale.element_size()


def written_width(module: torch.nn.Module) -> int:
    """Width of the tensor a layer writes; 0 for a layer writing none.

    Features or channels under the names the two carry, and read as attributes
    rather than off the class: a quantized layer is no longer the linear it
    replaces, while it still says how wide it writes.
    """
    for attribute in ("out_features", "out_channels"):
        width = getattr(module, attribute, None)

        if isinstance(width, int):
            return width

    return 0


def parameter_width(module: torch.nn.Module) -> int:
    """Width a layer's own parameters say it works at; 0 where it holds none.

    A normalization keeps one weight per feature and, across the families, names
    that width nowhere twice the same way: under an attribute of its own, as a
    shape rather than a number, or not at all. The parameter is the one place
    they all agree, and it is read rather than the attributes so that a family
    naming nothing is sized as well as one that does.
    """
    widths = [
        parameter.shape[-1]
        for parameter in module.parameters(recurse=False)
        if parameter.dim() > 0
    ]

    return max(widths, default=0)


def read_width(module: torch.nn.Module) -> int:
    """Width a layer reads, where it says; 0 where it says nothing.

    A layer that reshapes nothing writes as wide as it reads, which is how the
    normalizations are sized: they name what they expect, never what they hand on.
    """
    shape = getattr(module, "normalized_shape", None)

    if isinstance(shape, (tuple, list)) and shape:
        return int(shape[-1])

    for attribute in ("num_features", "in_features", "in_channels"):
        width = getattr(module, attribute, None)

        if isinstance(width, int):
            return width

    return parameter_width(module)


def outer_block_lists(
    module: torch.nn.Module,
) -> list[tuple[str, torch.nn.ModuleList]]:
    """The lists of repeated blocks a module holds, by name, nested ones excluded.

    A block is the unit a model repeats for its depth, and what a pass holds at
    its widest is one of them. Attention and feed-forward keep lists of their own,
    which are parts of a block rather than blocks: a list held by another names a
    depth already counted, so only the outermost are the unit wanted.
    """
    found = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.ModuleList) and len(child) > 0
    ]

    return [
        (name, block_list)
        for name, block_list in found
        if not any(name.startswith(f"{other}.") for other, _ in found if other != name)
    ]


def writes_by_block(
    module: torch.nn.Module, stream_size: int
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    """What each layer writes inside each repeated block, and outside them all.

    Every layer writes, not only the ones that declare an output width. The
    normalizations and the activations between them each hand on a tensor of
    their own, and the activation of a feed-forward is the widest tensor of its
    block: reading only what is declared leaves out the largest single thing a
    block holds.

    A layer saying nothing at all about its width is read at the width that
    reached it, which is what the layer before it wrote, the tree being walked
    in the order it was declared in. Only the ones that provably hand on what
    they were given untouched are passed over.

    An adapted layer is read as the parts it is made of, the adapted layer and the
    low rank pair beside it writing a tensor each before one is added into the other.

    Each write is paired with the bytes an element of it takes as it is written,
    which is the stream's for most layers and several times that for a quantized
    one.

    Args:
        module: The component the writes are read off.
        stream_size: Bytes an element flowing between the layers takes.

    Returns:
        The width and element size of what each layer writes, one list per
        block, and the same for the layers belonging to no block.
    """
    block_names = []

    for list_name, block_list in outer_block_lists(module):
        block_names += [f"{list_name}.{index}" for index in range(len(block_list))]

    per_block: dict[str, list[tuple[int, int]]] = {name: [] for name in block_names}
    outside: list[tuple[int, int]] = []
    carried = 0

    for name, child in module.named_modules():
        if list(child.children()):
            continue

        declared = written_width(child)

        if declared:
            carried = declared
            width = declared
        elif isinstance(child, PASS_THROUGH):
            width = 0
        else:
            width = read_width(child) or carried

        if not width:
            continue

        # A block that is itself a layer owns what it writes, so the name is
        # matched as well as anything under it.
        owner = next(
            (
                block
                for block in block_names
                if name == block or name.startswith(f"{block}.")
            ),
            None,
        )

        write = (width, write_dtype_size(child, stream_size))

        if owner is None:
            outside.append(write)
        else:
            per_block[owner].append(write)

    return list(per_block.values()), outside


def stream_width(widths: list[int]) -> int:
    """The width the blocks carry from one end of the trunk to the other.

    A block reads the stream and writes it back, and works at that width nearly
    throughout: the projections opening its attention, the one closing it, and
    every normalization in between. What it widens to it narrows again before
    handing anything on, so the width named most often is the stream and the
    ones named seldom are what a block passes through.

    Read over every block at once, never over one alone. A block fusing its
    projections names the fused width as often as the stream and would be read
    as carrying it, and a family holding a second, narrower stream beside the
    picture keeps far fewer blocks for it than for the trunk. The narrowest of a
    tie is taken: the alternative to a stream is something a block widens to.
    """
    counts: dict[int, int] = {}

    for width in widths:
        counts[width] = counts.get(width, 0) + 1

    return min(counts, key=lambda width: (-counts[width], width), default=0)


def scope_peak(writes: list[tuple[int, int]], stream: int, stream_size: int) -> int:
    """The most a chain of layers holds at once, in bytes per token.

    A layer hands its result on and lets go of it, so what a chain holds is the
    layer being computed next to the one it reads from, never the sum of
    everything the chain ever wrote. Three tensors, then: the costliest write,
    taken as its own layer writes it; the widest of the others, narrowed to the
    stream, since that is all that survives of the layer before; and the stream
    the trunk carries end to end and each block adds its result back into.

    Summing every width instead read as four times what a denoiser allocates, on
    every family here, and grew with the depth of what a block declares rather
    than with anything a pass holds. On a card the weights nearly fill, that
    difference is the whole of the denoiser's seat: it was refused one it had
    the room for and crossed the bus at every step.

    Args:
        writes: The width and element size of what each layer of the scope writes.
        stream: What the trunk carries, in elements per token.
        stream_size: Bytes an element flowing between the layers takes.
    """
    if not writes:
        return stream * stream_size

    costliest = max(width * size for width, size in writes)
    widest = max(width for width, _ in writes)

    return costliest + (widest + stream) * stream_size


def bytes_per_token(denoiser: torch.nn.Module) -> int:
    """What one token of the picture costs the denoiser at its peak, in bytes.

    A block lets go of its tensors once it has handed on its result, so depth
    doesn't sum into the peak: the widest block bounds it, and the widest is
    taken over every block of every stream, a family carrying its text beside
    its picture keeping a set of blocks for each.

    Read on top of that block is one more stream width, for what reaches the
    trunk and what leaves it. The embedders and the head write per token too,
    and while neither is alive when a block is at its widest, the room is left
    for them rather than argued about.

    What those same outer layers write per sample is not read at all. A timestep
    becomes a modulation as wide as several streams, and one number per picture
    was being charged to every token of it: on the families here that alone was
    a third of the figure, and it grew with a resolution it has nothing to do
    with.

    Attention is counted through the tensors its projections write, never the
    scores, which the backends this application installs don't materialize.
    """
    stream_size = stream_dtype_size(denoiser)
    per_block, outside = writes_by_block(denoiser, stream_size)
    trunk = [width for writes in per_block for width, _ in writes]

    # Nothing repeated found: the whole module is then one block of its own.
    if not trunk:
        outer = stream_width([width for width, _ in outside])

        return scope_peak(outside, outer, stream_size)

    stream = stream_width(trunk)
    widest = max(scope_peak(writes, stream, stream_size) for writes in per_block)

    return widest + stream * stream_size


def latent_channels(vae: torch.nn.Module) -> int:
    """Channels the latent handed to a VAE carries."""
    config = getattr(vae, "config", None)

    for name in ("latent_channels", "z_dim"):
        channels = getattr(config, name, None)

        if isinstance(channels, int):
            return channels

    decoder = getattr(vae, "decoder", None)
    first = next(
        (
            child
            for _, child in (decoder or vae).named_modules()
            if written_width(child) and not list(child.children())
        ),
        None,
    )

    return getattr(first, "in_channels", 0) or 0


def meta_copy(module: torch.nn.Module) -> torch.nn.Module | None:
    """Rebuild a module from its configuration with nothing behind its tensors.

    A shape is all that is asked of the copy, and on the meta device it answers
    without a byte allocated or a kernel run. Rebuilt rather than borrowed: the
    loaded module carries quantized weights whose layers reshape what passes
    through them, where the architecture alone is the question here.
    """
    from_config = getattr(type(module), "from_config", None)
    config = getattr(module, "config", None)

    if from_config is None or config is None:
        return None

    try:
        with torch.device("meta"):
            copy = from_config(config)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Can't rebuild {type(module).__name__} to size it: {e}.")
        return None

    return copy if isinstance(copy, torch.nn.Module) else None


@dataclass
class Life:
    """How large a tensor a walked pass wrote is, and how long it stayed."""

    elements: int
    born: int
    dies: int | None = None
    """The step the last reference to it went, `None` while it is still held."""


class LiveTensors(TorchDispatchMode):
    """The tensors a walked pass writes, and the most it ever holds at once.

    A tensor is born when an operation writes it and dies when the last reference
    to it goes, which is not the same as the last read of it: a decoder carrying
    a time axis writes, at nearly every layer, a feature it keeps for the call
    after and never reads again in this one. Read by last use, those weigh
    nothing; held by the list that owns them, they are alive from where they are
    written to the end of the decode, and on that family they are most of what a
    decode holds.

    So the walk watches the operations rather than the layers, and keeps no
    reference of its own: what the pass has let go of, Python has already
    collected, and the burial is what tells the life. Everything a layer computes
    inside itself is then seen as well, where watching the layer boundaries saw
    only what crossed them.

    The weights are not the question and are left out: a decode holds them
    whatever the picture, and the caller sizes them apart.
    """

    def __init__(self, weights: set[int]):
        super().__init__()

        self.weights = weights
        """Identities of the tensors that are the model rather than the pass."""

        self.lives: dict[int, Life] = {}
        """The life of each tensor, by the step it was born at."""

        self.seen = WeakTensorKeyDictionary()
        """The order each tensor was seen in, kept without holding it alive."""

        self.step = 0
        """Operations walked so far, which is the clock the lives are told on."""

    def note(self, value) -> None:
        """Note a tensor written at the step under way, at whatever depth."""
        if isinstance(value, dict):
            value = list(value.values())

        for written in value if isinstance(value, (tuple, list)) else (value,):
            if isinstance(written, (tuple, list, dict)):
                self.note(written)
                continue

            if not torch.is_tensor(written) or id(written) in self.weights:
                continue

            if written in self.seen:
                continue

            self.step += 1
            self.seen[written] = self.step
            self.lives[self.step] = Life(written.numel(), self.step)

            weakref.finalize(written, self.bury, self.step)

    def bury(self, born: int) -> None:
        """Note that the tensor born at this step has just been let go of."""
        self.lives[born].dies = self.step

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.step += 1
        written = func(*args, **(kwargs or {}))
        self.note(written)

        return written

    def peak(self) -> int:
        """The most elements the pass held at once.

        The widest tensor alive is counted twice, once for itself and once for
        the value in flight beside it: an operation writes its result while what
        it works from is still there, and a kernel asks for scratch of its own
        that no walk over shapes can see. Whatever those are, they are the shape
        of a tensor next to them, and the widest is the one to stand in for it.

        Read this way, the two families here land within a twentieth of what a
        decode was measured to allocate, one just over and one just under.
        """
        last = self.step
        widest = 0

        for step in range(last + 1):
            alive = [
                life.elements
                for life in self.lives.values()
                if life.born <= step <= (life.dies if life.dies is not None else last)
            ]
            widest = max(widest, sum(alive) + max(alive, default=0))

        return widest


def walk_decode(vae: torch.nn.Module) -> tuple[int, int] | None:
    """Walk a decode of a known latent and read what it holds at its widest.

    A decoder is convolutional throughout, so what it holds grows with the picture
    and with nothing else: one walk gives the cost of every resolution, the stages
    nearest the output dominating a figure the ones before them barely move.

    Nothing is computed. The walk carries shapes over the meta device, which is
    also what tells the compression apart: what the decoder gives back against
    what it was handed is the ratio, whatever a configuration calls it.

    Returns:
        Bytes one pixel of the picture costs a pass, and pixels one latent pixel
        decodes to, or `None` where the decode can't be walked.
    """
    copy = meta_copy(vae)

    if copy is None:
        return None

    channels = latent_channels(copy)

    if not channels:
        return None

    walk = LiveTensors(
        {id(tensor) for tensor in chain(copy.parameters(), copy.buffers())}
    )

    # A family carrying a time axis reads one axis more than a picture has. Asked
    # rather than guessed: tried, a wrong count raises like a decoder that can't be
    # walked at all, and the two are only told apart by reading the reason.
    axes = convolved_axes(copy)
    shape = (1, channels) + (1,) * (axes - 2) + (LATENT_PROBE_SIDE,) * 2
    decode = getattr(copy, "decode", None)

    if not callable(decode):
        return None

    try:
        with torch.no_grad(), walk:
            picture = decode(torch.zeros(shape, device="meta"))
    except (RuntimeError, TypeError, ValueError, NotImplementedError) as e:
        logger.warning(f"Can't walk a decode of {type(vae).__name__} to size it: {e}")

        return None

    picture = getattr(picture, "sample", picture)

    if not torch.is_tensor(picture) or not walk.lives:
        logger.warning(f"A decode of {type(vae).__name__} gave back nothing to size.")

        return None

    pixels = picture.shape[-1] * picture.shape[-2]

    # The dtype is the loaded model's, never the copy's: rebuilding from a
    # configuration says nothing of what the weights were cast to.
    return (
        round(walk.peak() * stream_dtype_size(vae) / pixels),
        round(pixels / LATENT_PROBE_SIDE**2),
    )


def convolved_axes(vae: torch.nn.Module) -> int:
    """Axes of the picture a VAE convolves over, two for a picture, three with time.

    A kernel carries one entry per axis it slides along. Two where none is found,
    that being what a picture has.
    """
    for _, child in vae.named_modules():
        size = getattr(child, "kernel_size", None)

        if isinstance(size, (tuple, list)) and size:
            return len(size)

    return 2


def structural_pixels_per_token(
    denoiser: torch.nn.Module, compression: int, channels: int
) -> int:
    """How many pixels one token holds, as the configurations give it away.

    A denoiser is handed the latent either as the VAE left it or already folded
    into patches, and the two show up in different places. Folded on the way in,
    the fold reads as an input wider than the latent by the latent pixels a patch
    covers. Folded inside the model, it reads as the patch the configuration names.
    Either one is asked for, never their product: a family doing both would fold
    twice, and reading one fold where there are two puts more tokens in a picture
    than it has, which is the side to be wrong on. The other way sizes a run short
    and buys the denoiser a seat the GPU has no room for.

    A patch is square across the two axes of the picture, whatever else a family
    lists beside them, so its side is the widest figure named and its area that
    side squared.

    Args:
        denoiser: The component a run calls at every step.
        compression: Pixels one latent pixel decodes to.
        channels: Channels the latent carries as the VAE leaves it.
    """
    config = getattr(denoiser, "config", None)
    taken = getattr(config, "in_channels", 0) or 0
    folded_in = taken // channels if channels else 1

    sides = [1]

    for name, value in (config or {}).items():
        if "patch_size" not in name:
            continue

        entries = value if isinstance(value, (tuple, list)) else [value]
        sides += [entry for entry in entries if isinstance(entry, int)]

    return compression * max(folded_in, max(sides) ** 2, 1)


def denoiser_footprint(denoiser: torch.nn.Module) -> int:
    """Weight of a denoiser, in bytes, as what keys the figures it leaves behind."""
    footprint = getattr(denoiser, "get_memory_footprint", None)

    return footprint() if footprint is not None else 0


@dataclass(frozen=True)
class RunMemory:
    """What a loaded model asks of the GPU beyond its weights, per pixel.

    Everything here is read once, off the shapes the architecture fixes, and it
    holds for every resolution after: a convolution and an attention block both
    write tensors that grow with the picture and with nothing else, so one walk
    answers the sizes a session never generates as well as the ones it does.

    That is the whole point of reading rather than measuring. What a generation
    was watched to reserve says what that resolution cost, and a figure divided by
    the picture it was taken on carries the fixed part of the reserve into every
    other: it lands far above the truth on a roomy GPU, where the allocator holds
    the blocks it caches rather than hand them back, and the model is then refused
    a seat it had the room for at every resolution above the one measured.
    """

    denoiser_bytes_per_token: int
    """What one token costs the denoiser at its widest block, in bytes."""

    decode_bytes_per_pixel: int
    """What one pixel of the picture costs a VAE pass, in bytes."""

    read_pixels_per_token: int
    """Pixels one token holds, as the configurations gave it away at load."""

    def tokens(self, pixels: int) -> int:
        """Tokens a picture of this many pixels becomes."""
        return -(-pixels // max(self.read_pixels_per_token, 1))

    def run_bytes(self, pixels: int) -> int:
        """What a generation of this many pixels asks beyond the weights, in bytes.

        The walk, and nothing else. A correction watched at runtime sat here twice
        and was wrong both times, once reading the crowding around a generation and
        once the weights arriving inside a call, each time inflating the margin
        enough to cost a seat the walk had sized correctly. A third reading, of the
        tokens a pass was handed, answered what the patch sizes in the configuration
        already say. Nothing measured belongs here.
        """
        return self.tokens(pixels) * self.denoiser_bytes_per_token

    def decode_bytes(self, pixels: int) -> int:
        """What decoding this many pixels asks beyond the weights, in bytes."""
        return pixels * self.decode_bytes_per_pixel


def read_architecture(
    denoiser: torch.nn.Module | None, vae: torch.nn.Module | None
) -> RunMemory | None:
    """Read what a loaded model asks of the GPU, walking it once and keeping it.

    Called when a model is loaded, so that every resolution of the session is
    answered from the walk rather than from the generation before it.

    Args:
        denoiser: The component a run calls at every step.
        vae: The component encoding and decoding the picture.

    Returns:
        The reading, or `None` where the model can't be walked, no figure at all
        being better than one the architecture didn't give.
    """
    if denoiser is None or vae is None:
        return None

    key = (denoiser_footprint(denoiser), type(vae).__name__)
    kept = READ_ARCHITECTURES.get(key)

    if kept is not None:
        return kept

    walked = walk_decode(vae)

    if walked is None:
        return None

    decode_bytes_per_pixel, compression = walked
    reading = RunMemory(
        denoiser_bytes_per_token=bytes_per_token(denoiser),
        decode_bytes_per_pixel=decode_bytes_per_pixel,
        read_pixels_per_token=structural_pixels_per_token(
            denoiser, compression, latent_channels(vae)
        ),
    )

    READ_ARCHITECTURES[key] = reading

    return reading
