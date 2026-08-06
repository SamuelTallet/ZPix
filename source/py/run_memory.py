"""What a generation asks of the GPU, read off the architecture that runs it."""

from dataclasses import dataclass

import torch

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

    return 0


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


def widths_by_block(module: torch.nn.Module) -> tuple[list[int], int]:
    """Widths written inside each repeated block, and the width written outside.

    Every layer writes, not only the ones that declare an output width. The
    normalizations and the activations between them each hand on a tensor of
    their own, as wide as the one they were given, and the activation of a
    feed-forward is the widest tensor of its block: counting only what is
    declared leaves out the largest single thing a block holds, and the figure
    resting on this comes out around half of what a pass really asks.

    A layer saying nothing at all about its width is read at the width that
    reached it, which is what the layer before it wrote, the tree being walked
    in the order it was declared in. Only the ones that provably hand on what
    they were given untouched are passed over.

    An adapted layer is read as the parts it is made of, the adapted layer and the
    low rank pair beside it writing a tensor each before one is added into the other.

    Returns:
        What each block writes, summed over its layers, and what the layers
        belonging to no block write, summed over all of them.
    """
    block_names = []

    for list_name, block_list in outer_block_lists(module):
        block_names += [f"{list_name}.{index}" for index in range(len(block_list))]

    per_block = dict.fromkeys(block_names, 0)
    outside = 0
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

        if owner is None:
            outside += width
        else:
            per_block[owner] += width

    return list(per_block.values()), outside


def bytes_per_token(denoiser: torch.nn.Module) -> int:
    """What one token of the picture costs the denoiser at its peak, in bytes.

    A block lets go of its tensors once it has handed on its result, so depth
    doesn't sum into the peak: the widest block bounds it, next to what the
    embedders and the head write and the pass keeps alive throughout.

    Counting every width as if a block held them all at once puts this above the
    truth, which is the side to err on: a seat wrongly refused costs the bus, one
    wrongly taken overflows into host memory, several times worse again.

    Attention is counted through the tensors its projections write, never the
    scores, which the backends this application installs don't materialize.
    """
    per_block, outside = widths_by_block(denoiser)
    widest = max(per_block, default=0)

    # Nothing repeated found: the whole module is then one block of its own.
    if not widest:
        return outside * stream_dtype_size(denoiser)

    return (widest + outside) * stream_dtype_size(denoiser)


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


class LiveTensors:
    """The tensors a walked pass holds, and the most it ever holds at once.

    A layer's output is born when the layer writes it and dies after the last
    layer that reads it, so what a pass really costs is the widest overlap of
    those lives, not the sum of everything it writes. A decoder makes that gap
    wide: it holds a handful of tensors at the resolution it has reached and lets
    each stage go as the next one takes over, where its layers, added up, would
    read as several times the picture.

    Only what crosses a layer boundary is seen. What a layer computes inside
    itself, and what the pass computes between two of them, is missed, which is
    part of why the caller keeps a reserve on top.
    """

    def __init__(self):
        self.lives: dict[int, list[int]] = {}
        """Elements, first step and last step of each tensor, by its identity."""

        self.held: list[torch.Tensor] = []
        """The tensors themselves, kept so that no identity is reused under us."""

        self.step = 0
        """Layers walked so far, which is the clock the lives are told on."""

    def note(self, value) -> None:
        """Note that a tensor is alive at the step under way."""
        for tensor in value if isinstance(value, (tuple, list)) else (value,):
            if not torch.is_tensor(tensor):
                continue

            life = self.lives.get(id(tensor))

            if life is None:
                self.held.append(tensor)
                self.lives[id(tensor)] = [tensor.numel(), self.step, self.step]
            else:
                life[2] = self.step

    def peak(self) -> int:
        """The most elements the pass held at once.

        The widest tensor alive is counted twice, once for itself and once for the
        value in flight beside it. A pass adds a residual, applies an activation
        and joins two paths between the layers it is walked through, and no layer
        owns what those write: it is exactly what a walk at this grain can't see.
        Whatever it is, it is the shape of a tensor next to it, and the widest of
        those is the one to stand in for it.
        """
        widest = 0

        for step in range(self.step + 1):
            alive = [
                elements
                for elements, born, died in self.lives.values()
                if born <= step <= died
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

    walk = LiveTensors()

    def watch(_module, args, output) -> None:
        walk.step += 1
        walk.note(args)
        walk.note(output)

    for _, child in copy.named_modules():
        if not list(child.children()):
            child.register_forward_hook(watch)

    # A family carrying a time axis reads one axis more than a picture has. Asked
    # rather than guessed: tried, a wrong count raises like a decoder that can't be
    # walked at all, and the two are only told apart by reading the reason.
    axes = convolved_axes(copy)
    shape = (1, channels) + (1,) * (axes - 2) + (LATENT_PROBE_SIDE,) * 2
    decode = getattr(copy, "decode", None)

    if not callable(decode):
        return None

    try:
        with torch.no_grad():
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
