"""CPU offload strategy."""

import torch
from accelerate.utils.memory import clear_device_cache
from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy

from source.py.custom_logger import logger

EVICTION_TIERS = (
    ("text_encoder", "conditioner", "image_encoder", "feature_extractor"),
    ("vae",),
    (),
    ("transformer", "unet"),
)
"""Component name fragments, ordered by how far off their next use is.

The encoders condition once at the start, the VAE decodes once at the end, the
denoiser runs at every step in between. An eviction costs its round trip, and
only that distance says whether the trip is worth taking: ranking on footprint
alone, as the parent strategy does, sends the denoiser back to seat an encoder.
"""

ENCODER_RANK = 0
"""Rank of the components a run calls once, before the loop starts."""

UNKNOWN_RANK = 2
"""Rank of the unmatched components: after the known far-off ones."""

DENOISER_RANK = 3
"""Rank of the component a run calls back at every step.

Denoiser-first: it is called once per step where its siblings are called once
per generation, so the room goes to it and the others arbitrate around what it
leaves. Its seat is freed by a projection made before the loop, and freeing it
means streaming for the whole resolution rather than paying a trip both ways.

It holds among components the generation still has a use for. The decode runs
after the last step, and `free_gpu_before_decode` may have the seat back there.
"""


def eviction_rank(model_id: str) -> int:
    """Rank a component by how far off its next use is; lowest evicts first.

    Args:
        model_id: Name of the component, as the pipeline exposes it.
    """
    for rank, fragments in enumerate(EVICTION_TIERS):
        if any(fragment in model_id for fragment in fragments):
            return rank

    return UNKNOWN_RANK


class DenoiserFirstOffloadStrategy(AutoOffloadStrategy):
    """Offload strategy seating the denoiser, then evicting by reuse distance.

    The denoiser keeps its seat whatever arrives (`DENOISER_RANK`); the rest is
    ranked on how far off its next use is. A cache can only estimate that
    distance; a run needn't guess, its order of calls being fixed. Sizes are
    unequal, so the selection is greedy, not optimal.

    It measures the memory actually available, then frees only what the arriving
    component is short of. PyTorch keeps the blocks it caches for the activations,
    which the parent strategy reads as used memory: past a resolution it believes
    no eviction can help and offloads everything on each forward pass.

    Reclaiming here is also the only defragmentation available where the
    allocator can't be set to `expandable_segments`, as on Windows. Without it,
    each generation gets slower than the one before.
    """

    def __call__(self, hooks, model_id, model, execution_device):
        # Collecting first: the offload hooks leave reference cycles behind, and
        # what they hold isn't handed back by an `empty_cache()` alone.
        clear_device_cache(garbage_collection=True)

        device_module = getattr(torch, execution_device.type, torch.cuda)
        free_memory = device_module.mem_get_info(execution_device.index)[0]
        needed = model.get_memory_footprint() + self.memory_reserve_margin
        evictable = sum(hook.model.get_memory_footprint() for hook in hooks)

        logger.info(
            f"Making room for {model_id} "
            f"({model.get_memory_footprint() / 1024**3:.1f}GB): "
            f"{free_memory / 1024**3:.1f}GB free, "
            f"{evictable / 1024**3:.1f}GB evictable."
        )

        # A run conditions once, before the first step, so the encoders are done
        # by the time the denoiser arrives and their next use is the generation
        # after this one. The room they hold is the loop's for the asking, and no
        # figure has to say so: this is the one eviction here that isn't weighed
        # against anything, and it runs before the reserve is even consulted.
        #
        # Left to the arbitration below, that room is kept whenever the arrival
        # fits without it, which is most of the time and never when it counts: a
        # generation that compiles asks for several times a settled one, and what
        # it overruns by is smaller than what a finished encoder holds. Sizing the
        # reserve to catch that is what the note in `WARMING_UP_RUN` says not to
        # try again.
        released = (
            [hook for hook in hooks if eviction_rank(hook.model_id) == ENCODER_RANK]
            if eviction_rank(model_id) >= DENOISER_RANK
            else []
        )

        if released:
            freed = sum(hook.model.get_memory_footprint() for hook in released)

            logger.info(
                f"Releasing {', '.join(hook.model_id for hook in released)}: "
                f"{freed / 1024**3:.1f}GB freed, never called in the loop."
            )

            free_memory += freed

        if needed <= free_memory:
            return released

        # Counted apart from the released ones all the way down: those are gone
        # for having finished, so they are neither weighed for the room nor handed
        # back by the trim below, and the room they freed is already in hand.
        for_room = []

        # The seat sticks to whoever is sitting: an arriving sibling may not take
        # the denoiser's place. Ranking alone only made it the last to go, which
        # is still gone, and an encoder called once was watched to unseat it on
        # its way in and be unseated on its way out, both trips paid each way.
        candidates = [
            hook
            for hook in hooks
            if hook not in released and eviction_rank(hook.model_id) < DENOISER_RANK
        ]

        # Unless nothing else can pay: short of the room once every sibling is
        # gone, the driver backs the newcomer with host memory instead.
        evictable_now = sum(hook.model.get_memory_footprint() for hook in candidates)

        if free_memory + evictable_now < needed:
            candidates = [hook for hook in hooks if hook not in released]

        for hook in sorted(
            candidates,
            key=lambda hook: (
                eviction_rank(hook.model_id),
                hook.model.get_memory_footprint(),
            ),
        ):
            if needed <= free_memory:
                break

            free_memory += hook.model.get_memory_footprint()
            for_room.append(hook)

        # Reaching a nearer-used tier can make the evictions that led there
        # pointless, and their return trip would buy nothing.
        for hook in list(for_room):
            footprint = hook.model.get_memory_footprint()

            if free_memory - footprint >= needed:
                free_memory -= footprint
                for_room.remove(hook)

        if for_room:
            logger.info(
                f"Evicting {', '.join(hook.model_id for hook in for_room)} "
                f"for {model_id}."
            )

        return released + for_room
