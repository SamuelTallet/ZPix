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
"""Rank of the component a run calls back at every step."""


def eviction_rank(model_id: str) -> int:
    """Rank a component by how far off its next use is; lowest evicts first.

    Args:
        model_id: Name of the component, as the pipeline exposes it.
    """
    for rank, fragments in enumerate(EVICTION_TIERS):
        if any(fragment in model_id for fragment in fragments):
            return rank

    return UNKNOWN_RANK


class ReuseDistanceOffloadStrategy(AutoOffloadStrategy):
    """Offload strategy evicting the component whose next use is the furthest off.

    A cache can only estimate reuse distance; a run needn't guess, its order of
    calls being fixed. Sizes are unequal, so the selection is greedy, not optimal.

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

        if needed <= free_memory:
            return []

        evicted = []

        for hook in sorted(
            hooks,
            key=lambda hook: (
                eviction_rank(hook.model_id),
                hook.model.get_memory_footprint(),
            ),
        ):
            if needed <= free_memory:
                break

            free_memory += hook.model.get_memory_footprint()
            evicted.append(hook)

        # Reaching a nearer-used tier can make the evictions that led there
        # pointless, and their return trip would buy nothing.
        for hook in list(evicted):
            footprint = hook.model.get_memory_footprint()

            if free_memory - footprint >= needed:
                free_memory -= footprint
                evicted.remove(hook)

        if evicted:
            logger.info(
                f"Evicting {', '.join(hook.model_id for hook in evicted)} "
                f"for {model_id}."
            )

        return evicted
