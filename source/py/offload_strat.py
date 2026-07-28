"""CPU offload strategy."""

import torch
from accelerate.utils.memory import clear_device_cache
from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy

from source.py.custom_logger import logger


class FreeMemoryOffloadStrategy(AutoOffloadStrategy):
    """Auto offload strategy measuring the GPU memory actually available.

    PyTorch doesn't hand the blocks it caches for the activations back to the
    driver, so the parent strategy reads them as used memory. The higher the
    resolution, the larger those blocks: it ends up believing no eviction can free
    enough room, and offloads *every* component on each forward pass.

    Reclaiming here is also the only defragmentation available where the allocator
    can't be set to `expandable_segments`, as on Windows. Without it, each
    generation gets slower than the one before.
    """

    def __call__(self, hooks, model_id, model, execution_device):
        # Collecting first: the offload hooks leave reference cycles behind, and
        # what they hold isn't handed back by an `empty_cache()` alone.
        clear_device_cache(garbage_collection=True)

        device_module = getattr(torch, execution_device.type, torch.cuda)
        free_memory = device_module.mem_get_info(execution_device.index)[0]
        evictable = sum(hook.model.get_memory_footprint() for hook in hooks)

        logger.info(
            f"Making room for {model_id} "
            f"({model.get_memory_footprint() / 1024**3:.1f}GB): "
            f"{free_memory / 1024**3:.1f}GB free, "
            f"{evictable / 1024**3:.1f}GB evictable."
        )

        return super().__call__(hooks, model_id, model, execution_device)
