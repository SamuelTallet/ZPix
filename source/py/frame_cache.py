"""Features a video VAE keeps for the frame after, dropped where none follows."""

import weakref

import torch

FEATURE_CACHES = ("_feat_map", "_enc_feat_map")
"""Lists a causal VAE keeps its features in between frames, decoder then encoder.

Named alike by every family Diffusers derives from Wan's VAE, Qwen-Image's among
them: each pass empties them through `clear_cache` before it starts.
"""


class DroppedFeatures(list):
    """A feature cache that takes nothing in.

    A causal VAE copies what each of its convolutions reads into a slot of its
    own, for the frame after to read first. A picture is a single frame: each
    slot is read once, while still empty, then written, and never read again.
    Kept, those copies outlive their stage to the end of the pass, at every
    resolution the decoder goes through, the widest included, and they are a good
    share of what a decode holds. Dropped, nothing the pass computes changes.
    """

    def __setitem__(self, index, value) -> None:
        pass


def drop_frame_features(vae: torch.nn.Module | None) -> None:
    """Have a VAE keep no features for a frame after, every pass being a picture.

    Every pass and every tile of one empties the caches through `clear_cache`
    before it starts, so that reset is wrapped to hand them caches that keep
    nothing. A VAE keeping no such caches is left as it was.

    Not for a VAE that decodes a video: the frame after would then read the
    empty slot where it expects the end of the one before.

    Args:
        vae: The VAE to stop keeping features, or `None`.
    """
    if vae is None:
        return

    reset = getattr(type(vae), "clear_cache", None)

    # Wrapped once: a second wrapper would only reset the first.
    if reset is None or "clear_cache" in vars(vae):
        return

    # Held weakly, the wrapper living on the module it would otherwise keep alive.
    owner = weakref.ref(vae)

    def clear_cache() -> None:
        module = owner()

        if module is None:
            return

        reset(module)

        for name in FEATURE_CACHES:
            cache = getattr(module, name, None)

            if isinstance(cache, list):
                setattr(module, name, DroppedFeatures(cache))

    vae.clear_cache = clear_cache  # ty: ignore
