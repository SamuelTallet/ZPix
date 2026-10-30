"""Custom SDXL UNet attention on the FlashAttention-2 (FA2) kernel."""

from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def is_plain(attn: Attention) -> bool:
    """Is this attention only its projections around the kernel?

    Every SDXL UNet one is: no group, spatial or q/k norm, no norm of the
    context, no residual and no rescale, all of which the stock processor
    applies and this one doesn't.
    """
    return (
        attn.spatial_norm is None
        and attn.group_norm is None
        and not attn.norm_cross
        and attn.norm_q is None
        and attn.norm_k is None
        and not attn.residual_connection
        and attn.rescale_output_factor == 1.0
    )


class SDXLFlashAttnProcessor:
    """A custom SDXL UNet attention processor built on the FA2 kernel.

    The `flash` backend can't serve here: the stock UNet processor calls SDPA
    directly rather than through the dispatcher, so `set_attention_backend`
    never reaches it. Staying in (B, S, H, D) also skips the transposes SDPA
    wants. Close to SDPA, not identical: rounding adds up over the steps.
    """

    def __init__(self, flash_attn_func):
        self.flash_attn_func = flash_attn_func
        self.stock = AttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        # FA2 takes no mask, and SDXL pipelines never send one; a feature map
        # rather than a sequence isn't what a UNet transformer block passes.
        if attention_mask is not None or hidden_states.ndim != 3:
            return self.stock(
                attn, hidden_states, encoder_hidden_states, attention_mask
            )

        # Absent an encoder context, this is self-attention.
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # (B, S, H, D): the layout the kernel reads.
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))

        # Default scale, as the stock processor, which ignores `attn.scale`.
        hidden_states = self.flash_attn_func(query, key, value, causal=False)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def install_sdxl_flash_attn(pipe: DiffusionPipeline | ModularPipeline) -> None:
    """Install a custom FA2 processor onto every plain SDXL UNet attention.

    The VAE is left on SDPA: its attention normalizes and adds a residual,
    which this processor doesn't.

    Raises:
        ImportError: If the `flash-attn` package is not installed, the
            caller is expected to fall back to stock SDPA processor.
        ValueError: If the UNet holds no attention this processor recognizes,
            leaving it on its stock kernel.
    """
    from flash_attn import flash_attn_func

    attentions = [
        module
        for module in pipe.unet.modules()
        if isinstance(module, Attention)
        and isinstance(module.processor, AttnProcessor2_0)
        and is_plain(module)
    ]

    if not attentions:
        raise ValueError("No plain attention found in the UNet.")

    proc = SDXLFlashAttnProcessor(flash_attn_func)
    for attention in attentions:
        attention.set_processor(proc)
