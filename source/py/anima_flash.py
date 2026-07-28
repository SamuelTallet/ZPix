"""Custom Anima (Cosmos) attention on the FlashAttention-2 (FA2) kernel."""

from typing import cast

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


def rotate(x: torch.Tensor, freqs) -> torch.Tensor:
    """Apply Cosmos RoPE to a (B, S, H, D) tensor.

    Diffusers types its own return as a pair, though it returns one tensor.
    """
    return cast(
        torch.Tensor,
        apply_rotary_emb(x, freqs, use_real_unbind_dim=-2, sequence_dim=1),
    )


class AnimaFlashAttnProcessor:
    """A custom Anima attention processor built on the FA2 kernel.

    The `flash` backend can't serve here: switching it on is process-wide, and
    Anima's text conditioner masks its own attention, which FA2 refuses. Only the
    transformer takes the kernel, so it takes it through a processor. Staying in
    (B, S, H, D) also skips the key and value copies the stock Cosmos processor
    makes to widen their heads. Matches SDPA (bf16 rounding only).
    """

    def __init__(self, flash_attn_func):
        self.flash_attn_func = flash_attn_func

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        # Absent an encoder context, this is self-attention.
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # (B, S, H, D): the layout the kernel reads.
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1))

        # Both norms are RMS over the head dimension, which stayed last.
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = rotate(query, image_rotary_emb)
            key = rotate(key, image_rotary_emb)

        # Narrower key heads get widened, as the stock processor does; Anima's match.
        head_dim = query.size(-1)
        if key.size(-1) != head_dim:
            key = key.repeat_interleave(head_dim // key.size(-1), dim=-1)
        if value.size(-1) != head_dim:
            value = value.repeat_interleave(head_dim // value.size(-1), dim=-1)

        if attention_mask is None:
            hidden_states = self.flash_attn_func(query, key, value, causal=False)
        else:
            # FA2 takes no mask, and Anima's transformer never sends one: the
            # padded cross-attention other Cosmos checkpoints use falls back here.
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=attention_mask,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def install_anima_flash_attn(pipe: DiffusionPipeline | ModularPipeline) -> None:
    """Install a custom FA2 processor onto every Anima transformer attention.

    The text conditioner is left on SDPA: it masks its attention, which FA2
    can't take.

    Raises:
        ImportError: If the `flash-attn` package is not installed, the
            caller is expected to fall back to stock SDPA processor.
        ValueError: If the transformer holds no attention this processor
            recognizes, leaving it on its stock kernel.
    """
    from flash_attn import flash_attn_func

    # An Anima block carries this processor on both its attentions; the
    # image-context one of other Cosmos checkpoints doesn't, and is left alone
    # since this processor has no second stream.
    attentions = [
        module
        for module in pipe.transformer.modules()
        if isinstance(getattr(module, "processor", None), CosmosAttnProcessor2_0)
    ]

    if not attentions:
        raise ValueError("No Cosmos attention found in the transformer.")

    proc = AnimaFlashAttnProcessor(flash_attn_func)
    for attention in attentions:
        attention.set_processor(proc)
