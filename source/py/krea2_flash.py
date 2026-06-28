"""Custom Krea 2 (K2) self-attention on the FlashAttention-2 (FA2) kernel."""

import torch
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    apply_rotary_emb,
)
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline


class Krea2FlashAttnProcessor:
    """A custom K2 self-attention processor built on the FA2 kernel.

    Diffusers' built-in `flash` backend rejects both GQA and any attn_mask, yet the
    flash_attn package supports GQA natively and handles padding via its varlen path.
    So we bypass the dispatcher: plain flash when unmasked, varlen (unpad/pad) when the
    text is padded. Numerically matches SDPA on the valid tokens (bf16 rounding only).
    """

    def __init__(self, flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input):
        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.pad_input = pad_input
        self.unpad_input = unpad_input

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(
            -1, (attn.num_kv_heads, attn.head_dim)
        )
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # query/key/value are (B, S, H, D); flash handles num_heads != num_kv_heads (GQA).
        scale = attn.head_dim**-0.5
        if attention_mask is None:
            out = self.flash_attn_func(
                query, key, value, softmax_scale=scale, causal=False
            )
        else:
            # (B, 1, 1, S) key-padding mask -> (B, S); self-attn shares it for q and k.
            keep = attention_mask[:, 0, 0, :]
            batch, seq_len = keep.shape
            q_u, idx, cu_q, max_q, _ = self.unpad_input(query, keep)
            k_u, _, cu_k, max_k, _ = self.unpad_input(key, keep)
            v_u, _, _, _, _ = self.unpad_input(value, keep)
            out_u = self.flash_attn_varlen_func(
                q_u,
                k_u,
                v_u,
                cu_q,
                cu_k,
                max_q,
                max_k,
                softmax_scale=scale,
                causal=False,
            )
            out = self.pad_input(out_u, idx, batch, seq_len)

        out = out.flatten(2, 3)
        out = out * torch.sigmoid(gate)

        return attn.to_out[0](out)


def install_krea2_flash_attn(pipe: Krea2Pipeline) -> None:
    """Install a custom FA2 processor onto every K2 self-attention module.

    Raises:
        ImportError: If the `flash-attn` package is not installed, the
            caller is expected to fall back to stock SDPA processor.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    proc = Krea2FlashAttnProcessor(
        flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input
    )
    for module in pipe.transformer.modules():
        if isinstance(module, Krea2Attention):
            module.set_processor(proc)
