"""Custom Qwen-Image 2.1 self-attention on the FlashAttention-2 (FA2) kernel."""

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21AttnProcessor,
    _qwenimage21_prepare_qkv,
)
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class Qwen21FlashAttnProcessor:
    """A custom Qwen-Image 2.1 self-attention processor built on the FA2 kernel.

    Diffusers' `flash` backend refuses any attn_mask, so we bypass the dispatcher.
    Only the decoding steps go through it: they read the KV cache and attend fully
    over [prefix, target], carrying nothing but the validity of the text keys, which
    the varlen path unpads. The prefill is block-causal, a structure no padding mask
    expresses, so it is left to the stock processor. Queries are all valid, so keys
    and values alone are unpadded, each side with its own cumulated lengths.
    Numerically matches SDPA on the valid tokens (bf16 rounding only).
    """

    # Read by `set_attention_backend()`, which walks every attention module: the
    # backend it sets reaches the stock processor below, the kernel here having no
    # use for it.
    _attention_backend = None
    _parallel_config = None

    def __init__(self, flash_attn_func, flash_attn_varlen_func, unpad_input):
        self.flash_attn_func = flash_attn_func
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.unpad_input = unpad_input
        self.prefill_processor = QwenImage21AttnProcessor()

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        if segments is not None:
            # Prefill. Handed over whole rather than prepared here and finished
            # there: the projection writes the KV cache, and it is written once.
            self.prefill_processor._attention_backend = self._attention_backend
            self.prefill_processor._parallel_config = self._parallel_config

            return self.prefill_processor(
                attn,
                hidden_states,
                attention_mask=attention_mask,
                rotary_emb=rotary_emb,
                layer_cache=layer_cache,
                kv_cache_mode=kv_cache_mode,
                cache_write_slice=cache_write_slice,
                segments=segments,
                key_valid=key_valid,
            )

        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )

        # query/key/value are (B, S, H, D), the layout the kernel takes.
        scale = query.shape[-1] ** -0.5

        if attention_mask is None:
            out = self.flash_attn_func(
                query, key, value, softmax_scale=scale, causal=False
            )
        else:
            # (B, 1, 1, S_kv) key-validity mask -> (B, S_kv). The queries are the
            # target image's alone, all of them valid, so they cross the kernel
            # dense, their lengths cumulated by hand.
            keep = attention_mask[:, 0, 0, :]
            batch, _ = keep.shape
            queries = query.shape[1]

            keys, _, cu_keys, max_keys, _ = self.unpad_input(key, keep)
            values, _, _, _, _ = self.unpad_input(value, keep)

            cu_queries = torch.arange(
                0,
                (batch + 1) * queries,
                queries,
                dtype=torch.int32,
                device=query.device,
            )

            out = self.flash_attn_varlen_func(
                query.reshape(batch * queries, *query.shape[2:]),
                keys,
                values,
                cu_queries,
                cu_keys,
                queries,
                max_keys,
                softmax_scale=scale,
                causal=False,
            )
            out = out.reshape(batch, queries, *out.shape[1:])

        out = out[:, :seq_len_q].flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)

        return attn.to_out[1](out)


def install_qwen21_flash_attn(
    pipe: DiffusionPipeline | ModularPipeline,
) -> None:
    """Install a custom FA2 processor onto every Qwen-Image 2.1 attention module.

    Raises:
        ImportError: If the `flash-attn` package is not installed, the
            caller is expected to fall back to stock SDPA processor.
    """
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import unpad_input

    proc = Qwen21FlashAttnProcessor(
        flash_attn_func, flash_attn_varlen_func, unpad_input
    )

    for module in pipe.transformer.modules():
        if isinstance(module, QwenImage21Attention):
            module.set_processor(proc)
