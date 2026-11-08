"""MOSS-Image transformer, from the model code OzzyGT/MOSS_Image_Base ships.

Copyright 2026 OpenMOSS and The HuggingFace Team, licensed under the Apache
License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Modified: gradient
checkpointing removed.
"""

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import apply_lora_scale
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device
from torch import nn


def situ(hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gated activation of the feed-forward layers, soft-clamped by tanh."""
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden_states / 25.0)
    )


def apply_rotary_emb(
    hidden_states: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    """Rotate-half RoPE, in the dtype of `hidden_states`."""
    x1, x2 = hidden_states.chunk(2, dim=-1)
    return hidden_states * freqs_cos + torch.cat((-x2, x1), dim=-1) * freqs_sin


class MOSSImageAttnProcessor:
    """Attend with RoPE, through the backend Diffusers dispatches to."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: MOSSImageAttention,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        if attn.fused_qkv:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        query = apply_rotary_emb(query, freqs_cos, freqs_sin)
        key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        hidden_states = dispatch_attention_fn(
            query.to(value.dtype),
            key.to(value.dtype),
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return attn.to_out(hidden_states.flatten(2, 3))


class MOSSImageAttention(nn.Module, AttentionModuleMixin):
    """Multi-head attention, its queries and keys RMS-normalized."""

    _default_processor_cls = MOSSImageAttnProcessor
    _available_processors = (MOSSImageAttnProcessor,)
    processor: MOSSImageAttnProcessor

    to_qkv: nn.Linear
    to_q: nn.Linear
    to_k: nn.Linear
    to_v: nn.Linear

    def __init__(
        self, dim: int, heads: int, bias: bool, fused_qkv: bool, eps: float = 1e-6
    ):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        head_dim = dim // heads

        # The text refiner blocks project qkv fused and biased, the main blocks
        # unfused and unbiased.
        if fused_qkv:
            self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        else:
            self.to_q = nn.Linear(dim, dim, bias=bias)
            self.to_k = nn.Linear(dim, dim, bias=bias)
            self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(dim, dim, bias=bias)

        self.processor = MOSSImageAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, freqs_cos, freqs_sin)


class MOSSImageTextRefinerBlock(nn.Module):
    """Unmodulated block refining the text states before the joint blocks."""

    def __init__(
        self, dim: int, ffn_dim: int, num_attention_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = MOSSImageAttention(
            dim, num_attention_heads, bias=True, fused_qkv=True, eps=eps
        )
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff_gate = nn.Linear(dim, ffn_dim)
        self.ff_up = nn.Linear(dim, ffn_dim)
        self.ff_down = nn.Linear(ffn_dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), freqs_cos, freqs_sin
        )
        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff_down(
            situ(self.ff_up(norm_hidden_states), self.ff_gate(norm_hidden_states))
        )
        return hidden_states + ff_output


class MOSSImageTransformerBlock(nn.Module):
    """Single-stream block over the image and text tokens, modulated by the
    shared timestep embedding."""

    def __init__(
        self, dim: int, ffn_dim: int, num_attention_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = MOSSImageAttention(
            dim, num_attention_heads, bias=False, fused_qkv=False, eps=eps
        )
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff_in = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ff_out = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.unbind(1)
        )

        norm_hidden_states = (
            self.norm1(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        attn_output = self.attn(norm_hidden_states, freqs_cos, freqs_sin)
        hidden_states = hidden_states + attn_output * gate_msa[:, None]

        norm_hidden_states = (
            self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_hidden_states, ff_gate = self.ff_in(norm_hidden_states).chunk(2, dim=-1)
        ff_output = self.ff_out(situ(ff_hidden_states, ff_gate))
        return hidden_states + ff_output * gate_mlp[:, None]


class MOSSImageRotaryPosEmbed(nn.Module):
    """4-axis RoPE over (unused, frame, height, width) coordinates.

    Image tokens sit at frame `text_seq_len` on a grid centered at zero, text
    token `i` at frame `i`. Each axis takes `head_dim // 8` frequencies, built in
    float32 on the CPU like the original model.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.theta = theta

    def forward(
        self, height: int, width: int, text_seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exponent = torch.arange(self.num_bands, dtype=torch.float32) / self.num_bands
        bands = (1 / (self.theta**exponent)).to(device)

        grid_h, grid_w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image_ids = torch.stack(
            (
                torch.zeros_like(grid_h),
                torch.full_like(grid_h, text_seq_len),
                grid_h,
                grid_w,
            ),
            dim=-1,
        )
        image_ids = image_ids.flatten(0, 1).float()
        image_ids[:, 2] -= (height - 1) / 2
        image_ids[:, 3] -= (width - 1) / 2
        text_ids = torch.zeros(text_seq_len, 4, device=device, dtype=torch.float32)
        text_ids[:, 1] = torch.arange(text_seq_len, device=device)

        phase = (torch.cat((image_ids, text_ids)).unsqueeze(-1) * bands).flatten(1)
        phase = torch.cat((phase, phase), dim=-1)

        # (1, seq_len, 1, head_dim), broadcast over the attention heads.
        return phase.cos()[None, :, None], phase.sin()[None, :, None]


class MOSSImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin
):
    """The MOSS-Image diffusion transformer.

    Image tokens are FLUX.2 VAE latents, patchified to 128 channels at 1/16 of
    the image resolution. The text states pass through unmodulated refiner
    blocks, then image and text tokens through single-stream blocks, modulated
    by one projection of the timestep embedding. Each sample runs at its own text
    length, so prompts are never padded.
    """

    _no_split_modules = (
        "MOSSImageTransformerBlock",
        "MOSSImageTextRefinerBlock",
    )
    _repeated_blocks = (
        "MOSSImageTransformerBlock",
        "MOSSImageTextRefinerBlock",
    )
    _skip_layerwise_casting_patterns = ("norm",)

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 42,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        ffn_dim: int = 8192,
        text_embed_dim: int = 4096,
        timestep_embed_dim: int = 256,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.rope = MOSSImageRotaryPosEmbed(attention_head_dim, rope_theta)
        self.x_embedder = nn.Linear(in_channels, inner_dim)
        self.context_embedder = nn.Linear(text_embed_dim, inner_dim)
        self.context_refiner = nn.ModuleList(
            [
                MOSSImageTextRefinerBlock(inner_dim, ffn_dim, num_attention_heads)
                for _ in range(num_refiner_layers)
            ]
        )
        self.time_embedder = TimestepEmbedding(timestep_embed_dim, inner_dim)
        self.block_modulation = nn.Linear(inner_dim, 6 * inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MOSSImageTransformerBlock(inner_dim, ffn_dim, num_attention_heads)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = RMSNorm(inner_dim, eps=1e-6)
        self.norm_out_modulation = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)

    def timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        # Cos first, then sin; in float64 where the device supports it, like the
        # original model.
        half = self.config["timestep_embed_dim"] // 2
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, timestep.device)
        freqs = 10000 ** (
            -torch.arange(half, device=timestep.device, dtype=freqs_dtype) / half
        )
        phase = timestep.to(freqs_dtype)[:, None] * freqs
        return torch.cat((phase.cos(), phase.sin()), dim=-1).float()

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        skip_layers: list[int] | None = None,
        attention_kwargs: dict[str, object] | None = None,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        """Predict the flow of patchified latents.

        Args:
            hidden_states: Latents, of shape (batch, in_channels, height, width).
            timestep: Timesteps on the [0, 1000] scale, one or one per sample.
            encoder_hidden_states: Unpadded text states, one tensor per sample.
            skip_layers: Indices of the transformer blocks to skip.
            attention_kwargs: Extra arguments; `scale` sets the LoRA scale.
            return_dict: Whether to return a `Transformer2DModelOutput`.
        """
        batch_size, _, height, width = hidden_states.shape
        if len(encoder_hidden_states) != batch_size:
            raise ValueError(
                f"`encoder_hidden_states` holds {len(encoder_hidden_states)} "
                f"tensors for a batch of {batch_size}"
            )
        device = hidden_states.device
        image_seq_len = height * width

        timestep = torch.as_tensor(timestep, device=device).reshape(-1)
        timestep = timestep.expand(batch_size)
        temb = self.time_embedder(
            self.timestep_embedding(timestep).to(hidden_states.dtype)
        )
        modulation = self.block_modulation(temb).view(batch_size, 6, -1)
        shift_out, scale_out = (
            self.norm_out_modulation(temb).view(batch_size, 2, -1).unbind(1)
        )

        # Image positions depend on each sample's text length, so samples run
        # one at a time.
        outputs = []
        for index in range(batch_size):
            text_seq_len = encoder_hidden_states[index].shape[0]
            freqs_cos, freqs_sin = self.rope(height, width, text_seq_len, device)
            freqs_cos = freqs_cos.to(hidden_states.dtype)
            freqs_sin = freqs_sin.to(hidden_states.dtype)
            text_freqs = (freqs_cos[:, image_seq_len:], freqs_sin[:, image_seq_len:])

            image_tokens = self.x_embedder(
                hidden_states[index : index + 1].flatten(2).transpose(1, 2).contiguous()
            )
            text_tokens = self.context_embedder(encoder_hidden_states[index][None])
            for block in self.context_refiner:
                text_tokens = block(text_tokens, *text_freqs)

            tokens = torch.cat((image_tokens, text_tokens), dim=1)
            sample_modulation = modulation[index : index + 1]
            for block_index, block in enumerate(self.transformer_blocks):
                if skip_layers is not None and block_index in skip_layers:
                    continue
                tokens = block(tokens, sample_modulation, freqs_cos, freqs_sin)

            image_tokens = self.norm_out(tokens[:, :image_seq_len])
            image_tokens = (
                image_tokens * (1 + scale_out[index : index + 1, None])
                + shift_out[index : index + 1, None]
            )
            image_tokens = self.proj_out(image_tokens)
            outputs.append(image_tokens.transpose(1, 2).reshape(1, -1, height, width))

        output = torch.cat(outputs)
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
