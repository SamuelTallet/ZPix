"""MOSS-Image LoRA key format conversion."""

import re

from torch import Tensor

MOSS_LORA_KEY = re.compile(
    r"(?:diffusion_model\.|lora_unet_)"
    r"(?P<stack>layers|text_embedder[._]blocks)[._](?P<index>\d+)[._](?:proj[._])?"
    r"(?P<layer>\w+?)\.(?P<matrix>lora_down|lora_up|lora_A|lora_B)\.weight"
)
"""A MOSS-Image LoRA key, in the original model's names, dotted or Kohya's."""

MOSS_LAYERS: dict[str, tuple[str, dict[str, str | tuple[str, ...]]]] = {
    "layers": (
        "transformer_blocks",
        {
            "qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
            "q": "attn.to_q",
            "k": "attn.to_k",
            "v": "attn.to_v",
            "o": "attn.to_out",
            "ffn_up_gate": "ff_in",
            "ffn_down": "ff_out",
        },
    ),
    "text_embedder_blocks": (
        "context_refiner",
        {
            "qkv": "attn.to_qkv",
            "proj": "attn.to_out",
            "ffn_gate": "ff_gate",
            "ffn_up": "ff_up",
            "ffn_down": "ff_down",
        },
    ),
}
"""MOSS-Image block stacks, mapped to their Diffusers names and layers."""


def moss_to_diffusers(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename the layers of a MOSS-Image LoRA to the Diffusers transformer's.

    Args:
        state: State dict of a LoRA, its alphas folded.

    Returns:
        The same tensors, under PEFT keys of the transformer where they match.
    """
    converted = {}

    for key, tensor in state.items():
        match = MOSS_LORA_KEY.fullmatch(key)
        stack, layers = (
            MOSS_LAYERS[match["stack"].replace(".", "_")] if match else ("", {})
        )

        if match is None or match["layer"] not in layers:
            converted[key] = tensor
            continue

        matrix = "lora_A" if match["matrix"] in ("lora_down", "lora_A") else "lora_B"
        layer = layers[match["layer"]]
        split_layers = (layer,) if isinstance(layer, str) else layer
        tensors = (
            tensor.chunk(len(split_layers))
            if matrix == "lora_B"
            else (tensor,) * len(split_layers)
        )

        for split_layer, split_tensor in zip(split_layers, tensors, strict=True):
            converted[
                f"transformer.{stack}.{match['index']}.{split_layer}.{matrix}.weight"
            ] = split_tensor

    return converted
