"""LoRA key format conversion."""

from torch import Tensor

# LoRA up matrix suffixes, mapped to their down matrix ones.
LORA_MATRICES = {
    "lora_B.weight": "lora_A.weight",
    "lora_up.weight": "lora_down.weight",
}


def _fold_alphas(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Fold the `.alpha` scalars of a LoRA into its up matrices, then drop them.

    Trainers scale a LoRA by `alpha / rank`, in `.alpha` keys that Diffusers
    doesn't handle. PEFT scales it by `lora_alpha / rank`, which Diffusers
    leaves at 1, so bake the trainer scale into the weights instead.

    Args:
        state: State dict of a LoRA.

    Returns:
        The same state dict, without any `.alpha` key.
    """
    folded = {}

    for key, tensor in state.items():
        if key.endswith(".alpha"):
            continue

        for up_matrix, down_matrix in LORA_MATRICES.items():
            if not key.endswith(up_matrix):
                continue

            module = key.removesuffix(up_matrix)
            alpha = state.get(f"{module}alpha")
            down = state.get(f"{module}{down_matrix}")

            # First dimension of a down matrix is the rank of the LoRA.
            if alpha is not None and down is not None:
                tensor = tensor * (float(alpha.item()) / down.shape[0])

        folded[key] = tensor

    return folded


def to_diffusers(state: dict[str, Tensor], family: str) -> dict[str, Tensor]:
    """Convert a LoRA to the key format Diffusers expects for a model family.

    Args:
        state: State dict of a LoRA.
        family: Family of the image model the LoRA is loaded on.

    Returns:
        A state dict Diffusers can load, or the given one if it already is.
    """
    if family not in ("FLUX.2", "Krea 2"):
        return state

    # Diffusers FLUX.2 and Krea 2 LoRA converters don't handle .alpha keys.
    return _fold_alphas(state)
