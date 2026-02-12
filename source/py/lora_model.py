from json import JSONDecodeError, loads
from logging import warning
from pathlib import Path

from safetensors import safe_open
from torch import bfloat16


class LoraModel:
    """LoRA model."""

    def __init__(self, path: str | Path):
        """Opens a LoRA model stored in safe tensors.

        Args:
            path: Path to the LoRA .safetensors file.
        """
        self.metadata = {}
        self.state = {}

        with safe_open(path, framework="pt") as file:
            self.metadata = file.metadata()
            for key in file.keys():
                self.state[key] = file.get_tensor(key)

    def base_model(self) -> str | None:
        """Base model of this LoRA, from its metadata.

        Raises:
            TypeError: If base model in metadata is not a string.

        Returns:
            Base model as `str` or `None` if it is not in metadata.
        """
        base = self.metadata.get("ss_base_model_version")

        if base is not None and type(base) is not str:
            raise TypeError(f"Base model must be a str, got {type(base)}")

        return base

    def trigger_phrase(self) -> str | None:
        """Trigger phrase of this LoRA, from its metadata.

        Raises:
            TypeError: If trigger phrase in metadata is not a string.

        Returns:
            Trigger phrase as `str` or `None` if it is not in metadata.
        """
        phrase = self.metadata.get("modelspec.trigger_phrase")

        if phrase is not None and type(phrase) is not str:
            raise TypeError(f"Trigger phrase must be a str, got {type(phrase)}")

        return phrase

    def frequent_tag(self) -> str | None:
        """Most frequent tag of this LoRA, from its metadata.

        Raises:
            JSONDecodeError: If tag frequency in metadata is invalid.
            ValueError: If tag frequency [...] has a bad structure.
            TypeError: If tag in metadata is not a string.

        Returns:
            Tag as `str` or `None` if it is not in metadata.
        """
        if "ss_tag_frequency" not in self.metadata:
            return None

        try:
            frequency = loads(self.metadata["ss_tag_frequency"])
            # Example: {"1_tag": {"tag": 1}}
            container = next(iter(frequency.values()))
            tag = next(iter(container))

        except JSONDecodeError as error:
            raise ValueError("Invalid tag frequency JSON") from error

        except StopIteration as error:
            raise ValueError("Bad tag frequency structure") from error

        if type(tag) is not str:
            raise TypeError(f"Tag must be a str, got {type(tag)}")

        return tag

    def trigger_word(self) -> str | None:
        """Trigger word of this LoRA, from its metadata.

        Returns:
            Trigger word as `str` or `None` if it is not in metadata.
        """
        # Explicit metadata first.
        try:
            if phrase := self.trigger_phrase():
                return phrase
            else:
                warning("Trigger phrase not found or empty, trying tag...")
        except Exception as error:
            warning(error)

        # Then most frequent tag.
        tag = None

        try:
            tag = self.frequent_tag()
        except Exception as error:
            warning(error)
            return None

        if not tag:
            warning("Tag not found or empty")
            return None

        if len(tag) > 32:
            warning("A long tag (> 32 chars) is unlikely a trigger word")
            return None

        return tag

    def to_bf16(self) -> dict:
        """Converts all LoRA tensors to `bfloat16` if needed.

        Returns:
            A state dict with all tensors stored as `bfloat16`.
        """
        return {key: tensor.to(bfloat16) for key, tensor in self.state.items()}
