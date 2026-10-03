"""Language model backbones and fine-tuning helpers.

Import subpackages explicitly:

- ``hale_vlm.language.hale`` — HF causal LM backbone + LoRA
"""

from hale_vlm.language.config import LLMConfig

__all__ = ["LLMConfig"]
