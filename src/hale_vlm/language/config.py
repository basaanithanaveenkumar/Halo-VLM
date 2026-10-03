"""HuggingFace LLM configuration for Hale VLM stacks."""

from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class LLMConfig(StrictModel):
    """HuggingFace LLM backbone settings."""

    backbone: Literal["qwen3-8b", "deepseek-r1-qwen-7b", "custom"] = "qwen3-8b"
    model_id: str = "Qwen/Qwen3-8B"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    freeze_llm: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    trust_remote_code: bool = True
    attn_implementation: str | None = "sdpa"
    image_token: str = "<image>"
    reasoning_mode: bool = False
