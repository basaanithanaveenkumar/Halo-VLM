"""Hale (HF) language model stack: backbone + LoRA helpers."""

from hale_vlm.language.hale.backbone import LLMBackbone, build_llm_backbone, resolve_llm_config
from hale_vlm.language.hale.lora import (
    apply_lora,
    configure_llm_trainability,
    count_parameters,
    iter_trainable_parameters,
)

__all__ = [
    "LLMBackbone",
    "apply_lora",
    "build_llm_backbone",
    "configure_llm_trainability",
    "count_parameters",
    "iter_trainable_parameters",
    "resolve_llm_config",
]
