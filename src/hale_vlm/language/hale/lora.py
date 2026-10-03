"""LoRA and parameter-freezing helpers for LLM fine-tuning."""

from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn
from loguru import logger
from peft import LoraConfig, get_peft_model

from hale_vlm.config.sections.llm import LLMConfig

DEFAULT_LORA_TARGETS: dict[str, list[str]] = {
    "qwen3-8b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek-r1-qwen-7b": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "custom": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def lora_target_modules(cfg: LLMConfig) -> list[str]:
    if cfg.lora_target_modules:
        return cfg.lora_target_modules
    return DEFAULT_LORA_TARGETS.get(cfg.backbone, DEFAULT_LORA_TARGETS["custom"])


def apply_lora(model: nn.Module, cfg: LLMConfig) -> nn.Module:
    """Wrap the causal LM with PEFT LoRA adapters; base weights stay frozen."""
    targets = lora_target_modules(cfg)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        "applied LoRA backbone={} targets={} trainable={:,} / {:,} params",
        cfg.backbone,
        targets,
        trainable,
        total,
    )
    return peft_model


def configure_llm_trainability(model: nn.Module, cfg: LLMConfig) -> nn.Module:
    """Freeze or LoRA-wrap the LLM according to config."""
    if cfg.use_lora:
        model = apply_lora(model, cfg)
    elif cfg.freeze_llm:
        model.requires_grad_(False)
    return model


def iter_trainable_parameters(*modules: nn.Module) -> Iterator[nn.Parameter]:
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                yield param


def count_parameters(*modules: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for module in modules:
        for param in module.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
    return trainable, total
