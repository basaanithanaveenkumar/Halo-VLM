"""Unit tests for LoRA adapter wiring."""

from __future__ import annotations

from unittest.mock import patch

from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.language.hale.lora import apply_lora, configure_llm_trainability
from tests.helpers.tiny_models import TinyCausalLM


def test_apply_lora_freezes_base_weights():
    model = TinyCausalLM()
    cfg = LLMConfig(use_lora=True, lora_r=4, lora_alpha=8, backbone="custom")

    with patch("hale_vlm.language.hale.lora.get_peft_model", wraps=__import__("peft").get_peft_model):
        peft_model = apply_lora(model, cfg)

    trainable = [name for name, param in peft_model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora" in name.lower() for name in trainable)

    base_trainable = [
        name
        for name, param in peft_model.named_parameters()
        if "lora" not in name.lower() and param.requires_grad
    ]
    assert not base_trainable


def test_configure_llm_trainability_freeze_without_lora():
    model = TinyCausalLM()
    cfg = LLMConfig(use_lora=False, freeze_llm=True)
    frozen = configure_llm_trainability(model, cfg)
    assert not any(param.requires_grad for param in frozen.parameters())


def test_configure_llm_trainability_prefers_lora_over_full_finetune():
    model = TinyCausalLM()
    cfg = LLMConfig(use_lora=True, freeze_llm=False, lora_r=4, lora_alpha=8, backbone="custom")
    adapted = configure_llm_trainability(model, cfg)
    trainable = [name for name, param in adapted.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora" in name.lower() for name in trainable)
