"""HuggingFace LLM backbone loaders for supported dense transformers."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.language.hale.lora import configure_llm_trainability

LLM_PRESETS: dict[str, dict[str, str | bool]] = {
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "reasoning_mode": False,
    },
    "deepseek-r1-qwen-7b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "reasoning_mode": True,
    },
}

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class LLMBackbone(nn.Module):
    """Wrapper around a pretrained causal LM from HuggingFace."""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        preset = LLM_PRESETS.get(cfg.backbone, {})
        model_id = cfg.model_id or str(preset.get("model_id", ""))
        dtype = _DTYPE_MAP[cfg.dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {
            "trust_remote_code": cfg.trust_remote_code,
            "torch_dtype": dtype,
        }
        if cfg.attn_implementation:
            load_kwargs["attn_implementation"] = cfg.attn_implementation

        base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model = configure_llm_trainability(base_model, cfg)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.model
        if hasattr(base, "get_base_model"):
            base = base.get_base_model()
        return base.get_input_embeddings()(input_ids)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )


def resolve_llm_config(cfg: LLMConfig) -> LLMConfig:
    """Apply backbone presets (model_id, reasoning flags) when using named backbones."""
    preset = LLM_PRESETS.get(cfg.backbone)
    if preset is None:
        return cfg
    updates = {}
    if cfg.model_id == LLMConfig.model_fields["model_id"].default:
        updates["model_id"] = str(preset["model_id"])
    if not cfg.reasoning_mode and preset.get("reasoning_mode"):
        updates["reasoning_mode"] = True
    return cfg.model_copy(update=updates) if updates else cfg


def build_llm_backbone(cfg: LLMConfig) -> LLMBackbone:
    return LLMBackbone(resolve_llm_config(cfg))
