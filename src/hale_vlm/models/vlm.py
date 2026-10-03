"""Assembled vision-language model."""

from __future__ import annotations

import torch
import torch.nn as nn
from hale_vlm.registry import register_model

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.language.hale import build_llm_backbone, iter_trainable_parameters, resolve_llm_config
from hale_vlm.vision.hale import build_projector, build_vision_tower

VLM_VARIANTS = ("qwen3_8b_vlm", "deepseek_r1_qwen_7b_vlm")


class HaleVLM(nn.Module):
    """Vision encoder + projector + dense LLM backbone."""

    fusion_mode = "token_replace"

    def __init__(self, vocab_size: int, cfg: VLMRunConfig | None = None, **kwargs) -> None:
        super().__init__()
        del vocab_size, kwargs
        if cfg is None:
            raise TypeError("HaleVLM requires cfg: VLMRunConfig")
        self.cfg = cfg
        llm_cfg = resolve_llm_config(cfg.model.llm)

        self.llm = build_llm_backbone(llm_cfg)
        self.vision = build_vision_tower(cfg.model.vision)
        self.projector = build_projector(
            self.vision.hidden_size,
            self.llm.hidden_size,
            cfg.model.vision,
        )
        self.image_token_id = self._resolve_image_token_id(llm_cfg.image_token)
        self._log_trainable_summary()

    @property
    def num_image_tokens(self) -> int:
        return self.cfg.model.vision.resolved_num_image_tokens()

    def trainable_parameters(self):
        """Parameters updated during fine-tuning: projector, optional vision, LoRA adapters."""
        yield from iter_trainable_parameters(self.projector)
        if not self.cfg.model.vision.freeze_encoder:
            yield from iter_trainable_parameters(self.vision)
        yield from iter_trainable_parameters(self.llm.model)

    def _log_trainable_summary(self) -> None:
        from loguru import logger

        from hale_vlm.language.hale import count_parameters

        modules = [self.projector, self.llm.model]
        if not self.cfg.model.vision.freeze_encoder:
            modules.insert(0, self.vision)
        trainable, total = count_parameters(*modules)
        llm_mode = (
            "lora"
            if self.cfg.model.llm.use_lora
            else ("frozen" if self.cfg.model.llm.freeze_llm else "full")
        )
        logger.info(
            "trainable components: vision_frozen={} llm_mode={} trainable={:,} / {:,}",
            self.cfg.model.vision.freeze_encoder,
            llm_mode,
            trainable,
            total,
        )

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    def _resolve_image_token_id(self, image_token: str) -> int:
        tokenizer = self.llm.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(image_token)
        if token_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            self.llm.model.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(image_token)
        return int(token_id)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision(pixel_values)
        projected = self.projector(vision_features)
        num_tokens = self.cfg.model.vision.resolved_num_image_tokens()
        if projected.shape[1] > num_tokens:
            projected = projected[:, :num_tokens]
        return projected

    def merge_image_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace placeholder image tokens with projected vision embeddings."""
        batch_size, seq_len, hidden = text_embeds.shape
        image_token_id = self.image_token_id
        num_image_tokens = image_embeds.shape[1]

        merged = []
        masks = []
        for batch_idx in range(batch_size):
            ids = input_ids[batch_idx]
            embeds = text_embeds[batch_idx]
            image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
            if image_positions.numel() == 0:
                merged.append(embeds)
                masks.append(torch.ones(seq_len, device=embeds.device, dtype=torch.long))
                continue

            start = int(image_positions[0].item())
            prefix = embeds[:start]
            suffix = embeds[start + num_image_tokens :]
            sample_embeds = torch.cat([prefix, image_embeds[batch_idx], suffix], dim=0)

            prefix_mask = torch.ones(prefix.shape[0], device=embeds.device, dtype=torch.long)
            image_mask = torch.ones(num_image_tokens, device=embeds.device, dtype=torch.long)
            suffix_mask = torch.ones(suffix.shape[0], device=embeds.device, dtype=torch.long)
            sample_mask = torch.cat([prefix_mask, image_mask, suffix_mask], dim=0)

            merged.append(sample_embeds)
            masks.append(sample_mask)

        max_len = max(t.shape[0] for t in merged)
        padded_embeds = []
        padded_masks = []
        for embeds, mask in zip(merged, masks, strict=True):
            pad_len = max_len - embeds.shape[0]
            if pad_len:
                pad = torch.zeros(pad_len, hidden, device=embeds.device, dtype=embeds.dtype)
                embeds = torch.cat([embeds, pad], dim=0)
                mask = torch.cat(
                    [mask, torch.zeros(pad_len, device=mask.device, dtype=mask.dtype)],
                    dim=0,
                )
            padded_embeds.append(embeds)
            padded_masks.append(mask)

        return torch.stack(padded_embeds), torch.stack(padded_masks)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        text_embeds = self.llm.embed_tokens(input_ids)
        if pixel_values is not None:
            image_embeds = self.encode_images(pixel_values)
            inputs_embeds, attention_mask = self.merge_image_embeddings(
                input_ids,
                text_embeds,
                image_embeds,
            )
            input_ids = None
        else:
            inputs_embeds = text_embeds

        return self.llm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @classmethod
    def from_config(cls, cfg: VLMRunConfig) -> HaleVLM:
        return cls(vocab_size=0, cfg=cfg)


def build_vlm(cfg: VLMRunConfig) -> HaleVLM:
    architecture = cfg.model.resolved_architecture(cfg.variant)
    if architecture == "hale":
        return HaleVLM.from_config(cfg)
    if architecture == "gwm_vla":
        from hale_vlm.models.vla.factory import build_vla_model

        return build_vla_model(cfg)
    from hale_vlm.models.scratch.factory import build_scratch_vlm

    return build_scratch_vlm(cfg)


def _register_vlm_variants() -> None:
    for name in VLM_VARIANTS:

        @register_model(name)
        class _RegisteredVLM(HaleVLM):
            pass


_register_vlm_variants()
