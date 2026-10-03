"""Custom ViT + MoE decoder scratch VLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import Parameter

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.models.scratch.components.lm_head import LMHead
from hale_vlm.models.scratch.components.transformer import DecoderTransformer
from hale_vlm.vision.config import TOKEN_CONNECTOR_TYPES, VisionConfig
from hale_vlm.vision.connectors import build_vision_connector
from hale_vlm.vision.scratch.projector import ImageProjector
from hale_vlm.vision.scratch.vit import VisTransformer


class HaloVLM(nn.Module):
    """Prefix-fusion VLM with patch-level image tokens and MoE decoder."""

    fusion_mode: Literal["prefix_concat"] = "prefix_concat"

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 512,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        vit_layers: int = 6,
        vit_heads: int = 16,
        decoder_layers: int = 16,
        decoder_heads: int = 32,
        vision_cfg: VisionConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = emb_dim
        self._num_image_tokens = (img_size // patch_size) ** 2
        self.vis_enc = VisTransformer(
            img_size=img_size,
            p_size=patch_size,
            in_chans=3,
            emb_dim=emb_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
            mlp_dim=512,
            drop_fact=0.0,
        )
        self.decoder_transformer = DecoderTransformer(
            num_layers=decoder_layers,
            emb_dim=emb_dim,
            num_heads=decoder_heads,
            mlp_dim=1024,
            drop_fact=0.0,
        )
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(5000, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.lm_head = LMHead(hidden_size=emb_dim, vocab_size=vocab_size)
        if vision_cfg is not None and vision_cfg.projector_type in TOKEN_CONNECTOR_TYPES:
            self.image_projector = build_vision_connector(emb_dim, emb_dim, vision_cfg)
            self._num_image_tokens = vision_cfg.resolved_num_image_tokens()
        else:
            self.image_projector = ImageProjector(vision_dim=emb_dim, llm_dim=emb_dim)

    @property
    def num_image_tokens(self) -> int:
        return self._num_image_tokens

    def trainable_parameters(self) -> Iterator[Parameter]:
        yield from self.parameters()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        batch_size = input_ids.size(0)
        device = input_ids.device

        img_proj = self.image_projector(self.vis_enc(images))
        text_embeds = self.token_emb(input_ids)
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
        positions = torch.arange(combined_embeds.size(1), device=device)
        combined_embeds = combined_embeds + self.pos_embed(positions).unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        transformer_out = self.decoder_transformer(combined_embeds)
        transformer_out = self.layer_norm(transformer_out)
        return self.lm_head(transformer_out)

    @classmethod
    def from_config(cls, cfg: VLMRunConfig) -> HaloVLM:
        scratch = cfg.model.scratch
        return cls(
            vocab_size=scratch.vocab_size,
            emb_dim=scratch.embed_dim,
            img_size=cfg.model.vision.image_size,
            patch_size=scratch.patch_size,
            vit_layers=scratch.vit_num_layers,
            vit_heads=scratch.vit_num_heads,
            decoder_layers=scratch.decoder_num_layers,
            decoder_heads=scratch.decoder_num_heads,
            vision_cfg=cfg.model.vision,
        )
