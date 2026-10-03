"""OpenCLIP + PyTorch decoder scratch VLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import Parameter

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.vision.config import TOKEN_CONNECTOR_TYPES, VisionConfig
from hale_vlm.vision.connectors import build_vision_connector
from hale_vlm.vision.scratch.projector import ImageProjector
from hale_vlm.models.scratch.components.lm_head import LMHead
from hale_vlm.models.scratch.components.positional_embeddings import SinusoidalPositionalEmbedding


class BasicVLM(nn.Module):
    """Prefix-fusion VLM with a single global image token."""

    fusion_mode: Literal["prefix_concat"] = "prefix_concat"
    num_image_tokens: int = 1

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        vision_cfg: VisionConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeds = SinusoidalPositionalEmbedding(embed_dim)
        from hale_vlm.vision.scratch.openclip import OpenCLIPEncoder

        self.vision_encoder = OpenCLIPEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            freeze=False,
            output_dim=embed_dim,
        )
        if vision_cfg is not None and vision_cfg.projector_type in TOKEN_CONNECTOR_TYPES:
            self.image_projector = build_vision_connector(embed_dim, embed_dim, vision_cfg)
            self.num_image_tokens = vision_cfg.resolved_num_image_tokens()
        else:
            self.image_projector = ImageProjector(vision_dim=embed_dim, llm_dim=embed_dim)
        self.lm_head = LMHead(hidden_size=embed_dim, vocab_size=vocab_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=64,
            dim_feedforward=1024,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=12)

    def trainable_parameters(self) -> Iterator[Parameter]:
        yield from self.parameters()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img_features = self.vision_encoder(images)
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(1)
        img_proj = self.image_projector(img_features)
        batch_size, num_img_tokens, _ = img_proj.size()

        text_embeds = self.token_embeds(input_ids)
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
        combined_embeds = combined_embeds + self.positional_embeds(combined_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                input_ids.size(1),
                device=input_ids.device,
                dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(dtype=torch.bool)

        img_mask = torch.ones(batch_size, num_img_tokens, device=input_ids.device, dtype=torch.bool)
        combined_mask = torch.cat([img_mask, attention_mask], dim=1)

        total_len = combined_embeds.size(1)
        attn_mask = torch.triu(
            torch.ones(total_len, total_len, device=combined_embeds.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = ~combined_mask

        transformer_out = self.transformer(
            tgt=combined_embeds,
            memory=combined_embeds,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=key_padding_mask,
        )
        return self.lm_head(transformer_out)

    @classmethod
    def from_config(cls, cfg: VLMRunConfig) -> BasicVLM:
        scratch = cfg.model.scratch
        return cls(
            vocab_size=scratch.vocab_size,
            embed_dim=scratch.embed_dim,
            vision_cfg=cfg.model.vision,
        )
