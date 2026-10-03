"""Vision tower wrapping HuggingFace SigLIP / CLIP encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel

from hale_vlm.vision.config import VisionConfig


class VisionTower(nn.Module):
    """Frozen (by default) vision encoder producing patch-level embeddings."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_type = cfg.encoder
        self.model = self._load_encoder(cfg.model_id, cfg.encoder)
        if cfg.freeze_encoder:
            self.model.requires_grad_(False)
            self.model.eval()

    def _load_encoder(self, model_id: str, encoder: str) -> nn.Module:
        if encoder == "siglip":
            return SiglipVisionModel.from_pretrained(model_id)
        if encoder == "clip":
            return CLIPVisionModel.from_pretrained(model_id)
        return AutoModel.from_pretrained(model_id)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.cfg.freeze_encoder:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state


def build_vision_tower(cfg: VisionConfig) -> VisionTower:
    return VisionTower(cfg)
