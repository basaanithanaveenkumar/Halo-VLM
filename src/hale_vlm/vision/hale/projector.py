"""Project vision patch embeddings into the LLM hidden dimension."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.vision.config import TOKEN_CONNECTOR_TYPES, VisionConfig
from hale_vlm.vision.connectors import build_vision_connector


class VisionProjector(nn.Module):
    """Map vision encoder outputs to LLM token embedding space."""

    def __init__(self, vision_dim: int, llm_dim: int, cfg: VisionConfig) -> None:
        super().__init__()
        hidden = cfg.projector_hidden_dim or llm_dim * 2
        if cfg.projector_type == "linear":
            self.net = nn.Linear(vision_dim, llm_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(vision_dim, hidden),
                nn.GELU(),
                nn.Dropout(cfg.projector_dropout),
                nn.Linear(hidden, llm_dim),
            )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.net(vision_features)


def build_projector(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module:
    """Build the vision-to-LLM projector selected by ``cfg.projector_type``.

    ``mlp`` / ``linear`` keep one output token per vision patch; ``qformer`` and
    ``gated_cross_attention`` compress them into a fixed number of learned tokens.
    """
    if cfg.projector_type in TOKEN_CONNECTOR_TYPES:
        return build_vision_connector(vision_dim, llm_dim, cfg)
    return VisionProjector(vision_dim, llm_dim, cfg)
