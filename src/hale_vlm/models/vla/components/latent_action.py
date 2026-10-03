"""Shared latent-action queries conditioned on multi-view observation and language."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.config.sections.gwm import GWMConfig
from hale_vlm.vision.geometry.encoder import GeometryEncoderOutput


class LatentActionModule(nn.Module):
    """Produce timestep-grouped latent action tokens A_{0:T-1} (GWM-VLA Eq. 9–11)."""

    def __init__(self, cfg: GWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.instruction_embed = nn.Embedding(cfg.instruction_vocab_size, cfg.vlm_hidden_dim)
        self.queries = nn.Parameter(
            torch.randn(cfg.action_horizon, cfg.latent_action_tokens, cfg.latent_action_dim) * 0.02
        )
        self.context_proj = nn.Linear(cfg.embed_dim, cfg.vlm_hidden_dim)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.latent_action_dim,
                nhead=4,
                dim_feedforward=cfg.flow_hidden_dim,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
        self.out_proj = nn.Linear(cfg.latent_action_dim, cfg.embed_dim)

    def forward(
        self,
        geometry: GeometryEncoderOutput,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        pooled = geometry.patch_tokens.mean(dim=(1, 2))
        context = self.context_proj(pooled)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        text = self.instruction_embed(input_ids.clamp(min=0, max=self.cfg.instruction_vocab_size - 1))
        text = (text * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        context = context + text

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1, -1)
        queries = queries.reshape(batch, self.cfg.action_horizon * self.cfg.latent_action_tokens, -1)
        queries = queries + context.unsqueeze(1)

        fused = self.fusion(queries)
        fused = fused.view(batch, self.cfg.action_horizon, self.cfg.latent_action_tokens, -1)
        return self.out_proj(fused)
