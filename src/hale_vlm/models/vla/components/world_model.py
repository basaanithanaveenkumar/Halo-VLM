"""Causal latent world model for target-view patch prediction (GWM-VLA Eq. 5–8)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.config.sections.gwm import GWMConfig


class LatentWorldModel(nn.Module):
    """Predict next target-view patch tokens from current patch/register/latent-action tokens."""

    def __init__(self, cfg: GWMConfig, *, num_patches: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.num_register = cfg.num_register_tokens
        self.latent_tokens = cfg.latent_action_tokens
        self.horizon = cfg.action_horizon
        dim = cfg.embed_dim

        self.time_embed = nn.Embedding(cfg.action_horizon, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.world_model_heads,
            dim_feedforward=cfg.world_model_ff,
            dropout=cfg.world_model_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.world_model_layers)
        self.pred_head = nn.Linear(dim, dim)

    def _build_causal_mask(self, seq_len: int, tokens_per_step: int, device: torch.device) -> torch.Tensor:
        """Causal over timesteps; bidirectional within each timestep (Eq. 6)."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        num_steps = seq_len // tokens_per_step
        for step in range(num_steps):
            start = step * tokens_per_step
            end = start + tokens_per_step
            mask[start:end, :end] = 0.0
        return mask

    def forward(
        self,
        patch_tokens: torch.Tensor,
        register_tokens: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced next-step patch prediction.

        Args:
            patch_tokens: (B, T, P, D) target-view patches for steps 0..T-1
            register_tokens: (B, T, R, D)
            latent_actions: (B, T, K, D)
        Returns:
            Predicted next patch tokens: (B, T, P, D) aligned with inputs 0..T-1
        """
        batch, horizon, _, dim = patch_tokens.shape
        if horizon < 1:
            raise ValueError("world model requires at least one timestep")

        tokens_per_step = patch_tokens.shape[2] + register_tokens.shape[2] + latent_actions.shape[2]
        step_tokens = []
        for t in range(horizon):
            step = torch.cat(
                [patch_tokens[:, t], register_tokens[:, t], latent_actions[:, t]],
                dim=1,
            )
            step = step + self.time_embed.weight[t].view(1, 1, dim)
            step_tokens.append(step)
        sequence = torch.cat(step_tokens, dim=1)

        mask = self._build_causal_mask(sequence.shape[1], tokens_per_step, sequence.device)
        encoded = self.encoder(sequence, mask=mask)

        predictions = []
        patch_count = patch_tokens.shape[2]
        for t in range(horizon):
            start = t * tokens_per_step
            patch_repr = encoded[:, start : start + patch_count]
            predictions.append(self.pred_head(patch_repr))
        return torch.stack(predictions, dim=1)

    @staticmethod
    def l1_world_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(predicted - target))
