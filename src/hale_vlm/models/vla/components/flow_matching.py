"""Conditional flow-matching action head (GWM-VLA Eq. 12)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.config.sections.gwm import GWMConfig


class FlowMatchingActionHead(nn.Module):
    """Predict action-chunk velocity field conditioned on latent actions and proprio."""

    def __init__(self, cfg: GWMConfig) -> None:
        super().__init__()
        self.horizon = cfg.action_horizon
        self.action_dim = cfg.action_dim
        cond_dim = cfg.action_horizon * cfg.latent_action_tokens * cfg.embed_dim + cfg.proprio_dim
        layers: list[nn.Module] = [nn.Linear(cond_dim + self.horizon * cfg.action_dim + 1, cfg.flow_hidden_dim), nn.GELU()]
        for _ in range(cfg.flow_num_layers - 1):
            layers.extend([nn.Linear(cfg.flow_hidden_dim, cfg.flow_hidden_dim), nn.GELU()])
        layers.append(nn.Linear(cfg.flow_hidden_dim, self.horizon * cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def _condition(self, latent_actions: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        batch = latent_actions.shape[0]
        flat_actions = latent_actions.reshape(batch, -1)
        return torch.cat([flat_actions, proprio], dim=-1)

    def predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        latent_actions: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        cond = self._condition(latent_actions, proprio)
        batch = noisy_actions.shape[0]
        action_flat = noisy_actions.reshape(batch, -1)
        flow_in = torch.cat([cond, action_flat, flow_time.view(batch, 1)], dim=-1)
        return self.net(flow_in).view(batch, self.horizon, self.action_dim)

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        latent_actions: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        batch = actions.shape[0]
        noise = torch.randn_like(actions)
        flow_time = torch.rand(batch, device=actions.device, dtype=actions.dtype)
        time_view = flow_time.view(batch, 1, 1)
        interpolated = (1.0 - time_view) * noise + time_view * actions
        target_velocity = actions - noise
        predicted = self.predict_velocity(interpolated, flow_time, latent_actions, proprio)
        return torch.mean((predicted - target_velocity) ** 2)
