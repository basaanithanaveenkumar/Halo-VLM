"""GWM-VLA: Geometry-Aware Latent World Modeling for VLA learning."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import Parameter

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.config.sections.gwm import GWMConfig
from hale_vlm.models.vla.components.flow_matching import FlowMatchingActionHead
from hale_vlm.models.vla.components.latent_action import LatentActionModule
from hale_vlm.models.vla.components.world_model import LatentWorldModel
from hale_vlm.vision.geometry.encoder import build_geometry_encoder


@dataclass
class GWMVLALosses:
    total: torch.Tensor
    action: torch.Tensor
    world: torch.Tensor


class GWMVLA(nn.Module):
    """Geometry-aware latent world model + shared latent-action flow policy."""

    fusion_mode = "prefix_concat"
    num_image_tokens: int

    def __init__(self, cfg: VLMRunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gwm: GWMConfig = cfg.model.gwm
        self.geometry = build_geometry_encoder(self.gwm)
        num_patches = (self.gwm.image_size // self.gwm.patch_size) ** 2
        self.num_image_tokens = num_patches

        self.latent_action = LatentActionModule(self.gwm)
        self.world_model = LatentWorldModel(self.gwm, num_patches=num_patches)
        self.action_head = FlowMatchingActionHead(self.gwm)

    def trainable_parameters(self) -> Iterator[Parameter]:
        for module in (self.latent_action, self.world_model, self.action_head):
            yield from module.parameters()
        if not self.gwm.freeze_geometry_encoder:
            yield from self.geometry.parameters()

    def encode_geometry(self, multi_view_images: torch.Tensor):
        return self.geometry(multi_view_images)

    def compute_losses(
        self,
        *,
        multi_view_images: torch.Tensor,
        next_multi_view_images: torch.Tensor,
        input_ids: torch.Tensor,
        proprio: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> GWMVLALosses:
        """Joint world-model and flow-matching objective (Eq. 13)."""
        horizon = self.gwm.action_horizon
        if actions.shape[1] != horizon:
            raise ValueError(f"expected action horizon {horizon}, got {actions.shape[1]}")

        current = self.encode_geometry(multi_view_images)
        nxt = self.encode_geometry(next_multi_view_images)

        latent_actions = self.latent_action(current, input_ids, attention_mask=attention_mask)
        action_loss = self.action_head.flow_matching_loss(actions, latent_actions, proprio)

        patch_tokens = current.target_patch.unsqueeze(1)
        register_tokens = current.target_register.unsqueeze(1)
        latent_step = latent_actions[:, :1]
        predicted = self.world_model(patch_tokens, register_tokens, latent_step)
        world_loss = LatentWorldModel.l1_world_loss(predicted[:, 0], nxt.target_patch)

        total = action_loss + self.gwm.world_loss_weight * world_loss
        return GWMVLALosses(total=total, action=action_loss, world=world_loss)

    def forward(
        self,
        multi_view_images: torch.Tensor,
        input_ids: torch.Tensor,
        proprio: torch.Tensor,
        *,
        next_multi_view_images: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> GWMVLALosses | torch.Tensor:
        if next_multi_view_images is not None and actions is not None:
            return self.compute_losses(
                multi_view_images=multi_view_images,
                next_multi_view_images=next_multi_view_images,
                input_ids=input_ids,
                proprio=proprio,
                actions=actions,
                attention_mask=attention_mask,
            )

        current = self.encode_geometry(multi_view_images)
        latent = self.latent_action(current, input_ids, attention_mask=attention_mask)
        noise = torch.randn(
            multi_view_images.shape[0],
            self.gwm.action_horizon,
            self.gwm.action_dim,
            device=multi_view_images.device,
        )
        flow_time = torch.zeros(multi_view_images.shape[0], device=multi_view_images.device)
        return self.action_head.predict_velocity(noise, flow_time, latent, proprio)

    @classmethod
    def from_config(cls, cfg: VLMRunConfig) -> GWMVLA:
        return cls(cfg)
