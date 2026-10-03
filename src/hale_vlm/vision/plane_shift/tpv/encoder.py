"""TPVFormer encoder: stacked self-attn + image cross-attn + FFN layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.vision.plane_shift.config import TPVConfig
from hale_vlm.vision.plane_shift.tpv.attention import (
    TPVCrossViewHybridAttention,
    TPVImageCrossAttention,
)


class TPVFormerLayer(nn.Module):
    def __init__(self, cfg: TPVConfig) -> None:
        super().__init__()
        self.self_attn = TPVCrossViewHybridAttention(cfg.embed_dim, cfg.num_heads, cfg.dropout)
        self.cross_attn = TPVImageCrossAttention(
            cfg.embed_dim,
            cfg.num_heads,
            tpv_h=cfg.tpv_h,
            tpv_w=cfg.tpv_w,
            tpv_z=cfg.tpv_z,
            pc_range=cfg.pc_range,
            num_z_anchors=cfg.num_z_anchors,
            image_size=cfg.image_size,
            dropout=cfg.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )
        self.ffn_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        planes: list[torch.Tensor],
        feat_maps: list[torch.Tensor],
        lidar2img: torch.Tensor | None,
        num_cams: int,
    ) -> list[torch.Tensor]:
        planes = self.self_attn(planes)
        planes = self.cross_attn(planes, feat_maps, lidar2img, num_cams)
        return [self.ffn_norm(p + self.ffn(p)) for p in planes]


class TPVFormerEncoder(nn.Module):
    def __init__(self, cfg: TPVConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TPVFormerLayer(cfg) for _ in range(cfg.num_layers)])
        self.num_cams = cfg.num_cams

    def forward(
        self,
        planes: list[torch.Tensor],
        feat_maps: list[torch.Tensor],
        lidar2img: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        for layer in self.layers:
            planes = layer(planes, feat_maps, lidar2img, self.num_cams)
        return planes
