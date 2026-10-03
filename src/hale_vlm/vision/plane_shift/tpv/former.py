"""End-to-end TPVFormer occupancy model in pure PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hale_vlm.vision.plane_shift.config import TPVConfig
from hale_vlm.vision.plane_shift.tpv.aggregator import TPVAggregator
from hale_vlm.vision.plane_shift.tpv.backbone import FPNNeck, build_image_backbone
from hale_vlm.vision.plane_shift.tpv.encoder import TPVFormerEncoder
from hale_vlm.vision.plane_shift.tpv.positional import TPVPositionalEncoding


@dataclass
class TPVFormerOutput:
    occupancy: torch.Tensor
    tpv_hw: torch.Tensor
    tpv_zh: torch.Tensor
    tpv_wz: torch.Tensor
    point_logits: torch.Tensor | None = None


class TPVFormer(nn.Module):
    """Tri-perspective view encoder + occupancy aggregator.

    Implements the TPVFormer pipeline from Huang et al., CVPR 2023
    (https://github.com/wzzheng/TPVFormer) without MMCV/MMDetection3D.
    """

    def __init__(self, cfg: TPVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = build_image_backbone(cfg.backbone, cfg.embed_dim, cfg.num_feature_levels)
        self.neck = FPNNeck(cfg.embed_dim)
        self.encoder = TPVFormerEncoder(cfg)
        self.aggregator = TPVAggregator(
            tpv_h=cfg.tpv_h,
            tpv_w=cfg.tpv_w,
            tpv_z=cfg.tpv_z,
            embed_dim=cfg.embed_dim,
            num_classes=cfg.num_classes,
            hidden_dim=cfg.ffn_dim,
            scale_h=cfg.scale_h,
            scale_w=cfg.scale_w,
            scale_z=cfg.scale_z,
        )
        self.positional_encoding = TPVPositionalEncoding(
            cfg.embed_dim, cfg.tpv_h, cfg.tpv_w, cfg.tpv_z
        )
        self.level_embeds = nn.Parameter(torch.zeros(cfg.num_feature_levels, cfg.embed_dim))
        self.cam_embeds = nn.Parameter(torch.zeros(cfg.num_cams, cfg.embed_dim))
        self.tpv_embedding_hw = nn.Embedding(cfg.tpv_h * cfg.tpv_w, cfg.embed_dim)
        self.tpv_embedding_zh = nn.Embedding(cfg.tpv_z * cfg.tpv_h, cfg.embed_dim)
        self.tpv_embedding_wz = nn.Embedding(cfg.tpv_w * cfg.tpv_z, cfg.embed_dim)
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cam_embeds)
        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_image_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """images: (B, N, 3, H, W) -> list of (B*N, C, h, w)."""
        batch, num_cams, channels, height, width = images.shape
        if num_cams != self.cfg.num_cams:
            raise ValueError(f"expected {self.cfg.num_cams} cameras, got {num_cams}")
        flat = images.reshape(batch * num_cams, channels, height, width)
        feats = self.neck(self.backbone(flat))
        decorated = []
        for level, feat in enumerate(feats):
            bn, dim, feat_h, feat_w = feat.shape
            feat = feat.view(batch, num_cams, dim, feat_h, feat_w)
            feat = feat + self.cam_embeds.view(1, num_cams, dim, 1, 1)
            feat = feat + self.level_embeds[level].view(1, 1, dim, 1, 1)
            decorated.append(feat.reshape(batch * num_cams, dim, feat_h, feat_w))
        return decorated

    def _init_queries(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        hw = self.tpv_embedding_hw.weight.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        zh = self.tpv_embedding_zh.weight.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        wz = self.tpv_embedding_wz.weight.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        hw = hw + self.positional_encoding(batch, device, "z")
        zh = zh + self.positional_encoding(batch, device, "w")
        wz = wz + self.positional_encoding(batch, device, "h")
        return [hw, zh, wz]

    def forward(
        self,
        images: torch.Tensor,
        *,
        lidar2img: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
    ) -> TPVFormerOutput:
        """
        Args:
            images: (B, num_cams, 3, H, W)
            lidar2img: optional (B, num_cams, 4, 4) camera matrices
            points: optional (B, N, 3) grid coordinates for sparse LiDAR heads
        """
        feat_maps = self.extract_image_features(images)
        planes = self._init_queries(images.shape[0], images.device, images.dtype)
        planes = self.encoder(planes, feat_maps, lidar2img)
        aggregated = self.aggregator(*planes, points=points)
        if isinstance(aggregated, tuple):
            occupancy, point_logits = aggregated
        else:
            occupancy, point_logits = aggregated, None
        return TPVFormerOutput(
            occupancy=occupancy,
            tpv_hw=planes[0],
            tpv_zh=planes[1],
            tpv_wz=planes[2],
            point_logits=point_logits,
        )
