"""Fuse HW / ZH / WZ planes into a 3D occupancy volume (TPVAggregator)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TPVAggregator(nn.Module):
    """Each voxel is the sum of its three TPV plane projections, then classified."""

    def __init__(
        self,
        *,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        scale_h: int = 1,
        scale_w: int = 1,
        scale_z: int = 1,
    ) -> None:
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        hidden = hidden_dim or embed_dim * 2
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Softplus(),
            nn.Linear(hidden, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _upsample(self, plane: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if plane.shape[-2:] == size:
            return plane
        return F.interpolate(plane, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        tpv_hw: torch.Tensor,
        tpv_zh: torch.Tensor,
        tpv_wz: torch.Tensor,
        points: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tpv_hw: (B, H*W, C)
            tpv_zh: (B, Z*H, C)
            tpv_wz: (B, W*Z, C)
            points: optional voxel indices (B, N, 3) in grid coordinates
        Returns:
            occupancy logits (B, num_classes, W', H', Z') or (voxel, point) pair
        """
        batch, _, channels = tpv_hw.shape
        hw = tpv_hw.permute(0, 2, 1).reshape(batch, channels, self.tpv_h, self.tpv_w)
        zh = tpv_zh.permute(0, 2, 1).reshape(batch, channels, self.tpv_z, self.tpv_h)
        wz = tpv_wz.permute(0, 2, 1).reshape(batch, channels, self.tpv_w, self.tpv_z)

        out_h = self.tpv_h * self.scale_h
        out_w = self.tpv_w * self.scale_w
        out_z = self.tpv_z * self.scale_z
        hw = self._upsample(hw, (out_h, out_w))
        zh = self._upsample(zh, (out_z, out_h))
        wz = self._upsample(wz, (out_w, out_z))

        # Broadcast each plane into a (B, C, W, H, Z) voxel tensor and sum
        hw_vox = hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, out_z)
        zh_vox = zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, out_w, -1, -1)
        wz_vox = wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, out_h, -1)
        fused = hw_vox + zh_vox + wz_vox

        if points is None:
            fused = fused.permute(0, 2, 3, 4, 1)
            logits = self.classifier(self.decoder(fused))
            return logits.permute(0, 4, 1, 2, 3)

        pts = points.reshape(batch, 1, -1, 3).clone()
        num_pts = pts.shape[2]
        pts[..., 0] = pts[..., 0] / max(out_w, 1) * 2 - 1
        pts[..., 1] = pts[..., 1] / max(out_h, 1) * 2 - 1
        pts[..., 2] = pts[..., 2] / max(out_z, 1) * 2 - 1
        hw_pts = F.grid_sample(hw, pts[..., [0, 1]], align_corners=False).squeeze(2)
        zh_pts = F.grid_sample(zh, pts[..., [1, 2]], align_corners=False).squeeze(2)
        wz_pts = F.grid_sample(wz, pts[..., [2, 0]], align_corners=False).squeeze(2)
        fused_pts = hw_pts + zh_pts + wz_pts
        fused_vox = fused.flatten(2)
        combined = torch.cat([fused_vox, fused_pts], dim=-1).permute(0, 2, 1)
        logits = self.classifier(self.decoder(combined)).permute(0, 2, 1)
        logits_vox = logits[:, :, :-num_pts].reshape(batch, -1, out_w, out_h, out_z)
        logits_pts = logits[:, :, -num_pts:].reshape(batch, -1, num_pts, 1, 1)
        return logits_vox, logits_pts
