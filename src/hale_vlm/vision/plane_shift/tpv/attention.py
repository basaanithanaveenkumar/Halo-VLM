"""Cross-view hybrid attention and image-to-TPV cross-attention (pure PyTorch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_vlm.vision.plane_shift.tpv.geometry import (
    default_lidar2img,
    denormalize_points,
    project_to_image,
    reference_points_hw,
    reference_points_wz,
    reference_points_zh,
)


class TPVCrossViewHybridAttention(nn.Module):
    """Self-attention over concatenated HW/ZH/WZ queries so planes can share context."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, planes: list[torch.Tensor]) -> list[torch.Tensor]:
        lengths = [p.shape[1] for p in planes]
        tokens = torch.cat(planes, dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + self.dropout(attended))
        return list(torch.split(tokens, lengths, dim=1))


class TPVImageCrossAttention(nn.Module):
    """Lift multi-view image features onto TPV queries via projected 3D anchors.

    Replaces CUDA multi-scale deformable attention with bilinear ``grid_sample``
    at pillar reference points, then a multi-head attention mix.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        tpv_h: int,
        tpv_w: int,
        tpv_z: int,
        pc_range: tuple[float, float, float, float, float, float],
        num_z_anchors: tuple[int, int, int],
        image_size: tuple[int, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.num_z_anchors = num_z_anchors
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.sample_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _sample_plane(
        self,
        queries: torch.Tensor,
        feat_maps: list[torch.Tensor],
        refs: torch.Tensor,
        lidar2img: torch.Tensor,
    ) -> torch.Tensor:
        """Sample image features at 3D anchors and mix them into each query."""
        batch, num_queries, _ = queries.shape
        world = denormalize_points(refs, self.pc_range)
        pixel_xy, valid = project_to_image(world, lidar2img, self.image_size)
        # pixel_xy: B, N, A, Q, 2
        sampled = []
        for feat in feat_maps:
            bn, channels, height, width = feat.shape
            num_cams = lidar2img.shape[1]
            feat = feat.view(batch, num_cams, channels, height, width)
            cam_samples = []
            for cam in range(num_cams):
                grid = pixel_xy[:, cam].reshape(batch, -1, 1, 2)
                pulled = F.grid_sample(
                    feat[:, cam],
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                pulled = pulled.reshape(batch, channels, refs.shape[0], num_queries)
                mask = valid[:, cam].unsqueeze(1).to(pulled.dtype)
                cam_samples.append(pulled * mask)
            sampled.append(torch.stack(cam_samples, dim=1).mean(dim=1))
        # average over FPN levels: (B, C, A, Q)
        keys = torch.stack(sampled, dim=0).mean(dim=0).permute(0, 3, 2, 1)
        keys = self.sample_proj(keys)
        query = self.query_proj(queries).unsqueeze(2)
        q = query.view(batch, num_queries, 1, self.num_heads, self.head_dim)
        k = keys.view(batch, num_queries, refs.shape[0], self.num_heads, self.head_dim)
        attn = (q * k).sum(-1) / (self.head_dim**0.5)
        weights = torch.softmax(attn, dim=2)
        mixed = (weights.unsqueeze(-1) * k).sum(dim=2)
        mixed = mixed.reshape(batch, num_queries, self.embed_dim)
        return self.out_proj(mixed)

    def forward(
        self,
        planes: list[torch.Tensor],
        feat_maps: list[torch.Tensor],
        lidar2img: torch.Tensor | None,
        num_cams: int,
    ) -> list[torch.Tensor]:
        device = planes[0].device
        dtype = planes[0].dtype
        batch = planes[0].shape[0]
        if lidar2img is None:
            lidar2img = default_lidar2img(batch, num_cams, self.image_size, device, dtype)

        hw_refs = reference_points_hw(
            self.tpv_h, self.tpv_w, self.num_z_anchors[0], device, dtype
        )
        zh_refs = reference_points_zh(
            self.tpv_z, self.tpv_h, self.num_z_anchors[1], device, dtype
        )
        wz_refs = reference_points_wz(
            self.tpv_w, self.tpv_z, self.num_z_anchors[2], device, dtype
        )
        sampled = [
            self._sample_plane(planes[0], feat_maps, hw_refs, lidar2img),
            self._sample_plane(planes[1], feat_maps, zh_refs, lidar2img),
            self._sample_plane(planes[2], feat_maps, wz_refs, lidar2img),
        ]
        return [
            self.norm(plane + self.dropout(delta))
            for plane, delta in zip(planes, sampled, strict=True)
        ]
