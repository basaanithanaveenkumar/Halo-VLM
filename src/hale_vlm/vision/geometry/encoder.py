"""Multi-view geometry encoder (VGGT-Ω-style aggregation with local fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_vlm.config.sections.gwm import GWMConfig


@dataclass
class GeometryEncoderOutput:
    """Register and patch tokens for each view after joint aggregation."""

    register_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    target_register: torch.Tensor
    target_patch: torch.Tensor


class _CrossViewBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, V*(R+P), D) — all views concatenated for cross-view mixing
        normed = self.norm1(tokens)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        normed = self.norm2(tokens)
        cross_out, _ = self.cross_attn(normed, normed, normed, need_weights=False)
        tokens = tokens + cross_out
        tokens = tokens + self.ff(self.norm3(tokens))
        return tokens


class MultiViewGeometryEncoder(nn.Module):
    """Jointly aggregate multi-view observations into geometry-aware patch/register tokens.

    Implements the per-timestep encoder E_Ω from GWM-VLA (Eq. 3–4). A lightweight
    cross-view transformer stack stands in for frozen VGGT-Ω when ``geometry_backend=local``.
    """

    def __init__(self, cfg: GWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.num_register = cfg.num_register_tokens
        self.embed_dim = cfg.embed_dim

        self.patch_embed = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.register_tokens = nn.Parameter(torch.randn(1, cfg.num_views, self.num_register, cfg.embed_dim) * 0.02)
        self.view_embed = nn.Parameter(torch.randn(1, cfg.num_views, 1, cfg.embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_CrossViewBlock(cfg.embed_dim, heads=8, dropout=0.0) for _ in range(4)]
        )
        self.out_norm = nn.LayerNorm(cfg.embed_dim)

        if cfg.freeze_geometry_encoder:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def target_view_index(self) -> int:
        if self.cfg.target_view == "wrist":
            return self.cfg.target_view_index
        if self.cfg.target_view == "third_person":
            return min(1, self.cfg.num_views - 1)
        return self.cfg.target_view_index

    def _encode_views(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multi-view images into per-view register and patch tokens."""
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch, views, _, height, width = images.shape
        if views != self.cfg.num_views:
            raise ValueError(f"expected {self.cfg.num_views} views, got {views}")

        flat = images.reshape(batch * views, 3, height, width)
        patches = self.patch_embed(flat)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches.view(batch, views, self.num_patches, self.embed_dim)

        registers = self.register_tokens.expand(batch, -1, -1, -1)
        view_bias = self.view_embed.expand(batch, -1, self.num_register + self.num_patches, -1)
        tokens = torch.cat([registers, patches], dim=2) + view_bias

        mixed = tokens.reshape(batch, views * (self.num_register + self.num_patches), self.embed_dim)
        for block in self.blocks:
            mixed = block(mixed)
        mixed = self.out_norm(mixed)

        per_view_len = self.num_register + self.num_patches
        mixed = mixed.view(batch, views, per_view_len, self.embed_dim)
        register_tokens = mixed[:, :, : self.num_register]
        patch_tokens = mixed[:, :, self.num_register :]
        return register_tokens, patch_tokens

    def forward(self, images: torch.Tensor) -> GeometryEncoderOutput:
        register_tokens, patch_tokens = self._encode_views(images)
        target_idx = self.target_view_index
        return GeometryEncoderOutput(
            register_tokens=register_tokens,
            patch_tokens=patch_tokens,
            target_register=register_tokens[:, target_idx],
            target_patch=patch_tokens[:, target_idx],
        )


def build_geometry_encoder(cfg: GWMConfig) -> MultiViewGeometryEncoder:
    if cfg.geometry_backend == "vggt_omega":
        try:
            from hale_vlm.vision.geometry.vggt_backend import VGGTGeometryEncoder

            return VGGTGeometryEncoder(cfg)
        except ImportError as exc:
            raise ImportError(
                "geometry_backend='vggt_omega' requires optional VGGT dependencies. "
                "Use geometry_backend='local' or install the vggt extra."
            ) from exc
    return MultiViewGeometryEncoder(cfg)
