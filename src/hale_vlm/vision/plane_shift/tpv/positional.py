"""Learned tri-plane positional encodings (TPVFormer CustomPositionalEncoding)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TPVPositionalEncoding(nn.Module):
    """Independent H/W/Z embeddings concatenated; one axis is zeroed per plane."""

    def __init__(self, embed_dim: int, tpv_h: int, tpv_w: int, tpv_z: int) -> None:
        super().__init__()
        num_feats = max(embed_dim // 3, 8)
        self.h_embed = nn.Embedding(tpv_h, num_feats)
        self.w_embed = nn.Embedding(tpv_w, num_feats)
        self.z_embed = nn.Embedding(tpv_z, num_feats)
        self.proj = nn.Linear(num_feats * 3, embed_dim)
        self.num_feats = num_feats
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z

    def forward(self, batch: int, device: torch.device, ignore_axis: str) -> torch.Tensor:
        if ignore_axis == "z":
            h_embed = self.h_embed(torch.arange(self.tpv_h, device=device))
            h_embed = h_embed.reshape(self.tpv_h, 1, -1).repeat(1, self.tpv_w, 1)
            w_embed = self.w_embed(torch.arange(self.tpv_w, device=device))
            w_embed = w_embed.reshape(1, self.tpv_w, -1).repeat(self.tpv_h, 1, 1)
            z_embed = torch.zeros(self.tpv_h, self.tpv_w, self.num_feats, device=device)
        elif ignore_axis == "w":
            h_embed = self.h_embed(torch.arange(self.tpv_h, device=device))
            h_embed = h_embed.reshape(1, self.tpv_h, -1).repeat(self.tpv_z, 1, 1)
            w_embed = torch.zeros(self.tpv_z, self.tpv_h, self.num_feats, device=device)
            z_embed = self.z_embed(torch.arange(self.tpv_z, device=device))
            z_embed = z_embed.reshape(self.tpv_z, 1, -1).repeat(1, self.tpv_h, 1)
        elif ignore_axis == "h":
            h_embed = torch.zeros(self.tpv_w, self.tpv_z, self.num_feats, device=device)
            w_embed = self.w_embed(torch.arange(self.tpv_w, device=device))
            w_embed = w_embed.reshape(self.tpv_w, 1, -1).repeat(1, self.tpv_z, 1)
            z_embed = self.z_embed(torch.arange(self.tpv_z, device=device))
            z_embed = z_embed.reshape(1, self.tpv_z, -1).repeat(self.tpv_w, 1, 1)
        else:
            raise ValueError(f"unknown ignore_axis {ignore_axis!r}")
        pos = torch.cat((h_embed, w_embed, z_embed), dim=-1).flatten(0, 1)
        return self.proj(pos).unsqueeze(0).expand(batch, -1, -1)
