"""Image encoder: always emit ``embed_dim`` feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyImageBackbone(nn.Module):
    """Two-scale CNN used for tests and CPU-friendly occupancy smoke runs."""

    def __init__(self, embed_dim: int, num_levels: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                )
                for _ in range(max(num_levels - 1, 0))
            ]
        )
        self.num_levels = num_levels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        feat = self.stem(images)
        feats = [feat]
        current = feat
        for stage in self.stages:
            current = stage(current)
            feats.append(current)
        return feats[: self.num_levels]


class ResNet18Backbone(nn.Module):
    """Optional torchvision ResNet-18 + 1x1 laterals."""

    def __init__(self, embed_dim: int, num_levels: int = 2) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        channels = [128, 256, 512][:num_levels]
        self.laterals = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in channels])
        self.num_levels = num_levels

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        feats = [c2, c3, c4][: self.num_levels]
        return [lat(f) for lat, f in zip(self.laterals, feats, strict=True)]


def build_image_backbone(name: str, embed_dim: int, num_levels: int) -> nn.Module:
    if name == "resnet18":
        return ResNet18Backbone(embed_dim, num_levels)
    if name == "tiny":
        return TinyImageBackbone(embed_dim, num_levels)
    raise ValueError(f"unknown TPV backbone {name!r}")


class FPNNeck(nn.Module):
    """Top-down fusion so every TPV level shares ``embed_dim``."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.smooth = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) == 1:
            return [self.smooth(feats[0])]
        fused: list[torch.Tensor] = [feats[-1]]
        for feat in reversed(feats[:-1]):
            up = F.interpolate(fused[-1], size=feat.shape[-2:], mode="nearest")
            fused.append(feat + up)
        fused = list(reversed(fused))
        return [self.smooth(f) for f in fused]
