"""Configurable TPVFormer (tri-perspective view) settings."""

from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class TPVConfig(StrictModel):
    """Hyperparameters for the pure-PyTorch TPVFormer occupancy encoder."""

    backbone: Literal["tiny", "resnet18"] = "tiny"
    num_cams: int = 6
    num_classes: int = 18
    embed_dim: int = 64
    ffn_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_feature_levels: int = 2
    dropout: float = 0.0

    tpv_h: int = 16
    tpv_w: int = 16
    tpv_z: int = 8
    scale_h: int = 1
    scale_w: int = 1
    scale_z: int = 1

    # LiDAR / ego volume, nuScenes default from TPVFormer occupancy config
    pc_range: tuple[float, float, float, float, float, float] = (
        -51.2,
        -51.2,
        -5.0,
        51.2,
        51.2,
        3.0,
    )
    num_z_anchors: tuple[int, int, int] = (4, 8, 8)
    image_size: tuple[int, int] = (128, 256)  # H, W
    use_grid_mask: bool = False
    freeze_backbone: bool = False
