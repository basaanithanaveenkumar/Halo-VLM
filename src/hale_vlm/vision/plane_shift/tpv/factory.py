"""Factory for TPVFormer occupancy models."""

from __future__ import annotations

from hale_vlm.vision.plane_shift.config import TPVConfig
from hale_vlm.vision.plane_shift.tpv.former import TPVFormer

TPV_VARIANTS = ("tpv_former",)


def build_tpv_former(cfg: TPVConfig | None = None, **overrides) -> TPVFormer:
    """Build a configurable TPVFormer. ``tiny`` defaults are test-friendly."""
    if cfg is None:
        cfg = TPVConfig(**overrides)
    elif overrides:
        cfg = cfg.model_copy(update=overrides)
    return TPVFormer(cfg)
