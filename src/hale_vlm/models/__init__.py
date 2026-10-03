"""Unified VLM model exports."""

from __future__ import annotations

from hale_vlm.models.scratch.factory import SCRATCH_VARIANTS
from hale_vlm.models.vla.factory import VLA_VARIANTS
from hale_vlm.models.vlm import HaleVLM, VLM_VARIANTS, build_vlm

ALL_VLM_VARIANTS = (*VLM_VARIANTS, *SCRATCH_VARIANTS, *VLA_VARIANTS)

__all__ = [
    "ALL_VLM_VARIANTS",
    "HaleVLM",
    "SCRATCH_VARIANTS",
    "VLA_VARIANTS",
    "VLM_VARIANTS",
    "build_vlm",
]


def __getattr__(name: str):
    if name == "BasicVLM":
        from hale_vlm.models.scratch.basic_vlm import BasicVLM

        return BasicVLM
    if name == "HaloVLM":
        from hale_vlm.models.scratch.halo_vlm import HaloVLM

        return HaloVLM
    if name == "GWMVLA":
        from hale_vlm.models.vla.gwm_vla import GWMVLA

        return GWMVLA
    if name == "build_scratch_vlm":
        from hale_vlm.models.scratch.factory import build_scratch_vlm

        return build_scratch_vlm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
