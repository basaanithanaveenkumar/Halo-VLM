"""Plane-shift 3D scene representations (tri-perspective view and related).

- ``hale_vlm.vision.plane_shift.tpv`` — TPVFormer occupancy model
"""

from hale_vlm.vision.plane_shift.config import TPVConfig
from hale_vlm.vision.plane_shift.tpv import TPVFormer, TPVFormerOutput, build_tpv_former

__all__ = ["TPVConfig", "TPVFormer", "TPVFormerOutput", "build_tpv_former"]
