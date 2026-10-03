"""Scratch vision stack: OpenCLIP, custom ViT, timm encoders, projectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hale_vlm.vision.scratch.projector import ImageProjector
from hale_vlm.vision.scratch.vit import PatchEmb, VisTransformer

if TYPE_CHECKING:
    from hale_vlm.vision.scratch.openclip import OpenCLIPEncoder
    from hale_vlm.vision.scratch.siglip import CLIPEncoder, SigLIPEncoder
    from hale_vlm.vision.scratch.timm import VisionEncoder

__all__ = [
    "CLIPEncoder",
    "ImageProjector",
    "OpenCLIPEncoder",
    "PatchEmb",
    "SigLIPEncoder",
    "VisTransformer",
    "VisionEncoder",
]

_LAZY_IMPORTS = {
    "CLIPEncoder": ("hale_vlm.vision.scratch.siglip", "CLIPEncoder"),
    "OpenCLIPEncoder": ("hale_vlm.vision.scratch.openclip", "OpenCLIPEncoder"),
    "SigLIPEncoder": ("hale_vlm.vision.scratch.siglip", "SigLIPEncoder"),
    "VisionEncoder": ("hale_vlm.vision.scratch.timm", "VisionEncoder"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
