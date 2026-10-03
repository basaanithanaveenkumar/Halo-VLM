"""Vision encoders, projectors, and configuration.

Import subpackages explicitly:

- ``hale_vlm.vision.hale`` — HF SigLIP/CLIP tower + projector
- ``hale_vlm.vision.scratch`` — OpenCLIP, custom ViT, timm encoders
"""

from hale_vlm.vision.config import VisionConfig

__all__ = ["VisionConfig"]
