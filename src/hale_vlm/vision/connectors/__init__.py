"""Vision-to-LLM token connectors that compress patch features into learned tokens.

- :class:`QFormer` — BLIP-2 style learned queries (``projector_type: qformer``)
- :class:`GatedCrossAttentionProjector` — Flamingo style tanh-gated cross-attention
  (``projector_type: gated_cross_attention``)
"""

from hale_vlm.vision.connectors.factory import build_vision_connector
from hale_vlm.vision.connectors.gated_cross_attention import (
    GatedCrossAttentionBlock,
    GatedCrossAttentionProjector,
)
from hale_vlm.vision.connectors.qformer import QFormer, QFormerLayer

__all__ = [
    "GatedCrossAttentionBlock",
    "GatedCrossAttentionProjector",
    "QFormer",
    "QFormerLayer",
    "build_vision_connector",
]
