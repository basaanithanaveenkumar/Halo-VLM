"""Build a vision-to-LLM token connector from :class:`VisionConfig`."""

from __future__ import annotations

import torch.nn as nn

from hale_vlm.vision.config import TOKEN_CONNECTOR_TYPES, VisionConfig
from hale_vlm.vision.connectors.gated_cross_attention import GatedCrossAttentionProjector
from hale_vlm.vision.connectors.qformer import QFormer


def build_vision_connector(vision_dim: int, llm_dim: int, cfg: VisionConfig) -> nn.Module:
    """Return the Q-Former or gated cross-attention connector selected by ``projector_type``."""
    if cfg.projector_type == "qformer":
        q = cfg.qformer
        return QFormer(
            vision_dim,
            llm_dim,
            num_queries=q.num_queries,
            hidden_dim=q.hidden_dim,
            num_layers=q.num_layers,
            num_heads=q.num_heads,
            cross_attention_freq=q.cross_attention_freq,
            ffn_mult=q.ffn_mult,
            dropout=q.dropout,
        )
    if cfg.projector_type == "gated_cross_attention":
        g = cfg.gated_cross_attention
        return GatedCrossAttentionProjector(
            vision_dim,
            llm_dim,
            num_latents=g.num_latents,
            hidden_dim=g.hidden_dim,
            num_layers=g.num_layers,
            num_heads=g.num_heads,
            ffn_mult=g.ffn_mult,
            dropout=g.dropout,
            gate_init=g.gate_init,
        )
    raise ValueError(
        f"projector_type={cfg.projector_type!r} is not a token connector; "
        f"expected one of {TOKEN_CONNECTOR_TYPES}"
    )
