from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.common import StrictModel

#: ``projector_type`` values that compress vision tokens into a fixed number of learned tokens.
TOKEN_CONNECTOR_TYPES = ("qformer", "gated_cross_attention")


class QFormerConfig(StrictModel):
    """Q-Former settings, used when ``projector_type: qformer``."""

    num_queries: int = 32
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    cross_attention_freq: int = 1  # cross-attend to vision in every N-th layer (BLIP-2 uses 2)
    ffn_mult: int = 4
    dropout: float = 0.0


class GatedCrossAttentionConfig(StrictModel):
    """Gated cross-attention settings, used when ``projector_type: gated_cross_attention``."""

    num_latents: int = 64
    hidden_dim: int | None = None  # None -> min(llm_dim, 768)
    num_layers: int = 2
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
    gate_init: float = 1.0  # 0.0 = Flamingo-style identity start; >0 lets images flow at step 0


class VisionConfig(StrictModel):
    """Vision tower and projector settings."""

    encoder: Literal["siglip", "clip"] = "siglip"
    model_id: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    freeze_encoder: bool = True
    projector_type: Literal["mlp", "linear", "qformer", "gated_cross_attention"] = "mlp"
    projector_hidden_dim: int | None = None
    projector_dropout: float = 0.0
    num_image_tokens: int = 256
    qformer: QFormerConfig = Field(default_factory=QFormerConfig)
    gated_cross_attention: GatedCrossAttentionConfig = Field(
        default_factory=GatedCrossAttentionConfig
    )

    def resolved_num_image_tokens(self) -> int:
        """Image tokens the LLM sees: the connector's own count, else ``num_image_tokens``."""
        if self.projector_type == "qformer":
            return self.qformer.num_queries
        if self.projector_type == "gated_cross_attention":
            return self.gated_cross_attention.num_latents
        return self.num_image_tokens
