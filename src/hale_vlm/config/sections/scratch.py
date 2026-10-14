from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.common import StrictModel


class ScratchConfig(StrictModel):
    """Hyperparameters for scratch BasicVLM / HaloVLM stacks."""

    embed_dim: int = 512
    vocab_size: int = 30522
    patch_size: int = 16
    vit_num_layers: int = 6
    vit_num_heads: int = 16
    decoder_num_layers: int = 16
    decoder_num_heads: int = 32
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"
    tokenizer_id: str = "bert-base-uncased"

    # ── Training objective (halo_moe architecture) ───────────────────────────
    objective: Literal["autoregressive", "mtp"] = "autoregressive"
    """``autoregressive``: next-token prediction only. ``mtp``: also predict the following
    ``mtp_num_future_tokens - 1`` tokens with extra depths (see
    ``hale_vlm.models.scratch.components.mtp``); generation still uses the next-token head."""
    mtp_num_future_tokens: int = Field(default=4, ge=2)
    """Tokens predicted per position including the next one (Gloeckle et al. use 4)."""
    mtp_style: Literal["parallel", "sequential"] = "parallel"
    """``parallel``: independent heads on the shared trunk (Gloeckle et al., 2024).
    ``sequential``: causal chain of MTP modules fed with future-token embeddings (DeepSeek-V3)."""
    mtp_loss_weight: float = Field(default=0.3, ge=0.0)
    """Weight of the mean extra-depth loss added to the next-token loss (DeepSeek-V3: 0.3)."""
    coco_max_length: int = 17
