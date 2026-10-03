from typing import Literal

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
    coco_max_length: int = 17
