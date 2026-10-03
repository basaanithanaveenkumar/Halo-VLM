"""GWM-VLA (Geometry-Aware Latent World Modeling) hyperparameters."""

from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class GWMConfig(StrictModel):
    """Config for GWM-VLA per arXiv geometry-aware latent world modeling."""

    # Multi-view geometry encoder (VGGT-Ω surrogate or HF checkpoint)
    geometry_backend: Literal["local", "vggt_omega"] = "local"
    geometry_model_id: str = "local/multiview-geometry"
    num_views: int = 2
    target_view: Literal["wrist", "third_person", "mixed"] = "wrist"
    target_view_index: int = 0
    geometry_layer: int = -1
    freeze_geometry_encoder: bool = True
    num_register_tokens: int = 4
    patch_size: int = 16
    image_size: int = 224

    # Latent world model
    world_model_layers: int = 4
    world_model_heads: int = 8
    world_model_ff: int = 1024
    world_model_dropout: float = 0.0
    world_loss_weight: float = 0.1

    # Shared latent-action representation
    action_horizon: int = 4
    latent_action_tokens: int = 8
    latent_action_dim: int = 256
    instruction_vocab_size: int = 32000
    vlm_hidden_dim: int = 256

    # Flow-matching action head
    action_dim: int = 7
    proprio_dim: int = 8
    flow_hidden_dim: int = 512
    flow_num_layers: int = 3

    embed_dim: int = 256
