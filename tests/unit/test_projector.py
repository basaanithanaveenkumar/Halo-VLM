import torch

from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.vision.hale import build_projector


def test_projector_shapes():
    cfg = VisionConfig(projector_type="mlp", projector_hidden_dim=64)
    projector = build_projector(vision_dim=32, llm_dim=16, cfg=cfg)
    x = torch.randn(2, 10, 32)
    out = projector(x)
    assert out.shape == (2, 10, 16)
