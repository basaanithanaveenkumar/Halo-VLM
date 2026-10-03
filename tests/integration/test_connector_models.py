"""Q-Former / gated cross-attention wired into the full Hale and scratch VLMs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hale_vlm.config import load_vlm_config
from hale_vlm.data.vlm.multimodal import MultimodalDataModule
from hale_vlm.models.scratch.halo_vlm import HaloVLM as ScratchHaloVLM
from hale_vlm.models.vlm import HaleVLM, build_vlm
from hale_vlm.registry import get_trainer
from hale_vlm.training.evaluator import VLMEvaluator
from hale_vlm.vision.connectors import GatedCrossAttentionProjector, QFormer
from tests.helpers.tiny_models import TinyCausalLM, TinyTokenizer, TinyVisionModel

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

HALE_CASES = [
    ("qwen3_8b_qformer.yaml", QFormer),
    ("qwen3_8b_gated_cross_attention.yaml", GatedCrossAttentionProjector),
]
SCRATCH_CASES = [
    ("halo_moe_qformer_overfit.yaml", QFormer),
    ("halo_moe_gated_cross_attention_overfit.yaml", GatedCrossAttentionProjector),
]


@pytest.fixture
def hf_mocks():
    with (
        patch(
            "hale_vlm.vision.hale.tower.SiglipVisionModel.from_pretrained",
            TinyVisionModel.from_pretrained,
        ),
        patch(
            "hale_vlm.language.hale.backbone.AutoModelForCausalLM.from_pretrained",
            TinyCausalLM.from_pretrained,
        ),
        patch(
            "hale_vlm.language.hale.backbone.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: TinyTokenizer(),
        ),
        patch(
            "hale_vlm.data.vlm.multimodal.AutoTokenizer.from_pretrained",
            lambda *_args, **_kwargs: TinyTokenizer(),
        ),
    ):
        yield


def _shrink_hale(cfg) -> None:
    """Fit the connector to the 32-dim tiny vision tower / LLM used in tests."""
    cfg.model.vision.qformer.num_queries = 4
    cfg.model.vision.qformer.hidden_dim = 16
    cfg.model.vision.qformer.num_heads = 4
    cfg.model.vision.gated_cross_attention.num_latents = 4
    cfg.model.vision.gated_cross_attention.hidden_dim = 16
    cfg.model.vision.gated_cross_attention.num_heads = 4


@pytest.mark.integration
@pytest.mark.parametrize(("config_name", "connector_cls"), HALE_CASES)
def test_hale_vlm_uses_connector(hf_mocks, config_name, connector_cls):
    cfg = load_vlm_config(CONFIGS / config_name)
    _shrink_hale(cfg)
    model = HaleVLM(vocab_size=0, cfg=cfg)

    assert isinstance(model.projector, connector_cls)
    assert model.num_image_tokens == 4
    images = torch.randn(2, 3, 224, 224)
    assert model.encode_images(images).shape == (2, 4, model.llm.hidden_size)
    assert any(p.requires_grad for p in model.projector.parameters())


@pytest.mark.integration
@pytest.mark.parametrize(("config_name", "connector_cls"), HALE_CASES)
def test_hale_trainer_learns_with_connector(hf_mocks, tmp_path, config_name, connector_cls):
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_overfit.yaml")
    connector_cfg = load_vlm_config(CONFIGS / config_name)
    cfg.model.vision.projector_type = connector_cfg.model.vision.projector_type
    _shrink_hale(cfg)
    cfg.train.steps = 12
    cfg.train.batch_size = 2
    cfg.train.checkpoint_path = str(tmp_path / "last.pt")
    cfg.train.checkpoint_every_epoch = False
    cfg.train.resume = False
    cfg.logging.backend = "noop"
    cfg.logging.log_file = None
    cfg.experiment.enabled = False
    cfg.eval.every_n_epochs = None
    cfg.device = "cpu"
    cfg.model.max_length = 32

    trainer = get_trainer("vlm")(
        cfg,
        data_module=MultimodalDataModule(cfg, tokenizer=None),
        evaluator=VLMEvaluator(),
    )
    model, _tokenizer, losses = trainer.fit()

    assert isinstance(model.projector, connector_cls)
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))


@pytest.mark.smoke
@pytest.mark.parametrize(("config_name", "connector_cls"), SCRATCH_CASES)
def test_scratch_halo_vlm_compresses_patches(config_name, connector_cls):
    cfg = load_vlm_config(CONFIGS / config_name)
    cfg.model.scratch.decoder_num_layers = 2
    cfg.model.scratch.vit_num_layers = 2
    model = build_vlm(cfg)

    assert isinstance(model, ScratchHaloVLM)
    assert isinstance(model.image_projector, connector_cls)
    assert model.num_image_tokens == 16  # 196 patches compressed to 16 tokens

    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, cfg.model.scratch.vocab_size, (2, 8))
    logits = model(images, input_ids, attention_mask=torch.ones(2, 8))
    assert logits.shape == (2, 16 + 8, cfg.model.scratch.vocab_size)
    logits.mean().backward()
    assert all(p.grad is not None for p in model.image_projector.parameters())


@pytest.mark.smoke
def test_default_projector_is_unchanged_for_scratch_models():
    cfg = load_vlm_config(CONFIGS / "halo_moe_overfit.yaml")
    cfg.model.scratch.decoder_num_layers = 1
    cfg.model.scratch.vit_num_layers = 1
    model = build_vlm(cfg)
    assert type(model.image_projector).__name__ == "ImageProjector"
    assert model.num_image_tokens == 196


@pytest.mark.smoke
@pytest.mark.parametrize("projector_type", ["mlp", "qformer"])
def test_basic_vlm_image_token_count(projector_type):
    """BasicVLM with a stubbed OpenCLIP encoder (open_clip need not be installed)."""
    import sys
    import types

    import torch.nn as nn

    class FakeEncoder(nn.Module):
        def __init__(self, *, output_dim, **_kwargs):
            super().__init__()
            self.proj = nn.Linear(3, output_dim)

        def forward(self, images):
            return self.proj(images.mean(dim=(2, 3)))  # (B, D): one global feature

    fake = types.ModuleType("hale_vlm.vision.scratch.openclip")
    fake.OpenCLIPEncoder = FakeEncoder
    cfg = load_vlm_config(CONFIGS / "basic_vlm_coco.yaml")
    cfg.model.scratch.embed_dim = 64
    cfg.model.vision.projector_type = projector_type
    cfg.model.vision.qformer.num_queries = 4
    cfg.model.vision.qformer.hidden_dim = 64
    with patch.dict(sys.modules, {"hale_vlm.vision.scratch.openclip": fake}):
        model = build_vlm(cfg)

    expected = 4 if projector_type == "qformer" else 1
    assert model.num_image_tokens == expected
    input_ids = torch.randint(0, cfg.model.scratch.vocab_size, (2, 8))
    logits = model(torch.randn(2, 3, 32, 32), input_ids, attention_mask=torch.ones(2, 8))
    assert logits.shape == (2, expected + 8, cfg.model.scratch.vocab_size)
