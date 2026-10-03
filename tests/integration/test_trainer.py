"""Integration tests for the VLM training loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hale_vlm.registry import get_trainer

from hale_vlm.config import load_vlm_config
from hale_vlm.data.vlm.multimodal import MultimodalDataModule
from hale_vlm.language.hale.lora import iter_trainable_parameters
from hale_vlm.models.vlm import HaleVLM
from hale_vlm.training.evaluator import VLMEvaluator
from tests.helpers.tiny_models import TinyCausalLM, TinyTokenizer, TinyVisionModel

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


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


@pytest.mark.integration
def test_vlm_lora_trainable_parameters(hf_mocks, tmp_path):
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    cfg.model.llm.use_lora = True
    cfg.model.vision.freeze_encoder = True

    model = HaleVLM(vocab_size=0, cfg=cfg)
    trainable = list(model.trainable_parameters())
    assert trainable, "expected trainable parameters"

    frozen_llm = [
        name
        for name, param in model.llm.model.named_parameters()
        if "lora" not in name.lower() and param.requires_grad
    ]
    assert not frozen_llm, f"base LLM weights should be frozen, got {frozen_llm}"

    projector_trainable = any(p.requires_grad for p in model.projector.parameters())
    assert projector_trainable
    vision_trainable = any(p.requires_grad for p in model.vision.parameters())
    assert not vision_trainable


@pytest.mark.integration
def test_trainer_runs_few_steps(hf_mocks, tmp_path):
    cfg = load_vlm_config(CONFIGS / "qwen3_8b_overfit.yaml")
    cfg.train.steps = 6
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
    cfg.model.vision.num_image_tokens = 8

    data_module = MultimodalDataModule(cfg, tokenizer=None)
    trainer = get_trainer("vlm")(
        cfg,
        data_module=data_module,
        evaluator=VLMEvaluator(),
    )
    model, tokenizer, losses = trainer.fit()

    assert len(losses) >= 4
    assert losses[-1] < losses[0]
    assert Path(cfg.train.checkpoint_path).exists()
    assert tokenizer is not None

    trainable = list(iter_trainable_parameters(model.projector, model.llm.model))
    assert all(p.requires_grad for p in trainable)
    assert not any(p.requires_grad for p in model.vision.parameters())
