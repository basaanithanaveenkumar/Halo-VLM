"""Tests for scratch data module integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hale_vlm.config import load_vlm_config
from hale_vlm.data.vlm.multimodal import MultimodalDataModule
from hale_vlm.models.scratch.halo_vlm import HaloVLM
from hale_vlm.models.vlm import build_vlm
from tests.helpers.tiny_models import TinyCausalLM, TinyTokenizer, TinyVisionModel

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def scratch_tokenizer():
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def hale_mocks():
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


@pytest.mark.smoke
def test_scratch_overfit_batch_uses_images_key(scratch_tokenizer):
    with patch(
        "hale_vlm.data.vlm.multimodal.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: scratch_tokenizer,
    ):
        cfg = load_vlm_config(CONFIGS / "halo_moe_overfit.yaml")
        data_module = MultimodalDataModule(cfg, tokenizer=None)
        batch = next(iter(data_module.train_loader()))

    assert "images" in batch
    assert "pixel_values" not in batch
    assert batch["images"].shape[0] == cfg.train.batch_size
    assert batch["labels"].shape == batch["input_ids"].shape


@pytest.mark.smoke
def test_hale_overfit_batch_uses_pixel_values_key(hale_mocks):
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    data_module = MultimodalDataModule(cfg, tokenizer=None)
    batch = next(iter(data_module.train_loader()))

    assert "pixel_values" in batch
    assert "images" not in batch


@pytest.mark.smoke
def test_scratch_overfit_training_step(scratch_tokenizer):
    with patch(
        "hale_vlm.data.vlm.multimodal.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: scratch_tokenizer,
    ):
        cfg = load_vlm_config(CONFIGS / "halo_moe_overfit.yaml")
        model = build_vlm(cfg)
        batch = next(iter(MultimodalDataModule(cfg, tokenizer=None).train_loader()))

    assert isinstance(model, HaloVLM)
    logits = model(
        batch["images"],
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    num_img = model.num_image_tokens
    targets = torch.cat(
        [
            torch.full((batch["labels"].shape[0], num_img), -100),
            batch["labels"],
        ],
        dim=1,
    )
    outputs_for_loss = logits[:, : targets.shape[1], :]
    loss = torch.nn.functional.cross_entropy(
        outputs_for_loss.reshape(-1, outputs_for_loss.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
