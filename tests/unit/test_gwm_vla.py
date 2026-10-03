"""Tests for GWM-VLA geometry-aware latent world modeling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hale_vlm.config import load_vlm_config
from hale_vlm.models.vla.gwm_vla import GWMVLA
from hale_vlm.models.vlm import build_vlm

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _synthetic_batch(cfg, batch_size: int = 2):
    gwm = cfg.model.gwm
    views = gwm.num_views
    size = gwm.image_size
    return {
        "multi_view_images": torch.randn(batch_size, views, 3, size, size),
        "next_multi_view_images": torch.randn(batch_size, views, 3, size, size),
        "input_ids": torch.randint(0, gwm.instruction_vocab_size, (batch_size, 12)),
        "attention_mask": torch.ones(batch_size, 12, dtype=torch.long),
        "proprio": torch.randn(batch_size, gwm.proprio_dim),
        "actions": torch.randn(batch_size, gwm.action_horizon, gwm.action_dim),
    }


@pytest.mark.smoke
def test_gwm_vla_config_loads():
    cfg = load_vlm_config(CONFIGS / "gwm_vla_overfit.yaml")
    assert cfg.variant == "gwm_vla"
    assert cfg.model.resolved_architecture(cfg.variant) == "gwm_vla"
    assert cfg.resolved_train_backend() == "vla"


@pytest.mark.smoke
def test_build_gwm_vla():
    cfg = load_vlm_config(CONFIGS / "gwm_vla_overfit.yaml")
    model = build_vlm(cfg)
    assert isinstance(model, GWMVLA)
    assert model.num_image_tokens == (cfg.model.gwm.image_size // cfg.model.gwm.patch_size) ** 2


@pytest.mark.smoke
def test_gwm_vla_forward_losses():
    cfg = load_vlm_config(CONFIGS / "gwm_vla_overfit.yaml")
    model = build_vlm(cfg)
    batch = _synthetic_batch(cfg)
    losses = model.compute_losses(**batch)
    assert losses.total.ndim == 0
    assert losses.action.ndim == 0
    assert losses.world.ndim == 0
    assert torch.isfinite(losses.total)


@pytest.mark.smoke
def test_gwm_vla_inference_actions():
    cfg = load_vlm_config(CONFIGS / "gwm_vla_overfit.yaml")
    model = build_vlm(cfg)
    batch = _synthetic_batch(cfg, batch_size=1)
    actions = model(
        batch["multi_view_images"],
        batch["input_ids"],
        batch["proprio"],
        attention_mask=batch["attention_mask"],
    )
    assert actions.shape == (1, cfg.model.gwm.action_horizon, cfg.model.gwm.action_dim)


@pytest.mark.smoke
def test_geometry_encoder_target_view():
    cfg = load_vlm_config(CONFIGS / "gwm_vla_overfit.yaml")
    model = build_vlm(cfg)
    out = model.encode_geometry(torch.randn(1, cfg.model.gwm.num_views, 3, 224, 224))
    assert out.target_patch.shape[1] == model.num_image_tokens
    assert out.target_register.shape[1] == cfg.model.gwm.num_register_tokens
