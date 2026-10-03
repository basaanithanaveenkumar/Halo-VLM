"""Tests for RL technique registry and training steps."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from hale_vlm.config import load_vlm_config
from hale_vlm.rl import RL_TECHNIQUE_NAMES, build_rl_technique, get_rl_technique
from hale_vlm.rl.base import RLStepResult, RLTechnique, sequence_logprobs
from hale_vlm.rl.techniques.dpo import DPORLTechnique
from hale_vlm.rl.techniques.grpo import GRPORLTechnique
from hale_vlm.rl.techniques.none import NoneRLTechnique

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, seq: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.seq = seq

    def forward(self, input_ids, attention_mask=None, pixel_values=None, labels=None):
        del attention_mask, pixel_values, labels
        x = self.embed(input_ids)
        return type("Out", (), {"logits": self.head(x)})()


@pytest.mark.smoke
def test_rl_defaults_to_none():
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    assert cfg.rl.technique == "none"
    assert cfg.rl.enabled() is False
    technique = build_rl_technique(cfg)
    assert isinstance(technique, NoneRLTechnique)


@pytest.mark.smoke
def test_all_rl_techniques_registered():
    for name in RL_TECHNIQUE_NAMES:
        cls = get_rl_technique(name)
        assert issubclass(cls, RLTechnique)


@pytest.mark.smoke
def test_sequence_logprobs_shape():
    logits = torch.randn(2, 6, 10)
    labels = torch.randint(0, 10, (2, 6))
    labels[:, 0] = -100
    logps = sequence_logprobs(logits, labels)
    assert logps.shape == (2,)


@pytest.mark.smoke
def test_dpo_training_step():
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    cfg.rl.technique = "dpo"
    technique = DPORLTechnique(cfg.rl)
    model = _TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "chosen_input_ids": torch.randint(0, 32, (2, 8)),
        "chosen_labels": torch.randint(0, 32, (2, 8)),
        "rejected_input_ids": torch.randint(0, 32, (2, 8)),
        "rejected_labels": torch.randint(0, 32, (2, 8)),
    }
    loss_fn = MagicMock()
    result = technique.training_step(
        model,
        batch,
        loss_fn=loss_fn,
        optimizer=optimizer,
        grad_clip=1.0,
    )
    assert torch.isfinite(result.loss)
    loss_fn.assert_not_called()


@pytest.mark.smoke
def test_grpo_training_step():
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    technique = GRPORLTechnique(cfg.rl.model_copy(update={"group_size": 2}))
    model = _TinyLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.randint(0, 32, (4, 8)),
        "labels": torch.randint(0, 32, (4, 8)),
        "rewards": torch.tensor([1.0, 0.5, 0.2, 0.8]),
    }
    result = technique.training_step(
        model,
        batch,
        loss_fn=MagicMock(),
        optimizer=optimizer,
        grad_clip=1.0,
    )
    assert torch.isfinite(result.loss)
