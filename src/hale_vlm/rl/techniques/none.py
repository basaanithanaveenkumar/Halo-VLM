"""Supervised training (RL disabled)."""

from __future__ import annotations

from collections.abc import Callable

import torch

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, supervised_step


class NoneRLTechnique(RLTechnique):
    name = "none"

    def __init__(self, cfg: RLConfig) -> None:
        del cfg

    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict,
        *,
        loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> RLStepResult:
        return supervised_step(
            model,
            batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
