"""Vanilla REINFORCE with moving-average baseline."""

from __future__ import annotations

from collections.abc import Callable

import torch

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, model_sequence_logprobs


class ReinforceRLTechnique(RLTechnique):
    name = "reinforce"

    def __init__(self, cfg: RLConfig) -> None:
        self.reward_scale = cfg.reward_scale
        self.baseline_momentum = cfg.baseline_momentum
        self._baseline: float | None = None

    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict,
        *,
        loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> RLStepResult:
        del loss_fn
        if "rewards" not in batch:
            raise KeyError("REINFORCE batch requires rewards tensor")

        logprobs = model_sequence_logprobs(model, batch)
        rewards = batch["rewards"].to(logprobs.device, dtype=logprobs.dtype) * self.reward_scale
        reward_mean = float(rewards.mean())
        if self._baseline is None:
            self._baseline = reward_mean
        else:
            self._baseline = (
                self.baseline_momentum * self._baseline
                + (1.0 - self.baseline_momentum) * reward_mean
            )
        advantages = rewards - self._baseline
        loss = -(logprobs * advantages).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return RLStepResult(
            loss=loss,
            metrics={
                "grad_norm": float(grad_norm),
                "reward_mean": reward_mean,
                "baseline": float(self._baseline),
            },
        )
