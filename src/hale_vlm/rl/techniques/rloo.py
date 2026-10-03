"""REINFORCE Leave-One-Out (RLOO) baseline for VLM RL."""

from __future__ import annotations

from collections.abc import Callable

import torch

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, model_sequence_logprobs


class RLOORLTechnique(RLTechnique):
    name = "rloo"

    def __init__(self, cfg: RLConfig) -> None:
        self.group_size = cfg.group_size
        self.reward_scale = cfg.reward_scale

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
            raise KeyError("RLOO batch requires rewards tensor")

        logprobs = model_sequence_logprobs(model, batch)
        rewards = batch["rewards"].to(logprobs.device, dtype=logprobs.dtype) * self.reward_scale
        batch_size = rewards.shape[0]
        if batch_size % self.group_size != 0:
            raise ValueError(
                f"batch size {batch_size} must be divisible by group_size={self.group_size}"
            )

        grouped = rewards.view(-1, self.group_size)
        group_sum = grouped.sum(dim=1, keepdim=True)
        # Leave-one-out baseline: (sum - r_i) / (G - 1)
        baselines = (group_sum - grouped) / max(self.group_size - 1, 1)
        advantages = (grouped - baselines).reshape(-1)
        group_logps = logprobs.view(-1, self.group_size).reshape(-1)
        loss = -(group_logps * advantages).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return RLStepResult(
            loss=loss,
            metrics={
                "grad_norm": float(grad_norm),
                "reward_mean": float(rewards.mean()),
                "advantage_mean": float(advantages.mean()),
            },
        )
