"""Proximal Policy Optimization (PPO) for VLM RL fine-tuning."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, model_sequence_logprobs


class PPORLTechnique(RLTechnique):
    name = "ppo"

    def __init__(self, cfg: RLConfig) -> None:
        self.clip_range = cfg.ppo_clip_range
        self.kl_coef = cfg.kl_coef
        self.entropy_coef = cfg.entropy_coef

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
        if "advantages" not in batch or "old_logprobs" not in batch:
            raise KeyError("PPO batch requires advantages and old_logprobs")

        logprobs = model_sequence_logprobs(model, batch)
        old_logprobs = batch["old_logprobs"].to(logprobs.device, dtype=logprobs.dtype)
        advantages = batch["advantages"].to(logprobs.device, dtype=logprobs.dtype)

        ratio = torch.exp(logprobs - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        kl = (old_logprobs - logprobs).mean()
        loss = policy_loss + self.kl_coef * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return RLStepResult(
            loss=loss,
            metrics={
                "grad_norm": float(grad_norm),
                "policy_loss": float(policy_loss.detach()),
                "approx_kl": float(kl.detach()),
                "ratio_mean": float(ratio.mean()),
            },
        )
