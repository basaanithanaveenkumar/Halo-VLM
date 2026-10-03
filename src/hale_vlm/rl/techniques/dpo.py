"""Direct Preference Optimization (DPO) for VLM alignment."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, model_sequence_logprobs, preference_batch


class DPORLTechnique(RLTechnique):
    name = "dpo"

    def __init__(self, cfg: RLConfig) -> None:
        self.beta = cfg.beta

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
        chosen = preference_batch(batch, "chosen")
        rejected = preference_batch(batch, "rejected")
        chosen_logps = model_sequence_logprobs(model, chosen)
        rejected_logps = model_sequence_logprobs(model, rejected)
        logits = self.beta * (chosen_logps - rejected_logps)
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        margin = float((chosen_logps - rejected_logps).mean().detach())
        return RLStepResult(
            loss=loss,
            metrics={
                "grad_norm": float(grad_norm),
                "chosen_logp": float(chosen_logps.mean().detach()),
                "rejected_logp": float(rejected_logps.mean().detach()),
                "preference_margin": margin,
            },
        )
