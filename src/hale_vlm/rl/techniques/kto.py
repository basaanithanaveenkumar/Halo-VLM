"""Kahneman-Tversky Optimization (KTO) for binary feedback alignment."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.rl.base import RLStepResult, RLTechnique, model_sequence_logprobs


class KTORLTechnique(RLTechnique):
    name = "kto"

    def __init__(self, cfg: RLConfig) -> None:
        self.beta = cfg.beta
        self.w_desirable = cfg.kto_desirable_weight
        self.w_undesirable = cfg.kto_undesirable_weight

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
        if "kto_labels" not in batch:
            raise KeyError("KTO batch requires kto_labels (+1 desirable, -1 undesirable)")

        logprobs = model_sequence_logprobs(model, batch)
        labels = batch["kto_labels"].to(logprobs.device, dtype=logprobs.dtype)
        desirable = labels > 0
        undesirable = labels < 0

        losses = []
        if desirable.any():
            losses.append(
                self.w_desirable
                * (-F.logsigmoid(self.beta * logprobs[desirable])).mean()
            )
        if undesirable.any():
            losses.append(
                self.w_undesirable
                * (-F.logsigmoid(-self.beta * logprobs[undesirable])).mean()
            )
        if not losses:
            raise ValueError("KTO batch contains no labeled examples")
        loss = torch.stack(losses).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return RLStepResult(
            loss=loss,
            metrics={
                "grad_norm": float(grad_norm),
                "desirable_frac": float(desirable.float().mean()),
                "undesirable_frac": float(undesirable.float().mean()),
            },
        )
