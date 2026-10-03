"""RL training primitives shared across techniques."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class RLStepResult:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class RLTechnique(ABC):
    """One RL alignment method applied during a training step."""

    name: str

    @abstractmethod
    def training_step(
        self,
        model: torch.nn.Module,
        batch: dict,
        *,
        loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> RLStepResult:
        raise NotImplementedError


def _shift_logits_for_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Align causal LM logits with next-token labels."""
    if logits.shape[1] == labels.shape[1]:
        return logits[:, :-1], labels[:, 1:]
    return logits, labels


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean log-probability per sequence (used by DPO/KTO/PPO)."""
    logits, labels = _shift_logits_for_labels(logits, labels)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp(min=0)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(ignore_index).float()
    return (token_logp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def model_sequence_logprobs(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Compute sequence log-probs for Hale or scratch VLM batches."""
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask")

    if "images" in batch and "pixel_values" not in batch:
        logits = model(
            batch["images"],
            batch["input_ids"],
            attention_mask=attention_mask,
        )
        if not isinstance(logits, torch.Tensor):
            raise TypeError("scratch VLM forward must return logits tensor")
    else:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            pixel_values=batch.get("pixel_values"),
            labels=None,
        )
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise TypeError("unsupported model output for RL logprob computation")
    return sequence_logprobs(logits, labels)


def supervised_step(
    model: torch.nn.Module,
    batch: dict,
    *,
    loss_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> RLStepResult:
    """Standard supervised backward pass."""
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return RLStepResult(loss=loss, metrics={"grad_norm": float(grad_norm)})


def preference_batch(batch: dict, prefix: str) -> dict:
    """Extract chosen/rejected sub-batch using ``{prefix}_*`` keys."""
    sub: dict = {}
    marker = f"{prefix}_"
    for key, value in batch.items():
        if key.startswith(marker):
            sub[key[len(marker) :]] = value
    if "input_ids" not in sub:
        raise KeyError(f"missing {prefix}_input_ids in RL preference batch")
    return sub
