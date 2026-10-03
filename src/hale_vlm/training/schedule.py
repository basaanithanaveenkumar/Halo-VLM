"""Training schedule helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

from hale_vlm.config.run import RunConfig


@dataclass(frozen=True)
class TrainSchedule:
    n_epochs: int
    total_steps: int

    @classmethod
    def from_config(cls, cfg: RunConfig, n_batches: int) -> TrainSchedule:
        if n_batches < 1:
            raise RuntimeError("empty dataloader")
        if cfg.train.epochs is not None:
            if cfg.train.steps is not None:
                logger.warning(
                    "both epochs={} and steps={} set; using epochs ({} steps)",
                    cfg.train.epochs,
                    cfg.train.steps,
                    cfg.train.epochs * n_batches,
                )
            return cls(n_epochs=cfg.train.epochs, total_steps=cfg.train.epochs * n_batches)
        if cfg.train.steps is None:
            logger.error("set train.epochs or train.steps")
            raise ValueError("set train.epochs or train.steps")
        return cls(n_epochs=math.ceil(cfg.train.steps / n_batches), total_steps=cfg.train.steps)


def resolve_schedule(cfg: RunConfig, n_batches: int) -> tuple[int, int]:
    schedule = TrainSchedule.from_config(cfg, n_batches)
    return schedule.n_epochs, schedule.total_steps
