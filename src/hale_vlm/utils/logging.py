"""Experiment loggers."""

from __future__ import annotations

from typing import Any

from hale_vlm.config.run import RunConfig
from hale_vlm.registry import register_logger
from hale_vlm.utils.logging_setup import setup_logging


class _NoOpExperimentLogger:
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        del step, metrics

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        del hparams, metrics

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        del tag, text, step

    def log_image(self, tag: str, image, step: int = 0) -> None:
        del tag, image, step

    def log_histogram(self, tag: str, values, step: int = 0) -> None:
        del tag, values, step

    def watch_model(self, model, log_freq: int = 100) -> None:
        del model, log_freq

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@register_logger("noop")
def _noop_logger(*, cfg: RunConfig | None = None, **kwargs) -> _NoOpExperimentLogger:
    del cfg, kwargs
    return _NoOpExperimentLogger()


@register_logger("loguru")
def _loguru_logger(*, cfg: RunConfig | None = None, **kwargs) -> None:
    if cfg is None:
        raise TypeError("loguru logger requires cfg=RunConfig")
    setup_logging(
        level=cfg.logging.level,
        log_file=cfg.logging.log_file,
        force=kwargs.get("force", True),
    )
    return None


@register_logger("experiment")
def _experiment_logger(*, cfg: RunConfig | None = None, **kwargs) -> _NoOpExperimentLogger:
    del kwargs
    if cfg is not None:
        setup_logging(
            level=cfg.logging.level,
            log_file=cfg.logging.log_file,
            force=True,
        )
    return _NoOpExperimentLogger()
