"""Variant-agnostic training loop resolved from registries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from loguru import logger
from tqdm import tqdm

from hale_vlm.config.experiment import apply_experiment_layout
from hale_vlm.config.run import RunConfig
from hale_vlm.utils.logging_setup import setup_logging
from hale_vlm.utils.optim import build_optimizer
from hale_vlm.registry import build_logger, get_loss, get_trainer, get_variant, register_trainer
from hale_vlm.rl.factory import build_rl_technique
from hale_vlm.utils.runtime.checkpoint import CheckpointStore
from hale_vlm.utils.runtime.device import get_device
from hale_vlm.utils.runtime.tensors import count_parameters, log_model_summary, move_batch_to_device
from hale_vlm.training.distributed import (
    cleanup_distributed,
    is_main_process,
    reduce_dict,
    setup_distributed,
    wrap_model_parallel,
)
from hale_vlm.training.schedule import TrainSchedule


class DataModule(Protocol):
    tokenizer: Any

    def train_loader(self): ...

    def val_loader(self, train_loader): ...

    def viz_prompt(self, step: int) -> str: ...


class Evaluator(Protocol):
    def run(self, model, loader) -> dict[str, float]: ...


@register_trainer("default")
class Trainer:
    def __init__(
        self,
        cfg: RunConfig,
        tokenizer=None,
        *,
        data_module: DataModule | None = None,
        checkpoints: CheckpointStore | None = None,
        evaluator: Evaluator | None = None,
        on_visualize: Callable[[Trainer, str, bool], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._tokenizer = tokenizer
        self.data = data_module
        self.checkpoints = checkpoints or CheckpointStore()
        self.evaluator = evaluator
        self.on_visualize = on_visualize
        self.dist_config = setup_distributed(
            strategy=cfg.train.parallel_strategy,
            backend=cfg.train.distributed_backend,
        )
        self.device = self._get_device(cfg.device)
        self.tokenizer = tokenizer
        self.model = None
        self.opt = None
        self.loss_fn = None
        self.rl_technique = None
        self.logger = None
        self.losses: list[float] = []
        self.global_step = 0
        self.start_epoch = 0
        self.schedule: TrainSchedule | None = None

    def _get_device(self, device_str: str | None) -> torch.device:
        if self.dist_config.strategy in ("ddp", "fsdp") and self.dist_config.is_initialized:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.dist_config.local_rank}")
                torch.cuda.set_device(device)
                return device
        return torch.device(get_device(device_str))

    def fit(self) -> tuple:
        cfg = self.cfg
        apply_experiment_layout(cfg, create=True)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        torch.manual_seed(cfg.train.seed)

        if self.data is None:
            raise TypeError("Trainer requires a data_module")
        self.tokenizer = self.data.tokenizer
        if self.evaluator is None:
            raise TypeError("Trainer requires an evaluator")

        vocab_size = len(self.tokenizer)
        model_cls = get_variant(cfg.variant)
        self.loss_fn = get_loss(cfg.variant)
        self.rl_technique = build_rl_technique(cfg)
        self.model = model_cls(vocab_size=vocab_size, cfg=cfg).to(self.device)
        n_params = count_parameters(self.model)
        self.model = wrap_model_parallel(self.model, self.dist_config, self.device)

        if is_main_process(self.dist_config):
            log_model_summary(self.model, title=f"model summary  variant={cfg.variant}")

        loader = self.data.train_loader()
        if len(loader) == 0:
            raise RuntimeError("empty dataloader")

        self.schedule = TrainSchedule.from_config(cfg, len(loader))
        n_epochs, total_steps = self.schedule.n_epochs, self.schedule.total_steps

        if is_main_process(self.dist_config):
            logger.info(
                "starting train variant={} device={} epochs={} steps={} lr={} rl={}",
                cfg.variant,
                self.device,
                n_epochs,
                total_steps,
                cfg.train.lr,
                cfg.rl.technique,
            )

        self.opt = build_optimizer(
            cfg.train.optimizer_type,
            self.model.parameters(),
            {
                "lr": cfg.train.lr,
                "weight_decay": cfg.train.weight_decay,
                "grad_clip": cfg.train.grad_clip,
            },
        )
        self._maybe_resume(n_epochs)

        self.logger = build_logger(
            cfg.logging.backend,
            cfg=cfg,
            main_process=is_main_process(self.dist_config),
        )
        self.logger.log_hparams(
            {
                "variant": cfg.variant,
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "d_model": cfg.model.d_model,
                "n_layers": cfg.model.n_layers,
                "epochs": n_epochs,
                "steps": total_steps,
            },
        )
        self.logger.log_text("config", str(cfg.model_dump()))
        self.logger.log_scalars(0, {"model/n_params": float(n_params)})

        if cfg.logging.wandb and cfg.logging.wandb_watch_model:
            self.logger.watch_model(self.model, log_freq=cfg.logging.wandb_log_freq)

        self.losses = []
        try:
            self._run_epochs(loader, vocab_size)
        except Exception:
            logger.exception("training crashed at last_step={}", self.global_step)
            if self.logger is not None:
                self.logger.close()
            cleanup_distributed(self.dist_config)
            raise

        if is_main_process(self.dist_config):
            self._save(epoch=n_epochs - 1, vocab_size=vocab_size)
        if self.logger is not None:
            if self.losses:
                self.logger.log_scalars(self.global_step, {"train/final_loss": self.losses[-1]})
            self.logger.close()
        cleanup_distributed(self.dist_config)
        return self.model, self.tokenizer, self.losses

    def _maybe_resume(self, n_epochs: int) -> None:
        cfg = self.cfg
        ckpt_path = cfg.train.checkpoint_path
        self.start_epoch = 0
        self.global_step = 0
        if cfg.train.resume and self.checkpoints.exists(ckpt_path):
            ckpt = self.checkpoints.load(ckpt_path, self.device)
            model_to_load = self.model.module if hasattr(self.model, "module") else self.model
            model_to_load.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.opt.load_state_dict(ckpt["optimizer_state_dict"])
            self.start_epoch = int(ckpt.get("epoch", -1)) + 1
            self.global_step = int(ckpt.get("step", 0))

    def _save(self, *, epoch: int, vocab_size: int) -> None:
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        self.checkpoints.save(
            self.cfg.train.checkpoint_path,
            model_state_dict=model_to_save.state_dict(),
            optimizer_state_dict=self.opt.state_dict(),
            variant=self.cfg.variant,
            vocab_size=vocab_size,
            epoch=epoch,
            step=self.global_step,
            config=self.cfg.model_dump(),
        )

    def _run_epochs(self, loader, vocab_size: int) -> None:
        cfg = self.cfg
        n_epochs = self.schedule.n_epochs
        total_steps = self.schedule.total_steps
        for epoch in range(self.start_epoch, n_epochs):
            self.model.train()
            epoch_losses: list[float] = []
            pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=True)
            for batch in pbar:
                if self.global_step >= total_steps:
                    break
                batch = move_batch_to_device(batch, self.device)
                step = self.rl_technique.training_step(
                    self.model,
                    batch,
                    loss_fn=self.loss_fn,
                    optimizer=self.opt,
                    grad_clip=cfg.train.grad_clip,
                )
                loss = step.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {self.global_step}")

                grad_norm = step.metrics.get("grad_norm", 0.0)
                lr = self.opt.param_groups[0]["lr"]
                self.losses.append(loss.item())
                epoch_losses.append(loss.item())
                log_payload = {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": float(epoch),
                    "train/rl_technique": float(cfg.rl.enabled()),
                }
                for key, value in step.metrics.items():
                    if key != "grad_norm":
                        log_payload[f"rl/{key}"] = value
                self.logger.log_scalars(self.global_step, log_payload)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                self.global_step += 1
                self._maybe_visualize(tag=f"step{self.global_step}")

            epoch_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
            self.logger.log_scalars(epoch, {"train/epoch_loss": epoch_mean})

            if cfg.eval.every_n_epochs and (epoch + 1) % cfg.eval.every_n_epochs == 0:
                eval_loader = self.data.val_loader(loader)
                eval_metrics = self.evaluator.run(self.model, eval_loader)
                eval_metrics = reduce_dict(eval_metrics, self.dist_config, average=True)
                self.logger.log_scalars(epoch, {f"eval/{k}": v for k, v in eval_metrics.items()})

            if is_main_process(self.dist_config):
                self._maybe_visualize(tag=f"epoch{epoch + 1}", by_epoch=True)
            if cfg.train.checkpoint_every_epoch and is_main_process(self.dist_config):
                self._save(epoch=epoch, vocab_size=vocab_size)

    def _maybe_visualize(self, *, tag: str, by_epoch: bool = False) -> None:
        if self.on_visualize is None:
            return
        cfg = self.cfg
        if not cfg.viz.enabled:
            return
        if by_epoch:
            n = cfg.viz.every_n_epochs
            if not n:
                return
            epoch_n = int(tag.replace("epoch", ""))
            if epoch_n % n != 0:
                return
        else:
            n = cfg.viz.every_n_steps
            if not n or self.global_step % n != 0:
                return
        self.on_visualize(self, tag, by_epoch)


def train(cfg: RunConfig, tokenizer=None, *, trainer: str = "default", **kwargs) -> tuple:
    return get_trainer(trainer)(cfg, tokenizer, **kwargs).fit()
