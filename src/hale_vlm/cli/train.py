"""Training CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from hale_vlm.config.experiment import apply_experiment_layout
from hale_vlm.registry import get_trainer

from hale_vlm.config import load_vlm_config
from hale_vlm.data.vlm import MultimodalDataModule
from hale_vlm.training.evaluator import VLMEvaluator
from hale_vlm.training.scratch_trainer import ScratchVLMTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VLM model (Hale or scratch backends)")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run-name", type=str, default=None, help="Optional experiment run name")
    args = parser.parse_args()

    cfg = load_vlm_config(str(args.config))
    if args.resume:
        cfg.train.resume = True

    if cfg.resolved_train_backend() == "scratch":
        ScratchVLMTrainer.from_config(cfg).fit()
        return

    apply_experiment_layout(cfg, create=not args.resume, name=args.run_name)

    data_module = MultimodalDataModule(cfg, tokenizer=None)
    trainer = get_trainer("vlm")(
        cfg,
        data_module=data_module,
        evaluator=VLMEvaluator(),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
