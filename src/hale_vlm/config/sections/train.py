from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.common import StrictModel


class TrainConfig(StrictModel):
    """Generic training loop settings."""

    steps: int | None = 300
    epochs: int | None = None
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True
    loss_type: Literal["ce", "focal", "label_smoothing"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    optimizer_type: str = "adamw"
    parallel_strategy: Literal["none", "dp", "ddp", "fsdp"] = "none"
    distributed_backend: str | None = None
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False


class VLMTrainConfig(TrainConfig):
    """Training settings extended for scratch and Hale backends."""

    backend: Literal["auto", "haleblocks", "scratch", "vla"] = "auto"
    max_epochs: int = 10
    log_tensorboard: bool = False
    log_dir: str = "./runs/hale_vlm"
    scratch_checkpoint_dir: str = "checkpoints/scratch"
