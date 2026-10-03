from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str | None = "logs/hale.log"
    backend: str = "experiment"

    # TensorBoard configuration
    tensorboard: bool = True
    tensorboard_dir: str = "runs"

    # Weights & Biases configuration
    wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_watch_model: bool = False
    wandb_log_freq: int = 100
