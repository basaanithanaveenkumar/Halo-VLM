"""Base and VLM run configs."""

from typing import Literal

from pydantic import Field

from hale_vlm.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)
from hale_vlm.config.sections.common import StrictModel
from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.config.sections.model import VLMModelConfig
from hale_vlm.config.sections.rl import RLConfig
from hale_vlm.config.sections.train import VLMTrainConfig


class RunConfig(StrictModel):
    """Generic training run configuration (shared trainer infrastructure)."""

    variant: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    device: str | None = None


class VLMRunConfig(RunConfig):
    """Full configuration for vision-language model training and inference."""

    model: VLMModelConfig = Field(default_factory=VLMModelConfig)
    train: VLMTrainConfig = Field(default_factory=VLMTrainConfig)
    data: VLMDataConfig = Field(default_factory=VLMDataConfig)
    rl: RLConfig = Field(default_factory=RLConfig)

    def resolved_train_backend(self) -> Literal["haleblocks", "scratch", "vla"]:
        if self.train.backend != "auto":
            return self.train.backend
        architecture = self.model.resolved_architecture(self.variant)
        if architecture == "gwm_vla":
            return "vla"
        return "scratch" if architecture in {"basic", "halo_moe"} else "haleblocks"
