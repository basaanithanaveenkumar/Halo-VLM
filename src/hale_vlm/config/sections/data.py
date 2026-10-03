from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.common import StrictModel
from hale_vlm.data.types.bridge import RoboticsVLMMode


class DataConfig(StrictModel):
    """Generic dataset settings (text LM defaults)."""

    source: Literal["huggingface", "overfit"] = "huggingface"
    dataset: str = "Salesforce/wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    cache_dir: str | None = None
    tokenizer_name: str = "gpt2"
    train_size: int | None = None
    val_size: int | None = None
    overfit_text: str | None = None
    n_overfit_copies: int = 64
    add_special_tokens: bool = False
    stride_words: int = 10


class VLMDataConfig(DataConfig):
    """VLM-specific data configuration."""

    source: Literal[
        "huggingface",
        "overfit",
        "registry",
        "vla_registry",
        "mixed_registry",
        "coco_lavis",
    ] = "huggingface"
    registry_stage: Literal["all", "vision", "video", "context"] = "all"
    registry_datasets: list[str] = Field(default_factory=list)
    vla_registry_stage: Literal["all", "community", "simulation", "real_world"] = "all"
    vla_registry_datasets: list[str] = Field(default_factory=list)
    robotics_vlm_mode: RoboticsVLMMode = RoboticsVLMMode.OFF
    max_samples_per_dataset: int | None = 256
    streaming: bool = True
    prefetch_workers: int = 2
    max_video_frames: int = 8

    def model_post_init(self, __context) -> None:
        if not self.registry_datasets:
            from hale_vlm.data.vlm.catalog import SMOLVLM_ALL_DATASETS

            self.registry_datasets = list(SMOLVLM_ALL_DATASETS)
        if not self.vla_registry_datasets:
            from hale_vlm.data.vla.catalog import SMOLVLA_ALL_DATASETS

            self.vla_registry_datasets = list(SMOLVLA_ALL_DATASETS)

    def resolved_registry_datasets(self) -> list[str]:
        from hale_vlm.data.vlm.catalog import (
            SMOLVLM_ALL_DATASETS,
            SMOLVLM_CONTEXT_DATASETS,
            SMOLVLM_VIDEO_DATASETS,
            SMOLVLM_VISION_DATASETS,
        )

        presets = {
            "vision": SMOLVLM_VISION_DATASETS,
            "video": SMOLVLM_VIDEO_DATASETS,
            "context": SMOLVLM_CONTEXT_DATASETS,
            "all": SMOLVLM_ALL_DATASETS,
        }
        if self.registry_stage != "all":
            return list(presets[self.registry_stage])
        return self.registry_datasets

    def resolved_vla_registry_datasets(self) -> list[str]:
        from hale_vlm.data.vla.catalog import (
            SMOLVLA_ALL_DATASETS,
            SMOLVLA_COMMUNITY_DATASETS,
            SMOLVLA_REAL_WORLD_DATASETS,
            SMOLVLA_SIMULATION_DATASETS,
        )

        presets = {
            "community": SMOLVLA_COMMUNITY_DATASETS,
            "simulation": SMOLVLA_SIMULATION_DATASETS,
            "real_world": SMOLVLA_REAL_WORLD_DATASETS,
            "all": SMOLVLA_ALL_DATASETS,
        }
        if self.vla_registry_stage != "all":
            return list(presets[self.vla_registry_stage])
        return self.vla_registry_datasets

    def robotics_in_vlm_enabled(self) -> bool:
        return self.robotics_vlm_mode != RoboticsVLMMode.OFF

    def uses_registry_stream(self) -> bool:
        return self.source in {"registry", "mixed_registry"} or self.robotics_in_vlm_enabled()
