"""VLA / robotics dataset types (SmolVLA paper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from PIL import Image


class VLAStage(StrEnum):
    """SmolVLA paper dataset stages."""

    COMMUNITY = "community"
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"


class VLATrainingPhase(StrEnum):
    """High-level training phase for VLA robot policies.

      PRETRAIN   — large-scale open robot teleoperation (Open-X, Bridge V2, RT-1)
      MID_TRAIN  — domain-specific manipulation data (DROID, AIRoA-MoMA)
      POST_TRAIN — task-specific fine-tuning on target embodiment
    """

    PRETRAIN = "pretrain"
    MID_TRAIN = "mid_train"
    POST_TRAIN = "post_train"


class RobotEmbodiment(StrEnum):
    """Robot platforms referenced in SmolVLA."""

    SO100 = "so100"
    SO101 = "so101"
    PANDA = "panda"
    SAWYER = "sawyer"
    MIXED = "mixed"


@dataclass(frozen=True)
class VLADatasetSpec:
    """Static metadata for a registered VLA / robotics dataset."""

    name: str
    hf_path: str
    stage: VLAStage
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    description: str = ""
    paper_reference: str = "SmolVLA (arXiv:2506.01844)"
    enabled: bool = True
    config_name: str | None = None
    default_split: str = "train"
    task_fields: tuple[str, ...] = ("task", "language_instruction", "prompt", "instruction")
    action_fields: tuple[str, ...] = ("action", "actions")
    state_fields: tuple[str, ...] = ("observation.state", "state", "proprio")
    image_fields: tuple[str, ...] = ("image", "images")
    camera_prefix: str = "observation.images"
    streaming: bool = True
    trust_remote_code: bool = False
    episodes: int | None = None
    # High-level training phase (pretrain / mid_train / post_train).
    # None means the dataset belongs only to the SmolVLA stage taxonomy.
    phase: VLATrainingPhase | None = None


@dataclass
class VLASample:
    """Normalized robotics demonstration frame."""

    dataset: str
    stage: VLAStage
    task: str
    images: list[Image.Image] = field(default_factory=list)
    action: list[float] | None = None
    state: list[float] | None = None
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return bool(self.images)
