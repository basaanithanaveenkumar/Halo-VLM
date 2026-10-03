"""Built-in SmolVLA paper VLA dataset registrations."""

from __future__ import annotations

from typing import TypeVar

from hale_vlm.data.vla.adapters.lerobot import VLADataAdapter
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name
from hale_vlm.data.vla.catalog import (
    EMBODIMENT_BY_STAGE,
    REAL_WORLD_DATASET_NOTES,
    SIMULATION_DATASET_NOTES,
    SMOLVLA_COMMUNITY_DATASETS,
)
from hale_vlm.data.vla.registry import register_vla_dataset

T = TypeVar("T", bound=VLADataAdapter)


def _make_community_adapter(hf_path: str) -> type[VLADataAdapter]:
    name = hf_path_to_registry_name(hf_path)

    @register_vla_dataset(name)
    class _CommunityAdapter(VLADataAdapter):
        spec = VLADatasetSpec(
            name=name,
            hf_path=hf_path,
            stage=VLAStage.COMMUNITY,
            embodiment=EMBODIMENT_BY_STAGE[VLAStage.COMMUNITY],
            description=f"Community SO-100 dataset: {hf_path}",
            paper_reference="SmolVLA Appendix A.1",
        )

    _CommunityAdapter.__name__ = f"Community_{name.replace('-', '_')}"
    _CommunityAdapter.__qualname__ = _CommunityAdapter.__name__
    return _CommunityAdapter


for _hf_path in SMOLVLA_COMMUNITY_HF_PATHS:
    _make_community_adapter(_hf_path)


@register_vla_dataset("libero")
class LiberoAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero",
        hf_path="physical-intelligence/libero",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        description=SIMULATION_DATASET_NOTES["libero"],
        episodes=1693,
    )


@register_vla_dataset("metaworld-mt50")
class MetaWorldMT50Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="metaworld-mt50",
        hf_path="lerobot/metaworld_mt50",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.SAWYER,
        description=SIMULATION_DATASET_NOTES["metaworld-mt50"],
        episodes=2500,
    )


@register_vla_dataset("svla-so100-pickplace")
class SvlaSO100PickPlaceAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so100-pickplace",
        hf_path="lerobot/svla_so100_pickplace",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO100,
        description=REAL_WORLD_DATASET_NOTES["svla-so100-pickplace"],
        episodes=50,
    )


@register_vla_dataset("svla-so100-stacking")
class SvlaSO100StackingAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so100-stacking",
        hf_path="lerobot/svla_so100_stacking",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO100,
        description=REAL_WORLD_DATASET_NOTES["svla-so100-stacking"],
        episodes=50,
    )


@register_vla_dataset("svla-so100-sorting")
class SvlaSO100SortingAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so100-sorting",
        hf_path="lerobot/svla_so100_sorting",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO100,
        description=REAL_WORLD_DATASET_NOTES["svla-so100-sorting"],
        episodes=50,
    )


@register_vla_dataset("svla-so101-pickplace")
class SvlaSO101PickPlaceAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="svla-so101-pickplace",
        hf_path="lerobot/svla_so101_pickplace",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.SO101,
        description=REAL_WORLD_DATASET_NOTES["svla-so101-pickplace"],
        episodes=50,
    )


__all__ = ["SMOLVLA_COMMUNITY_DATASETS"]
