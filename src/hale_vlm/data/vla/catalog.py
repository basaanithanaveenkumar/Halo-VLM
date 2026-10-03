"""SmolVLA paper dataset catalog and stage presets."""

from __future__ import annotations

from hale_vlm.data.types import RobotEmbodiment, VLAStage
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name

# Simulation benchmarks (§4.1)
SMOLVLA_SIMULATION_DATASETS: tuple[str, ...] = (
    "libero",
    "metaworld-mt50",
)

# Real-world author datasets (§4.1, Figure 4)
SMOLVLA_REAL_WORLD_DATASETS: tuple[str, ...] = (
    "svla-so100-pickplace",
    "svla-so100-stacking",
    "svla-so100-sorting",
    "svla-so101-pickplace",
)

# Community SO-100 pretraining (Appendix A.1)
SMOLVLA_COMMUNITY_DATASETS: tuple[str, ...] = tuple(
    hf_path_to_registry_name(path) for path in SMOLVLA_COMMUNITY_HF_PATHS
)

SMOLVLA_ALL_DATASETS: tuple[str, ...] = (
    *SMOLVLA_COMMUNITY_DATASETS,
    *SMOLVLA_SIMULATION_DATASETS,
    *SMOLVLA_REAL_WORLD_DATASETS,
)

VLA_STAGE_PRESETS: dict[VLAStage, tuple[str, ...]] = {
    VLAStage.COMMUNITY: SMOLVLA_COMMUNITY_DATASETS,
    VLAStage.SIMULATION: SMOLVLA_SIMULATION_DATASETS,
    VLAStage.REAL_WORLD: SMOLVLA_REAL_WORLD_DATASETS,
}

SIMULATION_DATASET_NOTES: dict[str, str] = {
    "libero": "LIBERO — 40 tasks, 1,693 episodes (Liu et al., 2023a)",
    "metaworld-mt50": "Meta-World MT50 — 50 tasks, 2,500 episodes (Yu et al., 2020)",
}

REAL_WORLD_DATASET_NOTES: dict[str, str] = {
    "svla-so100-pickplace": "Pick cube, place in box — 50 demos on SO-100",
    "svla-so100-stacking": "Stack red cube on blue cube — 50 demos on SO-100",
    "svla-so100-sorting": "Sort colored cubes into boxes — 50 demos on SO-100",
    "svla-so101-pickplace": "Pick pink Lego, place in box — 50 demos on SO-101",
}

COMMUNITY_STATS = {
    "paper_reported_datasets": 481,
    "appendix_hf_paths": len(SMOLVLA_COMMUNITY_HF_PATHS),
    "paper_reported_episodes": 22_900,
    "paper_reported_frames": 10_600_000,
}

EMBODIMENT_BY_STAGE: dict[VLAStage, RobotEmbodiment] = {
    VLAStage.COMMUNITY: RobotEmbodiment.SO100,
    VLAStage.SIMULATION: RobotEmbodiment.MIXED,
    VLAStage.REAL_WORLD: RobotEmbodiment.SO100,
}
