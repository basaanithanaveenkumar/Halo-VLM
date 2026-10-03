"""VLA / robotics data (SmolVLA paper). Install with ``pip install hale-vlm[vla]``."""

from hale_vlm.data.vla.bridge import (
    format_robotics_instruction,
    iter_vla_as_vlm_samples,
    vla_sample_to_vlm,
)
from hale_vlm.data.vla.catalog import (
    SMOLVLA_ALL_DATASETS,
    SMOLVLA_COMMUNITY_DATASETS,
    SMOLVLA_REAL_WORLD_DATASETS,
    SMOLVLA_SIMULATION_DATASETS,
    VLA_STAGE_PRESETS,
)
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name
from hale_vlm.data.vla.registry import (
    VLA_DATASETS,
    build_vla_dataset,
    get_vla_dataset,
    list_vla_datasets,
    register_vla_dataset,
)
from hale_vlm.data.vla.sequential import SequentialVLAStream, VLAStreamConfig

__all__ = [
    "SMOLVLA_ALL_DATASETS",
    "SMOLVLA_COMMUNITY_DATASETS",
    "SMOLVLA_COMMUNITY_HF_PATHS",
    "SMOLVLA_REAL_WORLD_DATASETS",
    "SMOLVLA_SIMULATION_DATASETS",
    "SequentialVLAStream",
    "VLAStreamConfig",
    "VLA_DATASETS",
    "VLA_STAGE_PRESETS",
    "build_vla_dataset",
    "format_robotics_instruction",
    "get_vla_dataset",
    "hf_path_to_registry_name",
    "iter_vla_as_vlm_samples",
    "list_vla_datasets",
    "register_vla_dataset",
    "vla_sample_to_vlm",
]
