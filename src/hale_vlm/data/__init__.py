"""Data layer for Hale-VLM.

- ``hale_vlm.data.vlm`` — SmolVLM registry and multimodal dataloaders (core)
- ``hale_vlm.data.vla`` — optional robotics / SmolVLA registry
- ``hale_vlm.data.types`` — shared sample and config types
"""

from hale_vlm.data.types import (
    DatasetSpec,
    Modality,
    RoboticsVLMMode,
    TrainingStage,
    VideoCategory,
    VisionCategory,
    VLMSample,
)

__all__ = [
    "DatasetSpec",
    "Modality",
    "RoboticsVLMMode",
    "TrainingStage",
    "VIDEO_CATEGORY_NOTES",
    "VISION_CATEGORY_NOTES",
    "VideoCategory",
    "VisionCategory",
    "VLMSample",
]

_LAZY_EXPORTS = {
    "DATASETS": ("hale_vlm.data.vlm.registry", "DATASETS"),
    "MultimodalDataModule": ("hale_vlm.data.vlm.multimodal", "MultimodalDataModule"),
    "MultimodalDataset": ("hale_vlm.data.vlm.multimodal", "MultimodalDataset"),
    "SMOLVLM_ALL_DATASETS": ("hale_vlm.data.vlm.catalog", "SMOLVLM_ALL_DATASETS"),
    "SMOLVLM_CONTEXT_DATASETS": ("hale_vlm.data.vlm.catalog", "SMOLVLM_CONTEXT_DATASETS"),
    "SMOLVLM_REJECTED_DATASETS": ("hale_vlm.data.vlm.catalog", "SMOLVLM_REJECTED_DATASETS"),
    "SMOLVLM_VIDEO_DATASETS": ("hale_vlm.data.vlm.catalog", "SMOLVLM_VIDEO_DATASETS"),
    "SMOLVLM_VISION_DATASETS": ("hale_vlm.data.vlm.catalog", "SMOLVLM_VISION_DATASETS"),
    "STAGE_PRESETS": ("hale_vlm.data.vlm.catalog", "STAGE_PRESETS"),
    "SequentialMixConfig": ("hale_vlm.data.vlm.sequential", "SequentialMixConfig"),
    "SequentialMultiDatasetStream": (
        "hale_vlm.data.vlm.sequential",
        "SequentialMultiDatasetStream",
    ),
    "VIDEO_CATEGORY_NOTES": ("hale_vlm.data.vlm.catalog", "VIDEO_CATEGORY_NOTES"),
    "VISION_CATEGORY_NOTES": ("hale_vlm.data.vlm.catalog", "VISION_CATEGORY_NOTES"),
    "build_dataset": ("hale_vlm.data.vlm.registry", "build_dataset"),
    "get_dataset": ("hale_vlm.data.vlm.registry", "get_dataset"),
    "iter_registry_vlm_samples": ("hale_vlm.data.vlm.stream", "iter_registry_vlm_samples"),
    "list_datasets": ("hale_vlm.data.vlm.registry", "list_datasets"),
    "register_dataset": ("hale_vlm.data.vlm.registry", "register_dataset"),
    "SMOLVLA_ALL_DATASETS": ("hale_vlm.data.vla.catalog", "SMOLVLA_ALL_DATASETS"),
    "SMOLVLA_COMMUNITY_DATASETS": ("hale_vlm.data.vla.catalog", "SMOLVLA_COMMUNITY_DATASETS"),
    "SMOLVLA_REAL_WORLD_DATASETS": ("hale_vlm.data.vla.catalog", "SMOLVLA_REAL_WORLD_DATASETS"),
    "SMOLVLA_SIMULATION_DATASETS": ("hale_vlm.data.vla.catalog", "SMOLVLA_SIMULATION_DATASETS"),
    "VLA_DATASETS": ("hale_vlm.data.vla.registry", "VLA_DATASETS"),
    "VLA_STAGE_PRESETS": ("hale_vlm.data.vla.catalog", "VLA_STAGE_PRESETS"),
    "VLADatasetSpec": ("hale_vlm.data.types.vla", "VLADatasetSpec"),
    "VLASample": ("hale_vlm.data.types.vla", "VLASample"),
    "VLAStage": ("hale_vlm.data.types.vla", "VLAStage"),
    "RobotEmbodiment": ("hale_vlm.data.types.vla", "RobotEmbodiment"),
    "build_vla_dataset": ("hale_vlm.data.vla.registry", "build_vla_dataset"),
    "get_vla_dataset": ("hale_vlm.data.vla.registry", "get_vla_dataset"),
    "list_vla_datasets": ("hale_vlm.data.vla.registry", "list_vla_datasets"),
    "register_vla_dataset": ("hale_vlm.data.vla.registry", "register_vla_dataset"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
