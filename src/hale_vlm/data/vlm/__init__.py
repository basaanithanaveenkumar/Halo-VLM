"""VLM data: SmolVLM registry, multimodal dataloaders, scratch COCO."""

from hale_vlm.data.vlm.catalog import (
    SMOLVLM_ALL_DATASETS,
    SMOLVLM_CONTEXT_DATASETS,
    SMOLVLM_REJECTED_DATASETS,
    SMOLVLM_VIDEO_DATASETS,
    SMOLVLM_VISION_DATASETS,
    STAGE_PRESETS,
    VIDEO_CATEGORY_NOTES,
    VISION_CATEGORY_NOTES,
)
from hale_vlm.data.vlm.multimodal import MultimodalDataModule, MultimodalDataset
from hale_vlm.data.vlm.registry import (
    DATASETS,
    build_dataset,
    get_dataset,
    list_datasets,
    register_dataset,
)
from hale_vlm.data.vlm.sequential import SequentialMixConfig, SequentialMultiDatasetStream
from hale_vlm.data.vlm.stream import iter_registry_vlm_samples, iter_vlm_registry_samples

__all__ = [
    "DATASETS",
    "MultimodalDataModule",
    "MultimodalDataset",
    "SMOLVLM_ALL_DATASETS",
    "SMOLVLM_CONTEXT_DATASETS",
    "SMOLVLM_REJECTED_DATASETS",
    "SMOLVLM_VIDEO_DATASETS",
    "SMOLVLM_VISION_DATASETS",
    "STAGE_PRESETS",
    "SequentialMixConfig",
    "SequentialMultiDatasetStream",
    "VIDEO_CATEGORY_NOTES",
    "VISION_CATEGORY_NOTES",
    "build_dataset",
    "get_dataset",
    "iter_registry_vlm_samples",
    "iter_vlm_registry_samples",
    "list_datasets",
    "register_dataset",
]
