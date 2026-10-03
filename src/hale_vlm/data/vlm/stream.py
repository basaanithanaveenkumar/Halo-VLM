"""VLM registry streaming and optional VLA bridge."""

from __future__ import annotations

from collections.abc import Iterator

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.data.types import RoboticsVLMMode, VLMSample
from hale_vlm.data.vlm.sequential import SequentialMixConfig, SequentialMultiDatasetStream


def iter_vlm_registry_samples(
    data_cfg: VLMDataConfig,
    *,
    image_size: int,
) -> Iterator[VLMSample]:
    """Yield samples from the SmolVLM dataset registry."""
    if data_cfg.source not in {"registry", "mixed_registry"}:
        return
    stream = SequentialMultiDatasetStream(
        SequentialMixConfig(
            dataset_names=data_cfg.resolved_registry_datasets(),
            max_samples_per_dataset=data_cfg.max_samples_per_dataset,
            split=data_cfg.train_split,
            cache_dir=data_cfg.cache_dir,
            prefetch_workers=data_cfg.prefetch_workers,
            streaming=data_cfg.streaming,
            max_video_frames=data_cfg.max_video_frames,
            image_size=image_size,
        )
    )
    yield from stream


def _needs_vla_bridge(data_cfg: VLMDataConfig) -> bool:
    return data_cfg.source in {"vla_registry", "mixed_registry"} or (
        data_cfg.source == "registry" and data_cfg.robotics_in_vlm_enabled()
    )


def iter_registry_vlm_samples(
    data_cfg: VLMDataConfig,
    *,
    image_size: int,
) -> Iterator[VLMSample]:
    """Yield VLMSample from VLM registry and optionally robotics (VLA) registry."""
    yield from iter_vlm_registry_samples(data_cfg, image_size=image_size)
    if not _needs_vla_bridge(data_cfg):
        return

    from hale_vlm.data.vla.bridge import iter_vla_as_vlm_samples

    mode = data_cfg.robotics_vlm_mode
    if mode == RoboticsVLMMode.OFF:
        mode = RoboticsVLMMode.INSTRUCTION_TUNING
    yield from iter_vla_as_vlm_samples(data_cfg, mode=mode)
