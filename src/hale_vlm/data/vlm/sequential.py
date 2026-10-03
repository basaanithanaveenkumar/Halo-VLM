"""Sequential multi-dataset streaming with parallel prefetch."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from loguru import logger

from hale_vlm.data.vlm.registry import build_dataset
from hale_vlm.data.types import VLMSample


@dataclass
class SequentialMixConfig:
    dataset_names: Sequence[str]
    max_samples_per_dataset: int | None = None
    split: str = "train"
    cache_dir: str | None = None
    prefetch_workers: int = 2
    streaming: bool = True
    max_video_frames: int = 8
    image_size: int | None = None


class SequentialMultiDatasetStream:
    """Iterate datasets one after another with bounded, on-demand downloads.

    The active dataset is streamed row-by-row (``streaming=True``) up to
    ``max_samples_per_dataset``. While the active dataset is consumed, the next
    dataset is warmed up in a background thread so downloads can overlap without
    materializing full corpora in local storage.
    """

    def __init__(self, config: SequentialMixConfig) -> None:
        if not config.dataset_names:
            raise ValueError("dataset_names must not be empty")
        self.config = config

    def __iter__(self) -> Iterator[VLMSample]:
        names = list(self.config.dataset_names)
        workers = max(1, self.config.prefetch_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prefetch: Future | None = None
            for index, name in enumerate(names):
                if prefetch is not None:
                    prefetch.result()

                if index + 1 < len(names):
                    next_name = names[index + 1]
                    prefetch = executor.submit(self._warmup, next_name)
                else:
                    prefetch = None

                yield from self._iter_dataset(name)

    def _adapter(self, name: str):
        adapter = build_dataset(name)
        if hasattr(adapter, "max_video_frames"):
            adapter.max_video_frames = self.config.max_video_frames  # type: ignore[attr-defined]
        if hasattr(adapter, "image_size"):
            adapter.image_size = self.config.image_size  # type: ignore[attr-defined]
        return adapter

    def _warmup(self, name: str) -> None:
        logger.info("prefetch warmup dataset={}", name)
        self._adapter(name).warmup(split=self.config.split, cache_dir=self.config.cache_dir)

    def _iter_dataset(self, name: str) -> Iterator[VLMSample]:
        logger.info(
            "sequential dataset start name={} max_samples={}",
            name,
            self.config.max_samples_per_dataset,
        )
        adapter = self._adapter(name)
        yield from adapter.iter_samples(
            split=self.config.split,
            max_samples=self.config.max_samples_per_dataset,
            cache_dir=self.config.cache_dir,
            streaming=self.config.streaming,
        )
        logger.info("sequential dataset complete name={}", name)


def all_builtin_dataset_names() -> list[str]:
    from hale_vlm.data.vlm.catalog import SMOLVLM_ALL_DATASETS

    return list(SMOLVLM_ALL_DATASETS)


__all__ = ["SequentialMixConfig", "SequentialMultiDatasetStream", "all_builtin_dataset_names"]
