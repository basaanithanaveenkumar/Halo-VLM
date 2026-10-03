"""LeRobot / robotics dataset adapters for SmolVLA registry."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from loguru import logger
from PIL import Image

from hale_vlm.data.vlm.adapters.base import _as_image, _first_present
from hale_vlm.data.types import VLADatasetSpec, VLASample


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return None


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            _flatten(path, nested, out)
    else:
        out[prefix] = value


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    _flatten("", row, flat)
    return flat


class VLADataAdapter(ABC):
    """Stream LeRobot-style rows and normalize to VLASample."""

    spec: VLADatasetSpec

    def open_stream(
        self,
        *,
        split: str | None = None,
        max_samples: int | None = None,
        cache_dir: str | None = None,
        streaming: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        split = split or self.spec.default_split
        use_streaming = self.spec.streaming if streaming is None else streaming
        logger.info(
            "opening vla dataset={} hf_path={} split={} streaming={} max_samples={}",
            self.spec.name,
            self.spec.hf_path,
            split,
            use_streaming,
            max_samples,
        )
        kwargs: dict[str, Any] = {
            "path": self.spec.hf_path,
            "split": split,
            "cache_dir": cache_dir,
            "streaming": use_streaming,
            "trust_remote_code": self.spec.trust_remote_code,
        }
        if self.spec.config_name:
            kwargs["name"] = self.spec.config_name

        dataset = load_dataset(**kwargs)
        if use_streaming and max_samples is not None:
            dataset = dataset.take(max_samples)
        elif not use_streaming and max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        yield from dataset

    def normalize(self, row: dict[str, Any]) -> VLASample | None:
        flat = _flatten_row(row)
        task = self._extract_task(flat)
        if not task:
            return None

        images = self._extract_images(flat)
        if not images:
            return None

        return VLASample(
            dataset=self.spec.name,
            stage=self.spec.stage,
            task=task,
            images=images,
            action=self._extract_action(flat),
            state=self._extract_state(flat),
            embodiment=self.spec.embodiment,
            metadata={"raw_keys": sorted(flat.keys())},
        )

    def iter_samples(
        self,
        *,
        split: str | None = None,
        max_samples: int | None = None,
        cache_dir: str | None = None,
        streaming: bool | None = None,
    ) -> Iterator[VLASample]:
        produced = 0
        for row in self.open_stream(
            split=split,
            max_samples=max_samples,
            cache_dir=cache_dir,
            streaming=streaming,
        ):
            sample = self.normalize(row)
            if sample is None:
                continue
            yield sample
            produced += 1
            if max_samples is not None and produced >= max_samples:
                break

    def _extract_task(self, row: dict[str, Any]) -> str | None:
        value = _first_present(row, self.spec.task_fields)
        if value is None:
            return None
        return str(value).strip() or None

    def _extract_action(self, row: dict[str, Any]) -> list[float] | None:
        return _as_float_list(_first_present(row, self.spec.action_fields))

    def _extract_state(self, row: dict[str, Any]) -> list[float] | None:
        return _as_float_list(_first_present(row, self.spec.state_fields))

    def _extract_images(self, row: dict[str, Any]) -> list[Image.Image]:
        images: list[Image.Image] = []

        prefix = self.spec.camera_prefix
        if prefix:
            for key, value in row.items():
                if key.startswith(f"{prefix}.") or key.startswith("observation.images."):
                    image = _as_image(value)
                    if image is not None:
                        images.append(image)

        if images:
            return images

        for key in self.spec.image_fields:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, list):
                for item in value:
                    image = _as_image(item)
                    if image is not None:
                        images.append(image)
            else:
                image = _as_image(value)
                if image is not None:
                    images.append(image)
        return images

    def warmup(self, *, split: str | None = None, cache_dir: str | None = None) -> None:
        iterator = self.open_stream(split=split, max_samples=1, cache_dir=cache_dir, streaming=True)
        try:
            next(iterator)
        except StopIteration:
            pass
