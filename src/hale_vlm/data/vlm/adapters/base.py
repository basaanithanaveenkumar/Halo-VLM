"""Base adapter skeleton for VLM datasets."""

from __future__ import annotations

import json
from abc import ABC
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from loguru import logger
from PIL import Image

from hale_vlm.data.types import DatasetSpec, Modality, VLMSample


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        from io import BytesIO

        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    return None


def _conversation_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for turn in value:
            if isinstance(turn, dict):
                role = turn.get("from") or turn.get("role") or "user"
                content = turn.get("value") or turn.get("content") or ""
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(value)


class VLMDataAdapter(ABC):
    """Skeleton adapter: stream rows from HuggingFace and normalize to VLMSample."""

    spec: DatasetSpec

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
            "opening dataset={} hf_path={} split={} streaming={} max_samples={}",
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

    def normalize(self, row: dict[str, Any]) -> VLMSample | None:
        text = self._extract_text(row)
        if not text:
            return None

        images = self._extract_images(row)
        video_frames = self._extract_video_frames(row)

        modality = self.spec.modality
        if modality == Modality.TEXT:
            return VLMSample(
                dataset=self.spec.name,
                modality=modality,
                text=text,
                stage=self.spec.stage,
                category=self.spec.category,
                metadata={"raw": row},
            )
        if modality == Modality.VIDEO and not video_frames:
            return None
        if modality in {Modality.IMAGE, Modality.MULTI_IMAGE} and not images:
            return None

        return VLMSample(
            dataset=self.spec.name,
            modality=modality,
            text=text,
            stage=self.spec.stage,
            category=self.spec.category,
            images=images,
            video_frames=video_frames,
            metadata={"raw_keys": sorted(row.keys())},
        )

    def iter_samples(
        self,
        *,
        split: str | None = None,
        max_samples: int | None = None,
        cache_dir: str | None = None,
        streaming: bool | None = None,
    ) -> Iterator[VLMSample]:
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

    def _extract_text(self, row: dict[str, Any]) -> str | None:
        if self.spec.conversation_field and self.spec.conversation_field in row:
            text = _conversation_to_text(row[self.spec.conversation_field])
            if text:
                return text

        value = _first_present(row, self.spec.text_fields)
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return _conversation_to_text(value)
        return str(value)

    def _extract_images(self, row: dict[str, Any]) -> list[Image.Image]:
        images: list[Image.Image] = []
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

    def _extract_video_frames(self, row: dict[str, Any]) -> list[Image.Image]:
        return []

    def warmup(self, *, split: str | None = None, cache_dir: str | None = None) -> None:
        """Touch the stream so metadata/shards begin downloading without buffering all rows."""
        iterator = self.open_stream(split=split, max_samples=1, cache_dir=cache_dir, streaming=True)
        try:
            next(iterator)
        except StopIteration:
            pass


def dumps_row_preview(row: dict[str, Any], limit: int = 240) -> str:
    preview = {k: type(v).__name__ for k, v in row.items()}
    text = json.dumps(preview, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."
