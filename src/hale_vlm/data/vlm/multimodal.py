"""Multimodal dataset and dataloader utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.data.vlm.stream import iter_registry_vlm_samples
from hale_vlm.data.vlm.scratch_batch import scratch_collate
from hale_vlm.data.vlm.scratch_encoding import (
    encode_scratch_image,
    encode_scratch_text,
    scratch_image_transform,
)
from hale_vlm.data.types import Modality, VLMSample
from hale_vlm.language.hale.backbone import resolve_llm_config


@dataclass
class MultimodalSample:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    modality: str
    dataset: str


FusionMode = Literal["token_replace", "prefix_concat"]


def _hale_collate(samples: list[MultimodalSample]) -> dict[str, torch.Tensor | list[str]]:
    pixel_values = torch.stack(
        [s.pixel_values if s.pixel_values.ndim == 3 else s.pixel_values[0] for s in samples]
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "modality": [s.modality for s in samples],
        "dataset": [s.dataset for s in samples],
    }


class MultimodalDataset(Dataset):
    """Image + caption pairs with Hale or scratch prompt templating."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer,
        image_token: str,
        image_size: int,
        max_length: int,
        fusion_mode: FusionMode = "token_replace",
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.fusion_mode = fusion_mode
        if fusion_mode == "prefix_concat":
            self.transform = scratch_image_transform(image_size)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MultimodalSample:
        row = self.records[idx]
        image = row["image"]
        caption = row["text"]

        if self.fusion_mode == "prefix_concat":
            pixel_values = encode_scratch_image(image, self.transform)
            input_ids, attention_mask, labels = encode_scratch_text(
                self.tokenizer,
                caption,
                max_length=self.max_length,
            )
        else:
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            pixel_values = self.transform(image)
            prompt = f"User: {self.image_token}\nDescribe the image.\nAssistant: {caption}"
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        return MultimodalSample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            modality="image",
            dataset=row.get("dataset", "local"),
        )


class EncodedRegistryDataset(Dataset):
    """Materialize a bounded slice from the sequential registry stream for map-style training."""

    def __init__(
        self,
        cfg: VLMRunConfig,
        tokenizer,
        *,
        fusion_mode: FusionMode,
    ) -> None:
        self.samples = list(RegistryStreamingDataset(cfg, tokenizer, fusion_mode=fusion_mode))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]


class RegistryStreamingDataset(IterableDataset):
    """Stream all registered datasets sequentially with bounded sample counts."""

    def __init__(
        self,
        cfg: VLMRunConfig,
        tokenizer,
        *,
        fusion_mode: FusionMode = "token_replace",
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.image_token = cfg.model.llm.image_token
        self.max_length = cfg.model.max_length
        self.image_size = cfg.model.vision.image_size
        if fusion_mode == "prefix_concat":
            self.transform = scratch_image_transform(self.image_size)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __iter__(self) -> Iterator[MultimodalSample]:
        data_cfg = self.cfg.data
        for sample in iter_registry_vlm_samples(
            data_cfg,
            image_size=self.image_size,
        ):
            yield self._encode(sample)

    def _encode(self, sample: VLMSample) -> MultimodalSample:
        pixel_values = self._visual_tensor(sample)
        if self.fusion_mode == "prefix_concat":
            if sample.modality == Modality.TEXT:
                text = sample.text
            else:
                text = sample.text or "Describe the image."
            input_ids, attention_mask, labels = encode_scratch_text(
                self.tokenizer,
                text,
                max_length=self.max_length,
            )
        else:
            prompt = self._build_prompt(sample)
            encoded = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        return MultimodalSample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            modality=sample.modality.value,
            dataset=sample.dataset,
        )

    def _build_prompt(self, sample: VLMSample) -> str:
        if sample.modality == Modality.TEXT:
            return sample.text
        if sample.modality == Modality.VIDEO:
            return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"
        if sample.modality == Modality.MULTI_IMAGE:
            tokens = " ".join(self.image_token for _ in sample.images) or self.image_token
            return f"User: {tokens}\n{sample.text}\nAssistant: {sample.text}"
        return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"

    def _visual_tensor(self, sample: VLMSample) -> torch.Tensor:
        frames = sample.visual_frames
        if not frames:
            return torch.zeros(3, self.image_size, self.image_size)
        if self.fusion_mode == "prefix_concat":
            return encode_scratch_image(frames[0], self.transform)
        tensors = [self.transform(frame) for frame in frames]
        if sample.modality == Modality.VIDEO and len(tensors) > 1:
            return torch.stack(tensors)
        return tensors[0]


class MultimodalDataModule:
    """HaleBlocks-compatible data module for Hale and scratch VLM training."""

    def __init__(self, cfg: VLMRunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.fusion_mode = self._resolve_fusion_mode()
        if tokenizer is None:
            self.tokenizer = self._build_tokenizer()
        else:
            self.tokenizer = tokenizer
        self._train: Dataset | IterableDataset | None = None
        self._val: Dataset | None = None
        self._coco_loaders: dict[str, DataLoader] | None = None

    def _resolve_fusion_mode(self) -> FusionMode:
        architecture = self.cfg.model.resolved_architecture(self.cfg.variant)
        return "prefix_concat" if architecture in {"basic", "halo_moe"} else "token_replace"

    def _build_tokenizer(self):
        if self.fusion_mode == "prefix_concat":
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.scratch.tokenizer_id)
        else:
            llm_cfg = resolve_llm_config(self.cfg.model.llm)
            tokenizer = AutoTokenizer.from_pretrained(
                llm_cfg.model_id,
                trust_remote_code=llm_cfg.trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _collate_fn(self):
        return scratch_collate if self.fusion_mode == "prefix_concat" else _hale_collate

    def _load_split(self, split: str) -> list[dict[str, Any]]:
        data_cfg = self.cfg.data
        if data_cfg.source == "overfit":
            text = data_cfg.overfit_text or "A red square on a white background."
            size = self.cfg.model.vision.image_size
            image = Image.new("RGB", (size, size), "white")
            copies = data_cfg.n_overfit_copies if split == data_cfg.train_split else 4
            return [
                {"image": image.copy(), "text": text, "dataset": "overfit"} for _ in range(copies)
            ]

        ds = load_dataset(
            data_cfg.dataset,
            data_cfg.subset,
            split=split,
            cache_dir=data_cfg.cache_dir,
        )
        records = []
        for row in ds:
            image = row.get("image")
            text = row.get(data_cfg.text_field) or row.get("caption") or row.get("text")
            if image is None or text is None:
                continue
            records.append({"image": image, "text": text, "dataset": data_cfg.dataset})
            limit = data_cfg.train_size if split == data_cfg.train_split else data_cfg.val_size
            if limit is not None and len(records) >= limit:
                break
        return records

    def _build_coco_loaders(self) -> dict[str, DataLoader]:
        if self._coco_loaders is not None:
            return self._coco_loaders

        from lavis.datasets.builders import load_dataset

        from hale_vlm.data.vlm.coco_lavis import create_vlm_dataloaders

        import os

        os.environ.setdefault("cache_root", str(Path.home() / ".cache" / "lavis" / "coco"))
        coco_dataset = load_dataset("coco_caption")
        self._coco_loaders = create_vlm_dataloaders(
            coco_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=2,
            tokenizer=self.tokenizer,
            max_length=self.cfg.model.scratch.coco_max_length,
        )
        return self._coco_loaders

    def train_loader(self) -> DataLoader:
        if self.cfg.data.source == "coco_lavis":
            return self._build_coco_loaders()["train"]

        if self._train is None:
            if self.cfg.data.source in {"registry", "vla_registry", "mixed_registry"}:
                self._train = EncodedRegistryDataset(
                    self.cfg,
                    self.tokenizer,
                    fusion_mode=self.fusion_mode,
                )
            else:
                records = self._load_split(self.cfg.data.train_split)
                self._train = MultimodalDataset(
                    records,
                    tokenizer=self.tokenizer,
                    image_token=self.cfg.model.llm.image_token,
                    image_size=self.cfg.model.vision.image_size,
                    max_length=self.cfg.model.max_length,
                    fusion_mode=self.fusion_mode,
                )
        return DataLoader(
            self._train,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.data.source not in {"registry", "vla_registry", "mixed_registry"},
            collate_fn=self._collate_fn(),
        )

    def val_loader(self, train_loader: DataLoader | None = None) -> DataLoader:
        del train_loader
        if self.cfg.data.source == "coco_lavis":
            return self._build_coco_loaders()["val"]

        if self._val is None:
            records = self._load_split(self.cfg.data.val_split)
            self._val = MultimodalDataset(
                records,
                tokenizer=self.tokenizer,
                image_token=self.cfg.model.llm.image_token,
                image_size=self.cfg.model.vision.image_size,
                max_length=self.cfg.model.max_length,
                fusion_mode=self.fusion_mode,
            )
        return DataLoader(
            self._val,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn(),
        )

    def viz_prompt(self, step: int) -> str:
        del step
        if self.fusion_mode == "prefix_concat":
            return "Describe the image."
        return f"User: {self.cfg.model.llm.image_token}\nDescribe the image.\nAssistant:"
