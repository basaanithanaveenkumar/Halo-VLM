"""Scratch batch collation helpers."""

from __future__ import annotations

import torch


def scratch_collate(samples: list) -> dict[str, torch.Tensor | list[str]]:
    images = torch.stack(
        [s.pixel_values if s.pixel_values.ndim == 3 else s.pixel_values[0] for s in samples]
    )
    return {
        "images": images,
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "modality": [s.modality for s in samples],
        "dataset": [s.dataset for s in samples],
    }
