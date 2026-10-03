"""Registry and sequential streaming tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hale_vlm.data.vlm.catalog import (
    SMOLVLM_ALL_DATASETS,
    SMOLVLM_CONTEXT_DATASETS,
    SMOLVLM_REJECTED_DATASETS,
    SMOLVLM_VIDEO_DATASETS,
    SMOLVLM_VISION_DATASETS,
)
from hale_vlm.data.vlm.registry import build_dataset, list_datasets
from hale_vlm.data.vlm.sequential import SequentialMixConfig, SequentialMultiDatasetStream
from hale_vlm.data.types import Modality, TrainingStage, VLMSample


class _FakeAdapter:
    def __init__(self) -> None:
        self.warmup_calls = 0
        self.max_video_frames = 8
        self.image_size = None

    def warmup(self, *, split=None, cache_dir=None) -> None:
        del split, cache_dir
        self.warmup_calls += 1

    def iter_samples(self, **kwargs):
        del kwargs
        yield VLMSample(
            dataset="fake",
            modality=Modality.IMAGE,
            text="hello",
        )


@pytest.mark.smoke
def test_builtin_registry_lists_smolvlm_datasets():
    import hale_vlm.data.vlm.datasets.builtin  # noqa: F401

    names = list_datasets(enabled_only=True)
    assert len(names) == len(SMOLVLM_ALL_DATASETS)
    assert "mathwriting" in names
    assert "magpie" in names
    assert "dolma-books" in names
    assert "smoltalk" not in names


def test_rejected_smoltalk_is_registered_but_disabled():
    import hale_vlm.data.vlm.datasets.builtin  # noqa: F401

    rejected = list_datasets(stage=TrainingStage.REJECTED, enabled_only=False)
    assert rejected == ["smoltalk"]
    assert "smoltalk" in SMOLVLM_REJECTED_DATASETS

    with pytest.raises(ValueError, match="disabled"):
        build_dataset("smoltalk")


def test_stage_presets_cover_paper_mixtures():
    assert len(SMOLVLM_VISION_DATASETS) == 4
    assert len(SMOLVLM_VIDEO_DATASETS) == 10
    assert len(SMOLVLM_CONTEXT_DATASETS) == 5
    assert len(SMOLVLM_ALL_DATASETS) == 19


def test_sequential_stream_order_and_prefetch():
    adapters = {
        "a": _FakeAdapter(),
        "b": _FakeAdapter(),
        "c": _FakeAdapter(),
    }

    def _build(name: str, **_kwargs):
        return adapters[name]

    stream = SequentialMultiDatasetStream(
        SequentialMixConfig(
            dataset_names=["a", "b", "c"],
            max_samples_per_dataset=1,
            prefetch_workers=2,
        )
    )

    with patch("hale_vlm.data.vlm.sequential.build_dataset", side_effect=_build):
        samples = list(stream)

    assert [s.dataset for s in samples] == ["fake", "fake", "fake"]
    assert adapters["b"].warmup_calls == 1
    assert adapters["c"].warmup_calls == 1


def test_build_dataset_returns_video_adapter():
    import hale_vlm.data.vlm.datasets.builtin  # noqa: F401

    adapter = build_dataset("finevideo")
    assert adapter.spec.modality == Modality.VIDEO
    assert adapter.spec.stage == TrainingStage.VIDEO
    assert adapter.spec.hf_path == "HuggingFaceFV/finevideo"
