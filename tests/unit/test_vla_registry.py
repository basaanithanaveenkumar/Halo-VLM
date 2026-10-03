"""VLA dataset registry tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hale_vlm.data.vla.bridge import format_robotics_instruction, vla_sample_to_vlm
from hale_vlm.data.types import Modality, RobotEmbodiment, RoboticsVLMMode, VLASample, VLAStage
from hale_vlm.data.vla.community_paths import SMOLVLA_COMMUNITY_HF_PATHS, hf_path_to_registry_name
from hale_vlm.data.vla.catalog import (
    COMMUNITY_STATS,
    SMOLVLA_ALL_DATASETS,
    SMOLVLA_COMMUNITY_DATASETS,
    SMOLVLA_REAL_WORLD_DATASETS,
    SMOLVLA_SIMULATION_DATASETS,
)
from hale_vlm.data.vla.registry import build_vla_dataset, list_vla_datasets


@pytest.mark.smoke
def test_vla_registry_lists_smolvla_datasets():
    import hale_vlm.data.vla.datasets.builtin  # noqa: F401

    names = list_vla_datasets(enabled_only=True)
    assert len(names) == len(SMOLVLA_ALL_DATASETS)
    assert "libero" in names
    assert "svla-so100-pickplace" in names
    assert hf_path_to_registry_name("satvikahuja/mixer_on_off_new_1") in names


def test_community_paths_cover_appendix():
    assert len(SMOLVLA_COMMUNITY_HF_PATHS) >= COMMUNITY_STATS["paper_reported_datasets"]
    assert len(SMOLVLA_COMMUNITY_DATASETS) == len(SMOLVLA_COMMUNITY_HF_PATHS)
    assert "aergogo/so100_pick_place" in SMOLVLA_COMMUNITY_HF_PATHS
    assert "liuhuanjim013/so100_th" in SMOLVLA_COMMUNITY_HF_PATHS


def test_vla_stage_presets():
    assert len(SMOLVLA_SIMULATION_DATASETS) == 2
    assert len(SMOLVLA_REAL_WORLD_DATASETS) == 4
    assert len(SMOLVLA_ALL_DATASETS) == len(SMOLVLA_COMMUNITY_DATASETS) + 6


def test_build_vla_simulation_adapter():
    import hale_vlm.data.vla.datasets.builtin  # noqa: F401

    adapter = build_vla_dataset("libero")
    assert adapter.spec.stage == VLAStage.SIMULATION
    assert adapter.spec.hf_path == "physical-intelligence/libero"
    assert adapter.spec.embodiment == RobotEmbodiment.PANDA


def test_robotics_to_vlm_conversion():
    sample = VLASample(
        dataset="svla-so100-pickplace",
        stage=VLAStage.REAL_WORLD,
        task="Pick up the cube and place it in the box",
        images=[],
        embodiment=RobotEmbodiment.SO100,
    )
    vlm = vla_sample_to_vlm(sample, RoboticsVLMMode.PRETRAINING)
    assert vlm.modality == Modality.IMAGE
    assert "Robot task:" in vlm.text
    assert vlm.category == "robotics:real_world"


def test_robotics_instruction_modes():
    task = "Pick up the cube"
    assert "Robot task" in format_robotics_instruction(task, RoboticsVLMMode.PRETRAINING)
    assert "manipulation" in format_robotics_instruction(task, RoboticsVLMMode.FINETUNING)
    assert "Instruction" in format_robotics_instruction(task, RoboticsVLMMode.INSTRUCTION_TUNING)


class _FakeVLAAdapter:
    def warmup(self, *, split=None, cache_dir=None) -> None:
        del split, cache_dir

    def iter_samples(self, **kwargs):
        del kwargs
        yield VLASample(
            dataset="fake-vla",
            stage=VLAStage.REAL_WORLD,
            task="pick cube",
            images=[],
        )


def test_sequential_vla_stream():
    from hale_vlm.data.vla.sequential import SequentialVLAStream, VLAStreamConfig

    adapters = {"a": _FakeVLAAdapter(), "b": _FakeVLAAdapter()}

    def _build(name: str, **_kwargs):
        return adapters[name]

    stream = SequentialVLAStream(
        VLAStreamConfig(dataset_names=["a", "b"], max_samples_per_dataset=1, prefetch_workers=1)
    )
    with patch("hale_vlm.data.vla.sequential.build_vla_dataset", side_effect=_build):
        samples = list(stream)
    assert len(samples) == 2
    assert samples[0].dataset == "fake-vla"
