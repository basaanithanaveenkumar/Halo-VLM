"""Tests for the phase-aware VLM and VLA dataset registries.

These tests verify that:
1. Phase datasets are in a SEPARATE registry from SmolVLM datasets.
2. Existing SmolVLM tests are not affected (registry isolation).
3. Phase filtering works correctly for both VLM and VLA.
4. All three training phases have registered datasets.
"""

from __future__ import annotations

import pytest

from hale_vlm.data.types import TrainingPhase, VLATrainingPhase
from hale_vlm.data.vlm.catalog import (
    MID_TRAIN_VLM_DATASETS,
    PHASE_NOTES,
    POST_TRAIN_VLM_DATASETS,
    PRETRAIN_VLM_DATASETS,
    VLM_PHASE_PRESETS,
)
from hale_vlm.data.vla.catalog import (
    MID_TRAIN_VLA_DATASETS,
    POST_TRAIN_VLA_DATASETS,
    PRETRAIN_VLA_DATASETS,
    VLA_PHASE_NOTES,
    VLA_PHASE_PRESETS_BY_PHASE,
)


# ---------------------------------------------------------------------------
# VLM phase catalog
# ---------------------------------------------------------------------------


def test_vlm_phase_presets_cover_all_three_phases():
    assert set(VLM_PHASE_PRESETS) == {
        TrainingPhase.PRETRAIN,
        TrainingPhase.MID_TRAIN,
        TrainingPhase.POST_TRAIN,
    }


def test_vlm_pretrain_datasets_non_empty():
    assert len(PRETRAIN_VLM_DATASETS) >= 3
    assert "cc3m" in PRETRAIN_VLM_DATASETS
    assert "cc12m" in PRETRAIN_VLM_DATASETS


def test_vlm_mid_train_datasets_non_empty():
    assert len(MID_TRAIN_VLM_DATASETS) >= 3
    assert "llava-pretrain-558k" in MID_TRAIN_VLM_DATASETS


def test_vlm_post_train_datasets_non_empty():
    assert len(POST_TRAIN_VLM_DATASETS) >= 3
    assert "llava-instruct-665k" in POST_TRAIN_VLM_DATASETS


def test_phase_notes_cover_all_three_phases():
    assert set(PHASE_NOTES) == set(VLM_PHASE_PRESETS)
    for note in PHASE_NOTES.values():
        assert len(note) > 20  # sanity: not empty


# ---------------------------------------------------------------------------
# VLM phase registry
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_phase_registry_lists_pretrain_datasets():
    from hale_vlm.data.vlm.phase_registry import list_phase_datasets

    names = list_phase_datasets(phase=TrainingPhase.PRETRAIN)
    assert len(names) >= len(PRETRAIN_VLM_DATASETS)
    assert "cc3m" in names
    assert "laion-aesthetics-v2-5plus" in names


def test_phase_registry_lists_mid_train_datasets():
    from hale_vlm.data.vlm.phase_registry import list_phase_datasets

    names = list_phase_datasets(phase=TrainingPhase.MID_TRAIN)
    assert len(names) >= len(MID_TRAIN_VLM_DATASETS)
    assert "llava-pretrain-558k" in names


def test_phase_registry_lists_post_train_datasets():
    from hale_vlm.data.vlm.phase_registry import list_phase_datasets

    names = list_phase_datasets(phase=TrainingPhase.POST_TRAIN)
    assert len(names) >= len(POST_TRAIN_VLM_DATASETS)
    assert "llava-instruct-665k" in names


def test_phase_registry_all_returns_all_phases():
    from hale_vlm.data.vlm.phase_registry import list_phase_datasets

    all_names = list_phase_datasets()
    pretrain = list_phase_datasets(phase=TrainingPhase.PRETRAIN)
    mid = list_phase_datasets(phase=TrainingPhase.MID_TRAIN)
    post = list_phase_datasets(phase=TrainingPhase.POST_TRAIN)
    assert set(all_names) == set(pretrain) | set(mid) | set(post)


def test_phase_registry_does_not_pollute_smolvlm_registry():
    """Phase datasets must live in PHASE_DATASETS, not DATASETS."""
    # Import both after phase registry is loaded.
    from hale_vlm.data.vlm.phase_registry import PHASE_DATASETS, list_phase_datasets
    from hale_vlm.data.vlm.registry import DATASETS

    list_phase_datasets()  # trigger registration
    phase_names = set(PHASE_DATASETS.items)
    smolvlm_names = set(DATASETS.items)
    overlap = phase_names & smolvlm_names
    assert len(overlap) == 0, f"Phase datasets leaked into SmolVLM registry: {overlap}"


def test_smolvlm_count_unchanged_after_phase_import():
    """Existing SmolVLM dataset count must stay at 19 after phase datasets load."""
    from hale_vlm.data.vlm.registry import DATASETS, list_datasets
    from hale_vlm.data.vlm.phase_registry import list_phase_datasets
    from hale_vlm.data.vlm.catalog import SMOLVLM_ALL_DATASETS
    import hale_vlm.data.vlm.datasets.builtin  # noqa: F401

    list_phase_datasets()  # trigger phase dataset registration
    smolvlm_names = list_datasets(enabled_only=False)
    assert len(smolvlm_names) == len(SMOLVLM_ALL_DATASETS) + 1  # +1 for smoltalk (rejected)


def test_build_phase_dataset_cc3m():
    from hale_vlm.data.vlm.phase_registry import build_phase_dataset
    from hale_vlm.data.types import TrainingPhase

    adapter = build_phase_dataset("cc3m")
    assert adapter.spec.phase == TrainingPhase.PRETRAIN
    assert adapter.spec.hf_path == "pixparse/cc3m-wds"


def test_build_phase_dataset_chartqa():
    from hale_vlm.data.vlm.phase_registry import build_phase_dataset
    from hale_vlm.data.types import TrainingPhase

    adapter = build_phase_dataset("chartqa")
    assert adapter.spec.phase == TrainingPhase.POST_TRAIN


# ---------------------------------------------------------------------------
# VLA phase catalog
# ---------------------------------------------------------------------------


def test_vla_phase_presets_cover_all_three_phases():
    assert set(VLA_PHASE_PRESETS_BY_PHASE) == {
        VLATrainingPhase.PRETRAIN,
        VLATrainingPhase.MID_TRAIN,
        VLATrainingPhase.POST_TRAIN,
    }


def test_vla_pretrain_datasets_non_empty():
    assert len(PRETRAIN_VLA_DATASETS) >= 3
    assert "open-x-embodiment" in PRETRAIN_VLA_DATASETS
    assert "bridge-v2" in PRETRAIN_VLA_DATASETS


def test_vla_mid_train_datasets_non_empty():
    assert len(MID_TRAIN_VLA_DATASETS) >= 3
    assert "droid-100" in MID_TRAIN_VLA_DATASETS


def test_vla_post_train_datasets_non_empty():
    assert len(POST_TRAIN_VLA_DATASETS) >= 3
    assert "libero-goal" in POST_TRAIN_VLA_DATASETS


# ---------------------------------------------------------------------------
# VLA phase registry
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_vla_phase_registry_lists_pretrain():
    from hale_vlm.data.vla.phase_registry import list_vla_phase_datasets

    names = list_vla_phase_datasets(phase=VLATrainingPhase.PRETRAIN)
    assert len(names) >= len(PRETRAIN_VLA_DATASETS)
    assert "bridge-v2" in names
    assert "fractal-rt1" in names


def test_vla_phase_registry_lists_mid_train():
    from hale_vlm.data.vla.phase_registry import list_vla_phase_datasets

    names = list_vla_phase_datasets(phase=VLATrainingPhase.MID_TRAIN)
    assert len(names) >= len(MID_TRAIN_VLA_DATASETS)
    assert "droid-100" in names


def test_vla_phase_registry_lists_post_train():
    from hale_vlm.data.vla.phase_registry import list_vla_phase_datasets

    names = list_vla_phase_datasets(phase=VLATrainingPhase.POST_TRAIN)
    assert len(names) >= len(POST_TRAIN_VLA_DATASETS)
    assert "libero-goal" in names


def test_vla_phase_registry_does_not_pollute_smolvla_registry():
    """VLA phase datasets must live in VLA_PHASE_DATASETS, not VLA_DATASETS."""
    from hale_vlm.data.vla.phase_registry import VLA_PHASE_DATASETS, list_vla_phase_datasets
    from hale_vlm.data.vla.registry import VLA_DATASETS

    list_vla_phase_datasets()  # trigger registration
    phase_names = set(VLA_PHASE_DATASETS.items)
    smolvla_names = set(VLA_DATASETS.items)
    overlap = phase_names & smolvla_names
    assert len(overlap) == 0, f"VLA phase datasets leaked into SmolVLA registry: {overlap}"


def test_build_vla_phase_dataset_bridge_v2():
    from hale_vlm.data.vla.phase_registry import build_vla_phase_dataset

    adapter = build_vla_phase_dataset("bridge-v2")
    assert adapter.spec.phase == VLATrainingPhase.PRETRAIN
    assert adapter.spec.hf_path == "lerobot/bridge_v2"


def test_build_vla_phase_dataset_libero_goal():
    from hale_vlm.data.vla.phase_registry import build_vla_phase_dataset

    adapter = build_vla_phase_dataset("libero-goal")
    assert adapter.spec.phase == VLATrainingPhase.POST_TRAIN


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def test_vlm_data_config_resolved_phase_pretrain():
    from hale_vlm.config.sections.data import VLMDataConfig

    cfg = VLMDataConfig(source="phase_registry", registry_phase="pretrain")
    names = cfg.resolved_phase_datasets()
    assert len(names) == len(PRETRAIN_VLM_DATASETS)
    assert "cc3m" in names


def test_vlm_data_config_resolved_phase_all():
    from hale_vlm.config.sections.data import VLMDataConfig

    cfg = VLMDataConfig(source="phase_registry", registry_phase="all")
    names = cfg.resolved_phase_datasets()
    total = (
        len(PRETRAIN_VLM_DATASETS)
        + len(MID_TRAIN_VLM_DATASETS)
        + len(POST_TRAIN_VLM_DATASETS)
    )
    assert len(names) == total


def test_vlm_data_config_resolved_vla_phase_mid_train():
    from hale_vlm.config.sections.data import VLMDataConfig

    cfg = VLMDataConfig(vla_registry_phase="mid_train")
    names = cfg.resolved_vla_phase_datasets()
    assert len(names) == len(MID_TRAIN_VLA_DATASETS)
    assert "droid-100" in names
