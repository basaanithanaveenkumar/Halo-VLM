"""Robotics-to-VLM config resolution tests."""

from __future__ import annotations

from pathlib import Path

from hale_vlm.config import load_vlm_config
from hale_vlm.data.types import RoboticsVLMMode
from hale_vlm.data.vla.catalog import SMOLVLA_SIMULATION_DATASETS

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_mixed_registry_config_loads():
    cfg = load_vlm_config(CONFIGS / "vlm_with_robotics_pretrain.yaml")
    assert cfg.data.source == "mixed_registry"
    assert cfg.data.robotics_vlm_mode == RoboticsVLMMode.PRETRAINING
    assert cfg.data.vla_registry_stage == "real_world"


def test_vla_registry_config_loads():
    cfg = load_vlm_config(CONFIGS / "smolvla_simulation.yaml")
    assert cfg.data.source == "vla_registry"
    assert cfg.data.resolved_vla_registry_datasets() == list(SMOLVLA_SIMULATION_DATASETS)
