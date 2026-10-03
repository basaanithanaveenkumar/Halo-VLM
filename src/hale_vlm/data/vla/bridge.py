"""Convert robotics (VLA) samples into VLM training samples."""

from __future__ import annotations

from collections.abc import Iterator

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.data.types import Modality, RoboticsVLMMode, VLASample, VLMSample
from hale_vlm.data.vla.sequential import SequentialVLAStream, VLAStreamConfig


def format_robotics_instruction(task: str, mode: RoboticsVLMMode) -> str:
    task = task.strip()
    if mode == RoboticsVLMMode.PRETRAINING:
        return f"Robot task: {task}"
    if mode == RoboticsVLMMode.FINETUNING:
        return f"Execute the following manipulation task: {task}"
    if mode == RoboticsVLMMode.INSTRUCTION_TUNING:
        return f"Instruction: {task}\nDescribe the robot action needed to complete this task."
    return task


def vla_sample_to_vlm(sample: VLASample, mode: RoboticsVLMMode) -> VLMSample:
    """Map a robotics frame into a vision-language sample for VLM training."""
    text = format_robotics_instruction(sample.task, mode)
    modality = Modality.MULTI_IMAGE if len(sample.images) > 1 else Modality.IMAGE
    return VLMSample(
        dataset=sample.dataset,
        modality=modality,
        text=text,
        stage=None,
        category=f"robotics:{sample.stage.value}",
        images=list(sample.images),
        metadata={
            "source": "vla_registry",
            "robotics_mode": mode.value,
            "embodiment": sample.embodiment.value,
            "has_action": sample.action is not None,
            "has_state": sample.state is not None,
        },
    )


def iter_vla_as_vlm_samples(
    data_cfg: VLMDataConfig,
    *,
    mode: RoboticsVLMMode,
) -> Iterator[VLMSample]:
    """Stream VLA registry datasets converted to VLM training samples."""
    vla_stream = SequentialVLAStream(
        VLAStreamConfig(
            dataset_names=data_cfg.resolved_vla_registry_datasets(),
            max_samples_per_dataset=data_cfg.max_samples_per_dataset,
            split=data_cfg.train_split,
            cache_dir=data_cfg.cache_dir,
            prefetch_workers=data_cfg.prefetch_workers,
            streaming=data_cfg.streaming,
        )
    )
    for sample in vla_stream:
        yield vla_sample_to_vlm(sample, mode)
