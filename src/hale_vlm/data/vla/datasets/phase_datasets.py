"""Phase-tagged VLA dataset registrations (pretrain / mid-train / post-train).

All datasets here are registered in VLA_PHASE_DATASETS (see vla/phase_registry.py),
NOT in the SmolVLA VLA_DATASETS registry, so existing SmolVLA tests are unaffected.
"""

from __future__ import annotations

from hale_vlm.data.vla.adapters.lerobot import VLADataAdapter
from hale_vlm.data.vla.phase_registry import register_vla_phase_dataset
from hale_vlm.data.types import RobotEmbodiment, VLADatasetSpec, VLAStage, VLATrainingPhase

# ---------------------------------------------------------------------------
# Phase 1 — PRETRAIN
# Large-scale open-world robot teleoperation across many embodiments.
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("open-x-embodiment")
class OpenXEmbodimentAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="open-x-embodiment",
        hf_path="jxu124/OpenX-Embodiment",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Open X-Embodiment — 22 robot types, ~2M demonstrations across "
            "kitchen, tabletop, and outdoor environments. "
            "Broadest available robot pretraining corpus."
        ),
        paper_reference="Open X-Embodiment Collaboration (2023)",
        episodes=2_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("bridge-v2")
class BridgeV2Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="bridge-v2",
        hf_path="lerobot/bridge_v2",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Bridge Data V2 — diverse household robot manipulation across "
            "24 environments and 13 task families. ~60K demonstrations."
        ),
        paper_reference="Walke et al. (2023) Bridge Data V2",
        episodes=60_000,
    )


@register_vla_phase_dataset("fractal-rt1")
class FractalRT1Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="fractal-rt1",
        hf_path="google-deepmind/fractal20220817-data",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Google RT-1 training corpus (fractal20220817) — 130K episodes "
            "across 700+ task variants on a mobile manipulation robot."
        ),
        paper_reference="Brohan et al. (2022) RT-1",
        episodes=130_000,
    )


@register_vla_phase_dataset("bc-z")
class BCZAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="bc-z",
        hf_path="lerobot/bc_z",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "BC-Z — 25K robot episodes across 100 tasks on Google's robot, "
            "collected via human demonstration and teleoperation."
        ),
        paper_reference="Jang et al. (2022) BC-Z",
        episodes=25_000,
    )


@register_vla_phase_dataset("droid-v1")
class DroidV1Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="droid-v1",
        hf_path="lerobot/droid_1.0.1",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "DROID 1.0.1 — full corpus, 76K trajectories, 564 scenes, 86 tasks, "
            "50 operators. Collected in-the-wild on Franka Panda across many labs. "
            "High scene diversity; key cross-embodiment pretrain complement to OXE."
        ),
        paper_reference="Khazatsky et al. (2024) DROID",
        episodes=76_000,
    )


@register_vla_phase_dataset("libero-pretrain")
class LiberoPretrainAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-pretrain",
        hf_path="HuggingFaceVLA/libero",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "LIBERO (HuggingFaceVLA edition) — 130+ tasks, 5K+ episodes on Franka. "
            "Officially integrated into LeRobot v0.4.0. Used as pretraining corpus "
            "for instruction-following and multi-task generalisation."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=5_000,
    )


@register_vla_phase_dataset("openEAI-dataset")
class OpenEAIDatasetAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="openEAI-dataset",
        hf_path="OpenEAI/OpenEAI-Dataset",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "OpenEAI-Dataset — ~3.12 TB unified HDF5 corpus aggregating "
            "OXE + UMI Community + DROID + BC-Z. Provides a single-format "
            "pretrain pack for cross-embodiment robot learning."
        ),
        paper_reference="OpenEAI (2024)",
        episodes=3_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("lerobot-community-v3")
class LeRobotCommunityV3Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="lerobot-community-v3",
        hf_path="lerobot/community_dataset_v3",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "LeRobot Community Dataset v3 — 791 datasets across 46 robot types, "
            "consolidated from 851 community-contributed sources. "
            "LeRobot Datasets v3.0 format with chunked episode support for OXE-scale."
        ),
        paper_reference="Cadène et al. (2024) LeRobot",
        episodes=5_000_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("robogene")
class RoboGeneAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="robogene",
        hf_path="X-Humanoid/RoboGene",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "RoboGene — VLA-specific pretraining dataset. Diversity-driven "
            "agentic generation addresses limited scene variety and insufficient "
            "physical grounding in existing corpora. LeRobot-compatible."
        ),
        paper_reference="X-Humanoid (2024) RoboGene",
        episodes=500_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("being-h0")
class BeingH0Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="being-h0",
        hf_path="BeingBeyond/Being-H0",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "Being-H0 post-training dataset — pretrained from large-scale human "
            "videos via explicit hand motion modelling. "
            "Bridges the embodiment gap using 2D/3D cues from egocentric human video."
        ),
        paper_reference="BeingBeyond (2024) Being-H0",
        episodes=100_000,
        trust_remote_code=True,
    )


@register_vla_phase_dataset("agibot-world")
class AgibotWorldAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="agibot-world",
        hf_path="lerobot/xvla-agibot-world",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "AgiBot World — bimanual manipulation dataset used in X-VLA pretraining. "
            "High-quality dexterous manipulation across a variety of real-world tasks."
        ),
        paper_reference="AgiBot (2024) AgiBot World",
        episodes=200_000,
    )


@register_vla_phase_dataset("h-tac-ttp")
class HTacTTPAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="h-tac-ttp",
        hf_path="BeingBeyond/TTP",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.PRETRAIN,
        description=(
            "H-Tac TTP (Tactile Transformer Pretraining) — tactile sensor "
            "pretraining for dexterous manipulation. Captures rich contact "
            "dynamics unavailable in vision-only corpora."
        ),
        paper_reference="BeingBeyond (2024) H-Tac",
        episodes=50_000,
        trust_remote_code=True,
    )


# ---------------------------------------------------------------------------
# Phase 2 — MID_TRAIN
# Domain-specific manipulation / embodiment adaptation.
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("droid-100")
class Droid100Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="droid-100",
        hf_path="lerobot/droid_100",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "DROID 100K-frame Franka arm subset — 76K diverse manipulation "
            "demonstrations across labs and task categories."
        ),
        paper_reference="Khazatsky et al. (2024) DROID",
        episodes=76_000,
    )


@register_vla_phase_dataset("rh20t")
class RH20TAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="rh20t",
        hf_path="lerobot/rh20t",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "RH20T — 110K contact-rich multi-modal robot manipulation demos. "
            "Includes force/torque sensing and rich task diversity."
        ),
        paper_reference="Fang et al. (2023) RH20T",
        episodes=110_000,
    )


@register_vla_phase_dataset("taco-play")
class TACOPlayAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="taco-play",
        hf_path="lerobot/taco_play",
        stage=VLAStage.COMMUNITY,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "TACO-Play — 3.5K multi-view unstructured play episodes on a Franka arm. "
            "Rich contact and proprioception for tactile-rich manipulation."
        ),
        paper_reference="Rosete-Beas et al. (2023) TACO-Play",
        episodes=3_500,
    )


@register_vla_phase_dataset("pusht")
class PushTAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="pusht",
        hf_path="lerobot/pusht",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "Push-T — 2D planar pushing task for behavioural cloning benchmarking. "
            "Reference dataset for testing multi-modal action distributions."
        ),
        paper_reference="Chi et al. (2023) Diffusion Policy",
        episodes=206,
    )


@register_vla_phase_dataset("aloha-sim")
class AlohaSimAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="aloha-sim",
        hf_path="lerobot/aloha_sim_transfer_cube_scripted",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.MID_TRAIN,
        description=(
            "ALOHA simulated bimanual manipulation — cube transfer and "
            "peg insertion tasks scripted for ACT / Diffusion Policy."
        ),
        paper_reference="Zhao et al. (2023) ACT",
        episodes=50,
    )


# ---------------------------------------------------------------------------
# Phase 3 — POST_TRAIN
# Task-specific fine-tuning on target embodiment / benchmark.
# ---------------------------------------------------------------------------


@register_vla_phase_dataset("libero-goal")
class LiberoGoalAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-goal",
        hf_path="physical-intelligence/libero",
        config_name="libero_goal",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LIBERO-Goal — 10 goal-conditioned tasks with language instructions "
            "specifying desired final configurations."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=500,
    )


@register_vla_phase_dataset("libero-spatial")
class LiberoSpatialAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-spatial",
        hf_path="physical-intelligence/libero",
        config_name="libero_spatial",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LIBERO-Spatial — 10 tasks requiring spatial-relationship understanding "
            "('on top of', 'inside', 'to the right of')."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=500,
    )


@register_vla_phase_dataset("libero-object")
class LiberoObjectAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-object",
        hf_path="physical-intelligence/libero",
        config_name="libero_object",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LIBERO-Object — 10 tasks involving manipulation of a diverse "
            "object set with varying shapes, textures, and affordances."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=500,
    )


@register_vla_phase_dataset("libero-100")
class Libero100Adapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="libero-100",
        hf_path="physical-intelligence/libero",
        config_name="libero_100",
        stage=VLAStage.SIMULATION,
        embodiment=RobotEmbodiment.PANDA,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "LIBERO-100 — full 100-task benchmark combining spatial, goal, "
            "object and long-horizon task suites."
        ),
        paper_reference="Liu et al. (2023) LIBERO",
        episodes=5_000,
    )


@register_vla_phase_dataset("aloha-bimanual")
class AlohaBimanualAdapter(VLADataAdapter):
    spec = VLADatasetSpec(
        name="aloha-bimanual",
        hf_path="lerobot/aloha_mobile_shrimp",
        stage=VLAStage.REAL_WORLD,
        embodiment=RobotEmbodiment.MIXED,
        phase=VLATrainingPhase.POST_TRAIN,
        description=(
            "ALOHA real bimanual manipulation — shrimp cooking task "
            "requiring dexterous bimanual coordination."
        ),
        paper_reference="Zhao et al. (2023) ACT",
        episodes=50,
    )


__all__ = [
    # pretrain
    "AgibotWorldAdapter",
    "BCZAdapter",
    "BeingH0Adapter",
    "BridgeV2Adapter",
    "DroidV1Adapter",
    "FractalRT1Adapter",
    "HTacTTPAdapter",
    "LeRobotCommunityV3Adapter",
    "LiberoPretrainAdapter",
    "OpenEAIDatasetAdapter",
    "OpenXEmbodimentAdapter",
    "RoboGeneAdapter",
    # mid-train
    "AlohaSimAdapter",
    "Droid100Adapter",
    "PushTAdapter",
    "RH20TAdapter",
    "TACOPlayAdapter",
    # post-train
    "AlohaBimanualAdapter",
    "Libero100Adapter",
    "LiberoGoalAdapter",
    "LiberoObjectAdapter",
    "LiberoSpatialAdapter",
]
