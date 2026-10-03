"""Registered RL alignment techniques."""

from hale_vlm.rl.techniques.dpo import DPORLTechnique
from hale_vlm.rl.techniques.grpo import GRPORLTechnique
from hale_vlm.rl.techniques.kto import KTORLTechnique
from hale_vlm.rl.techniques.none import NoneRLTechnique
from hale_vlm.rl.techniques.ppo import PPORLTechnique
from hale_vlm.rl.techniques.reinforce import ReinforceRLTechnique
from hale_vlm.rl.techniques.rloo import RLOORLTechnique

__all__ = [
    "DPORLTechnique",
    "GRPORLTechnique",
    "KTORLTechnique",
    "NoneRLTechnique",
    "PPORLTechnique",
    "ReinforceRLTechnique",
    "RLOORLTechnique",
]
