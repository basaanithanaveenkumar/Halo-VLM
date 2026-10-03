"""Reinforcement-learning fine-tuning for VLM models.

Configure via ``rl.technique`` in the run YAML (default ``none`` = supervised).
"""

from hale_vlm.rl.base import RLStepResult, RLTechnique
from hale_vlm.rl.factory import (
    RL_TECHNIQUE_NAMES,
    build_rl_technique,
    get_rl_technique,
    register_rl_technique,
)

__all__ = [
    "RLStepResult",
    "RLTechnique",
    "RL_TECHNIQUE_NAMES",
    "build_rl_technique",
    "get_rl_technique",
    "register_rl_technique",
]
