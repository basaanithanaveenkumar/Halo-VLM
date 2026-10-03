"""Reinforcement-learning fine-tuning configuration for VLM training."""

from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class RLConfig(StrictModel):
    """Select and tune an RL alignment technique (default: supervised only)."""

    technique: Literal[
        "none",
        "dpo",
        "ppo",
        "grpo",
        "reinforce",
        "kto",
        "rloo",
    ] = "none"

    # Shared
    kl_coef: float = 0.02
    reward_scale: float = 1.0

    # DPO / KTO
    beta: float = 0.1
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0

    # PPO
    ppo_clip_range: float = 0.2
    ppo_epochs: int = 1
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # GRPO / RLOO
    group_size: int = 4
    grpo_normalize_advantage: bool = True

    # REINFORCE / RLOO
    baseline_momentum: float = 0.9

    def enabled(self) -> bool:
        return self.technique != "none"
