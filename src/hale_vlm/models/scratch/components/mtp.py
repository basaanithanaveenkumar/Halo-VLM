"""Multi-token prediction (MTP): extra training signal from predicting several future tokens.

Two published variants, selected by ``ScratchConfig.mtp_style``. Both keep the model's normal
next-token head (depth 0) unchanged and add ``D = mtp_num_future_tokens - 1`` extra depths;
depth ``d`` at position ``i`` predicts token ``i + d + 1``. All depths share the main LM head.

* ``parallel`` - Gloeckle et al., *Better & Faster Large Language Models via Multi-token
  Prediction*, ICML 2024, arXiv:2404.19737. Every depth is an independent causal transformer
  block on top of the shared trunk's hidden states.
* ``sequential`` - DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024, arXiv:2412.19437 (§2.2).
  Depth ``d`` keeps the causal chain: it combines the previous depth's state with the
  embedding of the token it has just "seen", ``h_i^d = TRM_d(M_d [RMSNorm(h_i^{d-1});
  RMSNorm(Emb(x_{i+d}))])``, then applies the shared head.

MTP is a training-time objective: inference uses the main head only (the extra depths can be
dropped, or reused for speculative decoding).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_vlm.models.scratch.components.transformer import TransformerBlock

MTPStyle = Literal["parallel", "sequential"]


class MultiTokenPredictionHeads(nn.Module):
    """The extra prediction depths of an MTP model.

    Args:
        emb_dim: hidden width of the trunk.
        num_future_tokens: tokens predicted per position including the next one (``>= 2``).
        style: ``parallel`` (Gloeckle et al.) or ``sequential`` (DeepSeek-V3).
        num_heads: attention heads of each depth's transformer block.
        mlp_dim: MLP width of each depth's transformer block.
    """

    def __init__(
        self,
        emb_dim: int,
        num_future_tokens: int,
        style: MTPStyle = "parallel",
        num_heads: int = 8,
        mlp_dim: int = 1024,
    ) -> None:
        super().__init__()
        if num_future_tokens < 2:
            raise ValueError(f"num_future_tokens must be >= 2 for MTP, got {num_future_tokens}")
        if style not in ("parallel", "sequential"):
            raise ValueError(f"mtp style must be 'parallel' or 'sequential', got {style!r}")
        self.style = style
        self.depths = num_future_tokens - 1
        self.blocks = nn.ModuleList(
            TransformerBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, causal_mask=True)
            for _ in range(self.depths)
        )
        self.norm = nn.LayerNorm(emb_dim)
        if style == "sequential":
            # M_d: projection of [RMSNorm(h); RMSNorm(Emb(x_{i+d}))] back to the model width.
            self.hidden_norms = nn.ModuleList(nn.RMSNorm(emb_dim) for _ in range(self.depths))
            self.embed_norms = nn.ModuleList(nn.RMSNorm(emb_dim) for _ in range(self.depths))
            self.merges = nn.ModuleList(nn.Linear(2 * emb_dim, emb_dim) for _ in range(self.depths))

    def forward(
        self,
        hidden: torch.Tensor,
        input_embeds: torch.Tensor,
        lm_head: Callable[[torch.Tensor], torch.Tensor],
    ) -> list[torch.Tensor]:
        """Logits of every extra depth.

        Args:
            hidden: ``(B, S, E)`` final trunk states (before the main LM head).
            input_embeds: ``(B, S, E)`` the sequence the trunk consumed, without positions
                (image tokens + token embeddings); ``sequential`` reads future entries of it.
            lm_head: the model's shared output head.

        Returns:
            ``D`` tensors; depth ``d`` (1-based) has shape ``(B, S - d, V)``: position ``i``
            predicts token ``i + d + 1``.
        """
        logits: list[torch.Tensor] = []
        state = hidden
        for d in range(1, self.depths + 1):
            idx = d - 1
            if self.style == "parallel":
                out = self.blocks[idx](hidden)[:, : hidden.size(1) - d]
            else:
                # The previous depth already dropped its last positions; align the embeddings of
                # the tokens x_{i+d} with the remaining states.
                prev = state[:, : hidden.size(1) - d]
                future = input_embeds[:, d:]
                merged = self.merges[idx](
                    torch.cat([self.hidden_norms[idx](prev), self.embed_norms[idx](future)], dim=-1)
                )
                out = self.blocks[idx](merged)
                state = out
            logits.append(lm_head(self.norm(out)))
        return logits


def shift_targets(targets: torch.Tensor, depth: int) -> torch.Tensor:
    """Targets for MTP depth ``depth``.

    ``targets[:, i]`` is the label of the next-token head at position ``i`` (the token at
    ``i + 1``, ``-100`` where ignored); depth ``d`` at position ``i`` predicts the token at
    ``i + d + 1``, i.e. ``targets[:, i + d]``.
    """
    return targets[:, depth:]


def mtp_loss(
    mtp_logits: list[torch.Tensor],
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean cross-entropy over the extra depths (each aligned with :func:`shift_targets`)."""
    losses = []
    for d, logits in enumerate(mtp_logits, start=1):
        tgt = shift_targets(targets, d)
        steps = min(logits.size(1), tgt.size(1))
        if steps == 0 or not (tgt[:, :steps] != ignore_index).any():
            continue
        losses.append(
            F.cross_entropy(
                logits[:, :steps].reshape(-1, logits.size(-1)),
                tgt[:, :steps].reshape(-1),
                ignore_index=ignore_index,
            )
        )
    if not losses:
        return mtp_logits[0].new_zeros(()) if mtp_logits else targets.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()
