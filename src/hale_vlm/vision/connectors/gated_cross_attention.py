"""Gated cross-attention connector (Flamingo style, Alayrac et al., 2022).

A set of learned latent tokens reads the vision features through *tanh-gated* cross-attention
and a *tanh-gated* feed-forward block::

    x = x + tanh(alpha_attn) * CrossAttn(LN(x), vision)
    x = x + tanh(alpha_ffn)  * FFN(LN(x))

The gates ``alpha`` are learned scalars. With ``gate_init=0`` the block is the identity at the
start of training (the LLM initially sees only the latents) and the image signal is blended in as
the gates open, which keeps a pretrained LLM undisturbed. A positive ``gate_init`` lets image
information flow from step 0, which converges faster at small learning rates.

Input  ``(B, N, vision_dim)``  ->  output ``(B, num_latents, llm_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossAttentionBlock(nn.Module):
    """One tanh-gated cross-attention + tanh-gated FFN residual block."""

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )
        self.attn_gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.ffn_gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        x = x + torch.tanh(self.attn_gate) * self.attn(h, memory, memory, need_weights=False)[0]
        return x + torch.tanh(self.ffn_gate) * self.ffn(self.ffn_norm(x))


class GatedCrossAttentionProjector(nn.Module):
    """Learned latents + gated cross-attention blocks, projected to the LLM hidden size.

    Args:
        vision_dim: channel size of the vision encoder tokens.
        llm_dim: hidden size of the language model.
        num_latents: number of output tokens.
        hidden_dim: internal width; ``None`` uses ``min(llm_dim, 768)``.
        num_layers: number of gated blocks.
        num_heads: attention heads (must divide ``hidden_dim``).
        ffn_mult: feed-forward expansion factor.
        dropout: dropout inside attention and the FFN.
        gate_init: initial value of every gate before ``tanh`` (0 = Flamingo identity start).
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        *,
        num_latents: int = 64,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or min(llm_dim, 768)
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(vision_dim)
        self.blocks = nn.ModuleList(
            GatedCrossAttentionBlock(
                hidden_dim, vision_dim, num_heads, ffn_mult, dropout, gate_init
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, llm_dim)

    def output_tokens(self, num_input_tokens: int | None = None) -> int:
        """Number of tokens produced per image (independent of the input length)."""
        del num_input_tokens
        return self.num_latents

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(vision_features)
        x = self.latents.expand(vision_features.shape[0], -1, -1).to(memory.dtype)
        for block in self.blocks:
            x = block(x, memory)
        return self.out_proj(self.out_norm(x))
