"""Q-Former: learned queries that compress vision tokens into a fixed set of LLM tokens.

BLIP-2 style (Li et al., 2023), vision-only: ``num_queries`` learned query vectors run through
transformer layers. Every layer lets the queries attend to each other (self-attention); every
``cross_attention_freq``-th layer also lets them attend to the frozen vision features
(cross-attention). The result is projected to the LLM hidden size.

Input  ``(B, N, vision_dim)``  ->  output ``(B, num_queries, llm_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormerLayer(nn.Module):
    """Pre-norm block: query self-attention, optional cross-attention to vision, FFN."""

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        cross_attention: bool,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim) if cross_attention else None
        self.cross_attn = (
            nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, kdim=kv_dim, vdim=kv_dim, batch_first=True
            )
            if cross_attention
            else None
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    @property
    def has_cross_attention(self) -> bool:
        return self.cross_attn is not None

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.self_norm(queries)
        queries = queries + self.self_attn(h, h, h, need_weights=False)[0]
        if self.cross_attn is not None:
            h = self.cross_norm(queries)
            queries = queries + self.cross_attn(h, memory, memory, need_weights=False)[0]
        return queries + self.ffn(self.ffn_norm(queries))


class QFormer(nn.Module):
    """Learned-query transformer that maps vision features to ``num_queries`` LLM tokens.

    Args:
        vision_dim: channel size of the vision encoder tokens.
        llm_dim: hidden size of the language model the output is fed to.
        num_queries: number of output tokens (the number of image tokens the LLM sees).
        hidden_dim: internal width; ``None`` uses ``min(llm_dim, 768)``.
        num_layers: transformer layers.
        num_heads: attention heads (must divide ``hidden_dim``).
        cross_attention_freq: layer ``i`` gets cross-attention when ``i % freq == 0``
            (BLIP-2 uses 2; 1 means every layer).
        ffn_mult: feed-forward expansion factor.
        dropout: dropout inside attention and the FFN.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        *,
        num_queries: int = 32,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 8,
        cross_attention_freq: int = 1,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or min(llm_dim, 768)
        if hidden_dim % num_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if cross_attention_freq < 1:
            raise ValueError("cross_attention_freq must be >= 1")
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(vision_dim)
        self.layers = nn.ModuleList(
            QFormerLayer(
                hidden_dim,
                vision_dim,
                num_heads,
                ffn_mult,
                dropout,
                cross_attention=(i % cross_attention_freq == 0),
            )
            for i in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, llm_dim)

    def output_tokens(self, num_input_tokens: int | None = None) -> int:
        """Number of tokens produced per image (independent of the input length)."""
        del num_input_tokens
        return self.num_queries

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(vision_features)
        queries = self.query_tokens.expand(vision_features.shape[0], -1, -1)
        queries = queries.to(memory.dtype)
        for layer in self.layers:
            queries = layer(queries, memory)
        return self.out_proj(self.out_norm(queries))
