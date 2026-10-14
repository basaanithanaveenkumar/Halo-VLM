# Multi-token prediction (MTP)

The scratch Halo MoE VLM (`architecture: halo_moe`) trains with either objective:

```yaml
model:
  scratch:
    objective: autoregressive   # default: next-token prediction only
    # objective: mtp            # also predict the following tokens
    mtp_style: parallel         # parallel | sequential
    mtp_num_future_tokens: 4    # tokens predicted per position, including the next one
    mtp_loss_weight: 0.3        # weight of the extra-depth loss
```

With `objective: mtp` the normal next-token head is unchanged and `mtp_num_future_tokens - 1`
extra depths are added (`src/hale_vlm/models/scratch/components/mtp.py`). Depth `d` at position
`i` predicts token `i + d + 1`, and all depths share the main LM head.

| `mtp_style` | paper | how depth `d` is computed |
|---|---|---|
| `parallel` | Gloeckle et al., *Better & Faster LLMs via Multi-token Prediction*, ICML 2024 ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)) | an independent causal transformer block on the shared trunk |
| `sequential` | DeepSeek-V3 Technical Report, §2.2 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) | `TRM_d(M_d[RMSNorm(h^{d-1}_i); RMSNorm(Emb(x_{i+d}))])`, keeping the causal chain |

Training loss = next-token cross-entropy + `mtp_loss_weight` × mean cross-entropy of the extra
depths. Validation reports the next-token loss only, so numbers compare across objectives.
Generation uses the next-token head; the extra depths add no inference cost. `HaloVLM.forward`
returns the next-token logits as before; `forward(..., return_mtp=True)` also returns the
extra-depth logits.

Tests (`tests/unit/test_mtp.py`) check the default autoregressive path is unchanged, depth shapes
and gradients, target alignment, that depth `d` never sees beyond token `i + d` (both styles),
the config switch, and a trainer epoch with each style.

This replaces the work-in-progress MTP code on the `mtp` and `mtp-viz` branches. Those were
written for the pre-restructure layout and did not run; their ideas (parallel heads, DeepSeek-style
modules) are implemented here.
