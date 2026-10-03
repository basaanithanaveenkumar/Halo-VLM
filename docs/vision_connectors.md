# Vision connectors: Q-Former and gated cross-attention

The vision-to-LLM projector is selected in the config with `model.vision.projector_type`:

| `projector_type` | Output tokens per image | What it is |
|---|---|---|
| `mlp` (default) | one per vision patch | 2-layer MLP (Hale) / 3-layer MLP (scratch) |
| `linear` | one per vision patch | single linear layer (Hale only) |
| `qformer` | `qformer.num_queries` | BLIP-2 style Q-Former |
| `gated_cross_attention` | `gated_cross_attention.num_latents` | Flamingo style tanh-gated cross-attention |

`qformer` and `gated_cross_attention` compress `N` patch features `(B, N, vision_dim)` into a fixed number of
LLM tokens `(B, K, llm_dim)`, so the LLM sequence gets shorter (SigLIP-base: 196 patches -> e.g. 32 tokens).
`VisionConfig.resolved_num_image_tokens()` returns `K`; `model.num_image_tokens` follows it.

## Q-Former (`projector_type: qformer`)

`num_queries` learned query vectors pass through `num_layers` pre-norm transformer layers. Each layer
self-attends over the queries; layer `i` also cross-attends to the (LayerNormed) vision features when
`i % cross_attention_freq == 0` (`1` = every layer, `2` = BLIP-2 style). A final LayerNorm and linear map the
queries to the LLM hidden size.

```yaml
model:
  vision:
    projector_type: qformer
    qformer:
      num_queries: 32
      hidden_dim: 768          # null -> min(llm_dim, 768); must be divisible by num_heads
      num_layers: 2
      num_heads: 12
      cross_attention_freq: 1
      ffn_mult: 4
      dropout: 0.0
```

## Gated cross-attention (`projector_type: gated_cross_attention`)

`num_latents` learned latents read the vision features through tanh-gated cross-attention and a tanh-gated FFN:

```text
x = x + tanh(alpha_attn) * CrossAttn(LN(x), vision)
x = x + tanh(alpha_ffn)  * FFN(LN(x))
```

The gates are learned scalars. `gate_init: 0.0` reproduces the Flamingo identity start (the LLM sees only the
latents until the gates open, which protects a pretrained LLM but trains slowly at small learning rates such as
`2e-5`). The default `gate_init: 1.0` lets image information flow from step 0.

```yaml
model:
  vision:
    projector_type: gated_cross_attention
    gated_cross_attention:
      num_latents: 64
      hidden_dim: 768
      num_layers: 2
      num_heads: 12
      ffn_mult: 4
      dropout: 0.0
      gate_init: 1.0
```

## Where it applies

* `HaleVLM` (SigLIP/CLIP + Qwen/DeepSeek): replaces `VisionProjector`.
* Scratch `HaloVLM` (custom ViT + MoE decoder): replaces `ImageProjector`; `num_image_tokens` becomes `K`.
* Scratch `BasicVLM` (one global OpenCLIP token): works, but the connector then expands a single token into
  `K` tokens, so it is rarely useful there.
* `gwm_vla` has no projector and is unaffected.

Ready-made configs: `configs/qwen3_8b_qformer.yaml`, `configs/qwen3_8b_gated_cross_attention.yaml`,
`configs/halo_moe_qformer_overfit.yaml`, `configs/halo_moe_gated_cross_attention_overfit.yaml`.

```bash
uv run hale-vlm-train configs/halo_moe_qformer_overfit.yaml
```

## Notes

* Existing checkpoints and configs are unaffected: the default is still `mlp`. Changing `projector_type`
  changes the projector's parameters, so a checkpoint trained with one type cannot be loaded into another.
* Source: `src/hale_vlm/vision/connectors/` (`qformer.py`, `gated_cross_attention.py`, `factory.py`).
