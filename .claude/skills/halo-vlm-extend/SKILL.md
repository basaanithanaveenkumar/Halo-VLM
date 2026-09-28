---
name: halo-vlm-extend
description: Extend Halo-VLM without editing core code — register a new model variant, loss, trainer, LLM backbone, vision encoder or dataset adapter, and route configs to it. Use when adding a new architecture or backbone, a new data source, or a new YAML preset.
---

# Extending Halo-VLM

## New model variant

See `examples/register_custom_model.py`:

```python
from hale_vlm.registry import register_model, register_loss

@register_model("my_vlm")
class MyVLM(nn.Module):
    fusion_mode = "prefix_concat"        # informational
    num_image_tokens = 64                 # used by the scratch loss to prepend -100 targets
    def __init__(self, vocab_size, cfg=None, **kw): ...
    def trainable_parameters(self): yield from self.parameters()
    def forward(self, images, input_ids, attention_mask=None): ...   # -> logits [B, T_img+T, V]
    @classmethod
    def from_config(cls, cfg): ...

register_loss("my_vlm")(my_loss_fn)       # or reuse training.losses._scratch_vlm_loss
```

Then:
1. Add `"my_vlm": "<architecture>"` to `_VARIANT_ARCHITECTURE` in
   `config/sections/model.py` (or set `model.architecture` in YAML).
2. If it is a scratch model, extend `build_scratch_vlm` in `models/scratch/factory.py` and
   add the name to `SCRATCH_VARIANTS` so the scratch loss is registered.
3. Import the module from `bootstrap.py`/`registry/plugins.py` so registration runs.
4. Add a `configs/my_vlm_overfit.yaml` and a shape test with a tiny config.

## New LLM backbone (Hale path)

1. Extend `LLMConfig.backbone` (`Literal[...]`) in `config/sections/llm.py`.
2. Add default LoRA targets in `llm/adapters.py::DEFAULT_LORA_TARGETS`.
3. Add a preset (`model_id`, optional `reasoning_mode`) to `LLM_PRESETS` in `llm/backbones.py`; `resolve_llm_config` applies it.
4. Add a variant name to `VLM_VARIANTS` and `_VARIANT_ARCHITECTURE` (`"hale"`).

## New vision encoder

`VisionConfig.encoder` is `Literal["siglip", "clip"]`. Add a branch in
`vision/encoders.py::VisionTower._load_encoder` returning a module whose forward gives
`[B, N, D_v]` and set `hidden_size`. Keep `num_image_tokens` consistent with N.

## New dataset

Same pattern as Hale-VLM: `@register_dataset("name")` with a `DatasetSpec` in
`data/datasets/builtin.py`, the name added to a stage tuple in `data/catalog.py`, and a
check in `tests/unit/test_dataset_registry.py`.

## Docs

Update `docs/architecture.md` (and the duplicate `docs/vlm_architecture.md`), the
`configuration.md` table and `docs/mkdocs.yml` nav when you add a page. Check that every
Mermaid block renders. In `sequenceDiagram`, don't use `&lt;`/`&gt;` entities, because `;`
ends a statement there.
