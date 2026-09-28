# Known issues

These were found while documenting the code (September 2026). They affect what the model
actually sees and learns, so fix them before reporting results.

## 1. Visual tokens overwrite text after `<image>`

`HaleVLM.merge_image_embeddings` (`src/hale_vlm/models/vlm.py`, Hale path only) finds the first `<image>`
position `s` and builds `prefix = embeds[:s]`, `image_embeds`, `suffix = embeds[s + N:]`, where
`N = num_image_tokens`. The prompt contains **one** `<image>` placeholder, so tokens
`s+1 … s+N−1` (the start of the instruction) are silently dropped. With `N = 196` and
`max_length = 512`, most short prompts lose their whole instruction.

Minimal reproduction (no model download):

```
ids    = [1, 2, <image>, 3, 4, 5, 6, 7, 8, 9],  N = 4
merged = [e1, e2, img, img, img, img, e6, e7, e8, e9]    # tokens 3, 4, 5 lost
```

**Fix options.** Expand the placeholder to `N` copies of `<image>` at tokenisation time (the
current merge then works), or insert the visual tokens instead of overwriting and pad `labels`
with `-100` over the inserted span.

## 2. Loss covers the entire prompt

The Hale-path encoders in `data/multimodal.py` set `labels = input_ids` (only padding → `-100`), and
`_build_prompt` uses `sample.text` as both user text and assistant answer. Mask everything
before the assistant turn and use a real target field.

## 3. Only the first image per sample

`_visual_tensor` returns `tensors[0]` for image and multi-image samples. Multi-image
datasets (M4-Instruct, MAmmoTH-VL) therefore contribute one image each.

## 4. Scratch HaloVLM ignores `attention_mask`

`HaloVLM.forward` discards `attention_mask` and relies on the causal mask. This is correct
for right-padded batches (as produced by the provided collators) but not for left padding.

## 5. Minor

- `format_scratch_caption` concatenates without a separator: `"Describe the image" + caption`.
- `docs/vlm_architecture.md` duplicates `docs/architecture.md` byte for byte.
