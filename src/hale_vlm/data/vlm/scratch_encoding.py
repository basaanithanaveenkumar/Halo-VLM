"""Text and image encoding for prefix-fusion scratch VLMs."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms


def scratch_image_transform(image_size: int, *, is_train: bool = True) -> transforms.Compose:
    del is_train
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def format_scratch_caption(text: str) -> str:
    caption = text.strip()
    return f"Describe the image{caption}"


def encode_scratch_text(
    tokenizer,
    text: str,
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize caption text and build shifted next-token labels."""
    prompt = format_scratch_caption(text)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None and hasattr(tokenizer, "sep_token_id"):
        eos_token_id = tokenizer.sep_token_id
    if eos_token_id is None:
        eos_token_id = 2

    encoded = tokenizer(
        prompt,
        max_length=max_length - 1,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    seq_len = int((attention_mask == 1).sum().item())

    if seq_len < len(input_ids):
        input_ids[seq_len] = eos_token_id
        attention_mask[seq_len] = 1
    else:
        input_ids[-1] = eos_token_id

    labels = input_ids[1:].clone()
    labels = torch.cat([labels, torch.tensor([-100], dtype=labels.dtype)])
    labels[labels == 0] = -100
    labels[attention_mask == 0] = -100
    if attention_mask[-1] == 1:
        labels[-1] = -100

    return input_ids, attention_mask, labels


def encode_scratch_image(image, transform) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image)
