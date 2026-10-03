"""Q-Former and gated cross-attention vision connectors."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from hale_vlm.config.sections.vision import VisionConfig
from hale_vlm.vision.connectors import (
    GatedCrossAttentionProjector,
    QFormer,
    build_vision_connector,
)
from hale_vlm.vision.hale import build_projector

VISION_DIM, LLM_DIM = 32, 16


def _qformer(**kwargs) -> QFormer:
    defaults = dict(num_queries=8, hidden_dim=16, num_layers=3, num_heads=4)
    return QFormer(VISION_DIM, LLM_DIM, **{**defaults, **kwargs})


def _gated(**kwargs) -> GatedCrossAttentionProjector:
    defaults = dict(num_latents=6, hidden_dim=16, num_layers=2, num_heads=4)
    return GatedCrossAttentionProjector(VISION_DIM, LLM_DIM, **{**defaults, **kwargs})


@pytest.mark.smoke
def test_qformer_output_shape_is_independent_of_input_length():
    model = _qformer()
    assert model(torch.randn(2, 49, VISION_DIM)).shape == (2, 8, LLM_DIM)
    assert model(torch.randn(2, 196, VISION_DIM)).shape == (2, 8, LLM_DIM)
    assert model.output_tokens(196) == 8


@pytest.mark.smoke
def test_qformer_cross_attention_frequency():
    model = _qformer(num_layers=4, cross_attention_freq=2)
    flags = [layer.has_cross_attention for layer in model.layers]
    assert flags == [True, False, True, False]
    assert all(layer.has_cross_attention for layer in _qformer(cross_attention_freq=1).layers)


@pytest.mark.smoke
def test_qformer_depends_on_image_and_trains_all_parameters():
    model = _qformer()
    a, b = torch.randn(2, 20, VISION_DIM), torch.randn(2, 20, VISION_DIM)
    assert not torch.allclose(model(a), model(b))

    model(a).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


@pytest.mark.smoke
def test_gated_cross_attention_zero_gate_blocks_the_image():
    model = _gated(gate_init=0.0)
    a, b = torch.randn(2, 20, VISION_DIM), torch.randn(2, 20, VISION_DIM)
    assert model(a).shape == (2, 6, LLM_DIM)
    assert torch.allclose(model(a), model(b)), "tanh(0) gate must hide the image at init"

    model(a).square().mean().backward()
    for block in model.blocks:
        assert block.attn_gate.grad is not None and block.attn_gate.grad.abs().item() > 0
        assert block.ffn_gate.grad is not None


@pytest.mark.smoke
def test_gated_cross_attention_open_gate_uses_the_image():
    model = _gated(gate_init=1.0)
    a, b = torch.randn(2, 20, VISION_DIM), torch.randn(2, 20, VISION_DIM)
    assert not torch.allclose(model(a), model(b))

    model(a).square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


@pytest.mark.smoke
@pytest.mark.parametrize("cls", [QFormer, GatedCrossAttentionProjector])
def test_connector_rejects_indivisible_heads(cls):
    with pytest.raises(ValueError, match="divisible"):
        cls(VISION_DIM, LLM_DIM, hidden_dim=10, num_heads=4)


@pytest.mark.smoke
def test_default_hidden_dim_is_capped_at_768():
    assert QFormer(32, 4096).out_proj.in_features == 768
    assert QFormer(32, 16, num_heads=4).out_proj.in_features == 16


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("projector_type", "kwargs", "cls", "tokens"),
    [
        ("qformer", {"qformer": {"num_queries": 5, "hidden_dim": 16, "num_heads": 4}}, QFormer, 5),
        (
            "gated_cross_attention",
            {"gated_cross_attention": {"num_latents": 7, "hidden_dim": 16, "num_heads": 4}},
            GatedCrossAttentionProjector,
            7,
        ),
    ],
)
def test_config_selects_connector(projector_type, kwargs, cls, tokens):
    cfg = VisionConfig(projector_type=projector_type, num_image_tokens=196, **kwargs)
    assert cfg.resolved_num_image_tokens() == tokens
    for builder in (build_vision_connector, build_projector):
        module = builder(VISION_DIM, LLM_DIM, cfg)
        assert isinstance(module, cls)
        assert module(torch.randn(2, 11, VISION_DIM)).shape == (2, tokens, LLM_DIM)


@pytest.mark.smoke
def test_mlp_projector_keeps_patch_count_and_default_token_count():
    cfg = VisionConfig(projector_type="mlp", num_image_tokens=196)
    assert cfg.resolved_num_image_tokens() == 196
    assert build_projector(VISION_DIM, LLM_DIM, cfg)(torch.randn(2, 11, VISION_DIM)).shape == (
        2,
        11,
        LLM_DIM,
    )


@pytest.mark.smoke
def test_config_rejects_unknown_projector_and_keys():
    with pytest.raises(ValidationError):
        VisionConfig(projector_type="perceiver")
    with pytest.raises(ValidationError):
        VisionConfig(qformer={"num_query": 3})
