"""Multi-token prediction for the scratch HaloVLM: switchable objective, alignment, causality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hale_vlm.config import load_vlm_config
from hale_vlm.models import build_vlm
from hale_vlm.models.scratch.components.mtp import mtp_loss, shift_targets
from hale_vlm.models.scratch.halo_vlm import HaloVLM
from tests.unit.test_scratch_data import scratch_tokenizer  # noqa: F401  (fixture)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
VOCAB = 50


def tiny(objective: str = "autoregressive", style: str = "parallel", k: int = 3) -> HaloVLM:
    torch.manual_seed(0)
    return HaloVLM(VOCAB, emb_dim=32, img_size=32, patch_size=16, vit_layers=1, vit_heads=2,
                   decoder_layers=1, decoder_heads=2, objective=objective, mtp_num_future_tokens=k,
                   mtp_style=style).eval()


def inputs(b: int = 2, length: int = 6):
    g = torch.Generator().manual_seed(1)
    return torch.rand(b, 3, 32, 32, generator=g), torch.randint(0, VOCAB, (b, length), generator=g)


def test_autoregressive_is_the_default_and_unchanged() -> None:
    model = tiny()
    images, ids = inputs()
    assert model.objective == "autoregressive" and model.mtp is None
    logits = model(images, ids)
    assert torch.is_tensor(logits) and logits.shape == (2, 4 + 6, VOCAB)
    same, extra = model(images, ids, return_mtp=True)
    assert torch.equal(same, logits) and extra == []


@pytest.mark.parametrize("style", ["parallel", "sequential"])
def test_mtp_depth_shapes_and_gradients(style: str) -> None:
    model = tiny("mtp", style, k=4).train()
    images, ids = inputs()
    logits, extra = model(images, ids, return_mtp=True)
    seq = 4 + 6
    assert [t.shape for t in extra] == [(2, seq - d, VOCAB) for d in (1, 2, 3)]
    targets = torch.cat([torch.full((2, 4), -100), ids[:, 1:], torch.full((2, 1), -100)], 1)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=-100)
    total = loss + 0.3 * mtp_loss(extra, targets)
    total.backward()
    # Routed MoE experts only get gradients when top-k routing selects them; everything else must.
    missing = [n for n, p in model.mtp.named_parameters() if p.grad is None and "routed_experts" not in n]
    assert not missing, missing
    assert model.token_emb.weight.grad is not None  # MTP gradients reach the shared trunk


def test_targets_are_aligned_with_depths() -> None:
    """Logits that put all mass on targets[:, i + d] give (almost) zero loss."""
    targets = torch.tensor([[-100, 7, 3, 9, 1, 4, -100]])
    extra = []
    for d in (1, 2):
        tgt = shift_targets(targets, d)
        logits = torch.full((1, targets.size(1) - d, VOCAB), -30.0)
        for i, t in enumerate(tgt[0].tolist()[: logits.size(1)]):
            if t >= 0:
                logits[0, i, t] = 30.0
        extra.append(logits)
    assert mtp_loss(extra, targets).item() < 1e-6
    wrong = [torch.roll(e, 1, dims=1) for e in extra]
    assert mtp_loss(wrong, targets).item() > 10


@pytest.mark.parametrize("style", ["parallel", "sequential"])
def test_depth_d_never_sees_beyond_token_i_plus_d(style: str) -> None:
    model = tiny("mtp", style, k=3)
    images, ids = inputs(b=1, length=8)
    changed = ids.clone()
    pos = 6  # text position; sequence position = 4 image tokens + 6
    changed[0, pos] = (ids[0, pos] + 1) % VOCAB
    with torch.no_grad():
        _, a = model(images, ids, return_mtp=True)
        _, b = model(images, changed, return_mtp=True)
    p = 4 + pos
    for d, (ta, tb) in enumerate(zip(a, b, strict=True), start=1):
        # Depth d at position i may read tokens up to i + d (the sequential variant reads x_{i+d}).
        safe = p - d
        assert torch.allclose(ta[:, :safe], tb[:, :safe], atol=1e-5), (style, d)


def test_config_switch_builds_mtp_and_basic_rejects_it(scratch_tokenizer) -> None:  # noqa: F811
    with patch("hale_vlm.data.vlm.multimodal.AutoTokenizer.from_pretrained", lambda *a, **k: scratch_tokenizer):
        cfg = load_vlm_config(CONFIGS / "halo_moe_overfit.yaml")
    cfg.model.scratch.decoder_num_layers, cfg.model.scratch.vit_num_layers = 1, 1
    cfg.model.scratch.objective = "mtp"
    cfg.model.scratch.mtp_style = "sequential"
    model = build_vlm(cfg)
    assert model.objective == "mtp" and model.mtp.style == "sequential" and model.mtp.depths == 3
    cfg.model.architecture = "basic"
    with pytest.raises(ValueError, match="halo_moe"):
        build_vlm(cfg)


@pytest.mark.parametrize("style", ["parallel", "sequential"])
def test_trainer_epoch_with_mtp_objective(scratch_tokenizer, style: str) -> None:  # noqa: F811
    from hale_vlm.training.scratch_trainer import ScratchVLMTrainer

    with patch("hale_vlm.data.vlm.multimodal.AutoTokenizer.from_pretrained", lambda *a, **k: scratch_tokenizer):
        cfg = load_vlm_config(CONFIGS / "halo_moe_overfit.yaml")
        cfg.model.scratch.embed_dim, cfg.model.scratch.decoder_num_layers = 64, 1
        cfg.model.scratch.vit_num_layers, cfg.model.scratch.decoder_num_heads = 1, 4
        cfg.model.scratch.vit_num_heads = 4
        cfg.model.scratch.objective, cfg.model.scratch.mtp_style = "mtp", style
        trainer = ScratchVLMTrainer.from_config(cfg)
    assert trainer.uses_mtp and trainer.mtp_loss_weight == pytest.approx(0.3)
    loss = trainer.train_epoch(0)
    assert loss == loss and loss > 0  # finite, positive
