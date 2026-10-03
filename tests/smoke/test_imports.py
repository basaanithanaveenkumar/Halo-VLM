import pytest


@pytest.mark.smoke
def test_imports():
    import hale_vlm  # noqa: F401
    from hale_vlm.config import VLMRunConfig, load_vlm_config
    from hale_vlm.models import ALL_VLM_VARIANTS, HaleVLM
    from hale_vlm.vision.hale import VisionProjector

    assert hale_vlm.__version__ == "0.2.0"
    assert "qwen3_8b_vlm" in ALL_VLM_VARIANTS
    assert "halo_vlm_moe" in ALL_VLM_VARIANTS
    assert VLMRunConfig is not None
    assert HaleVLM is not None
    assert VisionProjector is not None
    assert load_vlm_config is not None
