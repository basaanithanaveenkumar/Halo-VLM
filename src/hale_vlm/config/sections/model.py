from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.common import StrictModel
from hale_vlm.config.sections.gwm import GWMConfig
from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.config.sections.scratch import ScratchConfig
from hale_vlm.config.sections.vision import VisionConfig

_VARIANT_ARCHITECTURE = {
    "qwen3_8b_vlm": "hale",
    "deepseek_r1_qwen_7b_vlm": "hale",
    "basic_vlm_scratch": "basic",
    "halo_vlm_moe": "halo_moe",
    "gwm_vla": "gwm_vla",
}


class ModelConfig(StrictModel):
    """Generic transformer architecture knobs (scratch LM defaults)."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64
    attn_type: Literal["causal", "bidirectional", "block_causal"] = "causal"
    attn_impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    ffn_type: Literal["mlp", "geglu", "moe"] = "mlp"
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1
    use_time_cond: bool = False
    block_size: int | None = None
    n_mtp_heads: int = 2
    arch: Literal["default", "transformer", "lgt", "dit"] = "default"
    sliding_window: int = 512
    local_global_ratio: int = 5
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    p_rope: float = 0.25
    qk_norm: bool = False


class VLMModelConfig(ModelConfig):
    """VLM model section: architecture routing plus vision and LLM backbones."""

    architecture: Literal["auto", "hale", "basic", "halo_moe", "gwm_vla"] = "auto"
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scratch: ScratchConfig = Field(default_factory=ScratchConfig)
    gwm: GWMConfig = Field(default_factory=GWMConfig)

    def resolved_architecture(
        self, variant: str
    ) -> Literal["hale", "basic", "halo_moe", "gwm_vla"]:
        if self.architecture != "auto":
            return self.architecture
        return _VARIANT_ARCHITECTURE.get(variant, "hale")
