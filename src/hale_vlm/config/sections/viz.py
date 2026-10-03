from typing import Literal

from hale_vlm.config.sections.common import StrictModel


class VizConfig(StrictModel):
    enabled: bool = False
    every_n_steps: int | None = 500
    every_n_epochs: int | None = None
    sampling_steps: int | None = 8
    max_new_tokens: int | None = None
    prompt_source: Literal["dataset", "config"] = "dataset"
    prompt_words: int = 8
    output_dir: str = "data/viz"
