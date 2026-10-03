from hale_vlm.config.sections.common import StrictModel


class SampleConfig(StrictModel):
    max_new_tokens: int = 32
    temperature: float = 1.0
    sampling_steps: int = 32
    eps: float = 1e-3
    prompt: str = "hello"
