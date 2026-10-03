from pydantic import Field

from hale_vlm.config.sections.common import StrictModel


class EvalConfig(StrictModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "loss",
            "perplexity",
            "bits_per_token",
            "token_accuracy",
            "masked_accuracy",
        ]
    )
    every_n_epochs: int | None = 1
    max_batches: int | None = None
