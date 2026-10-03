from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Pydantic model that rejects unknown YAML keys."""

    model_config = ConfigDict(extra="forbid")
