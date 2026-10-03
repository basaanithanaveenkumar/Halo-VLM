from hale_vlm.config.sections.common import StrictModel


class ExperimentConfig(StrictModel):
    enabled: bool = True
    root: str = "data/experiments"
    name: str | None = None
    source_yaml: str | None = None
