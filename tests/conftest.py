import pytest

import hale_vlm  # noqa: F401 — registers VLM plugins
from hale_vlm.bootstrap import register_vla_plugins

register_vla_plugins()
