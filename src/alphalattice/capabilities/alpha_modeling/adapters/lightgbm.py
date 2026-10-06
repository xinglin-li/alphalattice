"""Development-only deterministic LightGBM capability for Alpha Research."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

LIGHTGBM_ADAPTER_ID = "lightgbm"
LIGHTGBM_RECIPE_SCHEMA_ID = "alpha-model.lightgbm"
LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.lightgbm.search-domain"
LIGHTGBM_CONTENT_FORMAT_ID = "alpha-model.lightgbm.model-text"
LIGHTGBM_STATE_SCHEMA_ID = "alpha-model.lightgbm.development-state"


def _load_lightgbm() -> Any:
    return import_module("lightgbm")


__all__ = [
    "LIGHTGBM_ADAPTER_ID",
    "LIGHTGBM_CONTENT_FORMAT_ID",
    "LIGHTGBM_RECIPE_SCHEMA_ID",
    "LIGHTGBM_SEARCH_DOMAIN_SCHEMA_ID",
    "LIGHTGBM_STATE_SCHEMA_ID",
]
