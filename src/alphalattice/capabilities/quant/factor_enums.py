"""Lightweight enums for the governed factor registry."""

from enum import StrEnum


class FactorFamily(StrEnum):
    """Governed family used to group registered factor formulas."""

    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    HIGHER_MOMENTS = "higher-moments"
    LOTTERY = "lottery"
    LIQUIDITY = "liquidity"
    SYSTEMATIC = "systematic"
    PRICE_LEVEL_TREND = "price-level-trend"
    PRICE_VOLUME = "price-volume"
    SEASONALITY = "seasonality"
    TECHNICAL = "technical"


class FactorTrack(StrEnum):
    """Whether a factor feeds models, display, or both."""

    MODEL = "model"
    DISPLAY = "display"
    BOTH = "both"


class FactorValueStatus(StrEnum):
    """Whether one factor value was computed or ruled ineligible."""

    COMPUTED = "COMPUTED"
    INELIGIBLE = "INELIGIBLE"


__all__ = ["FactorFamily", "FactorTrack", "FactorValueStatus"]
