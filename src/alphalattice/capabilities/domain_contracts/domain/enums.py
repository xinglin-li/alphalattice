"""Canonical product and runtime vocabulary."""

from enum import StrEnum


class DataValidityClass(StrEnum):
    """Classify the universe and temporal validity of research data.

    The vocabulary distinguishes a synthetic demonstration, fixed/current-universe conditional
    research, and point-in-time research; the enum itself grants no admission permission.
    """

    SYNTHETIC_DEMO = "SYNTHETIC_DEMO"
    FIXED_UNIVERSE_CONDITIONAL_RESEARCH = "FIXED_UNIVERSE_CONDITIONAL_RESEARCH"
    CURRENT_UNIVERSE_RESEARCH_ONLY = "CURRENT_UNIVERSE_RESEARCH_ONLY"
    POINT_IN_TIME_RESEARCH = "POINT_IN_TIME_RESEARCH"
