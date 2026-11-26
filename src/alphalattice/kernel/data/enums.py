"""Canonical vocabulary for the data truth layer."""

from enum import StrEnum


class LicenseClass(StrEnum):
    """Redistribution authority attached to acquired data."""

    REDISTRIBUTABLE = "REDISTRIBUTABLE"
    PERSONAL_RESEARCH_ONLY = "PERSONAL_RESEARCH_ONLY"
    USER_PROVIDED = "USER_PROVIDED"
    UNKNOWN_BLOCKED = "UNKNOWN_BLOCKED"


class ExecutionSessionStatus(StrEnum):
    """Official-open execution eligibility for one session."""

    VERIFIED_ELIGIBLE = "VERIFIED_ELIGIBLE"
    ASSUMED_ELIGIBLE_FROM_DAILY_BAR = "ASSUMED_ELIGIBLE_FROM_DAILY_BAR"
    NO_OFFICIAL_OPEN = "NO_OFFICIAL_OPEN"
    HALTED_OR_MARKET_RESTRICTED = "HALTED_OR_MARKET_RESTRICTED"
    ELIGIBILITY_UNKNOWN = "ELIGIBILITY_UNKNOWN"


class ExecutionVenueStatus(StrEnum):
    """Verified venue permission for a proposed execution."""

    VERIFIED_ELIGIBLE = "VERIFIED_ELIGIBLE"
    HALTED_OR_MARKET_RESTRICTED = "HALTED_OR_MARKET_RESTRICTED"


class UniverseIndex(StrEnum):
    """Supported source index for current-universe construction."""

    SP500 = "SP500"
    NASDAQ100 = "NASDAQ100"
    DJIA = "DJIA"
