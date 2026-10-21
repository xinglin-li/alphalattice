"""Acceptance checks for the portable evaluation contracts."""


def plan(**changes):
    return {
        "commands": ["alphalattice feature controls --binding <id>"],
        "fields": [],
        "steps": [],
        "acts": [],
        "decider": "agent",
        "ask": [],
        "refuse": [],
        **changes,
    }
