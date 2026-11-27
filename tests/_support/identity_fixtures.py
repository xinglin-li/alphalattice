"""Synthetic identities for fixtures that need a valid digest without a stored body."""

from alphalattice.kernel.identity import canonical_hash


def fixture_digest(label: object) -> str:
    """Give the same case label the same tagged identity."""
    return canonical_hash(label)
