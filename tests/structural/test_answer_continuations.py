"""The saved-answer continuation class, generated from its owning registries."""

from pathlib import Path

from devtools.architecture.answer_continuations import (
    continuation_checks,
    documented_edges,
    reference_checks,
)

ROOT = Path(__file__).resolve().parents[2]


def test_every_compact_locator_restores_by_the_same_rule() -> None:
    """Contract: schema references and answer locators, plain and prefixed,
    hashes and UUIDs, including list members, are whole after a compact round trip."""
    rows = reference_checks(ROOT)
    assert rows
    assert [row for row in rows if row["class"] != "BOUND"] == []


def test_every_continuation_reference_binds_or_refuses_with_words() -> None:
    """contract (TE12): each operation's reference fields and declared aliases bind;
    a required subject missing from a saved answer cannot silently become a default."""
    rows = continuation_checks(ROOT)
    assert rows
    assert [row for row in rows if row["class"] == "FAIL"] == []


def test_the_skill_saved_answer_chains_bind_or_name_their_missing_subject() -> None:
    """Contract: the Skill's commands and references generate the checked chains;
    a declared answer either supplies the next subject or refuses it with words."""
    edges = documented_edges(ROOT)
    assert edges
    probes = [probe for edge in edges for probe in edge["reference_probes"]]
    assert probes
    assert [probe for probe in probes if probe["class"] == "FAIL"] == []
