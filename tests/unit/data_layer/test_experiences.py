"""A note's one text form, and the one markdown body the Host renders of it."""

from __future__ import annotations

import json

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.experiences import note_text, render

TEST_SOURCE = "src/alphalattice/foundation/data_layer/experiences.py"
TEXT = note_text("Momentum pays", "12-1 momentum", "It paid, thinly", ["Mind costs"])
RUN = {"gen_ai.usage.input_tokens": 1200, "gen_ai.usage.output_tokens": 300, "cost_usd": 0.25}
NOTE = {
    "subject_kind": "trial",
    "subject_id": "t1",
    "workspace_id": "w1",
    "goal_id": "g1",
    "task_id": "k1",
    "occurred_us": 1_794_700_000_000_000,
    "actor_kind": "agent",
    "actor_id": "claude",
    "attrs": json.dumps({"verdict": "success", "seconds": 42.5} | RUN),
}


@pytest.mark.proves("data_layer.a_note_says_each_part")
@pytest.mark.parametrize(
    ("hypothesis", "method", "conclusion", "lessons", "refused"),
    [
        pytest.param("Momentum pays", "12-1", "It paid", [], False, id="no-lessons"),
        pytest.param("Momentum pays", "12-1", "It paid", ["Mind costs"], False, id="a-lesson"),
        pytest.param("", "12-1", "It paid", [], True, id="no-hypothesis"),
        pytest.param("Momentum pays", "  ", "It paid", [], True, id="blank-method"),
        pytest.param("Momentum pays", "12-1", None, [], True, id="no-conclusion"),
        pytest.param(
            "Momentum pays", "12-1", "It paid", ["Mind costs", " "], True, id="blank-lesson"
        ),
        pytest.param(
            "Momentum pays", "12-1", "It paid", "Mind costs", True, id="lessons-not-a-list"
        ),
    ],
)
def test_a_note_says_its_hypothesis_method_and_conclusion(
    hypothesis, method, conclusion, lessons, refused
):
    """A note's text holds each part it was given; one missing a part or with a blank lesson is
    refused by name, never stored as a review that says nothing."""
    if refused:
        with pytest.raises(DataError) as error:
            note_text(hypothesis, method, conclusion, lessons)
        assert error.value.failure.code == "experience.note_incomplete"
        return
    written = json.loads(note_text(hypothesis, method, conclusion, lessons))
    assert written == {
        "hypothesis": hypothesis,
        "method": method,
        "conclusion": conclusion,
        "lessons": lessons,
    }


@pytest.mark.proves("data_layer.a_note_renders_the_same_bytes")
def test_the_same_note_renders_the_same_bytes():
    """A body is a function of its note alone: rendered twice it is the same bytes (so its BLAKE3
    name is), and a different note renders a different body."""
    assert render(NOTE, TEXT) == render(dict(NOTE), TEXT)
    other = note_text("Momentum pays", "12-1 momentum", "It did not pay", ["Mind costs"])
    assert render(NOTE, other) != render(NOTE, TEXT)
    assert render(NOTE | {"actor_id": "codex"}, TEXT) != render(NOTE, TEXT)


@pytest.mark.proves("data_layer.an_experience_body_shows_its_run")
def test_a_body_shows_its_run_its_place_and_its_lessons():
    """A body names its scope, verdict and run (seconds, tokens and cost) beside the note, each
    measure's place in its spread (said so when there is none yet) and a missing lesson list."""
    place = [{"metric": "ic", "version": 1, "z": 1.5, "n": 9}]
    place.append({"metric": "sharpe", "version": 2, "z": None, "n": 1})
    attrs = json.loads(NOTE["attrs"]) | {"place": place}
    plain = note_text("Momentum pays", "12-1 momentum", "It paid, thinly", [])
    body = render(NOTE | {"goal_id": None, "attrs": json.dumps(attrs)}, plain).decode()
    for line in (
        "# Experience of trial t1",
        "- Verdict: success",
        "- Goal: none",
        "- Task: k1",
        "- Reviewed by: agent claude",
        "- Run: 42.5 s, 1200 tokens in, 300 out, 0.25 USD",
        "- Place: ic v1, z 1.5 among 9 trials",
        "- Place: sharpe v2, no spread yet among 1 trials",
        "## Conclusion\n\nIt paid, thinly",
        "## Lessons\n\nNone recorded.",
    ):
        assert line in body
