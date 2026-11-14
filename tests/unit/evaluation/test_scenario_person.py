"""The scenario person answers what it can, asks with a bound, and never fails a run for asking."""

import json

import pytest

from devtools.evaluation import scenario_person
from devtools.evaluation.scenario_person import UNKNOWN, ScenarioPerson, ask_command, asked, asking

TEST_SOURCE = "src/devtools/evaluation/scenario_person.py"
pytestmark = pytest.mark.proves("evaluation.scenario_person_answers_and_grades_asking")
SHEET = {
    "facts": [{"match": "which universe", "answer": "The S&P 500.", "avoidable": True}],
    "nonblocking": ["committee.daily"],
    "ask_bound_seconds": 30,
    "ask_interval_seconds": 10,
}


def person(out, *, reply=None, posted=0):
    ticks = iter(range(0, 1000, 10))
    calls = []

    def post(record, question):
        calls.append((record, question))
        if reply is not None:
            record.with_name(record.stem + ".answer.txt").write_text(reply, encoding="utf-8")
        return not posted

    made = ScenarioPerson(SHEET, out, ask=post, clock=lambda: next(ticks), sleep=lambda s: None)
    return made, calls


@pytest.mark.parametrize(
    ("question", "code", "reply", "posted", "words", "source"),
    [
        ("Which universe should I use?", None, None, 0, "The S&P 500.", "sheet"),
        ("Write in English?", None, "Yes, English.", 0, "Yes, English.", "asked"),
        ("Write in English?", None, None, 0, UNKNOWN, "unknown"),
        (
            "Write in English?",
            None,
            "ignored",
            1,
            UNKNOWN,
            "unknown",
        ),  # a refused post waits for nothing
        ("Convene the committee?", "committee.daily", None, 0, UNKNOWN, "unknown"),
        ("Keep the network on?", "network.daily_updates", None, 0, None, "stop"),
    ],
    ids=["sheet", "asked", "unknown", "refused-post", "nonblocking", "stop"],
)
def test_the_person_answers_by_source(
    tmp_path_factory, question, code, reply, posted, words, source
):
    out = tmp_path_factory.mktemp("person")  # each call's own rows: CodSpeed repeats a body
    made, calls = person(out, reply=reply, posted=posted)
    answer = made.answer(question, code=code, session="s1")
    assert (answer.words, answer.source) == (words, source)
    assert answer.stop == (
        f"scenario_person.product_question_unanswered:{code}" if source == "stop" else None
    )
    assert len(calls) == (0 if source == "sheet" else 1)
    rows = [json.loads(line) for line in (out / "questions.jsonl").read_text().splitlines()]
    assert rows[0] | dict(at=None) == dict(
        event="asked", at=None, id=rows[0]["id"], question=question, code=code, session="s1"
    )
    assert rows[-1]["event"] == "answered" and rows[-1]["source"] == source
    assert (out / "sheet-proposals.jsonl").exists() is (source == "asked")


def test_asking_is_graded_and_an_unset_ask_asks_no_one(tmp_path_factory):
    out = tmp_path_factory.mktemp("person")  # each call's own rows: CodSpeed repeats a body
    made = ScenarioPerson(SHEET, out)
    made.scripted("Keep the network on?", "Yes.", code="network.daily_updates", session="s")
    made.answer("Which universe should I use?")
    made.answer("Write in English?")
    assert made.answer("Keep the network on?", code="network.daily_updates").stop
    graded = asking(out)
    assert graded | dict(waited_s=0) == dict(
        questions=4,
        by_source=dict(script=1, sheet=1, unknown=1, stop=1),
        avoidable=1,
        waited_s=0,
        stops=["scenario_person.product_question_unanswered:network.daily_updates"],
    )
    assert asking(out / "absent") == dict(
        questions=0, by_source={}, avoidable=0, waited_s=0, stops=[]
    )


@pytest.mark.parametrize(
    ("rows", "question"),
    [
        (
            [{"type": "result", "result": "Done.\n\nShall I run the daily now?"}],
            "Shall I run the daily now?",
        ),
        ([{"type": "result", "result": "Is this right? I ran it.\n\nThe study is written."}], None),
        (
            [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Which date **?**"},
                }
            ],
            "Which date **?**",
        ),
        ([{"type": "assistant"}], None),
    ],
)
def test_the_agents_own_question_is_its_last_paragraph(rows, question):
    assert asked(rows) == question


@pytest.mark.parametrize(
    ("raw", "argv"),
    [
        (None, None),
        ('["board", "{ask}"]', ["board", "{ask}"]),
        ("board", ValueError),
        ("[]", ValueError),
    ],
)
def test_the_ask_command_is_a_json_argv_or_refused_by_name(raw, argv):
    environment = {} if raw is None else {scenario_person.ASK_ENV: raw}
    if argv is ValueError:
        with pytest.raises(ValueError, match=r"scenario_person\.ask_unreadable"):
            ask_command(environment)
    else:
        assert ask_command(environment) == argv


@pytest.mark.parametrize(
    ("case", "prompt", "terminal", "stop", "source"),
    [
        ("scripted", "Yes, keep it on.", False, None, "script"),
        (
            "product",
            "Continue.",
            True,
            "scenario_person.product_question_unanswered:net.daily",
            "stop",
        ),
        ("nonblocking", UNKNOWN, False, None, "unknown"),
        ("agent", "Yes, in English.", False, None, "asked"),
        (
            "relayed",
            "Continue.",
            False,
            None,
            None,
        ),  # a pending Host question owns the agent's words
        ("acted", "Continue.", False, None, None),
    ],
    ids=["scripted", "product", "nonblocking", "agent", "relayed", "acted"],
)
def test_a_native_turns_questions_go_to_the_person(
    tmp_path_factory, case, prompt, terminal, stop, source
):
    out = tmp_path_factory.mktemp("person")  # each call's own rows: CodSpeed repeats a body
    made, _ = person(out, reply="Yes, in English.")
    made.nonblocking = {"net.daily"} if case == "nonblocking" else set()
    made.ask = made.ask if case == "agent" else None
    answered, offered = set(), {"net.daily"}
    turn = dict(
        prompt=prompt if case == "scripted" else "Continue.",
        terminal=case in {"product", "nonblocking"},
        coded=[] if case in {"scripted", "agent", "acted"} else ["net.daily"],
        unavailable=None,
        acted="local_web.detached_breakaway_denied" if case == "acted" else None,
    )
    rows = [
        {"type": "result", "result": "The study is written.\n\nShould the report be in English?"}
    ]
    scripted = (
        {"net.daily": ("Keep the network on?", "Yes, keep it on.")} if case == "scripted" else {}
    )
    got = made.after_turn(
        turn,
        rows,
        scripted=scripted,
        words=str.upper,
        answered=answered,
        offered=offered,
        session="s",
    )
    assert got == (prompt, terminal, stop)
    graded = asking(out)
    assert graded["by_source"] == ({source: 1} if source else {}) and graded["stops"] == (
        [stop] if stop else []
    )
    assert ("net.daily" in answered) is (case == "nonblocking") and ("net.daily" in offered) is (
        case != "nonblocking"
    )
