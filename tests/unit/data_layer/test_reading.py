"""The reading views' reader: a process that does not answer is ended and refused by name."""

import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import reading

TEST_SOURCE = "src/alphalattice/kernel/data_layer/reading.py"


class _Silent:
    """A reader's process that takes each request and never answers: its output ends only when it
    is killed."""

    def __init__(self) -> None:
        self.stdin, self.stdout = io.BytesIO(), self
        self.code: int | None = None
        self._killed = threading.Event()

    def read(self, _size: int) -> bytes:
        self._killed.wait(10)
        return b""

    def poll(self) -> int | None:
        return self.code

    def kill(self) -> None:
        self.code = -9
        self._killed.set()


@pytest.mark.proves("data_layer.a_stuck_log_reader_is_ended")
def test_a_reader_that_does_not_answer_is_ended_and_refused_by_name(tmp_path, monkeypatch):
    """A reader's process that never answers is refused past READ_SECONDS, and is ended."""
    monkeypatch.setattr(reading, "READ_SECONDS", 0.2)
    silent = _Silent()
    with ThreadPoolExecutor(max_workers=1) as answers, pytest.raises(DataError) as refused:
        reading._ask(silent, ("experience", {"limit": 1}), answers, tmp_path)
    assert refused.value.failure.code == "data_layer.history_unavailable"
    assert silent.poll() == -9


@pytest.mark.proves("data_layer.a_class_width_is_named")
def test_an_unknown_class_width_is_refused_by_name_before_any_read():
    """A width other than recipe, method_version or method is refused by its code, naming the
    three."""
    with pytest.raises(DataError, match="recipe, method_version, method") as refused:
        reading.experience_class(None, trial="t1", width="family")
    assert refused.value.failure.code == "experience.class_width_unknown"


@pytest.mark.proves("data_layer.a_brief_budget_is_bounded")
@pytest.mark.parametrize("budget", [99, 32_001, 0, -1], ids=["under", "over", "zero", "negative"])
def test_a_brief_budget_out_of_range_is_refused_by_name_before_any_read(budget):
    """A budget outside 100 to 32,000 tokens is refused by its code, naming the range."""
    with pytest.raises(DataError, match="100 to 32000 tokens") as refused:
        reading.brief(None, goal="g1", budget=budget)
    assert refused.value.failure.code == "experience.brief_budget_invalid"


@pytest.mark.proves("data_layer.a_condition_digest_names_its_content")
@pytest.mark.parametrize(
    ("other", "same"),
    [
        pytest.param({"market_vol_63d": 0.2}, True, id="equal-values"),
        pytest.param({"market_vol_63d": 0.3}, False, id="another-value"),
        pytest.param({"market_vol_63d": None}, False, id="undefined"),
    ],
)
def test_a_condition_digest_is_its_values_and_their_question(other, same):
    """A market condition's content digest binds its values, so equal values of one question
    share it and different ones do not."""
    first, _ = reading.condition_events("s1", "u1", 5, {"market_vol_63d": 0.2}, 1)
    second, _ = reading.condition_events("s2", "u1", 5, other, 1)
    assert (first["content_digest"] == second["content_digest"]) is same


class _Answering:
    """A reader's process that answers each request with `answer`, as `serve` sends it."""

    def __init__(self, answer: object) -> None:
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO()
        reading._send(self.stdout, answer)
        self.stdout.seek(0)


@pytest.mark.proves("data_layer.a_reads_answer_and_refusal_cross_the_reader")
@pytest.mark.parametrize(
    ("answer", "code"),
    [
        pytest.param((True, {"members": [], "search": {}}), None, id="an-answer-whole"),
        pytest.param(
            (False, ("experience.trial_unknown", "not held")),
            "experience.trial_unknown",
            id="a-refusal-by-its-code",
        ),
        pytest.param((False, "OSError: gone"), "data_layer.history_unavailable", id="a-fault"),
    ],
)
def test_a_reads_answer_or_refusal_reaches_the_host_as_the_read_gave_it(tmp_path, answer, code):
    """A read's answer reaches the Host as the read returned it, a mapping too; the read's own
    refusal by its code; any other failure as the history being unavailable."""
    with ThreadPoolExecutor(max_workers=1) as answers:
        if code is None:
            assert reading._ask(_Answering(answer), ("q", {}), answers, tmp_path) == answer[1]
            return
        with pytest.raises(DataError) as refused:
            reading._ask(_Answering(answer), ("q", {}), answers, tmp_path)
    assert refused.value.failure.code == code
