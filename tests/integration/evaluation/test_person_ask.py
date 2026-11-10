"""The scenario person's ASK: one bounded process per question, its outcome beside the record."""

import json
import sys

import pytest

from devtools.evaluation.person_ask import poster

TEST_SOURCE = "src/devtools/evaluation/person_ask.py"


@pytest.mark.proves("evaluation.person_ask_posts_each_question_as_one_bounded_process")
def test_each_question_is_one_bounded_process_with_its_outcome_beside_the_record(tmp_path):
    """No command posts nothing; a post fills the record and question, a refusal or a command
    that cannot start is recorded as such."""
    assert poster(None) is None
    record, posted = tmp_path / "ask.json", tmp_path / "ask.posted.json"
    said = "import sys; sys.stderr.write('|'.join(sys.argv[1:])); sys.exit(len(sys.argv[2]) > 100)"
    post = poster([sys.executable, "-c", said, "{ask}", "{question}"])
    assert post(record, "May I use the network?") is True
    assert json.loads(posted.read_text("utf-8")) == {
        "exit": 0,
        "said": f"{record}|May I use the network?",
    }
    assert post(record, "x" * 200) is False  # the question is cut to 120 characters
    assert json.loads(posted.read_text("utf-8")) == {"exit": 1, "said": f"{record}|{'x' * 120}"}
    assert poster([str(tmp_path / "absent")])(record, "Anyone?") is False
    assert set(json.loads(posted.read_text("utf-8"))) == {"error"}
