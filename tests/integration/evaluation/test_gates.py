"""The native release gates run as one Inspect AI task over each host's one prepared leg."""

import json
import sys

import pytest

from devtools.evaluation import gates

TEST_SOURCE = "src/devtools/evaluation/gates.py"
pytestmark = pytest.mark.proves("evaluation.release_gates_run_as_one_inspect_task")
SHA = "a" * 40
LEG = r"""import hashlib, json, sys
from pathlib import Path
action, run = sys.argv[1], Path(sys.argv[3])
if action == "start":
    host, grades = sys.argv[sys.argv.index("--host") + 1], {}
    names = ("summary.json", "daily/summary.json", "chain-summary.json", "budget.json")
    for name in ("fresh/" + n for n in names):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text('{"held": true, "status": "COMPLETE"}')
        grades[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    previous = sys.argv[sys.argv.index("--previous") + 1] if "--previous" in sys.argv else None
    leg = dict(state="DONE", source=sys.argv[sys.argv.index("--source") + 1], host=host, stop=None,
               previous=previous, grades=grades)
    (run / "leg.json").write_text(json.dumps(leg))
    cancel = dict(held=True, task="t")
    upgrade = dict(held=True, book_verdict="ALL EQUAL", released_cancel=cancel)
    summary = dict(candidate_sha=leg["source"], held=True, upgrade=upgrade)
    (run / "acceptance-summary.json").write_text(json.dumps(summary))
    asked = dict(event="asked", id="q1", question="Should the report be in English?", code=None)
    (run / "fresh/questions.jsonl").write_text(json.dumps(asked) + "\n")
print(action, "done")
"""


def test_both_hosts_legs_give_every_native_gate_and_their_questions_join_the_log(
    tmp_path, monkeypatch
):
    from inspect_ai.log import read_eval_log

    script = tmp_path / "leg.py"
    script.write_text(LEG, encoding="utf-8")
    result = gates.run_gates(SHA, tmp_path / "run", leg=(sys.executable, str(script)))
    assert result["status"] == "success" and result["source"] == SHA
    assert result["hosts"]["claude-code"]["verdicts"] == {
        "native-claude-code": True,
        "resource-budget": True,
        "upgrade-axis": True,
    }
    assert result["hosts"]["codex"]["verdicts"] == {"native-codex": True, "resource-budget": True}
    assert result["hosts"]["codex"]["asking"]["questions"] == 1
    assert json.loads((tmp_path / "run/gates.json").read_text()) == json.loads(
        json.dumps(result, default=str)
    )
    log = read_eval_log(result["log"])
    events = [e for s in log.samples for e in s.events if getattr(e, "event", "") == "info"]
    assert sum(e.source == "scenario_person" for e in events) == 2  # one question per host's leg

    # A refused host must not be hidden by the other host's passing gates.
    for hosts in (
        {**result["hosts"], "claude-code": {"verdicts": {}}},
        {"codex": result["hosts"]["codex"]},
        {
            **result["hosts"],
            "claude-code": {"verdicts": {"native-claude-code": True, "resource-budget": True}},
        },
    ):
        monkeypatch.setattr(
            gates, "run_gates", lambda *a, hosts=hosts, **k: {**result, "hosts": hosts}
        )
        assert gates.main(["--source", SHA, "--run", str(tmp_path / "run")]) == 1
    monkeypatch.setattr(gates, "run_gates", lambda *a, **k: result)
    assert gates.main(["--source", SHA, "--run", str(tmp_path / "run")]) == 0
