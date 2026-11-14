"""The commit's wait rules refuse each unbounded wait and pass its repair."""

import json
import subprocess

import pytest

from devtools.architecture import waits

TEST_SCOPE = "whole-tree"

CASES = {  # rule: (a refused form, its repair)
    "while-true-without-deadline": (
        "while True:\n    log(time.monotonic())\n",
        "while True:\n    if time.monotonic() > deadline:\n        break\n",
    ),
    "retry-without-bound": (
        "while True:\n    try:\n        send()\n    except OSError:\n        log()\n",
        "while True:\n    left = deadline - time.monotonic()\n    if left <= 0:\n        raise X\n"
        "    try:\n        send(left)\n    except OSError:\n        log()\n",
    ),
    "busy-wait": (
        "while not ready:\n    pass\nwhile not ready():\n    counter += 1\n",
        "while not stop.wait(0.5):\n    counter += 1\nwhile step():\n    pass\n"
        "while i < end and ready(i):\n    i += 1\n",
    ),
    "sleep-outside-wait": ("time.sleep(period)\n", "waits.pause(1, stop=stop)\n"),
    "swallowed-in-loop": (
        "for row in rows:\n    try:\n        f(row)\n    except ValueError:\n        continue\n",
        "for row in rows:\n    try:\n        f(row)\n    except ValueError:\n        n += 1\n",
    ),
    "call-without-timeout": (
        "subprocess.run(argv)\nthread.join()\nq.put(x)\njobs_queue.get()\nf(timeout=None)\n"
        "async def pull():\n    await subscription.fetch(1)\n",
        "subprocess.run(argv, timeout=60)\nthread.join(5)\nlock.acquire(blocking=False)\n"
        "token.get()\nfuture.result(timeout=3)\n"
        "async def pull():\n    await subscription.fetch(1, timeout=5)\n",
    ),
    "wall-clock-deadline": ("end = time.time() + 30\n", "end = time.monotonic() + 30\n"),
    "deadline-reset-in-loop": (
        "while x < deadline:\n    deadline = time.monotonic() + 5\n",
        "deadline = time.monotonic() + 5\nwhile time.monotonic() < deadline:\n    step()\n",
    ),
    "blocking-under-lock": (
        "with self._lock:\n    time.sleep(period)\n",
        "with self._lock:\n    value = cache.get(key)\n",
    ),
}


@pytest.mark.proves("architecture.wait_rules_refuse_each_anti_pattern")
@pytest.mark.parametrize("rule", CASES, ids=list(CASES))
def test_each_wait_rule_refuses_its_case_and_passes_its_repair(rule):
    """requirement: each wait rule refuses its anti-pattern by name and passes its repair."""
    refused, repaired = CASES[rule]
    found = {f.rule for f in waits.findings("src/x.py", refused)}
    assert rule in found and waits.RULES[rule]
    rules = {f.rule for f in waits.findings("src/x.py", repaired)}
    assert rule not in rules, rules
    assert waits.findings(waits.PRIMITIVE, "time.sleep(period)\n") == []  # the one way to wait
    pages = "while True:\n    page = fetch(cursor)\n    if not page:\n        break\n"
    assert waits.findings("src/x.py", pages) == []  # a loop that ends on its data waits for nothing
    polls = "while not ready():\n    pass\nwhile not ready():\n    n += 1\n"
    assert [f.line for f in waits.findings("src/x.py", polls)] == [
        1,
        3,
    ]  # an idle poll and a busy one


@pytest.mark.proves("architecture.wait_baseline_only_falls")
def test_the_commit_refuses_a_new_case_and_a_raised_baseline(tmp_path, monkeypatch, capsys):
    """requirement: a staged case above the baseline, or a raised baseline, refuses the commit;
    a private file is counted in the private baseline, never the public one."""
    from scripts import check_waits

    monkeypatch.setattr(check_waits, "ROOT", tmp_path)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(tmp_path)]
    (tmp_path / "src").mkdir()
    (tmp_path / "config").mkdir()
    kind = {"src/b.py": "PRIVATE"}  # the manifest's answer for src/b.py, as for a private file
    monkeypatch.setattr(check_waits, "classify", lambda p, *_: (kind.get(p, "PUBLIC"), None, ""))
    for name in ("src/a.py", "src/b.py"):
        (tmp_path / name).write_text("import time\ntime.sleep(period)\n", encoding="utf-8")
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    assert check_waits.main(["--all", "--update"]) == 0
    sides = [json.loads((tmp_path / n).read_text("utf-8"))["counts"] for n in check_waits.SIDES]
    assert sides == [  # a private file's count lives in the private baseline alone
        {"src/a.py::sleep-outside-wait": 1},
        {"src/b.py::sleep-outside-wait": 1},
    ]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    (tmp_path / "src/a.py").write_text(
        "import time\ntime.sleep(period)\ntime.sleep(period)\n", "utf-8"
    )
    subprocess.run([*git, "add", "-A"], check=True)
    assert check_waits.main(["--staged"]) == 1
    assert "src/a.py:3: sleep-outside-wait" in capsys.readouterr().err
    subprocess.run([*git, "checkout", "--", "."], check=True)
    subprocess.run([*git, "reset", "-q", "--hard"], check=True)
    baseline = tmp_path / check_waits.BASELINE
    raised = json.loads(baseline.read_text(encoding="utf-8"))
    raised["counts"]["src/a.py::sleep-outside-wait"] = 2
    baseline.write_text(json.dumps(raised), encoding="utf-8")
    subprocess.run([*git, "add", "-A"], check=True)
    assert check_waits.main(["--staged"]) == 1
    assert "rose above HEAD's count" in capsys.readouterr().err
