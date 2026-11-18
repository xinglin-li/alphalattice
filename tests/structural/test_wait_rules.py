"""The commit's count ratchets: each wait rule refuses its unbounded wait and passes its repair,
and a wait finding or a type: ignore only falls (scripts/check_counts.py)."""

import json
import shutil
import subprocess

import pytest

from alphalattice.kernel.shared_kernel import spawn
from devtools.architecture import type_ignores, waits

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
    from scripts import check_counts

    kind = {"src/b.py": "PRIVATE"}  # the manifest's answer for src/b.py, as for a private file
    _counted(monkeypatch, tmp_path, kind)
    git = _counted_repo(
        tmp_path, {name: "import time\ntime.sleep(period)\n" for name in ("src/a.py", "src/b.py")}
    )
    sides = [
        json.loads((tmp_path / n).read_text("utf-8"))["counts"] for n in check_counts.WAITS.sides
    ]
    assert sides == [  # a private file's count lives in the private baseline alone
        {"src/a.py::sleep-outside-wait": 1},
        {"src/b.py::sleep-outside-wait": 1},
    ]
    (tmp_path / "src/a.py").write_text(
        "import time\ntime.sleep(period)\ntime.sleep(period)\n", "utf-8"
    )
    spawn.run([*git, "add", "-A"], check=True, timeout=30)
    assert check_counts.main(["--staged"]) == 1
    assert "src/a.py:3: sleep-outside-wait" in capsys.readouterr().err
    spawn.run([*git, "reset", "-q", "--hard"], check=True, timeout=30)
    baseline = tmp_path / check_counts.WAITS.baseline
    raised = json.loads(baseline.read_text(encoding="utf-8"))
    raised["counts"]["src/a.py::sleep-outside-wait"] = 2
    baseline.write_text(json.dumps(raised), encoding="utf-8")
    spawn.run([*git, "add", "-A"], check=True, timeout=30)
    assert check_counts.main(["--staged"]) == 1
    assert "rose above its base's count" in capsys.readouterr().err

    calls = []
    monkeypatch.setattr(
        check_counts.spawn,
        "run",
        lambda *a, **kw: (
            calls.append((a, kw)) or subprocess.CompletedProcess(a[0], 0, "answer", "")
        ),
    )
    assert check_counts._git("status") == "answer"
    assert calls[0][0][0] == ["git", "status"] and calls[0][1]["timeout"] == 60


def _counted(monkeypatch, root, kind=None):
    """Point the count ratchets at ``root``, with the manifest's answers ``kind`` (else public)."""
    from scripts import check_counts

    monkeypatch.setattr(check_counts, "ROOT", root)
    monkeypatch.setattr(
        check_counts, "classify", lambda p, *_: ((kind or {}).get(p, "PUBLIC"), None, "")
    )


def _git(root):
    return ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root)]


def _counted_repo(root, files):
    """A Git repository at ``root`` holding ``files``, both count baselines seeded and committed by
    the ratchets already pointed at it; its git argv."""
    from scripts import check_counts

    git = _git(root)
    for name, text in {**files, "config/.keep": ""}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    spawn.run([*git, "init", "-q"], check=True, timeout=30)
    spawn.run([*git, "add", "-A"], check=True, timeout=30)
    spawn.run([*git, "commit", "-qm", "files"], check=True, timeout=30)
    assert check_counts.main(["--all", "--update"]) == 0
    spawn.run([*git, "add", "-A"], check=True, timeout=30)
    spawn.run([*git, "commit", "-qm", "baselines"], check=True, timeout=30)
    return git


@pytest.fixture(scope="module")
def ignored_repo(tmp_path_factory):
    """One seeded repository whose src/a.py holds one [arg-type] ignore, copied by each case."""
    root = tmp_path_factory.mktemp("ignored")
    with pytest.MonkeyPatch.context() as patch:
        _counted(patch, root)
        _counted_repo(root, {"src/a.py": IGNORED})
    return root


IGNORED = "x = f()  # type: ignore[arg-type]\n"
RAISED = '{"counts": {"src/a.py::type-ignore[arg-type]": 2}}'
TYPE_IGNORE_CASES = {  # the change to src/a.py (one [arg-type] ignore): writes, a move, --update;
    # then the commit's exit, its words, and the baseline's counts after it
    "a-raised-count-is-refused": (
        {"src/a.py": IGNORED * 2},
        None,
        False,
        1,
        "src/a.py:2: type-ignore[arg-type]: a new type: ignore",
        None,
    ),
    "a-new-files-ignore-is-refused": (
        {"src/n.py": "y = g()  # type: ignore[attr-defined]\n"},
        None,
        False,
        1,
        "src/n.py:1: type-ignore[attr-defined]",
        None,
    ),
    "an-ignore-with-no-code-is-refused": (
        {"src/a.py": IGNORED + "z = h()  # type: ignore\n"},
        None,
        False,
        1,
        "src/a.py:2: type-ignore: a type: ignore with no error code hides every error",
        None,
    ),
    "a-broader-ignore-in-its-place-is-refused": (
        {"src/a.py": "x = f()  # type: ignore[arg-type, attr-defined]\n"},
        None,
        False,
        1,
        "src/a.py:1: type-ignore[attr-defined]",
        None,
    ),
    "a-raised-baseline-is-refused": (
        {"config/type-ignore-baseline.json": RAISED},
        None,
        False,
        1,
        "type-ignore[arg-type] rose above its base's count",
        None,
    ),
    "a-lowered-count-is-accepted-and-written-by-update": (
        {"src/a.py": "x = f()\n"},
        None,
        True,
        0,
        "type_ignores baseline lowered: 0 keys",
        {},
    ),
    "a-move-is-refused-until-update-carries-its-count": (
        {},
        ("src/a.py", "src/b.py"),
        False,
        1,
        "src/a.py moved; carry its counts",
        None,
    ),
    "a-move-carried-by-update-is-accepted": (
        {},
        ("src/a.py", "src/b.py"),
        True,
        0,
        "",
        {"src/b.py::type-ignore[arg-type]": 1},
    ),
    "an-ignore-inside-a-string-is-data": (
        {"src/n.py": 's = "x  # type: ignore[arg-type]"\n'},
        None,
        False,
        0,
        "",
        None,
    ),
}


@pytest.mark.proves("architecture.type_ignores_only_fall")
@pytest.mark.parametrize(
    ("writes", "move", "update", "code", "said", "after"),
    TYPE_IGNORE_CASES.values(),
    ids=list(TYPE_IGNORE_CASES),
)
def test_a_type_ignore_count_only_falls(
    tmp_path, monkeypatch, capsys, ignored_repo, writes, move, update, code, said, after
):
    """requirement: a commit is refused when a file's type: ignore count for an error code rises,
    a new file holds one, an ignore names no code or the baseline rose; a lowered count or a carried
    move is written by --update and then commits."""
    from scripts import check_counts

    root = shutil.copytree(ignored_repo, tmp_path / "repo")
    _counted(monkeypatch, root)
    git = _git(root)
    if move:
        spawn.run([*git, "mv", *move], check=True, timeout=30)
    for name, text in writes.items():
        (root / name).write_text(text, encoding="utf-8")
    spawn.run([*git, "add", "-A"], check=True, timeout=30)
    args = ["--staged", "--update"] if update else ["--staged"]
    assert check_counts.main(args) == code
    seen = capsys.readouterr()
    assert said in seen.out + seen.err
    if after is not None:  # the written baseline is the one the next commit holds
        baseline = root / check_counts.TYPE_IGNORES.baseline
        assert json.loads(baseline.read_text("utf-8"))["counts"] == after
        spawn.run([*git, "add", "-A"], check=True, timeout=30)
        assert check_counts.main(["--staged"]) == 0


FINDING_CASES = {
    "one-finding-per-code": (
        "x = f()  # type: ignore[arg-type, call-arg]\n",
        [(1, "type-ignore[arg-type]"), (1, "type-ignore[call-arg]")],
    ),
    "an-uncoded-ignore": ("x = f()  # type: ignore\n", [(1, "type-ignore")]),
    "an-ignore-inside-a-string-is-not-one": ('x = "# type: ignore"\n', []),
}


@pytest.mark.proves("architecture.type_ignores_only_fall")
@pytest.mark.parametrize(("text", "found"), FINDING_CASES.values(), ids=list(FINDING_CASES))
def test_a_type_ignore_is_counted_per_error_code(text, found):
    """An ignore comment is one finding per error code, an uncoded one its own; a string none."""
    assert [(f.line, f.rule) for f in type_ignores.findings("a.py", text)] == found
