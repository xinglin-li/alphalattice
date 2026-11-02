"""Collection resolves one owned size and refuses contradictory declarations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "declared",
    [(), ("small",), ("large",), ("medium", "medium"), ("small", "large"), ("registered",)]
    + [(name,) for name in ("timeout", "bad-reason", "bad-limit", "duplicate", "owner-large")]
    + [("strict",)]
    + [("post_" + stage,) for stage in ("commit", "post-merge", "candidate", "nightly")]
    + [("post-kind",), ("post-structural",), ("post-repeat",)]
    + [("lane_" + lane,) for lane in ("merge", "candidate", "nightly")],
)
def test_collection_resolves_owned_size_or_names_the_conflict(tmp_path, declared):
    """Owned size declarations enforce their bounds and provenance, refusing invalid catalogs."""
    root = Path(__file__).resolve().parents[2]
    lane = declared[0][5:] if declared and declared[0].startswith("lane_") else ""
    public = tmp_path / "public-only"
    policy_file = public / "config/registries/test-sizes.json"
    policy_file.parent.mkdir(parents=True)
    policy = json.loads((root / "config/registries/test-sizes.json").read_bytes())
    policy["limits"] = dict(small=1, medium=60, large=900)
    large = policy["large"]
    faults = {
        "bad-reason": (large[0], "reason", "slow", "large_reason"),
        "bad-limit": (policy["limits"], "medium", 0, "limits"),
        "duplicate": (policy, "large", large + large[:1], "large_duplicate"),
        "owner-large": (policy["defaults"][-1], "size", "large", "default_size"),
    }
    faults |= {
        "post-kind": (policy, "post_merge_only", "tests/x.py", "rule"),
        "post-structural": (policy, "post_merge_only", ["tests/structural/test_x.py"], "rule"),
        "post-repeat": (policy, "post_merge_only", ["tests/a.py", "tests/a.py"], "rule"),
    }
    case = declared[0] if declared else ""
    post_stage = case[5:] if case.startswith("post_") else ""
    if post_stage:
        lane = post_stage if post_stage in {"candidate", "nightly"} else "merge"
        contract = "tests/portfolio_strategy_lab/test_cli_contract.py"
        policy["mandatory"] = list(dict.fromkeys([*policy["mandatory"], contract + "*"]))
        policy["post_merge_only"] = [contract]
    if case in faults:
        record, key, value, refusal = faults[case]
        record[key] = value
    if case == "timeout":
        policy["limits"]["medium"] = 0.05
    policy_file.write_text(json.dumps(policy), encoding="utf-8")
    target = tmp_path / "test_example.py"
    target.write_text(
        "import pytest\n"
        + "".join(
            "@pytest.mark." + name + "\n"
            for name in declared
            if name in {"small", "medium", "large"}
        )
        + ("@pytest.mark.timeout(0.05)\n" if case == "strict" else "")
        + 'def test_example():\n    """Exercise the resolved execution limit."""\n'
        + ("    import time; time.sleep(0.2)\n" if case == "timeout" else "    pass\n"),
        encoding="utf-8",
    )
    registered = declared == ("registered",)
    if post_stage:
        target = tmp_path / contract
        target.parent.mkdir(parents=True, exist_ok=True)
    if lane:
        header = "import pytest\n"
        target.write_text(
            header + "@pytest.mark.small\ndef test_small(): pass\n"
            "def test_medium(): pass\n"
            "@pytest.mark.large\ndef test_large(): pass\n"
        )
        mandatory = tmp_path / "tests/structural/test_mandatory.py"
        mandatory.parent.mkdir(parents=True)
        mandatory.write_text(header + "@pytest.mark.large\ndef test_mandatory(): pass\n")
    if registered:
        target = str(root / "tests/portfolio_strategy_lab/test_strategy_activation.py")
        target += "::test_the_daily_update_resumes_its_deferral_then_goes_on_to_the_next_session"
    script = (
        "import pytest,json,sys\nfrom pathlib import Path\n"
        "from devtools.testing import sizes as test_sizes\nread=test_sizes.load_policy\n"
        f"test_sizes.load_policy=lambda _:read(Path({str(public)!r}))\n"
        "class Reader:\n"
        " def pytest_collection_finish(self,session):\n"
        "  rows=[]\n  for i in session.items:\n"
        "   m=i.get_closest_marker('timeout')\n"
        "   rows.append(dict(i.user_properties, timeout_marker=m.args[0] if m else None,\n"
        "     markers=sorted({m.name for m in i.iter_markers()\n"
        "       if m.name in {'small','medium','large'}})))\n"
        "  print('SIZE_ITEMS='+json.dumps(rows))\n"
        "raise SystemExit(pytest.main(sys.argv[1:],plugins=[Reader()]))\n"
    )
    command = [sys.executable, "-c", script]
    if not registered:
        command += ["-p", "tests.conftest", "-o", "markers=small\nmedium\nlarge"]
    command += ["-c", str(root / "pyproject.toml")]
    command += ["--basetemp", str(tmp_path / "child-temp")]
    command += ["-vv"] if case == "timeout" else ["--collect-only", "-q"]
    command += ["--no-cov", str(target)]
    if lane:
        command += ["--rootdir", str(tmp_path), "--keep-duplicates", str(target), str(mandatory)]
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(
            os.environ,
            ALPHALATTICE_NETWORK_DISABLED="1",
            PYTHONPATH=str(root / "src"),
            ALPHALATTICE_TEST_LANE=lane,
            ALPHALATTICE_PIPELINE_STAGE=post_stage,
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    if case in faults:
        assert result.returncode != 0 and "test_size." + refusal in result.stderr
    elif case == "timeout":
        assert result.returncode != 0 and "Timeout" in result.stdout + result.stderr
        assert "test_example" in result.stdout + result.stderr
    elif len(set(declared)) > 1:
        assert result.returncode != 0 and "test_size.conflicting" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        rows = json.loads(
            next(
                x.removeprefix("SIZE_ITEMS=")
                for x in result.stdout.splitlines()
                if x.startswith("SIZE_ITEMS=")
            )
        )
        assert rows
        if case == "strict":
            assert rows[0]["timeout_marker"] == 0.05
            return
        if post_stage:
            expected = (
                ["large"] if post_stage == "commit" else ["large", "large", "medium", "small"]
            )
            assert sorted(r["test_size"] for r in rows) == expected
            return
        if lane:
            expected = ["large", "medium", "small"] if lane == "merge" else ["large", "large"]
            assert sorted(r["test_size"] for r in rows) == expected
            assert sum("tests/structural/" in r["test_size_declaration"] for r in rows) == 1
            return
        expected_size = "large" if registered else declared[0] if declared else "medium"
        expected_owner = "tests.portfolio_strategy_lab" if registered else "tests"
        for row in rows:
            assert row["markers"] == [row.get("test_size")]
            assert row.get("test_size") == expected_size
            assert row.get("test_size_owner") == expected_owner
            assert row.get("test_size_declaration")
            limit = 5400 if registered else policy["limits"][expected_size]
            assert row.get("test_timeout_seconds") == limit
        if not declared:
            from devtools.testing.sizes import classify, load_policy

            assert not (public / "tests").exists()
            policy = load_policy(public)
            policy["defaults"].reverse()
            scope = "tests/portfolio_strategy_lab"
            node = scope + "/test_public_control_surface.py"
            node += "::test_coverage_names_the_owner_a_request_escapes"
            expected = ("medium", scope.replace("/", "."), scope + "/*", 60)
            assert classify(node, (), policy) == expected


def test_a_declared_reach_scopes_the_merge_and_fast_lanes(tmp_path, monkeypatch, request):
    """A per_file scan runs scoped unless its own file or input changed; a cross-file check runs
    when its inputs change; an undeclared structural test and nightly stay whole."""
    from devtools.testing.collection import CHANGED
    from devtools.testing.sizes import changed_paths, included, scoped

    scoped_run = os.environ.get("ALPHALATTICE_TEST_LANE") in {"merge", "fast"}  # read once
    assert request.config.stash.get(CHANGED, None) == (changed_paths() if scoped_run else None)
    scan, closure = "tests/structural/test_scan.py::test_a", "tests/structural/test_closure.py::t"
    reason = {"owner": "tests.structural", "reason": "declared"}
    policy = {
        "mandatory": ["tests/structural/*"],
        "reach": [
            {"pattern": scan, "per_file": True, "inputs": ["src/rules.py"], **reason},
            {"pattern": closure, "inputs": ["config/release/*", "+*"], **reason},
        ],
    }
    listing = tmp_path / "changed.log"
    listing.write_text("M\tsrc/a.py\nA\tsrc/new.py\n", encoding="utf-8")
    monkeypatch.setenv("ALPHALATTICE_CHANGED_PATHS", str(listing))
    changed = changed_paths()
    assert changed == {"src/a.py", "src/new.py", "+src/new.py"}
    assert scoped(scan, policy, changed) and included(scan, "medium", "fast", policy, changed)
    assert not scoped(scan, policy, changed | {"src/rules.py"})
    assert included(closure, "large", "merge", policy, changed)
    assert not included(closure, "large", "merge", policy, {"src/a.py"})
    assert not included(closure, "medium", "fast", policy, changed)
    assert included("tests/structural/test_other.py::t", "medium", "merge", policy, changed)
    assert included(closure, "large", "nightly", policy, None) and not scoped(scan, policy, None)


def test_a_card_whose_tests_outgrow_its_source_is_refused_by_name():
    """Net tests within max(20, 30% of net source), pruning tests fall, at most 3 new functions."""
    from devtools.testing.sizes import budget_problems

    assert "tests/b.py +31" in budget_problems([("src/a.py", 100), ("tests/b.py", 31)], [], "")[0]
    assert budget_problems([("src/a.py", 100), ("tests/b.py", 30)], ["t1", "t2", "t3"], "") == []
    assert budget_problems([("tests/b.py", 20)], [], "") == []  # the floor, at zero net source
    assert "exceed +20 for net source +0" in budget_problems([("tests/b.py", 21)], [], "")[0]
    assert "4 new test functions" in budget_problems([], ["t1", "t2", "t3", "t4"], "")[0]
    assert budget_problems([("scripts/a.py", -50), ("tests/b.py", 1)], [], "")
    assert budget_problems([("scripts/a.py", -50), ("tests/b.py", -9)], [], "") == []
    assert budget_problems([("tests/b.py", 31)], ["t1", "t2", "t3", "t4"], "R1 R2") == []
    assert budget_problems([("src/a.py", 100), ("tests/README.md", 99)], [], "") == []


def test_a_quarantined_node_needs_its_owner_reason_and_date():
    """A flaky node leaves the blocking tier only with an owner, a reason and an until date."""
    from devtools.testing.sizes import load_policy, validate_policy

    row = {"node": "tests/a.py::t", "owner": "tests.a", "reason": "flaky", "until": "2026-99-99"}
    with pytest.raises(ValueError, match="quarantine_invalid"):
        validate_policy(load_policy(Path(__file__).resolve().parents[2]) | {"quarantine": [row]})


HELD = "product_host.detached_start_outlives_the_session"


def _registry_tree(tmp_path):
    """A checkout with one requirement, its holder, an unregistered twin and an unrelated test."""
    root, registries = Path(__file__).resolve().parents[2], tmp_path / "config/registries"
    registries.mkdir(parents=True)
    for name in ("test-sizes.json", "owners-map.json"):
        (registries / name).write_bytes((root / "config/registries" / name).read_bytes())
    registry = json.loads((root / "config/registries/requirements.json").read_bytes())
    row = {"owner": "alphalattice.control.product_host", "layer": "integration", "seconds": 20}
    registry["entries"] = {HELD: row | {"description": "A held update runs after its session."}}
    (registries / "requirements.json").write_text(json.dumps(registry), "utf-8")
    (tmp_path / "tests/a").mkdir(parents=True)
    (tmp_path / "tests/a/test_host.py").write_text(
        "import pytest\n"
        f'@pytest.mark.proves("{HELD}")\n'
        "def test_breakaway_launch_survives(): pass\n"
        'def test_idle_expiry_keeps_owed_work():\n    """A detached Host keeps owed work."""\n'
        "def test_parquet_schema_round_trip(): pass\n",
        "utf-8",
    )
    return registry


@pytest.mark.proves("testing.find_matches_requirements_and_unregistered_tests")
def test_find_returns_requirements_with_holders_then_unregistered_tests(tmp_path, capsys):
    from scripts.tests_registry import main

    from devtools.testing.requirements import find, load, matches

    registry = _registry_tree(tmp_path)
    found = find(tmp_path, "detached host survives the session")
    assert found[0].startswith("1 requirements and 1") and "active" in found[1]
    assert "  held by tests/a/test_host.py::test_breakaway_launch_survives" in found
    assert "unregistered  tests/a/test_host.py::test_idle_expiry_keeps_owed_work" in found
    assert not any("parquet" in line for line in found)
    assert find(tmp_path, "detached host keeps work")[1].startswith(HELD)  # rows lead
    assert main(["--root", str(tmp_path), "find", "dividend ledger"]) == 0
    assert capsys.readouterr().out.startswith("Nothing matches") and matches("x", {}) == []
    registries, node = tmp_path / "config/registries", "tests/a/test_host.py::test_breakaway_launch"
    policy = json.loads((registries / "test-sizes.json").read_bytes())
    for until, state in (("2000-01-01", "active"), ("2999-01-01", "quarantined")):
        held = {"node": node + "_survives", "owner": "tests", "reason": "r", "until": until}
        (registries / "test-sizes.json").write_text(json.dumps(policy | {"quarantine": [held]}))
        assert state in find(tmp_path, "detached host survives the session")[1]
    bad = {HELD: registry["entries"][HELD] | {"layer": "system"}}
    for document in ([], registry | {"entries": []}, registry | {"entries": bad}):
        (registries / "requirements.json").write_text(json.dumps(document), "utf-8")
        with pytest.raises(ValueError, match=r"requirements\.(schema|row_invalid:product_host\.)"):
            load(tmp_path)


@pytest.mark.proves(
    "testing.unregistered_test_refused_with_its_find",
    "testing.row_admitted_or_refused_by_its_check",
)
@pytest.mark.parametrize(
    ("case", "edit", "refused"),
    [
        ("unregistered", {}, "proves nothing"),
        ("unknown", {}, "unregistered or retired"),
        ("product_host.port_reuse", {}, ""),
        ("host.port_reuse", {}, "its id is product_host.<words>"),
        ("product_host.port_reuse", {"description": "One holds. Two hold."}, "one 5-40 word"),
        ("nowhere.port_reuse", {"owner": "nowhere"}, "owns a noun"),
        ("product_host.port_reuse", {"layer": "e2e", "seconds": 600}, "(journey, large)"),
        ("product_host.owed_work", {"description": "A detached Host keeps owed work."}, "near"),
        ("product_host.port_reuse", {"layer": "e2e", "exception": "decided"}, ""),
        ("product_host.port_reuse", {"layer": "e2e", "exception": {"decided": True}}, "decision"),
        ("product_host.port_reuse", {"layer": "e2e", "exception": {"kinds": 1}}, "decision"),
        (HELD, None, "was removed"),
    ],
)
def test_a_new_test_or_row_is_admitted_or_refused_by_the_check_it_fails(
    tmp_path, case, edit, refused
):
    from devtools.testing import requirements as r
    from devtools.testing.collection import gate

    registry, node = _registry_tree(tmp_path), "tests/a/test_host.py::test_host_outlives_session"
    if case in {"unregistered", "unknown"}:
        test = r.Holder(node, ("x.y",) * (case == "unknown"), node.split("::")[1])
        found = r.registration(tmp_path, [test], registry["entries"])
        assert refused in found[0] and found[-1].lstrip().startswith("Next:")
        assert any(line.strip().startswith(HELD) for line in found)
        if case == "unknown":
            return
        kept, other = (r.Holder(f"t.py::{c}::test_anchor", (), "", "d") for c in ("TestA", "TestB"))
        assert r.fresh([kept], [kept, other]) == [other]  # a basename never matches
        moved, copy = (kept._replace(node=f"{f}::TestA::test_anchor") for f in ("u.py", "v.py"))
        assert r.fresh([kept], [moved]) == [] and r.fresh([kept], [moved, copy]) == [copy]
        suite, outside = tmp_path / "tests", [(node, [], tmp_path / "child/test_host.py")]
        gate("merge", outside, {}, suite)  # a child session's own tests are not the suite's
        with pytest.raises(ValueError, match=r"unregistered:tests/a/test_host\.py"):
            gate("merge", [(node, [], suite / "a/test_host.py")], {}, suite)
        gate("merge", [(node, [], suite / "a/test_host.py")], r.legacy(tmp_path), suite)  # public
        r.shrink(tmp_path, seed=True)
        with pytest.raises(ValueError, match="legacy_seeded"):  # seeded once, then only shrinks
            r.shrink(tmp_path, seed=True)
        (tmp_path / r.LEGACY).unlink()
        (tmp_path / r.PRIVATE_MARKER).parent.mkdir(parents=True)
        (tmp_path / r.PRIVATE_MARKER).write_text("{}", "utf-8")
        with pytest.raises(ValueError, match="legacy_missing"):  # a private checkout needs it
            r.legacy(tmp_path)
        return
    base, entries = json.dumps(registry), registry["entries"]
    row = entries[HELD] | {"description": "A restarted Host reuses its last free port."}
    decided = {"exception": {"kinds": ["journey"], "why": "w", "decided": "2026-10-24: w"}}
    if edit is None:
        del entries[case]
    else:
        entries[case] = row | edit | (decided if edit.get("exception") == "decided" else {})
    found = r.admission(tmp_path, entries, json.loads(base)["entries"])
    assert (refused in " ".join(found)) if refused else found == []
