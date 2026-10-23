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
    case = declared[0] if declared else ""
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
        "from devtools.architecture import test_sizes\nread=test_sizes.load_policy\n"
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
            from devtools.architecture.test_sizes import classify, load_policy

            assert not (public / "tests").exists()
            policy = load_policy(public)
            policy["defaults"].reverse()
            scope = "tests/portfolio_strategy_lab"
            node = scope + "/test_public_control_surface.py"
            node += "::test_coverage_names_the_owner_a_request_escapes"
            expected = ("medium", scope.replace("/", "."), scope + "/*", 60)
            assert classify(node, (), policy) == expected
