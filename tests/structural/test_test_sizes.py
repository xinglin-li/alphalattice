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
    + [("lane_" + lane,) for lane in ("merge", "candidate", "nightly")],
)
def test_collection_resolves_owned_size_or_names_the_conflict(tmp_path, declared):
    """Collection preserves defaults, markers and registered large owners, refusing conflicts."""
    root = Path(__file__).resolve().parents[2]
    lane = declared[0][5:] if declared and declared[0].startswith("lane_") else ""
    target = tmp_path / "test_example.py"
    marks = [name for name in declared if name in {"small", "medium", "large"}]
    target.write_text(
        "import pytest\n"
        + "".join("@pytest.mark." + name + "\n" for name in marks)
        + "def test_example(): pass\n",
        encoding="utf-8",
    )
    registered = declared == ("registered",)
    if lane:
        target.write_text(
            "import pytest\n"
            "@pytest.mark.small\ndef test_small(): pass\n"
            "def test_medium(): pass\n"
            "@pytest.mark.large\ndef test_large(): pass\n"
        )
        mandatory = tmp_path / "tests/structural/test_mandatory.py"
        mandatory.parent.mkdir(parents=True)
        mandatory.write_text(
            "import pytest\n"
            # Separate the newline escape from the generated decorator text.
            "@pytest.mark.large\ndef test_mandatory(): pass\n"
        )
    if registered:
        target = str(root / "tests/portfolio_strategy_lab/test_strategy_activation.py")
        target += "::test_the_daily_update_resumes_its_deferral_then_goes_on_to_the_next_session"
    script = (
        "import pytest,json,sys\nclass Reader:\n"
        " def pytest_collection_finish(self,session):\n"
        "  rows=[dict(i.user_properties, markers=sorted({m.name for m in i.iter_markers()\n"
        "         if m.name in {'small','medium','large'}})) for i in session.items]\n"
        "  print('SIZE_ITEMS='+json.dumps(rows))\n"
        "raise SystemExit(pytest.main(sys.argv[1:],plugins=[Reader()]))\n"
    )
    command = [sys.executable, "-c", script]
    if not registered:
        command += ["-p", "tests.conftest", "-o", "markers=small\nmedium\nlarge"]
    command += ["-c", str(root / "pyproject.toml")]
    command += ["--basetemp", str(tmp_path / "child-temp")]
    command += ["--collect-only", "-q", "--no-cov", str(target)]
    if lane:
        command += ["--rootdir", str(tmp_path), "--keep-duplicates", str(target), str(mandatory)]
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, ALPHALATTICE_NETWORK_DISABLED="1", ALPHALATTICE_TEST_LANE=lane),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    if len(set(declared)) > 1:
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
        if not declared:
            from devtools.architecture.test_sizes import classify, load_policy

            public = tmp_path / "public-only"
            policy_file = public / "config/registries/test-sizes.json"
            policy_file.parent.mkdir(parents=True)
            policy_file.write_bytes((root / "config/registries/test-sizes.json").read_bytes())
            assert not (public / "tests").exists()
            policy = load_policy(public)
            policy["defaults"].reverse()
            scope = "tests/portfolio_strategy_lab"
            node = scope + "/test_public_control_surface.py"
            node += "::test_coverage_names_the_owner_a_request_escapes"
            assert classify(node, (), policy) == ("medium", scope.replace("/", "."), scope + "/*")
