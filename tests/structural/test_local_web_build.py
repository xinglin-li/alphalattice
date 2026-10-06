"""Fresh-checkout tests prepare the same Local Web outputs every reader serves."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import build_local_web_ui

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = (
    "workbench.html",
    "workbench.css",
    "workbench.js",
    "workbench.zh.js",
    "workbench-prelude.js",
    "workbench-manifest.json",
)


@pytest.fixture
def copied_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Only disposable source/assets move; the check's syntax boundary has its own holder."""
    assets = tmp_path / "assets"
    shutil.copytree(build_local_web_ui.SRC, assets / "workbench-source")
    shutil.copytree(build_local_web_ui.ASSETS / "fonts", assets / "fonts")
    labels = tmp_path / "labels.json"
    shutil.copyfile(build_local_web_ui.LABELS, labels)
    monkeypatch.setattr(build_local_web_ui, "ASSETS", assets)
    monkeypatch.setattr(build_local_web_ui, "SRC", assets / "workbench-source")
    monkeypatch.setattr(
        build_local_web_ui, "PARAMETERS", assets / "workbench-source/design/parameters.json"
    )
    monkeypatch.setattr(build_local_web_ui, "LABELS", labels)
    monkeypatch.setattr(build_local_web_ui, "check", lambda: None)
    build_local_web_ui.build(product=True)
    return assets


@pytest.mark.parametrize("name", OUTPUTS)
@pytest.mark.parametrize("damage", ("missing", "different"))
def test_check_reads_every_output_without_rebuilding(
    copied_assets: Path, name: str, damage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manifest alone cannot hide a removed or stale file, including the host page."""
    target = copied_assets / name
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"synthetic stale output\n")
    before = {p.name: p.read_bytes() for p in copied_assets.iterdir() if p.is_file()}
    assert build_local_web_ui.main(["--product", "--check"]) == 1
    answer = json.loads(capsys.readouterr().out)
    assert answer["failure_code"] == "local_web.assets_stale"
    assert answer["outputs"] == [name]
    assert answer["next_commands"]["build"][-1] == "--product"
    assert {p.name: p.read_bytes() for p in copied_assets.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("source", ("shell", "labels"))
def test_check_compares_against_current_sources(
    copied_assets: Path, source: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source change moves its outputs even when the old manifest is intact."""
    if source == "shell":
        shell = build_local_web_ui.SRC / "shell.html"
        shell.write_text(
            shell.read_text(encoding="utf-8") + "\n<!-- synthetic change -->\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        labels = json.loads(build_local_web_ui.LABELS.read_text(encoding="utf-8"))
        labels["labels"][0]["title"] = "Synthetic changed label"
        build_local_web_ui.LABELS.write_text(
            json.dumps(labels) + "\n", encoding="utf-8", newline="\n"
        )
    assert build_local_web_ui.main(["--product", "--check"]) == 1
    assert json.loads(capsys.readouterr().out)["outputs"]


def test_check_leaves_current_output_bytes_and_write_times_alone(copied_assets: Path) -> None:
    """A passing session does not rebuild already-current assets."""
    before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in copied_assets.iterdir()
        if p.is_file()
    }
    assert build_local_web_ui.main(["--product", "--check"]) == 0
    assert {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in copied_assets.iterdir()
        if p.is_file()
    } == before


def _test_checkout(
    tmp_path: Path, *, state: str, failing: bool = False
) -> tuple[Path, dict[str, str]]:
    """The real shared conftest and a synthetic command expose controller/worker effects."""
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    scripts = checkout / "scripts"
    tests.mkdir(parents=True)
    scripts.mkdir()
    shutil.copyfile(ROOT / "tests/conftest.py", tests / "conftest.py")
    (checkout / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-p no:cacheprovider'\n",
        encoding="utf-8",
        newline="\n",
    )
    (checkout / "source").write_text("current", encoding="utf-8", newline="\n")
    if state != "missing":
        (checkout / "output").write_text(state, encoding="utf-8", newline="\n")
    (scripts / "build_local_web_ui.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "root = Path(__file__).resolve().parents[1]\n"
        "check = '--check' in sys.argv\n"
        "with (root / 'calls.jsonl').open('a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps({'check': check, "
        "'worker': os.environ.get('PYTEST_XDIST_WORKER')}) + '\\n')\n"
        "if check:\n"
        "    current = (root / 'output').is_file() and "
        "(root / 'output').read_text() == (root / 'source').read_text()\n"
        "    raise SystemExit(0 if current else 1)\n"
        f"if {failing!r}:\n"
        "    print('synthetic build owner error')\n"
        "    raise SystemExit(2)\n"
        "(root / 'output').write_text((root / 'source').read_text(), encoding='utf-8')\n",
        encoding="utf-8",
        newline="\n",
    )
    (tests / "test_consumers.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "@pytest.mark.parametrize('reader', range(16))\n"
        "def test_read(reader):\n"
        "    assert (ROOT / 'output').read_text() == 'current'\n",
        encoding="utf-8",
        newline="\n",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        ALPHALATTICE_NETWORK_DISABLED="1",
    )
    return checkout, env


@pytest.mark.parametrize("state", ("missing", "stale", "current"))
def test_session_controller_builds_once_before_all_workers(tmp_path: Path, state: str) -> None:
    """Sixteen simultaneous readers use one controller build, or none when current."""
    checkout, environment = _test_checkout(tmp_path, state=state)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "xdist",
            "-n",
            "4",
            "--basetemp",
            str(tmp_path / "pytest"),
        ],
        cwd=checkout,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    calls = [
        json.loads(line)
        for line in (checkout / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(call["worker"] is None for call in calls)
    assert sum(not call["check"] for call in calls) == (state != "current")
    assert sum(call["check"] for call in calls) == (1 if state == "current" else 2)


def test_session_propagates_the_build_owners_error(tmp_path: Path) -> None:
    """The owner error fails the session before any reader, never a skip or passing fallback."""
    checkout, environment = _test_checkout(tmp_path, state="missing", failing=True)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--basetemp", str(tmp_path / "pytest")],
        cwd=checkout,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert run.returncode == pytest.ExitCode.USAGE_ERROR, run.stdout + run.stderr
    assert "synthetic build owner error" in run.stderr
    assert not (checkout / "output").exists()
