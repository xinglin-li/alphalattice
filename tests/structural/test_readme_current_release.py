import re
import tomllib
from pathlib import Path

import pytest

TEST_SCOPE = "whole-tree"
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.proves("pipeline.readmes_name_project_release")
def test_both_readmes_name_the_current_release() -> None:
    """The current release heading, sentence, notes link and date follow pyproject."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    anchor = f"docs/public-source/release-notes.md#{version.replace('.', '')}"
    date = re.compile(r"(?m)^Last checked: \d{4}-\d{2}-\d{2} against " + re.escape(version) + "$")
    checks = {}
    for name, heading, current in (
        ("README.md", f"## {version}, now", f"**{version} is the current release.**"),
        ("README.zh-CN.md", f"## {version}\uff1a现已提供", f"**当前版本为 {version}。**"),
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        checks[name] = (heading in text, current in text, anchor in text, bool(date.search(text)))
    assert all(all(result) for result in checks.values()), checks
