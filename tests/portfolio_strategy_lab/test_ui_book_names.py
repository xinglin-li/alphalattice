"""Installed book names and exact routes through their public Workbench consumers."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("workbench_build")


@pytest.mark.parametrize("case", ["history-names", "evidence-book-choices"])
def test_installed_book_consumers_keep_names_and_exact_references(case):
    """BEHAVIOUR F30/: labelled owner answers, real readers and former-source controls."""
    root = Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    assert node, "Node.js development runtime required"
    source = root / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
    subprocess.run(
        [node, str(Path(__file__).with_name("workbench_book_names.cjs")), str(source), case],
        cwd=root,
        check=True,
        timeout=30,
    )
