"""Installed book names and exact routes through their public Workbench consumers."""

from pathlib import Path

import pytest

from tests.portfolio_strategy_lab.local_web_support import run_node


@pytest.mark.parametrize("case", ["history-names", "evidence-book-choices"])
def test_installed_book_consumers_keep_names_and_exact_references(case):
    """BEHAVIOUR F30/: labelled owner answers, real readers and former-source controls."""
    root = Path(__file__).resolve().parents[2]
    source = root / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
    run_node(
        [str(Path(__file__).with_name("workbench_book_names.cjs")), str(source), case],
        required=True,
        cwd=root,
        check=True,
        timeout=30,
    )
