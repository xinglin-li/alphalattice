"""Discovered content-store absence boundaries and the regression classes they hold."""

from pathlib import Path

import pytest

from devtools.architecture.structural import content_store_absence_violations

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", ["comparison", "translator", "index"])
def test_the_pin_refuses_each_erasing_reader_class(case: str) -> None:
    """CONTRACT: discover new comparisons, helper translators and index wrappers."""
    header = (
        "from alphalattice.control.workspace_runtime.content_store "
        "import ContentAddressedStoreError\n"
    )
    if case == "comparison":
        broken = """def read(code):
    if code == 'content_store.artifact_tampered':
        return 'changed'
"""
        fixed = broken + "    if code == 'content_store.artifact_missing':\n        return 'lost'\n"
    elif case == "translator":
        broken = """def translate(error: ContentAddressedStoreError):
    return 'portfolio_strategy_lab.artifact_tampered'
def read(store):
    try:
        return store.load_document()
    except ContentAddressedStoreError as error:
        raise ValueError(translate(error)) from error
"""
        fixed = broken.replace(
            "    return 'portfolio_strategy_lab.artifact_tampered'",
            "    if str(error).partition(':')[0] == 'content_store.artifact_missing':\n"
            "        return 'portfolio_strategy_lab.artifact_absent'\n"
            "    return 'portfolio_strategy_lab.artifact_tampered'",
        )
    else:
        broken = """def read(store):
    tampered = 'portfolio_application.result_index_tampered'
    try:
        return store.load_model()
    except (ContentAddressedStoreError, KeyError) as error:
        raise ValueError(tampered) from error
"""
        fixed = broken.replace(
            "        raise ValueError(tampered) from error",
            "        if str(error).partition(':')[0] == 'content_store.artifact_missing':\n"
            "            raise\n"
            "        raise ValueError(tampered) from error",
        )
    assert content_store_absence_violations({"new_reader.py": header + broken})
    assert content_store_absence_violations({"new_reader.py": header + fixed}) == ()


def test_content_store_readers_account_for_missing_artifacts() -> None:
    """CONTRACT: discover absence-to-corruption classifications over the whole product."""
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in (ROOT / "src/alphalattice").rglob("*.py")
    }
    assert content_store_absence_violations(sources) == ()
