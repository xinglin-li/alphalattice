"""Every product script that writes a location it chooses answers its failure in words (V539).

A product script is one the product names to a person or an agent: in its guide (AGENTS.md,
README.md, docs/public-source), its Skill, or its own answers (a setup command or an error that
names it). Each is classified here. One that writes a location its caller did not name -- a
default outside the checkout, or its own place inside it -- names the behaviour test proving
that a write it cannot make is refused in words with its way on
(`tests/workspace_maintenance/test_product_script_refusals.py`); one that writes only where its
caller names says so. A product script named anywhere new fails this pin until it is classified.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEHAVIOUR = ROOT / "tests/workspace_maintenance/test_product_script_refusals.py"
NAMED = re.compile(r"scripts/([a-z0-9_]+\.py)")
NAMED_MODULES = {
    "alphalattice.control.product_host.composition.web_launcher": "run_local_portfolio_web.py",
    "alphalattice.interface.local_application.retrieval_environment": (
        "create_retrieval_environment.py"
    ),
    "alphalattice.control.product_host.composition.retrieval_pack_setup": (
        "install_retrieval_pack.py"
    ),
    "alphalattice.control.product_host.composition.evidence_authority_setup": (
        "materialize_evidence_cro_authority.py"
    ),
}
"""Installed spellings of the same setup and launcher entries, counted with their checkout shims."""

_ENVIRONMENTS = "test_the_retrieval_environment_setup_answers_its_failure_in_words"
WRITERS = {
    "build_local_web_ui.py": "test_the_local_web_build_answers_an_asset_it_cannot_write",
    "create_gpu_environment.py": _ENVIRONMENTS,
    "create_retrieval_environment.py": _ENVIRONMENTS,
    "install_retrieval_pack.py": "test_the_pack_installer_refuses_an_unwritable_store_by_name",
    "materialize_claude_host.py": "test_the_claude_host_files_answer_a_write_they_cannot_make",
    "native_research.py": "test_the_native_bridge_answers_files_it_cannot_reach_in_words",
    # The backup a request makes writes the backup root, the application data by default.
    "run_alphalattice.py": "test_a_backup_refuses_an_unwritable_root_by_name_and_words_it",
}
"""Each product script that writes a location it chooses, and the test of its refusal."""

NAMED_ONLY = {
    "materialize_evidence_cro_authority.py": "the workspace `--workspace` names, and under it",
    "run_local_portfolio_web.py": "the workspace `--workspace` names",
}
"""Each product script that writes only where its caller names."""

SETUPS = {
    "build_local_web_ui.py",
    "create_gpu_environment.py",
    "create_retrieval_environment.py",
    "install_retrieval_pack.py",
    "materialize_claude_host.py",
    "materialize_evidence_cro_authority.py",
    "native_research.py",
}
"""The setup entries, including the caller-selected workspace installer."""

LAUNCHERS = {
    "run_alphalattice.py": "The product CLI, whose operation boundary words owner refusals; "
    "its backup writer remains covered by the writer pin above. It is not a setup script.",
    "run_local_portfolio_web.py": "A development server launcher, not an installer; its source "
    "admission and session operations belong to the maintained product entries.",
}
"""The deliberate non-setup entries among the product's named scripts."""

ADVISORY = {
    "native_research.py:hook": "Host lifecycle observations are advisory: malformed or unbound "
    "input returns the native hook protocol's empty reply with exit 0. It grants no authority.",
}

VERIFICATION_TOOLS = {
    "broad_ensemble_research_closure.py",
    "check_alpha_verification_impact.py",
    "check_playpen.py",
    "run_broad_ensemble_live_model_closure.py",
    "run_heterogeneous_live_score_closure.py",
    "run_monthly_alpha_refit_research.py",
}
"""File selectors in the verification router, not setup requests or product offers.

Their references are held to that one metadata owner below. A guide or another owner
offering one cannot inherit this exception.
"""


def _strings(path: Path) -> list[str]:
    """A module's string literals, its docstrings left out: what it can answer with."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    documented = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in documented
    ]


def _script_references() -> dict[str, set[Path]]:
    guides = [
        ROOT / "AGENTS.md",
        ROOT / "README.md",
        *sorted((ROOT / "docs/public-source").glob("*.md")),
        *sorted((ROOT / ".agents/skills").rglob("*.md")),
    ]
    references: dict[str, set[Path]] = {}
    for path in guides:
        for name in NAMED.findall(path.read_text(encoding="utf-8")):
            references.setdefault(name, set()).add(path)
    answering = [
        ROOT / "scripts/run_alphalattice.py",
        *sorted((ROOT / "src/alphalattice").rglob("*.py")),
    ]
    for path in answering:
        for value in _strings(path):
            for name in NAMED.findall(value):
                references.setdefault(name, set()).add(path)
            for module, name in NAMED_MODULES.items():
                if value == module:
                    references.setdefault(name, set()).add(path)
    return references


def _product_scripts() -> set[str]:
    return set(_script_references()) - VERIFICATION_TOOLS


def test_every_product_script_that_writes_a_location_it_chooses_answers_in_words() -> None:
    """requirement (V539, the class): the product scripts are exactly those classified, and
    each writer names a behaviour test of its worded refusal that exists."""

    references = _script_references()
    assert set(references) == set(WRITERS) | set(NAMED_ONLY) | VERIFICATION_TOOLS
    metadata_owner = ROOT / "src/alphalattice/investment/alpha_research/verification/impact.py"
    assert all(references[name] == {metadata_owner} for name in VERIFICATION_TOOLS)
    product_scripts = set(references) - VERIFICATION_TOOLS
    assert product_scripts == set(WRITERS) | set(NAMED_ONLY)
    assert not set(WRITERS) & set(NAMED_ONLY)
    for script in (*WRITERS, *NAMED_ONLY):
        assert (ROOT / "scripts" / script).is_file(), script
    test_nodes = [
        node
        for node in ast.parse(BEHAVIOUR.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    ]
    tests = {node.name for node in test_nodes}
    assert set(WRITERS.values()) <= tests, set(WRITERS.values()) - tests
    assert product_scripts == SETUPS | set(LAUNCHERS)
    assert not SETUPS & set(LAUNCHERS)
    assert all(LAUNCHERS.values()) and all(ADVISORY.values())
    # Each setup entry answers an unexpected failure, and every setup refusal claims only what
    # it carries with a way on fitting its mode (V590): both behaviour tests run every entry.
    for name in (
        "test_every_product_setup_script_words_an_unexpected_failure",
        "test_every_setup_refusal_claims_only_what_it_carries_and_its_way_on_fits_its_mode",
    ):
        assert name in tests, name
        behaviour = next(node for node in test_nodes if node.name == name)
        cases = next(
            decorator.args[1]
            for decorator in behaviour.decorator_list
            if isinstance(decorator, ast.Call)
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and decorator.args[0].value == "script"
        )
        assert isinstance(cases, ast.List)
        assert {str(ast.literal_eval(value)) + ".py" for value in cases.elts} == SETUPS, name
    owners = {
        "native_research.py": "src/alphalattice/interface/local_application/native_setup.py",
        "create_retrieval_environment.py": (
            "src/alphalattice/interface/local_application/retrieval_environment.py"
        ),
        "install_retrieval_pack.py": (
            "src/alphalattice/control/product_host/composition/retrieval_pack_setup.py"
        ),
        "materialize_evidence_cro_authority.py": (
            "src/alphalattice/control/product_host/composition/evidence_authority_setup.py"
        ),
    }
    for script in SETUPS:
        source = ROOT / owners.get(script, "scripts/" + script)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        # Expected exceptions have owner words; an unanticipated Exception also passes the
        # sanitizer. Help/usage SystemExit and a person's KeyboardInterrupt are not failures.
        assert any(
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "Exception"
            for node in ast.walk(main)
        ), script
