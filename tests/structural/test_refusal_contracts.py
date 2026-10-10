"""Source contracts and whole owner refusal sentences."""

import ast
import json
import re
from pathlib import Path

from tests.structural.refusal_support import refusal_codes

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


def test_every_refusal_the_client_raises_has_its_own_words() -> None:
    """Every refusal the client raises has its own words."""

    from alphalattice.interface.local_application import cli_contract

    codes = refusal_codes()["client"]
    generic = cli_contract.client_refusal("no_owner.unworded")
    assert "local_client.answer_names_two_books" in codes
    assert not sorted(c for c in codes if cli_contract.client_refusal(c) == generic)


def test_every_task_stop_sentence_fits_its_bound_and_reads_whole() -> None:
    """Every task stop sentence fits its bound and reads whole."""

    from alphalattice.control.guanyin.data.workspace_maintenance import (
        maintenance_failure_detail,
    )
    from alphalattice.control.product_host.composition.plain_refusals import (
        STOP_WORDS_BOUND,
        explain,
    )
    from alphalattice.control.product_host.composition.task_recovery import stop_detail

    registered = {"entries": refusal_codes()["all"]}
    worded = {}
    for code in registered["entries"]:
        try:
            words = explain(code).get("detail")
        except (KeyError, TypeError, ValueError):
            continue
        if words:
            worded[code] = words
    assert len(worded) > 50
    read_whole = refusal_codes()["read_whole"]
    for code in read_whole:
        worded[code] = explain(code)["detail"]
        assert stop_detail("research_experiment", code, "TASK_CONTROL") == worded[code], code
    long = {code: len(words) for code, words in worded.items() if len(words) > STOP_WORDS_BOUND}
    assert long == {}, long
    data = refusal_codes()["maintenance_dict_keys"]
    assert data and all(len(maintenance_failure_detail(code)) <= STOP_WORDS_BOUND for code in data)


def test_every_product_text_naming_the_workspace_flag_is_a_kept_full_form() -> None:
    """Every product text naming the workspace flag is a kept full form."""

    kept = {
        ("control/product_host/composition/entry.py", "serve"): "the launcher's own option",
        ("control/product_host/composition/evidence_authority_setup.py", "<module>"): (
            "its words: the launcher receives only --workspace"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_parser"): (
            "the source setup script's own option"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "setup_arguments"): (
            "the running Host binds the setup script to its own workspace"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "within_delegation"): (
            "the setup script's defaults are read for the same bound workspace"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_acquisition_command"): (
            "the source setup script's own command, a script, not an object and an action; "
            "its acquisition offered again at one cutoff"
        ),
        ("control/product_host/composition/evidence_authority_setup.py", "_import_command"): (
            "the source setup script's own command: a recorded import's check and way on"
        ),
        ("control/product_host/composition/evidence_source_ways.py", "evidence_setup"): (
            "the source setup's script commands"
        ),
        ("control/product_host/composition/evidence_source_ways.py", "source_ways"): (
            "the official consent way: the person's own command, in their shell"
        ),
        ("control/product_host/composition/goals.py", "goal_prompt.command"): (
            "a /goal prompt opens another session's initial phase"
        ),
        ("control/product_host/composition/web_launcher.py", "<module>"): (
            "the Local Web launcher script's usage"
        ),
        ("control/product_host/composition/web_launcher.py", "main"): (
            "the Local Web launcher script's own option"
        ),
        ("evidence/alternative_evidence/runtime/policy.py", "<module>"): (
            "the source setup script's command template"
        ),
        ("interface/local_application/cli.py", "<module>"): "`session bind`'s help",
        ("interface/local_application/cli.py", "_command"): "serve's arguments to its launcher",
        ("interface/local_application/cli.py", "_global_options"): "the option and its help",
        ("interface/local_application/cli.py", "_repair"): (
            "a malformed --workspace's repair, the unexpected case"
        ),
        ("interface/local_application/cli_contract.py", "<module>"): (
            "the unbound refusals' way on and the binding's words"
        ),
        ("interface/local_application/cli_contract.py", "entry"): (
            "the full form, for every workspace but the bound one"
        ),
        ("interface/local_application/client.py", "_write_bundle"): (
            "where an entry's workspace ends, never printed"
        ),
        ("interface/local_application/native_setup.py", "main"): "the bridge setup's bind option",
        (
            "interface/local_application/portfolio_research.py",
            "LocalPortfolioResearchService._manifest",
        ): "an export's sealed reproduction command",
        ("investment/portfolio_strategy_lab/application/contracts.py", "export_command"): (
            "an export's sealed reproduction command"
        ),
    }
    source = SCRIPT.parents[1] / "src" / "alphalattice"
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, owner: str, relative: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = owner
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = child.name if owner == "<module>" else f"{owner}.{child.name}"
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if re.search(r"--workspace(?![-\w])", child.value):
                    found.add((relative, name))
                continue
            visit(child, name, relative)

    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source).as_posix()
        visit(ast.parse(path.read_text(encoding="utf-8")), "<module>", relative)
    assert found == set(kept), (sorted(found - set(kept)), sorted(set(kept) - found))
    words = json.loads(
        (source / "interface/local_application/refusal_words.json").read_text("utf-8")
    )
    # backup restore's --workspace-id is a distinct option, not the global workspace flag.
    naming = {
        code for code, entry in words.items() if re.search(r"--workspace(?![-\w])", entry["detail"])
    }
    # The unexpected case's way on: a folder that holds no workspace names the right one.
    full_form_refusals = {
        "research_workspace.manifest_unreadable": "the folder cannot identify its workspace",
        "native_bridge.not_bound": "the initial session bind must explicitly name its workspace",
        "native_usage.read_limit_exceeded": (
            "unbind removes the implicit workspace; changing optional usage requires a fresh "
            "bind with that exact workspace, not a clean command against an absent binding"
        ),
        "native_bridge.existing_configuration_differs": (
            "changing this Session's scope first removes its implicit workspace"
        ),
        "native_bridge.binding_ambiguous": (
            "an unnamed Session cannot select another Session's implicit workspace"
        ),
        "native_bridge.binding_invalid": "an invalid record supplies no implicit workspace",
        "native_bridge.binding_limit_exceeded": (
            "an unbound Session can continue with an explicit workspace without replacing records"
        ),
        "native_bridge.binding_path_invalid": (
            "an unsafe record path supplies no implicit workspace"
        ),
        "native_bridge.binding_unreadable": "an unreadable record supplies no implicit workspace",
        "native_bridge.workspace_mismatch": (
            "another workspace cannot borrow this Session's binding; rebinding names its new scope"
        ),
    }
    assert naming == set(full_form_refusals)


def test_every_research_strategy_refusal_has_its_owner_words():
    """Every research strategy refusal in the owner has its declared door words."""
    from alphalattice.interface.local_application.cli_contract import refusal_words

    source = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/control/product_host/data_preparation/research_strategy.py"
    ).read_text(encoding="utf-8")
    internal = {
        "research_strategy.evidence_binding_mismatch",
        "research_strategy.evidence_invalid",
        "research_strategy.materialized_inputs",
        "research_strategy.plan_invalid",
        "research_strategy.retry_requested",
        "research_strategy.stage_unknown",
        "research_strategy.task_contract_invalid",
    }
    reachable = set(re.findall('"(research_strategy\\.[a-z_]+)', source)) - internal
    assert reachable and (not [c for c in sorted(reachable) if not refusal_words(c)]), reachable
