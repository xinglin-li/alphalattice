from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import get_args

from alphalattice.control.product_host.composition import saved_object_readback
from alphalattice.control.product_host.composition.evidence_review_application import SETUP_OPTIONS
from alphalattice.control.task_control.registry import TASK_CONTROL_DATABASE_FILENAME
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.panels.semantic_index import (
    FeaturePanelSemanticIndexService,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    FeaturePanelSnapshotPublisher,
)
from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.interface.local_application import client as client_module
from tests.researcher_methodology_surface.alternate_capability import (
    ALTERNATE_ADAPTER_ID,
    ALTERNATE_RECIPE_SCHEMA_ID,
)
from tests.researcher_methodology_surface.external_factor_method import (
    EXTERNAL_FACTOR_ID,
    EXTERNAL_METHOD_FAMILY,
)
from tests.structural.source_shape_samples import (
    _OPTION_CASES,
    bundle_refusal_declarations,
    evidence_review_refusal_samples,
    evidence_unit_failure_codes,
    specialist_answer_bound_samples,
    workbench_source_inputs,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"

GENERIC_LAYER = (
    "src/alphalattice/investment/portfolio_strategy_lab/application/resolution.py",
    "src/alphalattice/investment/portfolio_strategy_lab/inputs/shared_lanes.py",
    "src/alphalattice/control/product_host/composition/portfolio_research_operations.py",
    "scripts/run_local_portfolio_web.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/executor.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/contracts.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/controls.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/readback.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/report_projection.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/strategy_package.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/task.py",
    "src/alphalattice/investment/portfolio_strategy_lab/application/tranche_book_execution.py",
    "src/alphalattice/investment/portfolio_strategy_lab/reporting/static.py",
    "src/alphalattice/control/product_host/composition/portfolio_application.py",
    "src/alphalattice/control/product_host/composition/strategy_calibration.py",
    "src/alphalattice/control/product_host/composition/portfolio_finalization.py",
    "src/alphalattice/control/product_host/composition/local_web_session.py",
    "src/alphalattice/interface/local_application/portfolio_research.py",
    "src/alphalattice/interface/local_application/web.py",
    "src/alphalattice/interface/local_application/dispatcher.py",
)

FORBIDDEN_IN_GENERIC_LAYER = (
    r"iw184",
    r"heterogeneous[_a-z]",
    r"[_a-z]heterogeneous",
    r"\bgate[ _]i\b",
    r"\bgate[ _]m\b",
    r"\bg[0267]_",
)

ROOT = Path(__file__).resolve().parents[2]

CASE_ROOT = Path(__file__).resolve().parent

PLAYPEN_ROOT = CASE_ROOT.parents[1]


def test_retired_counting_owners_have_no_current_harness_reader() -> None:
    """the retired tools and their importing wrappers are gone."""
    root = Path(__file__).resolve().parents[2]
    retired = {
        "d5_count",
        "d5_rule",
        "post_run_mentions",
        "rr5g0",
        "rr5h_identity",
        "rr5h_screen",
        "rr5h_seal",
        "rr5h_source_audit",
        "rr5h_window",
        "rr5i_rule",
        "instruction_preflight",
        "packet_gate",
    }
    harness = root / "scripts/agent_eval"
    assert retired.isdisjoint(path.stem for path in harness.glob("*.py"))
    violations = []
    for path in sorted(harness.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or "", *(item.name for item in node.names)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                modules = [node.value]
            for module in modules:
                if module in retired or any(
                    f"scripts.agent_eval.{name}" in module
                    or f"scripts/agent_eval/{name}.py" in module.replace("\\", "/")
                    for name in retired
                ):
                    violations.append((path.relative_to(root).as_posix(), node.lineno, module))
    assert violations == []
    assert not (root / "config/agent-eval/count-rule-v588-2.yaml").exists()
    for path in sorted((root / "config/agent-eval").glob("*.yaml")):
        text = path.read_text("utf-8")
        assert "zero-count gate" not in text, path
        assert "counts only" not in text, path
        assert "instruction_preflight" not in text, path


def test_scripts_and_tests_have_no_machine_absolute_paths_outside_literal_fixtures():
    root = Path(__file__).resolve().parents[2]
    # Test functions and declared pytest fixtures may contain synthetic literal
    # inputs. Live helpers/module constants and every script may not name a machine.
    machine = re.compile(
        r"[A-Za-z]:[\\/]+(?:Users[\\/]+|[^\n]*alphalattice-internal)"
        r"|/(?:home|Users)/[A-Za-z0-9_.-]+/",
        re.I,
    )
    violations = []
    code_formats = {".py", ".js", ".cjs", ".mjs", ".ps1", ".sh", ".yaml", ".yml", ".json"}
    for directory in ("scripts", "tests"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix not in code_formats:
                continue
            if any(
                part.startswith(".tmp") or part in {".scratch", "__pycache__"}
                for part in path.parts
            ):
                continue
            text = path.read_text("utf-8")
            if path.suffix != ".py":
                # Data files in the explicit fixture directory are literal
                # test inputs; live scripts and browser helpers are checked.
                if directory == "tests" and "fixtures" in path.relative_to(root).parts:
                    continue
                if machine.search(text):
                    violations.append((path.relative_to(root).as_posix(), None))
                continue
            tree = ast.parse(text)
            parents = {
                child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
            }
            for ordinal, node in enumerate(ast.walk(tree)):
                if (
                    not isinstance(node, ast.Constant)
                    or not isinstance(node.value, str)
                    or not machine.search(node.value)
                ):
                    continue
                current, fixture = node, False
                while current in parents:
                    current = parents[current]
                    if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fixture = current.name.startswith("test_") or any(
                            "fixture" in ast.unparse(decorator)
                            for decorator in current.decorator_list
                        )
                        if fixture:
                            break
                if directory != "tests" or not fixture:
                    violations.append((path.relative_to(root).as_posix(), ordinal))
    assert violations == []  # Paths/ordinals only; never print matched source text.


def test_a_fixed_fit_measures_its_training_error_unless_the_plan_reads_none_source_shape() -> None:
    """Only model renewal explicitly opts out of measuring training error."""

    src = Path(__file__).resolve().parents[2] / "src"
    opting_out = sorted(
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if re.search(r"training_error=None", path.read_text(encoding="utf-8"))
    )
    assert opting_out == ["alphalattice/investment/alpha_research/scores/model_renewal.py"]


def test_every_fit_plan_call_site_states_its_protocol() -> None:
    """Every fit plan call site states its protocol."""

    root = Path(__file__).resolve().parents[2] / "src"
    untold: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if name != "build_alpha_model_fit_plan":
                continue
            if not any(keyword.arg == "protocol" for keyword in node.keywords):
                untold.append(f"{path.relative_to(root)}:{node.lineno}")
    assert untold == [], untold


def test_every_specialist_bundle_slice_has_a_named_disposition() -> None:
    """Every slice in the two renderers and packer has a named reading disposition."""
    sources = {
        "cro": "oversight/chief_risk_officer/decision/views.py",
        "analyst": "evidence/alternative_evidence/analysis/views.py",
        "packer": "protocols/actor_execution/bundles.py",
    }
    reviewed = {
        ("cro", "_finding_lines", "finding.affected_entities[1:]"): "first issuer is the heading",
        ("cro", "_index", "['F1', 'F2'][:max(1, min(2, len(dossier.findings)))]"): "answer example",
        ("cro", "_index", "coverage.unavailable_reasons[:MAXIMUM_COVERAGE_LINES]"): "coverage file",
        ("cro", "_index", "missing[:MAXIMUM_COVERAGE_LINES]"): "coverage file",
        ("cro", "_index", "unreported[:MAXIMUM_UNREPORTED_NAMES]"): "coverage file",
        ("analyst", "differing_words", "right[j1:j2]"): "complete differing words",
        (
            "analyst",
            "differing_words",
            "right[max(0, j1 - 2):j1]",
        ): "context beside differing words",
        (
            "analyst",
            "differing_words",
            "list(places.values())[:MAXIMUM_DIFFERENCE_PLACES]",
        ): "full excerpt below",
        ("analyst", "_shingles", "words[index:index + 3]"): "similarity computation",
        ("packer", "_fit", "word[:limit]"): "next piece below",
        ("packer", "_fit", "word[limit:]"): "remaining pieces below",
    }
    found = set()

    def visit(node: ast.AST, source: str, owner: str = "module") -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            owner = node.name
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            found.add((source, owner, ast.unparse(node)))
        for child in ast.iter_child_nodes(node):
            visit(child, source, owner)

    root = Path(__file__).resolve().parents[2] / "src/alphalattice"
    for source, path in sources.items():
        visit(ast.parse((root / path).read_text(encoding="utf-8")), source)
    assert found == set(reviewed), (found - reviewed.keys(), reviewed.keys() - found)


def test_every_bundle_manual_refusal_registers_its_words_and_way_on_source_shape() -> None:
    """Bundle manual refusals declare their complete vocabulary."""

    bundle_refusal_declarations()


def test_every_dossier_and_bundle_refusal_on_the_review_route_has_words_and_a_way_on_shape() -> (
    None
):
    """Review refusal declarations include every outcome and internal refusal."""

    evidence_review_refusal_samples()


def test_every_evidence_unit_failure_has_words_and_a_way_on_source_shape() -> None:
    """Evidence unit failure codes cover the source vocabulary."""

    evidence_unit_failure_codes()


def test_the_setup_offer_and_its_refusals_read_one_table_of_what_each_option_must_hold_shape() -> (
    None
):
    """Every setup refusal option has a real refused-input case and a public offer."""

    from scripts import materialize_evidence_cro_authority as setup

    tree = ast.parse(Path(setup.__file__).read_text(encoding="utf-8"))
    refused = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "OptionRefused"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
    }
    assert refused <= set(_OPTION_CASES) and refused <= set(SETUP_OPTIONS)


def test_every_coverage_floor_is_judged_by_one_rule_that_leaves_quiet_holdings_out_shape() -> None:
    """Coverage floors have one judge and no competing hand computation."""

    root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    judges: dict[str, bool] = {}
    by_hand: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "sources_short":
                    judges[f"{relative}::{function.name}"] = any(
                        keyword.arg == "quiet" for keyword in node.keywords
                    )
                parts = list(ast.walk(node)) if isinstance(node, ast.Compare) else []
                divides = any(isinstance(p, ast.BinOp) and isinstance(p.op, ast.Div) for p in parts)
                if divides and any(
                    (isinstance(p, ast.Name) and p.id == "floor")
                    or (isinstance(p, ast.Attribute) and p.attr == "minimum_entity_coverage")
                    for p in parts
                ):
                    by_hand.add(f"{relative}::{function.name}")
    composition = "control/product_host/composition"
    assert judges == {
        f"{composition}/evidence_authority_setup.py::_materialize": True,
        f"{composition}/evidence_review_application.py::units_short_of_sources": False,
        "evidence/alternative_evidence/runtime/task_adapter.py::_execute_stage": True,
    }
    assert by_hand == {"evidence/alternative_evidence/runtime/coverage.py::sources_short"}


def test_every_bound_a_specialist_answer_sets_is_screened_and_stated_source_shape() -> None:
    """Each specialist answer field names its bound in the owner documentation."""

    from alphalattice.evidence.alternative_evidence.analysis import views as analyst_views
    from alphalattice.evidence.alternative_evidence.analysis.contracts import (
        AlternativeEvidenceAnalystAnswer,
    )
    from alphalattice.oversight.chief_risk_officer.decision import views as cro_views
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        PortfolioReviewAnswer,
    )

    cro, analyst = specialist_answer_bound_samples()
    for model, screened, views in (
        (PortfolioReviewAnswer, cro, cro_views),
        (AlternativeEvidenceAnalystAnswer, analyst, analyst_views),
    ):
        readme = inspect.getsource(views)
        for name, (constant, _number) in screened.items():
            assert f"`{name}`" in readme and constant in readme, (model, name)


def test_feature_engine_does_not_query_diagnostic_compatibility_view() -> None:
    feature_root = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "alphalattice"
        / "foundation"
        / "feature_engine"
    )
    pattern = re.compile(r"\bFROM\s+feature_ineligibility\b", re.IGNORECASE)
    offenders = [
        path
        for path in feature_root.glob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_feature_engine_mutation_does_not_query_diagnostic_view() -> None:
    source = inspect.getsource(FeatureStateRepository.upsert_feature_materialization)
    assert "FROM feature_ineligibility\n" not in source


def test_factor_cutover_has_no_cli_writer_or_legacy_mandate_layer() -> None:
    playpen_root = Path(__file__).resolve().parents[2]
    runner = (playpen_root / "scripts" / "run_factor_research_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "--admit-live-review" not in runner
    assert "--promote-verified-evaluation" not in runner
    assert ".publish_current(" not in runner
    # The front desk's agent terminal, which rendered the confirmation, retired with AG2.
    assert not (playpen_root / "src/alphalattice/interface/front_desk/agent_interface").exists()


def test_the_panel_driver_never_deletes_a_development_workspace_source_shape() -> None:
    """The panel driver contains no deletion call and parses before capacity binding."""

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    path = scripts / "run_feature_observation_clock_panel.py"
    spec = importlib.util.spec_from_file_location("run_feature_observation_clock_panel", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
    }
    assert not called & {"rmtree", "remove", "unlink", "rmdir", "removedirs"}
    body = next(
        node.body
        for node in ast.walk(ast.parse(Path(driver.__file__).read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    parsed_at = [
        index
        for index, node in enumerate(body)
        if any(
            isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") == "parse_args"
            for inner in ast.walk(node)
        )
    ]
    bound_at = [
        index
        for index, node in enumerate(body)
        if any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", "") == "bind_process_logical_processors"
            for inner in ast.walk(node)
        )
    ]
    assert parsed_at and bound_at and max(parsed_at) < min(bound_at)


def test_availability_and_execution_are_separate_owners_source_shape() -> None:
    """The observation-clock owner imports no execution outcome owner."""

    module = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/foundation/feature_engine/catalog/observation_clock.py"
    )
    imported = {
        node.module or ""
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("causal_outcomes" in name for name in imported)


def test_the_publisher_owns_the_semantic_index_handoff() -> None:
    """The publisher owns the semantic index handoff."""

    annotations = FeaturePanelSnapshotPublisher.__init__.__annotations__
    assert "semantic_index" in annotations, (
        "the terminal publisher must own the handoff; a Panel nobody can resolve is not published"
    )
    assert annotations["semantic_index"] == "FeaturePanelSemanticIndexService | None"

    # The collaborator is the existing service, not a second writer: the default
    # constructs that one type, and `obtain` remains its only creator.

    constructed = inspect.getsource(FeaturePanelSnapshotPublisher.__init__)
    assert "FeaturePanelSemanticIndexService(resolver)" in constructed
    assert hasattr(FeaturePanelSemanticIndexService, "obtain")


def test_publication_refuses_to_return_before_the_index_resolves() -> None:
    """Publication refuses to return before the index resolves."""

    body = inspect.getsource(FeaturePanelSnapshotPublisher.publish)
    assert "self.semantic_index.obtain(" in body
    assert "find_feature_panel_semantic_index" in body
    assert "semantic_index_handoff_incomplete" in body
    # It runs after the lifecycle projection because the reader it scans is
    # ACTIVE-gated; ordering the other way would refuse its own Panel.
    assert body.index("publish_feature_panel_lifecycle_projection") < body.index(
        "self.semantic_index.obtain("
    )
    # And only for a gateway-qualified Panel. Without a Feature Input Gateway
    # admission the reader refuses by design; forcing an index there would
    # defeat that gate rather than finish a handoff, so such a Panel stays
    # unresolvable for a reason rather than for a missing artifact.
    assert "gateway_qualified" in body
    assert body.index("gateway_qualified") < body.index("self.semantic_index.obtain(")


def test_publication_refuses_rows_from_another_catalog_before_composing() -> None:
    """requirement: the wrong-catalog check happens before any value is read."""

    body = inspect.getsource(FeaturePanelSnapshotPublisher.publish)
    assert "assert_installed_catalog_rows" in body
    assert body.index("assert_installed_catalog_rows") < body.index("self._read_consistent_source(")


def test_a_completed_build_derives_its_stage_instead_of_asserting_it() -> None:
    """A completed build derives its stage instead of asserting it."""

    body = inspect.getsource(FeatureFoundationService.build)
    assert "build_stage=FeatureBuildStage.TERMINAL_PANEL_COMPLETE" not in body

    def outcome(call: ast.Call) -> str:
        named = {keyword.arg: keyword.value for keyword in call.keywords}
        if "failure_code" in named:
            return ast.unparse(named["failure_code"])
        details = call.args[3] if len(call.args) > 3 else None
        waits = isinstance(details, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "panel" for key in details.keys
        )
        return "COMPLETED, the Panel not composed" if waits else "COMPLETED"

    derived = {
        outcome(node)
        for node in ast.walk(ast.parse(textwrap.dedent(body)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FeatureBuildOutcome"
        and any(
            keyword.arg == "build_stage"
            and ast.unparse(keyword.value) == "self._durable_build_stage()"
            for keyword in node.keywords
        )
    }
    # Every blocked outcome after the base stage, and the completed ones, derive their stage
    # from durable state: the failed materialization, the sector, source-state and membership
    # refusals, completion, and a first use whose Panel waits for its qualified membership

    assert derived == {
        "failure_code",
        "'feature.membership_sector_unresolved'",
        "str(exc) if str(exc).startswith('feature.') else 'feature.panel_source_state_incomplete'",
        "str(exc) if str(exc) == 'feature.membership_admission_required' "
        "else 'feature.membership_manifest_mismatch'",
        "COMPLETED",
        "COMPLETED, the Panel not composed",
    }


def test_the_maintenance_path_completes_the_handoff_without_republishing(tmp_path: Path) -> None:
    """The maintenance path completes the handoff without republishing."""

    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )

    # ``run`` holds the cycle's connection hold and delegates the cycle body.
    assert "self._run(" in inspect.getsource(WorkspaceMaintenanceCoordinator.run)
    assert "self._run_cycle(" in inspect.getsource(WorkspaceMaintenanceCoordinator._run)
    body = inspect.getsource(WorkspaceMaintenanceCoordinator._run_cycle)
    assert "complete_semantic_index_handoff" in body
    assert "workspace_maintenance.semantic_index_handoff_failed" in body
    # Reached on the completed path, not only after a fresh publication.
    assert body.index("complete_semantic_index_handoff") > body.index(
        "_publish_snapshot_with_bounded_retry"
    )


def test_feature_panel_reader_streams_projected_batches_and_supports_concurrent_reads_shape() -> (
    None
):
    """The panel reader uses neither DuckDB nor whole-table materialization."""

    reader_tree = ast.parse(inspect.getsource(FeaturePanelReader))
    assert not any(
        isinstance(node, ast.Name) and node.id == "duckdb" for node in ast.walk(reader_tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "to_table"
        for node in ast.walk(reader_tree)
    )


def test_one_reader_names_the_session_a_request_and_its_launches_count_to_source_shape() -> None:
    """Session and launch provenance have only their declared environment readers."""

    variable = re.compile(r"environ\.get\(['\"](CODEX_THREAD_ID|CLAUDE_CODE_SESSION_ID)")
    words = ("AGENT_SESSION_VARIABLES", "CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID")
    readers: dict[str, set[str]] = {}
    for path in (SCRIPT.parents[1] / "src" / "alphalattice").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not any(word in text for word in words):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef):
                body = ast.unparse(node)
                if "AGENT_SESSION_VARIABLES" in body or variable.search(body):
                    readers.setdefault(path.name, set()).add(node.name)
    assert readers == {
        "cli_contract.py": {"agent_session", "request_provenance"},
        "client.py": {"_codex_thread"},
    }, readers


def test_every_read_of_work_planned_per_strategy_is_keyed_by_its_strategy_source_shape() -> None:
    """Every strategy readback has one reader and every caller supplies its strategy."""

    from alphalattice.control.product_host.composition import portfolio_research_operations
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperation,
        PortfolioResearchOperationRequest,
    )

    contract = PortfolioResearchOperationRequest.field_contract
    readbacks = {
        operation.replace("_PLAN", "_READBACK")
        for operation in get_args(PortfolioResearchOperation.__value__)
        if operation.endswith("_PLAN") and "strategy_package_id" in contract(operation)[0]
    }
    tree = ast.parse(inspect.getsource(portfolio_research_operations))
    table = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "_STRATEGY_READS"
    )
    assert isinstance(table, ast.Dict)
    assert {ast.literal_eval(key) for key in table.keys if key is not None} == readbacks

    def readers(node: ast.AST) -> set[str]:
        return {
            call.func.value.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "readback"
            and isinstance(call.func.value, ast.Attribute)
        }

    owners: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and ast.unparse(node.test.left) == "request.operation"
            and "self._strategy_task(request)" in ast.unparse(node)
        ):
            owners[ast.literal_eval(node.test.comparators[0])] = readers(node)
    assert owners.keys() == readbacks and all(len(owner) == 1 for owner in owners.values()), owners
    held = set().union(*owners.values())
    root = Path(portfolio_research_operations.__file__).resolve().parents[3]
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if ".readback(" not in text:
            continue
        for call in ast.walk(ast.parse(text)):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "readback"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr in held
            ):
                given = call.args[0] if call.args else None
                assert given is not None and ast.unparse(given) != "None", (path, ast.unparse(call))


def test_every_door_whose_way_on_reruns_a_plan_resumes_its_stopped_task_source_shape() -> None:
    """Every stopped data owner has a source admission, resume and answered submission route."""

    from alphalattice.control.product_host.composition import portfolio_research_operations
    from alphalattice.control.product_host.data_preparation import application as preparation
    from alphalattice.control.product_host.maintenance import data_update
    from alphalattice.interface.local_application import cli_contract

    words = json.loads(
        Path(cli_contract.__file__).with_name("refusal_words.json").read_text("utf-8")
    )
    reruns = {
        code
        for code, entry in words.items()
        if re.search(r"\b(run the plan|confirm) again\b|\bplan runs again\b", str(entry["detail"]))
    }
    owner = data_update.WorkspaceDataUpdateApplication
    resume = inspect.getsource(owner.resume_stopped)
    for code in reruns:
        if code == "workspace_data_update.retry_not_due":
            # A deferral: reopened once its retry time has passed, refused before it.
            assert "TaskLifecycle.DEFERRED" in resume and code in resume
        elif code.startswith("workspace_data_update."):
            pass
        else:
            confirm = inspect.getsource(preparation.WorkspacePreparationApplication.confirm)
            assert "allow_blocked=task.lifecycle is TaskLifecycle.BLOCKED" in confirm
            assert "self._sources_allowed()" in confirm

    def tree(module: object) -> ast.Module:
        return ast.parse(inspect.getsource(module))  # type: ignore[arg-type]

    root = Path(data_update.__file__).resolve().parents[2]
    running = {data_update.__name__}
    for path in (root / "product_host").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "data.execute_step(" in text:
            running.add(".".join(path.relative_to(root.parent.parent).with_suffix("").parts))
    commands = set()
    for name in running:
        module = __import__(name, fromlist=["_"])
        for node in ast.walk(tree(module)):
            if isinstance(node, ast.FunctionDef) and node.name == "admit":
                body = ast.unparse(node)
                if "task_control_registry" in body and "envelope" in body:
                    assert "resume_stopped(" in body, (name, node.lineno)
            if isinstance(node, ast.FunctionDef) and "data.execute_step(" in ast.unparse(node):
                # The data stages' deferral is passed through as the owner's own.
                assert "StageDisposition.DEFERRED" in ast.unparse(node), (name, node.name)
            if isinstance(node, ast.ClassDef) and node.name.endswith("Command"):
                commands.add(node.name)
    assert commands >= {"DecisionAdvancementCommand", "WorkspaceDataUpdateCommand"}, commands
    way = inspect.getsource(portfolio_research_operations.PortfolioResearchOperations._deferred_way)
    assert "DATA_UPDATE_TASK_KIND" in way and "DecisionAdvancementApplication.task_kind" in way
    commands.add(preparation.WorkspacePreparationCommand.__name__)
    door = tree(portfolio_research_operations)
    held_as = {
        id(node.value): ast.unparse(node.targets[0])
        for node in ast.walk(door)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    }
    submitted: dict[str, str] = {}
    for call in ast.walk(door):
        if not (isinstance(call, ast.Call) and ast.unparse(call.func) == "self.dispatcher.submit"):
            continue
        made = call.args[0]
        if isinstance(made, ast.Call) and ast.unparse(made.func) in commands:
            submitted[ast.unparse(made.func)] = held_as.get(id(call), ast.unparse(call))
    assert submitted.keys() == commands, submitted
    answered = {
        ast.unparse(call.args[0])
        for call in ast.walk(door)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "self._run_answer"
    }
    assert set(submitted.values()) <= answered, (submitted, answered)


def test_no_newer_plan_and_no_admitted_task_waits_behind_an_update_that_has_not_ended_shape() -> (
    None
):
    """Data deferrals, admitted work and waiting plans have their declared continuation routes."""

    from alphalattice.control.product_host.composition import (
        decision_advancement,
        portfolio_research_operations,
        research_update_automation,
    )
    from alphalattice.control.product_host.maintenance import data_update
    from alphalattice.interface.local_application import dispatcher

    root = Path(data_update.__file__).resolve().parents[2]
    deferring = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "product_host").rglob("*.py")
        if "StageDisposition.DEFERRED" in path.read_text(encoding="utf-8")
    )
    assert deferring == [
        "product_host/composition/decision_advancement.py",
        "product_host/data_preparation/application.py",
        "product_host/maintenance/data_update.py",
    ], deferring
    plans = {
        "research update": (
            # The plan's body; `plan` holds one request's completed scores around it.
            inspect.getsource(decision_advancement.DecisionAdvancementApplication._plan),
            "TaskLifecycle.DEFERRED",
            "workspace_score_source_identity(",
        ),
        "data update": (
            inspect.getsource(data_update.WorkspaceDataUpdateApplication.plan),
            "self._waiting()",
            "read_workspace_inputs(",
        ),
    }
    for owner, (source, waits, reads) in plans.items():
        assert 0 <= source.find(waits) < source.find(reads), owner
    waiting = inspect.getsource(data_update.WorkspaceDataUpdateApplication._waiting)
    assert "TaskLifecycle.DEFERRED" in waiting
    research_plan = plans["research update"][0]
    assert "self.data.plan(fresh=True)" in research_plan
    start = research_plan.find("TaskLifecycle.DEFERRED")
    waiting_pass = research_plan[start : research_plan.find("for task in tasks:", start)]
    assert "prior.package_id == package_id" in waiting_pass, waiting_pass
    assert "target" not in waiting_pass, waiting_pass
    automation = research_update_automation.ResearchUpdateAutomation
    cycle = inspect.getsource(automation._cycle)
    assert "RESEARCH_UPDATE_READBACK" in cycle and "retry_after_at" in cycle
    assert "self._retry" in cycle and "self._after" in cycle and "self._watched" in cycle
    assert '"DEFERRED"' in inspect.getsource(automation._settle)
    assert "self._watched" in inspect.getsource(automation.command_completed)
    operations = portfolio_research_operations.PortfolioResearchOperations
    way = inspect.getsource(operations._inputs_way)
    assert "read_workspace_inputs(" in way and "readiness_status" in way
    assert "running_place_holder()" in way and '"DATA_UPDATE_PLAN"' in way
    assert "DATA_UPDATE_TASK_KIND" in way and "DecisionAdvancementApplication.task_kind" in way
    assert "self._inputs_way(" in inspect.getsource(operations._execute)
    held = dispatcher.LocalBackgroundDispatcher
    assert "self._waits_its_turn(" in inspect.getsource(held._drain)
    assert "self._drive_waiting()" in inspect.getsource(held._drain)
    assert "self._drive_waiting()" in inspect.getsource(held.request_cancel)
    assert "self._waiting" in inspect.getsource(held.command_running)
    assert "self._tasks_wait()" in inspect.getsource(operations.sweep_if_due)
    assert "TaskLifecycle.DEFERRED" in inspect.getsource(operations._tasks_wait)


def test_no_generic_layer_module_names_a_strategy() -> None:
    """No generic layer module names a strategy."""

    root = Path(__file__).resolve().parents[2]
    offenders: dict[str, tuple[str, ...]] = {}
    for relative in GENERIC_LAYER:
        source = (root / relative).read_text(encoding="utf-8").lower()
        hits = tuple(value for value in FORBIDDEN_IN_GENERIC_LAYER if re.search(value, source))
        if hits:
            offenders[relative] = hits
    assert offenders == {}, offenders


def test_a_strategy_short_of_its_risk_window_names_it_and_offers_the_risk_study_source_shape() -> (
    None
):
    """Every reachable research-strategy refusal has public words."""

    from alphalattice.interface.local_application.cli_contract import refusal_words

    source = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/control/product_host/data_preparation/research_strategy.py"
    ).read_text(encoding="utf-8")
    internal = {  # integrity checks of the owner's own records, which no declaration reaches
        "research_strategy.evidence_binding_mismatch",
        "research_strategy.evidence_invalid",
        "research_strategy.materialized_inputs",
        "research_strategy.plan_invalid",
        "research_strategy.retry_requested",
        "research_strategy.stage_unknown",
        "research_strategy.task_contract_invalid",
    }
    reachable = set(re.findall(r'"(research_strategy\.[a-z_]+)', source)) - internal
    assert reachable and not [c for c in sorted(reachable) if not refusal_words(c)], reachable


def test_a_strategys_risk_history_is_judged_by_one_owner_for_its_plan_and_preparation_shape() -> (
    None
):
    """Strategy planning and preparation call the same risk-history judge."""

    from alphalattice.control.product_host.data_preparation.research_strategy import (
        ResearchStrategyPreparation,
    )
    from alphalattice.control.product_host.research_authoring import frozen_portfolio

    for owner in (
        ResearchStrategyPreparation.plan,
        frozen_portfolio.prepare_frozen_portfolio_authority,
    ):
        assert "risk_history_shortfall(" in inspect.getsource(owner), owner


def test_the_close_mark_is_the_route_the_decision_loop_actually_runs_source_shape() -> None:
    """The decision loop uses the close-mark lane at the formation session."""

    from alphalattice.capabilities.portfolio_backtesting import segments

    source = inspect.getsource(segments.run_portfolio_walk_forward_segment)
    assert "lane.value_at_decision(" in source
    assert "decision_session=workspace.formation_sessions[index]" in source
    assert "optimizer_reference = np.array(executed, copy=True)" not in source


def test_the_split_policy_owner_is_singular_and_correctly_placed() -> None:
    """The split policy owner is singular and correctly placed."""

    src = ROOT / "src" / "alphalattice" / "investment" / "portfolio_strategy_lab"
    definitions = [
        path.relative_to(ROOT).as_posix()
        for path in src.rglob("*.py")
        if "def anchored_fold_geometry(" in path.read_text(encoding="utf-8")
    ]
    assert definitions == [
        "src/alphalattice/investment/portfolio_strategy_lab/regularization/split_policy.py"
    ]

    regularization_consumers = sorted(
        path.name
        for path in (src / "regularization").glob("*.py")
        if "from .split_policy import" in path.read_text(encoding="utf-8")
    )
    assert regularization_consumers == ["contracts.py"]


def test_runtime_launchers_do_not_import_checkout_scripts_or_test_harnesses() -> None:
    """Requirement: runtime launch and saved-object readback need no checkout harness."""
    folder = Path(saved_object_readback.__file__).parent
    for name in (
        "entry",
        "web_launcher",
        "model_sandbox",
        "saved_object_readback",
        "retrieval_pack_setup",
        "evidence_authority_setup",
    ):
        tree = ast.parse((folder / f"{name}.py").read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert not any(module.split(".")[0] in {"scripts", "tests"} for module in modules), name


def test_every_wait_a_cap_ends_reads_pending() -> None:
    """Every exit of a waiter that a cap ends carries `wait_status`,
    which the outcome reads as pending; none answers a cap as a completed read."""

    tree = ast.parse(Path(client_module.__file__).read_text(encoding="utf-8"))
    exits: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            named = {key.value for key in node.keys if isinstance(key, ast.Constant)}
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "wait_event"
                    and isinstance(value, ast.Call)
                    and value.args
                    and isinstance(value.args[0], ast.Constant)
                    and value.args[0].value == "MAX_WAIT_REACHED"
                ):
                    exits.append((node.lineno, "wait_status" in named))
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "final"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "MAX_WAIT_REACHED"
        ):
            exits.append((node.lineno, any(kw.arg == "wait_status" for kw in node.keywords)))
    assert len(exits) >= 4 and all(pending for _line, pending in exits), exits


def test_the_answer_form_holds_a_draft_at_its_owners_bounds():
    """The page spells the bounds its owners refuse at (it has no read of them): the service's
    request cap, and each answer's own owner bound. The spelling must stay the owners'."""

    from alphalattice.evidence.alternative_evidence.analysis.submissions import (
        MAXIMUM_SUBMISSION_BYTES,
    )
    from alphalattice.interface.local_application.web import MAXIMUM_REQUEST_BODY_BYTES
    from alphalattice.oversight.chief_risk_officer.decision.submissions import (
        MAXIMUM_REVIEW_SUBMISSION_BYTES,
    )

    source = (
        ROOT
        / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
        / "live-review.js"
    ).read_text(encoding="utf-8")
    spelled = re.search(
        r"const REQUEST_LIMIT=(\d+)\*1024, ANSWER_LIMIT=\{analyst:(\d+)\*1024,cro:(\d+)\*1024\}",
        source,
    )
    assert spelled, "live-review.js no longer spells REQUEST_LIMIT and ANSWER_LIMIT"
    assert [int(kib) * 1024 for kib in spelled.groups()] == [
        MAXIMUM_REQUEST_BODY_BYTES,
        MAXIMUM_SUBMISSION_BYTES,
        MAXIMUM_REVIEW_SUBMISSION_BYTES,
    ]


def test_the_workbench_harnesses_pass_source_shape() -> None:
    """Workbench retention and Goal inputs enumerate their original source declarations."""

    workbench_source_inputs()


def test_seam_fixture_round_trips_one_native_shaped_event_and_one_product_event_source_shape() -> (
    None
):
    """Every refusal documented by the seam fixture is raised by an owner."""

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tests/portfolio_strategy_lab/workspace_activity_seam_fixture.json"
        ).read_text("utf-8")
    )
    source_root = Path(__file__).resolve().parents[2] / "src/alphalattice"
    owners = "".join(
        (source_root / name).read_text("utf-8")
        for name in (
            "interface/local_application/activity.py",
            "interface/local_application/failure_codes.py",
            "interface/local_application/web.py",
            "control/product_host/composition/workspace_activity.py",
            "control/product_host/composition/local_web_session.py",
            "control/product_host/composition/portfolio_research_operations.py",
            "control/observation_runtime/ledger.py",
            "control/observation_runtime/policy.py",
        )
    )
    for code, _why in fixture["typed_refusals"]:
        assert f'"{code}"' in owners, code


def test_no_product_branch_agent_or_admission_was_required_source_shape() -> None:
    """External factor methodology IDs require no product branch or admission."""

    for directory in (
        PLAYPEN_ROOT / "src" / "alphalattice" / "control" / "product_host",
        PLAYPEN_ROOT / "src" / "alphalattice" / "foundation" / "factor_research" / "agent",
    ):
        offenders = sorted(
            path.relative_to(PLAYPEN_ROOT).as_posix()
            for path in directory.rglob("*.py")
            for text in (path.read_text(encoding="utf-8"),)
            if EXTERNAL_FACTOR_ID in text or EXTERNAL_METHOD_FAMILY in text
        )
        assert offenders == [], directory.name


def test_no_product_source_names_the_alternate_method() -> None:
    """No product source names the alternate method."""

    needles = (
        ALTERNATE_ADAPTER_ID,
        ALTERNATE_RECIPE_SCHEMA_ID,
        "ShrunkDiagonal",
        "shrinkage_intensity",
    )
    offenders = [
        f"{path.relative_to(PLAYPEN_ROOT)}:{needle}"
        for path in (PLAYPEN_ROOT / "src").rglob("*.py")
        for needle in needles
        if needle in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_a_risk_run_resolves_no_factor_authority() -> None:
    """A risk run resolves no factor authority."""

    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "run_research_experiment.py").read_text(encoding="utf-8")
    body = script.split("def main(", maxsplit=1)[1]
    # Exactly one call, and it sits inside the Factor branch rather than above
    # the dispatch where every kind would pay for it.
    assert body.count("host_resolved_factor_inventory(") == 1
    assert body.index("if selected.kind == FACTOR_EXPERIMENT_KIND:") < body.index(
        "host_resolved_factor_inventory("
    )


def test_a_score_producer_cannot_state_an_execution_term_source_shape() -> None:
    """The score producer source states no execution term."""

    from alphalattice.investment.alpha_research.simple_signal import authority as producer

    source = Path(producer.__file__).read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = executable.split('"""')
    code = "".join(body[::2])
    assert "NEXT_COMMON_SESSION_OFFICIAL_OPEN" not in code
    assert "entry_offset_sessions" not in code


def test_only_the_owner_writes_the_manifest() -> None:
    """the product changes the manifest through the one write; the whole-file publisher
    is called only by its owner (tests' fixtures and lease-held scripts hold the workspace
    alone)."""

    owner = "src/alphalattice/control/product_host/composition/research_workspace.py"
    callers = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "publish_research_workspace_manifest"
            ):
                callers.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert callers and all(caller.startswith(owner + ":") for caller in callers), callers


def test_a_cancelled_successor_without_a_verified_stage_gives_its_source_back_source_shape() -> (
    None
):
    """Preparation and remediation share the superseded-preparation owner."""

    from alphalattice.control.product_host.data_preparation import application, remediation

    for owner in (
        application.WorkspacePreparationApplication.plan,
        remediation.WorkspaceDataIssueApplication.readback,
    ):
        code = inspect.getsource(owner)
        assert "superseded_preparations(" in code and "source_task_id" not in code, owner


def test_only_the_foundation_writes_its_rebuild_requirement() -> None:
    """Regression: Market Data's activation inserted the Foundation's rebuild requirement
    by its own SQL, a state the storage registry gives to Foundation; the owner defines the
    insert and the activation runs it as the caller's step, inside its transaction."""

    root = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
    writes = re.compile(r"(?:INSERT INTO|UPDATE)\s+feature_universe_rebuild_requirement")
    writers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if writes.search(path.read_text(encoding="utf-8"))
    )
    assert writers == ["foundation/research_foundation/storage/repository.py"]


def test_the_workspace_runtime_composes_the_canonical_store() -> None:
    """The workspace runtime composes the canonical store."""

    source = (
        PLAYPEN_ROOT
        / "src"
        / "alphalattice"
        / "control"
        / "product_host"
        / "composition"
        / "workspace.py"
    ).read_text(encoding="utf-8")
    assert "DuckDbTaskControlRegistry(database.path" not in source
    assert source.count("resolve_task_control_database(database.workspace)") == 2


def test_no_production_site_spells_the_task_database_as_a_literal() -> None:
    """Seventeen agreeing strings were what allowed one of them to disagree."""

    owner = (
        PLAYPEN_ROOT / "src" / "alphalattice" / "control" / "task_control" / "registry.py"
    ).resolve()
    offenders = [
        path.relative_to(PLAYPEN_ROOT).as_posix()
        for directory in ("src", "scripts")
        for path in (PLAYPEN_ROOT / directory).rglob("*.py")
        if path.resolve() != owner
        and f'"{TASK_CONTROL_DATABASE_FILENAME}"' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
