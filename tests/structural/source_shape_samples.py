from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

_OPTION_CASES = {
    "accessions": ("--acquire-sec", "--entities", "AAPL", "--accessions", "bad", "--preflight"),
    "entities": ("--acquire-sec", "--entities", *(f"QE{index}" for index in range(9))),
    "evidence_as_of": (
        "--acquire-sec",
        "--entities",
        "AAPL",
        "--evidence-as-of",
        "not-a-time",
        "--preflight",
    ),
    "maximum_documents_per_issuer": (
        "--acquire-sec",
        "--entities",
        "AAPL",
        "--maximum-documents-per-issuer",
        "2",
    ),
    "minimum_entity_coverage": (
        "--acquire-sec",
        "--entities",
        "AAPL",
        "--minimum-entity-coverage",
        "1.5",
    ),
    "model_name": ("--acquire-sec", "--entities", "AAPL", "--model-name", "fixture-model"),
    "research_input_id": (
        "--acquire-sec",
        "--entities",
        "AAPL",
        "--research-input-id",
        "fixture-input",
    ),
    "semantic_model": ("--acquire-sec", "--entities", "AAPL"),
    "source_artifact_root": ("--source-set-hash", "a" * 64),
    "source_set_hash": ("--source-set-hash", "../escape"),
}

ROOT = Path(__file__).resolve().parents[2]


def bundle_refusal_declarations():
    """Bundle manual refusals declare their complete vocabulary."""

    from alphalattice.control.product_host.composition.evidence_review_bundles import (
        agent_bundle_refusal,
    )

    tree = ast.parse(Path(inspect.getfile(agent_bundle_refusal)).read_text(encoding="utf-8"))
    declarations = [
        ast.literal_eval(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and node.keys
        and all(
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.startswith("agent_bundle.")
            for key in node.keys
        )
    ]
    assert declarations, "the bundle's declared refusal vocabulary is part of the census"
    return declarations


def evidence_review_refusal_samples():
    """Review refusal declarations include every outcome and internal refusal."""

    root = Path(__file__).resolve().parents[2] / "src/alphalattice"
    owners = (
        "control/product_host/composition/evidence_review_application.py",
        "control/product_host/composition/evidence_review_delivery.py",
        "control/product_host/composition/evidence_review_bundles.py",
        # The book a request's selector names, opened.
        "oversight/chief_risk_officer/decision/book_evidence.py",
    )
    readers = (
        "product_host.evidence_review_update_not_configured",
        "product_host.evidence_review_update_task_mismatch",
        "product_host.evidence_review_publication_origin_unavailable",
        "portfolio_update.publication_not_bound_to_task",
        "research_update.publication_not_bound_to_task",
        "portfolio_research.session_outside_report",
        "portfolio_research.date_selector_not_portfolio",
    )
    internal = {
        # The Host's own composition: a workspace served without these is refused earlier, by
        # name, as its setup; every Host that serves a review reads studies.
        "product_host.evidence_review_service_absent",
        "product_host.evidence_review_adapter_absent",
        "product_host.evidence_refresh_command_incomplete",
        "product_host.evidence_review_experiment_not_configured",
        # A sealed record that no longer reads or measures as it was sealed: an update whose
        # listing labels and weights disagree, a handoff whose report the ledger holds otherwise,
        # an analysis selected for the review that answers another question than the review's.
        "product_host.evidence_review_update_axis_mismatch",
        "product_host.evidence_review_report_mismatch",
        "product_host.evidence_review_obligation_mismatch",
        "product_host.evidence_review_axis_mismatch",
        "product_host.evidence_review_scope_mismatch",
        "product_host.evidence_coverage_run_unreadable",
        "alternative_evidence.delivery_measurement_unstable",
        "chief_risk_officer.prepared_submission_invalid",
        "chief_risk_officer.submission_identity_mismatch",
        "chief_risk_officer.submitted_actor_invalid",
        "chief_risk_officer.review_not_prepared",
        "chief_risk_officer.review_actor_unavailable",
        # Who may submit or select: a person or an agent through its bundle; the Host's own
        # automation never does.
        "alternative_evidence.external_submission_entry_required",
        "chief_risk_officer.external_submission_entry_required",
        "product_host.evidence_selection_actor_invalid",
        # A slot another writer filled: the Host is its workspace's one writer, and one
        # binding's answers take turns.
        "agent_bundle.answer_slot_taken",
    }
    raised: set[str] = set()
    outcomes = 0
    for name in owners:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
                argument = node.exc.args[0]
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and re.fullmatch(r"[a-z_]+\.[a-z_]+(:.*)?", argument.value)
                ):
                    raised.add(argument.value.partition(":")[0])
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ReviewOutcome":
                keywords = {value.arg: value.value for value in node.keywords}
                disposition = keywords.get("disposition")
                if isinstance(disposition, ast.Constant) and str(disposition.value).startswith(
                    "REFUSED"
                ):
                    outcomes += 1
                    assert "detail" in keywords, (name, disposition.value)
    assert outcomes > 10 and len(raised) > 30
    assert internal <= raised, internal - raised
    return raised, outcomes, internal, readers


def evidence_unit_failure_codes():
    """Evidence unit failure codes cover the source vocabulary."""

    root = Path(__file__).resolve().parents[2] / "src/alphalattice/evidence/alternative_evidence"
    codes = {
        code
        for path in root.rglob("*.py")
        for code in re.findall(r'"(alternative_evidence\.[a-z_]+)', path.read_text("utf-8"))
    }
    assert len(codes) > 300, len(codes)
    return codes


def specialist_answer_bound_samples():
    """Specialist answer bounds retain their named owner constants."""

    from alphalattice.evidence.alternative_evidence.analysis import (
        submissions as analyst_screen,
    )
    from alphalattice.evidence.alternative_evidence.analysis.contracts import (
        ANALYST_ANSWER_TEXT_FIELDS,
    )
    from alphalattice.oversight.chief_risk_officer.decision import submissions as cro_screen
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        REVIEW_ANSWER_TEXT_FIELDS,
    )

    cro = {
        "risks": ("MAXIMUM_ANSWER_RISKS", cro_screen.MAXIMUM_ANSWER_RISKS),
        "resolved": ("MAXIMUM_ANSWER_RESOLUTIONS", cro_screen.MAXIMUM_ANSWER_RESOLUTIONS),
        "summary": ("REVIEW_ANSWER_TEXT_FIELDS", REVIEW_ANSWER_TEXT_FIELDS["summary"]),
    }
    analyst = {
        "findings": ("MAXIMUM_ANSWER_FINDINGS", analyst_screen.MAXIMUM_ANSWER_FINDINGS),
        "notes": ("ANALYST_ANSWER_TEXT_FIELDS", ANALYST_ANSWER_TEXT_FIELDS["notes"]),
    }
    return cro, analyst


def workbench_source_inputs():
    """Workbench retention and Goal inputs enumerate their original source declarations."""

    root = ROOT
    retention_owner = root / "src/alphalattice/control/product_host/storage/input_references.py"
    additions = [
        call
        for call in ast.walk(ast.parse(retention_owner.read_text(encoding="utf-8")))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"add", "update"}
        and isinstance(call.func.value, ast.Subscript)
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "roots"
    ]
    assert additions and all(
        len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
        for call in additions
    ), "Every owner retention reason must participate in the language census"
    retention_roots = sorted({call.args[0].value for call in additions})
    goal_owner = root / "src/alphalattice/control/product_host/composition/goals.py"
    goal_tree = ast.parse(goal_owner.read_text(encoding="utf-8"))
    goal_reference_operations = next(
        ast.literal_eval(node.value)
        for node in goal_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REFERENCE_OPERATIONS"
            for target in node.targets
        )
    )
    goal_comparison_refusals = next(
        sorted(ast.literal_eval(node.value.args[0]))
        for node in goal_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "COMPARISON_REFUSALS"
            for target in node.targets
        )
    )
    return retention_roots, goal_reference_operations, goal_comparison_refusals


def workbench_request_inputs(root: Path, replacements: dict[str, str] | None = None):
    """Read action and POST edges through the existing Workbench source harness."""
    from tests.portfolio_strategy_lab.local_web_support import run_node

    reader = root / "scripts/ui_qa/checks/detail_routes_probe.cjs"
    script = (
        "const fs=require('node:fs'),q=JSON.parse(fs.readFileSync(0,'utf8'));"
        "console.log(JSON.stringify(require(q.reader).requestInventory(q.root,q.sources)));"
    )
    answer = run_node(
        ["-e", script],
        required=True,
        cwd=root,
        capture_output=True,
        text=True,
        input=json.dumps({"reader": str(reader), "root": str(root), "sources": replacements or {}}),
        timeout=55,
    )
    assert answer.returncode == 0, answer.stderr
    return json.loads(answer.stdout.strip().splitlines()[-1])


def generated_evidence_sentences():
    """Collect generated Evidence sentences from their declared source."""

    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        NOT_ADDRESSED_RATIONALE,
        ReviewState,
    )

    src = ROOT / "src/alphalattice"

    def body(relative: str, name: str) -> ast.AST:
        tree = ast.parse((src / relative).read_text(encoding="utf-8"))
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    def filled(node: ast.AST, n: str) -> str | None:
        if isinstance(node, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant) else n for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = filled(node.left, n), filled(node.right, n)
            return None if left is None or right is None else left + right
        return (
            node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
        )

    def plural(words: str) -> str:  # the page's ownerWords: `3 table(s)` reads `3 tables`
        return re.sub(
            r"(\d[\d,]*)(\D*?)\b([a-z]+)\(s\)",
            lambda m: m[1] + m[2] + m[3] + ("" if m[1] == "1" else "s"),
            words,
        )

    said: list[str] = []
    for relative, name in (
        ("evidence/alternative_evidence/analysis/contracts.py", "gap_lines"),
        ("evidence/alternative_evidence/analysis/routing.py", "route_documents"),
    ):
        for n in ("1", "3"):
            for call in ast.walk(body(relative, name)):
                if not (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "append"):
                    continue
                words = filled(call.args[0], n) if call.args else None
                if not words or ("(s)" not in words and "_GAP: " not in words):
                    continue
                words = words.split(": ", 1)[1] if re.match(r"^[A-Z_]+_GAP: ", words) else words
                said.append(plural(words))
    plan = next(
        n
        for n in ast.walk(
            ast.parse(
                (src / "evidence/alternative_evidence/analysis/packet.py").read_text(
                    encoding="utf-8"
                )
            )
        )
        if isinstance(n, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "scope_rule" for k in n.keys)
    )
    said.append(filled(plan.values[[k.value for k in plan.keys].index("scope_rule")], ""))
    summary = body(
        "oversight/chief_risk_officer/decision/submissions.py", "normalize_review_answer"
    )
    for n in ("1", "3"):
        for value in ast.walk(summary):
            words = filled(value, n)
            if not words or "named in the evidence read" not in words:
                continue
            if isinstance(value, ast.JoinedStr):  # `{count} risk{'' if count == 1 else 's'} ...`
                said.append(plural(words.replace(" risk" + n, " risk(s)")))
            elif n == "1" and words[:1].isupper():  # a whole sentence, not a part of the f-string
                said.append(words)
    said.append(NOT_ADDRESSED_RATIONALE)
    coverage = next(
        filled(n, "{subject}")
        for n in ast.walk(
            body("interface/local_application/evidence_cro.py", "project_published_review")
        )
        if isinstance(n, ast.JoinedStr) and "In scope:" in (filled(n, "") or "")
    )
    for state in ReviewState:
        values = iter((state.value, "100.000%", "44.000%", "not recorded", "not recorded"))
        said.append(re.sub(r"\{subject\}", lambda _, values=values: next(values), coverage))
    return said
