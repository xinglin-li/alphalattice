"""V661: every whole canonical Task reader declares whether it needs complete authority."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
COMPOSITION = "control/product_host/composition/"
PREPARATION = "control/product_host/data_preparation/"

# These readers select latest/reusable authority, establish absence/completeness, or mutate
# admission, recovery, incident or retention state. Missing canonical authority must stop them.
# Repeated names pin multiple calls; preparation wrappers are included because they carry the
# same complete scan. Changing a reader to partial requires reviewing that reader's meaning.
STRICT_READERS = {
    COMPOSITION + "decision_advancement.py": (
        "DecisionAdvancementApplication",
        "plan admit in_flight _completed_scores _completed_input "
        "reusable publication_task readback",
    ),
    COMPOSITION + "evidence_review_application.py": (
        "EvidenceReviewApplication",
        "_run_tasks _active_refresh_task _completed_task "
        "_accepted_answer_delivery recovery_commands",
    ),
    COMPOSITION + "feature_trials.py": ("FeatureTrials", "way_on"),
    COMPOSITION + "local_web_session.py": ("LocalPortfolioWebSession", "_recovery_commands"),
    COMPOSITION + "portfolio_research_operations.py": (
        "PortfolioResearchOperations",
        "sweep_if_due _execute _workspace_operation _strategy_task "
        "_automation_answer upgrade cpu_budget",
    ),
    COMPOSITION + "portfolio_updates.py": (
        "PortfolioUpdateApplication",
        "reusable admit readback publication_task",
    ),
    COMPOSITION + "research_experiments.py": (
        "ResearchExperimentApplication",
        "_prerequisites intents _qualification _risk_studies "
        "_task_for_plan_hash _matching_task admit",
    ),
    COMPOSITION + "strategy_activation.py": ("StrategyActivation", "standing offer"),
    COMPOSITION + "strategy_calibration.py": (
        "StrategyCalibrationApplication",
        "prepare published_input reusable admit readback",
    ),
    COMPOSITION + "strategy_scoring.py": (
        "StrategyScoringApplication",
        "prepare reusable admit published_scores readback",
    ),
    COMPOSITION + "task_supervision.py": ("TaskSupervisor", "supervise_once"),
    COMPOSITION + "verification_sweep.py": (
        "StudyVerificationSweep",
        "due saved_studies work_waits",
    ),
    PREPARATION + "application.py": (
        "WorkspacePreparationApplication",
        "tasks readback plan delegated_resume_allowed require_confirmation_caller "
        "confirm confirm confirm owns_published_manifest",
    ),
    PREPARATION + "feature_research.py": (
        "ResearchFeatureBuildApplication",
        "_submit admit prepared_source",
    ),
    PREPARATION + "input_capture.py": (
        "ResearchInputCaptureApplication",
        "plan confirm admit admit",
    ),
    PREPARATION + "model_training.py": ("ModelTrainingInputApplication", "prepare admit"),
    PREPARATION + "research_strategy.py": ("ResearchStrategyPreparation", "plan prepare admit"),
    "control/product_host/maintenance/data_update.py": (
        "WorkspaceDataUpdateApplication",
        "_approved_plans _waiting confirm.approve receipt_task "
        "readback reusable prepare admit execute_step",
    ),
    "control/product_host/storage/input_references.py": ("ResearchInputStorage", "_references"),
    "control/task_control/registry.py": (
        "DuckDbTaskControlRegistry",
        "active_task stale_active_tasks",
    ),
    "control/task_control/runner.py": ("TaskControlRunner", "stale_active_tasks"),
    "evidence/alternative_evidence/runtime/task_adapter.py": (
        "AlternativeEvidenceDocumentTaskAdapter",
        "_carried_index completed_unit run_tasks prepared_receipts",
    ),
    "evidence/alternative_evidence/storage/inventory.py": (
        "EvidenceStorageInventory",
        "generation_readers",
    ),
    "interface/local_application/dispatcher.py": (
        "LocalBackgroundDispatcher",
        "resume _waits_its_turn _drive_waiting",
    ),
}

# Display collections preserve readable peers and name unreadable canonical Task IDs. Data
# issues preserve cases, but withhold continuation grants when the complete Task scan is unknown.
# Recovery command construction has both boundaries: compatibility reads may inspect known peers,
# while the default actual-resume branch keeps its strict scan above.
PARTIAL_READERS = {
    COMPOSITION + "evidence_review_application.py": (
        "EvidenceReviewApplication",
        "recovery_commands",
    ),
    COMPOSITION + "research_history.py": ("ResearchHistory", "listing"),
    COMPOSITION + "research_experiments.py": ("ResearchExperimentApplication", "listing"),
    COMPOSITION + "portfolio_research_operations.py": (
        "PortfolioResearchOperations",
        "pending_decisions guardian upgrade",
    ),
    COMPOSITION + "upgrade_overview.py": ("", "upgrade_overview"),
    PREPARATION + "remediation.py": ("WorkspaceDataIssueApplication", "readback"),
}


def _expected_readers():
    expected = Counter()
    for action, groups in (("tasks", STRICT_READERS), ("record_collection", PARTIAL_READERS)):
        for path, (owner, functions) in groups.items():
            for function in functions.split():
                scope = f"{owner}.{function}" if owner else function
                expected[path, scope, action] += 1
    expected[COMPOSITION + "portfolio_research_operations.py", "reused_read", "tasks"] = 1
    # Prewarm selects the latest exact source-book update from complete canonical authority.
    expected[COMPOSITION + "local_web_session.py", "forward_update_read_requests", "tasks"] = 1
    expected[
        "control/product_host/maintenance/data_update.py",
        "bind_existing_data_workspace",
        "read_existing_tasks",
    ] = 1
    # One public operation dispatch, not a canonical-record reader.
    expected[
        COMPOSITION + "portfolio_research_operations.py",
        "PortfolioResearchOperations._workspace_operation",
        "public-TASKS",
    ] = 1
    return expected


def _record_reader_calls(sources, *, canonical=False):
    found = Counter()
    for path, source in sources.items():
        tree = ast.parse(source)
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (
                    node.func.attr == "model_validate_json"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "TaskRecord"
                    if canonical
                    else node.func.attr in {"tasks", "record_collection", "read_existing_tasks"}
                )
            ):
                continue
            scope = []
            parent = node
            while parent in parents:
                parent = parents[parent]
                if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope.append(parent.name)
            scope = ".".join(reversed(scope))
            action = node.func.attr
            if (
                path == COMPOSITION + "portfolio_research_operations.py"
                and scope == "PortfolioResearchOperations._workspace_operation"
                and action == "tasks"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                action = "public-TASKS"
            found[path, scope, action] += 1
    return found


def test_every_whole_task_reader_keeps_its_reviewed_authority_requirement():
    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in ROOT.rglob("*.py")
    }
    assert _record_reader_calls(sources) == _expected_readers(), (
        "Review added or moved canonical Task readers: display collections must name unreadable "
        "rows, and admission/control/supervision/retention must keep complete authority."
    )


def test_the_census_discovers_a_new_reader_and_a_strict_reader_changed_to_partial():
    """A planted bypass must change the census, not escape its source discovery."""
    path = COMPOSITION + "task_supervision.py"
    strict = (
        "class TaskSupervisor:\n"
        "    def supervise_once(self, registry):\n"
        "        return registry.tasks()\n"
    )
    partial = strict.replace("registry.tasks()", "registry.record_collection()")
    assert _record_reader_calls({path: strict}) != _record_reader_calls({path: partial})
    fresh = "def new_reader(registry):\n    return registry.tasks()\n"
    discovered = _record_reader_calls({"new_reader.py": fresh})
    assert discovered == Counter({("new_reader.py", "new_reader", "tasks"): 1})
    assert discovered - _expected_readers()


def test_canonical_task_parsers_cannot_bypass_stored_identity_validation():
    """V661: filtered SQL reads share validation; queue heads retain their dedicated refusal."""
    path = "control/task_control/registry.py"
    source = (ROOT / path).read_text(encoding="utf-8")
    expected = Counter(
        {
            (path, "DuckDbTaskControlRegistry._verified_record", "model_validate_json"): 1,
            (path, "DuckDbTaskControlRegistry.queued_task.head", "model_validate_json"): 1,
        }
    )
    assert _record_reader_calls({path: source}, canonical=True) == expected
    bypass = "\ndef bypass(document):\n    return TaskRecord.model_validate_json(document)\n"
    assert _record_reader_calls({path: source + bypass}, canonical=True) != expected
