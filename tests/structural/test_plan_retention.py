"""An owner reopens every plan an answer named by its hash, after a restart too,
and builds each answer from that plan alone."""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "alphalattice"

SINGLE_PLAN_STORES = frozenset(
    {
        # One preparation runs at a time: a newer preview supersedes the older one.
        "control/product_host/data_preparation/application.py",
    }
)
"""The owners whose plan store keeps its newest plan alone, each with its reason."""


def test_an_answer_and_its_run_read_their_own_plan_never_the_owners_last() -> None:
    """Regression: an owner kept only its
    last plan, so two score plans in one Host left the first answer's run refused
    `strategy_score.plan_required`. Then the owners kept every plan by its hash, yet an input's
    capture, a strategy's preparation, the score, the calibration and the data update still
    built an answer from their last plan after keeping this one, so two concurrent plans could
    answer the first with the second's hash; the research update read the data owner's last
    plan; and the workspace preparation and the data update ran a plan their memory held past
    the hour a restarted Host refused it at. No module reads another owner's last plan, and an
    owner reads its own only to compare a new preparation plan with its prior one."""

    reads: set[tuple[str, str]] = set()
    foreign: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        where = path.relative_to(SRC).as_posix()
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(function):
                if not (
                    isinstance(node, ast.Attribute)
                    and node.attr == "last_plan"
                    and isinstance(node.ctx, ast.Load)
                ):
                    continue
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    reads.add((where, function.name))
                else:
                    foreign.append(f"{where}:{node.lineno}")
    assert foreign == [], foreign
    assert reads == {("control/product_host/data_preparation/application.py", "plan")}, reads


def test_every_plan_is_sealed_on_disk_until_it_expires() -> None:
    """regression (S3, the user's review at d1ed8a25): a training plan saved by
    `training plan --output training.json` died with a Host restart, refused
    `model_training.preview_required`, since its owner kept plans in memory; the score, the
    calibration, a strategy's preparation, an input's capture and the workspace preparation
    did too, and a data update's maintenance plan. No owner keeps plans in a bare
    mapping: each keeps them in the plan store, rooted in the workspace's `runtime/`, and only a
    single-plan owner keeps its newest alone."""

    # A plan an answer named is keyed by its hash; a tracker of running Tasks keyed by their id
    # (the evidence preparation's thread widths) holds no plan a reader sends again.
    bare = re.compile(r"self\._plans(: dict\[str, [^\]]*\])? = \{\}")
    held = sorted(
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if bare.search(path.read_text(encoding="utf-8"))
    )
    assert held == [], held
    stores: dict[str, tuple[bool, bool]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PreviewRegistry":
                named = {keyword.arg: keyword.value for keyword in node.keywords}
                single = isinstance(named.get("single"), ast.Constant) and named["single"].value
                stores[path.relative_to(SRC).as_posix()] = ("root" in named, bool(single))
    assert set(stores) == {
        "control/product_host/composition/research_experiments.py",
        "control/product_host/composition/strategy_calibration.py",
        "control/product_host/composition/strategy_scoring.py",
        "control/product_host/data_preparation/application.py",
        "control/product_host/data_preparation/input_capture.py",
        "control/product_host/data_preparation/model_training.py",
        "control/product_host/data_preparation/research_strategy.py",
        "control/product_host/maintenance/data_update.py",
    }, sorted(stores)
    assert all(rooted for rooted, _single in stores.values()), stores
    assert {path for path, (_rooted, single) in stores.items() if single} == SINGLE_PLAN_STORES


def test_no_owner_keeps_a_preview_in_one_slot() -> None:
    """Plan-store previews survive later previews and restarts."""

    slots = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign) and node.value is not None
                else []
            )
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and "preview" in target.attr
                    and not (
                        isinstance(node.value, ast.Call)
                        and getattr(node.value.func, "id", None) == "PreviewRegistry"
                    )
                ):
                    slots.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    assert slots == [], slots


EXPIRY_REPLANS = {
    "model_training.preview_required": "composition/portfolio_research_operations.py",
    "portfolio_calibration.plan_required": "composition/portfolio_research_operations.py",
    "research_input.preview_required": "composition/portfolio_research_operations.py",
    "research_strategy.preview_required": "composition/portfolio_research_operations.py",
    "strategy_score.plan_required": "composition/portfolio_research_operations.py",
    "workspace_data_update.plan_required": "composition/portfolio_research_operations.py",
    "workspace_preparation.preview_required": "composition/portfolio_research_operations.py",
    "research_foundation.preview_required": "composition/research_experiments.py",
    "research_experiment.preview_expired": "composition/research_experiments.py",
}
"""Every plan-expiry refusal a request can meet, by the module that answers it with a re-plan."""

EXPIRY_WITHOUT_REPLAN = {
    # A hash never planned binds nothing to plan again; its words say to plan, then run.
    "research_experiment.preview_required",
    # A Command the Host built without its plan: an invariant, never a request's refusal.
    "portfolio_update.plan_required",
    "research_update.plan_required",
    "workspace_preparation.plan_required",
}


def test_every_plan_expiry_refusal_offers_its_replan() -> None:
    """Every plan-expiry refusal has words and a re-plan offer, or names its exception's reason."""

    import json

    raised = set()
    expiry = re.compile(r'"([a-z_]+\.(?:plan_required|preview_required|preview_expired))"')
    for path in SRC.rglob("*.py"):
        raised |= set(expiry.findall(path.read_text(encoding="utf-8")))
    assert raised == set(EXPIRY_REPLANS) | EXPIRY_WITHOUT_REPLAN, sorted(raised)
    words = json.loads(
        (SRC / "interface/local_application/refusal_words.json").read_text(encoding="utf-8")
    )
    for code, module in EXPIRY_REPLANS.items():
        assert code in words, code
        lines = (SRC / "control/product_host" / module).read_text(encoding="utf-8").splitlines()
        assert any(
            any("replan" in nearby for nearby in lines[max(0, at - 10) : at + 11])
            for at, line in enumerate(lines)
            if f'"{code}"' in line
        ), code


def test_the_study_kinds_goal_summaries_are_pinned() -> None:
    """A study Task's goal summary stays byte-identical so stored goal and plan hashes read back."""

    import inspect

    from alphalattice.control.product_host.composition import research_experiments

    source = inspect.getsource(research_experiments._task_contract)
    for summary in (
        "Run an admitted Factor experiment, not a strategy or current publication.",
        "Replay selected Alpha evidence through an EW research book; no activation.",
    ):
        assert summary in source, summary
    for part in (
        '"Run an admitted Alpha development experiment, "',
        '"not a strategy or current publication."',
        '"Run installed Risk diagnostics over a sealed research input; "',
        '"no allocation or activation."',
    ):
        assert part in source, part
