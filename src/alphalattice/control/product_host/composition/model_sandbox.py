"""`model sandbox`: an agent's model tried on a copy of the workspace before a person activates it.

EX, the model point. The workspace at rest (its writer lease held, so no Host serves it) is
copied beside it, hard links kept; saved-object readback opens every object of the copy;
the model is
installed as the copy's trial alone; one Alpha study runs on the copy -- the workspace's latest
Alpha study declared anew with the model and its reference recipe, or the study `--study` names
-- and is read back, its run time and the process's peak memory recorded;
saved-object readback opens the copy
again, and every earlier read must keep its verdict and the new study's must open. The trial's
record goes into the workspace's registry (`runtime/extensions/alpha-models.json`), where a
person's activation reads it, and the copies are deleted. The entry composes it into the CLI as
it composes `backup restore`: the product readback probe imports no test harness.
"""

from __future__ import annotations

import copy as copying
import json
import os
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "BLOCKED", "RECOVERY_REQUIRED"})


def _copy(source: Path, target: Path) -> None:
    """Copy a workspace keeping its hard links; its writer lock and its Host's connection record
    stay behind, since no Host of the original serves the copy."""
    seen: dict[tuple[int, int], str] = {}

    def copy_one(src: str, dst: str) -> str:
        """Read the captured source document under its admitted contract."""
        info = os.stat(src)
        key = (info.st_dev, info.st_ino)
        if info.st_nlink > 1 and key in seen:
            os.link(seen[key], dst)
            return dst
        shutil.copy2(src, dst)
        if info.st_nlink > 1:
            seen[key] = dst
        return dst

    shutil.copytree(
        source,
        target,
        symlinks=True,
        copy_function=copy_one,
        ignore=shutil.ignore_patterns(
            ".alphalattice-writer.lock", "local-research-connection.json"
        ),
    )


def _verdicts(path: Path) -> dict[tuple[str, str, str], str]:
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return {(row["kind"], row["key"], row["op"]): row["verdict"] for row in rows}


def _probe(copy: Path, work: Path, out: Path) -> dict[tuple[str, str, str], str]:
    import contextlib
    import io

    from .saved_object_readback import probe

    harvest = copy.parent / f"{copy.name}-harvest"
    harvest.mkdir()
    try:
        # The harness reads each workspace under a harvest; the copy is linked in, not copied.
        os.replace(copy, harvest / copy.name)
        # The harness's progress stays out of the command's one envelope.
        with contextlib.redirect_stdout(io.StringIO()):
            probe(harvest, work, out)
    finally:
        os.replace(harvest / copy.name, copy)
        shutil.rmtree(harvest, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    return _verdicts(out)


class _Peak:
    """The process's resident memory at its highest while a study runs, sampled."""

    def __init__(self) -> None:
        import psutil  # type: ignore[import-untyped]

        self._process = psutil.Process()
        self.bytes = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            self.bytes = max(self.bytes, self._process.memory_info().rss)

    def __enter__(self) -> _Peak:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.bytes = max(self.bytes, self._process.memory_info().rss)


def _operate(session: Any, document: dict[str, Any]) -> dict[str, Any]:
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument,
    )

    request = PortfolioResearchRequestDocument.model_validate(document).to_operation_request()
    return dict(session.operations.execute(request, caller="EXTERNAL_AUTOMATION"))


def _declared_study(session: Any, copy: Path, model_id: str, study: Path | None) -> dict[str, Any]:
    """The study the trial runs, the model in its place: `--file`'s, or the latest published.

    Either is an Alpha study that names a model; only the model and its recipe change, so a
    study declared for another model never runs as this one's trial.
    """
    import yaml  # type: ignore[import-untyped]

    from alphalattice.capabilities.alpha_modeling.catalog import (
        build_installed_alpha_model_catalog,
    )
    from alphalattice.capabilities.alpha_modeling.extension import extension_adapter
    from alphalattice.control.product_host.research_authoring.model_extensions import (
        activated_models,
    )
    from alphalattice.investment.alpha_research.experiments.mandate import (
        build_installed_alpha_model_capability_mandate,
    )

    document: dict[str, Any] | None = None
    if study is not None:
        document = dict(yaml.safe_load(study.read_text(encoding="utf-8")) or {})
        # A lifecycle study declares a component's recipe, not a model to swap.
        if "model_capability_handle" not in (document.get("alpha") or {}):
            raise ValueError("model_sandbox.study_names_no_model")
    else:
        listed = _operate(session, {"operation": "EXPERIMENTS"})["experiments"]
        # The latest development study that names a model.
        for row in reversed(listed):
            if row["kind"] != "alpha.model-development" or row["lifecycle"] != "SUCCEEDED":
                continue
            body = _operate(
                session, {"operation": "EXPERIMENT_READBACK", "task_id": row["task_id"]}
            )
            if "model_capability_handle" in (body.get("document") or {}).get("alpha", {}):
                document = copying.deepcopy(dict(body["document"]))
                break
    if document is None:
        raise ValueError("model_sandbox.no_alpha_study")
    mandate = build_installed_alpha_model_capability_mandate(
        catalog=build_installed_alpha_model_catalog(activated_models(copy))
    )
    index = [value.adapter_id for value in mandate.ordered_search_domains].index(model_id)
    declaration = cast(Any, extension_adapter(model_id)).declaration
    # The study's experiment section is its authority's (a handoff binds it); only the
    # model and its recipe change.
    document["alpha"]["model_capability_handle"] = mandate.capability_handle(index)
    document["alpha"]["model_parameters"] = dict(declaration.recipe)
    return document


def _run_study(copy: Path, model_id: str, study: Path | None) -> tuple[str, float, int, str]:
    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )

    session = LocalPortfolioWebSession.from_workspace(copy)
    session.start()
    try:
        document = _declared_study(session, copy, model_id, study)
        plan = _operate(session, {"operation": "EXPERIMENT_PLAN", "experiment_document": document})
        if plan.get("status") != "PLANNED":
            raise ValueError(f"model_sandbox.study_refused:{plan.get('failure_code')}")
        started = time.perf_counter()
        with _Peak() as peak:
            sent = _operate(
                session, {"operation": "EXPERIMENT_RUN", "experiment_plan_hash": plan["plan_hash"]}
            )
            task_id = str(sent.get("task_id") or sent.get("publication_task_id"))
            while True:
                lifecycle = _operate(session, {"operation": "STATUS", "task_id": task_id}).get(
                    "lifecycle"
                )
                if lifecycle in _TERMINAL:
                    break
                time.sleep(1.0)
        seconds = time.perf_counter() - started
        body = _operate(session, {"operation": "EXPERIMENT_READBACK", "task_id": task_id})
        if body.get("status") != "EXPERIMENT_PUBLISHED":
            raise ValueError(f"model_sandbox.study_not_published:{lifecycle}")
        # The record names the model the study ran, read back from it, never the one asked:
        # a trial of another model is no trial of this one.
        ran = next(
            (
                str(row.get("model_adapter_id"))
                for row in _operate(session, {"operation": "EXPERIMENTS"})["experiments"]
                if row["task_id"] == task_id
            ),
            None,
        )
        if ran != model_id:
            raise ValueError(f"model_sandbox.study_ran_another_model:{ran}")
        return task_id, seconds, peak.bytes, ran
    finally:
        session.stop()


def run_sandbox(
    workspace: Path,
    model_id: str,
    *,
    study: Path | None = None,
    keep: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Try a model on a copy of the workspace and record the trial.

    Args:
        workspace: The workspace, at rest.
        model_id: The model.
        study: An Alpha study declaration to run with the model in its place; the
            workspace's latest published Alpha study that names a model when omitted.
        keep: Keep the copy for inspection.
        root: Where the copy goes; `<workspace's parent>/.alphalattice-sandboxes` by default.

    Returns:
        `SANDBOXED` and the record: PASSED when every earlier read kept its verdict and the
        study opened, FAILED otherwise.

    Raises:
        ValueError: `model_sandbox.host_serving` while a Host holds the workspace; the model's
            contract's, the study's or the copy's refusal by name.
    """
    from alphalattice.capabilities.alpha_modeling.catalog import (
        build_installed_alpha_model_catalog,
    )
    from alphalattice.capabilities.alpha_modeling.model_contract import check_model
    from alphalattice.control.product_host.research_authoring.model_extensions import (
        ModelExtensionRegistry,
        ModelSandboxRecord,
    )
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    workspace = workspace.resolve()
    if model_id in build_installed_alpha_model_catalog().adapter_ids:
        raise ValueError(f"model_extension.installed:{model_id}")
    contract = check_model(model_id)
    failed = next((row["code"] for row in contract["findings"] if row["code"]), None)
    if failed is not None:
        raise ValueError(f"model_extension.contract_failed:{failed}")
    try:
        lease = WorkspaceWriterLease.acquire(workspace)
    except (OSError, RuntimeError) as error:  # the lease is held: a Host serves it
        raise ValueError("model_sandbox.host_serving") from error
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    folder = (root or workspace.parent / ".alphalattice-sandboxes") / f"{model_id}-{stamp}"
    copy = folder / "workspace"
    try:
        folder.mkdir(parents=True)
        _copy(workspace, copy)
        before = _probe(copy, folder / "u0-before", folder / "u0-before.json")
        registry = ModelExtensionRegistry.read(copy)
        registry.model_copy(update={"trial": (*registry.trial, model_id)}).write(copy)
        task_id, seconds, peak, ran = _run_study(copy, model_id, study)
        after = _probe(copy, folder / "u0-after", folder / "u0-after.json")
        changed = sum(1 for key, verdict in before.items() if after.get(key) != verdict)
        changed += sum(
            1
            for key, verdict in after.items()
            if key not in before and not verdict.startswith("OPENS")
        )
        record = ModelSandboxRecord(
            model_id=model_id,
            declaration_hash=contract["declaration_hash"],
            numerical_binding_hash=contract["numerical_binding_hash"],
            contract_receipt_hash=contract["contract_receipt_hash"],
            study_task_id=task_id,
            study_model_id=ran,
            study_seconds=round(seconds, 3),
            peak_memory_bytes=peak,
            u0_reads=len(before),
            u0_changed=changed,
            recorded_at=datetime.now(UTC),
        )
        held = ModelExtensionRegistry.read(workspace)
        held.model_copy(update={"sandboxes": (*held.sandboxes, record)}).write(workspace)
        return {
            "status": "SANDBOXED",
            "verdict": "PASSED" if record.passed else "FAILED",
            "record": record.model_dump(mode="json"),
            "copy": str(copy) if keep else None,
            "next_action": "ASK_THE_PERSON_TO_ACTIVATE_IT"
            if record.passed
            else "READ_THE_COPY_WITH_KEEP_AND_CORRECT_THE_MODEL",
        }
    finally:
        lease.close()
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)


__all__ = ["run_sandbox"]
