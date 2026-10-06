"""Updated research inputs use real Data/Feature/Outcome and Factor owners."""

from __future__ import annotations

import json
import math
import shutil
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation import input_capture
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    ResearchInputRevision,
    ResearchInputRevisions,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.reader import (
    FeaturePanelReader,
    FeaturePanelReadRequest,
)
from alphalattice.foundation.market_data_ops.runtime.refresh import normal_refresh_plan
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json, _request
from tests.researcher_methodology_surface.session_workspace import (
    copy_workspace,
    session_workspace,
)
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    recording_provider,
    unchanged_membership_source,
)
from tests.workspace_readiness.storage_support import published_panel_chunks


def _finish(live, phase, *, timeout):
    started = perf_counter()
    live.dispatcher.drain_for_tests(timeout=timeout)
    print(f"PLAYPEN_RESEARCH_PHASE {phase} {perf_counter() - started:.3f}s", flush=True)


def _build_revision_journey(root: Path, original_root: Path) -> dict:
    shutil.copytree(original_root, root)
    manifest_bytes = (root / "research-workspace.json").read_bytes()
    anchor = read_research_workspace_manifest(root).experiment_inputs[0]
    symbols = tuple(f"F{i:03d}" for i in range(120))
    clock_now = NOW
    effective_day = NOW + timedelta(days=1)
    provider = recording_provider(symbols=symbols, now=effective_day, sector_size=6)
    first = LocalPortfolioWebSession.from_workspace(root, clock=lambda: clock_now)
    first.data_provider = provider
    first.data_source_loader = unchanged_membership_source(symbols)
    with first as live:
        # The journey still runs every stage and session; only its CPU width is bounded.
        budget = live.operations.set_cpu_budget("2", chosen_by="EXTERNAL_AUTOMATION")
        assert budget["status"] == "CPU_BUDGET" and budget["cpu_budget"] == 2
        assert (
            _json(
                live,
                "/api/research-inputs/plan",
                method="POST",
                payload={"research_input_id": anchor.input_id},
            )["status"]
            == "REUSED_EXACT"
        )
        controls = _json(live, "/api/experiments/controls")
        document = controls["template"]
        document["factor"]["factor_ids"] = controls["factor_options"][:4]
        old_plan = _json(
            live, "/api/experiments/plan", method="POST", payload={"experiment_document": document}
        )
        old_run = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": old_plan["plan_hash"]},
        )
        _finish(live, "original-factor", timeout=240)
        old_task = old_run["task_id"]
        old_report = _json(live, f"/api/experiments/readback?task_id={old_task}")
        assert old_report["status"] == "EXPERIMENT_PUBLISHED", old_report
        old_export = _json(live, f"/api/experiments/export?task_id={old_task}")
        same_draft = _json(
            live,
            "/api/experiments/draft",
            method="POST",
            payload={"task_id": old_task, "input_binding_hash": anchor.binding_hash},
        )
        same_plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": same_draft["document"], "origin_task_id": old_task},
        )
        same_run = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": same_plan["plan_hash"]},
        )
        assert same_run["status"] == "REUSED_EXACT" and same_run["task_id"] is None
        assert same_run["publication_task_id"] == old_task

        # A real source departure, not a fixed-Universe update: historical
        # samples retain the leaver while the as-of cohort becomes smaller.
        live.operations.data_update.source_loader = unchanged_membership_source(symbols[1:])
        market = MarketDataRepository(root)
        assert market.universe_bootstrap("us-current-index-research") is not None
        live.operations.data_update._readiness(market).refresh_sources_if_due(observed_at=NOW)
        update = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert update["status"] == "CONFIRMATION_REQUIRED", update
        _json(
            live,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": update["plan_hash"]},
        )
        sent = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": update["plan_hash"]},
        )
        _finish(live, "data-update", timeout=240)
        assert _json(live, f"/api/status?task_id={sent['task_id']}")["lifecycle"] == "SUCCEEDED"
        # The transition update's own record: its members' market data came through the
        # transition itself (the candidate path), so its cycle recorded no maintenance run
        # of its own -- the readback says none, and never another update's units; its
        # partition account is the receipt's resulting Panel's, whose manifest the plan's
        # `before` does not name.
        transition = _json(live, f"/api/data-update?task_id={sent['task_id']}")
        assert transition["selected"] and transition["task_id"] == sent["task_id"]
        assert transition["cycle"]["status"] == "completed"
        assert transition["cycle"]["change_set"]["listings_with_new_sessions"] == len(symbols)
        assert transition["maintenance"] is None, transition["maintenance"]
        assert (
            transition["receipt"]["after"]["manifest_revision"]
            != transition["receipt"]["before"]["manifest_revision"]
        )
        assert transition["partition_reuse"]["source"] == "RECEIPT_PANEL"
        assert (
            transition["partition_reuse"]["panel_hash"]
            == transition["receipt"]["after"]["panel_hash"]
        )
        # Observed departure is effective tomorrow. Today's still-member source
        # tail must not be skipped merely because the request manifest changed.
        bootstrap = market.universe_bootstrap("us-current-index-research")
        old_members = market.load_universe_manifest_revision(bootstrap.manifest_revision)
        leaver = next(item for item in old_members.listings if item.symbol == symbols[0])
        assert market.raw_bars(leaver.listing_id, through=NOW.date())[-1].session_date == NOW.date()
        calls_at_departure = len(provider.calls)
        clock_now = effective_day
        effective_plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert effective_plan["status"] == "PLANNED", effective_plan
        effective_run = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": effective_plan["plan_hash"]},
        )
        _finish(live, "effective-day-update", timeout=240)
        assert (
            _json(live, f"/api/status?task_id={effective_run['task_id']}")["lifecycle"]
            == "SUCCEEDED"
        )
        # The effective-day update ran the members' maintenance itself: its cycle recorded
        # the run it admitted (the manifest it bound), and the readback resolves the units
        # from that record -- the new membership, one fewer than the transition's `before`.
        effective = _json(live, f"/api/data-update?task_id={effective_run['task_id']}")
        units = effective["maintenance"]
        assert units["availability"] == "AVAILABLE", units
        assert units["scope_source"] == "CYCLE_RECORD"
        assert units["cycle_id"] == effective["cycle"]["cycle_id"]
        assert units["manifest_revision"] == effective["receipt"]["after"]["manifest_revision"]
        assert units["counts"]["listings"] == len(symbols) - 1
        counts = units["counts"]
        assert counts["processed"] == counts["updated"] + counts["failed"]
        reuse = effective["partition_reuse"]
        assert reuse["source"] == "RECEIPT_PANEL"
        assert reuse["panel_hash"] == effective["receipt"]["after"]["panel_hash"]
        assert all(
            symbols[0] not in fetched
            for fetched, _start, _end in provider.calls[calls_at_departure:]
        )
        overlap = normal_refresh_plan(
            date.fromisoformat(document["experiment"]["sessions"]["end"]), NOW.date()
        ).fetch_start
        assert provider.calls and all(start >= overlap for _symbols, start, _end in provider.calls)
        capture = _json(
            live,
            "/api/research-inputs/plan",
            method="POST",
            payload={"research_input_id": anchor.input_id},
        )
        assert capture["status"] == "CONFIRMATION_REQUIRED", capture
        bridge = InstalledAgent(live.operations)
        refusal = json.loads(
            bridge.invoke(
                PortfolioResearchAgentRequest(
                    operation="RESEARCH_INPUT_CONFIRM",
                    research_input_plan_hash=capture["plan_hash"],
                )
            )
        )
        assert "human_confirmation_required" in str(refusal)
        calls_before = list(provider.calls)
        verify = live.operations.input_capture.verify_stage

        def crash_after_publish(**kwargs):
            verify(**kwargs)
            raise ConnectionError("interrupted after input version publication")

        live.operations.input_capture.verify_stage = crash_after_publish
        captured = _json(
            live,
            "/api/research-inputs/confirm",
            method="POST",
            payload={"research_input_plan_hash": capture["plan_hash"]},
        )
        capture_task = UUID(captured["task_id"])
        _finish(live, "input-capture", timeout=300)
        state = _json(live, f"/api/status?task_id={capture_task}")
        assert state["lifecycle"] == "RECOVERY_REQUIRED", state
        assert provider.calls == calls_before
        publication = live.operations.input_capture.revisions.for_task(capture_task)
        assert publication is not None
        assert publication.binding_hash != anchor.binding_hash
        assert (root / "research-workspace.json").read_bytes() == manifest_bytes

    with pytest.MonkeyPatch.context() as scoped:

        def forbidden(*_args, **_kwargs):
            raise AssertionError("recovery re-entered data/capture work")

        scoped.setattr(input_capture, "describe_input_source", forbidden)
        scoped.setattr(input_capture, "publish_prepared_factor_inputs", forbidden)
        with LocalPortfolioWebSession.from_workspace(root, clock=lambda: effective_day) as live:
            assert live.resumed_task_ids == (capture_task,)
            _finish(live, "capture-recovery", timeout=120)
            assert _json(live, f"/api/status?task_id={capture_task}")["lifecycle"] == "SUCCEEDED"
            assert live.operations.input_capture.revisions.for_task(capture_task) == publication

    with LocalPortfolioWebSession.from_workspace(root, clock=lambda: effective_day) as live:
        repeat = _json(
            live,
            "/api/research-inputs/plan",
            method="POST",
            payload={"research_input_id": anchor.input_id},
        )
        assert repeat["status"] == "REUSED_EXACT" and repeat["task_id"] is None
        again = _json(live, f"/api/experiments/readback?task_id={old_task}")
        assert _facts(again) == _facts(old_report)
        assert _json(live, f"/api/experiments/export?task_id={old_task}") == old_export
        draft = _json(
            live,
            "/api/experiments/draft",
            method="POST",
            payload={
                "task_id": old_task,
                "research_input_id": anchor.input_id,
                "input_binding_hash": publication.binding_hash,
            },
        )
        assert draft["document"]["factor"] == document["factor"]
        assert draft["document"]["experiment"]["sessions"] == document["experiment"]["sessions"]
        next_plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={
                "research_input_id": anchor.input_id,
                "input_binding_hash": publication.binding_hash,
                "origin_task_id": old_task,
                "experiment_document": draft["document"],
            },
        )
        assert next_plan["status"] == "PLANNED", next_plan
        new_run = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": next_plan["plan_hash"]},
        )
        _finish(live, "updated-factor", timeout=240)
        new_report = _json(live, f"/api/experiments/readback?task_id={new_run['task_id']}")
        assert new_report["status"] == "EXPERIMENT_PUBLISHED"
        assert new_report["origin_task_id"] == old_task
        assert new_report["program"] != old_report["program"]
        assert (root / "research-workspace.json").read_bytes() == manifest_bytes
    return {
        "anchor": anchor.model_dump(mode="json"),
        "publication": publication.model_dump(mode="json"),
        "old_task": old_task,
        "old_export": old_export,
    }


@pytest.fixture(scope="module")
def revision_seed(request, tmp_path_factory):
    return session_workspace(
        tmp_path_factory,
        "input_revision",
        lambda root: _build_revision_journey(root, request.getfixturevalue("prepared")),
    )


@pytest.fixture
def revision_journey(revision_seed, tmp_path):
    source, metadata = revision_seed
    root = copy_workspace(source, tmp_path / "input-revision")
    return (
        root,
        ResearchWorkspaceExperimentInput.model_validate(metadata["anchor"]),
        ResearchInputRevision.model_validate(metadata["publication"]),
        metadata["old_task"],
        deepcopy(metadata["old_export"]),
    )


def test_update_capture_copy_and_recovery_leave_defaults_and_reports_unchanged(
    revision_journey, monkeypatch
):
    root, anchor, publication, _old_task, _export = revision_journey
    bundle = read_factor_bundle(root, publication.binding_hash)
    _, artifacts = factor_input_paths(root, publication.binding_hash)
    resolver = ArtifactResolver(artifacts)
    panel = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
    )
    assert panel["active_listing_count"] == 120  # Historical union includes the leaver.
    assert panel["safe_summary"]["membership"]["as_of_member_count"] == 119
    assert [item["member_count"] for item in panel["safe_summary"]["membership"]["epochs"]] == [
        120,
        119,
    ]
    market = MarketDataRepository(root)
    bootstrap = market.universe_bootstrap("us-current-index-research")
    prior = market.load_universe_manifest_revision(bootstrap.manifest_revision)
    leaver = next(item for item in prior.listings if item.symbol == "F000")
    rows = [
        row
        for batch in FeaturePanelReader(resolver).batches(
            FeaturePanelReadRequest(
                manifest_ref=resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash),
                start_session=NOW.date(),
                end_session=(NOW + timedelta(days=1)).date(),
                factor_columns=("beta_63",),
            )
        )
        for row in batch.to_pylist()
        if row["listing_id"] == leaver.listing_id
    ]
    assert len(rows) == 1 and rows[0]["session_date"] == NOW.date()
    assert math.isfinite(rows[0]["beta_63"])
    with LocalPortfolioWebSession.from_workspace(root, clock=lambda: NOW) as live:
        rows = _json(live, "/api/research-inputs")["inputs"][0]["versions"]
        assert {v["binding_hash"] for v in rows} == {anchor.binding_hash, publication.binding_hash}
        history = _json(
            live,
            "/api/research-history?"
            + f"research_input_id={anchor.input_id}&history_kind=factor.screening-development",
        )
        assert history["status"] == "AVAILABLE"
        assert {v["input_binding_hash"] for v in history["entries"]} == {
            anchor.binding_hash,
            publication.binding_hash,
        }
        assert any(v["task_id"] == str(_old_task) for v in history["entries"])
        bridge = InstalledAgent(live.operations)
        assert json.loads(
            bridge.invoke(PortfolioResearchAgentRequest(operation="RESEARCH_INPUTS"))
        ) == _json(live, "/api/research-inputs")

        # Discovery isolates an unreadable publication by filename identity and keeps the
        # declared input's healthy history visible. Neither file's body can safely supply an
        # input_id. Strict selectors still use publications() and are exercised separately.
        publication_dir = root / "research-inputs" / "publications"
        malformed = publication_dir / f"{'e' * 64}.json"
        unavailable = publication_dir / f"{'f' * 64}.json"
        malformed.write_text("{", encoding="utf-8", newline="\n")
        unavailable.write_text("{}", encoding="utf-8", newline="\n")
        original_read_text = Path.read_text

        def fail_one_publication(path, *args, **kwargs):
            if path.resolve() == unavailable.resolve():
                raise OSError("simulated publication read failure")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_one_publication)
        try:
            discovered = _json(live, "/api/research-inputs")
            assert discovered["status"] == "AVAILABLE"
            assert {
                version["binding_hash"]
                for group in discovered["inputs"]
                for version in group["versions"]
            } == {anchor.binding_hash, publication.binding_hash}
            versions_by_binding = {
                version["binding_hash"]: version
                for group in discovered["inputs"]
                for version in group["versions"]
            }
            assert versions_by_binding[anchor.binding_hash]["available"]
            assert not versions_by_binding[publication.binding_hash]["available"]
            assert versions_by_binding[publication.binding_hash]["unreadable"] == (
                "research_input.publication_invalid"
            )
            refusals = {row["publication_hash"]: row for row in discovered["refusals"]}
            assert set(refusals) == {"e" * 64, "f" * 64}
            assert refusals["e" * 64]["failure_code"] == "research_input.publication_invalid"
            assert refusals["f" * 64]["failure_code"] == "research_input.publication_unavailable"
            for row in refusals.values():
                assert "input_id" not in row
                assert row["status"] == "REFUSED" and row["detail"]
                assert set(row["next_requests"]) == {"inputs", "workspace", "backups", "storage"}
                assert row["next_requests"]["inputs"] == {"operation": "RESEARCH_INPUTS"}
                assert row["next_requests"]["workspace"] == {"operation": "WORKSPACE_SHOW"}
                assert row["next_requests"]["backups"] == {"operation": "WORKSPACE_BACKUPS"}
                assert row["next_requests"]["storage"] == {"operation": "STORAGE_READBACK"}
        finally:
            monkeypatch.setattr(Path, "read_text", original_read_text)
            malformed.unlink(missing_ok=True)
            unavailable.unlink(missing_ok=True)

        storage = _json(live, "/api/workspace/storage")
        roots = {v["binding_hash"]: v["roots"] for v in storage["inputs"]}
        assert "CURRENT_ACTIVE" in roots[anchor.binding_hash]
        assert "CURRENT_RESEARCH_VERSION" in roots[publication.binding_hash]
        # Only chunks no generation published are offered (V207, V311); both inputs' stay.
        targets = _json(live, "/api/workspace/storage/plan", method="POST", payload={})["targets"]
        assert not set(targets) & published_panel_chunks(root)


def test_a_saved_capture_continues_on_its_published_revision(revision_journey, capsys):
    """contract (V523/V515): capture readback offers controls on its newly published input,
    and a saved JSON or YAML answer carries that exact binding rather than the old anchor."""
    from alphalattice.interface.local_application.cli_contract import command_table
    from alphalattice.interface.local_application.client import continuation
    from alphalattice.protocols.research_authoring.selection import dump_declaration
    from run_alphalattice import main

    root, anchor, publication, _old_task, _export = revision_journey
    with LocalPortfolioWebSession.from_workspace(root, clock=lambda: NOW) as live:
        answer = _json(live, f"/api/research-inputs/readback?task_id={publication.task_id}")
        allowed = frozenset(command_table()["fields"]["EXPERIMENT_CONTROLS"]["allowed"])
        for format in ("json", "yaml"):
            saved = root / f"capture-answer.{format}"
            text = json.dumps(answer) if format == "json" else dump_declaration(answer)
            saved.write_text(text, encoding="utf-8", newline="\n")
            assert (
                main(
                    [
                        "--workspace",
                        str(root),
                        "--view",
                        "full",
                        "study",
                        "controls",
                        "--from",
                        str(saved),
                    ]
                )
                == 0
            )
            cli_controls = json.loads(capsys.readouterr().out)["data"]
            assert (
                cli_controls["input_binding_hash"]
                == publication.binding_hash
                != anchor.binding_hash
            )
            assert cli_controls["input_id"] == publication.input_id
            assert answer["research_input_id"] == publication.input_id
            assert answer["input_binding_hash"] == publication.binding_hash != anchor.binding_hash
            request = continuation("EXPERIMENT_CONTROLS", saved, {}, allowed)
            assert request == answer["next_requests"]["controls"]
            controls = json.loads(
                InstalledAgent(live.operations).invoke(
                    PortfolioResearchAgentRequest.model_validate(request)
                )
            )
            assert controls["input_binding_hash"] == publication.binding_hash
            assert controls["input_id"] == publication.input_id
            assert (
                cli_controls["template"]["experiment"]["sessions"]
                == controls["template"]["experiment"]["sessions"]
            )


def test_unregistered_or_tampered_versions_refuse_before_work(revision_journey, monkeypatch):
    root, anchor, publication, old_task, old_export = revision_journey
    with LocalPortfolioWebSession.from_workspace(root, clock=lambda: NOW) as live:
        registry = live.session.task_control_registry
        before = len(registry.tasks())
        code, _headers, _body = _request(
            live,
            "/api/experiments/controls"
            f"?research_input_id={anchor.input_id}&input_binding_hash={'f' * 64}",
        )
        assert code == 200  # typed refusal body, not a missing route
        bad = _json(
            live,
            "/api/experiments/controls"
            f"?research_input_id={anchor.input_id}&input_binding_hash={'f' * 64}",
        )
        assert bad["status"] == "REFUSED" and "version_not_admitted" in bad["failure_code"]
        bundle = read_factor_bundle(root, publication.binding_hash)
        path = root / "research-inputs" / bundle.binding_hash / "manifest.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(bundle.binding_hash.encode(), b"0" * 64))
            with pytest.raises(ValueError):
                ResearchInputRevisions(live.session).select(anchor.input_id, bundle.binding_hash)
            tampered = _json(
                live,
                "/api/experiments/controls"
                f"?research_input_id={anchor.input_id}&input_binding_hash={bundle.binding_hash}",
            )
            assert tampered["status"] == "REFUSED"
            assert tampered["failure_code"] != "research_input.manifest_missing"
        finally:
            path.write_bytes(original)
        assert len(registry.tasks()) == before
        manifest_bytes = (root / "research-workspace.json").read_bytes()
        for binding in (anchor.binding_hash, publication.binding_hash):
            manifest = root / "research-inputs" / binding / "manifest.json"
            held_manifest = manifest.with_suffix(".held")
            original_manifest = manifest.read_bytes()
            manifest.rename(held_manifest)
            try:
                catalog = ResearchInputRevisions(live.session).versions()
                unavailable = next(
                    (
                        version
                        for group in catalog["inputs"]
                        if group["input_id"] == anchor.input_id
                        for version in group["versions"]
                        if version["binding_hash"] == binding
                    ),
                    None,
                )
                if unavailable is None:
                    # The publication also reads this manifest; no version is inferred
                    # when that strict publication check cannot establish its binding.
                    assert binding == publication.binding_hash
                    assert any(
                        row["failure_code"] == "research_input.publication_unavailable"
                        for row in catalog["refusals"]
                    )
                    with pytest.raises(FileNotFoundError) as failure:
                        live.operations.experiments.controls(anchor.input_id, binding)
                    assert failure.value.filename == str(manifest)
                    continue
                assert unavailable["unreadable"] == "research_input.manifest_missing"
                selectors = [
                    f"research_input_id={anchor.input_id}&input_binding_hash={binding}",
                    f"input_binding_hash={binding}",
                ]
                if binding == anchor.binding_hash:
                    selectors.extend((f"research_input_id={anchor.input_id}", ""))
                for kind in (
                    "factor.screening-development",
                    "alpha.model-development",
                    "risk.covariance-development",
                ):
                    for selector in selectors:
                        refused = _json(
                            live,
                            f"/api/experiments/controls?experiment_kind={kind}&{selector}",
                        )
                        assert refused["status"] == "REFUSED"
                        assert refused["failure_code"] == unavailable["unreadable"]
                        assert refused["input_id"] == anchor.input_id
                        assert refused["input_binding_hash"] == binding
                        assert refused["detail"] == unavailable["detail"]
                        assert (
                            refused["next_requests"]
                            == unavailable["next_requests"]
                            == {"storage": {"operation": "STORAGE_READBACK"}}
                        )
                # Only one exact unavailable row can explain the selection failure.
                duplicate = deepcopy(catalog)
                duplicate["inputs"][0]["versions"].append(deepcopy(unavailable))
                unclassified = deepcopy(catalog)
                for version in unclassified["inputs"][0]["versions"]:
                    if version["binding_hash"] == binding:
                        version["unreadable"] = "research_input.bundle_unreadable"
                for unreadable_catalog in (duplicate, unclassified, {"inputs": []}):
                    with monkeypatch.context() as patch:
                        patch.setattr(
                            ResearchInputRevisions,
                            "versions",
                            lambda _self, value=unreadable_catalog: value,
                        )
                        with pytest.raises(FileNotFoundError) as failure:
                            live.operations.experiments.controls(None, binding)
                    assert failure.value.filename == str(manifest)

                def catalog_unavailable(_self):
                    raise OSError("catalog unavailable")

                with monkeypatch.context() as patch:
                    patch.setattr(ResearchInputRevisions, "versions", catalog_unavailable)
                    with pytest.raises(FileNotFoundError) as failure:
                        live.operations.experiments.controls(None, binding)
                assert failure.value.filename == str(manifest)
            finally:
                held_manifest.rename(manifest)
                assert manifest.read_bytes() == original_manifest
                restored = _json(
                    live,
                    "/api/experiments/controls?experiment_kind=risk.covariance-development"
                    f"&research_input_id={anchor.input_id}&input_binding_hash={binding}",
                )
                assert restored["status"] == "READY"
                assert restored["input_binding_hash"] == binding
        assert len(registry.tasks()) == before
        assert (root / "research-workspace.json").read_bytes() == manifest_bytes
        # Missing numerical input prevents execution, not reopening immutable evidence.
        database = root / "research-inputs" / anchor.binding_hash / "source/market-data.duckdb"
        held = database.with_suffix(".held")
        database.rename(held)
        try:
            assert _json(live, f"/api/experiments/export?task_id={old_task}") == old_export
            refusal = _json(live, "/api/experiments/controls")
            assert refusal["status"] == "REFUSED"
            assert refusal["failure_code"] != "research_input.manifest_missing"
        finally:
            held.rename(database)


def _facts(body: dict) -> dict:  # type: ignore[type-arg]
    """A study answer's facts, without how this read proved them (`verification_basis`, L1)."""

    return {key: value for key, value in body.items() if key != "verification_basis"}
