"""Workspace-only installation of one real-authority Evidence/CRO runtime."""

from __future__ import annotations

import ast
import json
import re
import shlex
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.product_host.composition.evidence_review_application import (
    SETUP_OPTIONS,
    evidence_setup,
)
from alphalattice.control.product_host.composition.evidence_review_workspace import (
    admit_evidence_review_workspace,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceManifest,
)
from tests.alternative_evidence_desk.review_package import (
    CAPABILITY_HASH,
    _package,
)


@pytest.fixture
def installer_universe(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The installer's workspace universe, ticker to listing id, as each test sets it.

    This module's one patch of the installer's universe: its real sources, a workspace's
    risk surface or a selected research input, are not built here (TE5).
    """
    from scripts import materialize_evidence_cro_authority as setup

    chosen: dict[str, str] = {}
    monkeypatch.setattr(setup, "_universe_listing_ids", lambda _workspace: chosen)
    return chosen


def test_source_setup_preflights_acquires_and_preserves_prior_authority(
    tmp_path, monkeypatch, installer_universe
):
    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.control.product_host.composition.evidence_review_application import (
        PACKAGE_RULE,
    )
    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        EvidenceReviewWorkspaceManifest,
        RecordedEvidenceDocumentBundle,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )
    from tests.alternative_evidence_desk.sec_fixture_transport import SecFixtureTransport

    old_binding, _ = _package(tmp_path)
    original = ResearchWorkspaceManifest.create(
        workspace_id="source-setup",
        default_strategy_package_id="fixture-package",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
    ).with_bindings(evidence_review=old_binding)
    publish_research_workspace_manifest(tmp_path, original)
    old_path = tmp_path / old_binding.relative_path
    old_bytes = old_path.read_bytes()
    transport = SecFixtureTransport()
    installer_universe.update({"AAPL": "US-AAPL", "MSFT": "US-MSFT"})
    monkeypatch.setattr(
        setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash=CAPABILITY_HASH),
    )
    monkeypatch.setattr(setup, "HttpxSecOfficialTransport", lambda **_: nullcontext(transport))
    args = setup._parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--semantic-model",
            str(tmp_path / "authority/semantic-model"),
            "--acquire-sec",
            "--entities",
            "AAPL",
            "--preflight",
            "--install",
        ]
    )
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    preview = setup.materialize(args)
    assert preview["status"] == "EVIDENCE_SOURCE_PREFLIGHT" and not preview["acquisition_performed"]
    assert preview["evidence_as_of"] is None and preview["accession_scopes"] == {}
    args.entities = ["AAPL", "MSFT"]
    args.accessions = ["0000320193-26-000001"]
    with pytest.raises(ValueError, match="accession_scope_requires_one_entity"):
        setup.materialize(args)
    args.entities = ["AAPL"]
    assert setup.materialize(args)["accession_scopes"] == {"AAPL": ["0000320193-26-000001"]}
    args.accessions = None
    assert not preview["sec_contact_configured"] and transport.calls == []
    assert old_path.read_bytes() == old_bytes
    args.entities = ["NOT-IN-UNIVERSE"]
    with pytest.raises(ValueError, match="entity_scope_outside_universe"):
        setup.materialize(args)
    args.entities = ["AAPL"]
    with monkeypatch.context() as unavailable:
        unavailable.setattr(
            setup,
            "probe_hybrid_retrieval_capabilities",
            lambda *_: SimpleNamespace(status="UNAVAILABLE"),
        )
        with pytest.raises(RuntimeError, match="semantic_capability_not_ready"):
            setup.materialize(args)
    assert transport.calls == []
    args.preflight = False
    with pytest.raises(ValueError, match="explicit_sec_network_consent_required"):
        setup.materialize(args)
    args.network_consent = True
    with pytest.raises(ValueError, match="workspace_network_not_allowed"):
        setup.materialize(args)
    assert transport.calls == []
    from alphalattice.control.workspace_runtime.network_access import (
        network_access,
        set_network_access,
    )

    set_network_access(tmp_path, enabled=True)
    # Exercise the typed control with a fixture transport while the test process stays offline.
    monkeypatch.setattr(
        setup, "network_access", lambda workspace: network_access(workspace, environment={})
    )
    args.network_consent = True
    with pytest.raises(ValueError, match="sec_user_agent_required"):
        setup.materialize(args)
    assert transport.calls == []
    monkeypatch.setenv("SEC_USER_AGENT", "Recorded protocol QA fixture@example.invalid")
    # The cutoff is declared, timezone-aware and never in the future; named
    # accessions need exactly one issuer and the official accession form.
    for invalid in ("2026-08-12", "2026-08-12T00:00:00", "not-a-time"):
        args.evidence_as_of = invalid
        with pytest.raises(ValueError, match="evidence_as_of_invalid"):
            setup.materialize(args)
    args.evidence_as_of = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="evidence_as_of_in_the_future"):
        setup.materialize(args)
    args.evidence_as_of = "2026-08-12T00:00:00+00:00"
    args.accessions = ["0000320193-26-000001", "bad"]
    with pytest.raises(ValueError, match="accession_scope_invalid"):
        setup.materialize(args)
    args.accessions = None
    assert transport.calls == []
    started = datetime.now(UTC)
    result = setup.materialize(args)
    assert len(transport.calls) == 3  # Registry, submissions and one recorded filing.
    assert result["installed"] and result["model_profile_hash"] is None
    assert result["acquisition"]["evidence_as_of"] == "2026-08-12T00:00:00+00:00"
    assert result["acquisition"]["http_attempts"] is None  # the fixture transport counts nothing
    manifest_path = tmp_path / result["binding"]["relative_path"]
    manifest = EvidenceReviewWorkspaceManifest.model_validate_json(manifest_path.read_bytes())
    documents_path = tmp_path / manifest.recorded_documents.relative_path
    bundle = RecordedEvidenceDocumentBundle.model_validate_json(documents_path.read_bytes())
    document = bundle.documents[0]
    assert document.captured_at >= started
    assert document.available_at >= document.captured_at > document.published_at
    # The recorded import keeps the original's acceptance and date precision
    # beside its capture-bounded availability: the guard stands, the
    # original time is not lost to it.
    assert document.accepted_at == datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    assert document.published_precision == "DATE"
    assert document.published_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert document.available_at > document.accepted_at
    assert manifest.model_profile is None and old_path.read_bytes() == old_bytes
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in manifest_path.parent.iterdir()}
    args.acquire_sec = False
    args.source_artifact_root = Path(result["source_artifact_root"])
    args.source_set_hash = result["source_set_hash"]
    replay = setup.materialize(args)
    # The first install replaced the workspace's prior package and says so, with what that does
    # to the packets prepared under it; the replay installs the same package again and
    # replaces nothing.
    assert result["replaced"] == old_binding.relative_path
    assert result["replaced_package_rule"] == PACKAGE_RULE
    assert "replaced" not in replay
    said = {"acquisition", "replaced", "replaced_package_rule"}
    assert {k: v for k, v in replay.items() if k != "acquisition"} == {
        k: v for k, v in result.items() if k not in said
    }
    assert result["acquisition"]["network_calls"] == 3
    assert replay["acquisition"]["network_calls"] == 0
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files} == files
    assert len(transport.calls) == 3
    documents_path.write_bytes(documents_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable_package_conflict"):
        setup.materialize(args)
    assert old_path.read_bytes() == old_bytes


def test_recorded_source_conversion_does_not_claim_a_new_sec_capture(
    tmp_path, monkeypatch, installer_universe
):
    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        EvidenceReviewWorkspaceManifest,
        RecordedEvidenceDocumentBundle,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )
    from tests.alternative_evidence_desk.document_intelligence_support import _registry, _request
    from tests.alternative_evidence_desk.planted_corpus import _NOW, _recorded_document, _runtime

    runtime = _runtime(tmp_path / "source")
    try:
        _snapshot, source_set = runtime.acquire_recorded(
            request=_request(("AAPL", "MSFT")),
            registry=_registry(("AAPL", "MSFT")),
            documents=(_recorded_document(),),
            published_at=_NOW,
        )
        source_root = runtime.artifacts.root
    finally:
        runtime.close()
    target = tmp_path / "target"
    publish_research_workspace_manifest(
        target,
        ResearchWorkspaceManifest.create(
            workspace_id="recorded-setup",
            default_strategy_package_id="fixture-package",
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(),
        ),
    )
    installer_universe.update({"AAPL": "US-AAPL"})
    monkeypatch.setattr(
        setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash=CAPABILITY_HASH),
    )
    args = setup._parser().parse_args(
        [
            "--workspace",
            str(target),
            "--semantic-model",
            str(target / "models"),
            "--source-artifact-root",
            str(source_root),
            "--source-knowledge-root",
            str(tmp_path / "source" / "workspace"),
            "--source-set-hash",
            source_set.source_set_hash,
        ]
    )
    with pytest.raises(ValueError, match="materialization_issuer_coverage_incomplete"):
        setup.materialize(args)
    args.entities = ["AAPL"]
    result = setup.materialize(args)
    manifest = EvidenceReviewWorkspaceManifest.model_validate_json(
        (target / result["binding"]["relative_path"]).read_bytes()
    )
    bundle = RecordedEvidenceDocumentBundle.model_validate_json(
        (target / manifest.recorded_documents.relative_path).read_bytes()
    )
    document = bundle.documents[0]
    assert document.captured_at == _recorded_document().captured_at
    assert "Admitted recorded-local source; no new SEC acquisition." in document.limitations
    assert not any("Captured SEC EDGAR filing" in value for value in document.limitations)
    assert result["installed"] is False
    # A fresh install asks for the integrated selection, the one current
    # method; the retired production plan and candidate selection are no
    # choice (first-release integration T5).
    assert manifest.matter_selection is not None
    assert manifest.matter_selection.method == "INTEGRATED_TOPIC_ROUTING"
    assert result["matter_selection"]["method"] == "INTEGRATED_TOPIC_ROUTING"
    for retired in ("PRODUCTION_PLAN_PREFIX", "CANDIDATE_UNMET_NEEDS"):
        with pytest.raises(SystemExit):
            setup._parser().parse_args(["--workspace", str(target), "--matter-selection", retired])
    args.source_set_hash = "../escape"
    with pytest.raises(ValueError, match="source_set_hash_invalid"):
        setup.materialize(args)


CUTOFF = "2026-08-12T00:00:00+00:00"


def _rr5f_book(
    tmp_path, monkeypatch, capsys, universe, *, filed: int, quiet: int, failed: int
) -> Any:
    """RR5f's first Evidence setup in small: a book whose `filed` holdings filed an 8-K
    inside the 30-day window before the cutoff, whose `quiet` ones filed a 10-Q a quarter
    before it, and whose `failed` ones filed inside it a body the source fails to serve, the
    installer's universe set to them. The workspace, the holdings by kind, the transport, and
    the real setup entry run on argv."""
    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )
    from alphalattice.control.workspace_runtime.network_access import (
        network_access,
        set_network_access,
    )
    from tests.alternative_evidence_desk.sec_scenario_transport import (
        ScenarioFiling,
        SecScenarioTransport,
        filing_body,
    )

    tickers = tuple(f"QB{index:02d}" for index in range(1, filed + quiet + failed + 1))
    kinds = {
        "filed": tickers[:filed],
        "quiet": tickers[filed : filed + quiet],
        "failed": tickers[filed + quiet :],
    }
    ciks = {ticker: f"{9100 + index:010d}" for index, ticker in enumerate(tickers, start=1)}
    filings: dict[str, list[ScenarioFiling]] = {}
    bodies: dict[str, bytes] = {}
    for ticker in tickers:
        on, form = ("2026-05-01", "10-Q") if ticker in kinds["quiet"] else ("2026-08-01", "8-K")
        cik = ciks[ticker]
        filing = ScenarioFiling(
            f"{cik}-26-000001", form, on, f"{on}T20:00:00.000Z", f"{ticker.casefold()}.htm"
        )
        filings[cik] = [filing]
        bodies[SecScenarioTransport.locator(cik, filing)] = filing_body(filing.accession)
    transport = SecScenarioTransport(
        registry={ticker: (ciks[ticker], f"{ticker} Inc.") for ticker in tickers},
        filings=filings,
        bodies=bodies,
    )
    for ticker in kinds["failed"]:
        transport.fail_body(
            ciks[ticker],
            filings[ciks[ticker]][0],
            lambda: RuntimeError("scenario: body refused"),
        )
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    publish_research_workspace_manifest(
        workspace,
        ResearchWorkspaceManifest.create(
            workspace_id="rr5f-setup",
            default_strategy_package_id="fixture-package",
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(),
        ),
    )
    universe.clear()
    universe.update({ticker: f"US-{ticker}" for ticker in tickers})
    monkeypatch.setattr(
        setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash=CAPABILITY_HASH),
    )
    monkeypatch.setattr(setup, "HttpxSecOfficialTransport", lambda **_: nullcontext(transport))
    set_network_access(workspace, enabled=True)
    # The typed control with a fixture transport while the test process stays offline.
    monkeypatch.setattr(setup, "network_access", lambda at: network_access(at, environment={}))
    monkeypatch.setenv("SEC_USER_AGENT", "Recorded protocol QA fixture@example.invalid")
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")

    def run(*argv: str) -> tuple[int, dict[str, Any]]:
        code = setup.main(list(argv))
        return code, json.loads(capsys.readouterr().out)

    book = (
        "--workspace",
        str(workspace),
        "--semantic-model",
        str(workspace / "authority/semantic-model"),
    )
    acquire = (*book, "--acquire-sec", "--entities", *tickers, "--evidence-as-of", CUTOFF)
    return SimpleNamespace(
        workspace=workspace,
        tickers=tickers,
        kinds=kinds,
        ciks=ciks,
        filings=filings,
        transport=transport,
        book=book,
        acquire=(*acquire, "--network-consent", "--install"),
        run=run,
    )


def test_the_installer_leaves_quiet_holdings_out_of_its_floor_as_the_coverage_run_does(
    tmp_path, monkeypatch, capsys, installer_universe
):
    """requirement (RR5f): the first Evidence authority setup judges its floor as the
    coverage run does -- an issuer whose filing index at the cutoff shows nothing filed in the
    window holds nothing to count -- the floor, cutoff and source checks unchanged. Four holdings
    with filings and four quiet install at 0.60 and the answer names which were quiet; the
    recorded import of that acquisition judges alike. A book whose every holding was quiet has
    no document at all, refused in words with its numbers and no acquisition offered again."""

    book = _rr5f_book(
        tmp_path / "eight", monkeypatch, capsys, installer_universe, filed=4, quiet=4, failed=0
    )
    code, result = book.run(*book.acquire)
    assert code == 0, result
    assert result["installed"] and result["minimum_entity_coverage"] == 0.6
    assert result["admitted_document_count"] == 4
    standing = {
        "hold_a_document": list(book.kinds["filed"]),
        "filed_nothing": list(book.kinds["quiet"]),
        "failed": [],
    }
    assert result["issuer_coverage"] == standing
    code, replay = book.run(
        *book.book,
        "--source-artifact-root",
        result["source_artifact_root"],
        "--source-set-hash",
        result["source_set_hash"],
        "--install",
    )
    assert code == 0, replay
    assert replay["package_id"] == result["package_id"] and replay["issuer_coverage"] == standing

    quiet = _rr5f_book(
        tmp_path / "quiet", monkeypatch, capsys, installer_universe, filed=0, quiet=8, failed=0
    )
    code, refused = quiet.run(*quiet.acquire)
    numbers = "8 of 8 issuers filed nothing in the window, 0 failed"
    assert code == 2
    assert refused["failure_code"] == (
        f"evidence_review.materialization_has_no_admitted_documents:{numbers}"
    )
    assert numbers in refused["detail"]
    assert refused["next_action"] == "ACQUIRE_AGAIN_AT_THE_CUTOFF_OR_NAME_ISSUERS_THAT_FILED"
    # Acquiring again at the cutoff finds the same nothing: no acquisition is offered.
    assert set(refused["next_commands"]) == {"preflight"}


def test_a_failed_acquisition_counts_against_the_floor_and_its_refusal_offers_the_cutoff_again(
    tmp_path, monkeypatch, capsys, installer_universe
):
    """requirement (TE12): a quiet issuer leaves the share and a failed acquisition stays
    in it. Four holdings with filings, two quiet and two failed install at 0.60 -- four of six
    counted -- and the same book under a floor of 0.70 is refused with its numbers in words:
    covered, needed, quiet and failed. The refusal offers the same official acquisition again,
    bound to the one cutoff it counted at, and that request as printed, run once the source
    serves the two bodies, installs all six."""

    from alphalattice.interface.local_application.cli_contract import refusal_words

    book = _rr5f_book(tmp_path, monkeypatch, capsys, installer_universe, filed=4, quiet=2, failed=2)
    code, result = book.run(*book.acquire)
    assert code == 0, result
    assert result["issuer_coverage"] == {
        "hold_a_document": list(book.kinds["filed"]),
        "filed_nothing": list(book.kinds["quiet"]),
        "failed": list(book.kinds["failed"]),
    }
    code, refused = book.run(*book.acquire, "--minimum-entity-coverage", "0.7")
    numbers = "4 of 6 counted issuers hold a document, 5 needed, 2 filed nothing, 2 failed"
    assert code == 2
    assert refused["failure_code"] == (
        f"evidence_review.materialization_issuer_coverage_incomplete:{numbers}"
    )
    assert refused["detail"] == refusal_words(refused["failure_code"])["detail"]
    assert numbers in refused["detail"]
    assert refused["next_action"] == "RUN_THE_OFFERED_WAY_AT_ONE_CUTOFF"
    preflight = shlex.split(refused["next_commands"]["preflight"])
    acquire = shlex.split(refused["next_commands"]["acquire"])
    for line in (preflight, acquire):
        assert line[line.index("--evidence-as-of") + 1] == CUTOFF
        assert line[line.index("--entities") + 1 :][: len(book.tickers)] == list(book.tickers)
    assert "--preflight" in preflight and "--install" not in preflight
    assert "--preflight" not in acquire and {"--install", "--network-consent"} <= set(acquire)
    assert acquire[acquire.index("--minimum-entity-coverage") + 1] == "0.7"
    for ticker in book.kinds["failed"]:
        book.transport.heal_body(book.ciks[ticker], book.filings[book.ciks[ticker]][0])
    code, again = book.run(*acquire[2:])
    assert code == 0, again
    assert again["minimum_entity_coverage"] == 0.7 and again["admitted_document_count"] == 6
    assert again["issuer_coverage"]["failed"] == []
    assert again["issuer_coverage"]["filed_nothing"] == list(book.kinds["quiet"])


def test_a_recorded_import_names_the_root_it_could_not_read_and_its_own_check_is_its_way_on(
    tmp_path, monkeypatch, capsys, installer_universe
):
    """requirement (RR5f): a recorded import given the artifact store's parent as its root
    is refused naming the option it read, the file it failed on relative to that root and what
    the option should hold, no absolute path, and its way on is the import's own check, offline,
    offered at the one directory below that root holding the named source set. That check, run
    as printed, reads and judges without installing; the corrected root installs. A recorded
    import short of its floor is offered the import's check, never an acquisition or the network
    decision."""

    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )
    from alphalattice.interface.local_application.cli_contract import refusal_words

    book = _rr5f_book(tmp_path, monkeypatch, capsys, installer_universe, filed=4, quiet=2, failed=2)
    code, acquired = book.run(*book.acquire)
    assert code == 0, acquired
    store = Path(acquired["source_artifact_root"])
    assert store.name == "alternative-evidence"
    bound = read_research_workspace_manifest(book.workspace).evidence_review
    source = ("--source-set-hash", acquired["source_set_hash"], "--entities", *book.tickers)

    code, refused = book.run(
        *book.book, "--source-artifact-root", str(store.parent), *source, "--install"
    )
    assert code == 2
    assert refused["failure_code"] == "evidence_review.source_file_unavailable:source_artifact_root"
    assert refused["detail"] == refusal_words(refused["failure_code"])["detail"]
    assert refused["location"] == {
        "option": "source_artifact_root",
        "file": f"source-document-sets/{acquired['source_set_hash']}.json",
        "expected": SETUP_OPTIONS["source_artifact_root"],
    }
    assert any(cause["kind"] == "OS" and cause["errno"] == 2 for cause in refused["causes"])
    assert "next_requests" not in refused, "an offline import is offered no network decision"
    assert set(refused["next_commands"]) == {"preflight"}
    check = shlex.split(refused["next_commands"]["preflight"])
    assert check[check.index("--source-artifact-root") + 1] == str(store)
    assert "--preflight" in check and "--install" not in check
    code, checked = book.run(*check[2:])
    assert code == 0, checked
    assert (checked["status"], checked["source"], checked["installed"]) == (
        "EVIDENCE_SOURCE_PREFLIGHT",
        "RECORDED_IMPORT",
        False,
    )
    assert checked["issuer_coverage"] == acquired["issuer_coverage"]
    assert read_research_workspace_manifest(book.workspace).evidence_review == bound

    code, short = book.run(
        *book.book,
        "--source-artifact-root",
        str(store),
        *source,
        "--minimum-entity-coverage",
        "0.7",
        "--install",
    )
    assert code == 2, short
    assert short["failure_code"].startswith(
        "evidence_review.materialization_issuer_coverage_incomplete:"
    )
    assert "next_requests" not in short and set(short["next_commands"]) == {"preflight"}
    assert "--acquire-sec" not in short["next_commands"]["preflight"]
    code, installed = book.run(
        *book.book, "--source-artifact-root", str(store), *source, "--install"
    )
    assert code == 0, installed
    assert installed["installed"] and installed["package_id"] == acquired["package_id"]


def test_a_recorded_import_whose_knowledge_root_lacks_an_object_names_it_under_that_option(
    tmp_path, monkeypatch, capsys, installer_universe
):
    """Requirement: a recorded import whose knowledge root exists but lacks an object
    the source set names is refused by that option, naming the object relative to the root and
    what the option must hold, its way on the import's own check and nothing created in the
    root it was given; a root that is not there is refused the same way, at the root itself,
    and not made; the acquiring workspace's own store installs."""

    book = _rr5f_book(tmp_path, monkeypatch, capsys, installer_universe, filed=4, quiet=0, failed=0)
    code, acquired = book.run(*book.acquire)
    assert code == 0, acquired
    source = (
        "--source-artifact-root",
        acquired["source_artifact_root"],
        "--source-set-hash",
        acquired["source_set_hash"],
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    code, refused = book.run(
        *book.book, *source, "--source-knowledge-root", str(elsewhere), "--install"
    )
    assert code == 2, refused
    assert refused["failure_code"] == (
        "evidence_review.source_file_unavailable:source_knowledge_root"
    )
    location = refused["location"]
    assert (location["option"], location["expected"]) == (
        "source_knowledge_root",
        SETUP_OPTIONS["source_knowledge_root"],
    )
    assert re.fullmatch(r"\.system/source-objects/[0-9a-f]{64}", location["file"]), location
    assert any(cause["kind"] == "OS" and cause["errno"] == 2 for cause in refused["causes"])
    assert "next_requests" not in refused and set(refused["next_commands"]) == {"preflight"}
    assert list(elsewhere.iterdir()) == [], "nothing is created in the root the import named"
    absent = tmp_path / "absent"
    code, missing = book.run(
        *book.book, *source, "--source-knowledge-root", str(absent), "--install"
    )
    assert code == 2 and missing["location"]["file"] == "." and not absent.exists()
    code, installed = book.run(*book.book, *source, "--install")
    assert code == 0, installed


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
"""A value of each option the setup refuses on, through its real entry; the
knowledge root's own case is the recorded import's above."""


@pytest.mark.parametrize("option", sorted(_OPTION_CASES))
def test_every_option_the_setup_refuses_on_names_what_it_must_hold_from_the_offers_table(
    option, tmp_path, monkeypatch, capsys, installer_universe
):
    """requirement (TE12): a setup refused on what an option holds names that
    option and what it must hold, from the one table the Host's offer states (`SETUP_OPTIONS`),
    whichever option it is."""

    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    publish_research_workspace_manifest(
        workspace,
        ResearchWorkspaceManifest.create(
            workspace_id="option-refusals",
            default_strategy_package_id="fixture-package",
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(),
        ),
    )
    installer_universe.update({"AAPL": "US-AAPL"})
    monkeypatch.setattr(
        setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash=CAPABILITY_HASH),
    )
    pack = tmp_path / "outside" if option == "semantic_model" else workspace / "authority/pack"
    roots = ("--source-artifact-root", str(tmp_path / "artifacts")) if "source" in option else ()
    argv = ["--workspace", str(workspace), "--semantic-model", str(pack), *roots]
    assert setup.main([*argv, *_OPTION_CASES[option]]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_code"].startswith("evidence_review."), payload
    location = payload["location"]
    assert (location["option"], location["expected"]) == (option, SETUP_OPTIONS[option])


def test_the_setup_offer_and_its_refusals_read_one_table_of_what_each_option_must_hold() -> None:
    """requirement (TE12): the Host's setup offer states, for each option it leaves to
    the person, what it must hold, from `SETUP_OPTIONS`, as every refusal on an option names it;
    the installer keeps no text of its own, every option it refuses on is in the table, and the
    table holds exactly the options offered or refused on."""

    from scripts import materialize_evidence_cro_authority as setup

    offered: set[str] = set()
    for authored in (False, True):
        for choice in evidence_setup(Path("workspace"), authored=authored)["authority"]["choose"]:
            for flag in re.findall(r"--[a-z-]+", choice["arg"]):
                option = flag[2:].replace("-", "_")
                assert f"{flag} names {SETUP_OPTIONS[option]}" in choice["why"], choice
                offered.add(option)
    assert setup.SETUP_OPTIONS is SETUP_OPTIONS and not hasattr(setup, "SOURCE_ROOTS")
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
    roots = {"source_artifact_root", "source_knowledge_root"}
    assert set(SETUP_OPTIONS) == offered | set(_OPTION_CASES) | roots


def test_listing_authority_admits_the_universe_and_rebinds_offline(
    tmp_path, monkeypatch, installer_universe
):
    """Identity is the registry's; the listing authority admits every universe
    symbol so a holding the registry cannot identify is a typed gap, and an
    install made under the acquired-issuers-only authority widens offline."""

    from scripts import materialize_evidence_cro_authority as setup

    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        EvidenceReviewWorkspaceManifest,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
        read_research_workspace_manifest,
    )
    from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
        AdmittedListingTickerAuthority,
    )

    binding, _registry_path = _package(tmp_path, managed=False)
    publish_research_workspace_manifest(
        tmp_path,
        ResearchWorkspaceManifest.create(
            workspace_id="rebind",
            default_strategy_package_id="fixture-package",
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(),
        ).with_bindings(evidence_review=binding),
    )
    installed = EvidenceReviewWorkspaceManifest.model_validate_json(
        (tmp_path / binding.relative_path).read_bytes()
    )
    before = {p: p.read_bytes() for p in (tmp_path / "authority").iterdir() if p.is_file()}
    installer_universe.update({"AAPL": "US-AAPL", "ZZZZ": "US-ZZZZ", "brk.b": "US-BRKB"})
    monkeypatch.setattr(
        setup,
        "probe_hybrid_retrieval_capabilities",
        lambda *_: SimpleNamespace(status="READY", logical_hash=CAPABILITY_HASH),
    )
    args = setup._parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--semantic-model",
            str(tmp_path / "authority/semantic-model"),
            "--rebind-installed",
        ]
    )
    args.source_set_hash = "a" * 64
    with pytest.raises(ValueError, match="rebind_takes_no_source"):
        setup.materialize(args)
    args.source_set_hash = None
    # The installed authority asks for the retired production plan (written
    # absent): a rebind that names no selection would re-seal it, and is
    # refused by name before anything is written; naming the integrated
    # selection is the migration.
    assert installed.matter_selection is None
    with pytest.raises(ValueError, match="installed_matter_selection_retired"):
        setup.materialize(args)
    assert {p: p.read_bytes() for p in before} == before
    args.matter_selection = "INTEGRATED_TOPIC_ROUTING"
    result = setup.materialize(args)
    assert result["acquisition"] == {"mode": "REBIND_INSTALLED", "network_calls": 0}
    assert result["rebound_from_authority_hash"] == installed.authority_hash
    assert result["listing_entries_before"] == 1 and result["listing_entries"] == 3
    assert result["entity_ids"] == ("AAPL",) and result["installed"] is False
    manifest = EvidenceReviewWorkspaceManifest.model_validate_json(
        (tmp_path / result["binding"]["relative_path"]).read_bytes()
    )
    # Registry, documents, pack, profile, authority id and floor are the
    # installed ones; only the listing authority moved.
    assert manifest.issuer_registry.content_hash == installed.issuer_registry.content_hash
    assert manifest.recorded_documents.content_hash == installed.recorded_documents.content_hash
    assert manifest.model_profile is None and manifest.authority_id == installed.authority_id
    assert manifest.minimum_entity_coverage == installed.minimum_entity_coverage == 0.6
    # The matter selection moved as named.
    assert manifest.matter_selection is not None
    assert manifest.matter_selection.method == "INTEGRATED_TOPIC_ROUTING"
    listing = AdmittedListingTickerAuthority.model_validate_json(
        (tmp_path / manifest.listing_authority.relative_path).read_bytes()
    )
    assert listing.ticker_by_listing == {"US-AAPL": "AAPL", "US-BRKB": "BRK.B", "US-ZZZZ": "ZZZZ"}
    assert manifest.listing_authority.content_hash != installed.listing_authority.content_hash
    assert {p: p.read_bytes() for p in before} == before
    assert read_research_workspace_manifest(tmp_path).evidence_review == binding
    # Re-declared floor and id make a distinct package; installing binds it
    # (still naming the selection: the installed one is the retired plan).
    args.install = True
    args.minimum_entity_coverage = 0.0
    args.authority_id = "rebound-qa"
    rebound = setup.materialize(args)
    assert rebound["package_id"] != result["package_id"] and rebound["installed"] is True
    bound = read_research_workspace_manifest(tmp_path).evidence_review
    assert bound is not None and bound.relative_path == rebound["binding"]["relative_path"]
    again = EvidenceReviewWorkspaceManifest.model_validate_json(
        (tmp_path / bound.relative_path).read_bytes()
    )
    assert again.minimum_entity_coverage == 0.0 and again.authority_id == "rebound-qa"
    assert again.listing_authority.content_hash == manifest.listing_authority.content_hash
    # Installed, the integrated selection is kept by a rebind that names none.
    assert again.matter_selection == manifest.matter_selection
    args.matter_selection = None
    kept = setup.materialize(args)
    assert kept["matter_selection"] == result["matter_selection"]
    assert kept["package_id"] == rebound["package_id"]
    # A universe that no longer names an acquired issuer's symbol is refused.
    installer_universe.clear()
    installer_universe.update({"ZZZZ": "US-ZZZZ"})
    with pytest.raises(RuntimeError, match="universe_missing_admitted_ticker:AAPL"):
        setup.materialize(args)


def test_research_only_evidence_binding_preserves_inputs_without_installing_a_strategy(tmp_path):
    binding, _registry = _package(tmp_path)
    original = ResearchWorkspaceManifest.research_only("authored-review").with_bindings(
        experiment_inputs=(
            ResearchWorkspaceExperimentInput(input_id="research", binding_hash="a" * 64),
        )
    )
    installed = original.with_bindings(evidence_review=binding)
    assert installed.strategy_installation == "NOT_INSTALLED"
    assert installed.default_strategy_package_id is None and installed.strategy_artifacts == ()
    assert installed.evidence_review == binding
    assert installed.model_dump(
        exclude={"manifest_hash", "evidence_review"}
    ) == original.model_dump(exclude={"manifest_hash", "evidence_review"})
    assert installed.manifest_hash != original.manifest_hash
    assert ResearchWorkspaceManifest.model_validate_json(installed.model_dump_json()) == installed


def test_cli_selects_only_the_declared_local_retrieval_environment(tmp_path, monkeypatch, capsys):
    from alphalattice.control.product_host.composition import entry as run_alphalattice
    from alphalattice.control.product_host.composition import (
        web_launcher as run_local_portfolio_web,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        publish_research_workspace_manifest,
    )
    from alphalattice.interface.local_application import (
        retrieval_environment as create_retrieval_environment,
    )

    binding, _registry = _package(tmp_path)
    publish_research_workspace_manifest(
        tmp_path,
        ResearchWorkspaceManifest.research_only("bootstrap").with_bindings(evidence_review=binding),
    )
    interpreter = tmp_path / "declared-python.exe"
    interpreter.write_bytes(b"selector-test-only")
    monkeypatch.setattr(create_retrieval_environment, "interpreter_path", lambda: interpreter)
    # `serve` imports these itself (a CLI call never pays for them), so the modules are patched.
    import importlib.util
    import os
    import subprocess
    import sys

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    spawned = []
    foreign_source = tmp_path / "other-source"
    foreign_package = foreign_source / "alphalattice"
    foreign_package.mkdir(parents=True)
    (foreign_package / "__init__.py").write_text(
        "raise RuntimeError('must not import the caller source')\n", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(foreign_source))
    monkeypatch.setattr(
        subprocess, "call", lambda args, **kwargs: spawned.append((args, kwargs)) or 0
    )
    args = ["--workspace", str(tmp_path), "--no-browser", "--stop-on-stdin"]
    assert run_alphalattice.serve(args) == 0
    command, options = spawned[0]
    assert command[0] == str(interpreter)
    assert command[1:3] == ["-m", "alphalattice.control.product_host.composition.web_launcher"]
    assert command[3:] == args
    assert options["env"]["PYTHONPATH"].split(os.pathsep) == [
        str(run_alphalattice.ROOT / "src"),
        str(foreign_source),
    ]
    assert os.environ["PYTHONPATH"] == str(foreign_source)
    assert set(options) == {"env"}  # cwd and stdin retain their caller meaning.
    # Run the actual module from outside the checkout with that child's source
    # binding: dependency-only environments and hostile sibling paths both work.
    opened = subprocess.run(
        [sys.executable, *command[1:3], "--help"],
        cwd=tmp_path,
        env=options["env"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert opened.returncode == 0, opened.stderr
    assert "--stop-on-stdin" in opened.stdout
    interpreter.unlink()
    assert run_alphalattice.serve(args) == 2
    import json

    missing = json.loads(capsys.readouterr().out)
    assert missing["next_action"] == "CREATE_DECLARED_RETRIEVAL_ENVIRONMENT"
    assert missing["setup_command"] == [*missing["network_setup_command"], "--offline"]
    assert "dependency-download permission" in missing["explanation"]
    assert len(spawned) == 1
    # An uninitialized workspace must still reach the ordinary initializer.
    started = []
    monkeypatch.setattr(run_local_portfolio_web, "main", lambda args: started.append(args) or 0)
    empty_args = ["--workspace", str(tmp_path / "empty"), "--no-browser"]
    assert run_alphalattice.serve(empty_args) == 0 and started == [empty_args]


def test_workspace_authority_installs_the_existing_runtime_and_no_model_actor(
    tmp_path: Path,
) -> None:
    """requirement (AG2): the Analyst's and the CRO's answers come through the seam, so the
    admission installs the runtime and the recorded documents and admits no model."""

    binding, _registry_path = _package(tmp_path)
    admitted = admit_evidence_review_workspace(
        workspace=tmp_path,
        binding=binding,
        semantic_capability_reader=lambda _runtime: CAPABILITY_HASH,
    )
    try:
        assert admitted.registry.entries[0].ticker == "AAPL"
        assert admitted.listing_authority.ticker_by_listing == {"US-AAPL": "AAPL"}
        assert admitted.resources.recorded_documents[0].revision == "0000320193-26-000079"
        assert admitted.resources.admitted_authority_hash == admitted.authority_hash
        assert admitted.evidence_policy.admit_model_review is False
        assert admitted.evidence_policy.network_consent is False
        assert admitted.review_actor is None and admitted.resources.analysis_actor is None
        assert admitted.model_authority_admitted is False
    finally:
        admitted.runtime.close()


def test_child_tampering_refuses_before_capability_or_model_work(tmp_path: Path) -> None:
    binding, registry_path = _package(tmp_path)
    registry_path.write_bytes(registry_path.read_bytes() + b" ")
    calls: list[str] = []
    with pytest.raises(ResearchWorkspaceError, match="child_file_identity_invalid"):
        admit_evidence_review_workspace(
            workspace=tmp_path,
            binding=binding,
            semantic_capability_reader=lambda _runtime: calls.append("semantic") or CAPABILITY_HASH,
        )
    assert calls == []


def test_a_semantic_capability_mismatch_refuses_installation(tmp_path: Path) -> None:
    binding, _registry_path = _package(tmp_path)
    with pytest.raises(ResearchWorkspaceError, match="semantic_capability_invalid"):
        admit_evidence_review_workspace(
            workspace=tmp_path,
            binding=binding,
            semantic_capability_reader=lambda _runtime: "d" * 64,
        )


@pytest.mark.parametrize("managed", [True, False])
def test_a_valid_workspace_admits_with_its_capability_verified(
    tmp_path: Path, managed: bool
) -> None:
    """requirement: the admission verifies the semantic capability and leaves the stored
    manifest as it was; no Provider credential is read (AG2)."""

    binding, _registry_path = _package(tmp_path, managed=managed)
    stored_manifest = tmp_path / binding.relative_path
    stored_bytes = stored_manifest.read_bytes()
    seen: list[str] = []

    def reader(_runtime: Any) -> str:
        seen.append("semantic")
        return CAPABILITY_HASH

    admitted = admit_evidence_review_workspace(
        workspace=tmp_path, binding=binding, semantic_capability_reader=reader
    )
    assert seen == ["semantic"]
    assert admitted.model_authority_admitted is False
    assert admitted.review_actor is None
    assert admitted.resources.analysis_actor is None
    assert admitted.evidence_policy.admit_model_review is False
    assert stored_manifest.read_bytes() == stored_bytes
    admitted.runtime.close()


def test_research_workspace_identity_binds_the_optional_evidence_authority(
    tmp_path: Path,
) -> None:
    binding, _registry_path = _package(tmp_path)
    without = ResearchWorkspaceManifest.create(
        workspace_id="qa-workspace",
        default_strategy_package_id="qa-strategy",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
    )
    with_authority = ResearchWorkspaceManifest.create(
        workspace_id="qa-workspace",
        default_strategy_package_id="qa-strategy",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
        evidence_review=binding,
    )
    assert without.manifest_hash != with_authority.manifest_hash
    assert with_authority.evidence_review == binding
