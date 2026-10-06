"""Replay verifies the whole evidence graph, and output never touches the source.

These cover the findings that were about what replay and the workflow *failed to
check*, so each case is written as a rejection: delete a child artifact, tamper
with one, pair a valid surface with another run's valid diagnostics, or aim the
output at the source, and the operation must refuse.

The cross-lineage cases are the ones content addressing cannot cover by itself.
Every artifact here hashes to its own name, so each verifies in isolation no
matter which run produced it; what makes a set wrong is that its members do not
belong to each other.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.investment.risk_research.experiments.verification import (
    REQUIRED_CATEGORIES,
    RiskEvidenceVerifier,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    SealedResearchProgram,
)
from tests.researcher_methodology_surface.conftest import RiskDevelopmentRun

_SURFACES = "development/covariance-surfaces"
_DIAGNOSTICS = "development/covariance-diagnostics"
_INPUT_BINDINGS = "development/input-bindings"


def _workflow(*, source: Path, root: Path) -> ResearchProgramWorkflow:
    """A workflow with no compilers or executors -- only path policy is exercised."""

    return ResearchProgramWorkflow(
        dispatcher=None,  # type: ignore[arg-type]
        executors=(),
        workspace_root=root,
        source_workspace=source,
    )


@pytest.mark.parametrize(
    "declared",
    [
        "../escaped",
        "../../escaped",
        "nested/../../escaped",
    ],
)
def test_output_workspace_cannot_escape_the_authorized_root(tmp_path: Path, declared: str) -> None:
    """Rejecting absolute paths was never enough -- `..` is relative."""

    root = tmp_path / "root"
    root.mkdir()
    workflow = _workflow(source=tmp_path / "source", root=root)
    with pytest.raises(AuthoringError, match="output_workspace_escapes_root"):
        workflow._output_workspace(declared)


@pytest.mark.parametrize("alias", ["ws", "WS"], ids=["same_path", "case_alias"])
def test_output_workspace_cannot_be_the_source_workspace(tmp_path: Path, alias: str) -> None:
    """Under either spelling: NTFS is case-insensitive, so `WS` and `ws` are one directory."""

    root = tmp_path / "root"
    (root / "ws").mkdir(parents=True)
    workflow = _workflow(source=root / "ws", root=root)
    with pytest.raises(AuthoringError, match="output_workspace_is_source"):
        workflow._output_workspace(alias)


def test_output_workspace_cannot_nest_inside_the_source(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "ws" / "inner").mkdir(parents=True)
    workflow = _workflow(source=root / "ws", root=root)
    with pytest.raises(AuthoringError, match="output_workspace_overlaps_source"):
        workflow._output_workspace("ws/inner")


def test_source_nested_under_output_is_also_refused(tmp_path: Path) -> None:
    """The containment check runs both ways: writing above the source is worse."""

    root = tmp_path / "root"
    (root / "out" / "ws").mkdir(parents=True)
    workflow = _workflow(source=root / "out" / "ws", root=root)
    with pytest.raises(AuthoringError, match="output_workspace_overlaps_source"):
        workflow._output_workspace("out")


def test_a_sibling_sharing_a_name_prefix_is_allowed(tmp_path: Path) -> None:
    """Containment is component-wise: `ws-backup` is not inside `ws`."""

    root = tmp_path / "root"
    (root / "ws").mkdir(parents=True)
    workflow = _workflow(source=root / "ws", root=root)
    assert workflow._output_workspace("ws-backup").name == "ws-backup"


@pytest.fixture
def tampered_run(risk_development_run: RiskDevelopmentRun, tmp_path: Path) -> RiskDevelopmentRun:
    """A private copy of the shared run, because these cases destroy it.

    The build is session-scoped so it is paid once, but deleting a chunk in one
    case would otherwise decide the outcome of the next. Copying keeps each
    rejection independent of test order.
    """

    target = tmp_path / "run"
    shutil.copytree(risk_development_run.output, target)
    return replace(risk_development_run, output=target)


def _store_root(output_workspace: Path) -> Path:
    """Ask the store where it writes rather than restating its layout here."""

    return RiskArtifactStore(output_workspace).root


def _verify(run: RiskDevelopmentRun) -> None:
    RiskEvidenceVerifier().verify(
        program=run.program,
        evidence=run.evidence,
        authority=run.authority,
        output_workspace=run.output,
    )


def _uri(run: RiskDevelopmentRun, category: str) -> str:
    prefix = f"playpen://risk-research/{category}/"
    return next(value for value in run.evidence.artifact_uris if value.startswith(prefix))


def _artifact_path(run: RiskDevelopmentRun, category: str) -> Path:
    """The on-disk file backing the evidence's artifact of one category."""

    prefix = f"playpen://risk-research/{category}/"
    content_hash = _uri(run, category)[len(prefix) :]
    return _store_root(run.output) / category / f"{content_hash}.json"


def _reseal(evidence: ResearchExecutionEvidence, **overrides: object) -> ResearchExecutionEvidence:
    """Rebuild evidence around a change so its own hash stays self-consistent.

    Mutating a field in place would fail the evidence contract's identity
    validator, and the case would then be asserting that pydantic works rather
    than that the verifier checks lineage.
    """

    values = evidence.model_dump(mode="json")
    values.pop("evidence_hash")
    values.update(overrides)
    # Back to tuples: these fields are declared as tuples, and re-sealing from
    # lists makes pydantic serialize a shape the identity hash was not computed
    # over.
    for field in ("artifact_uris", "formation_sessions"):
        values[field] = tuple(values[field])
    return ResearchExecutionEvidence.create(**values)


def _reseal_program(program: SealedResearchProgram, **overrides: object) -> SealedResearchProgram:
    values = program.model_dump(mode="json")
    values.pop("program_hash")
    values.update(overrides)
    values["resolved_sessions"] = tuple(values["resolved_sessions"])
    return SealedResearchProgram.create(**values)


def test_a_complete_untouched_evidence_graph_verifies(tampered_run: RiskDevelopmentRun) -> None:
    """The control. Every rejection below has to start from a passing graph."""

    _verify(tampered_run)
    assert len(tampered_run.evidence.artifact_uris) == len(REQUIRED_CATEGORIES)


def test_replay_verification_fails_when_a_chunk_is_deleted(
    tampered_run: RiskDevelopmentRun,
) -> None:
    """The original finding, exactly: exact reuse claimed over deleted artifacts."""

    _verify(tampered_run)
    chunk = sorted((_store_root(tampered_run.output) / "covariance" / "chunks").glob("*.bin"))[0]
    chunk.unlink()
    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        _verify(tampered_run)


def test_replay_verification_fails_when_a_chunk_is_tampered(
    tampered_run: RiskDevelopmentRun,
) -> None:
    chunk = sorted((_store_root(tampered_run.output) / "covariance" / "chunks").glob("*.bin"))[0]
    original = chunk.read_bytes()
    # Same length, different bytes: a size check alone would not notice.
    chunk.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        _verify(tampered_run)


@pytest.mark.parametrize("category", [_INPUT_BINDINGS, _SURFACES, _DIAGNOSTICS])
def test_deleting_a_terminal_artifact_is_refused(
    tampered_run: RiskDevelopmentRun,
    category: str,
) -> None:
    """The input binding was absent from the evidence entirely."""

    _artifact_path(tampered_run, category).unlink()
    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        _verify(tampered_run)


@pytest.mark.parametrize(
    ("category", "field", "value"),
    [
        (_INPUT_BINDINGS, "lookback_sessions", 7),
        (_SURFACES, "capability_handle", "CASE_STUDY_TAMPERED"),
        (_DIAGNOSTICS, "recipe_hash", "f" * 64),
    ],
)
def test_tampering_with_a_terminal_artifact_is_refused(
    tampered_run: RiskDevelopmentRun,
    category: str,
    field: str,
    value: object,
) -> None:
    target = _artifact_path(tampered_run, category)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[field] = value
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        _verify(tampered_run)


@pytest.mark.parametrize(
    "field",
    [
        "numerical_environment_hash",
        "selected_numerical_binding_hash",
        "selected_adapter_id",
        "parameter_domain_hash",
    ],
)
def test_a_surface_whose_method_fields_were_edited_cannot_be_read_back(
    tampered_run: RiskDevelopmentRun,
    field: str,
) -> None:
    """The surface answers for its own method, before replay compares anything.

    These used to be free-standing strings copied onto the artifact, so a surface
    could name a ``development_binding_hash`` that no combination of its other
    fields would ever produce and nothing could tell. Embedding the Program
    binding means reading the artifact re-derives the method identity from the
    fields underneath it, so an edited field fails to parse rather than verifying
    against itself.
    """

    target = _artifact_path(tampered_run, _SURFACES)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["program_binding"][field] = (
        "f" * 64 if field.endswith("hash") else "case-study-tampered"
    )
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        _verify(tampered_run)


def test_a_surface_paired_with_another_valid_diagnostics_is_refused(
    tampered_run: RiskDevelopmentRun,
) -> None:
    """Both artifacts individually valid; the pairing is what is wrong.

    Content addressing cannot catch this on its own. A diagnostics artifact
    hashes to its own name whichever surface it describes, so a verifier that
    only read artifacts back by hash would accept the pair.

    The donor is built by re-sealing the run's real diagnostics over a different
    epoch, which keeps it a genuinely valid artifact rather than a corrupt one --
    the distinction the lineage check exists to make.
    """

    diagnostics_path = _artifact_path(tampered_run, _DIAGNOSTICS)
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    from alphalattice.investment.risk_research.contracts import RiskFormationEvaluation
    from alphalattice.investment.risk_research.experiments.development_artifacts import (
        RiskDevelopmentDiagnostics,
    )

    # A genuinely valid diagnostics artifact for a *different* input binding.
    donor = RiskDevelopmentDiagnostics.create(
        input_binding_hash="e" * 64,
        recipe_hash=payload["recipe_hash"],
        evaluations=tuple(RiskFormationEvaluation(**value) for value in payload["evaluations"]),
    )
    store = RiskArtifactStore(tampered_run.output)
    store.publish_json(
        category=_DIAGNOSTICS,
        payload=donor.model_dump(mode="json"),
        identity_field="diagnostics_hash",
    )
    stray = _reseal(
        tampered_run.evidence,
        artifact_uris=[
            f"playpen://risk-research/{_DIAGNOSTICS}/{donor.diagnostics_hash}"
            if value.startswith(f"playpen://risk-research/{_DIAGNOSTICS}/")
            else value
            for value in tampered_run.evidence.artifact_uris
        ],
    )
    with pytest.raises(AuthoringError, match="evidence_diagnostics_not_this_surface"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


def test_an_input_binding_from_another_authority_is_refused(
    tampered_run: RiskDevelopmentRun,
) -> None:
    """The binding must answer for the authority the Program was sealed against."""

    stray = _reseal(tampered_run.evidence, authority_hash="c" * 64)
    program = _reseal_program(tampered_run.program, authority_hash="c" * 64)
    with pytest.raises(AuthoringError, match="evidence_input_binding_authority_mismatch"):
        RiskEvidenceVerifier().verify(
            program=program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


def test_evidence_naming_a_different_input_binding_is_refused(
    tampered_run: RiskDevelopmentRun,
) -> None:
    stray = _reseal(tampered_run.evidence, desk_input_binding_hash="d" * 64)
    with pytest.raises(AuthoringError, match="evidence_input_binding_mismatch"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


def test_reordered_formation_sessions_are_refused(tampered_run: RiskDevelopmentRun) -> None:
    """Same shape, same set, different order -- a different computation.

    The covariance axis is positional, so a reordering that a set comparison
    would call equal is not the run the binding describes.
    """

    reversed_sessions = [
        value.isoformat() for value in reversed(tampered_run.evidence.formation_sessions)
    ]
    stray = _reseal(tampered_run.evidence, formation_sessions=reversed_sessions)
    with pytest.raises(AuthoringError, match="evidence_formation_axis_mismatch"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


def test_an_unknown_artifact_uri_is_refused(tampered_run: RiskDevelopmentRun) -> None:
    stray = _reseal(
        tampered_run.evidence,
        artifact_uris=[
            *tampered_run.evidence.artifact_uris,
            "playpen://something-else/thing/" + "a" * 64,
        ],
    )
    with pytest.raises(AuthoringError, match="evidence_artifact_uri_unknown"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


def test_a_duplicated_artifact_category_is_refused(tampered_run: RiskDevelopmentRun) -> None:
    """Two candidate answers to which surface this Program produced."""

    stray = _reseal(
        tampered_run.evidence,
        artifact_uris=[*tampered_run.evidence.artifact_uris, _uri(tampered_run, _SURFACES)],
    )
    with pytest.raises(AuthoringError, match="evidence_artifact_duplicated"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )


@pytest.mark.parametrize("category", list(REQUIRED_CATEGORIES))
def test_an_incomplete_evidence_graph_is_refused(
    tampered_run: RiskDevelopmentRun,
    category: str,
) -> None:
    """Every terminal artifact is required; none of the four is optional."""

    stray = _reseal(
        tampered_run.evidence,
        artifact_uris=[
            value
            for value in tampered_run.evidence.artifact_uris
            if not value.startswith(f"playpen://risk-research/{category}/")
        ],
    )
    with pytest.raises(AuthoringError, match="evidence_artifact_missing"):
        RiskEvidenceVerifier().verify(
            program=tampered_run.program,
            evidence=stray,
            authority=tampered_run.authority,
            output_workspace=tampered_run.output,
        )
