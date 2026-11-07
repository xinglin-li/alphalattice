"""The grounded recall rule, the casebook and the reranker binding.

The rule credits only what the source confirms was shown, multi-fact cases
declare any-or-all, a casebook that left its corpus is refused, every
artifact the reranker loader reads is bound, frozen probes stay in the
denominator, and a generation sealed under another binding is refused. (The
production plan's per-issuer batches and its candidate-depth guarantee went
with the plan: first-release integration T5.)
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.contracts import (
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.retrieval.contracts import (
    AlternativeEvidenceRetrievalGeneration,
)
from alphalattice.kernel.knowledge import _reranking as reranking
from alphalattice.kernel.knowledge._reranking import verify_reranker_pack
from alphalattice.kernel.knowledge.hybrid_contracts import (
    RERANKER_PACK,
    HybridIndexSpec,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _corpus_passes,
    _counted_runtime,
)
from tests.alternative_evidence_desk.gate_9c6e_grounded_recall import (
    RECALL_ALL,
    RECALL_ANY,
    GroundedCasebookError,
    evaluable_cases,
    evaluate_grounded_recall,
    legacy_keyword_recall,
    load_casebook,
)

_GROUNDED_FACT = "The Company recorded an impairment charge of twenty three million dollars."


_GROUNDED_SECOND = "The Board approved the sale of the northern mill on March 3, 2026."


_GROUNDED_CURRENT = (
    "Routine background. " * 40 + _GROUNDED_FACT + " Routine tail. " * 40 + _GROUNDED_SECOND
)


_GROUNDED_SUPERSEDED = "Earlier background. " * 40 + "No charge was recorded. " * 40


def _grounded_fixture(
    *,
    bundle: str = "b" * 64,
    anchor_digest: str | None = None,
    recall: str = RECALL_ANY,
) -> tuple[dict[str, object], SimpleNamespace, SimpleNamespace]:
    """A casebook, a document set and a library that agree with each other."""

    first_start = _GROUNDED_CURRENT.index(_GROUNDED_FACT)
    first_end = first_start + len(_GROUNDED_FACT)
    second_start = _GROUNDED_CURRENT.index(_GROUNDED_SECOND)
    second_end = second_start + len(_GROUNDED_SECOND)

    def digest(start: int, end: int) -> str:
        return hashlib.sha256(_GROUNDED_CURRENT[start:end].encode("utf-8")).hexdigest()

    casebook = {
        "label": "DEVELOPMENT_CALIBRATION",
        "document_bundle_hash": bundle,
        "cases": [
            {
                "probe_id": "TEST-IMPAIRMENT",
                "entity_id": "TEST",
                "legacy_terms": ["impairment"],
                "classification": "SOURCE_CONFIRMED",
                "topic": "OPERATIONS_SUPPLY",
                "fact": "an impairment charge, and the approved sale of the northern mill",
                "recall": recall,
                "fact_ranges": [
                    {
                        "sub_fact": "charge",
                        "document_type": "10-Q",
                        "revision": "rev-current",
                        "character_start": first_start,
                        "character_end": first_end,
                        "anchor_sha256": anchor_digest or digest(first_start, first_end),
                        "excerpt": _GROUNDED_FACT,
                    },
                    {
                        "sub_fact": "sale",
                        "document_type": "10-Q",
                        "revision": "rev-current",
                        "character_start": second_start,
                        "character_end": second_end,
                        "anchor_sha256": digest(second_start, second_end),
                        "excerpt": _GROUNDED_SECOND,
                    },
                ],
            }
        ],
    }
    document_set = SimpleNamespace(
        documents=(
            SimpleNamespace(
                entity_id="TEST",
                document_type="10-Q",
                revision_label="rev-current",
                semantic_handle="DOC-TEST-001",
                workspace_document_id="ae.test.current",
                workspace_revision=1,
            ),
            SimpleNamespace(
                entity_id="TEST",
                document_type="10-Q",
                revision_label="rev-superseded",
                semantic_handle="DOC-TEST-002",
                workspace_document_id="ae.test.superseded",
                workspace_revision=1,
            ),
        )
    )
    bodies = {
        "ae.test.current": _GROUNDED_CURRENT.encode("utf-8"),
        "ae.test.superseded": _GROUNDED_SUPERSEDED.encode("utf-8"),
    }
    library = SimpleNamespace(
        read_revision=lambda document_id, revision: (None, bodies[document_id])
    )
    return casebook, document_set, library


def _grounded_span(
    excerpt: str,
    *,
    document_handle: str = "DOC-TEST-001",
    entity_id: str = "TEST",
    document_type: str = "10-Q",
    revision_label: str = "rev-current",
    character_start: int | None = None,
    character_end: int | None = None,
) -> SimpleNamespace:
    """A resolved span as the session emits it, defaulting to a range that holds its excerpt."""

    if character_start is None:
        # Default to the range that genuinely holds this excerpt in the
        # current revision; tests that want a lying range set one explicitly.
        located = _GROUNDED_CURRENT.find(excerpt)
        if located >= 0:
            character_start, character_end = located, located + len(excerpt)
        else:
            character_start, character_end = 0, min(len(_GROUNDED_CURRENT), len(excerpt) + 20)
    return SimpleNamespace(
        document_handle=document_handle,
        entity_id=entity_id,
        document_type=document_type,
        revision_label=revision_label,
        character_start=character_start,
        character_end=character_end,
        excerpt=excerpt,
    )


def _score(casebook: dict[str, object], document_set: Any, library: Any, *spans: Any) -> Any:
    return evaluate_grounded_recall(
        casebook=casebook,
        spans=tuple(spans),
        document_set=document_set,
        library=library,
        document_bundle_hash=str(casebook["document_bundle_hash"]),
    )[0]


def test_the_grounded_rule_credits_only_what_the_source_confirms_was_shown() -> None:
    """The grounded rule credits only what the source confirms was shown."""

    casebook, document_set, library = _grounded_fixture()
    first = _GROUNDED_CURRENT.index(_GROUNDED_FACT)
    shown = _grounded_span(_GROUNDED_CURRENT[first - 20 : first + len(_GROUNDED_FACT) + 14])
    assert _score(casebook, document_set, library, shown).recalled

    # The keyword without the fact: exactly what the legacy rule counted.
    boilerplate = _grounded_span("Routine background. Routine background.")
    assert not _score(casebook, document_set, library, boilerplate).recalled
    keyword = _grounded_span("risks include impairment of goodwill and other assets")
    assert legacy_keyword_recall(casebook=casebook, spans=(keyword,))[0].recalled

    # Truncated before the fact is stated; contradicted by a changed figure.
    assert not _score(
        casebook, document_set, library, _grounded_span(_GROUNDED_FACT[: len(_GROUNDED_FACT) // 2])
    ).recalled
    assert not _score(
        casebook,
        document_set,
        library,
        _grounded_span(_GROUNDED_FACT.replace("twenty three", "nine hundred")),
    ).recalled

    # An invented excerpt: real handle, real range, text the source does not
    # hold at that range. This is the case the previous rule could not refuse.
    invented = _grounded_span(
        _GROUNDED_FACT,
        character_start=0,
        character_end=len(_GROUNDED_FACT) + 20,
    )
    assert not _score(casebook, document_set, library, invented).recalled

    # A crossed revision: the current document's handle with the superseded
    # revision declared, and the reverse.
    assert not _score(
        casebook,
        document_set,
        library,
        _grounded_span(_GROUNDED_FACT, revision_label="rev-superseded"),
    ).recalled
    assert not _score(
        casebook,
        document_set,
        library,
        _grounded_span(
            _GROUNDED_FACT, document_handle="DOC-TEST-002", revision_label="rev-superseded"
        ),
    ).recalled

    # The wrong issuer under a real handle, and a range in the wrong place.
    assert not _score(
        casebook, document_set, library, _grounded_span(_GROUNDED_FACT, entity_id="OTHER")
    ).recalled
    wrong_place = _grounded_span(_GROUNDED_FACT, character_start=5, character_end=60)
    assert not _score(casebook, document_set, library, wrong_place).recalled

    # A handle the set never issued.
    assert not _score(
        casebook,
        document_set,
        library,
        _grounded_span(_GROUNDED_FACT, document_handle="DOC-TEST-404"),
    ).recalled


def test_multi_fact_cases_declare_any_or_all_and_report_both() -> None:
    """requirement: any/all semantics are explicit, never silently changed."""

    charge = _grounded_span(_GROUNDED_FACT)
    sale = _grounded_span(_GROUNDED_SECOND)

    casebook, document_set, library = _grounded_fixture(recall=RECALL_ANY)
    outcome = _score(casebook, document_set, library, charge)
    assert outcome.recalled and outcome.any_sub_fact and not outcome.all_sub_facts
    assert outcome.sub_facts_matched == ("charge",) and outcome.sub_facts_total == 2

    casebook, document_set, library = _grounded_fixture(recall=RECALL_ALL)
    outcome = _score(casebook, document_set, library, charge)
    assert not outcome.recalled and outcome.any_sub_fact and not outcome.all_sub_facts
    outcome = _score(casebook, document_set, library, charge, sale)
    assert outcome.recalled and outcome.all_sub_facts
    assert outcome.sub_facts_matched == ("charge", "sale")

    casebook["cases"][0]["recall"] = "most_sub_facts"
    with pytest.raises(GroundedCasebookError, match="unknown recall rule"):
        _score(casebook, document_set, library, charge)


def test_the_grounded_rule_refuses_a_casebook_that_left_its_corpus() -> None:
    """requirement: annotations are checked against the bytes they name."""

    casebook, document_set, library = _grounded_fixture()
    span = _grounded_span(_GROUNDED_FACT)

    # Another corpus.
    with pytest.raises(GroundedCasebookError, match="another document bundle"):
        evaluate_grounded_recall(
            casebook=casebook,
            spans=(span,),
            document_set=document_set,
            library=library,
            document_bundle_hash="c" * 64,
        )

    # An anchor that no longer holds at the recorded range.
    moved, document_set, library = _grounded_fixture(anchor_digest="d" * 64)
    with pytest.raises(GroundedCasebookError, match="no longer holds its anchor"):
        _score(moved, document_set, library, span)

    # A revision that is not admitted at all.
    absent, document_set, library = _grounded_fixture()
    absent["cases"][0]["fact_ranges"][0]["revision"] = "rev-never-admitted"
    with pytest.raises(GroundedCasebookError, match="not admitted"):
        _score(absent, document_set, library, span)


@contextmanager
def monkeypatched(owner: Any, name: str, value: Any) -> Iterator[None]:
    """Substitute one attribute for the body of a block, then restore it."""

    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


def test_every_artifact_the_reranker_loader_reads_is_bound(tmp_path: Path) -> None:
    """Every artifact the reranker loader reads is bound."""

    spec = HybridIndexSpec.fixed_v2()
    with pytest.raises(KnowledgeRetrievalError, match="missing a consumed artifact"):
        verify_reranker_pack(tmp_path, spec)

    def _pack(root: Path) -> Path:
        pack = root / "reranker"
        for name, _digest in RERANKER_PACK:
            artifact = pack / name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"stand-in:{name}".encode())
        return pack

    # A pack whose bytes are not the admitted ones is refused by name.
    root = tmp_path / "altered"
    _pack(root)
    with pytest.raises(KnowledgeRetrievalError, match="artifact hash differs"):
        verify_reranker_pack(root, spec)

    # With the digests stood in, the manifest verifies -- and then each single
    # artifact is shown to be load-bearing on its own.
    pinned = {name: digest for name, digest in RERANKER_PACK}
    original = reranking._hash_file

    def hash_file(path: Path) -> str:
        relative = path.relative_to(root / "reranker").as_posix()
        return pinned.get(relative) or str(original(path))

    with (
        monkeypatched(reranking, "_hash_file", hash_file),
        monkeypatched(reranking, "_runtime_versions", lambda: ("0.8.0", "0.23.2")),
    ):
        assert verify_reranker_pack(root, spec)[0].startswith("reranker-pack:")
        for name, _digest in RERANKER_PACK:
            held = pinned[name]
            pinned[name] = "0" * 64
            with pytest.raises(KnowledgeRetrievalError, match="artifact hash differs"):
                verify_reranker_pack(root, spec)
            pinned[name] = held
        (root / "reranker" / "preprocessor_config.json").write_bytes(b"{}")
        with pytest.raises(KnowledgeRetrievalError, match="unadmitted artifact"):
            verify_reranker_pack(root, spec)
        (root / "reranker" / "preprocessor_config.json").unlink()
        with (
            monkeypatched(reranking, "_runtime_versions", lambda: ("0.7.0", "0.23.2")),
            pytest.raises(KnowledgeRetrievalError, match="fastembed version differs"),
        ):
            verify_reranker_pack(root, spec)
        with (
            monkeypatched(reranking, "_runtime_versions", lambda: ("0.8.0", "0.20.0")),
            pytest.raises(KnowledgeRetrievalError, match="tokenizers version differs"),
        ):
            verify_reranker_pack(root, spec)

    assert not any(path.stat().st_size > 4096 for path in root.rglob("*") if path.is_file()), (
        "nothing may be downloaded into the model root"
    )


def test_every_frozen_probe_stays_in_the_denominator() -> None:
    """requirement: a failing probe is never quietly dropped."""

    casebook = load_casebook()
    assert [case["probe_id"] for case in casebook["cases"]] == [
        "MMM-PFAS",
        "MOS-IMPAIRMENT",
        "CF-IMPAIRMENT",
        "MELI-REGULATORY",
        "ISRG-LITIGATION",
        "DG-LITIGATION",
    ]
    assert len(evaluable_cases(casebook)) == 6
    for case in casebook["cases"]:
        assert case["classification"] in {
            "SOURCE_CONFIRMED",
            "BOILERPLATE_ONLY",
            "ABSENT_FROM_CORPUS",
        }
        assert case["fact_ranges"], "a source-confirmed probe must carry its ranges"
        for value in case["fact_ranges"]:
            assert value["character_end"] > value["character_start"]
            assert value["revision"] and value["document_type"]


def test_a_generation_sealed_under_another_retrieval_binding_is_refused(
    tmp_path: Path,
) -> None:
    """A generation sealed under another retrieval binding is refused."""

    runtime, passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    stale = seal_contract(
        AlternativeEvidenceRetrievalGeneration,
        "generation_hash",
        **{
            **generation.model_dump(exclude={"generation_hash", "kind"}),
            "retrieval_binding_hash": "9" * 64,
        },
    )
    assert stale.generation_hash != generation.generation_hash

    before = _corpus_passes(passes, generation)
    with pytest.raises(ValueError, match="retrieval_generation_lineage_invalid"):
        runtime.retrieval.open_session(
            document_set=document_set,
            generation=stale,
            evidence_as_of=request.evidence_as_of,
        )
    assert _corpus_passes(passes, generation) == before, "refused before any retrieval work"

    # The generation this build actually sealed still opens.
    session = runtime.retrieval.open_session(
        document_set=document_set,
        generation=generation,
        evidence_as_of=request.evidence_as_of,
    )
    try:
        assert session.search(query="operating", top_k=3).hits
    finally:
        session.close()
