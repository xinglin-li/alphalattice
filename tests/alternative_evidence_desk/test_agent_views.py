"""What an agent reads: the Host's Markdown bundles for the Analyst and the CRO.

The packing rule (at most 32 KB or 1,500 lines a file, 1,000 characters a
line, so each file is one whole read on every agent host; a subject never split
unless it alone exceeds the bound; one index naming every file with what it
covers and its counts, and how to read them), the answer's account of what was
read, near-identical excerpts shown by the words that differ with every alias
still citable, no hash in any file, the holdings stated once, and the built-in
agents reading the same views. Controlled material only; no model, no network.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from alphalattice.evidence.alternative_evidence.analysis.packet import (
    AlternativeEvidencePacket,
    span_aliases,
)
from alphalattice.evidence.alternative_evidence.analysis.views import (
    MAXIMUM_DIFFERENCE_PLACES,
    excerpt_groups,
    render_analyst_bundle,
)
from alphalattice.oversight.chief_risk_officer.decision.views import (
    MAXIMUM_COVERAGE_LINES,
    MAXIMUM_UNREPORTED_NAMES,
    render_review_bundle,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import ExposureBand
from alphalattice.protocols.actor_execution.bundles import (
    BUNDLE_FILE_BYTES,
    BUNDLE_FILE_LINES,
    BUNDLE_LINE_CHARACTERS,
    READING_RULE,
    BundleBlock,
    BundleSection,
    answer_read,
    compose_bundle,
    pack_sections,
    wrap_text,
)
from tests.alternative_evidence_desk.document_intelligence_support import (
    _obligation,
    _open_recorded,
)
from tests.alternative_evidence_desk.planted_corpus import _recorded_document
from tests.alternative_evidence_desk.review_dossiers import _NOW, _dossier, _finding, _issuer

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
CODEX_TOOL_OUTPUT_BYTES = 40_000
"""What Codex hands the model of one tool call's output: 10,000 tokens at four bytes a token,
the middle of the rest cut (measured 2026-09-27)."""


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


@pytest.mark.parametrize(
    "kind,bound,location_words",
    [
        (
            "unavailable_reasons",
            MAXIMUM_COVERAGE_LINES,
            "complete list, with each unit's issuers and reason",
        ),
        ("missing_evidence", MAXIMUM_COVERAGE_LINES, "complete gap list"),
        ("unreported", MAXIMUM_UNREPORTED_NAMES, "complete issuer list"),
    ],
)
@pytest.mark.parametrize("extra", [0, 1, 400], ids=["at-bound", "one-over", "multiple-files"])
def test_every_cro_coverage_preview_retains_its_complete_list(
    kind: str, bound: int, location_words: str, extra: int
) -> None:
    """Each bounded list retains every entry and its read location."""
    count = bound + extra
    if kind == "unreported":
        values = tuple(f"QAUN{index:04d}" for index in range(count))
        dossier = _dossier(
            issuers=tuple(
                _issuer(value, weight_rank=index + 1) for index, value in enumerate(values)
            ),
            findings=(),
        )
    else:
        values = tuple(
            f"unit {index} (QAUN{index:04d}) is not reviewed: " + "Missing source evidence. " * 5
            for index in range(count)
        )
        dossier = _dossier(**{kind: values})
    bundle = render_review_bundle(dossier, task_procedure="")
    _within_bounds(bundle.files)
    index = bundle.text("README.md")
    overflow = [(name, text) for name, text in bundle.files if name.startswith("coverage-")]
    if extra:
        assert overflow
        assert location_words in index
        assert f"{extra} more" in index
        assert all(f"`{name}`" in index for name, _ in overflow)
        complete = " ".join(" ".join(text for _, text in overflow).split())
        assert all(" ".join(value.split()) in complete for value in values)
        if extra == 400 and kind != "unreported":
            assert len(overflow) > 1
    else:
        assert not overflow
        if kind == "unreported":
            assert f"- No finding was reported for {count} issuer(s): {', '.join(values)}." in index
        elif kind == "missing_evidence":
            assert (
                f"- The analysis records {count} gap(s) in what could be read, among them:" in index
            )
            assert "\n".join(f"  - {value}" for value in values) in index
        else:
            assert "\n".join(f"- {value}" for value in values) in index
        assert location_words not in index


@pytest.mark.parametrize("extra", [0, 1], ids=["at-bound", "one-over"])
def test_an_analyst_preview_names_omitted_differences_and_retains_the_full_excerpt(
    tmp_path: Path, extra: int
) -> None:
    """BEHAVIOUR (TE12): overflowing differences have a full read; filing titles are uncut."""
    packet = _packet(tmp_path)
    count = MAXIMUM_DIFFERENCE_PLACES + extra
    representative = " ".join(
        " ".join(f"clause{index}word{word}" for word in range(40)) + f" OLD{index}"
        for index in range(count)
    )
    member = representative
    for index in range(count):
        member = member.replace(f"OLD{index}", f"NEW{index}")
    base = packet.spans[0]
    spans = (
        base.model_copy(update={"span_handle": "SPAN-DIFF-1", "excerpt": representative}),
        base.model_copy(
            update={
                "span_handle": "SPAN-DIFF-2",
                "excerpt": member,
                "available_at": base.available_at - timedelta(days=1),
            }
        ),
    )
    title = "Recorded source " + "long title " * 23 + "TITLE-END"
    document_set = packet.document_set.model_copy(
        update={
            "documents": tuple(
                document.model_copy(update={"title": title})
                for document in packet.document_set.documents
            )
        }
    )
    packet = replace(packet, spans=spans, document_set=document_set)
    groups = excerpt_groups(spans, packet.request.ordered_entity_ids)
    assert len(groups) == 1 and groups[0].members == (spans[1],)
    bundle = render_analyst_bundle(packet, task_procedure="")
    _within_bounds(bundle.files)
    material = "\n".join(text for name, text in bundle.files if name != "README.md")
    assert title in " ".join(material.split())
    alias = next(
        alias
        for alias, span in span_aliases(spans, packet.request.ordered_entity_ids).items()
        if span == spans[1]
    )
    if extra:
        assert (
            f"{extra} differing place(s) are not shown above; "
            f"read {alias} in full immediately below." in " ".join(material.split())
        )
        assert f"#### {alias} (full excerpt)" in material
        full = material.partition(f"#### {alias} (full excerpt)")[2]
        assert member in " ".join(full.split())
    else:
        assert "differing place(s) are not shown above" not in material
        assert "(full excerpt)" not in material


def _within_bounds(files: tuple[tuple[str, str], ...]) -> None:
    for name, text in files:
        assert len(text.encode("utf-8")) <= BUNDLE_FILE_BYTES, name
        # One whole read on Codex, with room for the reading command's own lines.
        assert len(text.encode("utf-8")) + 2048 <= CODEX_TOOL_OUTPUT_BYTES, name
        assert text.count("\n") <= BUNDLE_FILE_LINES, name
        assert max(len(line) for line in text.splitlines()) <= BUNDLE_LINE_CHARACTERS, name
        assert not HEX64.search(text), f"a hash reached {name}"


def test_files_are_packed_by_size_and_a_subject_splits_only_when_it_alone_exceeds() -> None:
    """requirement (S2 packing rule): subjects fill a file in order; the next
    file opens when one does not fit; only a subject larger than the bound is
    split, between its blocks; the index names every file, what it covers and
    its counts."""

    line = "x" * 900
    per_file = BUNDLE_FILE_BYTES // (len(line) + 1)  # the 900-character lines a file holds

    def section(key: str, blocks: int, lines_each: int) -> BundleSection:
        return BundleSection(
            key=key,
            heading=f"## {key}",
            blocks=tuple(
                BundleBlock(tuple(line for _ in range(lines_each)), {"filings": 1})
                for _ in range(blocks)
            ),
        )

    small = [section(f"S{index}", 1, per_file // 6) for index in range(3)]
    large = section("BIG", 6, per_file // 2 + 1)  # six filings over half a file: one a file
    after = section("AFTER", 1, per_file // 6)
    files = pack_sections([*small, large, after], stem="material")
    bundle = compose_bundle(["# Index"], files)

    assert [value.covers for value in files] == [
        ("S0", "S1", "S2"),
        ("BIG part 1 of 6",),
        ("BIG part 2 of 6",),
        ("BIG part 3 of 6",),
        ("BIG part 4 of 6",),
        ("BIG part 5 of 6",),
        ("BIG part 6 of 6", "AFTER"),
    ]
    assert files[0].counts == {"filings": 3}
    _within_bounds(bundle.files)
    index = bundle.text("README.md")
    assert bundle.files[0][0] == "README.md"
    for value in files:
        assert f"`{value.name}`" in index
    assert "`material-01.md` -- S0; S1; S2 (3 filings)" in index
    assert index.rstrip("\n").endswith(READING_RULE)
    # A long paragraph wraps at a sentence, never past the line bound.
    wrapped = wrap_text(" ".join(f"Sentence {n} states a fact." for n in range(400)))
    assert max(len(value) for value in wrapped) <= BUNDLE_LINE_CHARACTERS
    assert all(value.endswith(".") for value in wrapped)


def test_an_answer_names_the_files_it_read_as_its_own_word() -> None:
    """requirement (OP11): an external answer names under `read` the files of its
    bundle it read whole; the names, each once in the order given, are kept as provenance and
    never reach the answer's own fields, and no list names none. Nothing is refused by them."""

    files = ("README.md", "material-01.md", "material-02.md")
    findings = {"findings": [], "notes": "Nothing material."}
    own, read = answer_read({**findings, "read": [*files, files[0]]})
    assert own == findings and read == files
    assert answer_read({**findings, "read": ["README.md"]})[1] == ("README.md",)
    assert answer_read(findings) == (findings, ())
    assert answer_read({**findings, "read": "README.md"})[1] == ()


def _packet(tmp_path: Path) -> AlternativeEvidencePacket:
    runtime, request, _registry, snapshot, document_set, generation = _open_recorded(
        tmp_path,
        entities=("AAPL", "MSFT"),
        documents=(_recorded_document("AAPL"), _recorded_document("MSFT")),
    )
    receipt, spans = runtime.select_evidence(
        request=request, document_set=document_set, generation=generation
    )
    runtime.close()
    # Two near-identical passages of AAPL's filing, as templated disclosures
    # repeat: the same sentence with another party, date and case number.
    base = next(value for value in spans if value.entity_id == "AAPL")
    template = (
        "On March {day}, 2025, Plaintiff Number {n} filed a complaint in the District of "
        "Delaware (Case No. 1:25-cv-0000{n}) alleging that the Company's products infringed "
        "the plaintiff's patents. The Company filed an answer denying the allegations, and "
        "discovery has commenced in the ordinary course of the proceeding. The Company "
        "believes it has meritorious defenses and intends to defend the action vigorously; "
        "an estimate of the possible loss or range of loss cannot be made at this time, and "
        "the Company has not recorded an accrual with respect to this matter."
    )
    copies = tuple(
        base.model_copy(
            update={
                "span_handle": f"SPAN-S9{n}-R01",
                "excerpt": template.format(day=n + 1, n=n),
                "available_at": base.available_at - timedelta(days=n),
            }
        )
        for n in (1, 2, 3)
    )
    return AlternativeEvidencePacket(
        request=request,
        obligation=_obligation(request),
        snapshot=snapshot,
        document_set=document_set,
        receipt=receipt,
        spans=(*spans, *copies),
    )


def test_the_analyst_bundle_is_clean_packed_and_merges_near_identical_excerpts(
    tmp_path: Path,
) -> None:
    """requirement (S2): every excerpt under its alias, near-identical ones
    shown by the words that differ and still citable, the index first with
    the answer's schema and example, no hash anywhere."""

    packet = _packet(tmp_path)
    bundle = render_analyst_bundle(packet, task_procedure="Extract cited facts.")
    _within_bounds(bundle.files)
    index = bundle.text("README.md")
    assert '"cite": [' in index and "a subset is an answer" in index
    assert "## Coverage (written by the program)" in index
    material = "\n".join(text for name, text in bundle.files if name != "README.md")
    aliases = span_aliases(packet.spans, packet.request.ordered_entity_ids)
    for alias in aliases:
        headed = material.count(f"#### {alias}\n")
        listed = len(re.findall(rf"^- {alias} · ", material, flags=re.MULTILINE))
        assert headed + listed == 1, (alias, headed, listed)
    groups = excerpt_groups(packet.spans, packet.request.ordered_entity_ids)
    (templated,) = [group for group in groups if len(group.members) == 2]
    assert templated.representative.excerpt.startswith("On March 2, 2025, Plaintiff Number 1")
    member_lines = [value for value in material.splitlines() if "differs:" in value]
    assert any(
        "Plaintiff Number [2]" in value and "[1:25-cv-00002)]" in value for value in member_lines
    )
    assert "#### " + next(a for a, s in aliases.items() if s == templated.representative) in (
        material
    )
    merged = sum(len(group.members) for group in groups)
    assert f"{merged} near-identical excerpt(s) shown by their differing words" in index


def test_the_review_bundle_states_the_holdings_once_and_packs_the_findings() -> None:
    """requirement (S2): the dossier's semantics as Markdown -- holdings in
    their own file once, findings under their aliases by issuer, a program
    coverage note -- and no hash; measured on a 490-issuer book."""

    dossier = _dossier()
    bundle = render_review_bundle(dossier, task_procedure="Review.")
    names = [name for name, _ in bundle.files]
    assert names == ["README.md", "holdings-01.md", "findings-01.md"]
    _within_bounds(bundle.files)
    assert bundle.text("holdings-01.md").count("| AAPL |") == 1
    assert "### F1 · OPERATIONS_SUPPLY · ADVERSE" in bundle.text("findings-01.md")
    assert '"findings": [' in bundle.text("README.md")
    # Written by the program (S6): when each finding's documents were filed,
    # and the issuers in scope no finding names.
    filed = (_NOW - timedelta(days=1)).date().isoformat()
    assert f"Filed: {filed}." in bundle.text("findings-01.md")
    assert "No finding was reported" not in bundle.text("README.md")
    quiet = _dossier(issuers=(_issuer(), _issuer("ZZZ", weight_rank=5)))
    assert "- No finding was reported for 1 issuer(s): ZZZ." in (
        render_review_bundle(quiet, task_procedure="").text("README.md")
    )

    issuers = tuple(
        _issuer(f"Q{index:03d}", band=ExposureBand.LOW, weight_rank=index)
        for index in range(1, 491)
    )
    findings = tuple(
        _finding(handle=f"FIND-Q{index:03d}-1", entity=f"Q{index:03d}") for index in range(1, 491)
    )
    wide = _dossier(issuers=issuers, findings=findings)
    measured = render_review_bundle(wide, task_procedure="").measure()
    names = [row["name"] for row in measured["per_file"]]
    assert names[0] == "README.md" and names[1] == "holdings-01.md"
    assert all(name.startswith("findings-") for name in names[2:])
    material = sum(int(row["bytes"]) for row in measured["per_file"][1:])
    # Packed, never a file an issuer: every material file but each kind's last is full.
    assert measured["files"] - 1 <= 2 + material // (BUNDLE_FILE_BYTES - 2048), measured["files"]
    _within_bounds(render_review_bundle(wide, task_procedure="").files)
