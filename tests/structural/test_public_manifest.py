"""Public scope rules, immutable hashes and disclosure redaction hold the snapshot contract."""

import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from alphalattice.capabilities.domain_contracts import spawn
from devtools.pipeline import definition
from release.public_manifest import (
    _HOST_ROUTES,
    INTERNAL_ID_BASELINE,
    RULES,
    _carry_refusal,
    attach_reviews,
    classify,
    content_keys,
    declare_github,
    declared_private,
    generate,
    internal_id_label,
    internal_id_pattern,
    internal_id_ratchet,
    scan_public,
    validate_internal_id_history,
    validate_path,
)
from release.public_manifest import (
    main as public_manifest_main,
)
from tests._support.process_starts import observe_starts

_REVIEW_FIXTURE = "sk-" + "X" * 40 + " x@example.invalid"
_CHANGED_REVIEW_FIXTURE = "sk-" + "Y" * 40 + " x@example.invalid"


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
@pytest.mark.parametrize(
    ("path", "kind", "rule"),
    [
        ("LICENSE", "PUBLIC", "PUBLIC_LEGAL"),
        ("NOTICE", "PUBLIC", "PUBLIC_LEGAL"),
        ("README.md", "PUBLIC", "PUBLIC_DOC"),
        ("README.zh-CN.md", "PUBLIC", "PUBLIC_DOC"),
        ("CONTRIBUTING.md", "PUBLIC", "PUBLIC_DOC"),
        ("CLA.md", "PUBLIC", "PUBLIC_DOC"),
        ("GOVERNANCE.md", "PUBLIC", "PUBLIC_DOC"),
        ("SECURITY.md", "PUBLIC", "PUBLIC_DOC"),
        (".github/CODEOWNERS", "PUBLIC", "PUBLIC_DOC"),
        (".github/ISSUE_TEMPLATE/bug_report.yml", "PUBLIC", "PUBLIC_DOC"),
        (".github/ISSUE_TEMPLATE/feature_request.yml", "PUBLIC", "PUBLIC_DOC"),
        (".github/PULL_REQUEST_TEMPLATE.md", "PUBLIC", "PUBLIC_DOC"),
        (".github/workflows/ci.yml", "PUBLIC", "PUBLIC_BUILD"),
        (".github/workflows/supply-chain.yml", "PUBLIC", "PUBLIC_BUILD"),
        (".github/gitleaks.toml", "PUBLIC", "PUBLIC_BUILD"),
        (".test_durations", "PUBLIC", "PUBLIC_BUILD"),
        (".github/workflows/internal.yml", "UNSETTLED", None),
        ("benchmark/measure.py", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("case-study/readme.md", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("probe/measure.py", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("scripts/antipattern-" + "decisions.md", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("config/alpha-arm-control.yaml", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("config/dynamic-panel-risk.yaml", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("tests/agent_eval/test_one.py", "PRIVATE", "PRIVATE_TEST"),
        ("tests/unit/evaluation/test_accepted_routing.py", "PRIVATE", "PRIVATE_TEST"),
        ("tests/integration/evaluation/test_as_of.py", "PRIVATE", "PRIVATE_TEST"),
        ("tests/integration/evaluation/test_first_use_harness.py", "PRIVATE", "PRIVATE_TEST"),
        ("tests/unit/evaluation/test_unlisted_scope_contract.py", "PUBLIC", "PUBLIC_TEST"),
        ("tests/integration/evaluation/test_unlisted_scope_contract.py", "PUBLIC", "PUBLIC_TEST"),
        ("tests/structural/test_catalog_gate.py", "PRIVATE", "PRIVATE_TEST"),
        ("src/devtools/evaluation/forms.py", "PUBLIC", "PUBLIC_VERIFICATION"),
        ("scripts/test-shape-tools/package.json", "PUBLIC", "PUBLIC_VERIFICATION"),
        ("scripts/test-shape-tools/package-lock.json", "PUBLIC", "PUBLIC_VERIFICATION"),
        ("scripts/test-shape-tools/node_modules/jscpd/run-jscpd.js", "UNSETTLED", None),
        ("tests/integration/evaluation/test_forms.py", "PUBLIC", "PUBLIC_TEST"),
        (
            "tests/integration/local_application/test_native_answer_route_class.py",
            "PRIVATE",
            "PRIVATE_TEST",
        ),
        ("AGENTS.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        ("CLAUDE.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        ("AGENTS.override.md", "PRIVATE", "PRIVATE_AGREEMENT"),
        ("src/alphalattice/assets/AGENTS.md", "PRIVATE", "PRIVATE_AGREEMENT"),
        ("src/alphalattice/assets/private.key", "PRIVATE", "PRIVATE_CREDENTIAL"),
        (".env.production", "PRIVATE", "PRIVATE_CREDENTIAL"),
        (".claude/projects/one/session.json", "PRIVATE", "PRIVATE_STATE"),
        ("workspaces/one/record.json", "PRIVATE", "PRIVATE_STATE"),
        (".alphalattice/user/memory/MEMORY.md", "PRIVATE", "PRIVATE_STATE"),
        ("implemented-plans/round.md", "PRIVATE", "PRIVATE_RECORD"),
        ("experience/one.md", "PRIVATE", "PRIVATE_RECORD"),
        ("product-design/execution-plans/plan.md", "PRIVATE", "PRIVATE_RECORD"),
        ("scripts/agent_eval/one.py", "PRIVATE", "PRIVATE_QA_TOOL"),
        ("scripts/qa_retention.py", "PRIVATE", "PRIVATE_QA_TOOL"),
        ("tests/integration/product_host/test_qa_retention.py", "PRIVATE", "PRIVATE_TEST"),
        ("config/agent-eval/one.yaml", "PRIVATE", "PRIVATE_QA_TOOL"),
        (".claude/agents/research.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        (".codex/config.toml", "PUBLIC", "PUBLIC_INTEGRATION"),
        (".agents/skills/research/SKILL.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        ("docs/public-source/README.md", "PUBLIC", "PUBLIC_DOC"),
        ("docs/images/research-performance-light.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/research-performance-dark.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/research-positions-light.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/research-positions-dark.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/research-conversation-light.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/research-conversation-dark.png", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/generate_reference.py", "PUBLIC", "PUBLIC_DOC"),
        ("scripts/run_alphalattice.py", "PUBLIC", "PUBLIC_USER_SCRIPT"),
        ("config/some-rule-baseline-private.json", "PRIVATE", "PRIVATE_ALLOWANCE"),
        ("scripts/antipattern_rules.yml", "PUBLIC", "PUBLIC_VERIFICATION"),
        ("scripts/antipattern_examples.py.txt", "PUBLIC", "PUBLIC_VERIFICATION"),
        ("scripts/unexamined.py", "UNSETTLED", None),
        ("config/unexamined.yaml", "UNSETTLED", None),
        ("src/alphalattice/unreached.py", "UNSETTLED", None),
    ],
    ids=str,
)
def test_public_manifest_prioritizes_private_rules_and_never_guesses(path, kind, rule):
    decision = classify(path, set(), set(), set())
    assert decision[:2] == (kind, rule)
    assert decision[2]


PUBLISHED, PRIVATE = (
    "tests/unit/x/test_ids.py::test_one",
    "tests/unit/agent_eval/test_x.py::test_one",
)


@pytest.mark.proves("pipeline.github_declared_on_published_tests")
@pytest.mark.parametrize(
    ("node", "docstring", "marks", "contract", "declared"),
    [
        pytest.param(PUBLISHED, None, (), None, True, id="published"),
        pytest.param(PRIVATE, None, (), None, False, id="private-by-rule"),
        pytest.param(
            PUBLISHED, "PRIVATE_TEST: an inventory", (), None, False, id="private-by-docstring"
        ),
        pytest.param(
            PUBLISHED, None, ("real_evidence",), None, False, id="published-real-evidence"
        ),
        pytest.param(PUBLISHED, None, ("paid_agent",), None, False, id="published-paid-agent"),
        pytest.param(PUBLISHED, None, ("performance",), None, False, id="published-performance"),
        pytest.param(PUBLISHED, None, (), "nightly_only", False, id="published-nightly-only"),
    ],
)
def test_github_is_declared_on_exactly_the_published_tests_github_can_run(
    node, docstring, marks, contract, declared
):
    """Each test has one runner: GitHub's public Verify (-m github), or else the nightly."""
    added = []
    item = SimpleNamespace(nodeid=node, module=SimpleNamespace(__doc__=docstring))
    item.add_marker, item.get_closest_marker = added.append, lambda name: name in marks or None
    declare_github([item], lambda _: contract)
    assert added == (["github"] if declared else [])


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_the_full_release_closure_check_names_its_private_inventory_input():
    assert classify("tests/structural/test_public_release_closure.py", set(), set(), set()) == (
        "PRIVATE",
        "PRIVATE_TEST",
        "needs the private release inventory",
    )
    # Private by what it is, wherever it lives: a private data schema, or a test declaring
    # PRIVATE_TEST:, each read as parsed; what cannot be parsed cannot show it is public, so it is
    # private too. Content only ever makes a file private.
    blobs = {
        "config/anywhere/listing.json": b'{"schema": "alphalattice.release.public-manifest"}',
        "config/anywhere/escaped.json": b'{"schema": "alphalattice.tests\\u002dunregistered.v1"}',
        "config/anywhere/admissions.json": b'{"schema": "alphalattice.source-ratchet.v1"}',
        "config/anywhere/other.json": b'{"schema": "alphalattice.public.v1"}',
        "config/anywhere/torn.json": b'{"schema": "alphalattice.public.v1"',
        "tests/structural/test_declared.py": b'"""Reads it.\n\nPRIVATE\\x5fTEST: needs it."""\n',
        "tests/structural/test_said.py": b'"""Reads it."""\nNOTE = "PRIVATE_TEST: in a string"\n',
        "tests/structural/test_torn.py": b'"""Reads it."""\ndef broken(:\n',
    }
    declared = declared_private(blobs)
    assert declared == {
        "config/anywhere/listing.json",
        "config/anywhere/escaped.json",
        "config/anywhere/admissions.json",
        "config/anywhere/torn.json",
        "tests/structural/test_declared.py",
        "tests/structural/test_torn.py",
    }
    for path in sorted(declared):
        assert classify(path, {path}, {path}, {path}, declared=declared)[:2] == (
            "PRIVATE",
            "PRIVATE_BY_CONTENT",
        )


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
@pytest.mark.parametrize(
    "path", ["", "../src/one.py", "/src/one.py", "src\\one.py", "C:/one.py", "src/one\0.py"]
)
def test_public_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="unsafe manifest path"):
        validate_path(path)


def _runtime_census(*roots: str) -> dict[str, bytes]:
    """Build the synthetic entry inventory for the release scanner fixtures."""
    census = {"base": "b" * 40, "modules": [{"path": root, "class": "LIVE"} for root in roots]}
    return {
        "census/entry_reach.json": json.dumps(census).encode(),
    }


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_public_manifest_hashes_supplied_git_bytes_and_covers_every_file():
    sha = "a" * 40
    blobs = {
        "LICENSE": b"license\n",
        "AGENTS.override.md": b"internal instructions\n",
        **_runtime_census("src/alphalattice/main.py"),
        "src/alphalattice/main.py": b"from alphalattice.new import calculate\n",
        "src/alphalattice/new.py": b"def calculate():\n    return 1\n",
        "unexamined.bin": b"unknown\x00payload",
    }
    modes = dict.fromkeys(blobs, "100644")
    manifest, audit = generate(sha, blobs, modes)
    again = generate(sha, dict(reversed(list(blobs.items()))), dict(reversed(list(modes.items()))))
    assert (manifest, audit) == again
    assert set(manifest) == {
        "schema",
        "schema_version",
        "source_sha",
        "rules",
        "public",
        "private_count",
        "unsettled",
    }
    assert manifest["schema"] == "alphalattice.release.public-manifest"
    assert manifest["source_sha"] == sha and manifest["schema_version"] == 1
    assert manifest["private_count"] == 2
    assert manifest["unsettled"] == [
        {"path": "unexamined.bin", "evidence": audit["files"][-1]["evidence"]}
    ]
    assert [r["path"] for r in manifest["public"]] == sorted(r["path"] for r in manifest["public"])
    assert len(audit["files"]) == len(blobs) == sum(audit["counts"].values())
    assert all(
        r["sha256"] == hashlib.sha256(blobs[r["path"]]).hexdigest() for r in manifest["public"]
    )
    assert (
        next(r for r in manifest["public"] if r["path"].endswith("new.py"))["rule"]
        == "PUBLIC_RUNTIME"
    )


@pytest.mark.proves("pipeline.development_labels_cannot_grow_in_public")
def test_generate_walks_development_labels_once_and_merges_its_findings(
    monkeypatch, tmp_path, capsys
):
    """Generation runs the development-label walk once and keeps its findings."""
    walk = Mock(wraps=internal_id_ratchet)
    monkeypatch.setattr("release.public_manifest.internal_id_ratchet", walk)
    blobs = {
        "census/entry_reach.json": b'{"modules": []}',
        INTERNAL_ID_BASELINE: b'{"cards": [], "lines": [], "occurrences": {}}',
        "README.md": b"V" + b"900\n",  # Positive detector fixture; avoid public self-scan.
        "scripts/probe_panel_identity.py": b"print('demo')\n",
    }
    manifest, audit = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert walk.call_count == 1
    assert audit["findings"] == audit["internal_id_ratchet"]["violations"]
    assert len(audit["findings"]) == 1 and audit["internal_id_ratchet"]["policy_present"]
    assert INTERNAL_ID_BASELINE not in {row["path"] for row in manifest["public"]}
    # A rejected dotted secret must not be persisted as a private-reference token.
    import sqlite3

    monkeypatch.setenv("ALPHALATTICE_SCAN_CACHE_DIR", str(tmp_path))
    secret = "synthetic" + ".secretvalue123456"
    blobs["README.md"] = ('password = "' + secret + '"\n').encode()
    with sqlite3.connect(tmp_path / "public-manifest.sqlite") as db:
        db.execute(
            "CREATE TABLE facts (part TEXT, key TEXT, value TEXT, seal TEXT, "
            "PRIMARY KEY (part, key))"
        )
        old_key = hashlib.sha256(b"print('demo')\n").hexdigest()  # the probe script's bytes
        old_value = json.dumps([[1, secret, 0, [], False]])
        db.execute("INSERT INTO facts VALUES (?, ?, ?, ?)", ("imports", old_key, old_value, "old"))
    _, disclosure = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert any(row["rule"] == "NO_SECRET" for row in disclosure["findings"])
    with sqlite3.connect(tmp_path / "public-manifest.sqlite") as db:
        assert all(secret not in value for (value,) in db.execute("SELECT value FROM facts"))
        assert db.execute("SELECT count(*) FROM facts WHERE part = 'imports'").fetchone()[0] == 0
    retired = capsys.readouterr().err
    assert "purged 1 predecessor facts" in retired and secret not in retired
    for path, secret, text in (
        ("README.md", "V" + "8" * 15, 'password = "{}"\n'),
        ("tests/structural/test_probe.py", "ghp_" + "Q" * 30, "import {}\n"),
    ):
        blobs[path] = text.format(secret).encode()
        _, disclosure = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
        assert any(row["rule"] == "NO_SECRET" for row in disclosure["findings"])
        with sqlite3.connect(tmp_path / "public-manifest.sqlite") as db:
            assert all(secret not in value for (value,) in db.execute("SELECT value FROM facts"))
    # The store is an optimization: clean uncached, cold and warm outputs agree.
    blobs["README.md"] = b"ordinary documentation\n"
    del blobs["tests/structural/test_probe.py"]
    monkeypatch.delenv("ALPHALATTICE_SCAN_CACHE_DIR")
    uncached = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    monkeypatch.setenv("ALPHALATTICE_SCAN_CACHE_DIR", str(tmp_path / "fresh-cache"))
    cold = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert cold == uncached == generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))


@pytest.mark.proves("pipeline.public_review_never_copies_a_disclosure")
def test_public_manifest_records_locations_without_copying_disclosures():
    secret = "sk-" + "X" * 40
    session = "12345678-1234-4123-8123-123456789abc"
    text = "\n".join(
        [
            secret,
            "D:/" + "AI Agent/private/file",  # Disclosure negative fixture; avoid self-scan.
            "session=" + session,
            "contact=person@private.invalid",
            "experience/one.md",
        ]
    )
    blobs = {
        "src/public.py": text.encode(),
        "experience/one.md": b"private content",
        "config/package-architecture.json": b"",  # a sparse changed-file scan
    }
    rows = [
        {"path": "src/public.py", "kind": "PUBLIC"},
        {"path": "experience/one.md", "kind": "PRIVATE"},
    ]
    findings, binary = scan_public(blobs, rows)
    assert not binary
    assert {r["rule"] for r in findings} == {
        "NO_SECRET",
        "NO_PRIVATE_ABSOLUTE_PATH",
        "NO_INTERNAL_ID",
        "NO_PERSONAL_CONTACT",
        "NO_PRIVATE_REFERENCE",
    }
    serialized = json.dumps(findings)
    assert secret not in serialized and session not in serialized and "person@" not in serialized
    assert all(r["path"] == "src/public.py" and r["line"] > 0 for r in findings)

    # The same scan also resolves fully qualified and logical JSON writer references.
    families = "config/registries/example.json"
    identity = "config/identity-roles.json"
    private = "src/alphalattice/kernel/probe/dev_only.py"
    public = "src/alphalattice/kernel/probe/reader.py"
    blobs = {
        private: b"",
        public: b"",
        "config/package-architecture.json": json.dumps(
            {
                "packages": [
                    {
                        "package_id": "probe",
                        "physical_namespace_target": "alphalattice.kernel.probe",
                    }
                ],
            }
        ).encode(),
    }
    modules = [{"path": private, "kind": "PRIVATE"}, {"path": public, "kind": "PUBLIC"}]
    for registry, field, value, target in (
        (families, "writers", '"probe.dev_only:Writer"', private),
        (families, "writers", r'"probe.\u0064ev_only:Writer"', private),
        (families, "writers", '"probe.reader:Reader"', None),
        (identity, "installed", '"alphalattice.kernel.probe.dev_only:Writer.open"', private),
        (identity, "installed", '"alphalattice.kernel.probe.reader:Reader"', None),
    ):
        blobs[registry] = ('{\n  "' + field + '": ' + value + "\n}\n").encode()
        findings, binary = scan_public(blobs, [{"path": registry, "kind": "PUBLIC"}, *modules])
        assert not binary
        assert [
            (row["path"], row["line"], row["rule"], row["private_targets"]) for row in findings
        ] == ([(registry, 2, "NO_PRIVATE_REFERENCE", [target])] if target else [])
    with pytest.raises(ValueError, match=r"registry_layout_missing.*package-architecture\.json"):
        scan_public(
            {**blobs, "config/package-architecture.json": b""},
            [{"path": identity, "kind": "PUBLIC"}, *modules],
        )


@pytest.mark.proves("pipeline.development_labels_cannot_grow_in_public")
def test_development_label_ratchet_pins_each_public_files_count_and_spelling():
    owner = "src/public.py"
    private = "experience/private.md"
    labels = [  # Required detector inputs; assembled to avoid public self-scan.
        "V" + "509",
        "U" + "112",
        "Round" + " 12",
        "Law" + " 42",
        "SYNTHETIC-CARD",
        "SYNTHETIC-LINE",
    ]
    policy = {"cards": [labels[4]], "lines": [labels[5]]}
    text = "\n".join(labels) + "\nfixture_V509_name u1 alpha-SYNTHETIC-CARD-beta\n"
    blobs = {owner: text.encode(), private: text.encode()}
    rows = [{"path": owner, "kind": "PUBLIC"}, {"path": private, "kind": "PRIVATE"}]
    occurrences = Counter(internal_id_label(label) for label in labels)
    policy["occurrences"] = {owner: dict(occurrences)}
    measured = internal_id_ratchet(blobs, rows, policy)
    assert measured["current"] == {owner: len(labels)}
    assert measured["allowed"] == len(labels) and not measured["violations"]

    removed = {**blobs, owner: "\n".join(labels[1:]).encode()}
    after_removal = internal_id_ratchet(removed, rows, policy)
    assert after_removal["count"] == len(labels) - 1 and not after_removal["violations"]
    lowered = {**policy, "occurrences": after_removal["current_occurrences"]}
    assert not internal_id_ratchet(removed, rows, lowered)["violations"]
    resurrected = {**removed, owner: "\n".join([labels[0], *labels[2:]]).encode()}
    returned = internal_id_ratchet(resurrected, rows, lowered)["violations"]
    assert [(row["path"], row["line"], row["column"]) for row in returned] == [(owner, 1, 1)]
    changed = {**blobs, owner: text.replace(labels[0], "V" + "510", 1).encode()}
    findings = internal_id_ratchet(changed, rows, policy)["violations"]
    assert [(row["path"], row["line"], row["column"]) for row in findings] == [(owner, 1, 1)]
    assert all(row["rule"] == "NO_DEVELOPMENT_ID" for row in findings)
    assert "510" not in json.dumps(findings)
    new = "src/new.py"
    more = {**blobs, new: ("prefix " + labels[0]).encode()}
    new_rows = [*rows, {"path": new, "kind": "PUBLIC"}]
    new_findings = internal_id_ratchet(more, new_rows, policy)["violations"]
    assert [(row["path"], row["line"], row["column"]) for row in new_findings] == [(new, 1, 8)]
    assert len(internal_id_ratchet(blobs, rows)["violations"]) == 4

    # A password which is also a numbered label keeps its frozen count without echoing its value.
    sensitive = "V" + "8" * 15
    private_text = 'password = "' + sensitive + '"\n' + sensitive + "\n"
    fixture_policy = {
        "cards": [],
        "lines": [],
        "occurrences": {"README.md": {sensitive: 2}, "README.zh-CN.md": {sensitive: 1}},
    }
    synthetic = {
        "census/entry_reach.json": b'{"modules": []}',
        INTERNAL_ID_BASELINE: json.dumps(fixture_policy).encode(),
        "README.md": private_text.encode(),
        "README.zh-CN.md": sensitive.encode(),
    }
    _, audit = generate("a" * 40, synthetic, dict.fromkeys(synthetic, "100644"))
    assert any(row["rule"] == "NO_SECRET" for row in audit["findings"])
    assert audit["internal_id_ratchet"]["count"] == audit["internal_id_ratchet"]["allowed"] == 3
    assert sensitive not in json.dumps(audit)
    fixture_policy["occurrences"]["README.md"][sensitive] = 1
    synthetic[INTERNAL_ID_BASELINE] = json.dumps(fixture_policy).encode()
    _, lowered_audit = generate("a" * 40, synthetic, dict.fromkeys(synthetic, "100644"))
    ratchet = lowered_audit["internal_id_ratchet"]
    assert (ratchet["count"], ratchet["allowed"], len(ratchet["violations"])) == (3, 2, 1)
    assert sensitive not in json.dumps(lowered_audit)

    # Product-form controls need actual detector bytes, assembled to avoid public self-scan.
    owner_path = "config/registries/owners-map.json"
    alias, other, prefix = "C" + "1", "C" + "2", "V" + "900."
    source = "src/alphalattice/control/product_host/composition/committee_materials.py"
    test = "tests/integration/product_host/test_committee.py"
    form = {
        "name": "component floor alias",
        "pattern": r"C[1-9][0-9]*",
        "prefix": "C",
        "producers": {source: {"holdings": {"scores": "keys"}}},
        "assertions": {test: {"scores": "keys"}},
    }
    noun = {"owner": "alphalattice.control.product_host.composition", "identifier_forms": [form]}
    source_text = f'def holdings():\n    return {{"scores": {{"{alias}": 1}}}}\n'
    test_text = f'assert [v["scores"] for v in rows] == [{{"{other}": 0.5}}]\n'
    product = {
        source: source_text.encode(),
        test: test_text.encode(),
        owner_path: json.dumps({"nouns": {"the committee": noun}}).encode(),
    }
    choices = [{"path": p, "kind": "PUBLIC"} for p in (source, test, owner_path)]
    catalog = {"cards": [alias, other], "lines": [], "occurrences": {}}
    assert internal_id_ratchet(product, choices, catalog)["count"] == 0
    for text in (
        f"# card {alias}\n",
        f'note = "card {alias}"\n',
        f'request = {{"scores": {{"{alias}": 1}}}}\n',
        f'def holdings():\n    return {{"note": "{alias}"}}\n',
        f'assert v["other"] == "{alias}"\n',
        f'broken "{alias}"\n',
        f'assert v["scores"] == {{}} != "{alias}"\n',
        f'assert (v["scores"], other) == ({{}}, "{alias}")\n',
        f'assert v["scores"] == ({{}}, "{alias}")\n',
        f"def test_{alias}(): pass\n",
        "def test_" + "V" + "900(): pass\n",
    ):
        changed = product | {test: text.encode()}
        assert internal_id_ratchet(changed, choices, catalog)["violations"]
    # Scope and byte-column controls: neither another path nor nonoutput producer text qualifies.
    outside = "tests/another.py"
    assert internal_id_ratchet(
        product | {outside: test_text.encode()},
        [*choices, {"path": outside, "kind": "PUBLIC"}],
        catalog,
    )["violations"]
    for text in (
        f'def holdings():\n    request = {{"scores": {{"{alias}": 1}}}}\n    return {{}}\n',
        f'def holdings():\n    return {{"scores": "card {alias}"}}\n',
        f'def another():\n    return {{"scores": {{"{alias}": 1}}}}\n',
        f'def holdings():\n    return {{"scores": load_card("{alias}")}}\n',
        f"def holdings():\n    def unused():\n"
        f'        return {{"scores": {{"{alias}": 1}}}}\n    return {{}}\n',
    ):
        assert internal_id_ratchet(product | {source: text.encode()}, choices, catalog)[
            "violations"
        ]
    unicode_text = 'assert [v["scores"] for v in 中文] == [{"' + alias + '": 1}]\n'
    assert (
        internal_id_ratchet(product | {test: unicode_text.encode()}, choices, catalog)["count"] == 0
    )
    # Registry edits cannot widen the product namespace or detach its owner.
    for owned, changes in (
        (False, {"pattern": "[CV][1-9][0-9]*"}),
        (False, {"prefix": "V", "pattern": "V[1-9][0-9]*"}),
        (True, {"owner": "alphalattice.another"}),
        (False, {"prefix": prefix, "pattern": re.escape(prefix) + r"[1-9][0-9]*"}),
    ):
        altered = json.loads(product[owner_path])
        entry = altered["nouns"]["the committee"]
        (entry if owned else entry["identifier_forms"][0]).update(changes)
        encoded = json.dumps(altered).replace("V", r"\u0056").encode()
        for text in (source_text.replace(alias, prefix + "1"), f"# {prefix}1\n"):
            with pytest.raises(ValueError):
                internal_id_ratchet(
                    product | {owner_path: encoded, source: text.encode()}, choices, catalog
                )
    # One permitted literal never exempts a development label in the same line's comment.
    changed = product | {test: (test_text.rstrip() + f"  # card {alias}\n").encode()}
    assert len(internal_id_ratchet(changed, choices, catalog)["violations"]) == 1


@pytest.mark.proves("pipeline.development_labels_cannot_grow_in_public")
def test_development_vocabulary_matches_internal_forms_without_refusing_ordinary_words():
    """Ordinary words stay prose while qualified line and card forms remain labels."""
    cards = ["UI", "LEAD", "IS", "ACCEPT", "BADGE", "JOURNEY", "LAUNCH", "LAWS", "SEAL"]
    ux, perf, fix = "U" + "X", "PERF" + "2", "FIX" + "1"
    policy = {"cards": cards, "lines": ["UI", "LEAD", ux, perf, fix]}
    text = " ".join(cards) + f" UX\nCodex {ux} Claude {perf} {fix}\ncard {cards[6]} {cards[4]}:"
    assert [
        internal_id_label(match.group()) for match in internal_id_pattern(policy).finditer(text)
    ] == [ux, perf, fix, cards[6], cards[4]]


@pytest.mark.parametrize("part", ["moves", "batches", "history"])  # its three seams
@pytest.mark.proves("pipeline.development_labels_cannot_grow_in_public")
def test_committed_development_label_allowance_cannot_grow_or_reset(
    tmp_path, synthetic_history, monkeypatch, part
):
    _git, commit = synthetic_history
    with pytest.raises(ValueError, match="no frozen internal-label policy"):
        validate_internal_id_history(tmp_path, "HEAD")
    owner, label = "src/demo.py", "V" + "901"  # Required history-control label; avoid self-scan.
    policy = {
        "cards": ["UI", "IS", "LEAD", "SYNTHETIC-CARD"],
        "lines": [],
        "occurrences": {owner: {label: 1}},
    }
    baseline = tmp_path / INTERNAL_ID_BASELINE
    baseline.parent.mkdir()
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Establish synthetic allowance")
    validate_internal_id_history(tmp_path, "HEAD")
    base = _git("rev-parse", "HEAD")
    # A mapped move carries its allowance whole; anything else is refused by name.
    moved, table = "src/moved.py", tmp_path / "config/registries/test-layout.json"
    if part == "moves":
        table.parent.mkdir(parents=True)
        extra = "SYNTHETIC-CARD"
        for index, (named, occurrences, refusal) in enumerate(
            (
                ({moved: owner}, {moved: {label: 1}}, None),
                ({moved: owner}, {moved: {label: 1, extra: 1}}, "labels differ"),
                ({moved: owner}, {moved: {label: 2}}, "labels differ"),
                ({}, {moved: {label: 1}}, "map does not name it"),
                ({moved: "src/absent.py"}, {moved: {label: 1}}, "holds no allowance"),
                ({moved: owner}, {owner: {label: 1}, moved: {label: 1}}, "keeps its allowance"),
            )
        ):
            reason = _carry_refusal(
                moved, occurrences[moved], {"occurrences": occurrences}, policy, named
            )
            if refusal is None:
                assert reason is None
            else:
                assert reason is not None and re.search(refusal, reason)
            # One passing and one refusing Git history prove the seam; the other rows
            # vary the same move decision on the in-memory policies above.
            if index > 1:
                continue
            rows = {new: {"domain": "d", "old": old, "reason": "r"} for new, old in named.items()}
            table.write_text(json.dumps({"files": rows}), encoding="utf-8", newline="\n")
            baseline.write_text(
                json.dumps({**policy, "occurrences": occurrences}), encoding="utf-8"
            )
            _git("add", "--all")
            candidate = _git("commit-tree", _git("write-tree"), "-p", base, "-m", "Move")
            if refusal is None:
                validate_internal_id_history(tmp_path, candidate)
            else:
                with pytest.raises(ValueError, match=rf"allowance increased.*{refusal}"):
                    validate_internal_id_history(tmp_path, candidate)
        # A move Git records byte for byte carries its allowance without a map row; edited, not.
        mapped = table.read_text(encoding="utf-8")
        table.write_text(json.dumps({"files": {}}), encoding="utf-8", newline="\n")
        baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
        (tmp_path / owner).parent.mkdir(exist_ok=True)
        (tmp_path / owner).write_text("text = 1\n", encoding="utf-8", newline="\n")
        _git("add", "--all")
        holding = _git("commit-tree", _git("write-tree"), "-p", base, "-m", "Hold the file")
        (tmp_path / owner).unlink()
        twin = "src/twin.py"  # the same bytes added twice: no unique move, so neither carries
        for text, twins, refusal in (
            ("text = 1\n", (), None),
            ("text = 2\n", (), "map does not name it"),
            ("text = 1\n", (twin,), "map does not name it"),
        ):
            claims = {path: {label: 1} for path in (moved, *twins)}
            baseline.write_text(json.dumps({**policy, "occurrences": claims}))
            for path in (moved, *twins):
                (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
            _git("add", "--all")
            candidate = _git("commit-tree", _git("write-tree"), "-p", holding, "-m", "Git mv")
            if refusal is None:
                validate_internal_id_history(tmp_path, candidate)
            else:
                with pytest.raises(ValueError, match=rf"allowance increased.*{refusal}"):
                    validate_internal_id_history(tmp_path, candidate)
        # An edited move the map names and a byte-for-byte copy Git pairs, each a carry alone,
        # both claim the one old allowance: it carries to one of them only.
        row = {"domain": "d", "old": owner, "reason": "r"}
        table.write_text(json.dumps({"files": {moved: row}}), encoding="utf-8", newline="\n")
        claims = {moved: {label: 1}, twin: {label: 1}}
        baseline.write_text(json.dumps({**policy, "occurrences": claims}))
        (tmp_path / moved).write_text("text = 2\n", encoding="utf-8", newline="\n")
        (tmp_path / twin).write_text("text = 1\n", encoding="utf-8", newline="\n")
        _git("add", "--all")
        candidate = _git("commit-tree", _git("write-tree"), "-p", holding, "-m", "Mixed")
        with pytest.raises(ValueError, match=r"allowance increased.*already carried to"):
            validate_internal_id_history(tmp_path, candidate)
        for path in (moved, twin):
            (tmp_path / path).unlink()
        table.write_text(mapped, encoding="utf-8", newline="\n")
        held = {**policy, "occurrences": {owner: {label: 1}, moved: {label: 1}}}
        baseline.write_text(json.dumps(held), encoding="utf-8", newline="\n")
        _git("add", "--all")
        both = _git("commit-tree", _git("write-tree"), "-p", base, "-m", "Both")
        baseline.write_text(
            json.dumps({**policy, "occurrences": {moved: {label: 2}}}),
            encoding="utf-8",
            newline="\n",
        )
        _git("add", "--all")
        grown = _git("commit-tree", _git("write-tree"), "-p", both, "-m", "Grown")
        with pytest.raises(ValueError, match=r"allowance increased.*an allowance of its own"):
            validate_internal_id_history(tmp_path, grown)
        table.write_text("not a map", encoding="utf-8", newline="\n")
        baseline.write_text(
            json.dumps({**policy, "occurrences": {moved: {label: 1}}}), encoding="utf-8"
        )
        _git("add", "--all")
        candidate = _git("commit-tree", _git("write-tree"), "-p", base, "-m", "Unreadable")
        with pytest.raises(ValueError, match="test layout map is unreadable"):
            validate_internal_id_history(tmp_path, candidate)
        return
    if part == "batches":
        # Long valid histories, including a merge, acquire immutable inputs in bounded batches.
        branch = _git("rev-parse", "HEAD")
        stream = b""
        for number in range(12):
            raw = json.dumps({**policy, "note": number}).encode()
            parent = f":{number}" if number else branch
            stream += (
                (
                    f"commit refs/heads/batched\nmark :{number + 1}\n"
                    "committer Synthetic Author <synthetic@example.test> 0 +0000\n"
                    f"data 4\nKeep\nfrom {parent}\nM 100644 inline {INTERNAL_ID_BASELINE}\n"
                    f"data {len(raw)}\n"
                ).encode()
                + raw
                + b"\n\n"
            )
        spawn.check_output(["git", "-C", str(tmp_path), "fast-import", "--quiet"], input=stream)
        tip = _git("rev-parse", "batched")
        side = _git("commit-tree", _git("write-tree"), "-p", branch, "-m", "Side branch")
        merged = _git("commit-tree", _git("write-tree"), "-p", tip, "-p", side, "-m", "Merge")
        check_output, responses = spawn.check_output, {}
        run, resolved = spawn.run, {}
        sent = bytearray()

        def recorded(
            start, command, **kwargs
        ):  # the object session's commands, kept for the replays below
            process = start(command, **kwargs)
            if "--batch-command" in command:
                stdin, sent[:] = process.stdin, b""
                write = lambda data: sent.extend(data) or stdin.write(data)
                process.stdin = SimpleNamespace(
                    write=write, flush=stdin.flush, close=stdin.close, closed=False
                )
            return process

        def record_run(command, **kwargs):
            answer = run(command, **kwargs)
            resolved[tuple(command)] = answer
            return answer

        with monkeypatch.context() as patch:
            starts = Mock(wraps=recorded)
            observe_starts(patch, starts)
            patch.setattr(spawn, "run", record_run)
            patch.setattr(
                spawn,
                "check_output",
                lambda command, **kwargs: responses.setdefault(
                    tuple(command), check_output(command, **kwargs)
                ),
            )
            validate_internal_id_history(tmp_path, merged)
        assert starts.call_count == 4, "one rev-parse, rev-list, log and object session per check"
        assert all("creationflags" in call.kwargs for call in starts.call_args_list), "windowless"
        requested = re.findall(rb"contents ([0-9a-f]{40})", sent)
        assert requested and len(requested) == len(set(requested)), "read each immutable blob once"
        said = check_output(
            ["git", "cat-file", "--batch-command", "--buffer", "-z"],
            cwd=tmp_path,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            input=bytes(sent),
        )
        for option, damage in (  # the session's replies or the graph, damaged: each refuses by name
            ("--batch-command", lambda raw: b""),
            ("--batch-command", lambda raw: b"malformed\n"),
            ("--batch-command", lambda raw: raw[:-2]),
            ("--batch-command", lambda raw: raw + b"trailing"),
            ("--batch-command", lambda raw: raw.replace(b" blob ", b" tree ", 1)),
            ("--parents", lambda raw: merged + " " + "0" * 40 + "\n"),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(spawn, "run", lambda command, **_kwargs: resolved[tuple(command)])
                patch.setattr(
                    spawn,
                    "check_output",
                    lambda command, option=option, damage=damage, **kwargs: (
                        damage(responses[tuple(command)])
                        if option in command
                        else responses[tuple(command)]
                    ),
                )
                if option == "--batch-command":

                    class DamagedSession:
                        def __init__(self, *, damage=damage, **kwargs):
                            self.child = MagicMock(
                                stdin=io.BytesIO(), stdout=io.BytesIO(damage(said)), returncode=0
                            )

                        def start(self, *args, **kwargs):
                            return self.child

                        def send(self, data):
                            self.child.stdin.write(data)

                        def close_input(self):
                            self.child.stdin.close()

                        def check(self):
                            pass

                        def stop(self):
                            pass

                        def __exit__(self, *args):
                            pass

                    patch.setattr(spawn, "InteractiveSession", DamagedSession)
                with pytest.raises(ValueError, match=r"git object response|history graph"):
                    validate_internal_id_history(tmp_path, merged)
        return
    baseline.write_text(json.dumps({**policy, "cards": []}), encoding="utf-8", newline="\n")
    candidate = commit("Remove vocabulary", advance=False)
    with pytest.raises(ValueError, match="vocabulary changed"):
        validate_internal_id_history(tmp_path, candidate)
    policy.update(cards=["SYNTHETIC-CARD"], occurrences={})
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Lower synthetic allowance")
    validate_internal_id_history(tmp_path, "HEAD")
    policy["occurrences"] = {owner: {label: 1}}
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Increase synthetic allowance")
    with pytest.raises(ValueError, match="allowance increased"):
        validate_internal_id_history(tmp_path, "HEAD")
    shallow = tmp_path.parent / (tmp_path.name + "-shallow")
    spawn.check_output(["git", "clone", "--quiet", "--depth=1", tmp_path.as_uri(), str(shallow)])
    shallow_head = spawn.check_output(
        ["git", "-C", str(shallow), "rev-parse", "HEAD"], text=True
    ).strip()
    assert _git("rev-parse", "HEAD") == shallow_head
    with pytest.raises(ValueError, match=r"shallow.*history"):
        validate_internal_id_history(shallow, "HEAD")
    source = candidate = _git("rev-parse", "HEAD")
    parent = _git("rev-parse", "HEAD^")
    for named_parent, allowed, accepted in (
        (parent, {owner: {label: 1}}, True),
        ("0" * 40, {owner: {label: 1}}, False),
        (parent, {owner: {label: 2}}, False),
        (parent, {"src/unlisted.py": {label: 1}}, False),
    ):
        policy["history_admissions"] = {candidate: {"parent": named_parent, "occurrences": allowed}}
        baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
        _git("add", "--all")
        source = _git("commit-tree", _git("write-tree"), "-p", source, "-m", "Admission")
        if accepted:
            validate_internal_id_history(tmp_path, source)
            # a later policy admits the earlier commit by name; the commit alone still refuses
            validate_internal_id_history(tmp_path, candidate, admissions_from=source)
            with pytest.raises(ValueError, match="allowance increased"):
                validate_internal_id_history(tmp_path, candidate)
        else:
            with pytest.raises(ValueError, match="allowance increased"):
                validate_internal_id_history(tmp_path, source)
    policy.pop("history_admissions")
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    (tmp_path / "README.md").write_text("Unrelated change\n", encoding="utf-8", newline="\n")
    commit("Change an unrelated file")
    with pytest.raises(ValueError, match="allowance increased"):
        validate_internal_id_history(tmp_path, "HEAD")
    baseline.unlink()
    commit("Remove synthetic policy")
    with pytest.raises(ValueError, match="no frozen internal-label policy"):
        validate_internal_id_history(tmp_path, "HEAD")
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Reintroduce synthetic policy")
    with pytest.raises(ValueError, match="policy was removed"):
        validate_internal_id_history(tmp_path, "HEAD")


@pytest.mark.proves("pipeline.development_labels_cannot_grow_in_public")
def test_first_landing_freezes_only_its_prepolicy_anchors_labels(tmp_path, synthetic_history):
    """First landing admits anchor labels, then keeps its allowance, source and format fixed."""
    git, commit = synthetic_history
    earlier = git("rev-parse", "HEAD")
    owner = "src/demo.py"
    label, later = "V" + "901", "V" + "902"
    artifact = tmp_path / owner
    artifact.parent.mkdir()
    artifact.write_text(label + "\n", encoding="utf-8", newline="\n")
    commit("Record prepolicy labels")
    anchor = git("rev-parse", "HEAD")
    baseline = tmp_path / INTERNAL_ID_BASELINE
    baseline.parent.mkdir()
    policy = {"schema_version": 1, "source_sha": anchor, "cards": [], "lines": []}
    policy["occurrences"] = {}
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Record a draft allowance")
    artifact.write_text(label + "\n" + later, encoding="utf-8", newline="\n")
    policy.update(schema_version=2, occurrences={owner: {label: 1, later: 1}})
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="allowance increased"):
        validate_internal_id_history(tmp_path, commit("Admit a post-anchor label", advance=False))
    baseline.write_text(
        json.dumps({**policy, "occurrences": {"../foreign.py": {label: 1}}}),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="unsafe manifest path"):
        validate_internal_id_history(tmp_path, commit("Unsafe anchor path", advance=False))
    artifact.write_text(label + "\n", encoding="utf-8", newline="\n")
    policy["occurrences"] = {owner: {label: 1}}
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Land the anchored allowance")
    validate_internal_id_history(tmp_path, "HEAD")
    unrelated = git("commit-tree", git("write-tree"), "-m", "Unrelated anchor")
    for update, refusal in (
        ({"occurrences": {owner: {label: 2}}}, "allowance increased"),
        ({"source_sha": earlier}, "anchor or format changed"),
        ({"schema_version": 1}, "anchor or format changed"),
        ({"source_sha": unrelated}, "anchor must precede"),
    ):
        baseline.write_text(json.dumps({**policy, **update}), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match=refusal):
            validate_internal_id_history(
                tmp_path, commit("Change the frozen policy", advance=False)
            )


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_candidate_script_needs_a_consumer_outside_release_policy():
    script = "scripts/probe_panel_identity.py"
    blobs = {
        "census/entry_reach.json": b'{"modules": []}',
        "scripts/release/public_manifest.py": script.encode(),
        "tests/structural/test_public_manifest.py": script.encode(),
        script: b"print('demo')\n",
    }
    modes = dict.fromkeys(blobs, "100644")
    manifest, audit = generate("a" * 40, blobs, modes)
    assert script not in {row["path"] for row in manifest["public"]}
    assert next(row for row in audit["files"] if row["path"] == script)["kind"] == "PRIVATE"
    blobs["README.md"] = script.encode()
    manifest, _ = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert script in {row["path"] for row in manifest["public"]}


@pytest.mark.proves("pipeline.public_manifest_follows_registered_host_routes")
@pytest.mark.parametrize(
    ("module", "target_mode", "private", "refusal"),
    [
        pytest.param('"handler"', "100644", False, None, id="public-handler-and-imports"),
        pytest.param('"handler"', "100644", True, "door_route_private", id="private-handler"),
        pytest.param("module_name", "100644", False, "door_route_nonliteral", id="nonliteral"),
        pytest.param(
            '"alphalattice.kernel.handler"',
            "100644",
            False,
            "door_route_outside_package",
            id="outside-package",
        ),
        pytest.param('"handler"', None, False, "door_route_untracked", id="untracked"),
    ],
)
def test_public_runtime_closure_follows_the_registered_door_routes(
    monkeypatch, module, target_mode, private, refusal
):
    """The real scanner follows literal public routes and refuses invalid targets by name."""
    registry = _HOST_ROUTES
    package = registry.rsplit("/", 1)[0]
    handler, dependency = f"{package}/handler.py", f"{package}/dependency.py"
    blobs = {
        **_runtime_census(registry),
        registry: f'_ROUTES = (FamilyRoute(("SHOW",), {module}),)\n'.encode(),
        dependency: b"answer = 1\n",
    }
    if target_mode is not None:
        blobs[handler] = b"from .dependency import answer\n"
    modes = dict.fromkeys(blobs, "100644")
    if private:
        monkeypatch.setattr(
            "release.public_manifest.RULES",
            [
                {
                    "rule": "PRIVATE_FIXTURE",
                    "kind": "PRIVATE",
                    "patterns": [handler],
                    "reason": "private fixture",
                },
                *RULES,
            ],
        )
    if refusal:
        with pytest.raises(ValueError, match=f"public_release\\.{refusal}:"):
            generate("a" * 40, blobs, modes)
    else:
        manifest, _ = generate("a" * 40, blobs, modes)
        public = {row["path"]: row["rule"] for row in manifest["public"]}
        assert public[handler] == public[dependency] == "PUBLIC_RUNTIME"


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_candidate_script_reference_composed_from_path_parts_is_public():
    script = "scripts/run_alpha_model_adapter_parity.py"
    blobs = {
        "census/entry_reach.json": b'{"modules": []}',
        "tests/consumer.py": (
            b"from pathlib import Path\n"
            b'script = Path(__file__).parents[2] / "scripts" / '
            b'"run_alpha_model_adapter_parity.py"\n'
        ),
        script: b"print('demo')\n",
    }
    manifest, _ = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert script in {row["path"] for row in manifest["public"]}


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_release_audit_tool_and_pin_stay_private_even_with_public_consumers():
    rule = next(row for row in RULES if row["rule"] == "PRIVATE_RELEASE_AUDIT")
    assert rule["patterns"]
    assert rule["reason"] == "the reuse audit's tool and its pin serve an internal release record"
    for path in rule["patterns"]:
        assert classify(path, {path}, {path}, {path}) == (
            "PRIVATE",
            rule["rule"],
            rule["reason"],
        )


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
def test_private_test_trees_keep_all_file_kinds_out_of_public_scope():
    rule = next(row for row in RULES if row["rule"] == "PRIVATE_TEST")
    trees = [pattern for pattern in rule["patterns"] if pattern.endswith("/**")]
    assert trees
    for pattern in trees:
        for name in ("run.py", "files.json", "measurement.json", "README.md", "future/new.toml"):
            path = pattern.removesuffix("**") + name
            assert classify(path, {path}, {path}, {path}) == (
                "PRIVATE",
                rule["rule"],
                rule["reason"],
            )


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
@pytest.mark.parametrize(
    "path",
    [
        "tests/integration/local_application/test_workbench_readback.py",
        "tests/integration/portfolio_strategy_lab/test_workbench_owner_words.py",
        "tests/portfolio_strategy_lab/workbench_dom.cjs",
        "tests/portfolio_strategy_lab/workbench_minor_ui.cjs",
        "tests/structural/test_workbench_agent_parity.py",
    ],
)
def test_workbench_verification_needs_the_private_ui_qa_kit(path):
    assert classify(path, {path}, {path}, {path}) == (
        "PRIVATE",
        "PRIVATE_TEST",
        "needs the private UI QA kit",
    )


@pytest.mark.proves("pipeline.public_manifest_settles_every_file_by_rule")
@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("tests/integration/alternative_evidence/test_evidence_review_route.py", None),
        (
            "tests/integration/product_host/test_ui_qa_launch_session.py",
            "imports the private UI QA launch-session owner and executes its private drivers",
        ),
        (
            "tests/integration/product_host/test_ui_goal_composer.py",
            "requires the private UI QA browser and kit from its embedded Node.js program",
        ),
        (
            "tests/integration/portfolio_strategy_lab/test_workbench_portfolio_provenance.py",
            "runs the private Portfolio provenance harness and UI QA page census",
        ),
        (
            "tests/integration/task_control/test_workbench_recovery_stack.py",
            "runs the private recovery-stack harness and UI QA stack census",
        ),
        (
            "tests/portfolio_strategy_lab/workbench_portfolio_provenance.cjs",
            "reads the private UI QA page census and pinned-browser runtime",
        ),
        (
            "tests/portfolio_strategy_lab/workbench_recovery_stack.cjs",
            "requires the private UI QA stack census and pinned-browser runtime",
        ),
    ],
)
def test_private_browser_dependencies_keep_their_callers_private(path, reason):
    """A candidate's private browser inputs cannot be omitted under public verification."""
    found = classify(path, {path}, {path}, {path})
    assert found[:2] == ("PRIVATE", "PRIVATE_TEST") and found[2]
    if reason is not None:
        assert found[2] == reason


def test_every_test_path_the_public_ci_runs_exists_and_is_public(tmp_path):
    """A renamed or split test file cannot leave the public CI running a path that is gone, and
    the nightly full run is split by durations that name no private test."""
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text("utf-8")
    triggers = workflow.split("permissions:", 1)[0]
    for event in ("push", "pull_request"):
        declared = re.search(rf"  {event}:\s+branches: \[([^]]+)\]", triggers)
        assert declared and {value.strip() for value in declared[1].split(",")} == {"main", "next"}
    pipeline = definition.load(root)
    body = workflow.split("\njobs:", 1)[1]
    jobs = re.findall(r"^  ([\w-]+):$", body, re.M)
    blocks = dict(zip(jobs, re.split(r"^  [\w-]+:$", body, flags=re.M)[1:], strict=True))
    rows = {row.job for row in pipeline.rows.values() if row.kind == "ci-job"}
    # a row's job may need parts of its own (the public tests' plan and shards); no other job runs
    needs = [re.search(r"^    needs: \[?([^\]\n]+)", blocks.get(job, ""), re.M) for job in rows]
    parts = {part.strip() for found in needs if found for part in found[1].split(",")}
    assert set(jobs) == {"checkout", *rows, *parts}
    assert "scripts/pipeline.py run commit --host public" in workflow
    named = set(re.findall(r"tests/[\w/]+\.py", workflow)) | {*pipeline.rows["tests"].public}
    assert named
    assert sorted(path for path in named if not (root / path).is_file()) == []
    kinds = {path: classify(path, set(), set(), set())[0] for path in named}
    assert {path: kind for path, kind in kinds.items() if kind != "PUBLIC"} == {}
    # every public test runs nightly and on dispatch, collected at run time, never a file list
    assert re.search(r"^  schedule:\n    - cron: ", workflow, re.M)
    assert "\n  workflow_dispatch:\n" in workflow
    assert "public-tests-junit-${{ matrix.os }}-${{ matrix.group }}" in workflow
    assert "public-tests-result-${{ matrix.os }}-${{ github.run_id }}" in workflow
    # every workflow's actions are pinned to a full commit, never a tag or branch
    uses = [
        found
        for path in sorted((root / ".github/workflows").glob("*.yml"))
        for found in re.findall(r"^\s*-?\s*uses:\s*(\S+)", path.read_text("utf-8"), re.M)
    ]
    assert (
        uses
        and [found for found in uses if not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", found)]
        == []
    )
    # Each OS marker follows the matching shard-evidence verifier and is never emitted on failure.
    checked = workflow.index("      - name: Total and verify this operating system's shards\n")
    marker = workflow.index(
        "          name: public-tests-${{ matrix.os }}-${{ needs.public-tests-plan.outputs.sha }}\n"
    )
    assert checked < marker and "--pass-dir" in workflow[checked:marker]
    assert "        if: success()\n" in workflow[checked:marker]
    # every shard proves it ran: a missing JUnit fails it, and the totals refuse a missing record
    assert "          if-no-files-found: error" in workflow and "--expect 1 2 3 4" in workflow
    durations = json.loads((root / ".test_durations").read_text("utf-8"))
    files = {node.split("::", 1)[0] for node in durations}
    assert files and all(type(seconds) is float and seconds >= 0 for seconds in durations.values())
    split = {path: classify(path, set(), set(), set())[0] for path in files}
    assert {path: kind for path, kind in split.items() if kind != "PUBLIC"} == {}

    import yaml

    jobs = yaml.safe_load(workflow)["jobs"]
    shard_command = next(
        step["run"]
        for step in jobs["public-tests-shard"]["steps"]
        if step.get("name") == "Run this shard of every public test"
    )
    launcher, pytest_options = " ".join(shard_command.split()).split(" -- ", 1)
    assert "python scripts/release/public_test_shard.py " in launcher
    assert pytest_options.startswith(
        "tests -m github -n 4 "
    )  # exactly the tests declared for GitHub
    assert "--splits 4 --group ${{ matrix.group }}" in pytest_options
    assert "--durations-path .test_durations" in pytest_options
    assert not jobs["public-tests-shard"].get("continue-on-error", False)
    assert not jobs["public-tests"].get("continue-on-error", False)
    supply = yaml.safe_load((root / ".github/workflows/supply-chain.yml").read_text("utf-8"))
    sbom = supply["jobs"]["sbom"]
    upload = supply["jobs"]["sbom-release"]
    assert sbom.get("permissions", {}).get("contents", "read") == "read"
    assert upload["permissions"]["contents"] == "write"
    assert upload["needs"] == "sbom" and "release" in upload["if"]
    assert all("gh release upload" not in step.get("run", "") for step in sbom["steps"])
    assert any("gh release upload" in step.get("run", "") for step in upload["steps"])
    for name in ("checkout", "public-tests-shard"):
        job = jobs[name]
        assert "npm_config_cache" not in job.get("env", {})
        install = next(
            step for step in job["steps"] if step.get("name") == "Install the locked clone detector"
        )
        assert "npm ci --prefix scripts/test-shape-tools" in install["run"]
        consumer = next(
            step
            for step in job["steps"]
            if step.get("name")
            == (
                "Run the commit stage of config/pipeline.toml"
                if name == "checkout"
                else "Run this shard of every public test"
            )
        )
        for step in (install, consumer):
            assert step["env"]["npm_config_cache"] == "${{ runner.temp }}/jscpd-npm-cache"
    assert not jobs["checkout"].get("continue-on-error", False)


@pytest.mark.proves("pipeline.public_worker_timeout_diagnostic")
@pytest.mark.parametrize(
    "failure", ("timeout", "timeout-end-blocked", "crash", "prior-timeout", "malformed")
)
def test_public_worker_timeout_is_visible_without_relabeling_a_crash(tmp_path, failure):
    """The workflow's real worker plugin reports a timed-out worker but not an ordinary crash, and
    ends a worker whose timeout recorded itself but then blocked (Windows' thread method, the
    public Verify shards cancelled at their 60-minute limit on 11-21)."""
    from tests.structural.pytest_support import child_environment

    root = Path(__file__).resolve().parents[2]
    plugin = (root / "scripts/release/public_test_shard.py").read_text("utf-8")
    plugin = plugin.replace("HANG_SECONDS = 900", "HANG_SECONDS = 2", 1)
    plugin = plugin.replace("HANG_GRACE = 60", "HANG_GRACE = 2", 1)
    (tmp_path / "ci_worker_diagnostics.py").write_text(plugin, encoding="utf-8")
    if failure == "timeout-end-blocked":  # pytest-timeout's own end never reaches os._exit
        (tmp_path / "conftest.py").write_text(
            "import pytest_timeout\nfrom threading import Event\n\n\n"
            "def pytest_configure(config):\n"
            "    pytest_timeout.timeout_timer = lambda item, settings: Event().wait()\n",
            encoding="utf-8",
        )
    body = "Event().wait()" if failure.startswith("timeout") else "os._exit(5)"
    if failure == "prior-timeout":
        body = (
            "log = Path(os.environ['PYTEST_WORKER_DIAGNOSTICS']) / "
            "(os.environ['PYTEST_XDIST_WORKER'] + '-' + str(os.getpid()) + '.log')\n"
            "    with log.open('a', encoding='utf-8') as stream:\n"
            "        for event in [{'event': 'start', 'node': 'previous'}, "
            "{'event': 'timeout', 'node': 'previous', 'seconds': 2.0}, "
            "{'event': 'end', 'node': 'previous'}, "
            "{'event': 'start', 'node': 'test_probe.py::test_probe'}]:\n"
            "            stream.write(json.dumps(event) + '\\n')\n"
            "    os._exit(5)"
        )
    if failure == "malformed":
        body = (
            "log = Path(os.environ['PYTEST_WORKER_DIAGNOSTICS']) / "
            "(os.environ['PYTEST_XDIST_WORKER'] + '-' + str(os.getpid()) + '.log')\n"
            "    with log.open('a', encoding='utf-8') as stream:\n"
            "        stream.write('{\\n')\n"
            "    os._exit(5)"
        )
    (tmp_path / "test_probe.py").write_text(
        (
            "import json, os\nfrom pathlib import Path\n"
            "from threading import Event\ndef test_probe():\n    "
        )
        + body
        + "\n",
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    env = child_environment(
        PYTHONPATH=str(tmp_path),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_WORKER_DIAGNOSTICS=str(diagnostics),
    )
    run = spawn.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist",
            "-p",
            "pytest_timeout",
            "-p",
            "ci_worker_diagnostics",
            "-n",
            "1",
            "--max-worker-restart=0",
            "--timeout-method=thread",
            "-q",
            "test_probe.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    visible = run.stdout + run.stderr
    assert run.returncode != 0 and "crashed while running" in visible
    assert ('"event": "timeout"' in visible) == failure.startswith("timeout")
    if failure.startswith("timeout"):
        assert "test_probe.py::test_probe" in visible and '"seconds": 2.0' in visible
        assert visible.index('"event": "timeout"') < visible.index("crashed while running")
    if failure == "malformed":
        assert "timeout diagnostic unavailable" in visible and "INTERNALERROR" not in visible


@pytest.mark.proves("pipeline.public_worker_timeout_diagnostic")
@pytest.mark.parametrize(
    ("phase", "node", "expected"),
    [
        ("worker startup", None, "gw0: worker startup"),
        ("collection", None, "gw0: collection"),
        (
            "fixture setup",
            "test_probe.py::test_probe",
            "gw0: fixture setup test_probe.py::test_probe",
        ),
        ("test call", "test_probe.py::test_probe", "gw0: test call test_probe.py::test_probe"),
        (
            "fixture teardown",
            "test_probe.py::test_probe",
            "gw0: fixture teardown test_probe.py::test_probe",
        ),
        (
            "worker handoff",
            "test_probe.py::test_probe",
            "gw0: worker handoff test_probe.py::test_probe",
        ),
        ("session finish", None, "gw0: session finish"),
        ("worker shutdown", None, "gw0: worker shutdown"),
    ],
    ids=(
        "startup",
        "collection",
        "setup",
        "call",
        "teardown",
        "handoff",
        "session-finish",
        "shutdown",
    ),
)
def test_a_shard_timeout_names_the_process_phase(phase, node, expected):
    """A phase has a deadline, and a test transition keeps its original deadline."""
    from release.public_test_shard import HANG_SECONDS, ShardWatch

    watch = ShardWatch(began=0)
    watch.observe(
        dict(
            event="phase",
            owner="controller",
            phase="worker coordination",
            tick=1,
            until=None,
            node=None,
            pid=1,
        )
    )
    watch.observe(
        dict(
            event="phase",
            owner="gw0",
            phase="test protocol" if node else phase,
            tick=2,
            until=10,
            node=node,
            pid=2,
        )
    )
    deadline = 10
    if phase == "worker handoff":
        watch.observe(dict(event="end", owner="gw0", node=node, tick=9))
        deadline = 9 + HANG_SECONDS
        watch.observe(
            dict(
                event="phase",
                owner="gw1",
                phase="test call",
                tick=deadline - 1,
                until=deadline + 1,
                node="test_probe.py::test_healthy",
                pid=3,
            )
        )
    elif node is not None:
        watch.observe(dict(event="test phase", owner="gw0", phase=phase, node=node, tick=9))
    before = deadline - 0.1
    assert watch.expired(deadline, cutoff=deadline + 10, alive=lambda _pid: True) == expected
    assert watch.expired(before, cutoff=deadline + 10, alive=lambda _pid: True) is None
    assert watch.expired(deadline, cutoff=deadline + 10, alive=lambda pid: pid == 1) is None
    assert watch.expired(before, cutoff=before, alive=lambda _pid: True) == expected


@pytest.mark.proves("pipeline.public_worker_timeout_diagnostic")
@pytest.mark.parametrize(
    "escaped",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(sys.platform != "linux", reason="Linux process sessions"),
            id="detached-host",
        ),
    ],
)
def test_a_stalled_shard_keeps_the_failure_that_preceded_worker_session_finish(tmp_path, escaped):
    """A real xdist worker that hangs after its last report leaves a failed JUnit."""
    from xml.etree import ElementTree as ET

    from tests.structural.pytest_support import child_environment

    probe = "def test_prior_failure():\n"
    if escaped:
        detached = (
            "import json, os, subprocess, sys\n"
            "from pathlib import Path\n"
            "import psutil\n"
            "from alphalattice.capabilities.domain_contracts import spawn\n"
            "child = spawn.detached(\n"
            "    [sys.executable, '-c', 'from threading import Event; Event().wait(60)'],\n"
            "    log=subprocess.DEVNULL, env=dict(os.environ),\n"
            ")\n"
            "birth = psutil.Process(child.pid).create_time()\n"
            "Path('host.json').write_text(json.dumps([child.pid, birth]), encoding='utf-8')\n"
        )
        probe += (
            "    import sys\n"
            "    from alphalattice.capabilities.domain_contracts import spawn\n"
            f"    spawn.run([sys.executable, '-c', {detached!r}], timeout=8, check=True)\n"
        )
    line = probe.count("\n") + 1
    probe += "    assert False, 'child output stays out'\n"  # a message quoting a child's output
    probe += "\ndef test_value_only():\n    import pytest\n"
    probe += (
        "    pytest.fail('child output stays out', pytrace=False)\n"  # no frame, only its crash
    )
    value_line = probe.count("\n")
    (tmp_path / "test_probe.py").write_text(probe, encoding="utf-8")
    # Import and syntax errors reach pytest as text alone; their own error names the line.
    (tmp_path / "test_broken.py").write_text("import child_output_stays_out\n", encoding="utf-8")
    (tmp_path / "test_syntax.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "from threading import Event\n"
        "import pytest\n\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_sessionfinish(session):\n"
        "    if hasattr(session.config, 'workerinput'):\n"
        "        Event().wait()\n",
        encoding="utf-8",
    )
    junit = tmp_path / "junit.xml"
    env = child_environment(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_WORKER_DIAGNOSTICS=str(tmp_path / "diagnostics"),
    )
    root = Path(__file__).resolve().parents[2]
    guard = (root / "scripts/release/public_test_shard.py").read_text("utf-8")
    # Shorten the phase bound without changing which lifecycle hooks arm it.
    guard = guard.replace("HANG_SECONDS = 900", "HANG_SECONDS = 4", 1)
    launcher = tmp_path / "public_test_shard.py"
    launcher.write_text(guard, encoding="utf-8")
    run = spawn.run(
        [
            sys.executable,
            str(launcher),
            "--budget",
            "20",
            "--junit",
            str(junit),
            "--",
            "-p",
            "xdist",
            "-p",
            "pytest_timeout",
            "-n",
            "1",
            "--max-worker-restart=0",
            "--continue-on-collection-errors",
            "-q",
            "test_probe.py",
            "test_broken.py",
            "test_syntax.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if escaped:
        import psutil

        pid, created = json.loads((tmp_path / "host.json").read_text("utf-8"))
        try:
            host = psutil.Process(pid)
            same_host = host.create_time() == created and host.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            same_host = False
        if same_host:
            host.kill()  # Clean only the identified planted process, even on the red base.
        assert not same_host, f"shard-owned Host {pid} survived the stop"
    assert run.returncode == 1, run.stdout + run.stderr
    assert "public shard timeout: gw0: session finish" in run.stderr
    rows = {row.get("name"): row for row in ET.parse(junit).findall(".//testcase")}
    assert len(rows) == 5, rows
    failure, value_only = rows["test_prior_failure"], rows["test_value_only"]
    assert failure.get("classname") == "test_probe"
    # Each keeps its type, then the failing file and line, and never its message or child text.
    assert failure.findtext("failure").splitlines() == ["AssertionError", f"test_probe.py:{line}"]
    assert value_only.findtext("failure").splitlines() == ["Failed", f"test_probe.py:{value_line}"]
    assert rows["test_broken"].findtext("error").splitlines()[-1] == "test_broken.py:1"
    assert rows["test_syntax"].findtext("error").splitlines()[-1] == "test_syntax.py:1"
    diagnostics = [log.read_text("utf-8") for log in (tmp_path / "diagnostics").glob("*.log")]
    kept = [junit.read_text("utf-8"), *diagnostics]
    assert diagnostics and not any(re.search("stays.out", text) for text in kept)
    timeout = rows["public shard timeout: gw0: session finish"]
    assert timeout.find("error").get("message") == timeout.get("name")


@pytest.mark.proves("pipeline.public_durations_come_from_junit")
def test_the_durations_come_from_junit_keep_the_latest_run_and_leave_out_private_and_skipped(
    tmp_path, capsys
):
    """The nightly refreshes .test_durations from its shards' JUnit; bad records refuse by name."""
    from release.test_durations import main

    (tmp_path / "tree/tests/a").mkdir(parents=True)
    (tmp_path / "tree/tests/a/test_one.py").touch()
    case = (
        '<testcase classname="tests.a.test_one{cls}" name="{name}" time="{time}">{body}</testcase>'
    )
    records = []
    for index, cases in enumerate(
        (
            [("", "test_kept", "9.5", ""), ("", "test_skipped", "1", "<skipped/>")],
            [
                ("", "test_kept", "2.25", ""),
                (".TestGroup", "test_in_class", "0.5", ""),
                ("", "test_torn", "1", ""),
                ("", "test_torn", "0", "<error/>"),
                # ids the export's scan would flag: a contact, and a path outside the tree
                ("", "test_named[x" + chr(64) + "private.company]", "4", ""),
                ("", "test_named[tests/p/test_gone.py]", "4", ""),
            ],
        )
    ):
        rows = "".join(case.format(cls=c, name=n, time=t, body=b) for c, n, t, b in cases)
        rows += case.format(cls="", name="test_private", time="3", body="").replace(
            "tests.a.test_one", "tests.p.test_gone"
        )
        records.append(tmp_path / f"junit-{index}.xml")
        records[-1].write_text(
            f"<testsuites><testsuite>{rows}</testsuite></testsuites>", encoding="utf-8"
        )
    out, summary = tmp_path / ".test_durations", tmp_path / "summary.json"
    assert (
        main(
            [
                str(out),
                *map(str, records),
                "--tree",
                str(tmp_path / "tree"),
                "--summary",
                str(summary),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text("utf-8")) == {
        "tests/a/test_one.py::TestGroup::test_in_class": 0.5,
        "tests/a/test_one.py::test_kept": 2.25,  # the later run wins
        "tests/a/test_one.py::test_torn": 0.0,
    }  # skipped and outside the tree are left out
    assert json.loads(summary.read_text("utf-8")) == dict(
        passed=4,
        failed=0,
        errors=1,
        skipped=1,
        nodes=3,
        test_seconds=2.8,  # both still ran
    )
    capsys.readouterr()
    (broken := tmp_path / "broken.xml").write_text("<testsuite>", encoding="utf-8")
    shard = {}  # each shard's record in its artifact's folder, as CI downloads them
    for name in ("1", "2", "3", "4", "9", "extra-4", "other"):
        folder = "other-junit-4" if name == "other" else f"public-tests-junit-{name}"
        shard[name] = tmp_path / folder / "junit.xml"
        shard[name].parent.mkdir()
        shard[name].write_text(records[0].read_text("utf-8"), encoding="utf-8")
    for given, expected in (
        ("1 2 3", "1 2 3 4"),  # one shard left no JUnit
        ("1 2 3 9", "1 2 3 4"),  # as many records, but one is another shard's
        ("1 2 3 extra-4", "1 2 3 4"),  # a name that only ends in the shard
        ("1 2 3 other", "1 2 3 4"),  # another artifact's prefix
        ("1 1 2 3", "1 2 3 4"),  # one shard's record twice
        ("1 2 3 4", "1 1 2 3 4"),  # a shard declared twice
    ):
        with pytest.raises(SystemExit, match=r"test_durations.records_mismatch: "):
            main([str(out), *(str(shard[n]) for n in given.split()), "--expect", *expected.split()])
    every = [str(shard[n]) for n in ("4", "2", "3", "1")]  # each shard once, in any order
    assert main([str(out), *every, "--expect", "1", "2", "3", "4"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit, match=r"test_durations.record_unreadable: .*broken.xml"):
        main([str(out), str(broken)])
    # Per-OS totals keep the Windows and Linux evidence independent and bind each passing shard.
    sha = "a" * 40
    for os_name in ("windows", "linux"):
        os_records, pass_root = [], tmp_path / f"passes-{os_name}"
        pass_root.mkdir()
        for group in range(1, 5):
            folder = tmp_path / f"public-tests-junit-{os_name}-{group}"
            folder.mkdir()
            xml = folder / "junit.xml"
            xml.write_text(
                f'<testsuite><testcase classname="tests.a.test_one" name="test_{group}" '
                'time="1"/></testsuite>',
                encoding="utf-8",
            )
            os_records.append(xml)
            marker = pass_root / f"public-tests-pass-{os_name}-{group}"
            marker.mkdir()
            (marker / "passed.json").write_text(
                json.dumps({"os": os_name, "sha": sha, "group": group}), encoding="utf-8"
            )
        result, os_summary = (
            tmp_path / f"result-{os_name}.json",
            tmp_path / f"summary-{os_name}.json",
        )

        def run_os(
            records,
            os_name=os_name,
            pass_root=pass_root,
            os_summary=os_summary,
            result=result,
            tree=None,
        ):
            args = [
                str(out),
                *map(str, records),
                "--expect",
                "1",
                "2",
                "3",
                "4",
                "--os",
                os_name,
                "--sha",
                sha,
                "--pass-dir",
                str(pass_root),
                "--summary",
                str(os_summary),
                "--result",
                str(result),
            ]
            if tree is not None:
                args += ["--tree", str(tree)]
            try:
                return main(args)
            except SystemExit as error:
                pytest.fail(f"per-OS result producer refused: {error}")

        assert run_os(os_records) == 0
        assert json.loads(result.read_text(encoding="utf-8")) == {
            "os": os_name,
            "sha": sha,
            "status": "success",
        }
        assert json.loads(os_summary.read_text(encoding="utf-8"))["passed"] == 4
        marker = pass_root / f"public-tests-pass-{os_name}-2" / "passed.json"
        marker_bytes, original = marker.read_bytes(), os_records[1].read_bytes()
        os_records[1].write_text(
            '<testsuite><testcase classname="tests.a.test_one" name="test_2" time="1">'
            '<skipped/></testcase><testcase classname="tests.a.test_one" name="test_2" '
            'time="1"/></testsuite>',
            encoding="utf-8",
        )
        assert run_os(os_records) == 0
        assert json.loads(os_summary.read_text(encoding="utf-8"))["skipped"] == 0
        os_records[1].write_bytes(original)
        marker.unlink()
        marker.parent.rmdir()
        for bad in ("failure", "error"):
            for cases in (
                f'<testcase classname="tests.a.test_one" name="test_2" time="1">'
                f"<{bad}/><skipped/></testcase>",
                f'<testcase classname="tests.a.test_one" name="test_2" time="1"><{bad}/></testcase>'
                '<testcase classname="tests.a.test_one" name="test_2" time="1">'
                "<skipped/></testcase>",
            ):
                os_records[1].write_text(
                    f'<testsuite>{cases}<testcase classname="tests.a.test_one" '
                    'name="sibling" time="1"/></testsuite>',
                    encoding="utf-8",
                )
                assert run_os(os_records) == 1
                assert json.loads(result.read_text(encoding="utf-8"))["status"] == "failed"
                assert (
                    run_os(os_records, tree=tmp_path) == 1
                )  # filtering durations cannot clear a shard failure
                assert json.loads(result.read_text(encoding="utf-8"))["status"] == "failed"
                assert (
                    json.loads(out.read_text(encoding="utf-8")) == {}
                )  # filtered names stay private
        marker.parent.mkdir()
        marker.write_bytes(marker_bytes)
        os_records[1].write_bytes(original)
        assert (
            run_os(os_records) == 0
        )  # restore this OS's passing result for the isolation control below
        if os_name == "windows":
            assert main([str(out), *map(str, os_records), "--expect", "1", "2", "3", "4"]) == 0
        if os_name == "linux":
            with pytest.raises(SystemExit, match=r"test_durations\.records_mismatch"):
                main([str(out), *map(str, os_records), "--expect", "1", "2", "3", "4"])
            marker = pass_root / "public-tests-pass-linux-2" / "passed.json"
            marker.unlink()
            marker.parent.rmdir()  # a failed shard emits no pass artifact
            os_records[1].write_text(
                '<testsuite><testcase classname="tests.a.test_one" name="test_2" time="1">'
                "<failure/></testcase></testsuite>",
                encoding="utf-8",
            )
            assert run_os(os_records) == 1
            assert json.loads(result.read_text(encoding="utf-8"))["status"] == "failed"
            assert (
                json.loads((tmp_path / "result-windows.json").read_text(encoding="utf-8"))["status"]
                == "success"
            )
            assert run_os(os_records[:-1]) == 2
            assert json.loads(result.read_text(encoding="utf-8"))["status"] == "unknown"
            os_records[1].write_text(
                '<testsuite><testcase classname="tests.a.test_one" name="test_2" '
                'time="1"/></testsuite>',
                encoding="utf-8",
            )
            assert (
                run_os(os_records) == 2
            )  # a passing shard without its success artifact is unknown
            marker.parent.mkdir()
            marker.write_text(
                json.dumps({"os": "windows", "sha": sha, "group": 2}), encoding="utf-8"
            )
            assert run_os(os_records) == 2  # an artifact from another OS cannot prove this shard
            marker.write_text(json.dumps({"os": "linux", "sha": sha, "group": 2}), encoding="utf-8")
            os_records[1].write_text(
                '<testsuite><testcase classname="tests.a.test_one" name="test_2" time="1">'
                "<failure/></testcase></testsuite>",
                encoding="utf-8",
            )
            assert run_os(os_records) == 2  # a failure and a pass marker conflict
            assert json.loads(result.read_text(encoding="utf-8"))["status"] == "unknown"


@pytest.mark.proves("pipeline.public_review_never_copies_a_disclosure")
@pytest.mark.parametrize(
    "path,text,rule,mode,admitted",
    [
        ("README.md", "\r\n备注 x@private.company\r\n", "NO_PERSONAL_CONTACT", "key", True),
        ("README.md", " 备注 x@private.company", "NO_PERSONAL_CONTACT", "key", False),
        ("README.md", "备注 other@private.company", "NO_PERSONAL_CONTACT", "key", False),
        ("README.zh-CN.md", "备注 x@private.company", "NO_PERSONAL_CONTACT", "key", False),
        ("README.md", "备注 x@private.company", "NO_INTERNAL_ID", "key", False),
        ("README.md", "\n备注 x@private.company", "NO_PERSONAL_CONTACT", "same", True),
        ("README.md", "备注 x@private.company", "NO_PERSONAL_CONTACT", "conflict", "error"),
        ("README.md", "备注 x@private.company", "NO_PERSONAL_CONTACT", "unknown", "error"),
        ("README.md", "备注 x@private.company", "NO_PERSONAL_CONTACT", "legacy", "error"),
        ("README.md", "\n备注 x@example.invalid", "NO_PERSONAL_CONTACT", "auto", True),
        ("README.md", "备注 x@private.company", "NO_PERSONAL_CONTACT", "auto", False),
        (
            "README.md",
            "备注 x@example.invalid y@private.company",
            "NO_PERSONAL_CONTACT",
            "auto",
            False,
        ),
        *(
            pytest.param(
                "README.md",
                _CHANGED_REVIEW_FIXTURE if mode == "default-changed" else _REVIEW_FIXTURE,
                "NO_SECRET",
                mode,
                admitted,
                id=mode,
            )
            for mode, admitted in (
                ("default-kept", True),
                ("default-changed", False),
                ("default-new", False),
                ("default-missing", False),
                ("default-fresh", False),
                ("default-bad", False),
                ("default-duplicate", False),
                ("default-explicit", True),
                ("default-collision", False),
            )
        ),
    ],
)
def test_a_disclosure_review_follows_only_its_proved_line_content(
    path, text, rule, mode, admitted, tmp_path, monkeypatch
):
    """Moves retain content reviews; edits reopen; reserved-only contacts self-review."""
    version = "alphalattice.release.review-key.v1"
    digest = hashlib.sha256("备注 x@private.company".encode()).hexdigest()
    row = dict(
        path="README.md",
        line=1,
        column=4,
        rule=rule,
        disposition="KEEP_SYNTHETIC",
        evidence="fixture",
    )
    if mode.startswith("default-"):
        row["review_key"] = [
            version,
            path,
            rule,
            hashlib.sha256(_REVIEW_FIXTURE.encode()).hexdigest(),
        ]
        previous = {**row, "reopen_reason": "historical"}
        rows = [row, {**row, "evidence": "corroborating"}]
        standalone = {
            "current_findings": [] if mode == "default-new" else rows,
            "reopened_findings": [previous],
            "original_findings": [],
            "corrections": [],
            "rounds": [],
        }
        review_path = (
            "config/release/public-manifest.json"
            if mode == "default-collision"
            else "review-ledger.json"
        )
        blobs = {
            path: (text + "\n").encode(),
            review_path: json.dumps({"standalone": standalone}).encode(),
        }
        if mode in {"default-missing", "default-fresh"}:
            blobs.pop(review_path)
        elif mode in {"default-bad", "default-explicit"}:
            blobs[review_path] = b"{"
        elif mode == "default-duplicate":
            blobs[review_path] = b'{"standalone":{"current_findings":[{}],"current_findings":[]}}'
        findings, _ = scan_public(blobs, [{"path": path, "kind": "PUBLIC"}])
        content_keys(findings, blobs)
        monkeypatch.setattr(
            "release.public_manifest.generate",
            lambda *_: (
                {},
                {
                    "source_sha": "a" * 40,
                    "findings": findings,
                    "counts": {},
                    "finding_counts": {},
                    "command": "fixture",
                },
            ),
        )
        monkeypatch.setattr("release.public_manifest.render", lambda _audit: "fixture")
        monkeypatch.setattr(
            "release.public_manifest.read_tree",
            lambda _repo, _source: ("a" * 40, blobs, dict.fromkeys(blobs, "100644")),
        )
        argv = ["public_manifest.py", "--out", str(tmp_path), "--census-path", review_path]
        if mode == "default-explicit":
            override = tmp_path / "verified.json"
            override.write_text(json.dumps({"standalone": standalone}), encoding="utf-8")
            argv += ["--verification-json", str(override)]
        monkeypatch.setattr(sys, "argv", argv)
        if mode == "default-collision":
            with pytest.raises(SystemExit, match=r"public_release\.output_collision"):
                public_manifest_main()
            assert not list(tmp_path.rglob("*.json"))
            return
        if mode in {"default-missing", "default-bad", "default-duplicate"}:
            target = tmp_path / review_path
            target.write_text("existing review", encoding="utf-8")
            with pytest.raises(SystemExit, match=r"public_release\.verification_invalid"):
                public_manifest_main()
            assert target.read_text(encoding="utf-8") == "existing review"
            return
        try:
            code = public_manifest_main()
        except SystemExit as error:
            code = error.code
        assert code == 0
        if mode == "default-fresh":
            audit = json.loads((tmp_path / review_path).read_text(encoding="utf-8"))
            assert "--census-path=" + review_path in audit["command"]
            assert audit["findings"] and all("review" not in f for f in audit["findings"])
            monkeypatch.setattr(sys, "argv", [*argv, "--check"])
            assert public_manifest_main() == 0
            blobs[review_path] = (tmp_path / review_path).read_bytes()
            monkeypatch.setattr(sys, "argv", argv)
            assert public_manifest_main() == 0
            return
        audit = json.loads((tmp_path / review_path).read_text(encoding="utf-8"))
        assert "--census-path=" + review_path in audit["command"]
        assert "standalone" in audit
        assert audit["standalone"]["reopened_findings"][0] == previous
        finding = next(f for f in audit["findings"] if f["path"] == path and f["rule"] == rule)
        assert ("review" in finding) is admitted
        current = [f for f in audit["standalone"]["current_findings"] if f["rule"] == rule]
        policy = next(f for f in audit["findings"] if f["rule"] == "NO_PERSONAL_CONTACT")
        assert policy["review"]["reviewer"] == "reserved-example-domain-policy"
        if mode in {"default-kept", "default-explicit"}:
            assert current == rows
        elif mode == "default-changed":
            assert current == []
            assert audit["standalone"]["reopened_findings"][1:] == [
                {**item, "reopen_reason": "content_changed"} for item in rows
            ]
        else:
            assert current == []
        return
    if mode != "legacy":
        row["review_key"] = ["unknown" if mode == "unknown" else version, "README.md", rule, digest]
    rows = [] if mode == "auto" else [row]
    if mode in {"same", "conflict"}:
        rows.append(
            dict(
                row,
                evidence="corroborating",
                disposition="REJECT" if mode == "conflict" else "KEEP_SYNTHETIC",
            )
        )
    blobs = {path: text.encode(), "census/entry_reach.json": b'{"modules": []}'}
    _, audit = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    if admitted == "error":
        with pytest.raises(ValueError, match="review key"):
            attach_reviews(audit, {"standalone": {"current_findings": rows}}, Path(), blobs)
        return
    attach_reviews(audit, {"standalone": {"current_findings": rows}}, Path(), blobs)
    finding = audit["findings"][0]
    assert all(("review" in f) == admitted for f in audit["findings"])
    assert finding["review_key"][:3] == [version, path, "NO_PERSONAL_CONTACT"]
    if admitted:
        assert finding["line"] == 2
        assert (
            finding["review_key"][-1]
            == hashlib.sha256(text.replace("\r\n", "\n").split("\n")[1].encode()).hexdigest()
        )
