"""Public scope rules, immutable hashes and disclosure redaction hold the snapshot contract."""

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from release.public_manifest import (
    INTERNAL_ID_BASELINE,
    RULES,
    classify,
    generate,
    internal_id_label,
    internal_id_pattern,
    internal_id_ratchet,
    scan_public,
    validate_internal_id_history,
    validate_path,
)


@pytest.mark.parametrize(
    ("path", "kind", "rule"),
    [
        ("LICENSE", "PUBLIC", "PUBLIC_LEGAL"),
        ("NOTICE", "PUBLIC", "PUBLIC_LEGAL"),
        ("README.md", "PUBLIC", "PUBLIC_DOC"),
        ("CONTRIBUTING.md", "PUBLIC", "PUBLIC_DOC"),
        ("CLA.md", "PUBLIC", "PUBLIC_DOC"),
        ("GOVERNANCE.md", "PUBLIC", "PUBLIC_DOC"),
        ("SECURITY.md", "PUBLIC", "PUBLIC_DOC"),
        (".github/CODEOWNERS", "PUBLIC", "PUBLIC_DOC"),
        (".github/ISSUE_TEMPLATE/bug_report.yml", "PUBLIC", "PUBLIC_DOC"),
        (".github/ISSUE_TEMPLATE/feature_request.yml", "PUBLIC", "PUBLIC_DOC"),
        (".github/PULL_REQUEST_TEMPLATE.md", "PUBLIC", "PUBLIC_DOC"),
        (".github/workflows/ci.yml", "PUBLIC", "PUBLIC_BUILD"),
        (".github/workflows/internal.yml", "UNSETTLED", None),
        ("benchmark/measure.py", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("case-study/readme.md", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("probe/measure.py", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("config/alpha-arm-control.yaml", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("config/dynamic-panel-risk.yaml", "PRIVATE", "PRIVATE_DEVELOPMENT_TOOL"),
        ("tests/agent_eval/test_one.py", "PRIVATE", "PRIVATE_TEST"),
        (
            "tests/portfolio_strategy_lab/test_native_answer_route_class.py",
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
        ("config/release/public-manifest.json", "PRIVATE", "PRIVATE_RECORD"),
        ("scripts/agent_eval/one.py", "PRIVATE", "PRIVATE_QA_TOOL"),
        ("scripts/qa_retention.py", "PRIVATE", "PRIVATE_QA_TOOL"),
        ("tests/researcher_methodology_surface/test_qa_retention.py", "PRIVATE", "PRIVATE_TEST"),
        ("config/agent-eval/one.yaml", "PRIVATE", "PRIVATE_QA_TOOL"),
        (".claude/agents/research.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        (".codex/config.toml", "PUBLIC", "PUBLIC_INTEGRATION"),
        (".agents/skills/research/SKILL.md", "PUBLIC", "PUBLIC_INTEGRATION"),
        ("docs/public-source/README.md", "PUBLIC", "PUBLIC_DOC"),
        ("docs/images/workbench-light.webp", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/workbench-dark.webp", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/images/goal-conversation.webp", "PUBLIC", "PUBLIC_ASSET"),
        ("docs/generate_reference.py", "PUBLIC", "PUBLIC_DOC"),
        ("scripts/run_alphalattice.py", "PUBLIC", "PUBLIC_USER_SCRIPT"),
        ("scripts/unexamined.py", "UNSETTLED", None),
        ("config/unexamined.yaml", "UNSETTLED", None),
        ("src/alphalattice/unreached.py", "UNSETTLED", None),
    ],
)
def test_public_manifest_prioritizes_private_rules_and_never_guesses(path, kind, rule):
    decision = classify(path, set(), set(), set())
    assert decision[:2] == (kind, rule)
    assert decision[2]


def test_the_full_release_closure_check_names_its_private_inventory_input():
    assert classify("tests/structural/test_public_release_closure.py", set(), set(), set()) == (
        "PRIVATE",
        "PRIVATE_TEST",
        "needs the private release inventory",
    )


@pytest.mark.parametrize("path", ["../src/one.py", "/src/one.py", "src\\one.py", "C:/one.py"])
def test_public_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="unsafe manifest path"):
        validate_path(path)


def test_public_manifest_hashes_supplied_git_bytes_and_covers_every_file():
    sha = "a" * 40
    census = {"base": "b" * 40, "modules": [{"path": "src/alphalattice/main.py", "class": "LIVE"}]}
    blobs = {
        "LICENSE": b"license\n",
        "AGENTS.override.md": b"internal instructions\n",
        "census/entry_reach.json": json.dumps(census).encode(),
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


def test_generate_walks_development_labels_once_and_merges_its_findings(monkeypatch):
    """Generation runs the development-label walk once and keeps its findings."""
    walk = Mock(wraps=internal_id_ratchet)
    monkeypatch.setattr("release.public_manifest.internal_id_ratchet", walk)
    blobs = {
        "census/entry_reach.json": b'{"modules": []}',
        INTERNAL_ID_BASELINE: b'{"cards": [], "lines": [], "occurrences": {}}',
        "README.md": b"V" + b"900\n",
        "scripts/probe_panel_identity.py": b"print('demo')\n",
    }
    manifest, audit = generate("a" * 40, blobs, dict.fromkeys(blobs, "100644"))
    assert walk.call_count == 1
    assert audit["findings"] == audit["internal_id_ratchet"]["violations"]
    assert len(audit["findings"]) == 1 and audit["internal_id_ratchet"]["policy_present"]
    assert INTERNAL_ID_BASELINE not in {row["path"] for row in manifest["public"]}


def test_public_manifest_records_locations_without_copying_disclosures():
    secret = "sk-" + "X" * 40
    session = "12345678-1234-4123-8123-123456789abc"
    text = "\n".join(
        [
            secret,
            "D:/" + "AI Agent/private/file",
            "session=" + session,
            "contact=person@private.invalid",
            "experience/one.md",
        ]
    )
    blobs = {"src/public.py": text.encode(), "experience/one.md": b"private content"}
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


def test_development_label_ratchet_pins_each_public_files_count_and_spelling():
    owner = "src/public.py"
    private = "experience/private.md"
    labels = [
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


def test_development_vocabulary_matches_internal_forms_without_refusing_ordinary_words():
    """Ordinary words stay prose while qualified line and card forms remain labels."""
    cards = ["UI", "LEAD", "IS", "ACCEPT", "BADGE", "JOURNEY", "LAUNCH", "LAWS", "SEAL"]
    ux, perf, fix = "U" + "X", "PERF" + "2", "FIX" + "1"
    policy = {"cards": cards, "lines": ["UI", "LEAD", ux, perf, fix]}
    text = " ".join(cards) + f" UX\nCodex {ux} Claude {perf} {fix}\ncard {cards[6]} {cards[4]}:"
    assert [
        internal_id_label(match.group()) for match in internal_id_pattern(policy).finditer(text)
    ] == [ux, perf, fix, cards[6], cards[4]]


@pytest.fixture
def synthetic_history(tmp_path):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Synthetic Author",
                "-c",
                "user.email=synthetic@example.test",
                "-c",
                "core.hooksPath=",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(tmp_path),
                *args,
            ],
            text=True,
        ).strip()

    def commit(message, *, advance=True):
        git("add", "--all")
        if not advance:
            return git("commit-tree", git("write-tree"), "-p", "HEAD", "-m", message)
        git("commit", "-q", "-m", message)

    git("init", "-q")
    (tmp_path / "README.md").write_text("Synthetic tree\n", encoding="utf-8", newline="\n")
    commit("Start synthetic history")
    return git, commit


def test_committed_development_label_allowance_cannot_grow_or_reset(tmp_path, synthetic_history):
    _git, commit = synthetic_history
    with pytest.raises(ValueError, match="no frozen internal-label policy"):
        validate_internal_id_history(tmp_path, "HEAD")
    owner, label = "src/demo.py", "V" + "901"
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
    artifact.write_text(label + "\n", encoding="utf-8", newline="\n")
    policy["occurrences"] = {owner: {label: 1}}
    baseline.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    commit("Land the anchored allowance")
    validate_internal_id_history(tmp_path, "HEAD")
    for update, refusal in (
        ({"occurrences": {owner: {label: 2}}}, "allowance increased"),
        ({"source_sha": earlier}, "anchor or format changed"),
        ({"schema_version": 1}, "anchor or format changed"),
    ):
        baseline.write_text(json.dumps({**policy, **update}), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match=refusal):
            validate_internal_id_history(
                tmp_path, commit("Change the frozen policy", advance=False)
            )


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


@pytest.mark.parametrize(
    "path",
    [
        "tests/portfolio_strategy_lab/test_workbench_readback.py",
        "tests/portfolio_strategy_lab/test_workbench_owner_words.py",
        "tests/portfolio_strategy_lab/workbench_dom.cjs",
        "tests/portfolio_strategy_lab/workbench_minor_ui.cjs",
    ],
)
def test_workbench_verification_needs_the_private_ui_qa_kit(path):
    assert classify(path, {path}, {path}, {path}) == (
        "PRIVATE",
        "PRIVATE_TEST",
        "needs the private UI QA kit",
    )


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("tests/alternative_evidence_desk/test_evidence_review_route.py", None),
        (
            "tests/portfolio_strategy_lab/test_ui_qa_launch_session.py",
            "imports the private UI QA launch-session owner and executes its private drivers",
        ),
        (
            "tests/portfolio_strategy_lab/test_ui_goal_composer.py",
            "requires the private UI QA browser and kit from its embedded Node.js program",
        ),
        (
            "tests/portfolio_strategy_lab/test_workbench_portfolio_provenance.py",
            "runs the private Portfolio provenance harness and UI QA page census",
        ),
        (
            "tests/portfolio_strategy_lab/test_workbench_recovery_stack.py",
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


def test_every_test_path_the_public_ci_runs_exists_and_is_public():
    """A renamed or split test file cannot leave the public CI running a path that is gone."""
    root = Path(__file__).resolve().parents[2]
    named = set(
        re.findall(r"tests/[\w/]+\.py", (root / ".github/workflows/ci.yml").read_text("utf-8"))
    )
    manifest = json.loads((root / "config/release/public-manifest.json").read_text("utf-8"))
    public = {row["path"] for row in manifest["public"]}
    assert named
    assert sorted(path for path in named if not (root / path).is_file()) == []
    assert sorted(named - public) == []
