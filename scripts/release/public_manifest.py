"""Inventory one committed tree for RR4; never build or publish a snapshot.

The manifest uses the RR2 builder's contract. The companion census retains
PRIVATE decisions, unsettled evidence and redacted scan findings. Blob bytes,
not checkout files, determine both classifications and PUBLIC SHA-256 values.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import posixpath
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_ID_BASELINE = "census/public-internal-id-baseline.json"
RULES = [
    {
        "rule": "PRIVATE_CREDENTIAL",
        "kind": "PRIVATE",
        "patterns": [
            ".env",
            ".env.*",
            "**/.env",
            "**/.env.*",
            "*.key",
            "*.pem",
            "credentials.*",
            "**/credentials.*",
        ],
        "reason": "Credentials and machine-specific secret files are never release inputs.",
    },
    {
        "rule": "PRIVATE_STATE",
        "kind": "PRIVATE",
        "patterns": [
            "workspaces/**",
            "**/sessions/**",
            ".claude/projects/**",
            "**/artifacts/**",
            "playpen-board/**",
            ".alphalattice/user/**",
        ],
        "reason": (
            "Workspace, QA, session and board state belong to their private owners; the user "
            "layer is the person's own, and no release contains it."
        ),
    },
    {
        "rule": "PRIVATE_AGREEMENT",
        "kind": "PRIVATE",
        "patterns": [
            "AGENTS.override.md",
            "**/AGENTS.override.md",
            "**/AGENTS.md",
            "**/CLAUDE.md",
        ],
        "reason": (
            "A worktree's maintainer instructions and a nested working agreement are not "
            "installed research-agent integrations; the root guide and its Claude import are."
        ),
    },
    {
        "rule": "PRIVATE_RECORD",
        "kind": "PRIVATE",
        "patterns": [
            "product-design/**",
            "implemented-plans/**",
            "experience/**",
            "census/**",
            "config/release/public-manifest.json",
        ],
        "reason": (
            "Internal design, development records and release provenance "
            "stay outside the public snapshot."
        ),
    },
    {
        "rule": "PRIVATE_QA_TOOL",
        "kind": "PRIVATE",
        "patterns": [
            "scripts/agent_eval/**",
            "scripts/ui_qa/**",
            "config/agent-eval/**",
            "third_party/code-analysis/**",
            "scripts/document_indexes.py",
            "scripts/qa_retention.py",
            "probe/source_release/**",
            "probe/test-asset-governance/**",
        ],
        "reason": (
            "These tools operate internal evaluation, UI QA, inventories or "
            "the superseded private source candidate."
        ),
    },
    {
        "rule": "PRIVATE_DEVELOPMENT_TOOL",
        "kind": "PRIVATE",
        "patterns": [
            "benchmark/**",
            "case-study/**",
            "probe/**",
            "config/alpha-arm-*",
            "config/dynamic-panel-*",
            "config/factor-alpha-campaign.yaml",
            "config/g6-model-lifecycle.yaml",
            "config/risk-estimator-campaign.yaml",
            "config/evidence-roots.json",
            "config/performance-baselines.json",
            "config/store-structural-baseline.json",
        ],
        "reason": (
            "OW8 keeps development campaigns, probes and private QA location declarations "
            "outside the product snapshot."
        ),
    },
    {
        "rule": "PRIVATE_RELEASE_AUDIT",
        "kind": "PRIVATE",
        "patterns": [
            "scripts/release/reuse_inventory.py",
            "tests/structural/test_reuse_spec.py",
        ],
        "reason": "the reuse audit's tool and its pin serve an internal release record",
    },
    {
        "rule": "PRIVATE_TEST",
        "kind": "PRIVATE",
        "patterns": [
            "tests/agent_eval/**",
            "tests/operation_suite/**",
            "tests/alpha_research/test_alpha_product_replay.py",
            "tests/alpha_research/test_live_closure_entry_points.py",
            "tests/alternative_evidence_desk/test_evidence_review_route.py",
            "tests/alternative_evidence_desk/test_retrieval_product_acceptance.py",
            "tests/feature_engine/test_feature_closure_genesis.py",
            "tests/feature_engine/test_panel_runtime_authority.py",
            "tests/portfolio_strategy_lab/test_frozen_strategy_packages.py",
            "tests/portfolio_strategy_lab/test_native_answer_route_class.py",
            "tests/portfolio_strategy_lab/test_strategy_scoring.py",
            "tests/portfolio_strategy_lab/test_workbench_readback.py",
            "tests/portfolio_strategy_lab/test_workbench_zh_catalog.py",
            "tests/portfolio_strategy_lab/workbench_stack_census.cjs",
            "tests/portfolio_strategy_lab/test_workbench_goal_follow_browser.py",
            "tests/portfolio_strategy_lab/test_workbench_owner_words.py",
            "tests/portfolio_strategy_lab/test_team_scene_producers.py",
            "tests/portfolio_strategy_lab/test_workbench_goal_request_words.py",
            "tests/portfolio_strategy_lab/test_workbench_usage_reading.py",
            "tests/portfolio_strategy_lab/test_ui_home_forward.py",
            "tests/portfolio_strategy_lab/test_ui_qa_conditions.py",
            "tests/portfolio_strategy_lab/test_ui_qa_launch_session.py",
            "tests/portfolio_strategy_lab/test_ui_goal_composer.py",
            "tests/portfolio_strategy_lab/test_workbench_portfolio_provenance.py",
            "tests/portfolio_strategy_lab/test_workbench_recovery_stack.py",
            "tests/portfolio_strategy_lab/test_workbench_home_recent.py",
            "tests/portfolio_strategy_lab/test_workbench_hover_witness.py",
            "tests/portfolio_strategy_lab/test_workbench_pointer_and_copy.py",
            "tests/portfolio_strategy_lab/test_workbench_review_publication.py",
            "tests/portfolio_strategy_lab/test_workbench_shared_readers.py",
            "tests/portfolio_strategy_lab/workbench_dom.cjs",
            "tests/portfolio_strategy_lab/workbench_handoffs.cjs",
            "tests/portfolio_strategy_lab/workbench_home_forward.cjs",
            "tests/portfolio_strategy_lab/workbench_home_recent.cjs",
            "tests/portfolio_strategy_lab/workbench_hover_witness.cjs",
            "tests/portfolio_strategy_lab/workbench_minor_ui.cjs",
            "tests/portfolio_strategy_lab/workbench_pointer_and_copy.cjs",
            "tests/portfolio_strategy_lab/workbench_portfolio_provenance.cjs",
            "tests/portfolio_strategy_lab/workbench_recovery_stack.cjs",
            "tests/portfolio_strategy_lab/workbench_review_publication.cjs",
            "tests/portfolio_strategy_lab/workbench_shared_readers.cjs",
            "tests/structural/test_public_release_closure.py",
            "tests/structural/test_workbench_agent_parity.py",
            "tests/workspace_maintenance/test_workspace_backup.py",
            "tests/researcher_methodology_surface/panel_value_parity_probe.py",
            "tests/researcher_methodology_surface/test_dogfood_authority_probe.py",
            "tests/researcher_methodology_surface/test_qa_retention.py",
            "tests/structural/test_source_candidate.py",
            "tests/structural/test_structural_guards.py",
            "tests/structural/test_playpen_gate_routing.py",
            "tests/alpha_research/test_dynamic_panel_portfolio_campaign.py",
            "tests/alpha_research/test_verification_impact.py",
            "tests/portfolio_strategy_lab/cli_golden_capture.py",
            "tests/portfolio_strategy_lab/cli-golden-transcripts.json",
            "tests/portfolio_strategy_lab/test_cli_golden_transcripts.py",
        ],
        "reason": (
            "This test or helper reads private QA records, internal tools or development history; "
            "the private checkout retains its coverage."
        ),
        "path_reasons": {
            "tests/alternative_evidence_desk/test_evidence_review_route.py": (
                "runs the private Review publication harness for exact admitted Task subjects"
            ),
            "tests/portfolio_strategy_lab/test_team_scene_producers.py": (
                "runs the private UI QA Team scene producers"
            ),
            "tests/portfolio_strategy_lab/test_workbench_goal_request_words.py": (
                "uses the private UI QA handoffs harness for Goal request words"
            ),
            "tests/portfolio_strategy_lab/test_workbench_usage_reading.py": (
                "uses the private UI QA handoffs harness for Settings usage reading"
            ),
            "tests/portfolio_strategy_lab/test_native_answer_route_class.py": (
                "needs the private native instruction source validator"
            ),
            "tests/portfolio_strategy_lab/test_workbench_readback.py": (
                "needs the private UI QA kit"
            ),
            "tests/portfolio_strategy_lab/test_workbench_zh_catalog.py": (
                "needs the private UI QA catalog census"
            ),
            "tests/portfolio_strategy_lab/workbench_stack_census.cjs": (
                "plants geometry cases for the private stack census"
            ),
            "tests/portfolio_strategy_lab/workbench_dom.cjs": "needs the private UI QA kit",
            "tests/portfolio_strategy_lab/test_workbench_goal_follow_browser.py": (
                "needs the private UI QA kit"
            ),
            "tests/portfolio_strategy_lab/workbench_handoffs.cjs": "needs the private UI QA kit",
            "tests/portfolio_strategy_lab/workbench_home_forward.cjs": (
                "measures Home density with the private UI QA page census"
            ),
            "tests/portfolio_strategy_lab/workbench_home_recent.cjs": (
                "reads the private UI QA pinned-browser runtime for native disclosure"
            ),
            "tests/portfolio_strategy_lab/workbench_hover_witness.cjs": (
                "requires the private UI QA painted-ink and hover witness checks"
            ),
            "tests/portfolio_strategy_lab/workbench_minor_ui.cjs": "needs the private UI QA kit",
            "tests/portfolio_strategy_lab/workbench_pointer_and_copy.cjs": (
                "requires the private UI QA hover, pointer and copy checks"
            ),
            "tests/portfolio_strategy_lab/workbench_review_publication.cjs": (
                "measures published Review and Report with the private UI QA stack census"
            ),
            "tests/portfolio_strategy_lab/workbench_shared_readers.cjs": (
                "measures shared readback controls with the private UI QA stack census"
            ),
            "tests/portfolio_strategy_lab/test_workbench_owner_words.py": (
                "needs the private UI QA kit"
            ),
            "tests/portfolio_strategy_lab/test_ui_home_forward.py": (
                "runs the private Home density harness and UI QA page census"
            ),
            "tests/portfolio_strategy_lab/test_ui_qa_conditions.py": (
                "loads private UI QA checks from embedded Node.js programs"
            ),
            "tests/portfolio_strategy_lab/test_ui_qa_launch_session.py": (
                "imports the private UI QA launch-session owner and executes its private drivers"
            ),
            "tests/portfolio_strategy_lab/test_ui_goal_composer.py": (
                "requires the private UI QA browser and kit from its embedded Node.js program"
            ),
            "tests/portfolio_strategy_lab/test_workbench_portfolio_provenance.py": (
                "runs the private Portfolio provenance harness and UI QA page census"
            ),
            "tests/portfolio_strategy_lab/test_workbench_recovery_stack.py": (
                "runs the private recovery-stack harness and UI QA stack census"
            ),
            "tests/portfolio_strategy_lab/workbench_portfolio_provenance.cjs": (
                "reads the private UI QA page census and pinned-browser runtime"
            ),
            "tests/portfolio_strategy_lab/workbench_recovery_stack.cjs": (
                "requires the private UI QA stack census and pinned-browser runtime"
            ),
            "tests/portfolio_strategy_lab/test_workbench_home_recent.py": (
                "runs the private Home disclosure harness and pinned-browser runtime"
            ),
            "tests/portfolio_strategy_lab/test_workbench_hover_witness.py": (
                "runs the private UI QA painted-ink and hover witness harness"
            ),
            "tests/portfolio_strategy_lab/test_workbench_pointer_and_copy.py": (
                "runs the private UI QA pointer and copy callback harness"
            ),
            "tests/portfolio_strategy_lab/test_workbench_review_publication.py": (
                "runs the private published Review and Report stack harness"
            ),
            "tests/portfolio_strategy_lab/test_workbench_shared_readers.py": (
                "runs the private shared-reader controls and stack harness"
            ),
            "tests/structural/test_workbench_agent_parity.py": "needs the private UI QA kit",
            "tests/structural/test_public_release_closure.py": (
                "needs the private release inventory"
            ),
        },
    },
    {
        "rule": "PUBLIC_LEGAL",
        "kind": "PUBLIC",
        "patterns": ["LICENSE", "NOTICE"],
        "reason": "The user's license grant and applicable notices accompany redistributed code.",
    },
    {
        "rule": "PUBLIC_INTEGRATION",
        "kind": "PUBLIC",
        "patterns": [
            "AGENTS.md",
            "CLAUDE.md",
            ".agents/skills/**",
            ".codex/agents/**",
            ".codex/config.toml",
            ".claude/agents/**",
            ".claude/skills/**",
            ".claude/settings.json",
            "skills/**",
        ],
        "reason": (
            "Installed research Skills, role cards and native hooks are inputs to the user's agent."
        ),
    },
    {
        "rule": "PUBLIC_DOC",
        "kind": "PUBLIC",
        "patterns": [
            "README.md",
            "CONTRIBUTING.md",
            "CLA.md",
            "GOVERNANCE.md",
            "SECURITY.md",
            ".github/CODEOWNERS",
            ".github/ISSUE_TEMPLATE/bug_report.yml",
            ".github/ISSUE_TEMPLATE/feature_request.yml",
            ".github/PULL_REQUEST_TEMPLATE.md",
            "docs/public-source/**",
            "docs/reference/**",
            "docs/index.md",
            "docs/generate_reference.py",
            "docs/reference_hooks.py",
            "docs/verify_reference.py",
            "mkdocs.yml",
        ],
        "reason": (
            "Public product guidance, generated API reference and its "
            "builders explain the shipped product."
        ),
    },
    {
        "rule": "PUBLIC_BUILD",
        "kind": "PUBLIC",
        "patterns": [
            ".gitattributes",
            ".gitignore",
            ".python-version",
            ".pre-commit-config.yaml",
            ".githooks/**",
            ".github/workflows/ci.yml",
            "pyproject.toml",
            "uv.lock",
            "LAWS.md",
            "scripts/release/**",
        ],
        "reason": (
            "Dependency declarations, source rules, build hygiene and "
            "release tooling make the source checkout reproducible."
        ),
    },
    {
        "rule": "PUBLIC_BROWSER",
        "kind": "PUBLIC",
        "patterns": ["third_party/playwright/**"],
        "reason": (
            "The maintained browser dependency is used for Local Web "
            "operation and its verification."
        ),
    },
    {
        "rule": "PUBLIC_TEST",
        "kind": "PUBLIC",
        "patterns": ["tests/**"],
        "reason": (
            "Tracked test code and synthetic fixtures hold product behavior; "
            "scan findings still require review before redistribution."
        ),
    },
    {
        "rule": "PUBLIC_ASSET",
        "kind": "PUBLIC",
        "patterns": [
            "src/alphalattice/**",
            "docs/images/workbench-light.webp",
            "docs/images/workbench-dark.webp",
            "docs/images/goal-conversation.webp",
        ],
        "reason": (
            "Non-Python product resources, Local Web source/build assets and "
            "public Workbench illustrations support operation and product guidance."
        ),
    },
    {
        "rule": "PUBLIC_RUNTIME",
        "kind": "PUBLIC",
        "patterns": ["src/alphalattice/*.py", "src/alphalattice/**/*.py"],
        "reason": (
            "ER3 LIVE modules and their current parsed import closure implement product entries."
        ),
    },
    {
        "rule": "PUBLIC_TEST_DEPENDENCY",
        "kind": "PUBLIC",
        "patterns": ["src/alphalattice/*.py", "src/alphalattice/**/*.py"],
        "reason": (
            "Public tests import this product source directly or through its parsed import closure."
        ),
    },
    {
        "rule": "PUBLIC_VERIFICATION",
        "kind": "PUBLIC",
        "patterns": [
            "src/devtools/**",
            "scripts/check_*.py",
            "scripts/run_playpen_precommit.py",
            "scripts/configure_playpen_dev.py",
            "scripts/count_ui_laws.py",
            "scripts/identity_readout.py",
        ],
        "reason": (
            "Source verification and the maintained local gate hold the "
            "public product and test contracts."
        ),
    },
    {
        "rule": "PUBLIC_USER_SCRIPT",
        "kind": "PUBLIC",
        "patterns": ["scripts/*.py", "scripts/playwright.ps1"],
        "reason": (
            "This explicit maintained command launches, configures, builds "
            "or operates a user-selected product workspace."
        ),
    },
    {
        "rule": "PUBLIC_CONFIG",
        "kind": "PUBLIC",
        "patterns": ["config/**"],
        "reason": (
            "This configuration is referenced by PUBLIC source, tests, "
            "integrations or their environment setup."
        ),
    },
]
USER_SCRIPTS = frozenset(
    {
        "bind_local_web_data_workspace.py",
        "bind_local_web_research_inputs.py",
        "build_local_web_ui.py",
        "create_gpu_environment.py",
        "create_retrieval_environment.py",
        "install_retrieval_pack.py",
        "materialize_claude_host.py",
        "materialize_evidence_cro_authority.py",
        "model_sandbox.py",
        "native_research.py",
        "publish_simple_signed_score.py",
        "run_alphalattice.py",
        "run_canonical_alpha_development.py",
        "run_current_universe_onboarding.py",
        "run_factor_research_pipeline.py",
        "run_feature_observation_clock_panel.py",
        "run_full_universe_whitelist.py",
        "run_local_portfolio_web.py",
        "run_monthly_alpha_refit_research.py",
        "run_public_portfolio_research.py",
        "run_research_experiment.py",
        "run_sector_research_development.py",
        "u0_probe.py",
        "playwright.ps1",
    }
)
STRICT_CANDIDATE_SCRIPTS = frozenset(
    {
        "scripts/bind_portfolio_decision_checkpoint.py",
        "scripts/broad_ensemble_research_closure.py",
        "scripts/materialize_post_governance_strategy_authority.py",
        "scripts/playpen_review_context.py",
        "scripts/probe_panel_identity.py",
        "scripts/run_alpha_model_adapter_parity.py",
        "scripts/run_heterogeneous_live_score_closure.py",
        "scripts/run_broad_ensemble_live_model_closure.py",
    }
)
next(r for r in RULES if r["rule"] == "PRIVATE_DEVELOPMENT_TOOL")["patterns"].extend(
    sorted(STRICT_CANDIDATE_SCRIPTS)
)
PRIVATE_PREFIXES = (
    "product-design/",
    "implemented-plans/",
    "experience/",
    "census/",
    "scripts/agent_eval/",
    "scripts/ui_qa/",
    "config/agent-eval/",
    "workspaces/data-platform-qa/",
    "playpen-board/",
)
SCAN_RULES = {
    "NO_SECRET": "A PUBLIC file must not contain a credential, access token or private key.",
    "NO_PRIVATE_ABSOLUTE_PATH": (
        "A PUBLIC file must not disclose a maintainer's private machine path."
    ),
    "NO_INTERNAL_ID": (
        "A PUBLIC file must not disclose an actual internal thread or session identifier."
    ),
    "NO_PERSONAL_CONTACT": "A PUBLIC file must not disclose an unapproved personal contact detail.",
    "NO_PRIVATE_REFERENCE": (
        "A PUBLIC file must not require or link an excluded internal file or QA tree."
    ),
    "NO_DEVELOPMENT_ID": (
        "A PUBLIC file's internal development labels must not exceed its frozen count."
    ),
}
SCANS = {
    "NO_SECRET": re.compile(
        "-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE "
        "KEY-----|\\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0"
        "-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\\b|(?i:(?"
        ":api[_-]?key|access[_-]?token|client[_-]?secret|password)\\s*[:=]"
        "\\s*[\\\"'][^\\\"'\\s]{12,}[\\\"'])"
    ),
    "NO_PRIVATE_ABSOLUTE_PATH": re.compile(
        r"(?i)\b[A-Z]:[/\\]+(?:AI Agent|Users)[/\\]+|/Users/[^/\s]+/|/home/[^/\s]+/"
    ),
    "NO_INTERNAL_ID": re.compile(
        r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
    ),
    "NO_PERSONAL_CONTACT": re.compile(
        "(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b|(?:phone|telephone"
        "|contact)\\s*[:=]\\s*[\\\"']?\\+?\\d[\\d ()-]{8,}\\d"
    ),
}


def internal_id_pattern(policy: dict | None = None) -> re.Pattern:
    """Match numbered labels and internal card or line forms, never ordinary prose."""
    policy_values = policy or {}
    lines = set(policy_values.get("lines", ()))
    cards = set(policy_values.get("cards", ())) - lines - {"UI", "LEAD", "IS"}

    def vocabulary(tokens):
        return "|".join(re.escape(token) for token in sorted(tokens, key=lambda t: (-len(t), t)))

    alternatives = [r"[VU][0-9]+", r"(?i:(?:round|law)\s+[0-9]+)"]
    if lines:
        alternatives.append(r"(?:Codex|Claude)\s+(?:" + vocabulary(lines) + ")")
    distinct = {token for token in cards | lines if re.search(r"[0-9-]", token)}
    if distinct:
        alternatives.append(vocabulary(distinct))
    words = cards - distinct
    if words:
        alternatives.extend(
            [r"(?i:card)\s+(?:" + vocabulary(words) + ")", "(?:" + vocabulary(words) + r")(?=\s*:)"]
        )
    return re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9_-])")


def internal_id_label(value: str) -> str:
    """Normalize a numeric prose label; exact card and line spellings stay distinct."""
    if re.fullmatch(r"(?i:(?:round|law)\s+[0-9]+)", value):
        return " ".join(value.casefold().split())
    return re.sub(r"^(?:Codex|Claude|(?i:card))\s+", "", value)


def internal_id_ratchet(
    blobs: dict[str, bytes], decisions: list[dict], policy: dict | None = None
) -> dict:
    """Refuse public development labels beyond their per-file, per-label frozen allowance."""
    policy_values = policy or {}
    pattern = internal_id_pattern(policy)
    occurrences = policy_values.get("occurrences", {})
    limits = [n for row in occurrences.values() for n in row.values()]
    if any(type(count) is not int or count < 0 for count in limits):
        raise ValueError("internal development label counts must be nonnegative integers")
    current = {}
    current_occurrences = {}
    allowed = 0
    violations = []
    for row in decisions:
        if row["kind"] != "PUBLIC":
            continue
        path = row["path"]
        try:
            source = blobs[path].decode("utf-8-sig")
        except UnicodeError:
            continue
        matches = [
            (number, match.start() + 1, internal_id_label(match.group()))
            for number, line in enumerate(source.splitlines(), 1)
            for match in pattern.finditer(line)
        ]
        if matches:
            current[path] = len(matches)
        seen = Counter()
        for line, column, label in matches:
            seen[label] += 1
            if seen[label] <= occurrences.get(path, {}).get(label, 0):
                allowed += 1
                continue
            violations.append(
                {
                    "kind": "FINDING",
                    "path": path,
                    "line": line,
                    "column": column,
                    "rule": "NO_DEVELOPMENT_ID",
                    "assessment": "FROZEN_COUNT_EXCEEDED",
                }
            )
        if seen:
            current_occurrences[path] = dict(sorted(seen.items()))
    return {
        "policy_present": policy is not None,
        "count": sum(current.values()),
        "allowed": allowed,
        "current": dict(sorted(current.items())),
        "current_occurrences": dict(sorted(current_occurrences.items())),
        "violations": sorted(violations, key=lambda row: (row["path"], row["line"], row["column"])),
    }


def validate_internal_id_history(
    repo: Path, source: str, admissions_from: str | None = None
) -> None:
    """Validate the first landing against its source, then keep the policy lower-only."""
    if source.startswith("-"):
        raise ValueError("a revision cannot be an option")
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{source}^{{commit}}"], cwd=repo, text=True
    ).strip()
    cache = {}
    anchor_counts = {}

    def policy_at(commit: str) -> dict | None:
        if commit not in cache:
            result = subprocess.run(
                ["git", "-c", "core.longpaths=true", "show", f"{commit}:{INTERNAL_ID_BASELINE}"],
                cwd=repo,
                capture_output=True,
                check=False,
            )
            cache[commit] = json.loads(result.stdout) if result.returncode == 0 else None
        return cache[commit]

    source_policy = policy_at(source)
    if source_policy is None:
        raise ValueError("the source has no frozen internal-label policy")
    # a later reviewed policy (develop's) may admit an earlier commit's exact rows by name; a
    # commit's own policy never widens its own history beyond what it names
    admitting = source_policy
    if admissions_from is not None:
        later = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{admissions_from}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        admitting = (policy_at(later) if later else None) or source_policy
    admissions = admitting.get("history_admissions", {})
    commits = subprocess.check_output(
        ["git", "log", "--full-history", "--format=%H", source, "--", INTERNAL_ID_BASELINE],
        cwd=repo,
        text=True,
    ).splitlines()
    for commit in commits:
        policy = policy_at(commit)
        if policy is None:
            raise ValueError(f"the frozen internal-label policy was removed at {commit}")
        if {"cards", "lines", "occurrences"} - policy.keys():
            raise ValueError(f"the frozen internal-label policy is incomplete at {commit}")
        version = policy.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"the frozen internal-label format is unknown at {commit}")
        anchor = policy.get("source_sha", "")
        if version == 2 and (
            not re.fullmatch(r"[0-9a-f]{40}", anchor)
            or subprocess.run(
                ["git", "merge-base", "--is-ancestor", anchor, commit],
                cwd=repo,
                capture_output=True,
                check=False,
            ).returncode
            or policy_at(anchor) is not None
        ):
            raise ValueError(f"the first-landing anchor must precede the policy at {commit}")
        parents = subprocess.check_output(
            ["git", "rev-list", "-1", "--parents", commit], cwd=repo, text=True
        ).split()[1:]
        for parent in parents:
            previous = policy_at(parent)
            if previous is None:
                if version == 1:
                    continue  # The legacy policy established its draft allowance.
                previous = {**policy, "schema_version": 1, "occurrences": {}}
            if previous.get("schema_version", 1) == 2 and (
                version != 2 or anchor != previous["source_sha"]
            ):
                raise ValueError(f"the frozen internal-label anchor or format changed at {commit}")
            if (
                set(policy["cards"]) - set(previous["cards"])
                or set(previous["cards"]) - set(policy["cards"]) - {"UI", "IS", "LEAD"}
                or policy["lines"] != previous["lines"]
            ):
                raise ValueError(f"the frozen internal-label vocabulary changed at {commit}")
            for owner, labels in policy["occurrences"].items():
                for label, count in labels.items():
                    if count <= previous["occurrences"].get(owner, {}).get(label, 0):
                        continue
                    if version == 2 and previous.get("schema_version", 1) == 1:
                        key = (anchor, owner)
                        if key not in anchor_counts:
                            validate_path(owner)
                            raw = subprocess.run(
                                ["git", "-c", "core.longpaths=true", "show", f"{anchor}:{owner}"],
                                cwd=repo,
                                capture_output=True,
                                check=False,
                            )
                            try:
                                text = raw.stdout.decode("utf-8-sig") if raw.returncode == 0 else ""
                            except UnicodeError:
                                text = ""
                            pattern = internal_id_pattern(policy)
                            anchor_counts[key] = Counter(
                                internal_id_label(match.group())
                                for line in text.splitlines()
                                for match in pattern.finditer(line)
                            )
                        if count <= anchor_counts[key][label]:
                            continue
                    admitted = admissions.get(commit, {})
                    if parent == admitted.get("parent") and count == admitted.get(
                        "occurrences", {}
                    ).get(owner, {}).get(label):
                        continue
                    raise ValueError(f"the frozen internal-label allowance increased at {commit}")


def validate_path(path: str) -> None:
    """Reject paths that a file manifest cannot safely describe."""
    parsed = PurePosixPath(path)
    if not path or "\\" in path or parsed.is_absolute() or ".." in parsed.parts or ":" in path:
        raise ValueError(f"unsafe manifest path: {path!r}")


def read_tree(repo: Path, revision: str) -> tuple[str, dict[str, bytes], dict[str, str]]:
    """Read a commit's exact tracked blobs with one batch, without checking them out."""
    if revision.startswith("-"):
        raise ValueError("a revision cannot be an option")
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=repo, text=True
    ).strip()
    listing = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", sha], cwd=repo)
    objects = []
    modes = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        info, name = record.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        path = name.decode("utf-8")
        validate_path(path)
        modes[path] = mode
        if kind == "blob":
            objects.append((path, oid))
    raw = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input="".join(oid + "\n" for _, oid in objects).encode(),
    )
    blobs = {}
    cursor = 0
    for path, oid in objects:
        end = raw.index(b"\n", cursor)
        header = raw[cursor:end].decode().split()
        if header[:2] != [oid, "blob"]:
            raise ValueError("unexpected git object response")
        size = int(header[2])
        blobs[path] = raw[end + 1 : end + 1 + size]
        cursor = end + 2 + size
    return sha, blobs, modes


def import_closure(blobs: dict[str, bytes], roots: set[str]) -> set[str]:
    """Follow absolute and relative Python imports by parsing, never importing product code."""
    modules = {
        p[4:-3].replace("/", ".").removesuffix(".__init__"): p
        for p in blobs
        if p.startswith("src/") and p.endswith(".py")
    }
    edges = {}
    for path, raw in blobs.items():
        if not path.endswith(".py"):
            continue
        tree = ast.parse(raw.decode("utf-8-sig"), filename=path)
        own = (
            path[4:-3].replace("/", ".").removesuffix(".__init__")
            if path.startswith("src/")
            else ""
        )
        package = own if path.endswith("/__init__.py") else own.rpartition(".")[0]
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                names.add(base)
                names.update(base + "." + alias.name for alias in node.names)
        edges[path] = {
            modules[n] for name in names for n in modules if n == name or name.startswith(n + ".")
        }
    reached = set(roots)
    todo = list(roots)
    while todo:
        for target in edges.get(todo.pop(), set()) - reached:
            reached.add(target)
            todo.append(target)
    return reached


def classify(
    path: str,
    runtime: set[str],
    test_dependencies: set[str],
    config_dependencies: set[str],
    strict_scripts: set[str] | None = None,
) -> tuple[str, str | None, str]:
    """The first applicable named rule wins; absence of evidence stays unsettled."""
    validate_path(path)
    for rule in RULES:
        name = rule["rule"]
        if not any(fnmatch.fnmatchcase(path, p) for p in rule["patterns"]):
            continue
        if name == "PRIVATE_DEVELOPMENT_TOOL" and path in STRICT_CANDIDATE_SCRIPTS:
            continue  # generate settles these from PUBLIC references and their scans.
        if name == "PUBLIC_ASSET" and path.endswith(".py"):
            continue
        if name == "PUBLIC_RUNTIME" and path not in runtime:
            continue
        if name == "PUBLIC_TEST_DEPENDENCY" and path not in test_dependencies:
            continue
        if (
            name == "PUBLIC_USER_SCRIPT"
            and path.rsplit("/", 1)[-1] not in USER_SCRIPTS
            and path not in (strict_scripts or set())
        ):
            continue
        if name == "PUBLIC_CONFIG" and path not in config_dependencies:
            continue
        reason = rule.get("path_reasons", {}).get(path, rule["reason"])
        return rule["kind"], name, reason
    return (
        "UNSETTLED",
        None,
        (
            "No named rule establishes a product entry, public "
            "verification/document consumer, or private-state purpose; "
            "Claude must decide this file's scope."
        ),
    )


def scan_public(blobs: dict[str, bytes], decisions: list[dict]) -> tuple[list[dict], list[str]]:
    """Return location-only findings; never retain matched secrets, IDs or contact values."""
    private = {r["path"] for r in decisions if r["kind"] == "PRIVATE"}
    private_modules = {}
    for path in sorted(private):
        if not path.endswith(".py"):
            continue
        for relative in {path, path.removeprefix("src/"), path.removeprefix("scripts/")}:
            module = relative[:-3].replace("/", ".").removesuffix(".__init__")
            private_modules.setdefault(module, set()).add(path)
    findings = []
    binary = []
    for row in decisions:
        if row["kind"] != "PUBLIC":
            continue
        raw = blobs[row["path"]]
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            binary.append(row["path"])
            continue
        if row["path"].endswith(".py"):
            try:
                tree = ast.parse(text, filename=row["path"])
            except SyntaxError:
                tree = None  # Text-pattern scans still apply to an invalid/synthetic source.
            own = row["path"].removeprefix("src/")[:-3].replace("/", ".")
            package = (
                own.removesuffix(".__init__")
                if own.endswith(".__init__")
                else own.rpartition(".")[0]
            )
            imports = {}
            for node in ast.walk(tree) if tree is not None else ():
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parts = package.split(".")
                        base = ".".join(
                            parts[: len(parts) - node.level + 1] + ([base] if base else [])
                        )
                    names = [base, *(base + "." + alias.name for alias in node.names)]
                else:
                    continue
                targets = {p for name in names for p in private_modules.get(name, ())}
                if targets:
                    imports.setdefault(node.lineno, set()).update(targets)
            findings.extend(
                {
                    "kind": "FINDING",
                    "path": row["path"],
                    "line": line,
                    "rule": "NO_PRIVATE_REFERENCE",
                    "assessment": "PRIVATE_PYTHON_IMPORT",
                    "private_targets": sorted(targets),
                }
                for line, targets in sorted(imports.items())
            )
        for number, line in enumerate(text.splitlines(), 1):
            for rule, pattern in SCANS.items():
                for match in pattern.finditer(line):
                    assessment = "REVIEW_REQUIRED"
                    if rule == "NO_INTERNAL_ID" and not re.search(
                        r"thread|session|codex|claude|agent", line, re.I
                    ):
                        assessment = "UUID_PURPOSE_UNSETTLED"
                    if row["path"].startswith("tests/"):
                        assessment = "TEST_LITERAL_REVIEW"
                    findings.append(
                        {
                            "kind": "FINDING",
                            "path": row["path"],
                            "line": number,
                            "column": match.start() + 1,
                            "rule": rule,
                            "assessment": assessment,
                        }
                    )
            normalized = line.replace("\\\\", "/").replace("\\", "/")
            refs = set()
            for token in re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+", normalized):
                direct = token[2:] if token.startswith("./") else token
                relative = posixpath.normpath(posixpath.join(posixpath.dirname(row["path"]), token))
                if relative in private or (relative not in blobs and direct in private):
                    refs.add("TRACKED_PRIVATE_FILE")
            if any(prefix in normalized for prefix in PRIVATE_PREFIXES):
                refs.add("INTERNAL_TREE_REFERENCE")
            for ref in sorted(refs):
                findings.append(
                    {
                        "kind": "FINDING",
                        "path": row["path"],
                        "line": number,
                        "rule": "NO_PRIVATE_REFERENCE",
                        "assessment": ref,
                    }
                )
    return sorted(
        findings, key=lambda r: (r["path"], r["line"], r["rule"], r.get("column", 0))
    ), binary


def names_script(reader: str, raw: bytes, script: str) -> bool:
    """Recognize a named script, including a path composed with pathlib division."""
    text = raw.decode("utf-8", errors="replace")
    if script in text:
        return True
    if not reader.endswith(".py") or PurePosixPath(script).name not in text:
        return False

    def parts(node: ast.AST) -> list[str]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return parts(node.left) + parts(node.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        return []

    return any(
        "/".join(parts(node)).endswith(script)
        for node in ast.walk(ast.parse(text, filename=reader))
        if isinstance(node, ast.BinOp)
    )


def generate(sha: str, blobs: dict[str, bytes], modes: dict[str, str]) -> tuple[dict, dict]:
    """Produce the builder contract and complete private audit deterministically."""
    census_raw = blobs.get("census/entry_reach.json")
    if census_raw is None:
        raise ValueError("the source commit must contain the entry reachability census")
    er3 = json.loads(census_raw)
    runtime = import_closure(
        blobs, {r["path"] for r in er3["modules"] if r["class"] == "LIVE" and r["path"] in blobs}
    )
    tests = import_closure(
        blobs,
        {
            p
            for p in blobs
            if p.startswith("tests/")
            and p.endswith(".py")
            and classify(p, set(), set(), set())[0] == "PUBLIC"
        },
    )
    initial = [dict(path=p, kind=classify(p, runtime, tests, set())[0]) for p in sorted(blobs)]
    config_dependencies = set()
    for row in initial:
        if row["kind"] != "PUBLIC":
            continue
        text = blobs[row["path"]].decode("utf-8", errors="replace")
        for path in blobs:
            if path.startswith("config/") and (path in text or path.rsplit("/", 1)[-1] in text):
                config_dependencies.add(path)
    # These files are named by environment setup and by the maintained gate.
    config_dependencies.update(
        p
        for p in blobs
        if p.startswith("config/requirements-") or p.startswith("config/registries/")
    )
    initial = [
        dict(path=p, kind=classify(p, runtime, tests, config_dependencies)[0])
        for p in sorted(blobs)
    ]
    strict_scripts = set()
    private_script_reasons = {}
    for path in sorted(STRICT_CANDIDATE_SCRIPTS & blobs.keys()):
        readers = [
            r["path"]
            for r in initial
            if r["kind"] == "PUBLIC"
            and r["path"]
            not in {
                "scripts/release/public_manifest.py",
                "tests/structural/test_public_manifest.py",
            }
            and names_script(r["path"], blobs[r["path"]], path)
        ]
        if not readers:
            private_script_reasons[path] = "OW8: no PUBLIC file names this development script."
            continue
        hits, _ = scan_public(
            blobs,
            [{"path": path, "kind": "PUBLIC"}, *[r for r in initial if r["kind"] == "PRIVATE"]],
        )
        if any(
            h["rule"] in {"NO_PRIVATE_ABSOLUTE_PATH", "NO_INTERNAL_ID", "NO_PRIVATE_REFERENCE"}
            for h in hits
        ):
            private_script_reasons[path] = (
                "OW8: its scan contains a private path or identifier; PUBLIC references "
                "must be resolved at their owners."
            )
        else:
            strict_scripts.add(path)
    decisions = []
    for path in sorted(modes):
        if modes[path] not in {"100644", "100755"}:
            kind, rule, reason = (
                "UNSETTLED",
                None,
                "A symlink or gitlink needs an explicit redistribution/safety decision.",
            )
        else:
            kind, rule, reason = classify(path, runtime, tests, config_dependencies, strict_scripts)
            if path in private_script_reasons:
                kind, rule, reason = (
                    "PRIVATE",
                    "PRIVATE_DEVELOPMENT_TOOL",
                    private_script_reasons[path],
                )
        decisions.append({"path": path, "kind": kind, "rule": rule, "reason": reason})
    er3_by_path = {r["path"]: r for r in er3["modules"]}
    public_paths = [r["path"] for r in decisions if r["kind"] == "PUBLIC"]
    for row in decisions:
        if row["kind"] != "UNSETTLED":
            continue
        path = row["path"]
        refs = []
        for reader in public_paths:
            text = blobs[reader].decode("utf-8", errors="replace")
            refs.extend(
                f"{reader}:{line}"
                for line, value in enumerate(text.splitlines(), 1)
                if path in value
            )
        prior = er3_by_path.get(path)
        suffix = PurePosixPath(path).suffix or "extensionless"
        size = len(blobs.get(path, b""))
        facts = [f"Tracked {suffix} file; {size} bytes."]
        if prior:
            facts.append(
                f"ER3 class {prior['class']}; runtime and public test imports do not establish use."
            )
        else:
            facts.append("ER3 supplies no module-level product reach decision for this file.")
        facts.append(
            f"PUBLIC literal path references: {len(refs)}; "
            + (", ".join(refs[:4]) or "none found")
            + "."
        )
        row["evidence"] = " ".join(facts) + " " + row["reason"]
        row["public_reference_sites"] = refs
    public = [
        {
            "path": r["path"],
            "sha256": hashlib.sha256(blobs[r["path"]]).hexdigest(),
            "rule": r["rule"],
        }
        for r in decisions
        if r["kind"] == "PUBLIC"
    ]
    unsettled = [
        {"path": r["path"], "evidence": r["evidence"]}
        for r in decisions
        if r["kind"] == "UNSETTLED"
    ]
    manifest = {
        "schema": "alphalattice.release.public-manifest",
        "schema_version": 1,
        "source_sha": sha,
        "rules": RULES,
        "public": public,
        "private_count": sum(r["kind"] == "PRIVATE" for r in decisions),
        "unsettled": unsettled,
    }
    policy_raw = blobs.get(INTERNAL_ID_BASELINE)
    policy = json.loads(policy_raw) if policy_raw is not None else None
    findings, binary = scan_public(blobs, decisions)
    ratchet = internal_id_ratchet(blobs, decisions, policy)
    findings = sorted(
        [*findings, *ratchet["violations"]],
        key=lambda row: (row["path"], row["line"], row["rule"], row.get("column", 0)),
    )
    audit = {
        "source_sha": sha,
        "er3_source_sha": er3.get("base"),
        "command": f"python scripts/release/public_manifest.py --source {sha}",
        "tracked_count": len(decisions),
        "counts": dict(sorted(Counter(r["kind"] for r in decisions).items())),
        "by_rule": dict(sorted(Counter(r["rule"] for r in decisions if r["rule"]).items())),
        "files": decisions,
        "scan_rules": SCAN_RULES,
        "findings": findings,
        "finding_counts": dict(sorted(Counter(r["rule"] for r in findings).items())),
        "internal_id_ratchet": ratchet,
        "binary_public_files": binary,
        "limits": [
            (
                "Findings locate candidate disclosures without reproducing their "
                "values; tests and licensed copyright contacts need human "
                "assessment."
            ),
            (
                "Only committed files are scanned. Untracked state, Git history "
                "and external dependencies are not public manifest inputs."
            ),
            (
                "Names cannot be reliably inferred as personal data by syntax; "
                "the public author's expressly approved attribution is not a "
                "privacy finding."
            ),
            (
                "Non-UTF-8 fonts/assets are listed for license and metadata "
                "review, not reported as text-scan clean."
            ),
            (
                "ER3 is the recorded baseline; current static imports extend it, "
                "while unresolved scope remains unsettled."
            ),
        ],
    }
    return manifest, audit


def render(audit: dict) -> str:
    """The internal reading of decisions, unsettled files and every scan finding."""
    lines = [
        "# Public scope, disclosure assessment and standalone verification",
        "",
        f"Measured `{audit['source_sha']}`; ER3 measured `{audit['er3_source_sha']}`.",
        f"Rebuild: `{audit['command']}`. This command inventories; verification is recorded below.",
        "",
        f"Tracked files: {audit['tracked_count']}; classes: {audit['counts']}.",
        "",
        "## Named rules",
        "",
        "| Rule | Kind | Files | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {r['rule']} | {r['kind']} | {audit['by_rule'].get(r['rule'], 0)} | {r['reason']} |"
        for r in RULES
    )
    lines.extend(["", "## Unsettled scope for Claude", ""])
    lines.extend(
        f"- `{r['path']}`: {r['evidence']}" for r in audit["files"] if r["kind"] == "UNSETTLED"
    )
    lines.extend(
        [
            "",
            "## PUBLIC scan findings",
            "",
            f"Counts: {audit['finding_counts']}.",
            (
                "Each row is a FINDING for review, not a claim that a test "
                "literal is an actual secret. Values are withheld."
            ),
            "",
            "| File | Line | Broken rule | Assessment |",
            "| --- | ---: | --- | --- |",
        ]
    )
    lines.extend(
        f"| `{r['path']}` | {r['line']} | {r['rule']} | "
        f"{r.get('review', {}).get('evidence', r['assessment'])} |"
        for r in audit["findings"]
    )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            *("- " + s for s in audit["limits"]),
            "",
            "Binary PUBLIC files:",
            "",
            *(f"- `{p}`" for p in audit["binary_public_files"]),
            "",
            (
                "The JSON records every tracked file's rule and reason; PUBLIC "
                "hashes are only in the builder manifest."
            ),
            "",
        ]
    )
    verification = audit.get("standalone")
    if verification:
        lines.extend(["", "## RR4b standalone verification", ""])
        lines.extend(
            ["| Round | Source | Public | Private | Outcome |", "| --- | --- | ---: | ---: | --- |"]
        )
        for row in verification["rounds"]:
            lines.append(
                f"| {row['round']} | `{row['source_sha']}` | {row['public']} | "
                f"{row['private']} | {row['outcome']} |"
            )
        for row in verification["rounds"]:
            lines.extend(["", f"### Round {row['round']} checks", ""])
            for check in row.get("checks", []):
                lines.append(f"- {check['name']}: {check['result'].rstrip('.')}.")
        lines.extend(["", "### Corrections at the source owners", ""])
        for row in verification["corrections"]:
            lines.append(f"- `{row['path']}`: {row['reason']}")
        lines.extend(["", "### Original 501 scan rows", ""])
        lines.append(
            str(
                dict(
                    sorted(
                        Counter(
                            row["disposition"] for row in verification["original_findings"]
                        ).items()
                    )
                )
            )
        )
        lines.append(
            "\nEvery original location and its evidence-based disposition is retained in JSON; "
            "values are omitted."
        )
        lines.append(
            "\nRecorded demo identifiers and scientific UUID namespaces remain. Native-session "
            "fixtures now use explicit synthetic UUIDs. Example contacts and scanner regex "
            "literals are synthetic; the user's approved author contacts remain. Private QA, "
            "campaign and history inputs remain in the private checkout."
        )
        lines.extend(["", "### Open findings for Claude", ""])
        lines.extend("- " + row for row in verification.get("open_findings", []))
        lines.extend(
            [
                "",
                "### Private structural check",
                "",
                verification.get("private_structural", "Pending."),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    """Rebuild manifest/audit or compare them byte for byte without writing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="develop")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--verification-json",
        type=Path,
        help="Read the standalone evidence field from a retained private verification report.",
    )
    args = parser.parse_args()
    sha, blobs, modes = read_tree(args.repo, args.source)
    if INTERNAL_ID_BASELINE in blobs:
        validate_internal_id_history(args.repo, sha, admissions_from="develop")
    manifest, audit = generate(sha, blobs, modes)
    if args.verification_json:
        audit["standalone"] = json.loads(args.verification_json.read_text(encoding="utf-8"))[
            "standalone"
        ]
        audit["command"] += " --verification-json census/public_release.json"
        reviews = {
            (row["path"], row["line"], row["rule"], row.get("column", 0)): row
            for row in audit["standalone"].get("current_findings", [])
        }
        for row in audit["findings"]:
            key = (row["path"], row["line"], row["rule"], row.get("column", 0))
            if key in reviews:
                row["review"] = {name: reviews[key][name] for name in ("disposition", "evidence")}
    outputs = {
        "config/release/public-manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False)
        + "\n",
        "census/public_release.json": json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        "census/public_release.md": render(audit),
    }
    for name, content in outputs.items():
        destination = args.out / name
        if args.check:
            if not destination.exists() or destination.read_bytes() != content.encode("utf-8"):
                raise SystemExit(f"stale: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {"source_sha": sha, "counts": audit["counts"], "findings": audit["finding_counts"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
