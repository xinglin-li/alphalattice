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
import os
import posixpath
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from devtools.architecture.structural import FileFacts

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_ID_BASELINE = "census/public-internal-id-baseline.json"
REVIEW_KEY = "alphalattice.release.review-key.v1"
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
            "tests/feature_engine/test_panel_upgrade.py",
            "tests/feature_engine/panel_upgrade.py",
            "tests/portfolio_strategy_lab/test_frozen_strategy_packages.py",
            "tests/portfolio_strategy_lab/test_native_answer_route_class.py",
            "tests/portfolio_strategy_lab/test_strategy_scoring.py",
            "tests/portfolio_strategy_lab/test_workbench_readback.py",
            "tests/portfolio_strategy_lab/test_workbench_zh_catalog.py",
            "tests/portfolio_strategy_lab/workbench_stack_census.cjs",
            "tests/portfolio_strategy_lab/test_workbench_goal_follow_browser.py",
            "tests/portfolio_strategy_lab/test_workbench_owner_words.py",
            "tests/portfolio_strategy_lab/test_daily_book_records_gate.py",
            "tests/portfolio_strategy_lab/test_team_scene_producers.py",
            "tests/portfolio_strategy_lab/test_workbench_goal_request_words.py",
            "tests/portfolio_strategy_lab/test_workbench_usage_reading.py",
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
            "tests/structural/test_catalog_gate.py",
            "tests/alpha_research/test_dynamic_panel_portfolio_campaign.py",
            "tests/alpha_research/test_verification_impact.py",
            "tests/portfolio_strategy_lab/cli_golden_capture.py",
            "tests/portfolio_strategy_lab/cli-golden-transcripts.json",
            "tests/portfolio_strategy_lab/test_cli_golden_transcripts.py",
            "tests/e2e/evaluation/test_first_use_completion.py",
            "tests/integration/evaluation/test_as_of.py",
            "tests/integration/evaluation/test_assignment_coordinate_rebinding.py",
            "tests/integration/evaluation/test_call_profile_accounting.py",
            "tests/integration/evaluation/test_cli_invocation_count.py",
            "tests/integration/evaluation/test_cli_launch_accounting.py",
            "tests/integration/evaluation/test_commentary_final_separation.py",
            "tests/integration/evaluation/test_controller_rehearsal_admission.py",
            "tests/integration/evaluation/test_copy_bundle_ownership.py",
            "tests/integration/evaluation/test_data_case_coverage.py",
            "tests/integration/evaluation/test_delivered_finding_scoring.py",
            "tests/integration/evaluation/test_event_continuation.py",
            "tests/integration/evaluation/test_evidence_pair_usage.py",
            "tests/integration/evaluation/test_explorer_assets.py",
            "tests/integration/evaluation/test_factor_baseline_provenance.py",
            "tests/integration/evaluation/test_factor_trial_completion.py",
            "tests/integration/evaluation/test_file_read_phase_attribution.py",
            "tests/integration/evaluation/test_first_use_and_daily_metrics.py",
            "tests/integration/evaluation/test_first_use_harness.py",
            "tests/integration/evaluation/test_fixture_dispatch_counts.py",
            "tests/integration/evaluation/test_fresh_prestart.py",
            "tests/integration/evaluation/test_historical_bundles.py",
            "tests/integration/evaluation/test_historical_cro_usage.py",
            "tests/integration/evaluation/test_home_read_once.py",
            "tests/integration/evaluation/test_host_meter.py",
            "tests/integration/evaluation/test_instruction_configs.py",
            "tests/integration/evaluation/test_intent_eval.py",
            "tests/integration/evaluation/test_intent_gate.py",
            "tests/integration/evaluation/test_linked_book_completion.py",
            "tests/integration/evaluation/test_live_runner.py",
            "tests/integration/evaluation/test_metrics_behaviour.py",
            "tests/integration/evaluation/test_model_run_completion.py",
            "tests/integration/evaluation/test_native_input_mapping.py",
            "tests/integration/evaluation/test_native_observation_fidelity.py",
            "tests/integration/evaluation/test_native_preflight.py",
            "tests/integration/evaluation/test_nightly_smoke.py",
            "tests/integration/evaluation/test_owned_runner.py",
            "tests/integration/evaluation/test_owner_fact_measurements.py",
            "tests/integration/evaluation/test_packet_runner.py",
            "tests/integration/evaluation/test_packet_runner_continuation.py",
            "tests/integration/evaluation/test_path_explorer.py",
            "tests/integration/evaluation/test_path_research_update.py",
            "tests/integration/evaluation/test_person_intent_environment.py",
            "tests/integration/evaluation/test_person_turn_usage_additivity.py",
            "tests/integration/evaluation/test_read_audit.py",
            "tests/integration/evaluation/test_refusal_route_scoring.py",
            "tests/integration/evaluation/test_replay_clock.py",
            "tests/integration/evaluation/test_request_attribution_arithmetic.py",
            "tests/integration/evaluation/test_runtime_paths.py",
            "tests/integration/evaluation/test_served_closure.py",
            "tests/integration/evaluation/test_session_view.py",
            "tests/integration/evaluation/test_setup_boundary_admission.py",
            "tests/integration/evaluation/test_standing_gates.py",
            "tests/integration/evaluation/test_trial_instruction_isolation.py",
            "tests/integration/evaluation/test_usage_aggregation_availability.py",
            "tests/integration/evaluation/test_workspace_fixture_isolation.py",
            "tests/integration/evaluation/test_x8_score.py",
            "tests/unit/evaluation/test_accepted_routing.py",
            "tests/unit/evaluation/test_book_comparison_alignment.py",
            "tests/unit/evaluation/test_checkpoint_measurement_fidelity.py",
            "tests/unit/evaluation/test_client_socket_accounting.py",
            "tests/unit/evaluation/test_comparison_measurement_fidelity.py",
            "tests/unit/evaluation/test_conversation_budget_accounting.py",
            "tests/unit/evaluation/test_dossier_clock_fidelity.py",
            "tests/unit/evaluation/test_historical_cost_regressions.py",
            "tests/unit/evaluation/test_interval_union_accounting.py",
            "tests/unit/evaluation/test_market_input_modes.py",
            "tests/unit/evaluation/test_person_universe.py",
            "tests/unit/evaluation/test_phase_ledger_completeness.py",
            "tests/unit/evaluation/test_phase_totals_audit.py",
            "tests/unit/evaluation/test_run_comparison_fidelity.py",
            "tests/unit/evaluation/test_verified_fakes.py",
        ],
        "reason": (
            "This test or helper reads private QA records, internal tools or development history; "
            "the private checkout retains its coverage."
        ),
        "path_reasons": {
            "tests/structural/test_catalog_gate.py": (
                "covers private locale-input registrations and their public-absence control"
            ),
            "tests/alternative_evidence_desk/test_evidence_review_route.py": (
                "runs the private Review publication harness for exact admitted Task subjects"
            ),
            "tests/portfolio_strategy_lab/test_daily_book_records_gate.py": (
                "needs the private served-Host fixture and first-use write meter"
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
            "README.zh-CN.md",
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
            ".test_durations",
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
            "docs/images/research-performance-light.png",
            "docs/images/research-performance-dark.png",
            "docs/images/research-positions-light.png",
            "docs/images/research-positions-dark.png",
            "docs/images/research-conversation-light.png",
            "docs/images/research-conversation-dark.png",
            "docs/images/first-use-codex-light.png",
            "docs/images/first-use-codex-dark.png",
            "docs/images/committee-codex-light.png",
            "docs/images/committee-codex-dark.png",
            "docs/assets/readme/home-1280.png",
            "docs/assets/readme/home-900.png",
            "docs/assets/readme/research-1280.png",
            "docs/assets/readme/activity-1280.png",
            "docs/assets/readme/sentence-to-result.gif",
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
            "scripts/tests_registry.py",
            "scripts/pipeline.py",
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

_PUBLIC_FACT_KIND = "public-v2:"


def _cache_safe(raw: bytes) -> bool:
    """Never persist source-derived names from a file with a disclosure finding."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        return False
    return not any(pattern.search(text) for pattern in SCANS.values())


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


def _identifier_outputs(tree: ast.Module, producers: dict, fields: dict):
    scopes = [tree, *[n for n in tree.body if isinstance(n, ast.FunctionDef)]]
    for scope in scopes:
        slots = producers.get(getattr(scope, "name", None), {})
        for node in scope.body:
            match node:
                case ast.Assert(test=ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])):
                    if isinstance(left, ast.ListComp):
                        left = left.elt
                    match left:
                        case ast.Subscript(slice=ast.Constant(value=field)) if field in fields:
                            yield right, fields[field]
                case ast.Return(value=ast.Dict(keys=keys, values=values)) if slots:
                    for key, value in zip(keys, values, strict=True):
                        if isinstance(key, ast.Constant) and key.value in slots:
                            yield value, slots[key.value]


def _product_identifier_spans(path: str, source: str, forms: list) -> set[tuple[int, int, int]]:
    """Keep literal spans, so adjacent prose and comments remain development labels."""
    relevant = [f for f in forms if path in f.get("producers", {}) | f.get("assertions", {})]
    if not relevant or not path.endswith(".py"):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    lines, spans = source.splitlines(), set()
    for form in relevant:
        pattern = form["compiled"]
        pending = list(
            _identifier_outputs(
                tree,
                form.get("producers", {}).get(path, {}),
                form.get("assertions", {}).get(path, {}),
            )
        )
        while pending:
            literal, slot = pending.pop()
            if isinstance(literal, ast.IfExp):
                pending.extend((v, slot) for v in (literal.body, literal.orelse))
            elif slot == "keys" and isinstance(literal, (ast.List, ast.Tuple)):
                pending.extend((v, slot) for v in literal.elts)
            elif slot == "keys" and isinstance(literal, ast.Dict):
                pending.extend((key, "value") for key in literal.keys)
            elif (
                slot == "value"
                and isinstance(literal, ast.Constant)
                and isinstance(literal.value, str)
                and pattern.fullmatch(literal.value)
                and literal.lineno == literal.end_lineno
            ):
                line = lines[literal.lineno - 1].encode("utf-8")
                start = len(line[: literal.col_offset].decode("utf-8"))
                end = len(line[: literal.end_col_offset].decode("utf-8"))
                spans.add((literal.lineno, start, end))
    return spans


_TEST_NAMES = re.compile(r"(?m)^\s*(?:async\s+)?def\s+test_\w+(?=\s*\()")


def _label_matches(
    path: str, raw: bytes, forms: list[Any], pattern: re.Pattern[str], rules: str
) -> list[tuple[int, int, str, bool]]:
    """One PUBLIC file's development labels, as (line, column, label, inside a secret finding),
    outside its product identifier spans and test names; kept by its path, bytes and ``rules``
    within a `generate`."""

    def matched(data: bytes) -> list[list[Any]]:
        source = data.decode("utf-8-sig")
        permitted = _product_identifier_spans(path, source, forms)
        secret_spans = {
            number: [(match.start(), match.end()) for match in SCANS["NO_SECRET"].finditer(line)]
            for number, line in enumerate(source.splitlines(), 1)
        }
        source = _TEST_NAMES.sub(lambda m: m[0].replace("_", " "), source)
        return [
            [
                number,
                match.start() + 1,
                internal_id_label(match.group()),
                any(
                    match.start() < end and start < match.end()
                    for start, end in secret_spans[number]
                ),
            ]
            for number, line in enumerate(source.splitlines(), 1)
            for match in pattern.finditer(line)
            if not any(
                n == number and a <= match.start() and match.end() <= b for n, a, b in permitted
            )
        ]

    def shaped(rows: Any) -> bool:
        return isinstance(rows, list) and all(
            isinstance(row, list)
            and len(row) == 4
            and type(row[0]) is int
            and type(row[1]) is int
            and type(row[2]) is str
            and type(row[3]) is bool
            for row in rows
        )

    rows = (
        matched(raw)
        if _FACTS is None or not _cache_safe(raw)
        else _FACTS.of(f"{_PUBLIC_FACT_KIND}labels {rules} {path}", raw, matched, shaped)
    )
    return [(line, column, label, sensitive) for line, column, label, sensitive in rows]


def internal_id_ratchet(
    blobs: dict[str, bytes], decisions: list[dict], policy: dict | None = None
) -> dict:
    """Refuse public development labels beyond their per-file, per-label frozen allowance."""
    policy_values = policy or {}
    pattern = internal_id_pattern(policy)
    owners = json.loads(blobs.get("config/registries/owners-map.json", b"{}"))
    forms = [
        dict(f, owner=noun["owner"])
        for noun in owners.get("nouns", {}).values()
        for f in noun.get("identifier_forms", [])
    ]
    reserved = internal_id_pattern()
    for form in forms:
        owned, producers = "src/" + form["owner"].replace(".", "/") + "/", form["producers"]
        if not producers or any(p not in blobs or not p.startswith(owned) for p in producers):
            raise ValueError("product identifier needs its owning production source")
        expected = re.escape(form["prefix"]) + r"[1-9][0-9]*"
        if form["pattern"] != expected or reserved.search(form["prefix"] + "1"):
            raise ValueError("product identifier form cannot widen a reserved namespace")
        form["compiled"] = re.compile(expected)
    occurrences = policy_values.get("occurrences", {})
    limits = [n for row in occurrences.values() for n in row.values()]
    if any(type(count) is not int or count < 0 for count in limits):
        raise ValueError("internal development label counts must be nonnegative integers")
    current = {}
    current_occurrences = {}
    sensitive_labels = set()
    allowed = 0
    violations = []
    rules = hashlib.sha256(  # what a file's matches depend on beside its path and bytes
        json.dumps(
            [pattern.pattern, [{k: v for k, v in f.items() if k != "compiled"} for f in forms]],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    for row in decisions:
        if row["kind"] != "PUBLIC":
            continue
        path = row["path"]
        try:
            blobs[path].decode("utf-8-sig")
        except UnicodeError:
            continue
        matches = _label_matches(path, blobs[path], forms, pattern, rules)
        if matches:
            current[path] = len(matches)
        seen = Counter()
        for line, column, label, sensitive in matches:
            if sensitive:
                sensitive_labels.add(label)
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
            current_occurrences[path] = seen
    # Baseline decisions above use the original label. Only the returned keys are redacted,
    # including occurrences elsewhere of any label that appeared inside a secret finding.
    occupied = {label for seen in current_occurrences.values() for label in seen}
    aliases = {}
    ordinal = 0
    for label in sorted(sensitive_labels):
        while True:
            ordinal += 1
            alias = f"<NO_SECRET:{ordinal}>"
            if alias not in occupied:
                aliases[label] = alias
                occupied.add(alias)
                break
    current_occurrences = {
        path: dict(sorted((aliases.get(label, label), count) for label, count in seen.items()))
        for path, seen in current_occurrences.items()
    }
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
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if shallow.returncode or shallow.stdout.strip() != "false":
        raise ValueError(
            "shallow or unreadable Git history; restore complete Git history and retry"
        )
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{source}^{{commit}}"], cwd=repo, text=True
    ).strip()
    later = ""
    if admissions_from is not None:
        later = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{admissions_from}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    graph = {}
    for row in subprocess.check_output(
        ["git", "rev-list", "--parents", source],
        cwd=repo,
        text=True,
    ).splitlines():
        fields = row.split()
        if not fields or any(not re.fullmatch(r"[0-9a-f]{40}", oid) for oid in fields):
            raise ValueError("unreadable internal-label history graph; restore Git history")
        commit, *parents = fields
        if commit in graph:
            raise ValueError("repeated internal-label history graph entry; restore Git history")
        graph[commit] = parents
    if source not in graph or any(
        parent not in graph for parents in graph.values() for parent in parents
    ):
        raise ValueError("incomplete internal-label history graph; restore Git history")
    commits = subprocess.check_output(
        ["git", "log", "--full-history", "--format=%H", source, "--", INTERNAL_ID_BASELINE],
        cwd=repo,
        text=True,
    ).splitlines()
    if any(commit not in graph for commit in commits):
        raise ValueError("incomplete internal-label history graph; restore Git history")
    policies = set(commits) | {source} | ({later} if later else set())
    policies.update(parent for commit in commits for parent in graph[commit])
    resolved, contents = {}, {}

    def acquire(specs: set[str]) -> None:
        pending = sorted(specs - resolved.keys())
        if not pending:
            return
        raw = subprocess.check_output(
            ["git", "cat-file", "--batch-check", "-z"],
            cwd=repo,
            input="".join(spec + "\0" for spec in pending).encode(),
        )
        cursor = 0
        for spec in pending:
            missing = (spec + " missing\n").encode()
            if raw.startswith(missing, cursor):
                resolved[spec] = None
                cursor += len(missing)
                continue
            end = raw.find(b"\n", cursor)
            fields = raw[cursor:end].split() if end >= 0 else []
            if (
                len(fields) != 3
                or not re.fullmatch(rb"[0-9a-f]{40}", fields[0])
                or fields[1] != b"blob"
                or not fields[2].isdigit()
            ):
                raise ValueError(f"unexpected git object response for {spec}; restore Git objects")
            resolved[spec] = fields[0].decode("ascii")
            cursor = end + 1
        if cursor != len(raw):
            raise ValueError("unexpected trailing git object response; restore Git objects")
        contents.update(
            _read_blob_batch(repo, {oid for oid in resolved.values() if oid} - contents.keys())
        )

    acquire({f"{commit}:{INTERNAL_ID_BASELINE}" for commit in policies})
    cache = {
        commit: json.loads(contents[oid])
        if (oid := resolved[f"{commit}:{INTERNAL_ID_BASELINE}"])
        else None
        for commit in policies
    }
    anchor_counts = {}

    def policy_at(commit: str) -> dict | None:
        return cache[commit]

    source_policy = policy_at(source)
    if source_policy is None:
        raise ValueError("the source has no frozen internal-label policy")
    # a later reviewed policy (develop's) may admit an earlier commit's exact rows by name; a
    # commit's own policy never widens its own history beyond what it names
    admitting = source_policy
    if later:
        admitting = policy_at(later) or source_policy
    admissions = admitting.get("history_admissions", {})
    anchors = {
        policy.get("source_sha", "")
        for commit in commits
        if (policy := cache[commit])
        if policy and policy.get("schema_version", 1) == 2
    }
    anchors = {anchor for anchor in anchors if re.fullmatch(r"[0-9a-f]{40}", anchor)}
    acquire({f"{anchor}:{INTERNAL_ID_BASELINE}" for anchor in anchors})
    for anchor in anchors:
        oid = resolved[f"{anchor}:{INTERNAL_ID_BASELINE}"]
        cache[anchor] = json.loads(contents[oid]) if oid else None
    owners = set()
    for commit in commits:
        policy = policy_at(commit)
        if (
            not policy
            or policy.get("schema_version", 1) != 2
            or policy.get("source_sha") not in anchors
        ):
            continue
        for parent in graph[commit]:
            previous = policy_at(parent) or {}
            if previous.get("schema_version", 1) == 1:
                for owner, labels in policy.get("occurrences", {}).items():
                    if any(
                        count > previous.get("occurrences", {}).get(owner, {}).get(label, 0)
                        for label, count in labels.items()
                    ):
                        validate_path(owner)
                        owners.add(f"{policy['source_sha']}:{owner}")
    acquire(owners)
    ancestor_sets, children = {}, {}
    for child, parents in graph.items():
        for parent in parents:
            children.setdefault(parent, []).append(child)

    def is_ancestor(anchor: str, commit: str) -> bool:
        if anchor not in ancestor_sets:
            # Descendants are shared by every policy commit with this immutable anchor.
            descendants, todo = {anchor}, [anchor]
            while todo:
                for child in children.get(todo.pop(), []):
                    if child not in descendants:
                        descendants.add(child)
                        todo.append(child)
            ancestor_sets[anchor] = descendants
        if commit in ancestor_sets[anchor]:
            return True
        # Git remains authoritative at shallow boundaries or for unrelated anchors.
        return (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", anchor, commit],
                cwd=repo,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )

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
            or not is_ancestor(anchor, commit)
            or policy_at(anchor) is not None
        ):
            raise ValueError(f"the first-landing anchor must precede the policy at {commit}")
        parents = graph[commit]
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
                            oid = resolved[f"{anchor}:{owner}"]
                            try:
                                text = contents[oid].decode("utf-8-sig") if oid else ""
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
    if (
        not path
        or "\0" in path
        or "\\" in path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or ":" in path
    ):
        raise ValueError(f"unsafe manifest path: {path!r}")


def _read_blob_batch(repo: Path, objects: set[str]) -> dict[str, bytes]:
    """Read each immutable blob once and refuse truncated or mismatched batch output."""
    if not objects:
        return {}
    ordered = sorted(objects)
    raw = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input="".join(oid + "\n" for oid in ordered).encode(),
    )
    blobs, cursor = {}, 0
    for oid in ordered:
        end = raw.find(b"\n", cursor)
        header = raw[cursor:end].split() if end >= 0 else []
        if len(header) != 3 or header[:2] != [oid.encode(), b"blob"] or not header[2].isdigit():
            raise ValueError(f"unexpected git object response for {oid}; restore Git objects")
        size = int(header[2])
        stop = end + 1 + size
        if stop >= len(raw) or raw[stop : stop + 1] != b"\n":
            raise ValueError(f"incomplete git object response for {oid}; restore Git objects")
        blobs[oid] = raw[end + 1 : stop]
        cursor = stop + 1
    if cursor != len(raw):
        raise ValueError("unexpected trailing git object response; restore Git objects")
    return blobs


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
    contents = _read_blob_batch(repo, {oid for _, oid in objects})
    blobs = {path: contents[oid] for path, oid in objects}
    return sha, blobs, modes


_IMPORT_NAMES: dict[tuple[str, bytes], frozenset[str]] = {}
"""Each Python file's imported names by its place and content digest, parsed once a process: both
closures of a `generate`, and every later one, read the same answer. A
parse error is raised every time, never kept."""


_FACTS: FileFacts | None = None
"""The store a `generate` holds for its files' import statements, kept by their bytes when the
gate names one; None outside a `generate`."""


def _statements(path: str, raw: bytes) -> list[list]:
    """Each import statement of one file as its line, module, level, names and whether it is a
    `from` import, read from the bytes alone; each reader resolves relative ones itself."""

    def parsed(data: bytes) -> list[list]:
        return [
            [
                n.lineno,
                getattr(n, "module", None) or "",
                getattr(n, "level", 0),
                [a.name for a in n.names],
                isinstance(n, ast.ImportFrom),
            ]
            for n in ast.walk(ast.parse(data.decode("utf-8-sig"), filename=path))
            if isinstance(n, ast.Import | ast.ImportFrom)
        ]

    def shaped(rows: Any) -> bool:
        return all(
            type(line) is int
            and type(module) is str
            and type(level) is int
            and all(type(a) is str for a in names)
            and type(frm) is bool
            for line, module, level, names, frm in rows
        )

    return (
        parsed(raw)
        if _FACTS is None or not _cache_safe(raw)
        else _FACTS.of(f"{_PUBLIC_FACT_KIND}imports", raw, parsed, shaped)
    )


def _absolute(package: str, module: str, level: int, names: list[str], frm: bool) -> list[str]:
    """The absolute names one import statement imports, a relative one against ``package``."""
    if not frm:
        return names
    if level:
        parts = package.split(".")
        module = ".".join(parts[: len(parts) - level + 1] + ([module] if module else []))
    return [module, *(module + "." + name for name in names)]


def _import_names(path: str, raw: bytes) -> frozenset[str]:
    """The absolute names one file imports, relative imports resolved against its package."""
    key = (path, hashlib.sha256(raw).digest())
    if (known := _IMPORT_NAMES.get(key)) is not None:
        return known
    own = path[4:-3].replace("/", ".").removesuffix(".__init__") if path.startswith("src/") else ""
    package = own if path.endswith("/__init__.py") else own.rpartition(".")[0]
    names = {n for row in _statements(path, raw) for n in _absolute(package, *row[1:])}
    known = _IMPORT_NAMES[key] = frozenset(names)
    return known


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
        # A module is reached when it is a dotted prefix of an imported name (or the name itself):
        # each prefix is looked up, not every module tested against every name.
        edges[path] = {
            modules[prefix]
            for name in _import_names(path, raw)
            for prefix in (".".join(name.split(".")[:end]) for end in range(1, name.count(".") + 2))
            if prefix in modules
        }
    reached = set(roots)
    todo = list(roots)
    while todo:
        for target in edges.get(todo.pop(), set()) - reached:
            reached.add(target)
            todo.append(target)
    return reached


PRIVATE_SCHEMAS = frozenset(
    {
        "alphalattice.release.public-manifest",
        "alphalattice.tests-unregistered.v1",
        "alphalattice.agent-eval.host-facts",
    }
)
"""Private data by what it is: a tracked JSON document carrying one of these schemas is PRIVATE
wherever it lives, so no public rule names its path (CHECK-PLAYPEN-PUBLIC-SAFE)."""


def declared_private(blobs: dict[str, bytes]) -> frozenset[str]:
    """The files private by their own content: a JSON document carrying a private schema, or a test
    module whose docstring declares `PRIVATE_TEST:`, each read as parsed, so an escaped spelling is
    the same declaration. A JSON document or test module that cannot be parsed cannot show it is
    not private, so it is private too: content only ever makes a file private."""
    found = set()
    for path, raw in blobs.items():
        if path.endswith(".json"):
            try:
                document = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeError, ValueError, RecursionError):
                found.add(path)
                continue
            if isinstance(document, dict) and document.get("schema") in PRIVATE_SCHEMAS:
                found.add(path)
        elif path.startswith("tests/") and path.endswith(".py"):
            try:
                docstring = ast.get_docstring(ast.parse(raw.decode("utf-8-sig"))) or ""
            except (UnicodeError, SyntaxError, ValueError, RecursionError):
                found.add(path)
                continue
            if "PRIVATE_TEST:" in docstring:
                found.add(path)
    return frozenset(found)


def classify(
    path: str,
    runtime: set[str],
    test_dependencies: set[str],
    config_dependencies: set[str],
    strict_scripts: set[str] | None = None,
    declared: frozenset[str] = frozenset(),
) -> tuple[str, str | None, str]:
    """The first applicable named rule wins; absence of evidence stays unsettled. A file private by
    its own content (`declared_private`) is PRIVATE before any rule."""
    validate_path(path)
    if path in declared:
        return (
            "PRIVATE",
            "PRIVATE_BY_CONTENT",
            (
                "Its own content declares it private (a private data schema or a PRIVATE_TEST:"
                " test), or cannot be parsed to show it is not."
            ),
        )
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
                statements = _statements(row["path"], raw)
            except SyntaxError:
                statements = []  # Text-pattern scans still apply to an invalid/synthetic source.
            own = row["path"].removeprefix("src/")[:-3].replace("/", ".")
            package = (
                own.removesuffix(".__init__")
                if own.endswith(".__init__")
                else own.rpartition(".")[0]
            )
            imports = {}
            for line, *statement in statements:
                names = _absolute(package, *statement)
                targets = {p for name in names for p in private_modules.get(name, ())}
                if targets:
                    imports.setdefault(line, set()).update(targets)
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
        scanned = _text_facts(raw, text)
        for number, column, rule, assessment in scanned:
            findings.append(
                {
                    "kind": "FINDING",
                    "path": row["path"],
                    "line": number,
                    "column": column,
                    "rule": rule,
                    "assessment": (
                        "TEST_LITERAL_REVIEW" if row["path"].startswith("tests/") else assessment
                    ),
                }
            )
        for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
            refs = private_references(row["path"], line, private, blobs.keys())
            findings.extend(
                {
                    "kind": "FINDING",
                    "path": row["path"],
                    "line": number,
                    "rule": "NO_PRIVATE_REFERENCE",
                    "assessment": ref,
                }
                for ref in sorted(refs)
            )
    return sorted(
        findings, key=lambda r: (r["path"], r["line"], r["rule"], r.get("column", 0))
    ), binary


def private_references(reader: str, text: str, private: set[str], tracked) -> set[str]:
    """The NO_PRIVATE_REFERENCE assessments one line of a PUBLIC file earns: a token naming a
    tracked private file, or a reference into an internal tree. The scan and the registry
    projection ask this one rule, so "private" means one thing."""
    return _references(reader, *_reference_tokens(text), private, tracked)


def _reference_tokens(text: str) -> tuple[list[str], bool]:
    """One line's path-like tokens, and whether it names an internal tree, by its text alone."""
    normalized = text.replace("\\\\", "/").replace("\\", "/")
    tokens = sorted(set(re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_-]+", normalized)))
    return tokens, any(prefix in normalized for prefix in PRIVATE_PREFIXES)


def _references(
    reader: str, tokens: list[str], internal: bool, private: set[str], tracked: Any
) -> set[str]:
    """``private_references`` from one line's tokens: the half of the rule that reads the tree."""
    refs = set()
    for token in tokens:
        direct = token[2:] if token.startswith("./") else token
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(reader), token))
        if relative in private or (relative not in tracked and direct in private):
            refs.add("TRACKED_PRIVATE_FILE")
    if internal:
        refs.add("INTERNAL_TREE_REFERENCE")
    return refs


_TEXT_RULES = hashlib.sha256(  # SCANS and PRIVATE_PREFIXES, which a loaded code identity omits
    json.dumps(
        [{rule: [p.pattern, p.flags] for rule, p in SCANS.items()}, list(PRIVATE_PREFIXES)]
    ).encode()
).hexdigest()[:16]


def _text_facts(raw: bytes, text: str) -> list[Any]:
    """One PUBLIC file's location-only pattern findings, kept by its bytes within a `generate`."""

    def found(_data: bytes) -> list[list[Any]]:
        scanned: list[list[Any]] = []
        for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
            for rule, pattern in SCANS.items():
                for match in pattern.finditer(line):
                    assessment = "REVIEW_REQUIRED"
                    if rule == "NO_INTERNAL_ID" and not re.search(
                        r"thread|session|codex|claude|agent", line, re.I
                    ):
                        assessment = "UUID_PURPOSE_UNSETTLED"
                    scanned.append([number, match.start() + 1, rule, assessment])
        return scanned

    def shaped(value: Any) -> bool:
        return isinstance(value, list) and all(
            isinstance(row, list) and len(row) == 4 for row in value
        )

    return (
        found(raw)
        if _FACTS is None
        else _FACTS.of(f"{_PUBLIC_FACT_KIND}text {_TEXT_RULES}", raw, found, shaped)
    )


NOT_PROJECTED = {
    "config/registries/owners-map.json": (
        "Its rows are nouns, and a noun whose consumer list names a private tool is still a public"
        " owner: the public tree's message catalog and requirement checks need every noun. Its"
        " private consumer paths stay scan findings for review (10-29, public build at head)."
    ),
}
"""Public registries the projection leaves whole, each with the reason a public test needs it."""


def project_public(path: str, raw: bytes, private: set[str], tracked) -> tuple[bytes, list[str]]:
    """A PUBLIC registry as it is published: each row holding a private reference dropped.

    A row is an element of a top-level list or an entry of a top-level object; it is dropped when
    any string in it, key or value, earns a private reference by the scan's own rule. Any other
    file, an unreadable registry, or one with nothing to drop keeps its exact bytes; the private
    registry is never changed. Returns the bytes and each dropped row's place, such as
    `reach[3]` or `entries/<key>` (REGISTRY-PUBLIC-PROJECTION)."""
    if not (path.startswith("config/registries/") and path.endswith(".json")):
        return raw, []
    if path in NOT_PROJECTED:
        return raw, []
    try:
        text = raw.decode("utf-8-sig")
        data = json.loads(text)
    except (UnicodeError, ValueError):
        return raw, []  # scanned as it is; the scan names what it finds

    def strings(node: object):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from strings(value)

    def flagged(*nodes: object) -> bool:
        return any(private_references(path, s, private, tracked) for n in nodes for s in strings(n))

    # The test-size policy's inputs are dependencies, not the identity of their public row.
    # Classify a wildcard as private only when it matches tracked paths and all are private.
    tracked = frozenset(tracked)
    sizes = (
        path == "config/registries/test-sizes.json"
        and isinstance(data, dict)
        and data.get("schema") == "alphalattice.test-sizes.v1"
    )

    def private_selector(value: object) -> bool:
        if not isinstance(value, str):
            return False
        if flagged(value):
            return True
        pattern = value.replace("\\", "/").removeprefix("./").split("::", 1)[0]
        if not any(char in pattern for char in "*?["):
            return False
        matched = {name for name in tracked if fnmatch.fnmatchcase(name, pattern)}
        return bool(matched) and matched <= private

    projected: object
    dropped: list[str] = []
    if isinstance(data, list):
        dropped = [f"[{i}]" for i, row in enumerate(data) if flagged(row)]
        projected = [row for row in data if not flagged(row)]
    elif isinstance(data, dict):
        kept: dict[str, object] = {}
        for key, value in data.items():
            if isinstance(value, list):
                rows = []
                for i, row in enumerate(value):
                    place = f"{key}[{i}]"
                    if sizes and isinstance(row, dict):
                        rest = {name: item for name, item in row.items() if name != "inputs"}
                        if (
                            flagged(rest)
                            or private_selector(row.get("pattern"))
                            or private_selector(row.get("node"))
                        ):
                            dropped.append(place)
                            continue
                        inputs = row.get("inputs")
                        if not isinstance(inputs, list) and (
                            flagged(inputs) or private_selector(inputs)
                        ):
                            dropped.append(place)
                            continue
                        if isinstance(inputs, list):
                            kept_inputs = []
                            for j, item in enumerate(inputs):
                                if flagged(item) or private_selector(item):
                                    dropped.append(f"{place}.inputs[{j}]")
                                else:
                                    kept_inputs.append(item)
                            if len(kept_inputs) != len(inputs):
                                row = {**row, "inputs": kept_inputs}
                    elif flagged(row) or (sizes and private_selector(row)):
                        dropped.append(place)
                        continue
                    rows.append(row)
                kept[key] = rows
            elif isinstance(value, dict):
                dropped += [f"{key}/{name}" for name, row in value.items() if flagged(name, row)]
                kept[key] = {n: row for n, row in value.items() if not flagged(n, row)}
            else:
                kept[key] = value
        projected = kept
    if not dropped:
        return raw, []
    return _like(projected, raw, text), dropped


def _like(value: object, raw: bytes, text: str) -> bytes:
    """``value`` as JSON in the form ``text`` was written in: the indent of its first indented
    line, or one line with its own separators when it has none; its line ending, final newline,
    byte-order mark and escaping of non-ASCII text."""
    newline = "\r\n" if "\r\n" in text else "\n"
    later = [line for line in text.splitlines()[1:] if line.strip()]
    if later:
        indent = later[0][: len(later[0]) - len(later[0].lstrip())]
        body = json.dumps(value, indent=indent, ensure_ascii=text.isascii())
    else:
        separators = (
            "," + (" " if re.search(r",\s", text) else ""),
            ":" + (" " if re.search(r":\s", text) else ""),
        )
        body = json.dumps(value, separators=separators, ensure_ascii=text.isascii())
    body = body.replace("\n", newline) + (newline if text.endswith("\n") else "")
    return (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b"") + body.encode("utf-8")


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
    """Produce the builder contract and complete private audit deterministically; each file's
    import statements are kept by its bytes when the gate names a store."""
    global _FACTS
    if not os.environ.get("ALPHALATTICE_SCAN_CACHE_DIR"):  # structural.SCAN_CACHE_ENV
        return _generate(sha, blobs, modes)  # the export runs this scanner alone, without it
    from devtools.architecture.structural import kept_facts  # named only where it is importable

    with kept_facts("public-manifest", sys.modules[__name__]) as _FACTS:
        try:
            if (purged := _FACTS.retire_predecessors(_PUBLIC_FACT_KIND)) is not None:
                print(f"public-manifest: purged {purged} predecessor facts", file=sys.stderr)
            return _generate(sha, blobs, modes)
        finally:
            _FACTS = None


def _generate(sha: str, blobs: dict[str, bytes], modes: dict[str, str]) -> tuple[dict, dict]:
    declared = declared_private(blobs)  # private by content, whatever the path
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
            and classify(p, set(), set(), set(), declared=declared)[0] == "PUBLIC"
        },
    )
    initial = [
        dict(path=p, kind=classify(p, runtime, tests, set(), declared=declared)[0])
        for p in sorted(blobs)
    ]
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
        dict(path=p, kind=classify(p, runtime, tests, config_dependencies, declared=declared)[0])
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
            kind, rule, reason = classify(
                path, runtime, tests, config_dependencies, strict_scripts, declared
            )
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
    private = {r["path"] for r in decisions if r["kind"] == "PRIVATE"}
    blobs, projections = dict(blobs), []  # the bytes as published; the caller's private blobs stay
    for row in decisions:
        if row["kind"] == "PUBLIC":
            published, dropped = project_public(
                row["path"], blobs[row["path"]], private, blobs.keys()
            )
            if dropped:
                projections.append(
                    {
                        "path": row["path"],
                        "source_sha256": hashlib.sha256(blobs[row["path"]]).hexdigest(),
                        "public_sha256": hashlib.sha256(published).hexdigest(),
                        "dropped": dropped,
                    }
                )
                blobs[row["path"]] = published
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
    content_keys(findings, blobs)
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
        "registry_projections": projections,
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
    if verification and (
        verification["rounds"] or verification["corrections"] or verification["original_findings"]
    ):
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


def content_keys(rows: list[dict], blobs: dict[str, bytes]) -> None:
    """Bind findings to exact UTF-8 lines, normalizing CRLF only; cache each path and line."""
    lines, hashes = {}, {}
    for row in rows:
        path, number = row["path"], row["line"]
        if path not in lines:
            lines[path] = blobs[path].decode("utf-8").replace("\r\n", "\n").split("\n")
        if not 0 < number <= len(lines[path]):
            raise ValueError("review key has no source line")
        if (path, number) not in hashes:
            hashes[path, number] = hashlib.sha256(
                lines[path][number - 1].encode("utf-8")
            ).hexdigest()
        row["review_key"] = [REVIEW_KEY, path, row["rule"], hashes[path, number]]


def _unique_review_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON key in review record")
    return result


def attach_reviews(
    audit: dict,
    verification: dict,
    repo: Path,
    blobs: dict[str, bytes],
    *,
    policy_reviewer: bool = False,
) -> None:
    """Admit emitted keys only; reserved example contacts retain their declared policy."""
    held = verification["standalone"]
    accepted = held.get("current_findings", [])
    reopened = held.setdefault("reopened_findings", [])
    reviews = {}
    for row in accepted:
        key = row.get("review_key")
        if (
            not isinstance(key, list)
            or len(key) != 4
            or key[:3] != [REVIEW_KEY, row["path"], row["rule"]]
            or not isinstance(key[3], str)
            or not re.fullmatch("[0-9a-f]{64}", key[3])
        ):
            raise ValueError("invalid review key")
        value, key = row, tuple(key)
        prior = reviews.setdefault(key, {"disposition": value["disposition"], "evidence": {}})
        if prior["disposition"] != value["disposition"]:
            raise ValueError("conflicting review key")
        prior["evidence"][value["evidence"]] = None
    current = {tuple(row["review_key"]) for row in audit["findings"]}
    held["current_findings"] = []
    for row in accepted:
        if tuple(row["review_key"]) in current:
            held["current_findings"].append(row)
        else:
            reopened.append(dict(row, reopen_reason="content_changed"))
    lines = {}
    for row in audit["findings"]:
        if tuple(row["review_key"]) in reviews:
            prior = reviews[tuple(row["review_key"])]
            row["review"] = {**prior, "evidence": "\n".join(prior["evidence"])}
        elif row["rule"] == "NO_PERSONAL_CONTACT":
            path = row["path"]
            if path not in lines:
                lines[path] = blobs[path].decode("utf-8").replace("\r\n", "\n").split("\n")
            text = lines[path][row["line"] - 1]
            addresses = re.findall(r"[\w.+-]+@((?:[\w-]+\.)+[\w-]+)", text)
            if addresses and all(
                re.search(r"(^|\.)(example\.(com|org|net)|invalid|test|example)$", d, re.I)
                for d in addresses
            ):
                row["review"] = {
                    "disposition": "KEEP_SYNTHETIC",
                    "evidence": (
                        f"{path}:{row['line']} at {audit['source_sha'][:9]}: every address on the "
                        "line uses a reserved example domain (RFC 2606/6761); a fixture, "
                        "not anyone's contact."
                    ),
                }
                if policy_reviewer:
                    row["review"]["reviewer"] = "reserved-example-domain-policy"
                held["current_findings"].append({**row, **row["review"]})
    audit["standalone"] = held


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
    parser.add_argument(
        "--census-path", help="The source tree's review record path and output location."
    )
    args = parser.parse_args()
    sha, blobs, modes = read_tree(args.repo, args.source)
    if INTERNAL_ID_BASELINE in blobs:
        validate_internal_id_history(args.repo, sha, admissions_from="develop")
    manifest, audit = generate(sha, blobs, modes)
    # contract: CENSUS-PATH-CALLER supplies the path before the legacy default is removed.
    if args.census_path is not None:
        try:
            validate_path(args.census_path)
            if (
                not args.check
                and args.census_path not in blobs
                and (args.out / args.census_path).exists()
            ):
                raise ValueError("source review record is absent but output exists")
            source = (
                args.verification_json.read_bytes()
                if args.verification_json
                else blobs.get(args.census_path)
            )
            if source is not None:
                verification = json.loads(source, object_pairs_hook=_unique_review_object)
                if not isinstance(verification, dict) or not isinstance(
                    verification.get("standalone"), dict
                ):
                    raise ValueError("standalone review record is missing or invalid")
                attach_reviews(audit, verification, args.repo, blobs, policy_reviewer=True)
            else:
                audit["standalone"] = {
                    key: []
                    for key in (
                        "rounds",
                        "corrections",
                        "original_findings",
                        "current_findings",
                        "reopened_findings",
                    )
                }
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise SystemExit(
                f"public_release.verification_invalid: {args.census_path}: {error}; "
                "restore the reviewed census or supply --verification-json"
            ) from error
    elif args.verification_json:
        verification = json.loads(args.verification_json.read_text(encoding="utf-8"))
        attach_reviews(audit, verification, args.repo, blobs)
    if args.verification_json:
        audit["command"] += " --verification-json census/public_release.json"
    if args.census_path is not None:
        audit["command"] += " --census-path=" + subprocess.list2cmdline([args.census_path])
    outputs = {
        "config/release/public-manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False)
        + "\n",
        "census/public_release.json": json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        "census/public_release.md": render(audit),
    }
    if args.census_path is not None:
        legacy = next(name for name in outputs if name.endswith("public_release.json"))
        target = (args.out / args.census_path).resolve()
        if not target.is_relative_to(args.out.resolve()) or any(
            target == (args.out / name).resolve()
            for name in outputs
            if name != legacy or args.census_path != legacy
        ):
            raise SystemExit("public_release.output_collision: choose a distinct census path")
        outputs[args.census_path] = outputs.pop(legacy)
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
