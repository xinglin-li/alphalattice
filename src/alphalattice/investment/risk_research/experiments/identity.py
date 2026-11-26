"""Source identity for the Risk development path, the one path Risk studies run.

``RiskEstimatorNumericalBinding`` hashes module-name strings and declared
semantics, not source content. A newly installed adapter module would not be
covered by that, and neither was the development compiler, executor, window
binding, or policy code: all of it could change while every Program hash stayed
put.

So the development path gets its own closures. Four identities stay deliberately
separate, and none of them is derived from another:

the numerical closure (``RISK_NUMERICAL_SOURCE_PATHS``) over the code that
  computes a covariance number;
the selected adapter's numerical/content identity;
this development executor/input-binding identity;
Host catalog governance identity.

Keeping them apart is what lets an unrelated newly installed adapter change the
catalog without masquerading as a numerical change to an already selected
method. (The frozen production closure and its pin, which the numerical closure
used to share its entries with, retired with M01 on 2026-09-29.)
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

RISK_DEVELOPMENT_SOURCE_PATHS = (
    "src/alphalattice/investment/risk_research/estimators/capability.py",
    "src/alphalattice/investment/risk_research/estimators/catalog.py",
    "src/alphalattice/investment/risk_research/estimators/contracts.py",
    "src/alphalattice/investment/risk_research/estimators/domains.py",
    "src/alphalattice/investment/risk_research/experiments/compiler.py",
    "src/alphalattice/investment/risk_research/experiments/contracts.py",
    "src/alphalattice/investment/risk_research/experiments/development.py",
    "src/alphalattice/investment/risk_research/experiments/development_artifacts.py",
    "src/alphalattice/investment/risk_research/experiments/execution.py",
    "src/alphalattice/investment/risk_research/experiments/formation_selection.py",
    "src/alphalattice/investment/risk_research/experiments/identity.py",
    "src/alphalattice/investment/risk_research/experiments/observation.py",
    "src/alphalattice/investment/risk_research/experiments/policy.py",
    "src/alphalattice/investment/risk_research/experiments/series.py",
    "src/alphalattice/investment/risk_research/experiments/verification.py",
    "src/alphalattice/investment/risk_research/experiments/window.py",
    "src/alphalattice/protocols/research_authoring/contracts.py",
)
"""Executable development code whose change must move development identity.

The list started as the compiler, the executor, the window and the policy, which
was the set that existed when it was written. It then stopped being audited while
the path around it grew, and by the time it was checked, the modules that
actually produce a development estimate and seal it into an artifact were not in
it at all:

``experiments/development.py``            runs every estimate and packs the chunks
``experiments/development_artifacts.py``  defines what the evidence *is*
``estimators/capability.py``              admission, sealing and determinism policy
``estimators/catalog.py``                 which implementation resolution returns
``estimators/contracts.py``               the numerical binding and bound-input schema
``estimators/domains.py``                 what a parameter domain will admit

Any of those could have been rewritten -- changing what runs, what is admitted,
or what the evidence records -- while every development Program hash stood still.

``experiments/verification.py`` is here for a different reason and it is worth
stating rather than blurring: it changes no number. It decides whether a graph is
*accepted* as an exact replay, which is a governance concern, not a numerical
one. But an ungoverned acceptance rule silently weakens every ``REUSED_EXACT``
claim ever made under it, so it is covered by the same closure rather than given
an identity system of its own.

Deliberately excludes the numerical modules: no entry here appears in
``RISK_NUMERICAL_SOURCE_PATHS``, so this closure moves with the development code
and that one with the code that computes a number. The numerical modules --
``estimators/covariance.py``, ``estimators/matrix_identity.py``,
``evaluation/formation.py``, ``surfaces/artifacts.py``, ``surfaces/returns.py``
and the returns they read -- are covered by that closure, which the Program
binds separately.
"""


def risk_development_source_closure_hash(playpen_root: Path) -> str:
    """Hash the development path's own bytes.

    Includes this module, so the declaration of what counts as development
    source is itself covered -- otherwise the list could be narrowed without
    the identity noticing.
    """

    return str(
        source_rule_closure_hash(
            root=playpen_root,
            tracked_paths=RISK_DEVELOPMENT_SOURCE_PATHS,
            semantic_owner="risk_research.experiments",
            numerical_role="RISK_DEVELOPMENT",
        )
    )


RISK_NUMERICAL_SOURCE_PATHS = (
    "src/alphalattice/foundation/market_data_ops/returns/execution.py",
    "src/alphalattice/foundation/market_data_ops/publication/projection.py",
    "src/alphalattice/investment/risk_research/contracts.py",
    "src/alphalattice/investment/risk_research/estimators/covariance.py",
    "src/alphalattice/investment/risk_research/estimators/matrix_identity.py",
    "src/alphalattice/investment/risk_research/evaluation/formation.py",
    "src/alphalattice/investment/risk_research/surfaces/artifacts.py",
    "src/alphalattice/investment/risk_research/surfaces/returns.py",
)
"""The numerical code a development run executes: the covariance estimator, the matrix
identity, the formation axis, the returns and their projection, taken by the rule (each
entry and what it imports inside the number-deciding packages, as syntax without
docstrings or comments, LAWS ID3). A study plan binds these bytes (binding plan, P); no
build file is an entry, and the installed versions are bound where they are read (ID6).

It used to be derived from the frozen production closure, whose pin retired with M01
(2026-09-29) together with the two production surface writers that were its other
entries. What measures a run stays out of it: ``process_metrics.py`` once sat in that
closure because a surface writer recorded ``peak_rss_bytes`` in a diagnostic dossier,
and a telemetry edit then rotated every Risk identity without moving a number.
"""


def risk_numerical_source_closure_hash(playpen_root: Path) -> str:
    """Hash the numerical code a development run executes, without the build file."""

    return str(
        source_rule_closure_hash(
            root=playpen_root,
            tracked_paths=RISK_NUMERICAL_SOURCE_PATHS,
            semantic_owner="risk_research.experiments",
            numerical_role="RISK_DEVELOPMENT_NUMERICS",
        )
    )


__all__ = [
    "RISK_DEVELOPMENT_SOURCE_PATHS",
    "RISK_NUMERICAL_SOURCE_PATHS",
    "risk_development_source_closure_hash",
    "risk_numerical_source_closure_hash",
]
