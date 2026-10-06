# Alpha Research Active Test Portfolio

Date: 2026-08-09

This directory covers the active Goal-driven Alpha runtime, deterministic
numerical owners, array integrity, authority readback, and the minimum retained
legacy boundary. It is intentionally not a museum for every superseded Agent
experiment.

## Test Classes

### Requirement and scientific regression

- `test_array_firewall.py` and `test_array_surface.py`: causal arrays, axes,
  duplicate detection, bounded lifetime, real read telemetry, and bitwise
  surface behavior.
- `test_numerical_goldens.py`, `test_full_inventory_execution.py`, and
  `test_current_viability_and_refit.py`: model mathematics, preprocessing
  prohibition, viability, and current-refit behavior.
- `test_metrics.py`: common-surface economic and statistical metrics.

### Authority, runtime, and recovery

- `test_alpha_qualification.py`: the qualification, its registry and the
  committer that seals a candidate set or a stop (what stayed of the goal loop,
  GR3).
- `test_artifacts_and_reuse.py`: content-addressed children, marker-last
  publication, tamper rejection, and exact reuse.
- `test_current_runtime_and_view.py`: retained historical readback/tamper
  boundaries plus shared array-surface fixtures pending final legacy-owner
  removal.
- `test_catalog_and_identity.py`: registered inventory and identity boundaries.
- `test_development_chunk_read_scope.py`: a verified development chunk is reused
  only inside the read that hashed it, and the streamed chunk hash equals the
  sealed definition byte for byte.

## Retired Tests

The following suites were deleted after their consumers were superseded by the
Goal-driven sparse runtime or ended as pre-adoption experiments:

- fixed Alpha Graph/Task and old Agent selector boundary tests;
- Stability Case one-shot Agent review tests;
- Todo-enabled Evidence Workspace pre-adoption and reading-efficiency tests.

Their immutable records and artifacts remain historical evidence. They are not
part of the active development gate and must not be revived by copying old
tests into the current runtime.

## Verification Routing

`scripts/check_alpha_verification_impact.py` maps changed responsibilities to
the smallest active test set. Array-value or model-kernel changes require a
separate full numerical parity decision; documentation, tests, artifacts, and
qualification orchestration do not automatically trigger 60 model fits.
