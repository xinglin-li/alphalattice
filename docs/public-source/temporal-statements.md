# Read a result's temporal statement
Date: 2026-10-08

Read the Panel's temporal marks with evaluation and input history. Later data never improves an earlier sealed result. Missing evidence stays `NOT_RECORDED`; infer no historical statement from a result date.

| Result | Full-answer location |
| --- | --- |
| Experiment | `timing.temporal_scope` |
| Feature build or trial | `temporal_scope` |
| Installed Portfolio report | `reading_context.temporal_scope` |

Use the saved full answer or `--view full`; `--output` retains fields compact display can omit.

## Membership and evaluation

`t0_session` ends the first qualified Panel. Its initial cohort backfills earlier membership; observed entries and exits supply forward history. Pre-T0 coverage is approximate, not complete point-in-time membership. A cohort collected from surviving listings carries survivorship bias that later observations cannot repair.

| Field | Read as |
| --- | --- |
| `initial_cohort_size` | Backfilled listing count |
| `window_start`, `window_end` | Evaluated window |
| `data_start` | Earliest input session, including lookback and training |
| `window_before_t0`, `data_before_t0` | These flags show whether the evaluated window and input history start before T0. Each is `null` when a required date is missing. |
| `universe_basis` | Membership basis; `INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME` names backfill |
| `survivorship_bias` | Cohort-policy bias |

## Sector and price history

The result's `sector_treatment` comes from the Panel's `sector_history_treatment`:

- `CURRENT_CLASSIFICATION_BACKFILLED`: current classification used throughout.
- `FIRST_RECORDED_CLASSIFICATION_BACKFILLED_THEN_AS_OBSERVED`: first classification backfilled, then changes from effective sessions.

Neither promises complete historical classification. `sector_observed_on`, `sector_reclassification_count` and `sector_first_reclassified` identify observations and changes. A refresh applies on its exchange trading day or next session, bounded by the first unpublished session. Published sessions never change; calculations use each session's effective Sector.

`price_basis: split_adjusted` uses current provider split basis and adjusted-close back-adjustment for later splits and dividends. `unadjusted` uses as-traded prices and volumes and only splits known per session. Scale invariance does not make split-basis data point in time. Formula `_pit` leaves reconstruct as-traded values re-based to their session, with the controls' fractional-adjustment volume assumption. Unknown marks remain in `statements`.

Source contracts: [temporal marks](../../src/alphalattice/foundation/feature_engine/publication/temporal_statement.py), [readback timing](../../src/alphalattice/control/product_host/research_authoring/timing.py), [Sector vocabulary](../../src/alphalattice/kernel/shared_kernel/sector_treatment.py), [effective sessions](../../src/alphalattice/foundation/market_data_ops/sources/sector_forward.py).
