# Read a result's temporal statement
Date: 2026-10-03

A result carries temporal marks from its Panel and generated sentences. Read them with the evaluated window and history used by the inputs. A later data update does not improve an earlier sealed result. Missing temporal evidence is `NOT_RECORDED`; do not infer it from a result date or invent a statement for an older record.

| Result | Location |
| --- | --- |
| Experiment readback | `timing.temporal_scope` |
| Feature build/trial | `temporal_scope` |
| Installed Portfolio report | `reading_context.temporal_scope` |

Use the saved full answer or `--view full`; compact CLI display can omit `timing`, while `--output` keeps the full answer.

## Membership and evaluated window

`t0_session` is the last session of the workspace's first qualified Panel. The initial cohort stands in for earlier membership; observed entries and exits supply forward history from T0. This is approximate point-in-time coverage with disclosed pre-T0 backfill.

| Field | Meaning |
| --- | --- |
| `initial_cohort_size` | Initial listings used for backfill |
| `window_start`, `window_end` | Evaluated result window |
| `data_start` | Earliest input session, including support before the window |
| `window_before_t0` | Whether evaluation begins before T0 |
| `data_before_t0` | Whether inputs reach before T0 |
| `universe_basis` | Recorded membership basis; backfill uses `INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME` |
| `survivorship_bias` | Whether the cohort policy carries survivorship bias |

When T0 or a start is missing, the related before-T0 flag is `null`. The statements distinguish a pre-T0 window from a later window whose inputs reach earlier history. A cohort gathered from listings still alive at collection carries survivorship bias; later membership observations do not repair its backfill.

## Sector history

The result field is `sector_treatment`; the Panel field is `sector_history_treatment`.

| Value | Meaning |
| --- | --- |
| `CURRENT_CLASSIFICATION_BACKFILLED` | One recorded current classification is used for every session; the backfill is not point in time. |
| `FIRST_RECORDED_CLASSIFICATION_BACKFILLED_THEN_AS_OBSERVED` | Each listing's first recorded classification is backfilled until its first reclassification; later values start on their effective sessions. |

`sector_observed_on`, `sector_reclassification_count` and `sector_first_reclassified` report the Panel's observation and captured changes. A refresh is effective on its exchange trading day, or the next session when there is none, bounded by the first session after the last published Panel. It changes no published session. Aggregates, outcomes and result context use the Sector in force on each session. The generated sentence does not claim complete historical classification.

## Price and unknown marks

`price_basis: split_adjusted` uses the provider's current split basis; adjusted close is back-adjusted for later splits/dividends, so these are not prices as observed on each historical day. `unadjusted` uses as-traded prices/volumes and projects only splits known by each session. Scale-invariance checks do not make split-basis data point in time. `_pit` formula leaves reconstruct as-traded values re-based to the formula session; controls state the provider-volume assumption before fractional price adjustment. A new or unavailable mark remains named in `statements`, not silently dropped.

Source contracts: [TemporalStatement](../../src/alphalattice/foundation/feature_engine/publication/temporal_statement.py), [readback timing](../../src/alphalattice/control/product_host/research_authoring/timing.py), [Sector vocabulary](../../src/alphalattice/kernel/shared_kernel/sector_treatment.py) and [effective-session rule](../../src/alphalattice/foundation/market_data_ops/sources/sector_forward.py).
