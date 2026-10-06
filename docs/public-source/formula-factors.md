# Author a formula factor
Date: 2026-10-03

A formula factor adds a declared expression to the research Feature catalog. The installed kernel parses and computes it; the author explicitly selects preprocessing. Plan, build, trial, review and a person's activation are distinct records. Keep their exact input, plan and Task references. This guide owns the numerical language; see [Extending](extending.md) for the short workflow overview.

## Declaration and formula contract

Use a qualified input binding and the controls' template. The answer gives `request_schema`, `definitions`, `base_revision` and `formula_language`; copy the template to an editable Feature change document and retain `input_binding_hash`, `base_revision_hash` and optional `parent_plan_hash`. A `CREATE` edit supplies a new `factor_id`, a complete `FactorSpec` and `preprocessing_recipe`. State its expression, family, economic skip (`lag_sessions`), literature, tolerances and track. Planning canonicalizes the formula and derives `formula_ref`, required fields, window, minimum observations and return convention. Changing a kept factor's tolerances is refused as `feature_research.validation_policy_change_not_admitted`.

Split-basis leaves are `open`, `high`, `low`, `close`, `volume` and `adjusted_close`; prices and volumes use the provider's current split basis. Their powers must preserve scale through later splits/dividends: price and volume exponents must match and adjusted-close exponent must be zero. Ratios and returns can pass; raw price levels or their logarithm fail as `factor_formula.not_scale_invariant`.

`open_pit`, `high_pit`, `low_pit`, `close_pit` and `volume_pit` are point-in-time leaves reconstructed as traded and re-based to the formula's session. Prices use recorded adjustment ratios; share volume uses splits. The provider's volume behavior before a fractional price adjustment cannot be verified offline; controls disclose the assumption. Point-in-time leaves admit level forms, but mixed expressions retain the scale rule for split-basis leaves.

`formula_language.operators` is authoritative. Element operations include arithmetic, comparisons, logic, `abs`, `log`, `sqrt`, `sign`, `clip`, `min`, `max` and `where`; per-listing series operations include `lag`, `delta`, `ts_*` reductions/statistics and `ewm`. A time-series window reads only that listing's own rows and no later row. The derived window plus `lag_sessions` must fit the 276-row kernel budget, including the formula's own row. Missing input, divide-by-zero and invalid log/square-root domains yield missing values. Cross-sectional `rank`, `zscore`, `demean`, `sector_demean` and `winsorize` belong to preprocessing, not the expression (`factor_formula.section_is_preprocessing:<operator>`). This bounded grammar accepts no Python imports or callables.

## Preprocessing and research-only leaves

| Recipe | Treatment and use | Daily admission |
| --- | --- | --- |
| `ROBUST_SECTOR_NEUTRAL_Z` | Median/MAD winsorization, equal-Sector demeaning and global robust Z; use for a structural Sector level such as liquidity/turnover | Eligible if other checks pass |
| `ROBUST_UNIVERSE_Z` | Same robust transform with one universe group, retaining Sector component for returns/moves | Refused; research overlay |
| `TIME_SERIES_ABSOLUTE_STATE_ROBUST` | Standardize each listing against its trailing year | Refused; research overlay |

The recipe is required and must be listed; non-formula factors keep their installed preprocessing. `sector_return` is the equal-weight Sector log return for that session's classification. Its plan uses `factor.formula.sector`; point-in-time formulas use `factor.formula.point_in_time` or `factor.formula.sector.point_in_time` when both leaves are used. Market-return and registered-factor leaves are not admitted. Sector-return factors are research-only until the daily build carries the Sector child (`feature_extension.sector_leaf_research_only:<factor_id>`). Daily Panels carry as-traded fields when their catalog reads them, so a point-in-time factor without a Sector leaf can be admitted with an eligible recipe. See [Temporal statements](temporal-statements.md) for the history limits.

## Trial and review

Plan the Feature change, build its exact input overlay (`PREPROCESSED_VALUES` stays in local research artifacts), then choose an exact compatible baseline for the trial. That baseline is a completed Alpha study handed off from Factor evidence on the same input, or a Portfolio built on that Alpha; a bare Factor study is only the start of curation/handoff. The trial performs screening, curation, the baseline Alpha at its existing scale and, if declared, its Portfolio policy. It reuses exact existing work and retains its Tasks. For `feature_trial.study_not_from_factor_evidence`, the refusal offers up to five eligible completed studies on the input; if none exist it offers completed Factor studies for curation/handoff, otherwise the study listing. An existing build stays intact. Do not relax the baseline.

Read `standing` in this order: comparison, execution, contract, evidence, activation; `statements` explain each mark and `reasons` carries refusal codes. `COMPLETED` alone is not activation; `NOT_COMPARED` claims no change. Read `outcome` and `comparison` for rank IC/correlation and Alpha/Portfolio comparisons where applicable. `FEATURE_NOT_ADMITTED_BY_SCREENING` gives no downstream improvement claim.

The exact review packet covers declaration/recipe, independent golden checks and tolerances, identity bindings, build coverage/missing share, trials and formula search counts by workspace/goal. `active_panel.admitted` and `reason` state daily admission. `next_requests.activate` is offered only after a passed contract, completed trial and admitted daily recipe. Its standing uses the latest completed trial (or latest trial if none completed) and is part of `packet_hash`. `reasons.activation` gives a hold/refusal code. Gate order holds shipped factors, failed contracts, missing completed trials, then definitions the daily Panel does not admit. Activation does not require `comparison: COMPARED` or a positive effect; `A_PERSON_MAY_ACTIVATE` leaves the choice with the person.

`FEATURE_ACTIVATE` and `FEATURE_DEACTIVATE` require the person in Local Web; agent/CLI attempts are refused as `feature_extension.human_confirmation_required`. Activation records the reviewed packet, methodology and trial; the next data update rebuilds the daily Panel with the catalog entry. It changes no kernel code identity. Deactivation removes the active entry, while sealed Panels and studies keep their exact definitions and values.

For a goal deliverable, cite the exact trial readback (`FEATURE_TRIAL_READBACK`, `feature_trial_id`) and review (`FEATURE_REVIEW`, `feature_plan_hash`, `feature_factor_id`) under `DATA_FEATURES`. The Host rereads them and the standing still limits any claim.
