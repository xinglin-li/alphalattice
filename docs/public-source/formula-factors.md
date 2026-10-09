# Author a formula factor
Date: 2026-10-08

Ask for a declared expression on a qualified research input, with explicit preprocessing. The kernel parses and computes it; a plan, build, trial and review show what it means and supports. Activation on **Features** is a separate decision under the [research-agent guide](../../AGENTS.md#what-only-a-person-decides). This page describes the numerical contract.

## Formula and preprocessing

The controls provide an editable declaration and authoritative `formula_language.operators`. Specify the expression, family, economic skip (`lag_sessions`), literature, tolerances, track and recipe. Planning canonicalizes the expression and derives its fields, window, minimum observations and return convention. A kept factor's tolerances cannot be changed by this path.

Split-basis leaves are `open`, `high`, `low`, `close`, `volume` and `adjusted_close`. Provider prices and volumes use the current split basis. Price and volume exponents must match and adjusted-close exponent must be zero, preserving scale through later splits and dividends. Ratios and returns can pass; raw price levels or their logarithm fail scale invariance.

`open_pit`, `high_pit`, `low_pit`, `close_pit` and `volume_pit` reconstruct as-traded values re-based to the formula session. Prices use adjustment ratios; volume uses splits. The provider's volume behavior before a fractional price adjustment cannot be verified offline; controls disclose the assumption. These leaves admit levels; mixed expressions retain the split-basis scale rule.

The operators include arithmetic, comparisons, logic, `abs`, `log`, `sqrt`, `sign`, `clip`, `min`, `max`, `where`, `lag`, `delta`, `ts_*` statistics and `ewm`. Each time-series window counts a listing's own rows and reads no later row. The derived window and economic skip must fit within 276 rows, including the formula's own row. Missing inputs, division by zero and values outside the domain of `log` or `sqrt` yield missing values. Cross-sectional `rank`, `zscore`, `demean`, `sector_demean` and `winsorize` belong to preprocessing. The grammar admits no Python imports or callables.

The declaration must name one of the listed preprocessing recipes.

| Preprocessing recipe | Treatment | Daily admission |
| --- | --- | --- |
| `ROBUST_SECTOR_NEUTRAL_Z` | The recipe applies median and MAD winsorization, equal-Sector demeaning and global robust Z scaling to structural Sector levels. | The recipe is eligible for daily admission if the other checks pass. |
| `ROBUST_UNIVERSE_Z` | The recipe applies the robust universe transform and retains the Sector component of returns and price moves. | Daily admission is refused. The recipe remains a research overlay. |
| `TIME_SERIES_ABSOLUTE_STATE_ROBUST` | The recipe standardizes each listing against its trailing year. | Daily admission is refused. The recipe remains a research overlay. |

Non-formula factors keep installed preprocessing. `sector_return` is the equal-weight Sector log return using each session's classification. Sector-return formulas remain research-only until the daily build carries the Sector child; point-in-time formulas without it can be eligible with an admitted recipe. Market-return and registered-factor leaves are excluded. Read the [temporal limits](temporal-statements.md).

## Trial and review

The build creates an exact input overlay; preprocessed values stay in local research artifacts. A trial requires a compatible completed Alpha study handed off from Factor evidence on the same input, or a Portfolio built on that Alpha. A bare Factor study needs curation and handoff first. The trial runs screening, curation, baseline Alpha at its existing scale and any declared Portfolio policy, reusing exact completed work and retaining its Tasks.

Read standing across comparison, execution, contract, evidence and activation. `COMPLETED` does not activate; `NOT_COMPARED` claims no change, and screening refusal supports no downstream improvement claim. The review covers the declaration and recipe, independent golden checks, tolerances, identity bindings, coverage, missing share, trials and search counts by workspace and goal.

Daily activation eligibility requires a passed contract, a completed trial and an admitted recipe. Neither a completed comparison nor a positive effect is required. Inspect the review's daily admission and activation standing before deciding. Activation records the packet, methodology and trial. The next data update rebuilds the daily Panel without changing kernel identity. Deactivation removes the active entry, while sealed Panels and studies retain their definitions and values.
