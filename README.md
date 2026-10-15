# AlphaLattice

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](docs/public-source/getting-started.md)
[![uv](https://img.shields.io/badge/uv-locked-261230.svg)](docs/public-source/getting-started.md)

**Turn an investment idea into a reviewed research portfolio.**

Tell your Codex or Claude Code agent what you want to investigate. It uses
AlphaLattice to run quantitative studies, check source evidence and bring the
results to a research committee. Follow the work and its results in your browser.

Numerical inputs are bounded by the date's cutoff, with a grader checking for
look-ahead in verification runs. The committee can decline endorsement;
AlphaLattice runs locally and is open source.

Give your agent this repository's link and one intent:

> Use https://github.com/xinglin-li/alphalattice to build me a reviewed Trend Rebound strategy from public data and show its dated research positions; open the Workbench in your browser so I can follow it.

<!-- Hero image reserved: committee discussion from the final acceptance run. -->

[Getting started](docs/public-source/getting-started.md) covers installation,
first use and the decisions that remain yours.

## Why AlphaLattice

- **A complete research workflow.** Data preparation, quantitative studies,
  portfolio construction, Evidence review and committee discussion lead to a
  delivery report in one workspace.
- **Evidence you can inspect.** Read source passages, information cutoffs and
  review coverage for the reviewed portfolio.
- **Disagreement you can read.** Objections, limits and the committee's final
  ruling remain in the record and report.
- **Factors and models you can extend.** Add factors and Alpha models through
  their registries and checked [extension paths](docs/public-source/extending.md).

## A worked example

A completed run asked for a reviewed Trend Rebound strategy from public data,
with research positions for 2026-09-11.

1. **Studies.** The agent prepared public-source data and ran the strategy's
   model and Risk studies.
2. **Dated positions.** It built the portfolio and delivered research positions
   for the requested date, with their Risk assessment.
3. **Evidence review.** Evidence Analysts read the prepared source material
   available at the information cutoff. The chief risk officer (CRO) assessed
   their findings.
4. **Committee discussion.** Alpha, Risk, CRO and the lead discussed the dated
   result, keeping objections and the lead's rulings in the record.
5. **Report.** The delivery brought together positions, Risk, review coverage,
   citations and the members' recorded views.

The report kept the reservations alongside the results: the review covered
the filings accepted by its source checks and available at its cutoff, and
members named limits in predictive support, liquidity and capacity evidence.

<!-- Example report link reserved: use the same final acceptance run as the hero image. -->

### Historical research views

A **book** is a recorded research portfolio with its holdings and results.
For an installed strategy, it records the strategy's historical replay.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-performance-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-performance-light.png">
  <img src="docs/images/research-performance-light.png" alt="Trend Rebound Book historical research performance from 2023-10-02 to 2026-09-08." width="1280">
</picture>

*Historical research performance for Trend Rebound Book, 2023-10-02 to 2026-09-08, from a replay of the strategy.*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-positions-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-positions-light.png">
  <img src="docs/images/research-positions-light.png" alt="Trend Rebound Book historical sector exposure and replay holdings on 2026-09-08." width="1280">
</picture>

*Sector exposure and historical replay weights for 2026-09-08.*

**Rebound Return Book** and **Trend Rebound Book** are installed strategies:
ready-to-use strategy definitions that can be activated to produce dated
positions. They are separate from exploratory research study portfolios.
[Research flows](docs/public-source/research-flows.md) explains both routes.

Read each result's information cutoff, coverage and limits on its use.
For inputs built from a current membership snapshot, earlier membership is
backfilled from the initial set of securities; later additions and removals
follow recorded observations. This is not a complete reconstruction of past
index membership. Read the reported survivorship limits, sector backfill and
historical source availability alongside the performance.

## Follow the work

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-conversation-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-conversation-light.png">
  <img src="docs/images/research-conversation-light.png" alt="An earlier run's session conversation, showing Evidence review assignments, answers and accepted submissions." width="1280">
</picture>

*An earlier run's recorded Evidence review assignments, answers and acceptances.*

- **Workbench.** Use **Goals** for the timeline, conversation and results,
  **History** for saved studies and reviews, and **Team** for sessions and
  usage. Review pages show which checks passed, what remains missing and how
  the result may be used.
- **First-use permission.** Your sentence opens a goal and gives the agent
  permission for specific steps for 24 hours from the goal's opening: prepare
  data, handle data issues within the declared rules, open the preparation's
  network, set up Evidence within the default budget and activate the strategy's
  completed book. It reports each act.
- **When permission ends.** It ends when the goal's submission is accepted,
  when the goal is abandoned or when the 24 hours expire. The network access
  opened under that permission ends with it.
- **Your controls.** Stop the strategy through your agent or on **Portfolio**.
  Stopping a goal and cancelling a Task are separate controls. The guide lists
  [what only you decide](AGENTS.md#what-only-a-person-decides).
- **Host boundaries.** The agent installs AlphaLattice and continues in the
  same session. Specialist tool limits apply through their instructions;
  checkout sessions also load the host's project configuration. Start one with
  `cd <checkout>; claude` in Claude Code or `codex -C <checkout>` in Codex, and
  follow the [agent guide](AGENTS.md) for the host's setup and checks.
- **Languages.** The Workbench offers English and Chinese. CLI detail uses
  `--lang zh`, for example `alphalattice workspace show --lang zh`, with English
  fallback.

## Before you start

- **Environment:** Windows and Linux have completed full agent runs; Linux was
  verified on Ubuntu 24.04 under WSL2. Setup uses Python 3.12, uv and locked
  dependencies, with Codex or Claude Code. On macOS, CI checks installation and
  an offline first use; a full agent run on a Mac is still pending. On macOS,
  install the OpenMP runtime:

  ```bash
  brew install libomp
  ```

  See [Getting started](docs/public-source/getting-started.md) for installation
  and first-use instructions.
- **Time:** Data preparation, multi-year model fitting and validation, and
  source review take time. One completed run using prepared public-source data
  and retained filings, replayed offline with four CPU cores on a shared Windows
  machine, took about **37 minutes** from the start of the first data preparation
  Task to writing the delivery report. Installation and live downloads were outside
  that interval. Data and model-pack downloads add time when needed, depending
  on your connection. This is an observation, not a runtime guarantee.
- **Agent costs:** Your Codex or Claude Code access and usage costs are separate
  from AlphaLattice; check your provider's terms.

See [Getting started](docs/public-source/getting-started.md) and the
[setup procedure](.agents/skills/alphalattice-research/references/operating.md)
for installation, downloads and the permissions each step needs.

## Guides and care

| Guide | What it explains |
| --- | --- |
| [Getting started](docs/public-source/getting-started.md) | Prepare for installation and first use. |
| [CLI](docs/public-source/cli.md) | Read answers and continue saved requests. |
| [Research flows](docs/public-source/research-flows.md) | Follow studies, portfolios and review. |
| [Agents](docs/public-source/agents.md) | Follow goals, sessions and specialist work. |
| [Extending](docs/public-source/extending.md) | Develop and inspect factors and models. |
| [Data and care](docs/public-source/data-and-care.md) | Update, inspect, back up and restore a workspace. |
| [Privacy](docs/public-source/privacy.md) | Read what usage records contain and disable their reader. |
| [Downloads and data terms](docs/public-source/third-party-and-data.md) | Read the terms for models and data sources. |

The [guide index](docs/public-source/README.md) links the other reference pages.

Use [backup and restore](docs/public-source/data-and-care.md) into a new directory
to preserve the saved records and their integrity checks. [Privacy](docs/public-source/privacy.md)
explains recorded usage and its switch.

## Tell us where you got stuck

[Open an issue](https://github.com/xinglin-li/alphalattice/issues) with the command
you were running, what you expected and the answer you received. Redact private
data and credentials. For proposed code changes, follow
[CONTRIBUTING](CONTRIBUTING.md), including its current contribution-opening
conditions and agreement.
For a security vulnerability, report privately as described in [SECURITY](SECURITY.md).

## About

Documentation updated: 2026-10-14.

The [design principles](docs/public-source/design-principles.md) explain how the
agent, the software and the person share the work.

Created and maintained by Xinglin Li: [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com).
See [governance](GOVERNANCE.md) for project decisions and contribution credit,
and [contributing](CONTRIBUTING.md) for the opening condition, agreement and workflow.

## Verify a source checkout

After `uv sync --locked --all-extras`, build Local Web with
`uv run python scripts/build_local_web_ui.py --product` before running the
[public checks](CONTRIBUTING.md). Tests check that build at session start,
rebuilding once if it is missing or stale; a failed build fails the run.

## Development history

AlphaLattice has been developed privately since 2026-08-06. Each public commit
contains one shipped change exported from that development; releases publish
the resulting versions.

## Disclaimer

AlphaLattice produces research results, not orders, and places no trades.
Historical research is not investment advice or a promise of future returns.
A completed review records findings and limits; it is not a recommendation to
trade or a guarantee of scientific validity.

The software is licensed under [Apache-2.0](LICENSE); [NOTICE](NOTICE) records
third-party material. Acquired data and downloaded model packs retain their
own [terms](docs/public-source/third-party-and-data.md). The software licence
grants no data licence. Agent access and usage costs follow your provider's terms.
