# AlphaLattice

English | [简体中文](README.zh-CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](docs/public-source/getting-started.md)
[![uv](https://img.shields.io/badge/uv-locked-261230.svg)](docs/public-source/getting-started.md)
[![Mentioned in Awesome](https://awesome.re/mentioned-badge.svg)](https://github.com/wilsonfreitas/awesome-quant)

**Ask for a dated research portfolio and inspect its positions, source evidence and recorded review in one workspace.**

![An edited example run from a recorded request through activity to dated research positions.](docs/assets/readme/sentence-to-result.gif)

*Example run on synthetic data: a request to inspect stored results, their activity and the dated positions, shown as an edited sequence of cropped views.*

Tell your Claude Code or Codex agent what you want to investigate. It uses
AlphaLattice to run quantitative studies, check source evidence and bring the
results to a research committee. Follow the work and its results in your coding app's
built-in browser.

Research calculations use the information cutoff stated in each result. The market
list comes from current listings, and the report explains the survivorship limit
and limits in older sector and price records. Verification runs check for
look-ahead. The committee can decline to endorse a result. AlphaLattice runs
locally and is open source.

## 0.2.3, now

- Network-permission questions have stable codes beside their English and Chinese words, with the question's scope and reply kept in the activity record.
- A Goal's time profile shows recorded requests, Task stages and gaps. Missing observations stay unobserved; elapsed time without an attribution is not called agent time.
- On Claude Code, the first start can ask you once to run the offered command in your own terminal. Codex detaches the Host on its own. This startup step is separate from the network-permission question.

See the [0.2.3 release notes](docs/public-source/release-notes.md#023) for the message catalog and the limits of these diagnostics. The versioned notes keep each measured workload and its limits.

Use Claude Code or Codex as the lead. To see what a run used, open a session's
**Participants** in the Workbench (**Team**, then **Sessions**): it shows the models and
tokens each member used, and marks any gaps.

Give your agent this repository's link and one intent:

> Use https://github.com/xinglin-li/alphalattice to build me a reviewed Trend Rebound strategy from public data and show its dated research positions; open the Workbench in your built-in browser so I can follow it.

## Two ways to use AlphaLattice

1. **First use, in a new folder.** Send your Claude Code or Codex agent the
   intent above, or `Positions for <date>, reviewed.` The agent prepares data
   and the strategy, produces that date's research positions and follows them
   through review, committee discussion and a report.

   **Expect a pause for your answer.** The agent stops once to ask whether
   AlphaLattice may keep this workspace's network on for daily updates, fetching
   market data and SEC filings within the default budget (3 per issuer,
   600 documents and 3 GB overall). **The work waits until you answer yes or no.**
   Your yes stays in force until you tell it to stop; each network use is disclosed
   and recorded in activity, including automated updates.

   At the first-use report, the agent also asks whether daily updates should
   convene the investment committee. Decide it in the conversation, then or at any
   later time, and change it the same way, for example "Stop convening the
   committee on daily updates." Until you say yes, later updates get the portfolio
   manager's view on the reviewed positions. The first use does not wait for this
   answer.

2. **Daily update, in a new session in the same folder.** Say
   `Positions for <next session>, reviewed.` The agent uses the installed strategy
   and your standing network yes to update data and produce that session's reviewed
   research positions, following your daily committee setting. It works in your
   first use's workspace, so a later date never starts a second first use, and it
   asks you only about decisions that are yours.

## See an example run

![The Home view of an example research workspace, with dated positions and their provenance.](docs/assets/readme/home-1280.png)

*Example run on synthetic data: Home brings the positions date and the result's provenance together.*

![A saved risk study with its input window and diagnostics in the Workbench.](docs/assets/readme/research-1280.png)

*Example run on synthetic data: a saved risk study shows its input window and diagnostics.*

![The activity timeline of an example research workspace.](docs/assets/readme/activity-1280.png)

*Example run on synthetic data: the activity record keeps the work behind the result.*

<details>
<summary>The same example run at a narrower width</summary>

![The example Home view at a 900-pixel browser width.](docs/assets/readme/home-900.png)

*Example run on synthetic data at a browser width of 900 pixels.*

</details>

## Before you start

- **Environment:** Windows is the supported platform. Setup uses Python 3.12, uv
  and locked dependencies, with Claude Code or Codex. On Linux and macOS, CI checks
  installation and an offline first use; other use there is not guaranteed. On
  macOS, install the OpenMP runtime:

  ```bash
  brew install libomp
  ```
- **Time:** In the 0.2.3 release run with a Codex lead, first use took **7,097.5
  seconds** (about **118 minutes**, including a recorded pause); the first daily
  did not finish within the measurement window. The 0.2.2 Claude Code (Haiku) run
  took **121.8 minutes** for first use and **43.6 minutes** for the first daily.
  These are single runs, not guarantees.
- **Disk and memory:** The 0.2.3 first-use Host peaked at **8.66 GB**, above the
  **8 GB** budget. The 0.2.2 workspace was **8.70 GB** after first use; its first
  daily also filled a one-time **4.27 GB** backup. Allow headroom for the operating system and
  coding app. The [release notes](docs/public-source/release-notes.md#023) give
  each run's workload and limits; MB and GB use decimal units.
- **Package downloads:** Locked retrieval packages install under the first use's
  network consent, including on a machine with an empty package cache. That
  consent covers installation without a separate download question.
- **Agent costs:** Your Claude Code or Codex access and usage costs are separate
  from AlphaLattice; check your provider's terms.

See [Getting started](docs/public-source/getting-started.md) and the
[setup procedure](.agents/skills/alphalattice-research/references/operating.md)
for installation, downloads and the permissions each step needs.

Already using AlphaLattice? Follow [Upgrading](docs/public-source/upgrading.md)
before installing a later 0.2.x patch in your existing workspace.

## Why AlphaLattice

- **A complete research workflow.** Data preparation, quantitative studies,
  portfolio construction, Evidence review and committee discussion lead to a
  delivery report in one workspace.
- **Evidence you can inspect.** Read source passages, information cutoffs and
  review coverage for the reviewed portfolio.
- **Disagreement you can read.** Objections, limits and the committee's final
  ruling remain in the record and report.
- **An agent that learns from your corrections.** When a goal completes, the
  agent offers to record what the work taught it. A correction you give is kept
  in your own words, with why it matters and how to apply it. These notes live
  in your own layer, `.alphalattice/user/`, which releases never overwrite. The
  next session reads them before it starts.
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
- **When permission ends.** The first-use delegation ends when the goal's
  submission is accepted, the goal is abandoned or its 24 hours expire. Network
  access opened only under that delegation ends with it. The separate standing
  permission is described in [Two ways to use AlphaLattice](#two-ways-to-use-alphalattice).
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

## This release and where it is going

**0.2.3 is the current release.** It adds coded network-permission questions,
a shared catalog for new bilingual messages, recorded Goal profiles with gaps,
and a check that a detached Host survives its launching session. The
[release notes](docs/public-source/release-notes.md#023) distinguish the delivered
behaviour, observed workloads and remaining limits.

Each minor version's scope is planned in advance. Its release notes state the
checks run, measured workloads and unproved limits; patches in between make the
current version smoother. Each version builds on
methods already proven elsewhere, adapted to AlphaLattice's own rules: dated
inputs, sealed results and your authority.

The next 0.2.x patches, in order. Each holds one theme, and each publishes its
measured time, disk and memory against a stated budget:

- **0.2.4: a smoother daily update.** A shorter wrap-up for the agent, a measured time
  for each stage, and a background service that stays up between sessions.
- **0.2.5: a measured baseline.** A first use and a daily, replayed without a model,
  report each stage's time and the program's peak memory in minutes; later speed and
  memory changes are measured against it.
- **0.2.6: memory within budget.** Bring first use and daily updates within 8 GB.
- **0.2.7: realistic replay.** Replay real-world events (corporate actions, late
  reports, provider outages) through the real data client and its cache.
- **0.2.8: data checks.** Check the health of the data every day, in the program and
  the Workbench.
- **0.2.9: nothing stored twice.** A storage data model; upgrades reuse unchanged data.
- **0.2.10: caching and checking, done once.** Hash once, verify at trust boundaries
  and recompute only what changed, borrowing from Git's index and Bazel's action
  cache.
- **0.2.11: less waiting on the program.** The program prepares likely next steps
  while the agent is thinking.
- **0.2.12: history you can query by date.** Ask what was known, tried and decided on
  a past date, with nothing learned after it.
- **0.2.13: an evaluation flywheel.** One runner and fixed gates check every patch
  from the product's own records.
- **0.2.14: a first result in ten minutes.** A short trial on a small sample, labelled
  as a trial.
- **0.2.15: ground for daily supervision.** A store for the forward record of
  positions.

Later minor versions:

- **0.3 Daily supervision: "watch my strategy; tell me only what matters."**
  - Factor monitors in Alphalens' terms: IC, decay and turnover, on top of the
    daily data-health checks.
  - Drift is flagged by CUSUM and EWMA control charts, so quiet days stay quiet.
  - At most one digest a day.
  - A forward record of positions as published, kept apart from the backtest on GIPS
    principles.
  - Agents' calls are scored by hit rate and the Brier score.
  - The daily run is scheduled by the operating system's own scheduler.
- **0.4 Agent research: "search for a better model, honestly."**
  - Overfitting guards come before any automated search: every result reports its
    trial count, with the deflated Sharpe ratio and the probability of backtest
    overfitting, a locked holdout and purged, time-ordered validation.
  - Then search with Optuna, models in scikit-learn's estimator form, and Features
    named after the 101 Formulaic Alphas.
- **0.5 Your own strategy: "use my tickers and schedule; add my risk model or data
  source."** Extensions join by registry entry. Risk methods take the
  `sklearn.covariance` form, and data sources an Airbyte-style connector contract:
  spec, check, discover, read.
- **Later, proposed:**
  - macro data as it was known at the time (ALFRED vintages), company fundamentals
    as filed with the SEC, and consented news;
  - survivorship-free research on a CRSP-style data model;
  - several strategies as one portfolio, by risk budgeting and hierarchical risk
    parity;
  - committee adjustments within your bounds, in Black-Litterman form, built only if
    the forward record shows the committee adds value;
  - a stable 1.0 that keeps every agent-facing surface working;
  - intraday schedules.

Previews of the next minor version land on the `next` branch first. Tell us what you
need in an [issue](https://github.com/xinglin-li/alphalattice/issues): requests from
people using AlphaLattice decide what comes after 0.5.

## Tell us where you got stuck

[Open an issue](https://github.com/xinglin-li/alphalattice/issues) with the command
you were running, what you expected and the answer you received. Redact private
data and credentials. For proposed code changes, follow
[CONTRIBUTING](CONTRIBUTING.md): open an issue first for nontrivial work, and sign
the contributor agreement on your first pull request.
For a security vulnerability, report privately as described in [SECURITY](SECURITY.md).

## About

Last checked: 2026-10-22 against 0.2.3

The [design principles](docs/public-source/design-principles.md) explain how the
agent, the software and the person share the work.

Created and maintained by Xinglin Li: [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com).
See [governance](GOVERNANCE.md) for project decisions and contribution credit,
and [contributing](CONTRIBUTING.md) for the agreement and workflow.

## Verify a source checkout

After `uv sync --locked --all-extras`, build Local Web with
`uv run python scripts/build_local_web_ui.py --product` before running the
[public checks](CONTRIBUTING.md). Tests check that build at session start,
rebuilding once if it is missing or stale; a failed build fails the run.

## Development history

AlphaLattice has been developed privately since 2026-08-06. Public commits carry
the shipped changes exported from that development; releases publish the
resulting versions.

## Disclaimer

AlphaLattice produces research results, not orders, and places no trades.
Historical research is not investment advice or a promise of future returns.
A completed review records findings and limits; it is not a recommendation to
trade or a guarantee of scientific validity.

The software is licensed under [Apache-2.0](LICENSE); [NOTICE](NOTICE) records
third-party material. Acquired data and downloaded model packs retain their
own [terms](docs/public-source/third-party-and-data.md). The software licence
grants no data licence. Agent access and usage costs follow your provider's terms.
