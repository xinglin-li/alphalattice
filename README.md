# AlphaLattice

**Turn an investment idea into a reviewed research portfolio.**

Tell your Codex or Claude Code agent what you want to investigate. It uses
AlphaLattice to run quantitative studies, check source evidence and bring the
results to a research committee. Follow the work and its results in your browser.

Give your agent this repository's link and one intent:

> Use https://github.com/xinglin-li/alphalattice to build me a reviewed installed strategy from public data and show its dated research positions; open the Workbench in your browser so I can follow it.

<!-- Hero image reserved: Codex conversation on the left, Workbench on the right. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-performance-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-performance-light.png">
  <img src="docs/images/research-performance-light.png" alt="Trend Rebound Book historical research performance from 2023-10-02 to 2026-09-08." width="1280">
</picture>

*Historical research performance for Trend Rebound Book, 2023-10-02 to 2026-09-08; this is not a record of executed trades. Historical research is not investment advice or a promise of future returns.*

[Getting started](docs/public-source/getting-started.md) covers installation,
first use and the decisions that remain yours.

## Why AlphaLattice

- **A complete research workflow.** Data preparation, quantitative studies,
  portfolio construction, Evidence review and committee discussion lead to a
  delivery report in one workspace.
- **Evidence you can inspect.** Read source passages, information cutoffs and
  review coverage for the reviewed book.
- **Disagreement you can read.** Objections, limits and the committee's final
  ruling remain in the record and report.
- **Factors and models you can extend.** Add factors and Alpha models through
  their registries and checked [extension paths](docs/public-source/extending.md).

## Before you start

- **Environment:** Windows is verified; macOS and Linux are unverified. Setup
  uses Python 3.12, uv and locked dependencies. Bring Codex or Claude Code.
- **Time:** First preparation and model training take time. One measured run
  used prepared public-source data and retained filings, offline, with four
  CPU cores on a shared Windows machine. It took 63 minutes from the start of
  the first data preparation Task to writing the delivery report. Installation
  and live downloads were outside that interval; this is an indicative example,
  not a runtime guarantee.
- **Software licence:** AlphaLattice is [Apache-2.0](LICENSE).
- **Data and model terms:** Acquired data and downloaded model packs keep their
  own [terms](docs/public-source/third-party-and-data.md); the software licence
  grants no data licence.
- **Agent costs:** Your Codex or Claude Code access and usage costs are separate.

See [Getting started](docs/public-source/getting-started.md) and the
[setup procedure](.agents/skills/alphalattice-research/references/operating.md)
for details. Downloads need applicable consent; measured time and memory are
observations, not runtime guarantees.

## A worked example

An earlier completed run began with this request:

> Build me a reviewed Trend Rebound book from public data and give me positions for 2026-09-11.

**One sentence → studies → dated positions → Evidence review → committee
discussion → report.**

The agent prepared the data, ran the strategy's model and Risk studies, built
the book and delivered research positions for the requested date. Evidence
Analysts read the prepared source material; the CRO assessed their findings;
Alpha, Risk, CRO and the lead discussed the dated result. The delivery included
positions, Risk, review coverage, citations and the members' recorded views.

Reservations remained: the review covered only the admitted filings available
at its cutoff, and members named limits in predictive support, liquidity and
capacity evidence. A completed review did not clear those limits or authorize
trades.

<!-- Committee discussion image reserved: use the final 0.2.0 acceptance run, not the earlier run's committee or verdict wording. -->
*Committee discussion image: reserved for the final 0.2.0 run.*

<!-- Example report link reserved: use the same final 0.2.0 acceptance run as the committee discussion image. -->
*Example report: link reserved for the final 0.2.0 run.*

### Historical research views

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-positions-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-positions-light.png">
  <img src="docs/images/research-positions-light.png" alt="Trend Rebound Book historical sector exposure and replay holdings on 2026-09-08." width="1280">
</picture>

*Sector exposure and historical replay weights for 2026-09-08; these are not executed trades.*

[Research flows](docs/public-source/research-flows.md) explains studies, installed
strategies and review. **Rebound Return Book** and **Trend Rebound Book** are
installed strategies, distinct from research study books. Read each position's
basis, information cutoff, first actionable session and horizon. Calculation
alone grants no activation or publication.

## Follow the work

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/research-conversation-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/research-conversation-light.png">
  <img src="docs/images/research-conversation-light.png" alt="An earlier run's session conversation, showing Evidence review assignments, answers and accepted submissions." width="1280">
</picture>

*An earlier run's recorded Evidence review assignments, answers and acceptances.*

Use **Goals** for Timeline, Conversation and Results, **History** for saved
studies and reviews, and **Team** for recorded sessions and usage. Acceptance
alone establishes no scientific validity or specialist authorship.

Local Web includes Chinese; CLI detail uses `--lang zh`, for example
`alphalattice workspace show --lang zh`, with English fallback.

Read result standing, coverage and time limits, including survivorship, Sector
backfill and historical source availability. A needed decision names its issue
and page.

The agent installs AlphaLattice and continues in the same session. Its specialists'
tool limits hold by instruction. For limits enforced by the host, open a checkout
session with `cd <checkout>; claude` in Claude Code or `codex -C <checkout>` in
Codex. The [agent guide](AGENTS.md) gives the procedure.

Your exact sentence opens the first-use goal. Its 24-hour delegation lets the
agent handle admitted preparation and activate its reviewed book, reporting
each act. Deactivate on **Portfolio**; stopping the goal and cancelling a Task
are separate. The guide names [what only you decide](AGENTS.md#what-only-a-person-decides).
Forward positions are research, not orders or advice.

## Guides and care

| Guide | What it explains |
| --- | --- |
| [Getting started](docs/public-source/getting-started.md) | Prepare for installation and first use. |
| [CLI](docs/public-source/cli.md) | Read answers and continue saved requests. |
| [Research flows](docs/public-source/research-flows.md) | Follow studies, books and review. |
| [Agents](docs/public-source/agents.md) | Follow goals, sessions and specialist work. |
| [Extending](docs/public-source/extending.md) | Develop and inspect factors and models. |
| [Data and care](docs/public-source/data-and-care.md) | Update, inspect, back up and restore a workspace. |
| [Privacy](docs/public-source/privacy.md) | Read what usage records contain and disable their reader. |
| [Downloads and data terms](docs/public-source/third-party-and-data.md) | Read the terms for models and data sources. |

The [guide index](docs/public-source/README.md) links the other reference pages.

Use [backup and restore](docs/public-source/data-and-care.md) into a new directory
and keep sealed records intact. [Privacy](docs/public-source/privacy.md) explains
recorded usage and its switch.

## Tell us where you got stuck

[Open an issue](https://github.com/xinglin-li/alphalattice/issues) with the command
you were running, what you expected and the answer you received. Redact private
data and credentials. For proposed code changes, follow
[CONTRIBUTING](CONTRIBUTING.md), including its current contribution-opening
conditions and agreement.
For a security vulnerability, report privately as described in [SECURITY](SECURITY.md).

## About and licence

Documentation updated: 2026-10-13.

The [design principles](docs/public-source/design-principles.md) explain how the
agent, the software and the person share the work.

Created and maintained by Xinglin Li: [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com).
[Apache-2.0](LICENSE) covers the software; [NOTICE](NOTICE) records third-party
material. Downloaded models and acquired data keep their own terms. See
[governance](GOVERNANCE.md) for project decisions and contribution credit, and
[contributing](CONTRIBUTING.md) for the opening condition, agreement and workflow.

## Verify a source checkout

After `uv sync --locked --all-extras`, build Local Web with
`uv run python scripts/build_local_web_ui.py --product` before running the
[public checks](CONTRIBUTING.md). Tests check that build at session start,
rebuilding once if it is missing or stale; a failed build fails the run.

## Development history

AlphaLattice has been developed privately since 2026-08-06. Each public commit
contains one shipped change exported from that development; releases publish
the resulting versions.

The public repository starts at 0.1.x; earlier development happened privately.
