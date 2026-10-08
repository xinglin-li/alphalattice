# AlphaLattice
Date: 2026-10-08

AlphaLattice is a local quantitative research workstation for people using Codex or Claude Code to investigate investment ideas.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/workbench-dark.webp">
  <img src="docs/images/workbench-light.webp" alt="AlphaLattice Workbench showing a recorded historical Portfolio report and its performance curve." width="1470">
</picture>

*A historical research replay on recorded US market data, from 2022-07-01 to 2026-09-08. The report's figures are shown as recorded; this book is not reviewed. Historical research is not investment advice or a promise of future returns.*

## What one sentence gets you

Your agent leads seven specialists: Data, Factor, Alpha, Risk, Portfolio,
Evidence Analyst and CRO. Ask it to research factors, compare models, assess
risk and build a reviewed Portfolio book. Follow the Tasks, results, source
passages and review in the Workbench.

For an installed strategy, your activation of a reviewed historical book lets
the agent run a forward update and show dated research positions. Historical
holdings are replay results; forward positions are research, not orders or advice.

## Start with one sentence

Give your Codex or Claude Code agent this repository's link and one intent:

> Use https://github.com/xinglin-li/alphalattice to build me a reviewed installed strategy from public data and show its dated research positions; open the Workbench in your browser so I can follow it.

The agent you give the link to installs AlphaLattice and carries on with your sentence in the same session. It starts the seven specialists there from their cards, so their tool limits hold by instruction; for limits your agent's host enforces, open a session in the checkout instead (Claude Code with `cd <checkout>; claude`, Codex with `codex -C <checkout>`). The [agent guide](AGENTS.md) carries the flow. On first use, your exact sentence
opens a `FIRST_USE` goal before preparation. Its 24-hour delegation covers
preparation network access, confirmation/resumes, data issues and membership
changes, and the activation of the book it reviewed, which you deactivate in one
click on **Portfolio**; other person-only decisions remain yours. You may interrupt with goal abandonment or
close network access in **Settings**; stopping a Task needs cancellation.
See [first-use details](AGENTS.md#run-the-first-use-from-the-persons-sentence).

## Setup and research

Use the [setup procedure](.agents/skills/alphalattice-research/references/operating.md#setup-and-launch) for the locked
Windows/Python 3.12/uv setup, editable checkout or wheel, PATH and Local Web
launch. **Windows is verified; macOS and Linux are unverified.** Installation, data acquisition and retrieval
model downloads have separate permission; a workspace starts with network
closed. Evidence needs its retrieval environment, local models and admitted
issuer package. The setup guide's measured time and memory are observations,
not requirements or guarantees. Adjust execution capacity, not the recipe.

The [factor research flow](docs/public-source/research-flows.md) connects Factor
curation to Alpha, parallel Risk on the same input, then a Portfolio book,
Evidence, Analyst findings and CRO review. Risk sizing consumes a completed Risk
study; a report-only association changes no weights. An empty workspace installs
no strategy. **Rebound Return Book** and **Trend Rebound Book** are installed
strategies, distinct from study books.

Read the [installed strategy's forward dates and horizon](AGENTS.md#run-an-installed-strategy-forward)
before activation or continuation. Activation binds models, required calibration
and sealed state without fitting; scheduling needs the service running.
Deactivation keeps history; a changed package needs a new book and activation.

## Read results and decide

![Goal Conversation with a completed study-check Task and a running data-update Task](docs/images/goal-conversation.webp)

*A demonstration Goal Conversation with real Host Tasks and scripted fixture messages: an empty study check finished, and a data update is running. This shows the interface, not native agent authorship or research validation.*

Use **Goals** for the checked ledger's Timeline, Conversation and Results,
**History** for saved studies/reviews, and the exact page links in your agent's
report. **Team** shows bound native sessions and permitted usage; observations
are not proof of research success. [CLI](docs/public-source/cli.md) explains saved
answers, continued selections, Task waits and Chinese detail (`--lang zh`, with
English fallback).
A bound session can read `alphalattice workspace show` for the selected workspace's inputs and intents.

Historical research promises no future returns, independent validation or trade
execution. Read each result's standing, coverage and time limits, including
survivorship and Sector classification; captured references prove no historical
availability. Calculation grants no activation or external publication. Network
and preparation outside first use, factor/model/strategy activation, storage,
automation, revocation, paid actions and external sharing remain yours; the
[decision pages](AGENTS.md#delegate-and-ask-for-decisions) identify each.
Provider deferral preserves inputs: follow its retry-time continuation.

## Guides and care

| Guide | Reference |
| --- | --- |
| [Getting started](docs/public-source/getting-started.md) | Install, launch, preparation and capacity |
| [CLI](docs/public-source/cli.md) | Answers, YAML and continuations |
| [Research flows](docs/public-source/research-flows.md) | Studies, books and review |
| [Agents](docs/public-source/agents.md) | Goals, cards, native sessions and Team |
| [Extending](docs/public-source/extending.md) | Factor/model trials and person activation |
| [Data and care](docs/public-source/data-and-care.md) | Updates, data decisions, backup/restore |
| [Privacy](docs/public-source/privacy.md) | Usage facts and disabling their reader |
| [Downloads and data terms](docs/public-source/third-party-and-data.md) | Models, sources and conditions |

This checkout is yours and your agent's to change: fix bugs and add strategies,
models and features through the [source-change path](docs/public-source/extending.md#change-the-code).

Use backup operations and restore into a new directory; do not repair sealed
records by editing them. The [native usage procedure](.agents/skills/alphalattice-research/references/native-visibility.md)
keeps usage facts, drops conversation text and lets the person disable usage reading.

## About and licence

Created and maintained by Xinglin Li: [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com).
[Apache-2.0](LICENSE) covers the software; [NOTICE](NOTICE) records third-party
material. Downloaded models and acquired data keep their own terms. See
[governance](GOVERNANCE.md) for project decisions and contribution credit, and
[contributing](CONTRIBUTING.md) for the opening condition, agreement and workflow.

## Verify a source checkout

After `uv sync --locked --all-extras`, build Local Web with
`uv run python scripts/build_local_web_ui.py --product` before running the
[public checks](CONTRIBUTING.md). Tests also check the built output at session
start and build it once if it is missing or stale; a failed build fails the run.

## Development history

AlphaLattice has been developed privately since 2026-08-06, with about 6,000
commits in private development by 2026-10-08. Each public commit contains one
shipped change exported from that development; releases publish the resulting
versions.
