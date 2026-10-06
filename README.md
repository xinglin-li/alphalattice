# AlphaLattice
Date: 2026-10-03

AlphaLattice is a local multi-agent quantitative research workstation. Your
Codex or Claude Code agent leads seven specialists: Data, Factor, Alpha, Risk,
Portfolio, Evidence Analyst and CRO. You give the question; the product computes
and records studies. Local Web shows their Tasks, results and reviews.

This checkout is yours and your agent's to change: fix bugs and add strategies, models and features through the [source-change path](docs/public-source/extending.md#change-the-code).

## Start with one sentence

Give your agent the repository's GitHub link and your intent, for example:

> Set up AlphaLattice in a new checkout and workspace, research a factor, carry its curated result into Alpha, run Risk on the same input, and build and review a Portfolio book; open Local Web in your browser so I can follow it.

The [agent guide](AGENTS.md) carries that flow. On first use, your exact sentence
opens a `FIRST_USE` goal before preparation. Its 24-hour delegation covers only
preparation network access, confirmation/resumes and data issues; other
person-only decisions remain yours. You may interrupt with goal abandonment or
close network access in **Settings**; stopping a Task needs cancellation.
See [first-use details](AGENTS.md#run-the-first-use-from-the-persons-sentence).

## Setup and research

Use the [setup procedure](.agents/skills/alphalattice-research/references/operating.md#setup-and-launch) for the locked
Windows/Python 3.12/uv setup, editable checkout or wheel, PATH and Local Web
launch. macOS/Linux are unverified. Installation, data acquisition and retrieval
model downloads have separate permission; a workspace starts with network
closed. Evidence needs its retrieval environment, local models and admitted
issuer package. The setup guide's measured time and memory are observations,
not requirements or guarantees. Adjust execution capacity, not the recipe.

The [research flow](docs/public-source/research-flows.md) connects Factor
curation to Alpha, parallel Risk on the same input, then a Portfolio book,
Evidence, Analyst findings and CRO review. Risk sizing consumes a completed Risk
study; a report-only association changes no weights. An empty workspace installs
no strategy. **Rebound Return Book** and **Trend Rebound Book** are installed
strategies, distinct from study books.

An installed book replays history until you activate a reviewed whole-support
run. Activation binds models, required calibration and sealed state without
fitting. Your daily-update selection publishes research positions, never orders;
scheduling needs the service running. Read the [forward dates and horizon](AGENTS.md#run-an-installed-strategy-forward)
before activation or continuation. Deactivation keeps history; a changed package
needs a new book and activation.

## Read results and decide

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

Use backup operations and restore into a new directory; do not repair sealed
records by editing them. The [native usage procedure](.agents/skills/alphalattice-research/references/native-visibility.md)
keeps usage facts, drops conversation text and lets the person disable usage reading.

## About and licence

Created and maintained by Xinglin Li: [LinkedIn](https://www.linkedin.com/in/xinglin-li-381571139/)
or [xinglin789@outlook.com](mailto:xinglin789@outlook.com).
[Apache-2.0](LICENSE) covers the software; [NOTICE](NOTICE) records third-party
material. Downloaded models and acquired data keep their own terms. The first
release accepts no outside contributions; [contributing](CONTRIBUTING.md) states
the agreement required before others join.

## Verify a source checkout

After `uv sync --locked --all-extras`, build Local Web with
`uv run python scripts/build_local_web_ui.py --product` before running the
[public checks](CONTRIBUTING.md). Tests also check the built output at session
start and build it once if it is missing or stale; a failed build fails the run.

## Development history

AlphaLattice has been developed privately since August 2026. This repository
starts at the 0.1.0 release and continues in the open.
