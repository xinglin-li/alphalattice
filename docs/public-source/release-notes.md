# AlphaLattice first release
Date: 2026-10-08

The `0.1.0` first release is a local equity research workstation for a person and a Codex or Claude Code agent. Factor evidence and curation lead to Alpha; Risk runs on the same input. Portfolio consumes Alpha and only uses Risk when its allocation reads it. Evidence and the CRO review the exact book and retain coverage gaps. A report-only Risk association does not mean weights used a Risk model.

## Included capabilities

The maintained CLI and Local Web reach the same calculation, permission, Task and publication owners. Local Web includes Chinese; `--lang zh` selects available Chinese CLI detail with English fallback. Editable declarations go through installed schemas. Answers record outcomes, refusals and continuations; saved reads retain their full selection, `--wait` follows named Task state, and Portfolio position metrics may include units. See [CLI](cli.md).

Native role cards and the research Skill let external agents interpret product evidence. Goals bind work to agent sessions; the product checks submissions against recorded Tasks and references but does not certify an agent's interpretation. Default research and Team need no product hooks. Team separates public messages, product receipts, sealed outputs, usage and optional native observations. See [Agents](agents.md).

Formula features have a bounded language, explicit preprocessing and separate plan, build, trial and review. A person can activate only an eligible daily definition. Sector-return leaves remain research-only until daily inputs are admitted. Alpha models have a contract check and copy-based sandbox before person-only activation; a passed sandbox is not qualification. See [Formula factors](formula-factors.md) and [Extending](extending.md).

Data updates preserve earlier inputs; issues expose evidence and permitted choices. Backup stores held state; restore targets a new directory and identifies values, inputs or packages that need rebuilding, recapture or installation. Native usage reading keeps whitelisted facts and drops conversation content; `--usage off` controls that reader, while the CLI launch counter remains separate. See [Data and care](data-and-care.md) and [Privacy](privacy.md).

Readbacks carry temporal statements for evaluated window, earlier input history, T0 membership backfill, survivorship, Sector treatment and price basis. A later observation does not rewrite a published session. Research publication does not establish future returns or an operating trading service. See [Temporal statements](temporal-statements.md).

## Platform and terms

Windows is verified for this source-checkout release; macOS and Linux are unverified. On 2026-09-30 at checkout `c109a7bc`, an empty-workspace preparation through its data decision took 407.1 seconds and 3.99 GB peak memory, using eight logical processors on four P-cores. The machine was Windows 11 Pro, Intel Core i9-13900K (24 cores, 32 logical processors) and 128 GB memory; a comparison ran concurrently on disjoint cores. This observation is neither a hardware minimum nor a runtime guarantee. [Getting started](getting-started.md) has the measurement context.

Xinglin Li created AlphaLattice alone and made this release. It is Apache-2.0; [NOTICE](../../NOTICE) lists included third-party material and separately installed software/model licenses. The release takes no outside contributions; see the [contributing note](../../CONTRIBUTING.md).

## 0.1.1

The seven shipped Codex specialist role cards run at `high` reasoning effort instead of `xhigh`:
in the agent evaluation, `high` completed the same tasks faster. A project configured from 0.1.0
keeps its copied cards; set `model_reasoning_effort` in its `.codex/agents/` files to choose.

## 0.1.2

**Data**

- A membership update stopped part-way resumes from its approved change and names the failed source.

**Agents**

- Research needs no product hooks or hook trust prompts. Observation needs no poller; usage is read at Goal and answer moments, with a person-only switch in Settings.
- Claude specialists use the `sonnet` alias at medium effort. On Bedrock, Vertex or Foundry, pin the alias with `ANTHROPIC_DEFAULT_SONNET_MODEL`.
- A project configured from 0.1.1 keeps its copied cards, just as the 0.1.1 note describes for earlier projects.
- Guides clarify first-use authority and recovery.

**Workbench**

- Review links open the named review; reading a document from a confirmation keeps that confirmation.
- Request statuses show in words, installed strategies are visible, and unreadable Tasks are named.
- Activation offers show the reviewed book's last holdings.

**Speed**

- Daily preparation and forward updates reuse verified inputs. Goal reads and cold training are faster; Task details show stage timing.

**Other**

- Saved answers can be read offline by section; CI verifies source checkouts and the installed wheel.
- A holding with an earlier finding or open issue stays in the CRO dossier when its new filing cannot be read; it remains unreviewed.

## 0.1.3

**Agents and Workbench**

- Sessions bind automatically on their first workspace command; research opens a Goal automatically. The installing agent continues in the same session.
- Only decisions outside the agent's authority wait for you; stopped steps keep their way on. Local Web follows the session's Goal and opens its progress, results and decisions.
- During the first use, the agent decides its own preparation's data issues under the first-use delegation, and the Data page shows what it is deciding.
- Answers carry the next step's fields. The agent can wait in one call instead of sending repeated status commands.
- Installing a research strategy no longer needs a Host restart: the running Host serves the new package at once.
- One call prepares an installed strategy's book and its review bundles; Evidence and the CRO still supply their judgments.
- Open result appears only where a Task's result has its own page; a preparation's opens the Data page.

**Data**

- A daily update that crosses an index membership change and stops part-way now resumes through its own update cycle, including one an earlier release left stopped. A departing member whose source ends before the update's session is recorded as a disclosed missing tail; a member that stays keeps the stop, with the listing, its last bar and the session named.
- A preparation whose data does not yet reach its target session stops at the data stage with the counts and the latest common bar, and resumes the same Task once the source catches up.

**Speed and Risk**

- The first use trains the light lifecycle by default, one seed of each model vintage; the full lifecycle stays available by name.
- Recorded Windows runs with a four-core execution budget reduced an Alpha lifecycle replay from 718 to about 170 seconds. Preparing training history took 8.3 seconds instead of 98 for one component, and 3.5 instead of 82 for another.
- A warm daily update, measured alone, took about 142 seconds (plan 16 s, Task 133 s), with every published value equal to the earlier build's. These are measured runs, not runtime guarantees.
- Risk diagnostics are now computed single-threaded and reproducible across machines; re-run Risk studies made before 0.1.3.

**Contributions and security**

- Contribution and governance policies are available; contributions open once the CLA bot is active. Report vulnerabilities privately through GitHub's Security tab.
