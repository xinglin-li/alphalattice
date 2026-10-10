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

## 0.1.4

**Your date, in one sentence**

- The first use runs to the date you name: that date's reviewed positions, their Risk and a delivery report.
- A date whose information is not yet complete is the usual case during the US session. The agent prepares everything now, and the Host finishes the update itself once the data is ready, about two hours after the close. The ready time is given in New York time and in yours.
- A decision only you can make, activation included, is asked in one line and relayed in your words. It never needs a Workbench click. Each relayed yes answers one request, once.
- Answers are checked to tell the truth:
  - a next step the product offers is accepted when sent as offered;
  - a stopped task's answer names its way on and never asks the agent to wait.

**The agent and your machine**

- The agent reads one readiness list with `doctor` and fills what is missing: the product's own dependencies itself, other software after one question to you. A background wake that cannot be delivered is refused up front, and the agent then waits within its turn.
- On Codex, while a goal is active the agent waits inside its turn instead of ending it for a queued wake. In goal mode the Codex app holds queued messages until you press Steer (reported upstream as openai/codex#52705).
- Independent studies and the Evidence install run side by side within the CPU budget. Results are byte-equal to running them one at a time, and Alpha studies still run alone.

**Evidence**

- SEC filings are fetched under your consent with the product's own contact; you are never asked for a personal name or email. Within the default budget of three filings per issuer, the agent proceeds and tells you in one line.
- The Evidence retrieval runtime installs inside the running Host as a task you can follow, cancel and resume. No restart is needed.

**Previews**

These are new and still being proven. 0.2.0 will promote them once they run end to end without a fix.

- **Investment committee.** After a date's positions, the lead agent chairs a committee as portfolio manager, with the Alpha, Risk and CRO specialists. Stances are blind until all are in. Figures are quoted, never typed. The CRO's dissent appears verbatim in the delivery report. The committee changes no number.
- **Your own agent layer.** `.alphalattice/user/` holds your memories, a local guide and additions to the cards and Skills. Releases never overwrite it, and it is backed up before an upgrade. The agent keeps it current and prunes stale or duplicate memories.
- **The Workbench reads as your date.** Home and Portfolio lead with the positions date, the date's Risk and the committee. Activate is a secondary control beside the sentence to tell your agent.

**Known limits**

- The delivery export is English and still shows some raw figures and codes. Words and number formatting follow in 0.1.5.
- The decision schedule is fixed: decided at the official close, entered at the next official open. Other schedules are a recorded feature request.
- Claude specialists use the `haiku` alias at high effort. On Bedrock, Vertex or Foundry, pin the alias with `ANTHROPIC_DEFAULT_HAIKU_MODEL`. A project configured from 0.1.3 keeps its copied cards, as earlier notes describe.
