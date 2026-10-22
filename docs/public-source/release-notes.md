# AlphaLattice first release
Date: 2026-10-08

The `0.1.0` first release is a local equity research workstation for a person and a Claude Code or Codex agent. Factor evidence and curation lead to Alpha; Risk runs on the same input. Portfolio consumes Alpha and only uses Risk when its allocation reads it. Evidence and the CRO review the exact book and retain coverage gaps. A report-only Risk association does not mean weights used a Risk model.

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

## 0.1.5

**Launch and activation**

- Local Web stays running when your agent launches it from a background shell with no open standard input.
- `alphalattice host stop` waits for the Host's workers to finish before releasing the workspace.
- Measuring a run's process tree visits each process once, so a reused parent id cannot loop it.
- Outside first-use delegation, the agent introduces the selected strategy, asks you once in one line about activation and relays your answer.

**Refusals and recovery**

- An unambiguous command or option typo offers a corrected request that has been checked; it does not run without your agent sending it.
- A missing or invalid selection names its field and offers the read that shows valid values.
- An unknown strategy points to installed strategies; an unknown Task points to the workspace's Task list.
- A preparation plan offers no confirmation when it directs you elsewhere or its inputs have already been published.
- Reading a stopped Task's recovery returns OK with its permitted way on; the read does not turn the stopped Task into a success.
- A message file that cannot be read names the line and column where reading stopped and offers that operation's schema, a few kilobytes rather than the whole catalogue.
- The waits accept `--section`, as their compact answers offer.
- A request refused only because a field was sent empty comes back with the corrected request to send as it stands.

**Committee**

- Each member receives the facts its role judges from, drawn from the sealed records: the signal and why each name is held for Alpha, concentration and risk for Risk, the evidence and the run's process for the CRO, and the change, cost and timing for the portfolio manager. Each fact names its source; a fact no record holds is named as unavailable.
- Members cite facts, components and holdings by alias and type no figure. A typed figure is refused with each figure named and, for a holding's name, the message offered back with its alias.
- Tension points come from the date's exceptions, and an issue card names the held positions it affects.
- Refusals explain their rule and offer a corrected message where its shape can be repaired.
- Resending an unchanged accepted message returns its first admission without spending another turn.
- A wrong update reference points to open committees and offers a corrected request when the member's key identifies one.
- The portfolio manager must rule on every outstanding challenge before giving a verdict.
- A reservation cites the floor's facts it rests on, or their named absence; one without them is refused with its rule.
- The review coverage fact names each part of the book's weight: filings read, checked with nothing filed, and not reached. A review that did not record the split says so.
- When a published review is available, the CRO's material shows its current status and route rather than only the status recorded with the positions.
- The report carries the published review and every position a member still holds, in the member's words with its message id.

**Workbench**

- A date's performance appears above its holdings.
- Forward realized names the session it waits for when no realized session exists yet.
- Team, Sessions lists the specialist sessions each lead started.
- An accepted Evidence Analyst or CRO answer opens in full from its session's record.

**Previews**

- **Investment committee (preview).** Specialists judge a date's research positions and the portfolio manager records their decision; they change no position number.
- **Problem report (preview).** Tell your agent to report a problem in one sentence; it shows you a redacted report and sends it only on your yes.

**For developers**

- [Public scenario forms](../../src/devtools/evaluation/forms.py) describe starting states, clocks, routes, decisions and forbidden acts.
- The [feedback grader](../../src/devtools/evaluation/feedback.py) checks an agent's next step against the answer and records which part failed.
- The [committee grader](../../src/devtools/evaluation/committee.py) checks a closed committee's report against its floor: standing positions, rulings, the review and typed figures.
- [Design principles](design-principles.md) explain the product's choices with examples you can check.

**Measured**

- One recorded run from the first product command to delivery-report export took about 46 minutes on a four-core execution budget without a product repair; this is indicative, not a runtime guarantee.

## 0.1.6

**Reviews by their own members**

- Each Evidence Analyst and the CRO submits its own answer with a key scoped to its review thread. The Host records the issue, any correction and the acceptance as messages on that thread. A closed thread admits only its accepted answer again.
- An accepted Analyst or CRO answer opens in full from the session that led the review, whichever session sent it.

**Sessions**

- A session's participants are the members who spoke in it and the threads that began within its span. Every other thread of the same agent session is summarized in one line with its usage.
- A member's role, its messages and its usage read as one participant.
- Usage is read incrementally, so a session kept open for weeks reads in full. Each thread reads its own usage; a reading still catching up says how far it has read.
- Message ids are unique within a session. Short thread ids lengthen until they are unique on the page, and every message kind is labelled.
- A session takes its title from the question of its goal.
- A reloaded page reads the selected session's own records, so its participants and messages stay whole however much usage other threads record.
- Usage readings are paced by elapsed time, never by the research date, and a reading's progress is kept.
- A reference to a review thread is labelled as one and opens that thread.

**Home**

- Home leads with the date's outcome: its positions, review, committee verdict and report, each linked to its page. Then the decisions waiting for you, then progress.
- Recent deliveries list each delivery once. "Running" counts only Tasks that are running.
- A stopped Task is closed from Tasks.

**Law books**

- Every law in the product's and the Workbench's law books names its source: a classic principle, a form borrowed from a product or standard, or AlphaLattice's own, with its reason. Law ids and their holders are unchanged.

**For developers**

- The [session grader](../../src/devtools/evaluation/session.py) checks a session page's participants, ids, kinds, references and title against the records, including after a reload.

## 0.1.7

### Added

- Committee members answer from their own sessions through the same admission used for reviews. Their messages and usage stay joined after a page reload.

### Changed

- The README shows one recorded research run in light and dark themes: historical performance, sector exposure and replay holdings, and the session conversation. The captions distinguish historical weights from the positions discussed; neither is a record of executed trades.

### Deprecated

- None.

### Removed

- `refusal:native_bridge.child_not_a_specialist` was removed in 0.1.6 without an earlier deprecation notice. Usage is read independently of specialist-role verification; the separate verification fact remains.

### Fixed

- Committee member instructions give full executable commands with quoted absolute paths to their message files.
- A refusal after the committee closes explains what is still allowed and offers the report export.

### Previews

- **Data decisions by rule (preview).** During first use, a declared rule retains an isolated verified data move with its caveat or quarantines it recoverably. Past its bound, the rule leaves the decision to your agent. The Data page labels recorded decisions "Decided by rule" and shows an amber row while a decision is waiting. Once the required decisions are complete, the stopped preparation resumes as the same Task, with its earlier work kept.
- **Investment committee (preview).** Members judge research positions and the portfolio manager records their decision; they change no position number. Own-session messages and usage have been checked in a native run, not investment outcomes.

## 0.1.8

### Added

- [Design principles](design-principles.md) now describe one Host door, the loop between program, agent and person, decisions by declared rules and compatibility for interfaces agents have learned.

### Changed

- The README begins with what you can ask your agent to do, explains the workflow and its limits, and shows an earlier worked example. The committee image and example-report link remain reserved for the final 0.2.0 run.
- The light-theme historical performance and holdings images and both Conversation images are sharper, captured at twice their displayed width. The dark performance and holdings images are unchanged.

### Deprecated

- None. The current deprecation list is empty.

### Removed

- No new removal in this release. The compatibility record documents a refusal for a native child not recognized as a specialist, removed in 0.1.6 without an earlier deprecation notice; usage reading is independent of specialist-role verification, which remains separately recorded.

### Fixed

- Home keeps your authored question for the selected result. Automatic session work never supplies that question.
- Session usage progress shows its numbers while a reading catches up and no banner once usage is fully read.
- A reloaded session shows its own product records, even when later activity fills the recent feed.
- A later date's update applies your declared workspace data rule within its bounds, as on the first day. It no longer cycles between recovery and replanning.
- While a data decision is unresolved, the offered next step reads that decision before any replan.
- Deciding a later date's data case leaves an earlier date's committee report and its export exactly as sealed.
- In a source checkout, serving refuses Workbench files older than their sources and names the build command. Installed packages are unaffected.

### Previews

- **Committee decisions (preview).** The committee can decline endorsement without changing research positions. The Risk bundle separately states whether the date's risk was evaluated and whether the historical Risk study applied. Where the portfolio manager made rulings, the report labels the last ruling as its final view; earlier standing views remain readable.
- **Agent compatibility (preview).** Answers name the agent interface version. A deprecated entry stays available while its notice names the replacement and earliest removal release; compatibility checks refuse an undeclared removal or an incompatible typed change.
- **Bundle routing (preview).** A review bundle carries its workspace location. If a binding points elsewhere, the refusal names the conflicting workspaces and offers the exact binding recovery. Member keys and authority checks still apply.

## 0.1.9

### Added

- None.

### Changed

- The guide and research Skill describe how a past date is read. A past date's positions are decided only with information through that date's close, and earlier forward sessions are shown as a causal replay.

### Deprecated

- None.

### Removed

- None.

### Fixed

- A first use for a past date whose information is already available now publishes that date's positions. Previously, preparation could use market data from after the date, and the date's positions could not be published. Prices, Features, outcomes and model fits are now bounded at the date's close. Current sector classifications remain the declared current-universe assumption.
- A date the first use cannot support is refused before any model is fitted, with the nearest supported date or the next request. This covers a date that is not a session, is too far ahead, falls before the data, or lacks enough history.
- Local data that extends past the requested date is refused before it is copied into the workspace.

### Previews

- **Committee agenda (preview).** The committee argues a bounded agenda of its most material points, twelve by default, ranked by affected holding weight. The remaining points are listed below the agenda line, and the portfolio manager may raise one through a reply.

## 0.2.0

0.2.0 is the public launch. One sentence to Claude Code or Codex leads to a dated set of research positions, discussed by an investment committee and written up as a report, and to the same positions kept current day by day. Its acceptance ran on the release tree with no code change during the runs, on Windows: an offline first use, and a first use followed by daily updates, with a Claude Code (Haiku) lead and with a Codex lead that started its own Host inside Codex's default sandbox. A daily update took 15 minutes with the Codex lead and 25 minutes with the Haiku lead, both including the review and the committee, on a machine running other work. Its own update stage took 3 to 7 minutes, down from more than 10. Most of the remaining time is the agents' own review, committee and write-up.

### Added

- Public CI installs AlphaLattice and runs an offline first use on Windows, macOS and Linux.
- A numbered [first-run checklist](getting-started.md) covers the locked environment, the doctor and its repair commands, source and wheel installation, agent configuration, the workspace and the goal a non-agent shell must name. ([#3](https://github.com/xinglin-li/alphalattice/issues/3))
- The [README](../../README.md#two-ways-to-use-alphalattice) distinguishes first use in a new folder from daily updates in a new session in the same folder, and shows first-use and committee views in the coding app.
- `doctor` names the system libraries a platform needs, with the command that installs them: the OpenMP runtime (`libomp`) on macOS, and `libatomic1` and `libgomp1` on Linux.
- `research-update plan --date <date>` plans a named date's positions on the daily road. Every plan answer states which date's positions it holds, when they are decided and when they are entered.
- A first use pauses for a yes or no on keeping the workspace network on for daily market updates and SEC filings within the default budget: 3 per issuer, 600 documents and 3 GB overall. A yes remains in force until revoked. Each network use is disclosed in its answer and recorded in activity, including scheduled updates.
- A review without Evidence offers its install ready to run: the eight largest issuers by weight, and the next eight after a quiet stop. The preflight offers the same install, and the daily yes admits only that offer.
- Once a date's review is published, `research-update show` offers that date's committee. So does a lead's continue on an earlier bundle refused as stale or ambiguous after a second preparation.
- The first-use report asks whether daily updates should convene the committee. You can decide in the conversation then or later and change the setting the same way. The question never holds the first use open: its goal can be completed while the choice is unanswered. Until enabled, later updates get the portfolio manager's view on the reviewed positions, with the committee offered.
- The doctor and first-use startup share the command and retrieval-environment checks, using the command a fresh shell can reach. The doctor also names a missing Workbench build and its repair; installed dependencies no longer appear to await a decision. ([#2](https://github.com/xinglin-li/alphalattice/issues/2))
- An incomplete goal submission offers its missing evidence reads to attach and, where the Host can correct it, the corrected submission to send next. Its compact schema shows the reference form, problem fields and the reads each stage admits.
- A completed goal offers the experience-maintenance handoff, with its Goal and Task readbacks, so verified lessons and corrections can be kept in your personal layer.
- Delivery exports appear in the activity log as recorded reads; exporting a report does not open a separate research goal.
- `serve --no-browser --detach` starts the Host as its own process, which outlives the agent's turn. Each session's first answer offers a fresh Workbench link, including when the Host is already running. The agent opens it visibly in Claude Code's or Codex's built-in browser, rather than a separate system browser; a terminal without a built-in browser receives the link instead.
- After the blind stances are revealed, the committee offers each specialist its discussion turn, including a challenge to fill or a pass. The floor settles when all have spoken or passed and no challenge remains unruled, or when its discussion window ends. A silent turn and an unruled challenge are recorded as such; a floor that closes without a portfolio manager's verdict says so.

### Changed

- Committee guidance treats concerns about particular holdings as notes and suggestions to avoid or reduce those holdings. Declining endorsement is reserved for a hard defect affecting the research portfolio as a whole, such as a portfolio-wide data error, a breached Risk bound or missing key evidence. Major negatives below the agenda line remain in the report appendix, and the portfolio manager can raise a below-line point. No position number changes; you decide whether to act. ([#7](https://github.com/xinglin-li/alphalattice/issues/7))
- The worker pool keeps at most eight workers by default, bounded by memory and the CPU budget, and idle workers exit. A worker holds about 0.07 GB of memory instead of about 1.8 GB.
- Introductions name Claude Code first.
- Deadlines and windows a person reads show the Host's local time beside UTC. Research dates stay the session dates the positions belong to.
- A command's `--file` accepts the whole request its own answer saved. A saved goal submission can be sent with the command offered for it, without manually extracting its nested document. ([#1](https://github.com/xinglin-li/alphalattice/issues/1))
- A bundle folder may be new, empty, or hold the same bundle's earlier output, so a step reruns in place after a stop. Any other content is refused with a way on.
- The Evidence install's estimate is labelled as the upper bound of a first, cold download; held sources and packs are reused.
- Evidence intake reads the first three acceptable findings per issuer and names how many it leaves out in the receipt and CRO material. An excerpt cut inside a sentence is visibly marked. ([#9](https://github.com/xinglin-li/alphalattice/issues/9))
- Compact answers begin with status, next action, Task and saved-file receipt fields before nested data. An accepted bundle submission exits successfully even while its review Task is queued. Long client commands report their stage, dated update readbacks include a positions summary, and committee waits name their current standing and who is awaited. ([#5](https://github.com/xinglin-li/alphalattice/issues/5))
- A relayed network permission keeps the person's words, the question, the relaying session, the time and the scope. Network readback and use disclosures carry that origin across sessions; older permissions say which provenance was not recorded. ([#13](https://github.com/xinglin-li/alphalattice/issues/13))
- Update claims are explained as research positions, not trade advice, in answers, the Workbench, review material and reports. Activation dates retain their separate replay disclosure.
- The Sessions page shows each specialist once, with its own usage, joined by the key it ran with. Each message names its recipient by the thread it belongs to.
- Windows is the supported platform. On Linux and macOS, CI checks installation and an offline first use; other use there is not guaranteed.
- The README, the guide and the Workbench use plain words, for example "strategy" where the text said "book". Commands, fields and message kinds keep their names.
- The Workbench names every status it shows in words, in English and Chinese, including the committee's and the review's. Codes stay in the records. A committee member's session reads as its part in that date's committee.
- A date the person names means the same session on every road: the session the positions are entered on, decided with information through the previous close.
- The workspace data rule also decides a daily update's data issues and approves the full-history audit a dividend back-adjustment needs, within its declared bounds, and discloses each decision with the rule that made it. Data cases and full-history audits each have their own bound. A case already decided by the rule keeps its decision while the same evidence would be decided the same way, and a person's later decision always ends it.
- On a later day, Evidence reads only filings that are new since it last read, and reuses sealed selections across a restart.
- On macOS and Linux, capacity reads total and available memory from the operating system, and macOS may use all its processors. Windows is unchanged.
- The automatic storage cap admits a write the disk can hold with its recovery headroom, and grows with free disk. A cap the operator set is unchanged. A short disk is refused by name with a way on, and a research stage stopped by it can be retried.
- A daily update recomputes only what changed. A listing whose history the provider restated rebuilds its own columns alone, and a quiet day reads only the current year. Publishing the day's Panel rereads only the current year, the day's corporate-action audit runs once per batch of listings instead of once per listing, and the receipts it verified are not verified again later in the same update. The first daily update after upgrading from 0.1.x recomputes once in full, its score inputs included. Maintenance progress continues to advance per listing while audits run in batches.
- Under an agent sandbox that protects the project's `.codex` folder (Codex's default), the Host keeps the Session's binding in the workspace's own folder, and every answer says so.
- Each Session keeps its own usage reading. Participants that cannot be verified are shown as a bounded sample with their total count.
- Operational commands such as `host stop` and `cpu-budget set` are recorded but never open or join a research goal. A committee member's request counts toward its committee's goal instead of opening its own.

### Deprecated

- None.

### Removed

- None.

### Fixed

- A committee could stall until its time box when a member stayed silent or a challenge went unseen. Its wait now reports the discussion standing, pending members and closing time; a submitted stance updates the member's status, and a member with an open turn is told to act rather than wait on itself. ([#12](https://github.com/xinglin-li/alphalattice/issues/12))
- Committee refusals explain the alias, target, message-shape and agenda rules, name what the sender may submit, and offer a corrected message where possible. The guide documents every refusal and that `--seen` takes the returned message number. ([#6](https://github.com/xinglin-li/alphalattice/issues/6))
- A committee report with a long message no longer fails to export. A cut is named and points to the full record.
- Committee facts now agree with their own data. A holding kept by its tranche is named as such, score moves come from the previous session, and a coverage total equals its parts.
- On Windows, workers no longer open console windows, and counting tokens no longer downloads an encoding at Host start: the encoding ships in the package.
- A relay of words that say no person gave them is refused.
- A Host that cannot reach Codex from inside an agent's sandbox says so, and the lead waits inside its turn instead of for a wake that cannot arrive.
- A lead could open the Workbench in a hidden browser pane without telling the person. It now opens it visibly in the coding app's built-in browser and tells the person in one line, with the link.
- A daily update asked for a named date reviewed and convened on the next session's positions. It now reviews that date's positions.
- A dividend that a provider added or corrected late, inside the daily window, no longer stops the daily update. A refusal of that check names its listing.
- A past Session's usage no longer disappears from its page after a later Session binds the workspace. One completed usage read no longer prevents another session's pending usage from being read.
- The workspace database waits for a file still held by an instance that is closing, instead of failing.
- Files a workspace publishes now carry the workspace folder's own access, also when the Host runs under another account, as in Codex's Windows sandbox. Before, a published folder kept a private temporary-folder access list that only the Host's account could read.
- A daily update raised the same data issues every day and sent them to the person, and could wait on a provider that was not down. The rule now decides them in line.
- The network control reads the Host's clock. A delegation opened while the workspace runs on a historical date stays open until its end.
- A held data update no longer wakes the Host repeatedly. Taking the hold disarms its wake, a refused package waits for its next attempt, failed wakes back off, and a lost read is answered as a read.
- A stop resolved through its successor chain reads as resolved and leads to its successor.
- A stop explains itself with the same words whatever kind of Task stopped on it.
- An Evidence setup refusal names its reason.
- A review with nothing new filed names its way on instead of stopping without one.
- A link among a row's facts can be pressed where the row itself also opens a page.
- A request that arrived while the Host published its own Evidence install was refused as a configuration change needing a restart. It now passes.
- The same Evidence install sent again after a stop continues the stopped one, which then reads as resolved.
- A first use already past its strategy build named the build as its next step. It now names where it stands. Repeating the same first-use sentence and date from another session reuses the workspace's existing first use.
- A far-past first-use date before the supported history is refused by name before a goal opens, rather than starting an unsupported preparation. A review whose Evidence postdates its positions names that later cutoff.
- A refused request no longer permanently occupies its output path: a retry can replace a saved refusal that admitted no Task, while answers tied to admitted work remain protected. ([#4](https://github.com/xinglin-li/alphalattice/issues/4))
- A cancelled Task reports its stop and keeps verified stages and published results. A replan remains available where supported, but is sent only if the person or the lead that cancelled the work wants to continue.
- A goal's claimed outcome must agree with its criterion answers. A contradiction is named and the offered correction uses the outcome those answers support.
- The shipped CRO role can run the committee commands its route needs. Fresh installations use the shipped specialist cards without requiring an optional personal card file. ([#8](https://github.com/xinglin-li/alphalattice/issues/8))
- Setup and selection refusals name the actual cause and the next command: the configured host, an explicit workspace when none is bound, and the strategy id when a package hash was supplied instead. ([#11](https://github.com/xinglin-li/alphalattice/issues/11))
- Review materials name the installed strategy, count held issuers once and show route gaps separately from recorded reading coverage. ([#10](https://github.com/xinglin-li/alphalattice/issues/10))
- Exits have their own section apart from current holdings, and the current positions' review no longer treats an exited name as held exposure. ([#14](https://github.com/xinglin-li/alphalattice/issues/14))
- Older review exports omit issuer facts and coverage they did not record; complete dated exports use the owner's cutoff explanation. A replay date without recorded holdings says so.
- A committee message that named a point or holding by its name was refused without a way on. The refusal now offers the same message with aliases.
- A report suggestion listed every holding its member mentioned, including ones the member set apart. It now lists only those suggested.
- Some status words and steps showed as undeclared on the Workbench. Each now has its words, in English and Chinese.

### Previews

- None.

## 0.2.1

0.2.1 reduces repeated history writes, keeps provider corrections visible and improves continuation and recovery. Contributions are open.

### Added

- Contributions are open. Pull requests to `main` need the public checks to pass and the maintainer's review. Your first pull request asks you to sign the [Contributor License Agreement](../../CLA.md). See [CONTRIBUTING](../../CONTRIBUTING.md).

### Changed

- Ordinary daily updates preserve earlier history. The Feature Panel uses quarter parts, and a daily update rewrites the open quarter. Frozen observation arrays use shared 21-session blocks, and an ordinary daily update writes the last block.
- A provider's split adjustments are recorded and applied in order instead of rewriting the stored raw history. A forward split by a power of two keeps stored Features. Other ratios recompute cells that move past the declared agreement; Features in Panel spans already being rewritten are rebuilt exactly.
- A dividend payer's resent adjusted history is kept when it differs only within the data rule's declared precision, with unadjusted closes unchanged. The default relative precision is 1e-6. A larger difference, or a corporate action recorded in the same update, still restates the history.
- The Goal and report name every listing whose history is kept, with its largest difference from the provider's history.
- Backups include the record of split adjustments. Upgrade checks against released 0.2.0 verified stored records and matched published values without changing their seals.
- An explicit input capture states its preview-time byte ceiling before confirmation. Unchanged inputs reuse the existing version without a new capture Task; the answer names the version to select.
- `serve --no-browser --detach` answers `STARTED` or `ALREADY_SERVING`, with the workspace and the Host's address. An answer that finds no Host names the workspace it checked.
- Long commands print their Task id and the read that follows it before they block, and, with `--output`, the file they save. A caller cut short sends that read and reaches the same Task instead of sending the command again.
- A publication's claim follows its timing. Positions published before their entry session opens are timely research. A replay, or a publication made after the observations it settles, is post-observed. Neither is trade advice. Publications stored before 0.2.1 keep their claim and read as before. Upgrading recomputes nothing extra on the first daily update; a Lab portfolio update that repeats an identical earlier request plans once afresh.
- A local write that would be refused is refused before a restore, sandbox or bind changes anything.
- More refusals offer their way on, ready to run:
  - a missing agent-project declaration offers its two steps: configure the project for this host, then bind the session from it;
  - an output file that is already taken offers the same command with the next free output path;
  - a committee submission with the wrong fields names them and offers the schema for those fields;
  - a usage error offers the corrected command where the client can make it, and otherwise the command's exact help, never a template with unfilled placeholders;
  - work attributed to a closed goal offers the goal's read and the same operation again;
  - a goal note missing its evidence lists the references already attached and offers the attach and note forms;
  - a committee message whose text names a point below the agenda line lists the points it may name.

**Daily writes.** These count retained-file growth, including backups, rather than cumulative disk I/O. Event figures were measured on a replayed history; different universe sizes and events are separate cases. The upgrade figure with an existing backup store is calculated from measured object inventories.

| Case | Observed writes |
| --- | --- |
| Ordinary day including book publication and review, 120 listings | About 40 MB, including about 1.3 MB of incremental backup. |
| Data and Feature maintenance only, 472 listings, five consecutive sessions | About 207 MB: total first-day growth of about 153 MB, dominated by about 138 MB of database growth, then about 14 MB a day. |
| A day with two dividends | About 57 MB. |
| First daily after upgrading from 0.2.0, preserving the previous backup objects | About 123 MB once, including backup growth, by content-deduplication accounting. An empty backup store made the measured upgrade about 1.25 GB. |
| A 3:1 split | About 130 MB. |
| A split that rounds share volumes, such as 3:2 or 1:2 | About 0.9 GB; the following day was about 45 MB. |

In the release-tree native observation, the first-use workspace occupied about 8.69 GB, with about 10.38 GB across managed installation, workspace, cache and model folders, excluding backup. After the person granted the daily network question, the recovery grew the workspace by about 181 MB, including the first day's database step; its first backup added about 4.26 GB of retained objects once, and later backups add changed objects. The 472-listing walk measured data and Feature maintenance only; book publication, book history, review and their backup growth are additional and unmeasured at that scale. A full-year footprint against the 50 GB ceiling has not been measured. Split-day work, refits and additional explicit captures also add to the footprint.

### Deprecated

- None.

### Removed

- None.

### Fixed

- A daily update rewrote copies of the whole history. In a fresh folder on live data, that was about 2.1 GB a day. It now writes the day's change.
- A daily catch-up could keep a dividend the provider had restated because a cached history response was reused. Each acquisition now fetches fresh history; a normal daily makes no extra requests.
- When a split is reported late for a session already covered by issued positions, the stop names the listing and effective date. Issued positions stay as published; continuing requires a newer reviewed book built from current data.
- Committee materials and update readback no longer fail on a listing with no eligible score. Such listings are excluded from score comparisons and named.
- Sector labels and dated Risk answers no longer stay stale in a running Host after a membership change or a settled correction.
- A backup could fail with a permission error while an SEC campaign held its lock. Runtime lock files are no longer copied, and unreadable sources or backup objects are named with their recovery steps.
- The goal link an agent offered could open a "historical revision" notice while the first use was still running. The link now opens the current revision. A link pinned to an older revision says a newer one exists and links to it.
- A daily request in a new session could start a second first use in a new workspace, and ran a full preparation. Once a project's first use is complete, `serve` on a new workspace and `first-use prepare` refuse a second first use by name. Each names the existing workspace and offers its `research-update plan`. Separate new research, asked for in one line, still runs with `--new-research`.
- A sealed goal showed finished assignments as open. Accepted submissions now read as complete, unanswered work is archived with the goal's seal, and work that is still open says why and what its specialist does next.

### Previews

- None.

### Known limits

- On 2026-10-19, a Windows native first use on the release tree with Codex at extra-high reasoning effort took 74.7 minutes; its committee proceeded with notes. After an interrupted attempt, the person granted the daily network question; an assisted daily recovery then issued positions, published a partial review and sealed its first backup in 35.3 minutes, exceeding the 20-minute daily wall-time budget. Recorded Host work, including the data update, accounted for about 12.6 minutes; the remaining 22.7 minutes included agent turns, tools and waits. The Goal completion route also existed in 0.2.0; smaller readbacks and fewer finishing steps are planned for 0.2.2. This is a declared exception, not a runtime guarantee or a fresh ordinary-day measurement.
- Host memory above the 8 GB budget remains a declared limit carried from 0.2.0. The native first use peaked at 8.37 GB, and the authorized daily recovery peaked at 9.39 GB. A matched 120-listing replay measured 7.878 GB on 0.2.0 and 7.890 GB on 0.2.1. These are sampled peaks across the Host's processes, not the whole computer; this release makes no memory-reduction claim. Per-worker memory limits are planned for 0.2.4.
- On a machine with an empty package cache, first use asks once for permission to download the retrieval packages. Installing locked packages under the first use's network consent is planned for a later release.
- An explicit capture still copies the whole market database, about 1 GB at 472 listings. If the immutable snapshot alias cannot be linked, it can need a second full copy. The preview discloses that ceiling and that other files add bytes. Block-based captures remain future work.
- Outside the spans rewritten after a split that is not a power of two, Panel values derived from kept Features differed from a fresh rebuild by a measured 8.3e-8. Rewritten spans are rebuilt exactly; checking every remaining Panel cell is planned for 0.2.2. Such split days still rewrite whole older parts and are exceptions to the 50 MB ordinary-day write budget.

## 0.2.2

0.2.2 reduces repeated book writes, checks the whole Panel after a split repair and improves setup and recovery. It also removes unused code and consolidates tests that checked the same requirement.

### Added

- [The record is the lever](design-principles.md#19-the-record-is-the-lever) explains how recorded decisions and outcomes let agents investigate recurring problems and propose checked improvements, within the person's scope.

### Changed

- Book publication and history write records in proportion to the day's change, preserving earlier stored parts. Books written with 0.2.1 remain readable, and retrying an unchanged write adds nothing. This removes the repeated book-history writes behind 0.2.1's two-dividend write-budget exception; the release measurements below distinguish daily growth from the initial fill.
- Locked retrieval packages install under the first use's network consent, including on a machine with an empty package cache. The consent covers that installation, so it needs no separate download question.
- Derived caches have declared size bounds. Unused code and duplicate test setup have been removed while retaining the requirements checked.
- Product source contains **357 fewer net lines than 0.2.1**.

**Measured budgets**

The Windows observations below used a four-core execution budget. The public-data first use and its next daily ran with Claude Code and Haiku; the earlier 0.2.1 observation used Codex at extra-high effort. Different agents, dates and workloads prevent treating their elapsed times as a controlled speed comparison. Both new books were reviewed: 92 positions after the first use and 95 after the daily, with 472 supporting listings verified in each component. These are research positions, published as conditional estimates.

| Case and workload | Elapsed time | Host memory peak | Retained storage and daily growth |
| --- | --- | --- | --- |
| Public-data first use, 472 supporting listings per component | 121.8 minutes; 50.75 minutes of recorded Host work | 9.32 GB | 8.70 GB workspace; the first backup had not yet been filled. |
| First daily in that workspace, same support size | 43.6 minutes; 19.65 minutes of recorded Host work | 9.25 GB | 8.88 GB workspace; 180 MB of workspace growth, plus a separate 4.27 GB first backup fill. |
| Daily on released 0.2.1, controlled 120-listing upgrade scenario | 185.9 seconds, Host route and controller | Not established | 41.11 MB workspace growth and 1.24 MB backup growth. |
| First daily after upgrading that workspace to 0.2.2 | 356.8 seconds, Host route and controller | Not established | 40.85 MB workspace growth and 1.34 MB backup growth. |
| Retry of that daily | 208.7 seconds, Host route and controller | Not established | 0.115 MB of operational growth; the book remained exact and backup growth was zero. |
| Second daily after upgrading, 120 listings | 513.6 seconds, Host route and controller | Not established | 7.86 MB workspace growth and 1.43 MB backup growth. |
| Full-system run over five consecutive days | Not run | Not established | Not established. |

MB and GB use decimal units. Daily growth is retained-file growth, with backup objects shown separately; it is not a measurement of physical disk I/O. The 180 MB native daily includes 119 MB of market-database growth. Its new or changed files totaled 1.09 GB in final size, a different measure. The first backup's 4.27 GB is a one-time fill, rather than a routine daily increment. The upgrade scenario began with a 4.01 GB workspace and a separate 1.13 GB backup fill; its timings include the controller and no agent turns.

Memory peaks are sampled private commit for identified Host processes, with ownership checked against the listening Host. Unidentified transient processes are excluded. Recorded Host time combines request and retained Task-stage intervals without counting overlaps twice. It includes validation and locking waits, omits unrecorded work and is a lower bound, rather than CPU time. One first-use experiment was interrupted when the agent's background hosting command reached its time limit, then resumed as the same Task. No operator intervention or numerical rerun was used to improve the reported figures.

**Controlled first-use comparison.** The first-use release criterion uses the same generated 120-listing workspace, attached references, fixed clock and Goal submission across fresh Hosts. Peak Host private commit was 1.170 GiB on 0.2.1 and 1.175 GiB on the 0.2.2 candidate, an increase of 0.41%, within the 5% bound. Each submission completed. This isolates source changes under one locked environment; the driver, client and child processes are excluded. It measures reference verification, not the whole first-use peak. No valid preparation-peak comparison was obtained. The separate native observations of 9.32 GB and 0.2.1's 8.37 GB used different leads and have no matched earlier peak stage, so their difference is not attributed to this release's code.

The upgrade checks reopened a workspace made by released 0.2.1, continued on 0.2.2 and retained equal published values. The managed first-use footprint was at least 9.88 GB, excluding semantic-model aliases; a complete installation footprint and a full-year footprint against the 50 GB ceiling remain unmeasured. The native browser comparison was not established; deterministic Workbench checks passed.

### Deprecated

- None.

### Removed

- None.

### Fixed

- Every Panel cell is checked after a split repair, including cells outside the rewritten spans. This closes the measured 8.3e-8 difference from a fresh rebuild recorded in 0.2.1.
- Readiness advice offers commands that parse against the command grammar, in English and Chinese.
- A failed read no longer leaves a stale cached result for the next attempt. Retrying reads the input afresh.
- A missing stored portfolio observation is reported as missing even when the storage error supplies no detail.

### Previews

- None.

### Known limits

- The native daily took 43.6 minutes against the 20-minute budget, with 19.65 minutes of recorded Host work. The smaller readbacks and fewer finishing steps promised in 0.2.1 for 0.2.2 have moved to 0.2.4. No wall-time improvement is claimed here.
- Host memory remains above the 8 GB budget: 9.32 GB in the native first use and 9.25 GB in its daily. Memory limits and controls are planned for 0.2.6. The controlled first-use comparison above does not remove this absolute limit.
- The first daily after a first use grew the workspace by 180 MB against the 50 MB ordinary-day budget, with its initial backup fill counted separately. The two upgrade dailies grew their workspaces by about 41 MB and 8 MB. Reducing the first daily's excess is planned for 0.2.7.
- The full-system five-day storage scenario was not run. It is planned for 0.2.5's replay checks; no annual projection substitutes for it.
- An explicit input capture still copies the whole market database, about 1 GB at 472 listings in the 0.2.1 measurement, and may need a second full copy when the immutable snapshot alias cannot be linked. Captures that store changed chunks are planned for 0.2.10.
- Split days can still rewrite whole older parts and exceed the 50 MB ordinary-day write budget, even though every Panel cell is now checked. Reducing these writes is considered for 0.2.10 if replay measurements justify it. A full-year footprint against the 50 GB ceiling remains unmeasured.
- Browser parity in the native run remains unproven. The passing deterministic Workbench cases do not establish that separate result.

## 0.2.3

0.2.3 gives network-permission questions stable codes, checks new English and Chinese messages together, and exposes the recorded stages of a Goal. Detached startup also checks that the local Host can continue after the launching session ends.

### Added

- Network-permission questions carry a stable code alongside their English and Chinese words, scope and permitted continuation. The activity record links the question and its answer. A reply code must belong to the operation it answers; existing uncoded replies remain supported.
- A Goal's time profile reports recorded requests and Task stages, overlap, stage and cache provenance, and gaps in coverage. Missing observations stay marked as unobserved. Unattributed elapsed time is not reported as agent time, and an incomplete record does not establish a critical path.
- Replay inputs record their source, dates, clock and workspace preparation. Checks can distinguish a cold preparation from copying an existing prepared workspace; these records are diagnostic evidence, not a completed performance baseline.

### Changed

- New interface messages use stable names and named arguments in one English and Chinese catalog. Checks reject missing translations, duplicate names, conflicting aliases and mismatched arguments. Existing sentence-keyed translations remain available while their owners migrate them; new entries cannot increase that legacy inventory.
- Scripted evaluation answers use the Host's recorded question code and provenance instead of matching question wording. A transcript that merely repeats a question cannot authorize a reply.
- The public-source scanner recognizes registered product identifiers in their declared output locations while retaining the frozen limits on development labels.

**Measured observations**

The Windows run used a four-core execution budget and Codex with GPT-6 Luna. The first use reached a completed Goal. Of its 14 native agent sessions, ten used high effort and four used extra-high effort with inherited parent context and no assigned role card. Committee-role fidelity was therefore not established. These are mixed-effort observations, not a controlled benchmark for GPT-6 Luna at high effort or a controlled comparison with 0.2.2.

| Case and workload | Observed elapsed time | Largest observed Host memory |
| --- | --- | --- |
| Public-data first use for the 2026-10-21 session | 7,097.5 seconds, including a recorded pause during measurement | 8.66 GB |
| Next daily in that workspace, for the 2026-10-22 session | 464.2 seconds before the measurement deadline; the Goal remained open | 9.39 GB |

Host memory is sampled private committed memory; it excludes the agent's own process. Some stage observations were missing or had unmatched clocks, so the run does not establish a complete split between Host and agent time. A separate upgrade check retained equal values across all eight phases.

The incomplete daily increased retained workspace files by 144.27 MB net. New or changed files totaled 1.05 GB; that is not a measurement of physical disk I/O. This run did not establish backup growth. MB and GB use decimal units, and the incomplete daily establishes neither its final elapsed time nor its final memory or storage footprint.

### Deprecated

- None.

### Removed

- None.

### Fixed

- Reading published positions reuses each sealed input's verification throughout one request, including the associated portfolio review. This correction is carried forward from 0.2.2.
- On Claude Code, the first start asks you once to run the offered command in your own terminal; Codex detaches the local Host on its own. This startup step is separate from the network-permission question. Detached startup reports success only after the Host is ready and has escaped a launching process group that would otherwise end its work.
- Goal profiles align recorded intervals across Host restarts using their recorded clocks. Missing or inconsistent clock mappings remain explicit coverage gaps.

### Previews

- None.

### Known limits

- Host memory remains above the 8 GB budget: 8.66 GB in the first use and at least 9.39 GB in the incomplete daily. Memory limits and controls are planned for 0.2.6.
- The 20-minute daily wall-time budget remains unproven by this run: the daily was still open when its measurement deadline arrived after 464.2 seconds. Smaller readbacks and fewer finishing steps remain planned for 0.2.4.
- The incomplete first daily after a first use grew the workspace by 144.27 MB at the measurement deadline, exceeding the 50 MB ordinary-day budget. The five-day scenario through the actual daily data and book writers, and the storage work it informs, are planned for 0.2.9.
- An update with no end date, including an activation's first update or an undated daily update, can be refused when it resumes if its data step re-admitted quarantined candidates before waiting: that step changed the update's own inputs. The answer offers to cancel the update and plan it again; this route completes the update. A fix is planned for 0.2.4. The same limit applies to 0.2.2.
- The other limits recorded in [0.2.2](#022), including explicit input captures, split-day writes and unproven native browser parity, still apply. Idle shutdown and authenticated reuse of an existing local Host are also later work.
