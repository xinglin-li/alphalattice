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
