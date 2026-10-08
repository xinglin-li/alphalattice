# Leading research
Date: 2026-10-07

You coordinate the user's question; product owners keep methods and authority. Establish the question, horizon, input, evidence, budget and lawful next steps. Separate source facts, inference, specialist judgment and proposed action; state what the evidence can and cannot establish. A backtest is not deployment evidence. A goal records multi-step work ([goals](goals.md)); subagents do not replace Host ledgers.

Give each specialist a distinct professional question and the complete relevant projection, including unfavorable evidence. Run independent Alpha and Risk questions in parallel; the Analyst precedes the CRO through product validation. Mutations share Host capacity and are serialized, even when specialists share an input. Advice grants no permission. Keep specialist text and your synthesis separate, with exact references; use one CRO assessment per review subject (Review budget below) and keep your response distinct ([review handoff](cro-handoff.md)).

## First use and review order

From the person's sentence, open `FIRST_USE` before preparation, follow the strategy controls' required Alpha and Risk studies over their whole support, prepare and install the strategy, and run its whole-support historical book. Reuse completed required studies; add no exploratory Factor study, shortened trial or alternative model unless the question calls for it.

Before CRO, read the exact book's current Evidence answer. Continue its exact selected packet lineage only under a declared cumulative allowance with positive sessions and windows remaining; analyze and publish each successor packet, then reread Evidence. A null first-reading allowance grants no continuation authority; never invent limits. `COMPLETE` describes the sealed reading plan only. An absent or exhausted allowance or `NOTHING_RESUMABLE` retains unread ranges and limits for bounded review: take current Evidence's `dossier` action, then that dossier answer's `cro_bundle` action ([Evidence handoff](evidence-analysis-handoff.md)).

### Review budget

Count review work by its subjects, not by attempts. One CRO assessment binds one subject: the exact book Task or update publication, its holdings date and the Evidence publication it reads. A first use that ends with its first forward update, as in the example under Reading the dates, has two subjects:

| Subject | Holdings date | Evidence it reads | CRO assessments |
| --- | --- | --- | --- |
| The whole-support historical book Task | Its last sealed holdings, entered 2026-10-02 | Current Evidence after continuation is settled | 1 |
| The first forward update's publication | The first actionable entry, 2026-10-06 | Current Evidence for that publication | 1 |

The Analyst answers once for each prepared unit and once for each successor packet the declared allowance admits; each answer allows at most two corrections. Load a stage specialist only for a judgment the question needs. A correction is the same assessment, and an unchanged dossier carries its review forward (`REVIEW_CARRIED_FORWARD`) without a new one. A CRO run before continuation is settled reads an Evidence publication that the continuation replaces, so its assessment no longer stands for the book: settle continuation first.

A budget is exhausted when the Evidence allowance has no sessions or windows left, when Evidence answers `NOTHING_RESUMABLE`, or when a bound the person declared for the goal is used. Then run nothing more against it. Take the bounded CRO on what was read, and report it in the goal submission: name the exhausted bound and what stays unread (`remainders`, unread ranges, unreviewed units) under `problems`, answer the affected criterion `NOT_MET` or `NOT_ASSESSED` with that note, and list any later subject, such as a newer update publication, as unreviewed. Never enlarge an allowance, and never let one subject's assessment stand for another.

## Which chain yields positions

Positions to hold come only from an activated installed strategy's forward update. Every other book's holdings are a historical replay: evidence about the past, never positions for the next session. Three requests show the difference:

1. **"Build me a reviewed book from public data."** First use: `FIRST_USE` goal → strategy controls → the required whole-support Alpha and Risk studies → prepare and install → the whole-support historical book → Evidence and CRO → the person's activation on Portfolio → `research-update plan` and `run` → `research-update show`. Only that last update yields positions; the historical book before it yields replayed holdings and the review that activation reads.
2. **"Try a short-term reversal factor and show me a book."** Development research: Factor → curation → handoff → Alpha → `book draft` → book study → `study show`. It yields a development book's replayed holdings and metrics, a backtest, and no positions. Forward positions would need strategy authoring, installation, a whole-support book and activation: a new decision for the person.
3. **"What should my active strategy hold next?"** An installed strategy: `workspace show` → `strategy-book controls --package <package>`. If `ACTIVE` and inside its horizon, its offered `research-update plan` → `run` → `show` yields the positions, read as Reporting positions describes. If `INACTIVE`, the person's activation comes first; past the horizon, a newer book does.

## After a reviewed installed book

On every fresh session, return to `workspace show`, its package's `RUN_FORWARD` intent and `strategy-book controls --package <package>` before a forward plan. Read `activation`: an `INACTIVE` strategy offers the newest completed whole-support book or a held reason; `ACTIVE` names its book, first forward session and horizon. Review the historical book, review standing, information cutoff and conditional first actionable session before asking the person to take the exact offered activation on Portfolio. First forward positions are produced by the update after activation; read them as Reading the dates and Reporting positions below describe.

Positions before activation: the controls answer's `activation.review_holdings` holds the reviewed book's last sealed holdings, their cash and the sessions they were decided and entered (claim `REVIEWED_BOOK_LAST_HOLDINGS_NOT_NEXT_POSITIONS`). Show them with the conditional dates; they are the review's last holdings, not the next positions. The product computes no preview, so run no forward update and build no book to make one. Activation is reversible and deactivation keeps history: the person may activate, read the first forward update's first-day positions, and deactivate if they decline them.

Installation enables historical replay; activation and enabling daily updates on Settings are separate person-only decisions. First-use delegation grants neither activation nor automation authority.

While active, follow `next_requests.update`: plan by package, run its offered plan and read back the Task that run returned. Report published positions with formation and entry dates and exact `claim`, as research rather than orders or advice. A blocked update from an earlier activation is history, not this Task's continuation.

## Reading the dates

`strategy-book controls` returns `strategy_dates`. Before activation the dates are `IF_ACTIVATED`, measured against the current clock; after it they are `ACTIVE`, measured against the activation time. Take this daily book: its sealed formations end on Thursday 2026-10-01, its component records run through Friday 2026-10-02, and the person activates it on Tuesday 2026-10-06 at 02:30:51 EDT (06:30:51Z), before the market opens.

| Field | Value | Why |
| --- | --- | --- |
| `information_cutoff` | 2026-10-02 | The latest date across the required component records |
| `forward_book_first_decided_session` | 2026-10-02 | The book continues from its last sealed formation, 2026-10-01 |
| `first_actionable_source.activated_at` | 2026-10-06T06:30:51Z | The person's activation |
| `first_actionable_source.latest_completed_session` | 2026-10-05 | The last session XNAS and XNYS had both completed |
| `first_actionable_source.formation_session` | 2026-10-05 | Decided at Monday's close, 16:00 EDT (20:00Z) |
| `first_actionable_source.entry_at` | 2026-10-06T13:30:00Z | Tuesday's open, 09:30 EDT: the first planned entry strictly after activation |
| `first_actionable_session` | 2026-10-06 | That entry's session |
| `replayed_in_sample_forward_sessions` | count 1, 2026-10-02 to 2026-10-02 | Forward decisions through the cutoff and before the first actionable session |

The 2026-10-02 decision entered at Monday 2026-10-05's open (13:30Z), before activation: it is causal replay inside the research window, never out-of-sample evidence. The 2026-10-05 decision is after the cutoff and enters at Tuesday's open, the first position to hold. An update run after activation and before Tuesday's open decides both sessions and publishes the conditional proposal for the 2026-10-06 entry. Read before activation, the same book names Monday 2026-10-05 as first actionable at 08:00 EDT on Monday (Friday's decision still enters at Monday's open), and Tuesday 2026-10-06 at 10:00 EDT on Monday.

## Reporting positions

Read `research-update show` for the update Task the run returned. Each decision forms at its formation session's close and enters at the next session's open (`schedule.formation_close_at`, `schedule.entry_open_at`); returns run from that open to the holding end's open. Name the basis of each position:

- `CONDITIONAL_ESTIMATE` (`status` `PROPOSAL_PUBLISHED`): weights marked at the formation close for a conditional entry at the next open, not an execution target (`CLOSE_MARKED_ESTIMATE_NOT_EXECUTION_TARGET`).
- `OBSERVED_RESEARCH_ENTRY`: the entry settled on that open's daily bar (`DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION`), with cost lanes of 5 and 10 basis points a side; no venue fill is verified.

`review_selector.position_basis` carries the basis, and each `position_rows` row states it in words. Quote the publication's `claim`, `POST_OBSERVED_QA_NOT_TIMELY_ADVICE`, with its `risk_status` and `cro_status`: Risk and the CRO have not assessed that proposal until its own review runs. Report positions as research with their formation and entry dates, never as orders or advice.
