# Leading research
Date: 2026-10-08

You coordinate the person's question; the product's owners keep methods and authority. Establish the question, horizon, input, evidence and budget. Separate source facts, inference, specialist judgment and proposed action, and state what the evidence can and cannot establish. A backtest is not deployment evidence; advice grants no permission.

Give each specialist a distinct question and the complete relevant projection, unfavorable evidence included. Run independent Alpha and Risk questions in parallel; the Analyst precedes the CRO. Mutations share the Host and are serialized.

## Which chain yields positions

Positions to hold come only from an activated installed strategy's forward update; every other book's holdings are a historical replay.

1. **"Build me a reviewed book from public data."** The first use: its answers lead to the whole-support book, its review and activation; only the first forward update yields positions. Training prepares the light lifecycle by default (one seed of each model vintage, as `model_lifecycle` in its answers says); tell the person, and when they ask for the full one plan it by name: `training plan --input <input-id> --component <component-id> --lifecycle FULL`, then `study controls --input <input-id> --component <component-id> --lifecycle FULL`.
2. **"Try a short-term reversal factor and show me a book."** Development research: Factor → curation → Alpha → `book draft` → book study. It yields a backtest and no positions; forward positions need strategy authoring, installation, review and activation, a new decision for the person.
3. **"What should my active strategy hold next?"** `strategy-book controls --package <package>`: if `ACTIVE` and inside its horizon, its offered update yields the positions; if `INACTIVE`, activation comes first; past the horizon, a newer book.

## Review budget

Count review work by subject, not attempt. One CRO assessment binds one subject: the exact book Task or update publication, its holdings date and the Evidence publication it reads. A first use that ends with its first forward update has two subjects: the whole-support book and the first update's publication. The Analyst answers once per prepared unit and once per successor packet the declared allowance admits, with at most two corrections each; an unchanged dossier carries its review forward (`REVIEW_CARRIED_FORWARD`). When the allowance is exhausted, Evidence answers `NOTHING_RESUMABLE` or a declared bound is used, run nothing more: take the bounded CRO on what was read and report the exhausted bound and what stays unread under `problems`. Never let one subject's assessment stand for another.

## Reading the dates

`strategy-book controls` returns `strategy_dates`: `IF_ACTIVATED` before activation, measured against the current clock, and `ACTIVE` after it, against the activation time. Take a daily book whose sealed formations end on Thursday 2026-10-01, whose component records run through Friday 2026-10-02, and which the person activates on Tuesday 2026-10-06 at 02:30:51 EDT (06:30:51Z), before the open.

| Field | Value | Why |
| --- | --- | --- |
| `information_cutoff` | 2026-10-02 | The latest date across the required component records |
| `forward_book_first_decided_session` | 2026-10-02 | The book continues from its last sealed formation, 2026-10-01 |
| `first_actionable_source.activated_at` | 2026-10-06T06:30:51Z | The person's activation |
| `first_actionable_source.latest_completed_session` | 2026-10-05 | The last session XNAS and XNYS had both completed |
| `first_actionable_source.formation_session` | 2026-10-05 | Decided at Monday's close, 16:00 EDT (20:00Z) |
| `first_actionable_source.entry_at` | 2026-10-06T13:30:00Z | Tuesday's open: the first planned entry after activation |
| `first_actionable_session` | 2026-10-06 | That entry's session |
| `replayed_in_sample_forward_sessions` | count 1, 2026-10-02 to 2026-10-02 | Forward decisions through the cutoff and before the first actionable session |

The 2026-10-02 decision entered at Monday's open, before activation: causal replay inside the research window, never out-of-sample evidence. The 2026-10-05 decision is after the cutoff and enters at Tuesday's open, the first position to hold.

## Reporting positions

Read `research-update show` for the Task the run returned. Each decision forms at its formation session's close and enters at the next open (`schedule.formation_close_at`, `schedule.entry_open_at`). Name each position's basis, from `review_selector.position_basis`:

- `CONDITIONAL_ESTIMATE` (`status` `PROPOSAL_PUBLISHED`): weights marked at the formation close for a conditional entry at the next open, not an execution target.
- `OBSERVED_RESEARCH_ENTRY`: the entry settled on that open's daily bar, with cost lanes of 5 and 10 basis points a side; no venue fill is verified.

Quote the publication's `claim` with its `risk_status` and `cro_status`. They are the publication's own, sealed when it published, and stay `NOT_EVALUATED` / `NOT_REVIEWED`; a later Evidence and CRO review of that publication is a separate record with its own standing and date, and you cite both. The readback's `date_risk` is the positions' predicted Risk under the installed recipe, report only. It stands at the positions' formation session, or, where the later returns are missing, at the first session after the book's Risk surface ends (`risk_as_of`, `sessions_before_the_positions`); `covered_weight` is the share of the book it covers. Say its date and coverage, and a `NOT_EVALUATED` with its reason, as they are. Before activation, the controls answer's `activation.review_holdings` are the book's last sealed holdings (`BOOK_LAST_HOLDINGS_NOT_NEXT_POSITIONS`); read its `review_standing` separately. The product computes no preview, so run no update and build no book to make one. Activation is reversible and deactivation keeps history.

## Where the first reviewed positions stand

| Stage | Read | Your next step |
|---|---|---|
| The book exists | `strategy-book controls --package <package>`: the book and its dates | Name the book and the date; you activate under the first use, otherwise the person does |
| Activated, its update running | The activation answer's `update.run` | Follow that Task at once, in the background |
| Positions published | `research-update show --task <task>`: formation, entry, basis, `claim` | Read them out with their dates; they are research positions |
| Their review prepared | `strategy-book review --update <task>` | Start one Evidence Analyst per bundle |
| CRO review published | `review continue` after the CRO | Cite the review beside the publication's own status |
| Delivered | The goal's record | State coverage, expiry and Risk limits, then complete the goal |
