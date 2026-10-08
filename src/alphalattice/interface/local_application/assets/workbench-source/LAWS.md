# The Local Web UI's laws

The rules the workbench's UI keeps today, in one place. From 2026-09-24 this file is their one
owner: a rule is added, changed or retired here. The 150 numbered laws of the earlier design rounds
(2026-09-18 to 2026-09-24) are its history; the appendix maps each old number to the rule that carries it
now, so an old citation in the code still resolves. How to build and prove a change, and where
each layer of proof lives, uses `scripts/build_local_web_ui.py --product --check`; internal QA guidance is private.

Each rule says what holds, then why where the reason is not obvious, then what holds it. In the
product checkout:

- a parameter in `design/parameters.json` (`scale.type`, `--table-rows`), refused off its value by
  the gate (`scripts/check_ui_parameters.py`, run by the build's `--check`);
- a builder or table in `js/app/` (`picker()`, `STATES`).

In the private external QA tree:

- a test in `tests/portfolio_strategy_lab/` (`workbench_dom.cjs`, `test_workbench_parameters.py`);
- a private QA-kit check, named (`glass_census`), run over the scenes;
- the counter, `scripts/count_ui_laws.py`.

A rule no check holds says so ("held by: review"); it is judged on stills.

The maintainer is the project's creator; many rules here come from the maintainer's readings of the
pages, each dated. An id in parentheses, a convergence-list row (V685) or a change-ledger row (U138),
points into the maintainers' records, which are kept privately.

## What the product is

A research workstation a person reads for hours. Quiet grounds without hue and one box material;
strong, readable type that no width shrinks; colour only where it carries meaning; every fact
owned by a record and none invented; one frosted glass for everything that floats. References
lend patterns, never a skin: Apple for type, material and proportion; Linear for lobbies, issue
pages and de-noising; GitHub and Primer for tables, tabs and the control ladder; Sentry and
Vercel for engineering state; Mercury for a financial product's precision. Never: grey helper
text in the name of minimalism, every section the same card, density read as "smaller is more
professional", another product's layout taken as the default answer.

## MA. Material and ground

**MA1. Two appearances, grounds without hue.** Light and Dark follow the host (the sun/moon
switches). Each is one lightness ladder: the canvas darkest, the lane a step lighter, a box and an
overlay lightest (Light: canvas L 96.2, lane L 99.2, box and field white; Dark: canvas 16, lane 20,
box 24, overlay 30). The page is one colour; no gradient, glow, texture or art on a ground. A
group's head bar (a list's group head, a table's group row) stands a step above what it sits on
(`surface-bar`: Dark a translucent white, Light the ink's 4 %); a well, darker than its ground, is
only for what is set into its card -- a darker bar read as a hole (the maintainer's reading, 2026-09-25).
*Held by:* `role.ground`, `role.ink`.

**MA2. Canvas by default.** Headings, fact lines, lists and paragraphs lie on the page's ground.
Structure is said by whitespace first, a hairline second, a container last. A box is declared
(`data-box`) for exactly one purpose: a figure, a decision or blocker, a workspace, or a table.
*Why:* a card around everything is noise; the box has to mean "this is a thing you act on or
read as one". *Held by:* `box_budget_census`, `box_purpose_audit`.

**MA3. One box material.** Every box wears one rule (`04-surfaces.css`, `[data-box]`): the box
fill, a 1 px `--edge-card` border, `--radius-card` (12), the rest shadow (Light two soft layers;
Dark a tight black ring under a lit 4 % hairline). No component draws its own box fill, border or
radius; a section laid on the ground is flattened. *Held by:* `role.depth`, `box_material_census`.

**MA4. A box never holds a box.** At most three depths: the canvas, a box, a raised layer. A
detail inside a box is a well inset (`--surface-well`), not a second card. A box keeps at least
8 px inside at its sides and 4 px at top and bottom; nothing it holds spills or is clipped.
*Held by:* `box_inset_census`, `control_spill_census`, `clip_census`,
`test_workbench_readback.py::test_collection_box_census_reads_decision_ancestry_in_the_pinned_browser`
(V660/U118: direct, wrapped, closed-fold and floating-reader ancestry).

**MA5. Hairlines.** Rows divide by the hairline (`--edge-hair`); nothing inside a box draws a
line stronger than the box's edge. Increased contrast makes every edge read at 3:1.
*Held by:* `role.line`, `edge_contrast_census` (V657: theme tokens matching their
declarations does not prove the alpha-composited edge reaches 3:1 on its ground),
`workbench_minor_ui.cjs` (V657/U116 and V672/U128: weak solid and glass edges fail;
an undecidable backdrop is named, never certified).

**MA6. Radii.** One box radius (card 12: a box, a panel, a menu, a popover), the dialog 22, a
control's face 6, a box inside a card 8, micro 4 (a key, code), the pill for buttons, search
fields, chips and badges. *Held by:* `scale.radius`.

## TY. Type and space

**TY1. One text ladder.** Every text run is one style of `scale.type`: caption 12/16 (chrome
only: a badge, a key, a chart's tick), callout 13/18 (the UI: a row's words, a label, a control,
a table head), body 14/22 (reading: a fact's value, a cell, prose at 24), title-3 15/20, title-2
17/22, title-1 22/28, large title 26/32 (an object's name), figure 22/28 (tabular). zh adds
leading and drops tracking. Nothing is uppercase. *Held by:* `scale.type`, `pages_walk` R5,
`ink_census`.

**TY2. Three weights.** 400 body, 500 titles and labels of rows, sections and the current tab,
600 for a form group's label, a report's sub-head and an alert's title. Nothing is bold for
emphasis. On Windows a zh title is Bold (YaHei has no Medium; 500 draws Regular).
*Held by:* `--weight-*`, `ink_census`.

**TY3. A floor for what is read.** Content text is never under 13 px and never small and grey at
once; the 12 px caption is chrome. The muted ink is the text ink mixed into its ground
(`--muted-mix`: 80 % Light, 72 % Dark, more in increased contrast), re-mixed in each box. Density
is never solved by shrinking text: wrap, fold or disclose. A notice or banner keeps a readable
text column: the smaller of `--width-xs` and its available column, with a short label needing
only its natural measure. Actions wrap before they squeeze that explanation (V655: contained
buttons had reduced the Risk study's explanation to zero width). *Held by:* `role.ink`,
`ink_census`, `box_inset_census` (text measure, including zero-width blocks),
`workbench_minor_ui.cjs` (V655/U114: the notice families and the former zero-width rule).

**TY4. Space is a scale, set by what meets.** Every gap, padding and margin is a step of
`scale.space` (2-64). The gap between neighbours in a body depends on what they are, set once in
the sheet: text after text 8, a control 12, a box on either side 16, a heading 20; a heading sits
on what it heads. No spacing classes. Every body that stacks parts is on the stack's list: the Lab's
guide card was not, and its field and action row touched (the maintainer, 2026-09-25: the spacing between
chips, buttons and option boxes is ruled and parameterised). Page cards and their visible stack wrappers belong to
the same rule and to its census (V685/U138: Reading's bare wrapper had hidden touching boxes).
Neighbouring boxes keep their shared step through every structural wrapper. The census
measures their actual neighbouring edges across wrappers and grid columns, so a wrapper
absent from a selector list cannot hide a seam (V694/U142).
*Held by:* the stack (`04-surfaces`, its steps
`stack-*` in `component.boxes`), `stack_census` (a control 12 after anything, a form's box 16 before
or after, on every page at the split width 900 and desktop width 1470, V650), `scale.space`, the gate's literal ratchet,
`test_workbench_readback.py::test_stack_census_reads_page_boxes_through_every_wrapper`
(planted zero gaps, unknown wrappers, plain panels, inert templates and grid columns;
explicit neutral carriers restore the parameter's step without adding painted boxes).

**TY5. One spine, few starts.** A row's lead (a dot, a glyph) sits in one 24 px column and its
title starts one gap after it, whatever the lead is. A page's text starts at no more than eight
measured x positions. Blocks side by side start at one top edge with their heads on one line; a
pair of boxes shares width and height. *Held by:* `role.spine`, `align_census`.

**TY6. A glyph stands on its words' line.** A glyph and its word are one pair centred on their line
(`glyphWord`); a glyph that leads words sits centred on their first line by that line's own height,
never by a hand-set nudge; a control that shows on hover never makes its line taller. An inline
glyph on the text's baseline sat on the x-height, a pixel and more low (the maintainer, 2026-09-25:
the icon before a word did not line up with it). *Held by:* `glyphWord`, `glyph_align_census` (every glyph with words
within 1 px of their line's middle).

## CO. Colour

**CO1. Colour only carries meaning.** Seven tones, each a mark and an ink (`--good`,
`--good-ink`, ...): green, orange, red, purple, blue (the accent), grey, cyan. A tone sits on a
dot, a glyph tile, a switch, the focus ring, a link, a chart series, a badge's tint and a signed
performance figure; never on a ground, a box's border, body text, the primary button or an
unsigned number. The selected row is a neutral fill; the highlighted picker option is the accent
at 12 % (18 % Dark). *Held by:* `role.tone`, `role.fill`, `color_census`.

**CO2. Colour lives on a mark, not a word.** A row leads with a bare coloured glyph (`tile()`: a
kind in its kind's colour, a run's state dot). A badge is the dot and the word in text ink on
the tone's tint. A count in the dock is a counter (Primer's): a pill in caption type, the muted
number on a faint wash, or -- a count that needs you -- the warning mark solid under `counter-ink`;
a tint of amber on the dark dock read as mud (the maintainer's reading, 2026-09-25). A word is coloured only as a
link, a reference or code. *Held by:* `color_census`, `counter-*`.

**CO3. Domains keep their colour.** Factor screening and CRO purple, Alpha cyan, Risk orange,
Portfolio green, data grey; the Main PM, live and links blue. A role's colour is on its mark, never
on its whole row, and a speaker's avatar is its colour wherever it stands: a reply under an
objection keeps its own (a Main PM reply drew the objection's orange; the maintainer's reading, 2026-09-25). A
rail's steps: done green, the current one the text ink with its halo (a green done beside a blue
current did not pair), a stopped one its state's tone (ST2). *Held by:* `KIND_MARKS`, `ROLE_TONES`,
`stepList`'s `data-tone`.

**CO4. A notice's tone is its mark.** A banner, refusal or callout says its tone by its mark
alone; its words are in the ink, any code small and muted beneath; never a tinted ground. A
warning floats as a glass box (LY1). *Held by:* `box_material_census`, `glass_census`.

**CO5. Code in GitHub's colours.** Keys green, strings deep blue, numbers and constants a brighter
blue, comments muted, punctuation in the text ink: three code inks per appearance. The editor
paints under its textarea, which stays the editor. *Held by:* `--code-*`.

**CO6. A colour is a meaning.** Seven meanings, each one tone (`role.meaning`, the scripts'
`TONE`): failure red, attention amber, decision purple, done green, active blue, stopped the
stop's grey dot, rest no colour. The state table and every caller name the meaning, never the
colour. Red is failure alone -- work that ran and failed, a read or a render that could not
complete, a check whose object did not hold; a stop by decision is attention: blocked, refused, a
refused step, a blocking finding, an incompatible comparison, a budget exceeded, a HIGH severity
(the maintainer, 2026-09-25: blocked is amber, and red is kept strictly for failure). *Held by:* `role.meaning`,
`test_workbench_parameters.py::test_red_is_failure_alone`.

## CT. Controls

**CT1. One control ladder.** A control is 24, 28, 32, 40 or 48 high; its context picks the step
and nothing else sets it. The vertical pad is derived, so a longer word never makes a taller
control. The workshop (`#page=kit`) shows every component at its step. *Held by:*
`scale.control`, `component.sizes`, `workbench_dom.cjs` (a control off the ladder fails).

**CT2. Buttons.** Buttons and search fields are pills. The primary is ink on both grounds, one
per page and per region; a panel's confirming step is a plain button. An icon button is the bare
glyph with a neutral hover fill. No box carries its own Refresh. Hover never flashes. *Held by:*
`workbench_dom.cjs` (one primary per page).

**CT3. Choosing.** A boolean is a switch (38 x 22, the accent track when on). A value among a
few is a segmented control: no track, the chosen word on a neutral pill. Every other choice is
`picker()`: a trigger showing the value and a top-layer list at the trigger's left edge, one-line
rows, a check on the chosen, a filter over 8 rows, a sheet under 640 px, windowed over 120 rows;
never a `<select>`. A picker with nothing to choose is held and says so (`None available`, or its
own placeholder); it opens no empty list, and the verbs it feeds say why they are held. An open
list keeps its size while it or the page scrolls, and a classic scrollbar widens it rather than
cutting its titles. *Held by:* `picker()`, `Picker.fit`, `workbench_dom.cjs`,
`checks/picker_scroll_probe.cjs`.

**CT4. One form shape.** Settings owns what a person sets once, as form groups: a label on the
ground over one box of rows; a row is a title, one line under it and one control at its right
(a detail, a chevron, a pop-up pill, a segment, a switch, a stepper). Free text is `.field`.
A unit field beside actions uses `field`'s inline layout: its visible unit, input,
readout and neighbouring actions share the input's centre, not the centre of a
stacked label and input (V680, the maintainer, 2026-10-04). *Held by:* `field()`,
`control_spill_census`, `field_alignment_probe` (a misaligned field fails the walk), review.

**CT5. Menus.** One line per item: glyph, word, muted note, a check on the current item, greyed
when unavailable. The current row is a grey fill; the focus ring shows only on
`:focus-visible`. A row that toggles reads its current word wherever the menu is drawn from
(Focus / Exit Focus); a row's press never ends in "Action needs attention"; a menu's own scroll
never closes it. *Held by:* review, `checks/menu_action_probe.cjs`.

**CT6. Dark controls.** Pickers, buttons, stepper buttons and chips wear `--ui-line` and
`--ui-face`; a text field the face only; a chip is a control, never a box. *Held by:*
`role.ground`, `role.line`.

**CT7. A held control says why on demand.** Its reason is a tooltip (`data-tip`) or
`aria-describedby`, never printed beside it and never the browser's `title`: a printed reason moved
its row when the control was held (the maintainer, 2026-09-25: pressing Submit the assessment was confusing).
Only a dialog's foot, where the decision is, prints it, above its verbs. *Held by:*
`Controls.setReason` (the note is for assistive technology), `workbench_dom.cjs` (no `title` on a
control).

**CT8. Every press answers.** A rule that tints a control at rest restates its hover and hover
edge; a fill that moves less than 2 % in luminance, or an edge less than 4 %, is no answer. A
state never lives on what the material owns (on the glass the border is the glass's, so a state
marks with an outline), and a thing at rest never wears the hover or selected fill.
*Held by:* `--glass-control-hover`, `glass_hover_census`.

## ST. States and truth

**ST1. Nothing is invented.** Every fact on a page has an owner and its exact record one
disclosure away. The UI recomputes nothing and never invents a name, date, state, relation or
event; a page reads its objects from their owner, never from a feed's tail. The same holds for
QA scenes and fixtures: a producer is labelled, nothing is made up. A positive check names its
checked set, scope and limit; an empty set has nothing to check, never a verified result
(V653/U113). A reference's integrity does not certify the research it references. *Held by:*
the harnesses, review, `workbench_reference_checks.cjs` (zero, unknown and mixed sets;
scoped positive words and their limits),
`test_workbench_readback.py::test_goal_page_paints_the_saved_narrative_first_and_verifies_evidence_second`.
A performance view reads each named metric or its recorded absence from the owner's selected
window, with its source and as-of date. A pinned reading stays pinned; the latest realized
view follows published outcomes and admits no future proposal return. Formation range and
outcomes observed through are stated separately; completed holding periods alone supply
returns, with open periods shown separately (V678, V690). *Held by:*
`test_cli_contract.py::test_the_installed_report_names_every_absent_performance_metric`,
`workbench_dom.cjs` interaction (selected/latest, publication refresh, absent metrics,
formation versus outcomes and open periods),
`test_rolling_history.py::test_two_updates_append_only_head_projects_the_verified_chain`.
Every coverage gauge and comparison with a floor reads the owner's accounted coverage,
with material reviewed, official no new filing and unaccounted weight shown separately.
A historical record without that accounting supplies no inferred total or floor comparison
(V681). *Held by:* `workbench_dom.cjs` interaction (44 reviewed plus 56 quiet; legacy missing
accounting supplies neither floor result nor meter),
`test_portfolio_review_decision.py::test_a_holding_that_filed_nothing_is_counted_apart_and_never_as_no_risk`.

**ST2. One state table, one state line.** Every state is one `STATES` entry (`status.js`): word,
its colour's meaning (CO6), what it means, next action where a person must move it, moving or held. It reads the same
everywhere (`stateLine`): the dot in its tone, the word, the duration muted, then the next step.
A neutral state has no dot; a dot always has its word, whole (a row's facts that hold a state never
shrink; the row's words wrap instead); no icon beside a state. A tinted badge only where a word
alone would mislead. *Held by:* `STATES`, `stateLine`, `.list-row-props > span:has(> .state)`,
`scripts/count_ui_laws.py`.

**ST3. No simulated progress.** Motion appears only while Task Control reports a moving Task (its
dot breathes); at rest only the live dot moves. An unknown extent is a still dashed track. Time
is never a bar: an ended duration uses the owner's two times. A moving duration reads the
owner's finite, nonnegative elapsed span as of the read; if that span is absent, its slot is
empty. The browser clock never extends a recorded owner interval (final walk F31/U174).
A wait says what it waits for, since when, and what happens next. *Held by:*
`test_workbench_task_duration.py` (owner spans, absent spans, selected Task identity and a
browser clock months away from the owner clock).

**ST4. A listed state follows Task Control.** A list shows a Task's state as Task Control last
reported it (`Data.lifecycleOf`); a reported move re-reads the listings once and repaints in
place. Every page showing a strategy's activation, schedule or daily standing follows changes
made elsewhere on the ordinary activity cadence, without Reload or a second mutation
(V676/U131). *Held by:*
`test_workbench_readback.py::test_listed_task_state_follows_task_control_without_a_reload`,
`workbench_routes.cjs` (external state changes on Portfolio, the book, Home and Settings;
the address and sealed reading stay put).

**ST5. Reading in place.** A first read shows a skeleton after the hold, never a spinner or
"Loading". Choosing what a page already shows re-reads it in place: nothing blinks or collapses,
the head never moves, repaints go by section; a slow read dims the lane after 300 ms
(`aria-busy`); only another book shows the skeleton, after 400 ms. No collection reads its items' full objects, and no read outlives its page (V683). Discovery uses the owner's metadata summary; selecting an object verifies it. Page cancellation preserves application-owned observations and admitted work. `workbench_collection_reads.cjs` holds these boundaries. *Held by:* the browser test
in `test_workbench_readback.py` (repaints keep scroll, focus and open folds).

**ST6. A refusal is said once.** A dot and a word, the cause in the owner's words (its code on
hover), the next lawful action beside it; its row echoes the state line and the record keeps it.
Where the state is already said (a Task's truth line), the stop's box heads with its cause, not the
state again, and its next action is the cause's own way on (`STOP_WAYS`), never the state's
generic one (the maintainer, 2026-09-25: "Blocked" was said twice in a row). A banner is only for what blocks the
page's primary. An answer the Host returns for correction is attention, not a refusal: its problems
by item, the answer kept as written beneath, the rounds left before the Host keeps the acceptable
items. A collection keeps readable peers beside its named refusals; its read failure stays in
that collection, leaving an accepted workspace session and independent readers available.
An unsuccessful reread retains the last successful rows with the refusal (V661/U119).
No way on, link or command sits inside a clipped line: it stands whole and clickable in the
action area or an ordinary disclosure (V682/U133). *Held by:* review, `live-tasks.js` `runBody`,
`workbench_handoffs.cjs` (the CORRECT round, a cold History refusal and failed Task resync),
`workbench_workspace.cjs` (independent Backups), `clip_census`, `control_spill_census`,
`test_workbench_readback.py::test_home_keeps_each_unreadable_tasks_recovery_ways_clickable`.

**ST7. An unknown is an empty slot.** An unknown fact leaves its slot empty or its row out,
never a dash or a placeholder; a sentence leaves an unknown clause out rather than keeping a hole. An
absence that is a state is worded once (`Not prepared`). *Held by:* `empty_slot_census` (no dash slot and no
hole on any scene page), `workbench_dom.cjs` (no placeholder on any route).

**ST8. One empty state.** In the list's place: the product's mark, one line of what would be
here, at most one sentence, and the one way to make it. A section's or a box's empty state is
its line alone. *Held by:* `empty_census`, `workbench_dom.cjs`.

**ST9. What needs you is counted where it is seen.** Home's Needs a decision is the one count of
what waits on the person (`LiveViews.needs`): the dock's Home carries it, and Data its own share,
as the warning counter before any neutral one (Tasks: only actual Task decisions); the rail counts only
what needs you, 15 px on the glyph's top-right corner with the glyph masked away under it (a ring
of a fixed colour bit the current item's fill); the document title leads with it as `(n) `. Nothing else feeds it and
nothing is kept as read: it falls when the owner's record does. An item explicitly waiting on
the agent does not enter a person's count; an absent owner field keeps the person default.
Activity updates carry the canonical Task version and retain an existing attention fact only
for that same version. In the background a Task that
ends is a system notification only when the viewer turned it on in Settings; the browser asks
once and a refusal is said on that row. An unrecoverable ledger-rebuilt stop holds no decision;
its reason and Re-PLAN remain readable. A later successful admission of the exact Task kind
and sealed plan, or the successful successor named by an exact recovery link, clears the old
stop. Canonical admission order takes precedence over a clock reading. A changed plan needs
that link. Tasks, History and Home show the successor once with the earlier stop folded
beneath it. A review's NONE action retains its words and limits in the neutral state; it
offers no way on and feeds no attention count (BADGE/U205). *Held by:* `workbench_activity.cjs`,
`test_pending_decisions.py::test_only_a_later_successful_exact_plan_supersedes_a_blocked_task`,
`test_pending_decisions.py::test_a_rebuilt_ledger_task_holds_no_decision_and_keeps_its_stopped_record`,
`test_workbench_badge.py`,
`test_workbench_review_publication.py::test_a_sealed_no_action_review_is_neutral_and_feeds_no_attention`, review.

**ST10. One wire answer, each reader's own rule.** Callers of the same exact unfinished GET
share one wire answer, consumed once; each applies its own strict or refusal reading. Different
selectors stay separate and a settled read is read fresh on the next request; the transport
never joins or replays mutations. One object open uses one bound readback. A read is accepted only in its owner's
current generation. Paging accepts one answer for a cursor in that generation, keeps unique
rows and refusals, and never cycles or moves the cursor back; a reread supersedes an older page.
A late continuation cannot navigate or rewrite the address after a newer addressed choice
(V689/U141). *Held by:* `workbench_read_lifetime.cjs` (wire, bound Task readbacks, Portfolio
boot and History races), `workbench_read_collections.cjs` (refresh, Goal and Team races),
`workbench_navigation_intent.cjs` (normal and late action continuations, including Facts and
Record), collected by `test_workbench_readback.py::test_the_workbench_harnesses_pass`.

**ST11. Following is the viewer's choice, scoped to the owner's work.** A Host launch and an
exact result or person-decision answer retain `follow=goal:<Goal UUID>`, or `follow=latest`
while no session Goal is known. The latest scope reads the workspace's latest active Goal;
legacy `follow=<Task UUID>` remains an exact Task choice. Only owner-attributed Tasks and
decisions can move a Goal follower: running progress, a finished result, a person decision,
or an exact published review. Missing `waits_on` keeps the person default. One state change
moves at most once. A temporarily unavailable decision read retries on the existing cadence,
even without a new Task event; an exact input family and cutoff generation is a distinct state.
Typing, an unsaved draft or a dialog holds the move; a late answer never
overrides a newer choice. Manual navigation pauses following, retained across Reload and
Back/Forward, with one `Follow again` control. This reuses the activity cadence and existing
object readers, not a new polling or execution path (UIFOLLOW/U208). *Held by:*
`test_workbench_goal_follow.py` (Task records through events into the real readers, state
changes and controlled latency), and `test_launch_and_answer_links_follow_the_session_goal_through_the_real_browser`.

## FT. Figures and tables

**FT1. Numbers are set like a statement.** One precision per kind; a sign only where it is the
message; tabular figures, right-aligned on the decimal; a percentage's unit in its cell, a
report's unit in the head's second line. *Held by:* review.

**FT2. A figure is a figure.** A number lives in a stat tile, a meter or a table cell, never in
a row's sentence, a caption or a lede. A panel has at most one caption, of at most 90
characters. Every figure tile stacks its label above its value at one left start, separated
by `--space-2`, whether plain or linked (V686/U139). *Held by:* `pages_walk` R1, R6,
`figureTile`, `workbench_minor_ui.cjs` (all destination shapes and the former link-only rule).

**FT3. Columns.** A table spans its box. Words first, figures last; each column has the floor of
its kind (`component.columns`: index 48, tight 84, figure 112, date or identity 140, text 180);
the slack goes to the substance column; a row's action is a fixed 48 at the right edge.
*Held by:* `component.columns`, `table_census`.

**FT4. Table anatomy.** Rows and head 44; the head in the muted callout at 500, aligned with its
column; a sortable head shows its glyph on hover or when sorted. A cell holds words and figures:
no bars, tiles or chips. *Held by:* `component.rows`, `table_census`.

**FT5. Too long rolls as a box.** A table wider than its lane keeps its columns and rolls sideways
on a visible thin scrollbar over a tinted track (`.table-scroll[data-edge]`), never clipped,
squeezed or folded; its substance column wraps above the table's floor. The same holds for a box
whose parts cannot wrap: a JSON or code block (`pre.code-block`, the floating document's too), a
stage rail at the split width (`.fv-pipeline`, V650), the Lab's row of property chips -- the box rolls as a
whole, one bar a box (the maintainer's rule of 2026-09-22: too long is a roll). A
value never grows its own bar: an identity is its reference (WD3), and a log or a list narrower
than a lane (`lane-narrow`) stacks each line -- its instant and author over its words -- rather
than rolling inside the flow (the maintainer, 2026-09-26: a bar under every hash of a Task's receipt, one
across a step's log; a box may roll sideways as a whole, never one bar per hash). A field
someone types in keeps its lines (a code editor's under its numbers). *Held by:* `table_roll_probe`,
`roll_census` (the boxes that may roll are named there), `scrollers_probe`, `detail_census` (inside
every detail).

**FT6. An unbounded record collection pages.** It has an index column, a search field once there
is more than one page, pages of `--table-rows` (50) and the foot's count with Previous / Next;
search keeps focus and caret. The same table contract holds for records in disclosures and
detail lists, including Evidence's issuers, limits, source bases, deferrals and preparation
Tasks; a long stack of folds cannot hide the collection (V684/U137). Object lobbies keep LS1's
shape. A bounded table keeps its count line and has no toolbar. *Held by:* `--table-rows`,
`table_census` (reads the served parameter, including untagged Evidence overflow),
`workbench_handoffs.cjs` (all dynamic Evidence collections; changing the parameter changes
the page size),
`test_workbench_readback.py::test_evidence_collection_census_refuses_untagged_overflow_in_the_pinned_browser`.

**FT7. One meter.** Every bar is `meter()`: 6 px, pill ends, accent fill, a required minimum's
mark in ink, no tone on a segment (a shortfall is said in words). A share or progress bar is at
most 480 px, left under its words; a composition or threshold gauge spans its figure; a
one-part composition draws no bar. *Held by:* `component.meter`, `scripts/count_ui_laws.py`.

**FT8. A figure is a box.** Every drawn chart and page figure is `figureBox()`: its head holds
the name, the (i) and the legend (keys wrap between keys, never inside one); its drawing fills
the box's width. A chart has one palette per appearance, a crosshair and readings on hover. An
inline meter in a row or tile is not a figure. *Held by:* `figures_probe`.

## PG. Pages and places

**PG1. The frame.** A dock, a top row and one lane. The dock is the fixed tree of groups and
lists and holds no data (no object's name): right or left, dragged 180-400 px, collapsed by
`Ctrl \`, a drawer under 1180 px and a 48 px rail under 900. The top row holds the path, the
page's `···` beside it and at most one quiet verb (glyph and word, no ground); views, search,
filters and display belong in the content's first row. The page never scrolls sideways.
*Held by:* `role.frame`, `places_probe`, `overflow_probe`.

**PG2. Every place has an address.** Every object, section and inspector mode has one
(`#page=...`, the object's key); `Copy link` is on every head. A choice that changes what a page
holds is a pushed address (filters, search and page numbers replace it); a cold open paints the
object; appearance and language never travel. An address that names no object where the page needs
one (`#page=evidence` from the dock) opens the owner's default -- that book, its head, path and tabs --
and the address becomes that book's exact one; with no default it opens the list (Books). A page never
shows an object its address does not name (the books scene's `evidence-none`, decided 2026-09-26).
There is no in-product Back for pages: the browser's Back is the history, the path and the dock are
the ways up and across (a detail's own levels have their way up, LS5). *Held by:* the browser test in `test_workbench_readback.py` (history entries, exact
Back / Forward), `places_probe`, `tabs_probe` (a page's tabs read against the address it ends on).
Every route a page offers opens its object, and recovery keeps its word: every row a collection
offers opens its object page; every kind a route can name has a page function; every object page's
tabs keep their object's address; every recovery promise holds across a reload (the maintainer,
2026-10-04). *Held by:* `workbench_routes.cjs`, `workbench_dom.cjs`
(V667/U124: every registered inspector mode restores its exact cold address;
V668/U125: every Goal reference opens its own reader and Back retains the Goal folder).

**PG3. The path.** A page's name is the top row's path, once: group / list / object / folder,
every segment but the last a link to its scope, an object's name cut at `--crumb-object`
(320 px), segments folding from the left. An explanation is the (i) on the last segment, never a
lede. *Held by:* `--crumb-object`, `places_probe`.

**PG4. One word per page.** `ROUTES[page][1]` owns each page's word; the dock, the path, the tabs
and the document title read it; no page has a second spelling. An object's page is headed by the
object's name, never by the page's word. *Held by:* `n3_census`.

**PG5. Titles.** A collection page (a list, a table, a form) has no H1 and no sentence: its first
row is its controls, then its content. An object's page opens with the object's own name (the
large title, at most `--object-title-lines` lines) and the owner's description under it; never
the page's kind or a sentence the product writes about itself. Inside a page a section is named
by a small muted label or a counted group band; a label never repeats the path, the object's name
or another label. *Held by:* `object_title_census`, `n3_census`, `workbench_dom.cjs` (h1 count and
order).

**PG6. One head for every view of an object.** The band: the name, its one state word in its
mark's colour, the id in short form with its copy glyph, the actions at its right with the
state's one next lawful action as the primary (from `cycle()`, never one that contradicts the
state). Under it the context line: properties, controls at 24, the chooser. Every view draws the
same head, so its tabs never move. *Held by:* `tabs_probe`.

**PG7. Tabs.** One level of underline tabs under the head, for a page's own views, an object's
folders, or one thing seen two ways; a list's views are its tabs (Data's pages, the Lab's
kinds). Words over a hairline, the current in ink over a 2 px indicator, the others muted, one
dense row high, never wrapped (overflow goes to More). The page remembers its last tab; `⌘K`
opens each. Tabs never nest. *Held by:* `component.tabs`, `tabs_probe`.

**PG8. One scroller.** A flowing page is one scroller; only a table's sideways roll scrolls
inside it. Two standing boxes become a console (the page still, each box scrolling alone) when
the window reaches the `console` height and the lane sets them side by side. Nothing grows
without bound: a long block pages, folds or shows its count. *Held by:* `scrollers_probe`, the
browser test (no nested scroller).

**PG9. One column.** A side column stands only for what changes with the main column: a chosen
item's detail (LS5) or a document's outline in its own margin. Verbs go to the top row or a
notice, decisions to a notice at the top of the content, properties to the head's context line
and a Properties section; nothing stacks under the content. A document's section list never
rides the scroll beside the text. *Held by:* `one_column_probe`.

**PG10. Split width first.** Every surface is designed at the 800-1100 CSS px split width; full
width is its extension. The product runs mainly in the Codex and Claude Code desktop apps'
browser pane, about half a desktop screen wide (the maintainer, 2026-10-04; V650). A narrow walk
means this split width, never a phone width: walk at 900, and at 1100 for a dense page.
QA viewport widths lie in 800-1673 CSS px. Content folds by container queries on its own width (the breakpoint
ladder), and a dense list drops a property before it truncates one. *Held by:* the gate
(breakpoints off the ladder), `overflow_probe`, `viewport_widths` (QA viewport range).

**PG11. Say it on one page.** Two pages of one object never restate one another: a shared fact
is said on one and linked from the other (a book's Overview holds its live standing; its Report
the sealed deliverable). *Held by:* review (its round probe: `book_b1_probe` in the plans).

**PG12. One anchor, a quiet rest.** One dominant visual anchor per viewport and one dominant
action per region. The UI at rest is quiet: an unchosen option is text, a Task at rest a row; a
hover, a selection, activity, a warning or focus lifts a thing. *Held by:* review.

**PG13. Making new things.** A group's `+` in its head is a menu of what it makes new, in the
chain's order, each opening its composer preset to its kind; a kind's list says `+ New` in its
first row. First use is the Home with one run: `Not prepared` with `Prepare workspace`, then the
preparation pinned, then the Home; no Welcome page. *Held by:* review.

## LS. Lists, threads and details

**LS1. Every object list is one lobby.** One-line rows under collapsible counted groups on the
list's own axis (state, time or family): the active, latest or current group open, the rest
folded; `--lobby-rows` (6) per group, then `Show n more` in place. Search, Filter and Display
are remembered. No row is marked while nothing is open. *Held by:* `--lobby-rows`,
`lobby_probe`, `pages_walk` R3.

**LS2. A row is one line.** A list row is dense (36 px): its title and words on one line (words
ellipsized, whole on hover), its sentence in what it opens; only the opened row grows. Its facts
are columns: one grid of named columns, each row a subgrid, each column as wide as its longest
value, while rows fit on one line (a lane of 1080, a list of 640); narrower, the facts go under
the title at its x, and an absent fact leaves an empty slot. *Held by:* `--row-dense`,
`component.columns`, `pages_walk` (R2, a drifting fact column), `workbench_dom.cjs`
(V665/U122: removed, reordered and translated named fact slots fail the column check).

**LS3. The rail.** The bar at a row's left edge marks the one being read (the exchange the
inspector shows) and nothing else: `component.rail`, the rule's width, the row's pad as inset, in
the link blue, drawn as a border so it snaps to device pixels. An exchange's kind is its word and an
objection's state its dot; a rail by kind (none on an answer) cut a Task from its answer (the maintainer's
reading, 2026-09-25). *Held by:* `component.rail`.

**LS4. Runs.** Everything that ran or runs is one list (`runRow` on `runsOf`): grouped by day,
newest first, the moving run pinned with its step, its state and starter in fixed slots. An
object's runs are the last section of its status page. A Task has one host and one body: the
summary line, the cause as one refusal line with the next action, the steps as folded rows (the
current and a stopped one open), the receipt with Copy per hash, the log one line each; a log
folds to one grey counted line until opened. *Held by:* `runRow`, review.

**LS5. The detail stands beside its list.** What a press opens is one detail card
(`data-layer="detail"`, the glass) beside the list it details (`detailSplit`); only that list
gives way. Its width is `--detail-width` (`min(480px, 40%)`); it opens whole; the same press, its
X or Esc closes it. Under 900 px it is the pane's layer. A page's own Facts and Record float
under its head. A saved object is never read in a dialog: it opens in its detail or as its page.
A press inside a detail that opens another object in it goes one level down, and the card offers a
`←` tab in its own left inset beside its head -- hidden past its edge until the card is under the
pointer or the tab has keyboard focus, shown where nothing hovers; the level it left is its tip --
one level up each press, back to the one the page opened (which has none); a detail opened from the
page, or closed, keeps no levels. The tab takes the inset's slot, never a layer over words, and the
head keeps its height and the body's edge (the maintainer, 2026-09-26: a layered floating box needs a way back
to the level above, one back arrow, which may be a tab hidden at its left; `Window.trailAfter`).
*Why:* the Evidence reading is the model (the maintainer's "textbook"): read the item without losing
the list. *Held by:* `component.detail`, `detail_census`, `detail_switch_probe`,
`detail_flicker_probe`, `chevron_census`.

**LS6. Judging a claim.** Where a reader judges a claim, its cited passages open beside it with
the cited span marked, and each passage names and opens the findings that cite it. *Held by:*
review.

**LS7. A session reads as an issue.** Its question is the title; exchanges are comments in time
order; product observations are one-line events; members filter the thread and its facts sit in
the head. A thread has two flat levels, a comment and its replies at one indent on one light
axis, with no elbows; who answers whom is the kept `→ @name`. *Held by:* `--thread-axis`,
`workbench_team.cjs`, `thread_fold_probe`.

**LS8. A thread shows only the record.** It counts only what the owner records, in recorded
order: no likes, views, ranks or runtime states, time order only, the newest reached by the
`n new` pill; Display filters and sets density but never reorders. Consecutive hooks and
observations fold to one counted line with their real time span; a long run of replies keeps its
first and last with `Show n more` between; pressing an exchange opens its record in place. The
Main PM's recorded response sits under the question it answers as a neutral inset marked `PM`,
and never marks an objection resolved. *Held by:* `thread_fold_probe`, `workbench_team.cjs`.

**LS9. The ways into Team by role.** The dock keeps one Sessions row; a role is a way into it,
never a dock row of its own (PG1). The Sessions lobby shows each session's specialists (Roles)
and filters by them (`Participant is CRO`): the Main PM leads every session, so it is no filter,
and a role's two cards' spellings are one role. An object's Facts names the Team sessions whose
declared references name it exactly -- a study, a book, an Evidence book's review or analysis --
each with its specialists and the way to it; none is said, never guessed (the maintainer, 2026-09-26:
the PM, the Alternative analyst and the CRO as ways into Team sessions). *Held by:*
`collaborationRows`, `LiveTeam.sessionsNaming`, the books scene's Team sessions
(`team_evidence.py`), the walk.

## LY. Layers and glass

**LY1. One glass for everything that floats.** The detail, dialogs, menus, pickers, the toast,
the peek, tooltips, the drawer, the pill and every warning wear `material.glass`: a translucent
fill over a frosted backdrop, the light inside it, the rim's light seeping in, a 1 px rim ring,
an outer hairline and the overlay shadow. Glass is the default; Opaque controls (a saved
setting) turns it off. *Held by:* `material.glass`, `glass_census`.

**LY2. How a host wears it.** A host that holds a `position: fixed` menu or picker (the detail,
a dialog, the drawer) wears the glass as a layer (`::after`, the ring on `::before`), never as a
filter on itself, because a filter makes it the frame of its fixed descendants; any other host
wears it as its face. A glass host never animates its own opacity: its parts fade and the host
scales, so the frost shows through the entrance. *Held by:* `glass_census`, `glass_fade_probe`.

**LY3. Inside the glass everything is glass.** A box keeps only its outline, a field or chip
takes the glass control's tint, a well is the glass, a warning is flat; only a primary press, a
mark or a key stays solid. *Held by:* mode `glass-parts`, `glass_census`.

**LY4. Covering content.** Nothing dims, masks or covers content at rest: no scroller fades its
edge, and a way shown on hover takes its own slot, never a layer over words. The named
exceptions: a modal that asks for a decision keeps half the old dim (`--backdrop`) and adds a
10 px fog (`--backdrop-blur`); a sheet that is only read (Keyboard shortcuts) has no scrim and
closes on an outside press; a floating detail frosts what lies under it; a slow re-read dims the
busy lane (ST5). *Held by:* `role.depth`, `reading_switches_probe`.

**LY5. Dialogs.** A dialog's X in its head is its way out; its foot holds only its verbs, the
confirming one primary, never a second Close, Cancel or Back. A short question with two or three
answers is an alert: a bold title, one line of body, equal pills sized by their answers. A
confirmation is key-value facts (what the owner reads, from where, under what, what never runs)
and one sentence of what confirming admits; the exact request stays inside, folded. Quick Open
is a palette: no head, no X, Esc closes it. *Held by:* review.

**LY6. One tooltip.** `data-tip` (its chord in `data-tip-key`), shown after 400 ms, at once within
300 ms of the last or on keyboard focus, gone on leave, blur, Esc, press or scroll; a top-layer
tip at most 320 px, placed in layout coordinates, worn as glass. A tip never repeats words the
reader has: a dock row has none -- its word is the row, and its chord shows at its end under the
pointer or the focus, the word giving way; a tip in the dock (a rail glyph, an icon) stands beside
it over the lane, level with it -- under a row it covered the next. The dock opens at 216 px, every
word whole, and drags to 180 (the maintainer, 2026-09-25). *Held by:* `workbench_dom.cjs` (a tip on focus,
gone on Escape), `dock_tips_probe`.

**LY7. A toast confirms what the person did, or says a Task ended.** Copied, exported, cleared,
sent: shown 3.5-8 s inside the window, clear of a right dock. A Task that stops moving in Task
Control's report is said once, by its name and its state's word, with one press to open it (the
followed Task says its own ending); an agent's event is counted as unseen, never toasted.
*Held by:* `workbench_activity.cjs` (a Task's ending), review.

**LY8. The configuration without glass is recorded.** Every `material.glass` value carries its
`opaque` value; Opaque controls keeps the solid dim with no fog; Reduce focus effects removes
both. *Held by:* the gate,
`test_workbench_parameters.py::test_the_gate_refuses_a_glass_value_without_its_opaque_one`,
`reading_switches_probe`.

## WD. Words

**WD1. A name a person would say.** The owner's label is the object's one name in its list, its
page, the path, pickers and every area. Without one, a per-kind rule composes it from owner
facts (a session: the verb table, else its first sentence cut at `--session-title-chars`, 28);
look-alike names show the fact that differs; one word names one thing. An actor is named as who
it is (You, Codex, Claude Code, a subagent's name, Unidentified). *Held by:*
`object_title_census`,
`test_workbench_readback.py::test_submission_words_name_submissions_in_page_labels_and_chinese`
(V666/U123: a revision's recorder is not a completion submitter).

**WD2. Codes are words.** Every code is a word (`codeWords`), a method its words with the
handle in mono beside, an owner's code its sentence (the code on hover), an attribution its role;
a count carries its noun by number, never `(s)`. Internal nouns are translated once (a packet is
the prepared evidence, a unit a group, a cell issuer x topic); the Glossary holds every noun a
page keeps. Every composed owner code has a declared word; an undeclared one reads
`Word not declared`, retaining the exact value in its tip or Facts, never a humanized fallback
(V659/U117). A literal the person types (a process switch, `ALPHALATTICE_NETWORK_DISABLED=1`) is not a
code in the words: it stands in the sentence as `code.literal`, mono, breaking anywhere at the split
width (V650: walk at 900, 1100 for a dense page). *Held by:* `codeWords`, `methodWords`, `s_census` (raw codes), `pages_walk` (raw codes, a
literal excepted),
`test_workbench_owner_words.py::test_every_composed_closed_owner_value_has_declared_bilingual_words`
(declared groups, unknown codes and the retired fallback).

**WD3. An id is a property.** At most one 8-hex reference sits beside a name, in mono; a hash
appears otherwise only in Facts, the record or a Copy control. A row leads with words and its
handle follows, small. An identity or hash shows as its reference (8 hex, 12 for a content hash)
and a locator or path by its ends (host … last segment), the whole on hover and in its copy glyph
beside -- never a value with its own scrollbar (a box may roll as a whole, FT5), never wrapped
mid-token, never cut without its copy (the maintainer, 2026-09-26: a scrollbar under every hash of a
Task's receipt was the 2026-09-22 roll misapplied to each value); phrases wrap rather than
ellipsize. A
name the owner gives (a policy, a recipe, a rule set) is a code, not an identity: it reads as its
words, the code on hover and its copy glyph beside, and never rolls (the maintainer, 2026-09-25:
a policy's name had grown its own scrollbar). *Held by:* `refCell`, `hashCell`, `codeCell`, `s_census`,
`clip_census`, `workbench_minor_ui.cjs` (V663/U120: plain and slotted hint phrases;
V664/U121: full-value tooltips remain reachable), `workbench_routes.cjs`
(composed refusal locators retain their exact copy).

**WD4. Few words.** An overview's first screen carries at most 130 of the product's words (the
Evidence Overview, where the number was fitted), a view's at most 180 -- Home's lists among them
(the maintainer, 2026-09-30, at 164 words: not dense); figures and closed folds are not counted, and the owner's text
(`.owner-text`) is content (three lines with More). At most twelve distinct evidence words appear
before a page's first disclosure. A grey line over 90 characters keeps its first clause and
folds the rest behind (i). *Held by:* `pages_walk` R4, `evidence_census`,
`Controls.foldLines`.

**WD5. Time in the reader's zone.** Every instant is in the reader's zone: the time alone within
the reader's day, the date before it otherwise, the same on every page; UTC only in Facts,
marked `UTC`; a market session is a date; no "ago" except the owner's own age words. The product
runs as a development replay, so no text makes a claim about the present (a relative day is its
date). *Held by:* `n3_census`, `test_local_web_product.py`
(`test_development_replay_copy_makes_no_forward_claim`).

**WD6. One evidence language.** The evidence table is the one source of names, readings and
states; `evidenceRow` is every evidence object's row and `citePill` a handle's one shape, with
its hover card. *Held by:* `EVIDENCE`, `evidence_census`.

**WD7. Every product word is translated.** Every product word goes through `t()` and has its zh
entry; an entry nothing uses is removed. Product sentences composed by an owner arrive as
fields for a keyed sentence or have an exact catalog template; authored source and participant
words keep their spelling (V671/U127). Catalog refusal guidance uses that catalog whether or
not a page lists its code; the owner's actual explanation precedes a code-only fallback,
with commands and their metavariables kept literal (V674/U130). A scope is said in neutral
terms: what a view reads (a window, a cadence, an order), never how much a feature matters.
*Held by:* `zh_keys` (the phrases asked for at run time) with the private `zh_dead` check,
review, `test_workbench_readback.py::test_every_chinese_translation_uses_its_owned_ui_vocabulary`
(V656/U115 and V670/U126: ordinary English nouns fail; reasoned literals stay exact),
`test_workbench_readback.py::test_every_generated_evidence_sentence_reads_in_chinese`,
`test_workbench_readback.py::test_evidence_sentence_census_discovers_new_outputs_and_raw_readers`,
`test_workbench_readback.py::test_every_door_refusal_reads_in_chinese`
(the real refusal renderer, including codes outside the page's list).

## AC. Motion and access

**AC1. Motion.** A page change is a cut, readable at its first frame; every leave is its
entrance reversed over `--dur-1`. Three durations (100 / 150 / 240 ms) and a linger, two
curves. Reduced motion makes each an instant change. A view operation paints under 25 ms, with
no frame over 20 ms. *Held by:* `scale.motion`, `11-accessibility.css` (reduced motion); the
paint budget by review (round 94 measured it).

**AC2. The first paint is the reader's.** A synchronous prelude sets theme, language, dock,
layout and text size before paint; the zh dictionary loads only for zh; assets are
content-hashed and immutable, the page and the API `no-store`; a failed render draws the refusal
card. *Held by:* `js/app/prelude.js`, `test_local_web_product.py`, the browser test.

**AC3. Text size and viewport.** Text size is the product's own zoom (80-175 %, `Ctrl =`, `-`,
`0`); viewport lengths read `--vh` / `--vw`, which divide by it. *Held by:* `role.frame`.

**AC4. The reading switches do what they say.** Opaque controls (saved, restored at boot),
Reduce focus effects, Wider scrollbars and reduced motion each change what their line says.
Increased contrast deepens the inks and every edge. *Held by:* `reading_switches_probe`, the
`contrast` modes.

**AC5. The keyboard.** One map (`KEYMAP`, controls.js) binds the keys and draws the shortcuts
sheet, and documents the keys another owner binds (the composer, a chart, the dock's edge). `G`
then a letter goes to every place the dock lists, named by the dock's word; a key whose target
the UI dropped leaves the map with it. A list walks by `J` / `K`; Space shows the hover card and
Enter opens; Esc steps out; `[` and `]` step through the list the record was opened from, or the
list its detail stands beside -- another object reached after it forgets the list; `⌘K` opens Quick Open. The focus ring is 2 px of
the accent 2 px outside (inside a strip that clips), only on `:focus-visible`.
*Held by:* `role.focus`, the browser test, `keys_probe`.

## PR. How the rules are kept

**PR1. A value has one owner.** Every design value lives once in `design/parameters.json` with
its reason and modes; the build writes it as the first stylesheet and hands it to the scripts. A
value someone chose is a parameter; a value that follows from others is a formula over
parameters; 0, a hairline, 100 %, centring, a flex ratio and an icon's own geometry are
structure; anything else is a named exception with its reason
(`design/literal-exceptions.json`). The gate refuses a re-declared or unknown property, an
unread parameter, a dangling `var()`, a width off the breakpoint ladder and any unseen literal;
the ratchet (`design/literals-baseline.json`) only shrinks. *Held by:* the gate,
`test_workbench_parameters.py`. The maintainer, 2026-10-04: these rules are
parameterised, never set by hand. Alignment follows the shared builder's layout and token gaps, never a
page's hand-tuned offset; measurable rules have a parameter, formula, builder or
check as their holder (PR3).

**PR2. A sheet places, a builder shapes.** A chapter's sheet places components (areas, order,
column) and never sizes, colours or spaces them; a shape a page needs is a builder in
`components.js` with a specimen in the workshop; a selector belongs to one sheet.
*Held by:* the gate's literal ratchet, review.

**PR3. A rule is held where it lives.** An enforceable rule is enforced by a parameter, a CSS
rule, a builder, a test or a check of the kit, and measured on every page by the walk; prose
records a rule but does not hold it. A defect seen on one page is a question about its class:
find the cause, fix it where it lives, then check every scene, page and state.
*Held by:* this book's "Held by" lines.

**PR4. Borrow legally.** Values come from several products measured as a user sees them, or
from an open licence (Primer, MIT); never from a kit whose licence excludes this product, never
an asset or an identity; each borrowed value's source is recorded. *Held by:* review.

**PR5. Judge with real eyes.** A change is judged on stills at the desktop matrix (1470 / 1180 / 900
wide, Light / Dark, en / zh, 100 / 125 %) and in the maintainer's window (1673 x 1186 at 125 %),
with the classic scrollbar the maintainer's Windows shows (the kit's default; Playwright hides it, and
a value's roll or a picker's cut title that looked clean in a still grew a bar in that window,
2026-09-25/26).
The glass is judged in real Chrome (`QA_CHROME`): the headless shell drops a large element's
frost at device scale 2 and overlays its scrollbars. The matrix judges this desktop workstation's
widths (PG10, V650); a narrow walk is at 900, and at 1100 for a dense page. *Held by:* the kit's
stills and `viewport_widths`.

**PR6. A rule the maintainer states is kept with its reason.** It enters this book in English,
with its holder and the date of the reading that made it. *Held by:* review.

**PR7. A merged UI is this book's.** Another line's UI is a reference for behaviour, never for
shape: every hunk a merge brings in, and every item built from its reference, is checked one by
one against these rules and rebuilt in this product's shapes (a count is a tile with its meter, a
row's action its trailing column, a sealed limit the Report's fold, a setting's line what it does)
(the maintainer, 2026-09-25: a merged UI is never taken as it comes; the patterns this product
found are the standard, and every merged item is checked one by one).
*Held by:* review, the kit's walk on every scene the merge touches.

## What this book settled

Where the old laws disagreed, the rule above follows what the product does today (measured on
the scenes, 2026-09-24):

- Every block a card (2) gave way to canvas by default (MA2); a card's foot holding the only
  buttons (9) to a row's `···` and a form row's control (LS2, CT4).
- Figure sizes 20 / 30 / 44 (6) and a 12 px floor (26) gave way to the ladder and the 13 px floor
  (TY1, TY3); a table head at 600 (37) to the muted callout at 500 (FT4); an object's name at
  24 / 600 (120) to the large title at 500 (TY1, TY2).
- A region's thin accent (29) and the accent on the selected state (48) gave way to neutral
  selection and tone only on marks (CO1); a badge's icon (12) to no icon beside a state (ST2); a
  neutral state's grey dot (65) to no dot (ST2).
- The browser `title` for codes and explanations (17, 68, 73) gave way to `data-tip` (LY6, CT7).
- Growing blocks capped in pixels (10) gave way to paging, folding and counting (PG8); lists at
  twenty a page (103) to lobbies of six per group (LS1) and tables at fifty (FT6).
- No dock, drawer, tabs or console (40) gave way to the frame (PG1), tabs (PG7) and the console
  (PG8); a global bar of eight entries (21) to the dock; an in-product `←` (68, 86) to the
  browser's Back (PG2); the path without its group (122, 87) to group / list / object / folder
  (PG3); a scope's pages as dock rows (123, 132) to a list's views as tabs (PG7).
- A state lede on every page (94) gave way to no product sentence under a title (PG5); pills for
  views (95) to tabs (PG7); a properties column (95) and Team's side column (125) to one column
  (PG9); an outline in a side box (127) to the document's own margin (PG9).
- The published decision as its page's one box (99) gave way to canvas by default and one fact
  on one page (MA2, PG11).
- "Nothing dims or covers content" (112) now names its exceptions (LY4); the reading in the
  inspector (62, 78) became the detail beside its list (LS5); Space and Enter on one Task body
  (46) became Space for the hover card and Enter to open (AC5).
- One box material (147) and the notice on the box ground (128) learned the glass: a warning
  floats (LY1, CO4); a bar's 480 px cap (89) exempts a composition or threshold gauge (FT7).
- Quick Open's missing X (84 against 131) is the palette's exception (LY5).

## Appendix: the old numbers

Laws 1-83 come from the design round of 2026-09-18 and 84-150 from the rounds of
2026-09-20..24. Code comments still cite the old numbers; this table is how they resolve here. Retired
means its rule is gone and nothing carries it.

| Old | Now | Old | Now | Old | Now | Old | Now |
|---|---|---|---|---|---|---|---|
| 1 | MA1, MA3 | 39 | CO5 | 77 | AC2 | 115 | LS3 |
| 2 | MA2 | 40 | PG1 | 78 | LS5 | 116 | PR1 |
| 3 | MA5 | 41 | PG2 | 79 | CT3 | 117 | ST4 |
| 4 | TY1 | 42 | PG6 | 80 | WD2 | 118 | PG6 |
| 5 | TY2 | 43 | LS5, ST6 | 81 | ST7, ST2 | 119 | PG3, PG5 |
| 6 | TY1, FT1 | 44 | LS1, LS2 | 82 | WD1 | 120 | PG5 |
| 7 | WD3 | 45 | WD1, WD2 | 83 | ST8, CT2 | 121 | PG5 |
| 8 | TY4 | 46 | LS4, AC5 | 84 | LY6, LY5 | 122 | PG3 |
| 9 | TY5 | 47 | ST1, LS8 | 85 | TY4 | 123 | PG1, LS1 |
| 10 | PG8 | 48 | MA1, CO1 | 86 | PG2 | 124 | PG2 |
| 11 | TY5 | 49 | CT4 | 87 | PG4 | 125 | LS7 |
| 12 | CO1 | 50 | ST2 | 88 | WD3 | 126 | PG9 |
| 13 | MA6 | 51 | ST2 | 89 | FT7 | 127 | PG9 |
| 14 | CT1, CT2, CT7 | 52 | ST3 | 90 | FT5 | 128 | CO4 |
| 15 | CT2 | 53 | ST5 | 91 | ST5 | 129 | PG1 |
| 16 | ST3 | 54 | LS4 | 92 | FT6 | 130 | ST8 |
| 17 | ST1, ST6 | 55 | LS4 | 93 | LY5 | 131 | LY5, LS5 |
| 18 | FT1, FT8 | 56 | PG13 | 94 | PG6 | 132 | PG7 |
| 19 | WD6 | 57 | ST3 | 95 | PG6, PG7 | 133 | WD5 |
| 20 | retired (MA1); its motion text AC1 | 58 | ST6 | 96 | WD3 | 134 | WD1 |
| 21 | PG10 | 59 | LY7 | 97 | WD2 | 135 | PG1, PG3, PG7 |
| 22 | MA2 | 60 | WD6 | 98 | LS4 | 136 | LS1 |
| 23 | MA2 | 61 | WD4 | 99 | retired (MA2, PG11) | 137 | PG5 |
| 24 | MA4 | 62 | LS5, AC5 | 100 | PR3 | 138 | MA1, MA3 |
| 25 | PG12 | 63 | PG11 (the Report's anatomy is its builder's) | 101 | FT2 | 139 | LS7 |
| 26 | TY3 | 64 | PG6 | 102 | LS2 | 140 | LS8 |
| 27 | PG12 | 65 | CO1, CO2, CT6 | 103 | LS1, FT6 | 141 | LS8 |
| 28 | PG12 | 66 | MA1, MA3 | 104 | WD4 | 142 | LS8 |
| 29 | CO1 | 67 | CT2, CT3, MA6 | 105 | PR1 | 143 | PG11 |
| 30 | MA2 | 68 | PG1, CT5, AC1 | 106 | CT1, PR2 | 144 | LS6 |
| 31 | retired (restated 21: PG10) | 69 | PG1, AC3 | 107 | TY5 | 145 | PG13 |
| 32 | FT3 | 70 | CT4 | 108 | PR2 | 146 | FT8 |
| 33 | FT1 | 71 | LY5 | 109 | PR4 | 147 | MA3 |
| 34 | FT3 | 72 | MA4, LS4 | 110 | PG7 | 148 | MA2, MA4 |
| 35 | FT3 | 73 | LS2 | 111 | PR1 | 149 | LS5 |
| 36 | FT3 | 74 | ST1 | 112 | LY4 | 150 | LY1 |
| 37 | FT4 | 75 | TY1-TY3, WD4 | 113 | PG8 | 150.1-150.2 | LY2 |
| 38 | FT4 | 76 | CO1, CO3 | 114 | LS2 | 150.3 | LY3 |

150.4 and 150.5 are CT8; 150.6 is LY4; 150.7 is LY8; 150.8 is AC4; 150.9 is PR5.
