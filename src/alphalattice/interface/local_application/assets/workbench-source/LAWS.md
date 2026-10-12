# The Local Web UI's laws

This book defines the Workbench's design rules. Each rule identifies its source,
its adaptation and its existing enforcement; product-specific rules also state
their reason. Law ids are stable, and the appendix preserves earlier numeric
references. Workbench PR1-PR7 are distinct from the product law book's ids.

## Index by source

| Source | Laws |
|---|---|
| Classic principles | `MA2`, `MA4`, `MA6`, `TY2`, `TY4`, `TY5`, `TY6`, `CO1`, `CT3`, `CT6`, `CT7`, `CT8`, `ST2`, `ST4`, `ST5`, `ST6`, `ST8`, `ST10`, `FT1`, `FT2`, `FT6`, `FT7`, `FT8`, `PG3`, `PG4`, `PG6`, `PG8`, `PG9`, `PG11`, `LS3`, `LS5`, `LY4`, `LY7`, `LY8`, `WD1`, `WD2`, `WD4`, `WD5`, `WD7`, `AC2`, `AC3`, `AC4`, `PR1`, `PR2`, `PR3`, `PR4` |
| Borrowed products and standards | `MA1`, `MA3`, `MA5`, `TY1`, `TY3`, `CO2`, `CO4`, `CO5`, `CT1`, `CT2`, `CT4`, `CT5`, `FT3`, `FT4`, `FT5`, `PG1`, `PG5`, `PG7`, `PG13`, `LS1`, `LS2`, `LS7`, `LY1`, `LY2`, `LY3`, `LY5`, `LY6`, `WD3`, `AC1`, `AC5` |
| Product-specific rules | `CO3`, `CO6`, `ST1`, `ST3`, `ST7`, `ST9`, `ST11`, `PG2`, `PG10`, `PG12`, `LS4`, `LS6`, `LS8`, `LS9`, `WD6`, `PR5`, `PR6`, `PR7` |

Holders identify a parameter, builder, owner table, check or review. The product
gate is `scripts/build_local_web_ui.py --product --check`; private QA checks
provide the named measurements. A review-only holder is not an automated proof.
Borrowed sources identify provenance, not wholesale adoption or a claim of
standards conformance. Dimensions and timing thresholds in adaptations are
Workbench choices unless explicitly attributed to a standard.

The Workbench is a desktop research workstation. Its visual system uses neutral
grounds, readable typography, semantic colour and one material for floating
surfaces. Owners supply research facts and state; the interface does not infer them.

## MA. Material and ground

**MA1. Neutral grounds and appearances.**

*Source:* borrowed: Apple HIG and Linear\
*Adapted:* Host-following appearances use role.ground; group heads rise above the ground, recessed wells below it.

Each page has one ground, without gradient, glow, texture or artwork.

*Held by:* `role.ground`, `role.ink`.

**MA2. Canvas by default.**

*Source:* classic: Gestalt proximity and common region\
*Adapted:* Canvas by default; data-box means a figure, decision or blocker, workspace, or table.

Headings, facts, lists and paragraphs lie on the ground. Structure uses whitespace first, a
hairline second and containment last.

*Held by:* `box_budget_census`, `box_purpose_audit`.

**MA3. One box material.**

*Source:* borrowed: Linear\
*Adapted:* Shared [data-box] material in 04-surfaces.css, --edge-card and --radius-card.

Components never define independent box material. Flatten sections laid on the ground.

*Held by:* `role.depth`, `box_material_census`.

**MA4. Containment depth.**

*Source:* classic: Gestalt common region\
*Adapted:* Three depths: canvas, box, raised layer; box insets are at least 8 px horizontally and 4 px vertically.

A box never contains another card: an inset detail is a --surface-well. Its contents never
spill or clip.

*Held by:* `box_inset_census`, `control_spill_census`, `clip_census`,
`test_workbench_readback.py::test_collection_box_census_reads_decision_ancestry_in_the_pinned_browser`
(direct, wrapped, closed-fold and floating-reader ancestry).

**MA5. Hairlines.**

*Source:* borrowed: WCAG 2.2 Non-text Contrast\
*Adapted:* Rows use --edge-hair; increased contrast requires 3:1 against the actual ground, including composited alpha.

An interior line is never stronger than its box edge. A matching token alone does not prove
contrast.

*Held by:* `role.line`, `edge_contrast_census` (theme tokens matching their
declarations does not prove the alpha-composited edge reaches 3:1 on its ground),
`workbench_minor_ui.cjs` (weak solid and glass edges fail;
an undecidable backdrop is named, never certified).

**MA6. Radii.**

*Source:* classic: design tokens\
*Adapted:* scale.radius assigns card, dialog, control, inset-box, micro and pill shapes.

Roles select the shared radius; pages never adjust it.

*Held by:* `scale.radius`.

## TY. Type and space

**TY1. One text ladder.**

*Source:* borrowed: Apple HIG\
*Adapted:* scale.type pairs size and leading by role; Chinese increases leading and removes tracking.

Every text run uses one shared style. Nothing is uppercase.

*Held by:* `scale.type`, `pages_walk` R5,
`ink_census`.

**TY2. Three weights.**

*Source:* classic: typographic hierarchy\
*Adapted:* --weight-* assigns 400 body, 500 titles and labels, 600 group labels, report subheads and alert titles; Windows Chinese titles use Bold because YaHei has no Medium.

Apply semantic weights consistently. Bold is not an extra emphasis style.

*Held by:* `--weight-*`, `ink_census`.

**TY3. Readable content.**

*Source:* borrowed: Apple HIG and Linear\
*Adapted:* Read content has a 13 px floor; 12 px is chrome. --muted-mix is 80% Light, 72% Dark, stronger in increased contrast and remixed per box; notices use min(--width-xs, available column).

Content is never small and grey together; density uses wrapping or disclosure. Actions wrap
before squeezing explanations, and short labels keep their natural measure.

*Held by:* `role.ink`,
`ink_census`, `box_inset_census` (text measure, including zero-width blocks),
`workbench_minor_ui.cjs` (the notice families and the former zero-width rule).

**TY4. Shared spacing.**

*Source:* classic: Gestalt proximity\
*Adapted:* scale.space (2–64) owns every gap, pad and margin: text 8, control 12, box on either side 16, heading 20.

Headings attach to their content; no spacing classes exist. The shared stack and census
measure visible neighbours through wrappers and grid columns; templates, scripts, styles and
hidden children add no step.

*Held by:* the stack (`04-surfaces`, its steps
`stack-*` in `component.boxes`), `stack_census` (a control 12 after anything, a form's box 16 before
or after, on every page at the split width 900 and desktop width 1470), `scale.space`, the gate's literal ratchet,
`test_workbench_readback.py::test_stack_census_reads_page_boxes_through_every_wrapper`
(planted zero gaps, unknown wrappers, plain panels, inert templates and grid columns;
explicit neutral carriers restore the parameter's step without adding painted boxes).

**TY5. Shared alignment.**

*Source:* classic: Gestalt good continuation\
*Adapted:* Shared 24 px lead column and title gap; no more than eight text starts.

Side-by-side blocks share a top edge and heading line. Paired boxes share width and height.

*Held by:* `role.spine`, `align_census`.

**TY6. Glyph alignment.**

*Source:* classic: Gestalt proximity\
*Adapted:* glyphWord centres glyph and word on their line; leading glyphs use the first line's height.

Never align by a hand-set nudge or the whole block's height. Hover controls never increase
line height.

*Held by:* `glyphWord`, `glyph_align_census` (every glyph with words
within 1 px of their line's middle).

## CO. Colour

**CO1. Semantic colour.**

*Source:* classic: semantic encoding\
*Adapted:* Seven role.tone pairs; neutral selection, accent picker highlights at 12% Light and 18% Dark.

Use colour on semantic marks, switches, focus, links, charts, badge tints and signed
performance. Never tone grounds, borders, body text, primary buttons or unsigned numbers.

*Held by:* `role.tone`, `role.fill`, `color_census`.

**CO2. Colour on marks.**

*Source:* borrowed: GitHub Primer Counter\
*Adapted:* tile() marks and state dots; badges pair a dot with a text-ink word on tone tint; faint-wash dock counts or solid warning counts.

Colour belongs to a row's mark. Only links, references and code colour words.

*Held by:* `color_census`, `counter-*`.

**CO3. Domains keep their colour.**

*Source:* ours\
*Adapted:* KIND_MARKS and ROLE_TONES own the domain palette; rails are green done, text ink with halo current, and ST2 tone stopped.\
*Why:* Our domain colours identify speakers and fields, rather than universal states.

Apply role colour to marks, never whole rows; replies retain their speaker's avatar colour.

*Held by:* `KIND_MARKS`, `ROLE_TONES`,
`stepList`'s `data-tone`.

**CO4. Notice tone.**

*Source:* borrowed: WCAG 2.2 Use of Color\
*Adapted:* Notices use a tone mark with text-ink words and small muted code beneath; warnings float as LY1 glass.

Marks and words carry meaning together. Notice grounds never communicate tone.

*Held by:* `box_material_census`, `glass_census`.

**CO5. Code in GitHub's colours.**

*Source:* borrowed: GitHub syntax highlighting\
*Adapted:* Shared --code-* palette: green keys, deep-blue strings, brighter-blue constants, muted comments and text-ink punctuation.

Paint beneath the textarea, retaining it as the editor.

*Held by:* `--code-*`.

**CO6. A colour is a meaning.**

*Source:* ours\
*Adapted:* role.meaning/TONE: failure red, attention amber, decision purple, done green, active blue, stopped grey dot, rest uncoloured.\
*Why:* A valid stop needing attention differs from failed execution.

Callers name meanings, not colours. Red is failed work, read, render or check; blocked/refused
work, refused steps, blocking findings, incompatible comparisons, exceeded budgets and HIGH
severity are attention.

*Held by:* `role.meaning`,
`test_workbench_parameters.py::test_red_is_failure_alone`.

## CT. Controls

**CT1. One control ladder.**

*Source:* borrowed: GitHub Primer\
*Adapted:* Primer heights 24, 28, 32, 40 and 48 px; context selects, padding is derived. Fitts's law motivates usable target sizes.

No other rule sets control height, and longer words never increase it. The workshop shows each
component at its step.

*Held by:* `scale.control`, `component.sizes`, `workbench_dom.cjs` (a control off the ladder fails).

**CT2. Buttons.**

*Source:* borrowed: Apple HIG\
*Adapted:* Buttons/search fields are pills; primary ink works on both grounds, one per page/region; panel confirmation is plain.

Icon buttons are bare glyphs with neutral hover. Boxes have no local Refresh, and hover never
flashes.

*Held by:* `workbench_dom.cjs` (one primary per page).

**CT3. Choosing.**

*Source:* classic: Hick–Hyman law\
*Adapted:* Booleans use 38 × 22 switches, accent when on; few values use trackless neutral-selected segments. Linear's filter begins above eight rows; the 120-row window is a rendering bound, not a choice-law threshold; sheets follow BREAKPOINTS.window.phone.

Other choices use picker(), never select: value trigger, left-aligned layer, one-line rows and
selected check; empty pickers stay held, explain dependent verbs and open no empty list. Open
lists retain size during scrolling, and classic scrollbars widen them without clipping titles.

*Held by:* `picker()`, `Picker.fit`, `workbench_dom.cjs`,
`checks/picker_scroll_probe.cjs`.

**CT4. One form shape.**

*Source:* borrowed: Apple HIG\
*Adapted:* Settings groups have a label over one box; rows have title, one supporting line and a right-hand control; free text uses .field.

A unit field uses field's inline layout. Its unit, input, readout and actions align to the
input centre, not a stacked label/input centre.

*Held by:* `field()`,
`control_spill_census`, `field_alignment_probe` (a misaligned field fails the walk), review.

**CT5. Menus.**

*Source:* borrowed: Apple HIG\
*Adapted:* One-line items: glyph, label, note and check; grey unavailable/selected states, focus-visible rings.

Toggles state their current word wherever drawn; presses never end in Action needs attention.
Menu scrolling never closes it.

*Held by:* review, `checks/menu_action_probe.cjs`.

**CT6. Dark controls.**

*Source:* classic: Nielsen consistency and standards\
*Adapted:* Dark pickers, buttons, stepper buttons and chips use --ui-line and --ui-face; text fields use the face only.

A chip is a control, never a box.

*Held by:* `role.ground`, `role.line`.

**CT7. A held control says why on demand.**

*Source:* classic: Nielsen's error prevention and progressive disclosure\
*Adapted:* data-tip or aria-describedby supplies held reasons; native title and adjacent printed reasons are excluded.

Only a dialog foot prints reasons above its verbs. Holding a control never moves its row.

*Held by:* `Controls.setReason` (the note is for assistive technology), `workbench_dom.cjs` (no `title` on a
control).

**CT8. Visible feedback.**

*Source:* classic: Nielsen visibility of system status\
*Adapted:* Tinted controls define hover fill/edge; feedback moves fill luminance at least 2% or edge 4%; glass state uses outline.

Interactions give visible feedback without altering material-owned properties. Resting
controls do not use hover or selected styling.

*Held by:* `--glass-control-hover`, `glass_hover_census`.

## ST. States and truth

**ST1. Owned facts.**

*Source:* ours\
*Adapted:* Owner facts and binding determine claim limits.\
*Why:* Inferences cannot replace bound research facts.

Every fact has an owner and its exact record one disclosure away. The UI computes no fact and
invents no name, date, state, relation or event, including in labelled QA fixtures. Objects
come from their owner, never a feed tail. Positive checks name their checked set, scope and
limit; empty sets cannot be verified and reference integrity does not certify research.
Performance reads the selected owner window, its source, as-of date and each metric or
recorded absence. Pinned readings remain pinned; latest realized readings follow published
outcomes without future proposal returns. Formation and observed-outcome ranges are distinct.
Only completed holding periods supply returns; open periods remain separate. Coverage and
floor comparisons use owner accounting, separating material reviewed, official no new filing
and unaccounted weight. Historical records lacking accounting supply neither inferred totals
nor floor comparisons.

*Held by:* the harnesses, review, `workbench_reference_checks.cjs` (zero, unknown and mixed sets;
scoped positive words and their limits),
`test_workbench_readback.py::test_goal_page_paints_the_saved_narrative_first_and_verifies_evidence_second`.

*Held by:* `test_cli_contract.py::test_the_installed_report_names_every_absent_performance_metric`,
`workbench_dom.cjs` interaction (selected/latest, publication refresh, absent metrics,
formation versus outcomes and open periods),
`test_rolling_history.py::test_two_updates_append_only_head_projects_the_verified_chain`.

*Held by:* `workbench_dom.cjs` interaction (44 reviewed plus 56 quiet; legacy missing
accounting supplies neither floor result nor meter),
`test_portfolio_review_decision.py::test_a_holding_that_filed_nothing_is_counted_apart_and_never_as_no_risk`.

**ST2. One state table, one state line.**

*Source:* classic: Nielsen's consistency and standards\
*Adapted:* `STATES` owns vocabulary; `stateLine` owns presentation.

Each entry declares its word, CO6 meaning, explanation, next person action and moving or held
state, rendered as dot, whole word, muted duration and next step. Neutral states have no dot,
other dots retain their word, state facts never shrink, surrounding words wrap, icons do not
accompany states and badges are used only to prevent ambiguity.

*Held by:* `STATES`, `stateLine`, `.list-row-props > span:has(> .state)`,
`scripts/count_ui_laws.py`.

**ST3. Owner-reported progress.**

*Source:* ours\
*Adapted:* Task Control owns progress and elapsed time.\
*Why:* Animation cannot imply unreported work.

Only a reported moving Task animates its dot; at rest only the live dot moves. Unknown extent
is a still dashed track, never a time bar. Ended durations use owner times; moving durations
use the finite, nonnegative owner span as read, with an empty slot when absent. The browser
clock never extends it. A wait names its dependency, start and next event.

*Held by:* `test_workbench_task_duration.py` (owner spans, absent spans, selected Task identity and a
browser clock months away from the owner clock).

**ST4. Canonical Task state.**

*Source:* classic: Nielsen's visibility of system status\
*Adapted:* Owner changes repaint on the activity cadence.

Lists read `Data.lifecycleOf` and reread once on a reported move. Activation, schedule and
daily-standing pages follow external changes without Reload or another mutation, preserving
the address and sealed reading.

*Held by:* `test_workbench_readback.py::test_listed_task_state_follows_task_control_without_a_reload`,
`workbench_routes.cjs` (external state changes on Portfolio, the book and Settings;
the address and sealed reading stay put).

**ST5. Reading in place.**

*Source:* classic: Nielsen's response-time limits\
*Adapted:* Stable rereads use the 300 ms and 400 ms feedback thresholds.

Initial reads show delayed skeletons, never spinners or `Loading`; rereading the displayed
object retains its head and content, dims its `aria-busy` lane after 300 ms and reserves the
400 ms skeleton for another book. Collections discover metadata rather than full objects,
selection verifies the object, and page cancellation ends page reads while retaining
application observations and admitted work.

*Held by:* the browser test
in `test_workbench_readback.py` (repaints keep scroll, focus and open folds),
`workbench_collection_reads.cjs` (collections discover metadata; no read outlives its page).

**ST6. Refusal and recovery.**

*Source:* classic: Nielsen's error recognition, diagnosis and recovery\
*Adapted:* Owner recovery offers remain visible beside readable peers.

State is said once, with the owner cause, hover code and lawful `STOP_WAYS` action whole and
clickable; the record persists, and only a primary blocker gets a banner. Host corrections
retain problems, authored answers and remaining rounds as attention, while collection failures
retain successful peers, workspace admission and independent readers beside their named
refusal. An unsuccessful reread keeps the last successful rows, with the refusal.

*Held by:* review, `live-tasks.js` `runBody`,
`workbench_handoffs.cjs` (the CORRECT round, a cold History refusal and failed Task resync),
`workbench_workspace.cjs` (independent Backups), `clip_census`, `control_spill_census`,
`test_workbench_readback.py::test_home_keeps_each_unreadable_tasks_recovery_ways_clickable`.

**ST7. Unknown values.**

*Source:* ours\
*Adapted:* Unknown facts remain absent.\
*Why:* Placeholders must not manufacture facts.

Unknown facts leave slots empty, rows absent and unknown clauses omitted, never dashes,
placeholders or sentence holes. An absence that is a state is worded once.

*Held by:* `empty_slot_census` (no dash slot and no
hole on any scene page), `workbench_dom.cjs` (no placeholder on any route).

**ST8. Empty states.**

*Source:* classic: progressive disclosure\
*Adapted:* One empty state, in place, with one next step.

An empty list shows the product mark, at most one sentence naming its contents and one way to
create them. A section or box shows only its line.

*Held by:* `empty_census`, `workbench_dom.cjs`.

**ST9. Person attention.**

*Source:* ours\
*Adapted:* Person decisions alone feed attention; exact successors clear stops.\
*Why:* Resolved work or stops without a decision must not require clearance.

Home owns `LiveViews.needs`; its dock count and Data share precede neutral counts, Tasks
counts actual Task decisions, the rail uses its declared counter geometry, and the document
title begins `(n) `. No event or read marker persists attention after the owner clears it.
Agent-waiting items are excluded; absent owner attribution retains the person default.
Activity retains attention only for the same canonical Task version. Background Task endings
become system notifications only with the Settings permission, requested once with any refusal
on that row. Unrecoverable ledger-rebuilt stops require no decision but keep their reason and
Re-PLAN. Later success of the exact Task kind and sealed plan, or the successor named by an
exact recovery link, clears a stop. Canonical admission order outranks clock order; changed
plans require the link. Tasks, History and Home show the successor once with the earlier stop
folded beneath it. Review `NONE` retains neutral words and limits, with no action or
attention.

*Held by:* `workbench_activity.cjs`,
`test_pending_decisions.py::test_only_a_later_successful_exact_plan_supersedes_a_blocked_task`,
`test_pending_decisions.py::test_a_rebuilt_ledger_task_holds_no_decision_and_keeps_its_stopped_record`,
`test_workbench_badge.py`,
`test_workbench_review_publication.py::test_a_sealed_no_action_review_is_neutral_and_feeds_no_attention`, review.

**ST10. Shared reads and generations.**

*Source:* classic: request coalescing and generation checks\
*Adapted:* Shared GET transport retains each consumer's policy and intent.

One unfinished exact GET yields one consumed wire answer, with caller-specific strict or
refusal policy, distinct selectors, fresh settled reads, no mutation coalescing or replay and
one bound readback per object open. Only current generations accept unique cursor pages and
refusals, never cycle or move backward, supersede older pages on reread and prevent late
answers from replacing newer addressed choices.

*Held by:* `workbench_read_lifetime.cjs` (wire, bound Task readbacks, Portfolio
boot and History races), `workbench_read_collections.cjs` (refresh, Goal and Team races),
`workbench_navigation_intent.cjs` (normal and late action continuations, including Facts and
Record), collected by `test_workbench_readback.py::test_the_workbench_harnesses_pass`.

**ST11. Viewer-controlled following.**

*Source:* ours\
*Adapted:* Following respects attribution and the viewer's choice.\
*Why:* Watching work must preserve reading and editing control.

Launch and exact result or person-decision links retain `follow=goal:<Goal UUID>`, or
`follow=latest` before a session Goal is known. Latest uses the workspace's latest active
Goal; legacy `follow=<Task UUID>` remains exact. Only owner-attributed running progress,
completed results, person decisions and published reviews move the follower. Missing
`waits_on` retains the person default. Each state change moves at most once. Unavailable
decisions retry on the existing cadence without another Task event; exact input family and
cutoff generation distinguish a state. Typing, unsaved drafts and dialogs hold movement, and
late answers cannot override newer choices. Manual navigation pauses across Reload and
Back/Forward with one `Follow again` control. Existing activity and object readers serve
following, never another polling or execution path.

*Held by:* `test_workbench_goal_follow.py` (Task records through events into the real readers, state
changes and controlled latency), and `test_launch_and_answer_links_follow_the_session_goal_through_the_real_browser`.

## FT. Figures and tables

**FT1. Numerical precision.**

*Source:* classic: numerical typography\
*Adapted:* Shared financial precision.

Use one precision per kind, tabular figures aligned right on the decimal and signs only when
meaningful. Percent units accompany cells; report units occupy the header's second line.

*Held by:* review.

**FT2. Figure placement.**

*Source:* classic: Gestalt proximity and similarity\
*Adapted:* `figureTile` owns label/value layout and `--space-2` spacing.

Numbers occupy tiles, meters or table cells, never row prose, captions or ledes. Panels have
one caption of at most 90 characters, and plain or linked tiles stack label above value at one
left start.

*Held by:* `pages_walk` R1, R6,
`figureTile`, `workbench_minor_ui.cjs` (all destination shapes and the former link-only rule).

**FT3. Columns.**

*Source:* borrowed: GitHub Primer tables\
*Adapted:* `component.columns` owns floors and action width.

Tables span their box with words first, figures last and remaining width assigned to
substance. The row action stays in its fixed rightmost column.

*Held by:* `component.columns`, `table_census`.

**FT4. Table anatomy.**

*Source:* borrowed: GitHub Primer tables\
*Adapted:* `component.rows` owns geometry and header type.

Headers align with columns and reveal their sort glyph on hover or when sorted. Cells hold
words and figures, never bars, tiles or chips.

*Held by:* `component.rows`, `table_census`.

**FT5. Horizontal overflow.**

*Source:* borrowed: WCAG 2.2 Reflow\
*Adapted:* The 2D-table exception guides desktop reading; the Workbench makes no WCAG claim.

Irreducible tables, code, JSON, stage rails and property-chip boxes scroll horizontally as one
box on the shared visible track, retaining columns and wrapping substance above its floor
instead of clipping or squeezing it. Values never gain individual scrollbars, narrow logs
stack time and author above words, identities use WD3 and editable fields preserve their
lines.

*Held by:* `table_roll_probe`,
`roll_census` (the boxes that may roll are named there), `scrollers_probe`, `detail_census` (inside
every detail).

**FT6. An unbounded record collection pages.**

*Source:* classic: Hick's law and progressive disclosure\
*Adapted:* `--table-rows` owns the empirical 50-row limit.

Unbounded record collections use indexed pages, footer counts and Previous / Next, adding
search beyond one page while retaining focus and caret, including collections inside
disclosures and details. Folding does not exempt records; lobbies retain LS1 and bounded
tables retain counts without toolbars.

*Held by:* `--table-rows`,
`table_census` (reads the served parameter, including untagged Evidence overflow),
`workbench_handoffs.cjs` (all dynamic Evidence collections; changing the parameter changes
the page size),
`test_workbench_readback.py::test_evidence_collection_census_refuses_untagged_overflow_in_the_pinned_browser`.

**FT7. Shared meters.**

*Source:* classic: Nielsen's consistency and standards\
*Adapted:* `meter()` and `component.meter` own geometry.

Meters use accent fill, pill ends and an ink minimum mark, with shortfall stated in words
rather than segment colour. Share and progress bars are bounded beneath their words,
composition or threshold gauges span the figure and one-part compositions draw no bar.

*Held by:* `component.meter`, `scripts/count_ui_laws.py`.

**FT8. A figure is a box.**

*Source:* classic: Gestalt common region\
*Adapted:* `figureBox()` groups title, legend and drawing.

Figures place name, information control and legend in their head, wrap between whole legend
keys and fill the box width. Charts keep one palette per appearance with crosshair and hover
readings; inline meters remain part of their row or tile.

*Held by:* `figures_probe`.

## PG. Pages and places

**PG1. The frame.**

*Source:* borrowed: Linear's desktop application frame\
*Adapted:* `role.frame` owns dock, rail and drawer.

One top row and lane sit beside a dock of groups and lists, never object data. The top row
holds path, adjacent page menu and at most one quiet glyph-and-word verb, content starts with
views/search/filters/display, and the page never scrolls sideways.

*Held by:* `role.frame`, `places_probe`, `overflow_probe`,
`KEYMAP` in `js/app/controls.js` (the dock shortcut), `onKeydown` in `js/app/actions.js`.

**PG2. Every place has an address.**

*Source:* ours\
*Adapted:* Exact addressed objects preserve history.\
*Why:* Addresses must never silently substitute another research object.

Each object, section and inspector mode has an exact `#page=...` address and head `Copy link`.
Object-changing choices push history; filters, search and page numbers replace it. Cold links
open their named object. Appearance and language stay local. A missing required object selects
and addresses the owner default, or its collection when none exists. No page displays an
unaddressed object. Every offered collection row opens its object, every route kind has a
page reader, tabs retain object addresses and recovery promises survive Reload. Browser
Back/Forward owns page history; path and dock provide ascent and traversal. Only detail levels
have an internal upward control (LS5).

*Held by:* the browser test in `test_workbench_readback.py` (history entries, exact
Back / Forward), `places_probe`, `tabs_probe` (a page's tabs read against the address it ends on).

*Held by:* `workbench_routes.cjs`, `workbench_dom.cjs`
(every registered inspector mode restores its exact cold address; every Goal reference opens its own reader and Back retains the Goal folder).

**PG3. The path.**

*Source:* classic: Nielsen's recognition rather than recall\
*Adapted:* One linked breadcrumb.

The top row names group / list / object / folder once, links all but the last scope and folds
from the left with object names bounded by `--crumb-object`. The last segment's information
control replaces a lede.

*Held by:* `--crumb-object`, `places_probe`.

**PG4. Canonical page names.**

*Source:* classic: Nielsen's consistency and standards\
*Adapted:* `ROUTES[page][1]` owns the label across navigation and chrome.

Dock, path, tabs and document title use one spelling. Object pages are headed by object name
rather than page kind.

*Held by:* `n3_census`.

**PG5. Titles.**

*Source:* borrowed: Linear's object pages\
*Adapted:* Collections start with controls; objects with owner titles.

Collections have no H1 or introductory sentence, while objects use their name within
`--object-title-lines` and owner description rather than kind or product narration. Sections
use muted labels or counted bands without repeating path, object or another label.

*Held by:* `object_title_census`, `n3_census`, `workbench_dom.cjs` (h1 count and
order).

**PG6. One head for every view of an object.**

*Source:* classic: Nielsen's consistency and standards\
*Adapted:* Shared object head and stable tabs.

The head holds name, one state word, short id with Copy and right-aligned actions with the
lawful `cycle()` primary. Its shared context line holds properties, controls and chooser so
tabs never move.

*Held by:* `tabs_probe`.

**PG7. Tabs.**

*Source:* borrowed: GitHub UnderlineNav, Vercel and Sentry tabs\
*Adapted:* `component.tabs` owns the remembered, single-level tab strip.

Page views and object folders use one underline-tab level, with current indicator, muted
alternatives and More for overflow instead of wrapping. The last tab is remembered, `⌘K` opens
each and tabs never nest.

*Held by:* `component.tabs`, `tabs_probe`.

**PG8. One scroller.**

*Source:* classic: progressive disclosure\
*Adapted:* One flowing vertical scroller, with an explicit console exception.

Flowing pages have one scroller, allowing only table horizontal scroll within it and paging,
folding or counting long content. At `console` height, two side-by-side standing panes may
scroll independently while the page remains still.

*Held by:* `scrollers_probe`, the
browser test (no nested scroller).

**PG9. One column.**

*Source:* classic: Gestalt proximity\
*Adapted:* Side columns serve information that changes with main content.

Only LS5 detail or a document outline in its own margin occupies a side column, never a
section list following the text scroll. Verbs go to top row or notice, decisions to a top
notice and properties to context or Properties, with nothing stacked below content.

*Held by:* `one_column_probe`.

**PG10. Split width first.**

*Source:* ours\
*Adapted:* Desktop application panes define narrow width.\
*Why:* Phone width would test a different use from the desktop split pane.

Design at the 800-1100 CSS px split width, with full width its extension. Narrow never means
phone width: walk at 900, and at 1100 for a dense page. QA viewport widths lie in 800-1673 CSS
px. Container queries use the breakpoint ladder at component width; dense lists drop a
property before truncation.

*Held by:* the gate
(breakpoints off the ladder), `overflow_probe`, `viewport_widths` (QA viewport range).

**PG11. Canonical fact location.**

*Source:* classic: don't repeat yourself\
*Adapted:* One fact, linked views.

Shared facts are stated once and linked from other views. A book's Overview owns live
standing; its Report owns the sealed deliverable.

*Held by:* review.

**PG12. Visual and outcome hierarchy.**

*Source:* ours\
*Adapted:* The dated outcome anchors Portfolio and Home.\
*Why:* Home explains outcomes and decisions without duplicating execution.

Use one dominant viewport anchor and one dominant action per region. Unchosen options are text
and resting Tasks rows; hover, selection, activity, warning or focus supplies emphasis. Dated
Portfolio order is summary, performance, holdings. Home names the current Goal, then dated
outcome, person decisions and compact progress. Research deliveries, positions and reports
appear once; intermediate Tasks belong in Tasks and History. Authorization or retained
Sessions do not establish running work.

*Held by:* review, `workbench_portfolio_layout.cjs`, `workbench_home_recent.cjs` and
`workbench_handoffs.cjs` (the person's decisions).

**PG13. Creation controls.**

*Source:* borrowed: Linear's creation menus\
*Adapted:* Creation menus follow the chain and preset their composer.

Group-head `+` and list `+ New` open the appropriate composer in creation-chain order. First
use stays on Home with `Not prepared` and `Prepare workspace`, then pinned preparation and
Home, never a Welcome page.

*Held by:* review.

## LS. Lists, threads and details

**LS1. Every object list is one lobby.**

*Source:* borrowed: Linear's grouped issue lists\
*Adapted:* `--lobby-rows` owns the six-row disclosure step.

Object lobbies group one-line rows by their own state, time or family axis, opening the
active/latest/current group and folding others behind counted groups and `Show n more`.
Search, Filter and Display persist; no row is marked with nothing open.

*Held by:* `--lobby-rows`,
`lobby_probe`, `pages_walk` R3.

**LS2. A row is one line.**

*Source:* borrowed: Linear's dense issue rows\
*Adapted:* Shared row and column owners define density and wrapping.

One-line rows keep ellipsized words whole on hover, sentences in the opened object and growth
limited to the opened row. Named fact columns share a subgrid sized by values, then align
under the title when narrower, with absent facts empty.

*Held by:* `--row-dense`,
`component.columns`, `pages_walk` (R2, a drifting fact column), `workbench_dom.cjs`
(removed, reordered and translated named fact slots fail the column check).

**LS3. The rail.**

*Source:* classic: Nielsen's visibility of system status\
*Adapted:* `component.rail` marks the inspected row alone.

The link-blue rail uses declared width and row-padding inset, drawn as a device-snapping
border. Kind remains a word and objection state a dot, never a second rail meaning.

*Held by:* `component.rail`.

**LS4. Runs.**

*Source:* ours\
*Adapted:* `runRow` on `runsOf` is the complete execution view.\
*Why:* Complete execution detail needs one consistent place.

Runs group by day newest first, with the moving run pinned and step, state and starter in
fixed slots. Object runs occupy the status page's last section. One Task host/body contains
summary, cause and next action, folded steps with current/stopped open, receipts with Copy per
hash and line-by-line logs folded to a grey count.

*Held by:* `runRow`, review.

**LS5. Master-detail reading.**

*Source:* classic: Nielsen's recognition rather than recall\
*Adapted:* `detailSplit` and `component.detail` retain the list and a single level-up trail.

One glass detail opens whole beside its own list at `--detail-width`, overlays under 900 px,
closes on repeat press/X/Esc and hosts Facts or Record beneath the head, while saved objects
never open in dialogs. Nested objects add one level with the inset `←` tab visible on
hover/focus or without hover, naming its previous level in the tip and returning one level per
press without covering words or moving head/body edges, with no retained trail after page-open
or close (`Window.trailAfter`).

*Held by:* `component.detail`, `detail_census`, `detail_switch_probe`,
`detail_flicker_probe`, `chevron_census`.

**LS6. Citation navigation.**

*Source:* ours\
*Adapted:* Exact cited spans.\
*Why:* The reader must judge a claim's evidence without inferring an uncited relationship.

Cited passages open beside the claim with its cited span marked. Each passage names and opens
its citing findings.

*Held by:* review.

**LS7. Session threads.**

*Source:* borrowed: GitHub issue conversations\
*Adapted:* Two flat thread levels preserve recorded replies.

The question titles a session of time-ordered comments and one-line observations, with member
filtering and facts in the head. Comments and replies share one light axis and indent without
elbows, retaining `→ @name`.

*Held by:* `--thread-axis`,
`workbench_team.cjs`, `thread_fold_probe`.

**LS8. Recorded exchanges.**

*Source:* ours\
*Adapted:* Threads disclose records, not invented engagement or resolution.\
*Why:* Conversation observation cannot prove liveness, quality or resolution.

Only owner records count in recorded time order, with no likes, views, ranks or runtime
states. `n new` reaches the newest; Display filters/density never reorder. Consecutive hooks
and observations fold with their real span; long replies keep first/last around `Show n more`.
Exchanges open their record in place. Recorded PM replies sit beneath their question in a
neutral `PM` inset, never resolving an objection by display.

*Held by:* `thread_fold_probe`, `workbench_team.cjs`.

**LS9. The ways into Team by role.**

*Source:* ours\
*Adapted:* Recorded references and native parentage determine membership independently of usage and identity.\
*Why:* Identity, usage and contribution require distinct evidence.

One Sessions dock row serves role entry. Lobby Roles names specialists and filters by them;
the common Main PM is not a filter and two card names for one role stay one role. Facts links
only sessions whose declared references name the exact study, book, Evidence review or
analysis, with specialists; missing relationships are stated, never guessed. Native parentage
determines Sessions and Participants membership, including usage-only children. Usage belongs
to the thread, specialist identity is separately verified, absent roles never hide usage and
failed parent reads never hide confirmed children. Goal participation counts only attributed
hooks/messages, never session-wide usage.

*Held by:* `collaborationRows`, `LiveTeam.sessionsNaming`, the books scene's Team sessions
(`team_evidence.py`), the walk.

*Held by:* `workbench_team.cjs`, `test_a_codex_lead_names_its_children_in_its_own_rollout`,
`test_the_host_reads_the_lead_and_the_children_its_own_session_records`.

## LY. Layers and glass

**LY1. Shared floating material.**

*Source:* borrowed: Apple HIG materials\
*Adapted:* One material.glass; outlined boxes, tinted fields/chips, glass wells and flat warnings; primary actions, marks and keys stay solid.

Every floating surface and its contents use one glass material. Glass is default; saved Opaque
controls disables it.

*Held by:* `material.glass`, `glass_census`.

**LY2. Material and positioning.**

*Source:* borrowed: CSS Filter Effects\
*Adapted:* Glass hosts preserve fixed descendants' viewport positioning and frost throughout entrance.

A detail, dialog or drawer never changes the positioning frame of its fixed menus or pickers.
Glass remains visible through its entrance.

*Held by:* `glass_census`, `glass_fade_probe`.

**LY3. Material inside floating surfaces.**

*Source:* borrowed: Apple HIG materials\
*Adapted:* Retained alias; glass-parts belongs to LY1.

See LY1.

*Held by:* mode `glass-parts`, `glass_census`.

**LY4. Content visibility.**

*Source:* classic: Nielsen user control and freedom\
*Adapted:* Decision modals use half opaque dim and 10 px --backdrop-blur; read-only sheets have no scrim and close on outside press; details frost their background, and ST5 dims slow rereads.

Resting content is never dimmed, masked or covered; scroller edges never fade. Hover actions
take their own slot.

*Held by:* `role.depth`, `reading_switches_probe`.

**LY5. Dialogs.**

*Source:* borrowed: Apple HIG dialogs\
*Adapted:* Head X exits; feet contain verbs, primary confirmation and no second Close/Cancel/Back; two/three-answer alerts use bold title, one body line and equal answer-sized pills; Quick Open has no head/X and exits with Esc.

Confirmation facts name reads, source, conditions and exclusions, followed by one sentence
stating what it admits. Keep its exact request folded inside.

*Held by:* review.

**LY6. Shared tooltips.**

*Source:* borrowed: WCAG 2.2 Content on Hover or Focus\
*Adapted:* Existing dismissible, focus-triggered tips; Workbench timing, placement and dock geometry.

The shared glass tooltip appears after 400 ms, or immediately within 300 ms of another tooltip
or on focus, closes on leave, blur, Escape, press or scroll, is at most 320 px wide and uses
layout coordinates. It does not repeat visible words: dock rows reveal their shortcut in
place, icon tips sit level beside the dock, and the dock opens at 216 px and resizes to 180
px.

*Held by:* `workbench_dom.cjs` (a tip on focus,
gone on Escape), `dock_tips_probe`.

**LY7. Action and Task notifications.**

*Source:* classic: Nielsen visibility of system status\
*Adapted:* Person-action confirmations last 3.5–8 s inside the window, clear of a right dock; Task endings announce name/state once with an opening action.

A followed Task announces its own ending. Agent events are unseen counts, never toasts.

*Held by:* `workbench_activity.cjs` (a Task's ending), review.

**LY8. Opaque material.**

*Source:* classic: graceful degradation\
*Adapted:* Glass-dependent values record opaque counterparts, with the gate's declared shape and control-tint exceptions; Opaque controls keeps solid dim without fog, Reduce focus effects removes both.

Keep both configurations together and apply the declared reading changes.

*Held by:* the gate,
`test_workbench_parameters.py::test_the_gate_refuses_a_glass_value_without_its_opaque_one`,
`reading_switches_probe`.

## WD. Words

**WD1. Canonical object names.**

*Source:* classic: Nielsen's match with the real world\
*Adapted:* One owner name, or the declared naming rule.

Use one owner name everywhere, derive absent names only by the per-kind rule (sessions use
the scoped Goal's question or the latest delivery's Goal question) and distinguish look-alikes
by owner facts. One word names one thing; actors retain their actual identity or Unidentified.

*Held by:* `object_title_census`,
`test_workbench_readback.py::test_submission_words_name_submissions_in_page_labels_and_chinese`
(a revision's recorder is not a completion submitter).

**WD2. Declared vocabulary.**

*Source:* classic: Nielsen's match with the real world\
*Adapted:* Declared words retain exact owner codes; commands remain literal.

Codes, methods, attributions and count nouns use `codeWords`, `methodWords` and the Glossary.
A count carries its noun by number, never `(s)`. Internal nouns are translated once: a packet
is the prepared evidence, a unit a group, a cell issuer × topic. The Glossary holds every noun
a page keeps. Method handles use mono and owner codes appear on hover; unknown composed codes retain `Word not
declared` and their exact value rather than a guess. Person-entered literals such as
`ALPHALATTICE_NETWORK_DISABLED=1` remain mono `code.literal`, breaking at split width walked
at 900 and at 1100 for a dense page.

*Held by:* `codeWords`, `methodWords`, `s_census` (raw codes), `pages_walk` (raw codes, a
literal excepted),
`test_workbench_owner_words.py::test_every_composed_closed_owner_value_has_declared_bilingual_words`
(declared groups, unknown codes and the retired fallback).

**WD3. References and identifiers.**

*Source:* borrowed: Git's short object references\
*Adapted:* Compact references retain full hover and Copy.

Rows lead with names and at most one short mono reference, hashes otherwise remain in
Facts/Record/Copy, locators show their ends and every truncation retains full-value hover and
Copy without mid-token wrapping or per-value scroll. Phrases wrap, and named
policies/recipes/rules read as code words with hover code and Copy rather than identities,
with only FT5 boxes allowed to roll.

*Held by:* `SHORT` in `js/app/components.js` (reference and content-hash lengths),
`refCell`, `hashCell`, `codeCell`, `s_census`,
`clip_census`, `workbench_minor_ui.cjs` (plain and slotted hint phrases; full-value tooltips remain reachable), `workbench_routes.cjs`
(composed refusal locators retain their exact copy).

**WD4. Text disclosure.**

*Source:* classic: progressive disclosure\
*Adapted:* Owned word budgets, separate from authored text.

First-screen product wording stays within 130 words for overview and 180 for views, excluding
figures and closed folds, with at most twelve evidence terms before disclosure and
`.owner-text` shown for three lines with More. Grey lines over 90 characters retain their
first clause and disclose the rest.

*Held by:* `pages_walk` R4, `evidence_census`,
`Controls.foldLines`.

**WD5. Time in the reader's zone.**

*Source:* classic: Nielsen's match with the real world\
*Adapted:* Reader-zone instants and market-session dates stay distinct.

Instants use reader-zone time within that day and date/time otherwise, labelled UTC only in
Facts. A market session is a date; no `ago` except owner age words. Replay text uses explicit dates,
never present-time claims.

*Held by:* `n3_census`, `test_local_web_product.py`
(`test_development_replay_copy_makes_no_forward_claim`).

**WD6. Evidence vocabulary.**

*Source:* ours\
*Adapted:* `EVIDENCE` owns names, readings and states.\
*Why:* Evidence terms must keep their recorded meaning.

`evidenceRow` supplies every evidence row and `citePill` the shared handle and hover card.

*Held by:* `EVIDENCE`, `evidence_census`.

**WD7. Product localization.**

*Source:* classic: internationalization and localization\
*Adapted:* Keyed product language preserves authored text and literal commands.

All product words use `t()` and Chinese keys, unused keys are removed, and owner-composed
sentences use keyed fields or exact catalog templates while authored source and participants
retain spelling. Catalog refusals cover unlisted codes too, with owner explanation before
fallback, literal commands/metavariables and neutral scope wording.

*Held by:* `zh_keys` (the phrases asked for at run time) with the private `zh_dead` check,
review, `test_workbench_readback.py::test_every_chinese_translation_uses_its_owned_ui_vocabulary`
(ordinary English nouns fail; reasoned literals stay exact),
`test_workbench_readback.py::test_every_generated_evidence_sentence_reads_in_chinese`,
`test_workbench_readback.py::test_evidence_sentence_census_discovers_new_outputs_and_raw_readers`,
`test_workbench_readback.py::test_every_door_refusal_reads_in_chinese`
(the real refusal renderer, including codes outside the page's list).

## AC. Motion and access

**AC1. Motion.**

*Source:* borrowed: Linear and WCAG reduced-motion guidance\
*Adapted:* Shared Linear motion scale and reduced motion; paint under 25 ms and frames under 20 ms.

Page changes cut to readable first frames; departures reverse entrances over --dur-1. Reduced
motion makes changes instant.

*Held by:* `scale.motion`, `11-accessibility.css` (reduced motion); the
paint budget by review (visual measurement).

**AC2. Initial rendering.**

*Source:* classic: deterministic initialization and cache coherence\
*Adapted:* Synchronous preferences, locale-loaded dictionary, immutable hashed assets and uncached page/API reads.

First paint uses the saved configuration. Failed renders show the refusal card.

*Held by:* `js/app/prelude.js`, `test_local_web_product.py`, the browser test.

**AC3. Text size and viewport.**

*Source:* classic: browser zoom\
*Adapted:* Product zoom 80–175% uses Ctrl =, -, 0; --vh/--vw divide viewport lengths by zoom. The Workbench makes no WCAG claim.

Use product zoom for text and compensated variables for viewport lengths.

*Held by:* `role.frame`.

**AC4. Reading preferences.**

*Source:* classic: Nielsen user control and freedom\
*Adapted:* Opaque controls restores at boot; Reduce focus effects, Wider scrollbars and reduced motion retain their meanings; contrast deepens inks and edges.

Every reading switch does exactly what its line says.

*Held by:* `reading_switches_probe`, the
`contrast` modes.

**AC5. The keyboard.**

*Source:* borrowed: GitHub keyboard navigation and command palettes\
*Adapted:* KEYMAP owns command-palette and list navigation; other owners' bindings appear in its help.

Use G plus the dock letter, J/K to move through lists, Space for the hover card, Enter to
open, Escape to exit, brackets within the source or adjacent list and ⌘K for Quick Open;
reaching another object forgets the source list. Remove bindings with their targets and retain
the 2 px accent focus ring, offset 2 px outside or inside a clipping strip, only on
focus-visible.

*Held by:* `role.focus`, the browser test, `keys_probe`.

## PR. How the rules are kept

**PR1. Design parameter ownership.**

*Source:* classic: single source of truth and don't repeat yourself\
*Adapted:* One owner per design value and lawful source.

Chosen values/reasons/modes/sources belong to `design/parameters.json`, derived values to
formulas and other literals to structural rules or declared exceptions, with shared builders
owning alignment rather than page offsets. The build/gate enforces these owners and shrinking
literal ratchet, and borrowing uses recorded multi-product measurement or applicable open
licence, never excluded kits, borrowed assets or identities.

*Held by:* the gate,
`test_workbench_parameters.py`.

**PR2. Placement and component structure.**

*Source:* classic: separation of concerns\
*Adapted:* Placement and shared component shape have separate owners.

Chapter sheets place components without sizing, colouring or spacing them. Shapes belong to
`components.js` with Workshop specimens and selectors to one sheet.

*Held by:* the gate's literal ratchet, review.

**PR3. Rule enforcement.**

*Source:* classic: design by contract\
*Adapted:* A holder at the rule's owner.

Parameters, CSS, builders, tests or QA checks enforce rules; prose only records them. A page
defect is fixed at its cause and checked across its scene/page/state class.

*Held by:* this book's "Held by" lines.

**PR4. Source and licence review.**

*Source:* classic: don't repeat yourself\
*Adapted:* Alias for PR1's recorded sources and lawful borrowing.

See PR1.

*Held by:* review.

**PR5. Visual validation.**

*Source:* ours\
*Adapted:* Stills use the desktop matrix (1470 / 1180 / 900 wide, Light / Dark, en / zh, 100 / 125 %) and the person's window (1673 x 1186 at 125 %), with classic Windows scrollbars and real Chrome for glass.\
*Why:* Split panes and real scrollbars reveal layout defects.

See PG10; walk at 900, and at 1100 for a dense page. Headless device scale 2 frost and
scrollbar differences do not replace real-Chrome proof.

*Held by:* the kit's
stills and `viewport_widths`.

**PR6. Rule traceability.**

*Source:* ours\
*Adapted:* A stated rule records English, source, reason, holder and date.\
*Why:* Laws retain provenance without workshop history.

See PR3.

*Held by:* review.

**PR7. Integration conformance.**

*Source:* ours\
*Adapted:* Integrated items are reviewed individually against shared tokens/builders and walked on affected scenes.\
*Why:* Integration preserves one visual vocabulary.

See PR1 and PR2.

*Held by:* review, the kit's walk on every scene the merge touches.

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
