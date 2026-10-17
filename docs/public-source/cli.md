# Use the CLI
Date: 2026-10-08

Use `alphalattice <noun> <verb>`. Global `--workspace`, `--view`,
`--goal` and `--lang` options work before the noun or after the action. Most
operations need the workspace's Host; an agent session binds itself by working
on that workspace. [Getting started](getting-started.md) covers installation;
the [agent guide](../../AGENTS.md) owns procedures and person decisions.

```powershell
alphalattice workspace show
```

`workspace show` offers `intents`. Each answer's top-level `next_action` and
`next_requests` decide the continuation; a nested hint describes only its part.
Returned commands are complete requests; templates still need their `choose`
fields. Help and `operation list` describe installed operations and grant no
permission. `--lang zh` changes detail wording with English fallback; codes,
fields and authority stay unchanged.

## Answers and refusals

The envelope carries `operation`, `outcome`, `status`, `data`, `failure_code`,
`detail`, `next_requests`, `next_commands`, `next_templates`, timing and context.
When observed, `session_launches` counts this session's CLI launches, including
help and schema reads. `data` contains the owner answer or its display projection.
An admitted network step also carries `network_use`: `under` names whose setting allowed it and `words` is its disclosure. The same permission source is kept in activity, including automated updates.
A refusal names the code, explanation and permitted continuation. Keep its
references and follow that continuation.

| Outcome | Exit | Meaning |
| --- | ---: | --- |
| `OK` | 0 | Read the returned status and result. |
| `INVALID_INPUT` | 1 | Correct the named request or declaration. |
| `REFUSED` | 2 | Read the allowed continuation or person decision. |
| `PENDING` | 3 | Follow admitted work or its decision. |
| `NO_HOST` | 4 | Follow the named connection route; start a Host only if absent. |

An accepted `bundle submit` exits `OK` (0). Its `task_lifecycle` and offered Task
read report the review's continuing work; `--wait` can follow that Task.

Declarations request installed owners. `@path` reads text, YAML or JSON according
to the field's type; `@@` starts a literal `@`. Boolean values are `true` or
`false`; repeated YAML keys and invalid Boolean values are refused. Existing
output paths are protected, except a saved JSON/YAML refusal without an admitted
Task: a retry replaces it and prints `output_note`. HTML is a report, not a
continuation file.

## Saved answers and live continuations

`--output` saves the full owner answer as JSON or YAML in either view. Its fields
are at the file's root: read `position`, not `data.position`.
Compact output leads with `status`, `next_action` and `task_id`; the original
fields remain in `data` and in the saved answer.
`--save-declaration` writes an editable declaration. Compact view names omitted
paths in `omitted_sections`; `--section <dotted-path>` selects one for display,
while `--output` still saves the full answer. `--view full` displays everything.

Read a saved full answer locally, without a workspace, session or Host:

```powershell
alphalattice answer show --file answer.json --list-sections
alphalattice answer show --file answer.json --section position
alphalattice answer show --file answer.yaml --section result.items.3:6
```

The snapshot is marked `HISTORICAL_SAVED_ANSWER_NOT_REVERIFIED` with its source
file. It performs no fresh verification and never executes saved requests.
Without a section option it returns the original under `answer`; either view
prints the selected reading whole with references and declared units. A valid
local read exits `OK`, including when the saved answer records refused or
pending work.

Its `--output` saves the full snapshot reading; `--format yaml` selects YAML.
Input is the exporter's JSON-compatible tree, up to 4 MiB. Printed envelopes,
compact displays, HTML, YAML aliases, cycles and non-finite numbers are invalid.
Unknown section paths return `INVALID_INPUT` and available sections.

For live continuation, `--from answer.json` repeats a saved read with its whole
selection, including a day or page. An explicit non-target selection flag wins;
a different Task or goal is refused. For a next operation, it takes the offered
request, draft's plan or named Task. Explicit values fill open choices while
bound fields stay fixed. `--list-next` lists actions; `--choices choices.yaml`
supplies missing values. Partly filled objects remain templates.

Compact view shortens issued references to twelve characters, resolved by the
workspace; ambiguity or an unknown reference is refused with candidates.
Authored identifiers, paths and commands stay whole. `--view full`, `--output`
and `--from` retain whole references. Run returned PowerShell commands as printed,
including quoting and the call operator; requests with double quotes use JSON
through stdin to `request --file -`.

## Dates and network permission

```powershell
alphalattice research-update plan --package <package> --date <date> --output update-plan.json
```

`--date` names the entry session for the positions, not the formation session or the last downloaded date. A non-refused plan answer carries `positions_for.trade_date`, `positions_for.as_of_session`, `positions_for.as_of_close_at` and its `reading`: positions entered at that session's open are decided at the preceding session's close. Without `--date`, the plan uses the next session after the latest completed formation close. Invalid or unavailable dates return their own refusal and offered way on; `--date` and `--through` cannot be combined.

`alphalattice network show` reads the effective control. `alphalattice network set --enabled true` and `alphalattice network set --enabled false` change it under the person's decision or a valid first-use delegation. A person's setting wins over a delegation; the operator's offline switch and a research run's offline hold win over either.

For the one daily-network question, the agent relays the person's yes through `network set --enabled true` and `evidence-consent set --per-issuer 3`, using `--person-said` and `--asked` for their words and the question. This authorizes the stated daily updates and default SEC budget until the person says stop. `alphalattice evidence-consent show` reads the source budget; `alphalattice evidence-consent set --per-issuer 10` changes it with the required authority. Omitted `--documents` and `--bytes` retain the current total bounds; a wider budget needs the person's decision. See [network consent and revocation](privacy.md#workspace-network-and-consent).

## The daily committee

The first use's report asks the person once whether each daily update also convenes the investment committee. The agent relays their answer with `alphalattice committee daily --enabled true` or `alphalattice committee daily --enabled false`, using `--person-said` and `--asked`. Unanswered, the setting is off: a daily update's published review then answers `GIVE_THE_PM_VIEW`, so the agent gives the portfolio manager's view of the reviewed positions, and the committee for that date stays offered. `alphalattice workspace show` reads the setting as `committee_daily`.

## Waits, document limits and units

`--wait` follows the named Task until completion, decision, deferral, incident or
`--max-wait`, which leaves work running. Pending work exits `PENDING`;
blocked or cancelled work exits `REFUSED`; completed non-refused work exits `OK`.
A deferred Task offers its retry time and resume. After disconnect or timeout,
read the existing Task before resubmitting.

The agent verbs `first-use prepare`, `strategy build`, `strategy-book review` and
`review continue` follow each Task they start to its end. On Claude Code, run a
wait or an agent verb in the background (the Bash tool's `run_in_background`)
and act on its completion notice. On Codex, they accept `--notify codex-queue` to
return at the first running Task while the Host keeps its notification; a notice
names the continuation when the Task ends, needs a decision or is deferred, and
failures to queue it appear in Task activity. Under an active Codex goal, wait inside
the turn instead: the goal's next turn starts before a queued notice is taken. A
registration answering `local_client.codex_unreachable` means the Host cannot reach
Codex (it runs inside the agent's sandbox), so no notice will come: wait inside the turn.

With `--wait --output`, admission is saved first and replaced by the final
answer; save failure does not undo admission. Request documents are limited to
128 KiB; `--from` reads up to 4 MiB. Document refusals name `document_location`
and applicable size limits.

Read Portfolio metrics with the unit declared for their `metric_units`
path. Unknown sections list available paths. [Research flows](research-flows.md)
explains result claims, and [Privacy](privacy.md) explains observed usage.
