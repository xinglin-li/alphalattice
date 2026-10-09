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
A refusal names the code, explanation and permitted continuation. Keep its
references and follow that continuation.

| Outcome | Exit | Meaning |
| --- | ---: | --- |
| `OK` | 0 | Read the returned status and result. |
| `INVALID_INPUT` | 1 | Correct the named request or declaration. |
| `REFUSED` | 2 | Read the allowed continuation or person decision. |
| `PENDING` | 3 | Follow admitted work or its decision. |
| `NO_HOST` | 4 | Follow the named connection route; start a Host only if absent. |

Declarations request installed owners. `@path` reads text, YAML or JSON according
to the field's type; `@@` starts a literal `@`. Boolean values are `true` or
`false`; repeated YAML keys and invalid Boolean values are refused. Existing
output paths are refused. HTML is a report, not a continuation file.

## Saved answers and live continuations

`--output` saves the full owner answer as JSON or YAML in either view. Its fields
are at the file's root: read `position`, not `data.position`.
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

## Waits, document limits and units

`--wait` follows the named Task until completion, decision, deferral, incident or
`--max-wait`, which leaves work running. Pending work exits `PENDING`;
blocked or cancelled work exits `REFUSED`; completed non-refused work exits `OK`.
A deferred Task offers its retry time and resume. After disconnect or timeout,
read the existing Task before resubmitting.

On Codex, `strategy-book review` and `review continue` accept
`--notify codex-queue` to return at the first running Task while the Host keeps
its notification; a notice names the continuation when the Task ends, needs a
decision or is deferred, and delivery failures appear in Task activity.

With `--wait --output`, admission is saved first and replaced by the final
answer; save failure does not undo admission. Request documents are limited to
128 KiB; `--from` reads up to 4 MiB. Document refusals name `document_location`
and applicable size limits.

Read Portfolio metrics with the unit declared for their `metric_units`
path. Unknown sections list available paths. [Research flows](research-flows.md)
explains result claims, and [Privacy](privacy.md) explains observed usage.
