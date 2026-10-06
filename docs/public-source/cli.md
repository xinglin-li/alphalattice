# Use the CLI
Date: 2026-10-03

Define the checkout launcher and bind its workspace as described in [Getting started](getting-started.md). The grammar is `alphalattice <noun> <verb>`; global options such as `--workspace`, `--view`, `--goal` and `--lang` can appear before the noun or after the action. `serve` starts the Host and `request` submits a complete request. Most operations need that workspace's Host. Once bound, use the short form:

```powershell
alphalattice workspace show
alphalattice study controls --input '<input-id>' --output controls.json --save-declaration study.yaml
alphalattice study plan --from controls.json --file study.yaml --output plan.json
```

Start with `workspace show` and its `intents`: each input's flows name `needs`, `present` results and `missing` prerequisites. Follow `next_commands`; fill the `choose` fields in `next_templates`. Use help or a schema only for fields the returned choices do not resolve. `operation list` names installed operations and person-only actions. Help grants no permission. An open, unexpired `FIRST_USE` goal delegates only the three first-preparation steps; activation and other person-only decisions remain with the person. `--lang zh` changes available detail wording, with English fallback; codes, field names and authority are unchanged.

Declarations are requests to installed owners, not executable Python or a way around refusal. `@path` reads a field from a file; text fields read text and structured fields read YAML or JSON. `@@` starts a literal `@`. Boolean flags accept `true` or `false`; repeated YAML keys and invalid Boolean values are refused. Inspect the returned plan before requesting its exact run. Use unused output paths: an existing path is refused. HTML output is a report, not a continuation file.

For `feature_research.input_binding_unresolved`, `fields` names the binding field and `expected.input_binding_hash` lists held workspace bindings; follow the offered controls request for the input you mean. When `EXPERIMENT_PLAN` separates Desk declarations, `declaration_sections` names them.

## Answer and refusal

The stable envelope has `operation`, `outcome`, `status`, `data`, optional `failure_code`, `detail`, `next_requests`, `next_commands`, `next_templates`, `timing.elapsed_seconds`, `session_launches` and `context`. `data` is the owner answer or the selected display projection. Read a refusal's exact code, explanation and `next_action` or offered requests; keep the returned references and do not edit the store or replace refused hashes. `next_templates.choose` lists the required choices still missing.

| Outcome | Exit | Meaning |
| --- | ---: | --- |
| `OK` | 0 | Read the returned status and result |
| `INVALID_INPUT` | 1 | Correct the named request or declaration |
| `REFUSED` | 2 | Read the allowed continuation or person decision |
| `PENDING` | 3 | Follow admitted work or its decision |
| `NO_HOST` | 4 | Start or reconnect to the workspace Host |

`--wait` follows the named Task state even when it appears as `lifecycle` or `task_lifecycle` beside another status. Pending work exits `PENDING`; blocked/cancelled work exits `REFUSED`; a completed non-refused answer exits `OK`. Waiting ends when work ends, needs a decision, is deferred, reports an incident or reaches `--max-wait`. The cap leaves work running. Follow a deferred Task's offered `resume` after its named `retry_after_at`. With `--wait --output`, admission is saved before following and the file is replaced by the final answer; save failure does not undo admission. After disconnect or timeout, reopen the Task before deciding whether to submit another request.

## Saved answers and display

`--output` saves the full owner answer as JSON or YAML in either view. In a saved answer the owner's fields are at the root, not under the printed envelope's `data` (for example, read `position`, not `data.position`). `--save-declaration` saves an editable declaration. `--view full` displays all content; compact view names omitted paths in `omitted_sections`, which can be read with `--section <dotted-path>`. A selected section affects display only; `--output` still saves the full answer. A cut compact answer cannot stand in for its full saved request.

For a repeated read, `--from answer.json` carries the whole saved selection, including input revision, session or page. An explicit non-target selection flag wins; a flag naming a different Task or goal is refused. For a next operation, `--from` selects the offered request, a draft's plan or a Task reference. Explicit values fill open choices; fields bound by the offer cannot be replaced. `--list-next` shows multiple actions for explicit selection; `--choices choices.yaml` supplies the selected action's missing values. A partly filled object remains a template until its inner required `choose` fields are set. A study plan can read controls and keep their input; a study draft can take an explicit declaration.

Compact view shortens issued references and answer identifiers to twelve characters, including list items; authored request identifiers stay whole. The CLI resolves a shortened issued reference against this workspace's returned references. Ambiguity is `local_client.short_reference_ambiguous`; an unknown short reference is `local_client.short_reference_unknown`. The refusal names the field, value and candidates. `--view full`, `--output` or `--from` keeps/carries whole references. Paths, links, commands and other text are never shortened.

Complete PowerShell commands returned by the product preserve quoting, caller view, language and goal; run them as printed, including the call operator. A complete request with double quotes is sent as JSON through stdin to `request --file -`. Templates still need their named choices.

A missing or unreadable document is refused with `document_location`; request documents are limited to 128 KiB, while `--from` reads saved answers up to 4 MiB. `local_client.document_too_large` reports `document_size.bytes` and the applicable `document_size.limit_bytes` (stdin byte size is null). Correct the named document before resubmitting.

Portfolio position metrics carry units only when `metric_units` names that exact path; units come from `POSITION_UNITS`. Read rates, fractions, sessions and dollar quantities with their unit. Unknown sections list available paths. For experiment claim marks and installed strategy continuations, see [Research flows](research-flows.md). The optional launch count is described in [Privacy](privacy.md).
