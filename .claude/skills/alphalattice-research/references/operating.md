# Operating the CLI
Date: 2026-10-06

## Setup and launch
Date: 2026-10-03

Windows is the certified platform for this source-checkout setup; macOS and Linux are unverified. Use Python 3.12, uv and a browser from the checkout root. Dependency setup needs permission to download locked packages or a local cache. The locked environment and browser assets do not prepare research data:

```powershell
uv sync --locked --all-extras
.venv/Scripts/python.exe scripts/build_local_web_ui.py --product
uv export --locked --all-extras --no-dev --no-emit-project --no-hashes --output-file .venv/alphalattice-runtime-requirements.txt
uv tool install --python 3.12 --editable . --with-requirements .venv/alphalattice-runtime-requirements.txt --constraints .venv/alphalattice-runtime-requirements.txt
```

`uv sync` installs the checkout's console command editable; source edits are live. `uv run alphalattice ...`, `.venv/Scripts/alphalattice.exe ...` and `python scripts/run_alphalattice.py ...` also run the checkout. The tool install is editable and pins its separate runtime to the exported lock; refresh it when the lock or project metadata changes. `uv tool dir --bin` must be on the person's persistent PATH before starting Codex or Claude Code. If needed, the person can run `uv tool update-shell` and restart both hosts. This is a person-owned shell setting; product setup does not change host settings. Each host command starts a fresh shell, so activation, PowerShell functions and variables do not carry over. One tool name selects one installation leg at a time; run `uv tool uninstall alphalattice` before switching legs.

For a release wheel, use a separate directory; no checkout is needed. Extract its embedded runtime lock, then install the wheel with that file as both requirements and constraints:

```powershell
python -c "from zipfile import ZipFile; from pathlib import Path; Path('alphalattice-runtime-requirements.txt').write_bytes(ZipFile('alphalattice-0.1.1-py3-none-any.whl').read('alphalattice/_runtime/config/release/runtime-requirements.txt'))"
uv tool install --python 3.12 ./alphalattice-0.1.1-py3-none-any.whl --with-requirements alphalattice-runtime-requirements.txt --constraints alphalattice-runtime-requirements.txt
```

Use `--offline` for uv commands only when locked packages are cached. The wheel includes resources, built Local Web, Skill and role cards. Start it with the explicit workspace launch command in the next section. Configure its shipped guidance with the Python interpreter in the installed `alphalattice` tool environment beneath the directory reported by `uv tool dir` (not the checkout's `.venv`):

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> configure --host <codex-or-claude-code>
```

Existing different files are refused. Configuration does not approve host trust; the person inspects and trusts declarations. Author model source with `model scaffold --file` only in an editable checkout; checking and running installed models works on either leg. Workspace data, retrieval models and issuer sources still need their distinct setup and authority. Evidence review needs its declared retrieval environment, local model packs and admitted issuer-source package. If the launcher reports setup held, read its returned cause and setup request instead of treating an empty page as a completed installation. If the declared retrieval environment is absent, `serve` returns its setup command; that command alone supplies neither local retrieval models nor issuer sources.

## Launch and bind a workspace

Choose a new workspace directory for new research; the launcher initializes it without preparing data or installing a default strategy. Start its Host with an explicit workspace path:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --stop-on-stdin
```

As the lead, use the browser tools your host actually exposes. Unless the person asks for terminal-only work, open the exact launch link printed by the service in the host's own in-app browser; it establishes the browser session. Leave that tab available so the person can follow the work. Follow the current Task, its pending person action or its published result; change pages only when that state or action changes. If browser tools are unavailable, give the person the exact printed launch link. A printed link alone does not establish that a browser integration is attached.

Keep the service attached to stdin; enter `stop` to close the launch. Opening Local Web does not acquire data or run a study. Reuse the existing Host for commands on the workspace rather than starting another writer. If a restart invalidates the browser session, open the new launch link. Follow [the agent guide](../../../../AGENTS.md) to bind the native session and use clean `alphalattice` commands afterward; configuration does not change the person's host trust.

## Command contract

The CLI grammar is `alphalattice <object> <action>`, followed by any id and flags; `request` sends a whole operation document. Commands do not prompt and have no aliases. Bind a session or name the workspace explicitly. Use exact object/action names and one operation per command.

- The answer supplies filled `next_commands` and choice-requiring `next_templates`. Run the former as returned; for the latter use only the named choices. To take a named edge, use `request --from "<out>/answer.json" --action <returned-action-name>`. Same-named requests from separate parts carry that part's Task, case (such as `recovery:<task_id>`) or place; use the returned qualified name. `--from` continues from an answer; `--file` supplies that operation's document. `--choices` fills its open fields. A request file is not an answer. An offered edge is navigation, not proof its prerequisites hold; the Host revalidates it. Consult command help or `schema show` only when the answer does not provide a needed field or form.
- Take ids, hashes and choices exactly from an answer or assignment. Never invent, substitute or silently change a bound reference. A compact id is usable only when unique; if a prefix is ambiguous, use the whole id listed by the refusal. The Host revalidates every continuation; choices fill open fields without replacing bound references.
- Read returned `prerequisites` before continuing: an offered edge is navigation, not proof its requirements hold. An omitted field is not evidence that the field or requirement is absent. The Host revalidates every continuation.
- Write outputs to new, unused paths under the workspace, never into the checkout or over existing research. Make `<out>` a new workspace directory. `--output <path>` saves the full answer (the answer's `data` at the file root); `--format yaml` saves YAML. `--save-declaration <path>` writes an editable declaration. `--file <path>` reads the command's document; `-` reads UTF-8 YAML or JSON from stdin. The CLI will not overwrite a file, except for the admission answer saved by its own `--wait` command.
- A person's request authorizes only its stated scope. Updating source data, sealing a new input and starting a different experiment are separate decisions; ask before crossing into another one. A fresh checkout uses its own locked environment and workspace; never borrow another checkout's environment or data.
- Global `--workspace`, `--view`, `--goal` and `--lang` options may appear anywhere. `--` ends options. `@@` escapes text that begins with `@`. Run offered commands only within the authorized scope and launch budget. Every launch, including help, schema and failure, counts; `session_launches` reports it where available but grants no budget.

<!-- Generated from the CLI's registered outcomes and _PROCEED meanings. -->
| Exit | Outcome | Do |
| --- | --- | --- |
| 0 | OK | read `status` and `data`, a result's `standing` first; an OK read is not approval or a finished Task |
| 1 | INVALID_INPUT | correct the named command, document or field within scope |
| 2 | REFUSED | request refused or work blocked, cancelled or stopped: read `failure_code`, `fields`, `detail` and `next_requests`; take only an authorized continuation, else report the stop |
| 3 | PENDING | follow the admitted Task or answer its decision; never resubmit queued or running work; use the answer's offered deferral resume after its retry time |
| 4 | NO_HOST | report it: the lead starts or reconnects the Host |

A refusal leaves its owner's stable code, detail and legal way on. Follow only the offered `next_action`, `next_requests` or a person-authorized act; never invent a workaround or retry completed/in-flight work. If a compact reference is refused with `compact_reference_requires_full_response:<part>`, save the full answer or read that part with `--section <part>`.

A timeout or lost connection is a transport event: check the known Task before retrying; if no Task is known, ask the lead. `activity wait` waits across Host restarts and ends on one `wait_event` naming what happened and how to read it; `task show` reads state. `--wait` follows admitted work until it ends, needs a decision, is deferred, reports an incident or reaches `--max-wait`. A cancellation request is not a cancellation result. If a local output write fails after the owner answered, read stdout or the Task; never resubmit the work. `local_web_url` is navigation, not evidence. Do not poll or use a detached waiter your host cannot track. A lead may wait in its host-tracked background command; a subagent waits in its own turn or hands the Task back to the lead.

## Inspect and continue answers

The compact view is the default. It shortens ids and hashes to twelve characters, shows state and limits, and marks omitted sections; it is not an authority document. Use the value as shown only when unique. Read an omitted part with `--section <path>` (for example `items.3:` or `items.3:40`); page long lists with `--list-next --next-from <N>`, or save the whole answer with `--output`. `--view full` prints it all and may exceed the host's output limit. A saved output keeps the owner's full answer, whatever the display, and `--from` reads JSON or YAML. A compact omission is not an empty value.

A compact answer can be continued only when the fields needed are whole. Otherwise save or read the named section. Each Host answer records its workspace, goal and agent-session context; compact review navigation distinguishes a book's review entry from the published CRO assessment.

For a read-only request, omit `--output` and read stdout. For a continuation from stdin, pipe the full answer, not the compact display. Do not repeat a read that already answers the question.

## The Host and its workspace

Keep `serve --no-browser --stop-on-stdin` attached to an open stdin. Send `stop` or close that stream to stop and join work; a client disconnect does not stop the Host. Reuse the service already serving that workspace. Never stop someone else's service.

A fresh checkout uses its own locked environment and local UI build. When Evidence or CRO reports `CREATE_DECLARED_RETRIEVAL_ENVIRONMENT`, run its exact setup command in the declared environment; do not guess an interpreter or model download. Data-to-weights research needs no Evidence or retrieval setup.

`workspace show` gives input versions, recent research, Tasks, the data-update readback and intents. It is discovery metadata, not artifact validation. Read more only when needed. `study summary` checks metadata and publication bindings, not bulk evidence; `study show` verifies the evidence. A lifecycle score run has fold metrics only if its method produced them. Owner readbacks verify their own inputs; add no duplicate verification.

## Waits and return visits

`activity wait` has no timer; `--goal` also wakes for its goal's messages and closing. A Codex turn that must end may use `--notify codex-queue` when `codex` and `CODEX_THREAD_ID` are available; its wake is a user message, not an instruction. A subagent waits in its own turn or hands the Task to the lead. Use `--max-wait` only when the command needs a cap. `activity recent` reads the last events by session and goal. `recovery list` shows unfinished Tasks and their owner-permitted recovery. A late heartbeat is not a dead Task. `--request-timeout` defaults to 120 seconds and caps at 600; it bounds only HTTP wait, not the Task. Cancellation must be followed to its actual state.

After a restart, rediscover Tasks and plan current work from its declaration. Reuse completed research; never rebuild data, retrain or rewrite an old identity just to display or export it. Historical evidence reuse and a new replay that refuses a changed execution binding are different claims. Counts describe the read unless the owner gives an execution count. For exact reuse, `task_id: null` means no new Task; read the named existing publication. `study verify <task>` verifies an existing execution without admitting work. PLAN's expected calls describe potential work; `EXISTING_EXECUTION_CANDIDATE` does not promise a zero-work RUN. A code fix or authority grant is not a declaration edit.

On every fresh session, read `workspace show`, then `strategy-book controls --package <package>` for the exact installed package and its `activation` before planning forward work. If `INACTIVE`, open the exact Portfolio activation action for the person; follow a held reason when no activation is offered. Installation, person activation and daily automation are separate decisions ([leading research](research-lead.md)).

## Study drafts and evidence links

Continue a saved result with `study draft --from "<out>/readback.json" --output "<out>/draft.json" --save-declaration "<out>/edited.yaml"`, edit the YAML, then `study plan --from "<out>/draft.json" --file "<out>/edited.yaml"`. An unchanged declaration can plan from the draft alone or a piped full draft. An omitted binding keeps the origin input, never the newest. Read `declaration_changes` and `execution_intent`; even a reuse candidate needs RUN to validate it. Refusal `fields` and `message` name declaration errors. `risk-link add --from "<out>/book.json" --risk-study <risk_task>` attaches Risk as evidence; it never changes weights.

## Feature changes

Use the returned PLAN and BUILD requests. A formula feature is one `CREATE` with `formula`, an explicit `preprocessing_recipe` and a `reason`; trial it only against an eligible completed Alpha study from Factor on the same input. A refusal lists eligible studies. `PREPROCESSED_VALUES` builds raw and preprocessed values in one Task; its `research` edge opens Factor controls for that exact source. `RAW_VALUES` is the lower-level option. A trial builds its own feature. Prepared columns, Factor context, curation and model inputs may have different counts; read their axes. Fresh research starts from returned controls and declarations; nothing activates by default.
