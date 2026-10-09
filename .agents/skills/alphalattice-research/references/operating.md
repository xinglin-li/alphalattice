# Operating the CLI
Date: 2026-10-08

## Setup and launch

Windows is the certified platform for this source-checkout setup; macOS and Linux are unverified. Use Python 3.12, uv and a browser from the checkout root. Dependency setup needs permission to download locked packages or a local cache; it prepares no research data:

```powershell
uv sync --locked --all-extras
.venv/Scripts/python.exe scripts/build_local_web_ui.py --product
uv export --locked --all-extras --no-dev --no-emit-project --no-hashes --output-file .venv/alphalattice-runtime-requirements.txt
uv tool install --python 3.12 --editable . --with-requirements .venv/alphalattice-runtime-requirements.txt --constraints .venv/alphalattice-runtime-requirements.txt
```

`uv sync` installs the checkout's command editable, so source edits are live; the tool install is editable too and pins its runtime to the exported lock. When `uv tool dir --bin` is not on the shell's PATH, run `uv run alphalattice` or `python scripts/run_alphalattice.py` from the checkout and go on; a persistent PATH is the person's shell setting. Each host command starts a fresh shell, so variables do not carry over. Run `uv tool uninstall alphalattice` before switching installation legs.

For a release wheel, use a separate directory; no checkout is needed. Extract its runtime lock, then install the wheel with it as requirements and constraints:

```powershell
python -c "from zipfile import ZipFile; from pathlib import Path; Path('alphalattice-runtime-requirements.txt').write_bytes(ZipFile('alphalattice-0.1.3-py3-none-any.whl').read('alphalattice/_runtime/config/release/runtime-requirements.txt'))"
uv tool install --python 3.12 ./alphalattice-0.1.3-py3-none-any.whl --with-requirements alphalattice-runtime-requirements.txt --constraints alphalattice-runtime-requirements.txt
```

Configure the shipped guidance with the tool environment's own interpreter, beneath `uv tool dir` (not the checkout's `.venv`):

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> configure --host <codex-or-claude-code>
```

`configure` refuses to overwrite different files and answers `continue_here`: the files to read, the cards to start specialists from and a one-line `disclosure` for the person. A session opened with its `open_session` command loads them itself; offer it, never require it. On Claude Code with Bedrock, Vertex or Foundry, set `ANTHROPIC_DEFAULT_SONNET_MODEL` to pin a Sonnet version. Evidence review needs its declared retrieval environment, local model packs and an admitted issuer-source package; when `serve` reports one missing, run the setup command it returns.

Start the Host on an explicit workspace, a new directory for new research; the launcher initializes it without preparing data:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --stop-on-stdin
```

Keep it attached to stdin; `stop` or a closed stream stops it after its workers join, and a client disconnect does not. Reuse the Host already serving a workspace, never start a second writer, and never stop someone else's service. After a restart, open the new launch link.

## Command contract

The grammar is `alphalattice <object> <action>`, then any id and flags; `request` sends a whole operation document. Commands do not prompt and have no aliases. Global `--workspace`, `--view`, `--goal` and `--lang` may appear anywhere.

- Every command prints one envelope: `{"outcome", "status", "data", "failure_code", "detail", "next_requests", "next_commands", "timing"}`. The owner's answer sits under `data`, its `next_action` with it; a refusal fills `failure_code` and `detail`. A file saved with `--output` is the owner's whole answer, not the envelope, and `answer show --file` reads it.
- The answer's top-level `next_action` and `next_requests` decide the next step. A part's own `next_action`, such as a nested `network_access`, describes that part only and never overrides them.
- Run `next_commands` as returned; fill a `next_templates` entry only with its named choices. Take a named edge with `request --from "<out>/answer.json" --action <returned-action-name>`; same-named requests from separate parts carry a qualified name. `--choices` fills open fields and never replaces a bound reference.
- Take ids, hashes and choices exactly from an answer. A compact id is usable only when unique. An offered edge is navigation, not proof its prerequisites hold; the Host revalidates every continuation.
- Write outputs to new paths under a new `<out>` directory in the workspace, never into the checkout. `--output` saves the full answer, `--save-declaration` an editable declaration; `--file` reads a document, `-` reads stdin. The CLI overwrites no file.
- Consult help or `schema show` only when the answer lacks a field or form you need.

<!-- Generated from the CLI's registered outcomes and _PROCEED meanings. -->
| Exit | Outcome | Do |
| --- | --- | --- |
| 0 | OK | read `status` and `data`, a result's `standing` first; an OK read is not approval or a finished Task |
| 1 | INVALID_INPUT | correct the named command, document or field within scope |
| 2 | REFUSED | request refused or work blocked, cancelled or stopped: read `failure_code`, `fields`, `detail` and `next_requests`; take only an authorized continuation, else report the stop |
| 3 | PENDING | follow the admitted Task or answer its decision; never resubmit queued or running work; use the answer's offered deferral resume after its retry time |
| 4 | NO_HOST | report it: the lead starts or reconnects the Host |

A refusal names its owner's code, detail and legal way on; take only that or a person-authorized act, never a workaround. If a compact reference is refused with `compact_reference_requires_full_response:<part>`, save the full answer or read that part with `--section <part>`.

## Failure and recovery

Keep the original Goal, Task and error, and tell a malformed request, a permission refusal, a data decision, a transport event and a product defect apart:

1. Correct a malformed request's named command, document or field. A permission refusal is a stop, not a bug; a code fix grants no authority and never weakens a scientific or integrity check.
2. Take a data decision through its offered preview and confirmation ([pipeline issues](pipeline-issues.md)).
3. After a timeout or lost connection, read the known Task before repeating anything: no answer does not prove no work was admitted.
4. Reproduce a product defect, fix its existing owner and add a regression. For changed source, restart only an idle Host you own and open its new link.
5. Take the stopped Task's offered resume yourself and say so in one line; replan or cancel only when none is offered. Changed method or input needs new evidence: keep the earlier result and follow the owner's new plan, never rewriting stored bindings.

Reuse completed research; never rebuild data, retrain or rewrite an identity to display or export it. For exact reuse, `task_id: null` means no new Task: read the named publication.

## Inspect and continue answers

The compact view shortens ids to twelve characters and marks omitted sections; an omission is not an empty value. Read a part with `--section <path>`, page with `--list-next --next-from <N>`, or save the whole answer with `--output` when you will continue from it, cite it or hand it on. `answer show --file "<out>/answer.json" --list-sections` rereads a saved answer offline, as saved and never reverified; send the owner's read again for the present state. `workspace show` is discovery, not validation: `study show` verifies a study's evidence, `study summary` only its metadata.

## Waits and return visits

A wait is one call, never a poll. `--wait` and `activity wait` follow a Task until it ends, needs a decision, is deferred or reports an incident, across Host restarts. Your host's shell call may return before the Task ends. On Claude Code, run the wait or agent verb with the Bash tool's `run_in_background` and act on its completion notice; never read its output or check its Task before the notice, and background tasks outlive the turn. On Codex, a turn ends its shell's children, so leave nothing running past it: `activity wait --task <task-id> --notify codex-queue` registers the wake with the Host and returns at once; end your turn, and the line the Host queues when the Task ends, needs a decision or is deferred opens the next, naming the command to read. An agent verb given `--notify codex-queue` registers its first running Task and returns; the wake re-runs the command to continue. A goal's wait and `--max-wait` stay inside the turn. `--each-stage` returns at each verified stage; use it only when a stage lets you act early, as when an Evidence unit becomes `PREPARED`. A subagent waits in its own turn or hands the Task to the lead. `recovery list` shows unfinished Tasks and their permitted recovery; a late heartbeat is not a dead Task, and a cancellation request is not a cancellation result.

## Studies and features

Continue a saved study with `study draft --from "<out>/run.json" --save-declaration "<out>/next.yaml" --output "<out>/draft.json"`, edit the YAML and plan it from the draft; an omitted binding keeps the origin input. A formula feature is one `CREATE` with a `formula`, an explicit `preprocessing_recipe` and a `reason`, trialled against a completed Alpha study from Factor on the same input; a refusal lists eligible studies. Nothing activates by default.
