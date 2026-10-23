# Operating the CLI
Last checked: 2026-10-22 against 0.2.3

## Report a problem

Run `alphalattice problem report --output "<out>/report.json"`; optionally name the
expected operations with `--expected-route <operation>`. Use only the admitted
sentence form `Report the <OPERATION> problem to the developers.` with a registered
operation for `--sentence`, never a conversation excerpt. The report lists its
included fields and contains no workspace paths, holdings, figures or file contents.
Show the whole report, then ask: "May I send this report to the developers?"
Without the person's yes, send nothing. After yes, run `gh auth status`; if signed
in, run `gh issue create --repo xinglin-li/alphalattice --title "[Bug]: Product problem" --label bug --body-file "<out>/report.json"`
with the exact file the person reviewed. If gh is unavailable or signed out, give
the report's prefilled `issue_url` for that same repository. Never add raw diagnostics,
conversation text or private data to either form.

## Setup and launch

At first use and any missing-dependency refusal, run the doctor from the checkout's root, `uv run python -m alphalattice.interface.local_application.native_setup doctor`, after the `uv sync` below and before any tool install. Fill each missing `host_readiness` item by its `who_decides` (only a missing item carries one), disclosing locked repairs and relaying one question before an outside-lock install.

Windows is the certified platform for this source-checkout setup; macOS and Linux are unverified. Use Python 3.12, uv and a browser from the checkout root; the lock brings the Node that builds the Workbench. macOS also needs libomp (`brew install libomp`), which LightGBM loads, and Linux libatomic1, which the locked Node loads, and libgomp1, which LightGBM loads (`sudo apt install libatomic1 libgomp1`). They are system packages outside the lock: doctor's `system_libraries` names those missing with one install command, and the person is asked once. The agent discloses locked dependency downloads in one line; they prepare no research data:

```powershell
uv sync --locked --all-extras
uv run python -m alphalattice.interface.local_application.native_setup doctor
uv run python scripts/build_local_web_ui.py --product
uv export --locked --all-extras --no-dev --no-emit-project --no-hashes --output-file .venv/alphalattice-runtime-requirements.txt
uv tool install --python 3.12 --editable . --with-requirements .venv/alphalattice-runtime-requirements.txt --constraints .venv/alphalattice-runtime-requirements.txt
```

Run each missing item's returned `facts.commands` (the Workbench's `facts.command`). The retrieval runtime resolves `config/requirements-retrieval.lock` online as the agent's act: say its `disclosure` in one line, even on an empty cache, and ask no second question or renewed first-use network consent. Explicit `--offline` reads only the local cache. Software outside the lock still needs the person's permission. Run the doctor again after repairs and configuration; optional user-layer readings and Host-dependent measurements do not prevent research by themselves.

`uv sync` installs the checkout command editable, and the tool install pins its runtime to the exported lock. From a checkout you may also run `python scripts/run_alphalattice.py` or `uv run alphalattice`. Off PATH, use the binding block in the shared guides and pass the matching absolute prefix to every specialist; no shell update or restart is needed. Portable examples use alphalattice as a placeholder. When serving from another shell, set ALPHALATTICE_SHELL to your specialists' shell: posix or powershell. Each host command starts a fresh shell, so variables do not carry over. The first use goal belongs to the session that ran `first-use prepare`; the same sentence and date reuse it from another session. With no agent session, put `--goal <goal_id>` before each later step. Run `uv tool uninstall alphalattice` before switching installation legs.

For a release wheel, use a separate directory; no checkout is needed. Extract its runtime lock, then install the wheel with it as requirements and constraints:

```powershell
python -c "from zipfile import ZipFile; from pathlib import Path; Path('alphalattice-runtime-requirements.txt').write_bytes(ZipFile('alphalattice-0.2.3-py3-none-any.whl').read('alphalattice/_runtime/config/release/runtime-requirements.txt'))"
uv tool install --python 3.12 ./alphalattice-0.2.3-py3-none-any.whl --with-requirements alphalattice-runtime-requirements.txt --constraints alphalattice-runtime-requirements.txt
```

A checkout is its own agent project: configure it from its root, `uv run python -m alphalattice.interface.local_application.native_setup configure --host <codex-or-claude-code>`. On the wheel leg, configure the separate agent folder with the tool environment's own interpreter, beneath `uv tool dir`. In a shell with no agent session, give the doctor the same `--host` choice; otherwise it defaults to Codex:

```powershell
$toolDir = uv tool dir
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> doctor --host <codex-or-claude-code>
& "$toolDir/alphalattice/Scripts/python.exe" -m alphalattice.interface.local_application.native_setup --project <agent-project> configure --host <codex-or-claude-code>
```

On macOS and Linux the interpreter is `bin/python`:

```bash
"$(uv tool dir)/alphalattice/bin/python" -m alphalattice.interface.local_application.native_setup --project <agent-project> configure --host <codex-or-claude-code>
```

`configure` refuses to overwrite different files and answers `continue_here`: the files to read, the cards to start specialists from and a one-line `disclosure` for the person. A session opened with its `open_session` command loads them itself; offer it, never require it. On Claude Code with Bedrock, Vertex or Foundry, set `ANTHROPIC_DEFAULT_SONNET_MODEL` to pin a Sonnet version. Evidence review needs its declared retrieval environment, local model packs and an admitted issuer-source package; when `serve` reports one missing, run the setup command it returns.

In released 0.2.3, when Claude Code startup returns `SERVE_FROM_A_PERSON_OWNED_TERMINAL`, give the person its offered command once and wait for their word before continuing. This is separate from research network consent.

The next terminal-panel probe is a later-checkout preview. Use it only when the installed checkout advertises that capability and offers its matching route.

For a new Host on Claude Code desktop, use its available `run_in_terminal` tool to probe the Terminal panel shell. Only a successful `ok: True, in_job: False` result permits declaring `ALPHALATTICE_TERMINAL_PANEL_HOST=claude-code` for this serve invocation (set the environment variable for the launch, then remove it). Codex's panel capability is undeclared; never infer it from the host's name.

In the Claude Code desktop Terminal panel (never the command tool), run this Windows probe; a failed API call is not proof of escape:

```powershell
uv run python -c "import ctypes; inside=ctypes.c_int(); ok=ctypes.WinDLL('kernel32',use_last_error=True).IsProcessInJob(ctypes.c_void_p(-1),None,ctypes.byref(inside)); print({'ok':bool(ok),'in_job':bool(inside.value)})"
```

<!-- message: host.start.terminal.guide -->
On Claude Code desktop, use the available Terminal panel through `run_in_terminal`; on Codex, use an available agent-driven Terminal panel in the same way. If detached start is refused, choose the first applicable `terminal_routes` offer and run `terminal_command` yourself there. Only without such a panel, give the person the same command for a terminal they own, then wait for their word. The Host cannot inspect your tools; its escape and authenticated readiness checks still decide STARTED.

The app may ask the person to approve that one line.

Start the Host on an explicit workspace, a new directory for new research; the launcher initializes it without preparing data. Keep naming `--workspace` on the first-use command unless the session is already bound:

```powershell
alphalattice --workspace "workspaces/my-research" serve --no-browser --detach
```

Use `serve --no-browser --detach` on both Claude Code and Codex. Never keep a long-lived serve in a background Bash task: its time limit can stop the Host and running Tasks. STARTED requires an authenticated ready Host proved outside its parent Job. A readiness timeout can leave the named process starting: follow its offered log or stop recovery before another launch. An explicitly session-only Host still accepts `--stop-on-stdin`; a held update then waits for the next session. Never bypass Job containment through a broker, service or scheduler. The next marked authenticated-reuse procedure is a later-checkout preview; in 0.2.3 follow the `ALREADY_SERVING` answer actually returned by the launcher.

<!-- message: host.reuse.guide --> A matching serving Host can be reused even when this shell cannot detach a new process. ALREADY_SERVING reuses only an authenticated Host for this workspace and instance, preserving its authority and idle policy. Activity's observer names its instance, process and start time; the Workbench shows the same Host. On an unverified or stale descriptor, read the named Host log and identify its owning process before retrying the original serve command; leave the descriptor and writer lease unchanged. `alphalattice host stop` stops it once its running work has ended: it lets its Tasks finish, joins its workers and releases the workspace, cancelling nothing; a client disconnect does not stop it. Where your shell holds the launch's standard input open, the Host is also tied to it: sending `stop` to it, or closing it, does the same. In 0.2.3 a detached Host serves until its explicit stop. Later checkouts add the following idle policy: a detached Host stays while a held update, Task, request or enabled automation needs it; otherwise it stops after 900 idle seconds (0 disables idle shutdown). Once it serves, its ready lines end with the next action, OPEN_THE_WORKBENCH. The rule: the Workbench opens in the coding app's built-in browser, visibly, once, never in a separate system browser, so keep `--no-browser` on every serve you start. In Claude Code's desktop app, load the browser pane's tools and open the printed link with `preview_start` (url), which shows the person a card in the chat; never `navigate` or a background tab for this first open. In Codex, open it in Codex's built-in browser. Then tell the person in one line that the Workbench is open there, with the link. Only a host with no built-in browser (a terminal CLI) gives the person the link alone. The first use's first `ask_now` line carries the same link, to relay as it stands. Open it so at the start of every session, first use or daily: on a Host already serving, your Session's first answer carries `workbench`, the same next action with a launch link of its own that opens once; open that link the same way, once. Never navigate or click the Workbench to show work. Reuse the Host already serving a workspace, never start a second writer, and never stop someone else's service. An installation upgrade does not authorize stopping or restarting a running Host; take any required transition from the upgrade or readiness answer, and stop your own Host with `host stop` where it asks for that.

## Command contract

The grammar is `alphalattice <object> <action>`, then any id and flags; `request` sends a whole operation document. Commands do not prompt and have no aliases. Global `--workspace`, `--view`, `--goal` and `--lang` may appear anywhere.

- Every command prints one envelope: `{"outcome", "status", "data", "failure_code", "detail", "next_requests", "next_commands", "timing"}`. The owner's answer sits under `data`, its `next_action` with it; a refusal fills `failure_code` and `detail`. A file saved with `--output` is the owner's whole answer, not the envelope, and `answer show --file` reads it.
- A refusal may add `error_info`: registered `category` and owner `domain`, exact failure `reason`, catalog `message_id`/`message` and safe `field_violations`. Unknown supplied keys appear as `<unrecognized>`; list indices stay numeric. A category grants no retry or authority: follow the unchanged offered request or ask the person where required.
- On Claude Code and Codex, use `--view summary` for reads. Its complete envelope stays within 32 KiB with whole offers and a `full_answer` pointer. `--output "<out>/answer.json"` saves the exact complete owner bytes; give it before reading stdin or sending a write in summary mode. Follow the pointer for omitted data; an omission is not an empty value. Legacy `--view full` stdout remains available with a deprecation notice; prefer the summary and saved answer.
- The answer's top-level `next_action` and `next_requests` decide the next step. A part's own `next_action`, such as a nested `network_access`, describes that part only and never overrides them.
- Run `next_commands` as returned; fill a `next_templates` entry only with its named choices. Take a named edge with `alphalattice request --from "<out>/answer.json" --action <returned-action-name>`; same-named requests from separate parts carry a qualified name. `--choices` fills open fields and never replaces a bound reference.
- For an invalid Feature document or strategy declaration, send its offered controls or inputs request, repair the invalid fields and plan again with the same research choices.
- Take ids, hashes and choices exactly from an answer. A compact id is usable only when unique. An offered edge is navigation, not proof its prerequisites hold; the Host revalidates every continuation.
- Write outputs under a new `<out>` directory in the workspace, never into the checkout. `--output` saves the full answer, `--save-declaration` an editable declaration; `--file` reads a document, `-` reads stdin. A retry may replace a saved JSON/YAML refusal without an admitted Task and prints `output_note`; other files stay protected.
- Consult help or `schema show` only when the answer lacks a field or form you need.
- Long workflows print request steps and Task stage changes on stderr; stdout stays one envelope. Compact update readback keeps counts, top holdings, dates, review and next steps; read each `omitted_sections` path with `--section <path>`, or save the complete answer with `--output "<out>/answer.json"`.

<!-- Generated from the CLI's registered outcomes and _PROCEED meanings. -->
| Exit | Outcome | Do |
| --- | --- | --- |
| 0 | OK | read `status` and `data`, a result's `standing` first; an OK read is not approval or a finished Task |
| 1 | INVALID_INPUT | correct the named command, document or field within scope |
| 2 | REFUSED | request refused or work blocked, cancelled or stopped: read `failure_code`, `fields`, `detail` and `next_requests`; take only an authorized continuation, else report the stop |
| 3 | PENDING | follow the admitted Task or answer its decision; never resubmit queued or running work; use the answer's offered deferral resume after its retry time |
| 4 | NO_HOST | report it: the lead starts or reconnects the Host |

An accepted bundle submission exits 0; its `task_lifecycle` and offered Task read name the continuing review. Compact answers lead with `status`, `next_action` and `task_id`, repeated from the owner without removing its fields.

A refusal names its owner's code, detail and legal way on; take only that or a person-authorized act, never a workaround. If a compact reference is refused with `compact_reference_requires_full_response:<part>`, save the full answer or read that part with `--section <part>`.

## Failure and recovery

Keep the original Goal, Task and error, and tell a malformed request, a permission refusal, a data decision, a transport event and a product defect apart:

1. For a spelling or shape mistake, send the filled correction as offered. For an unreadable document, read the offered schema, repair the named line and column, and resend the original request. For a content mistake, read its named rule and offered valid values, change only the named field and send the corrected request; restore a missing plan hash from the original answer, never the latest plan. A permission refusal is a stop, not a bug; a code fix grants no authority and never weakens a scientific or integrity check.
2. Take a data decision through its offered preview and confirmation ([pipeline issues](pipeline-issues.md)).
3. After a timeout or lost connection, send the executable `read` printed with the Task id before stage progress: no answer does not prove no work was admitted. Structured `--output` saves the admission first and names `output_file`; without it, no file is saved.
4. Reproduce a product defect, fix its existing owner and add a regression. For changed source, stop only an idle Host you own with `host stop`, serve it again and open its new link.
5. Take the stopped Task's offered resume yourself and say so in one line; replan or cancel only when none is offered. Changed method or input needs new evidence: keep the earlier result and follow the owner's new plan, never rewriting stored bindings.

Reuse completed research; never rebuild data, retrain or rewrite an identity to display or export it. For exact reuse, `task_id: null` means no new Task: read the named publication.

## Inspect and continue answers

The compact view shortens ids to twelve characters and marks omitted sections; an omission is not an empty value. Read a part with `--section <path>`, page with `--list-next --next-from <N>`, or save the whole answer with `--output` when you will continue from it, cite it or hand it on. `answer show --file "<out>/answer.json" --list-sections` rereads a saved answer offline, as saved and never reverified; send the owner's read again for the present state. `alphalattice workspace show` is discovery, not validation: `alphalattice study show` verifies a study's evidence, `alphalattice study summary` only its metadata.

## Waits and return visits

A wait is one call, never a poll. `--wait` and `alphalattice activity wait` follow a Task until it ends, needs a decision, is deferred or reports an incident, across Host restarts. Your host's shell call may return before the Task ends. On Claude Code, run the wait or agent verb with the Bash tool's `run_in_background` and act on its completion notice; never read its output or check its Task before the notice, and background tasks outlive the turn. On Codex, a turn ends its shell's children, so leave nothing running past it; the Host started with `serve --detach` is apart from them. With no active Codex goal (`get_goal` reads none), `alphalattice activity wait --task <task-id> --notify codex-queue` registers the wake with the Host and returns at once; end your turn, and the line the Host queues when the Task ends, needs a decision or is deferred opens the next, naming the command to read. An agent verb given `--notify codex-queue` registers its first running Task and returns; the wake re-runs the command to continue. When the registration answers `local_client.codex_unreachable`, the Host cannot reach Codex (it runs inside your sandbox, as a Host you started does), so no wake will come: wait inside the turn with `alphalattice activity wait --task <task-id>`, as under an active goal. Under an active Codex goal, the goal's next turn starts before a queued line is taken, which then waits for the person's Steer: wait inside the turn instead, with `alphalattice activity wait --task <task-id>` or the verb's `--wait` started once, without `--notify`, and read that same call's output again at each yield until it returns. When that wait is long, the Task's `resource_estimate` or a hold's `held_until` more than about 20 minutes away, tell the person in one line (held until about 18:00 ET; the Host finishes it; I pick it up when you return) and never mark the goal complete: read the Task once on each goal turn, and when the same wait has held three goal turns, set the goal blocked with that reason. The Host finishes the work, and the next session's first read finds it. A goal's wait and `--max-wait` stay inside the turn. `--each-stage` returns at each verified stage; use it only when a stage lets you act early, as when an Evidence unit becomes `PREPARED`. A subagent waits in its own turn or hands the Task to the lead. `recovery list` shows unfinished Tasks and their permitted recovery; a late heartbeat is not a dead Task, and a cancellation request is not a cancellation result. Whenever a first-use step goes to the person, a Claude Code lead keeps `alphalattice activity wait --goal <goal>` in the background: their act's Task ends it. The Host also wakes a Codex lead when the person's act admits work under the first use.

## Studies and features

For execution capacity, daily automation or JSON/YAML conversion, use the [request index](intents.md); these are separate from the study's numerical method.

Continue a saved study with `alphalattice study draft --from "<out>/run.json" --save-declaration "<out>/next.yaml" --output "<out>/draft.json"`, edit the YAML and plan it from the draft; an omitted binding keeps the origin input. A formula feature is one `CREATE` with a `formula`, an explicit `preprocessing_recipe` and a `reason`, trialled against a completed Alpha study from Factor on the same input; a refusal lists eligible studies. Nothing activates by default.
