"""A plugin run's staging folder, the Host's checks on what it staged, and its one commit."""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable, Collection
from concurrent.futures import Future
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from alphalattice.capabilities.domain_contracts.domain.errors import WorkspaceSecurityError
from alphalattice.control.task_control.plugin_child import child_answer, signal_started
from alphalattice.control.task_control.plugin_guard import STARTED, Guard, Refused, install
from alphalattice.control.task_control.plugin_staging import (
    OUTPUT,
    PROVENANCE,
    PluginStaging,
    StagingRefused,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.protocols.plugin_contract import (
    ActionReceipt,
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionContext,
    ExecutionMode,
    Lease,
    ReceiptStatus,
)

TEST_SOURCE = "src/alphalattice/control/task_control/plugin_staging.py"

ROLE = "blake3:" + "1" * 64
SPEC = ActionSpec(
    capability_id="factor.custom_momentum",
    plugin_id="sample-factor",
    code_role_hash=ROLE,
    effect_mode=EffectMode.STAGING_WRITE,
    execution_mode=ExecutionMode.SANDBOX_SUBPROCESS,
    lease=Lease(memory_bytes=1 << 30, threads=2, timeout_ms=60_000),
    input_schema={"type": "object"},
    output_schema={
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    },
    inputs=("raw_bar",),
)
READ = CatalogRead(snapshot_id="lake:42", input_versions={"raw_bar": 7})
Plugin = Callable[[ExecutionContext], ActionReceipt]


def link_directory(link: Path, target: Path) -> None:
    """A directory symlink: Windows makes one in developer mode or with the right, as CI has."""
    link.symlink_to(target, target_is_directory=True)


@pytest.mark.proves("task_control.a_plugin_run_commits_only_what_its_staging_proves")
def test_a_staged_run_commits_once_only_what_its_key_and_schema_prove(tmp_path: Path) -> None:
    """An admitted run commits its checked answer once and every caller gets that answer, while
    each inadmissible, foreign, linked, reserved, invalid or stale run is refused by name."""
    workspace, artifacts, outside = tmp_path / "workspace", tmp_path / "artifacts", tmp_path / "x"
    outside.write_text("{}", encoding="utf-8")
    workspace.mkdir()
    latest = {"raw_bar": 7}

    def catalog(tables: Collection[str], snapshot: str | None) -> CatalogRead:
        if snapshot is None:
            return CatalogRead(snapshot_id="lake:43", input_versions=dict(latest))
        older = {"lake:41": 6}  # a kept snapshot before the latest version
        return CatalogRead(snapshot_id=snapshot, input_versions={"raw_bar": older.get(snapshot, 7)})

    resolver = ArtifactResolver(artifacts)
    staging = PluginStaging(
        workspace, tmp_path / "data", resolver, WorkspaceMutationGate(), catalog
    )
    elsewhere, behind = tmp_path / "elsewhere", tmp_path / "behind"
    elsewhere.mkdir()
    (behind / "staging").mkdir(parents=True)
    link_directory(behind / "staging" / "plugin-runs", elsewhere)  # a linked parent of each run
    linked_runs = PluginStaging(
        behind, tmp_path / "data", resolver, WorkspaceMutationGate(), catalog
    )
    invocation = SPEC.invoke({"window": 20}, READ)
    calls: list[ExecutionContext] = []
    attached: list[object] = []

    def plugin(answer: object, key: str = invocation.idempotency_key, also: str = "") -> Plugin:
        def run(context: ExecutionContext) -> ActionReceipt:
            calls.append(context)
            (context.staging_dir / OUTPUT).write_text(json.dumps(answer), encoding="utf-8")
            if also == "link":
                os.link(outside, context.staging_dir / "data.json")
            elif also == "provenance":
                (context.staging_dir / PROVENANCE).write_text("{}", encoding="utf-8")
            elif also == "attach":  # requests for the same key while this run is in flight
                with pytest.raises(StagingRefused) as foreign:
                    staging.request(other_code, invocation, plugin({"score": 9.0}))
                running = staging.request(SPEC, invocation, plugin({"score": 9.0}))
                attached.extend([foreign.value.code, running.cancel(), running])
            return ActionReceipt(
                invocation_id=context.invocation_id,
                idempotency_key=key,
                status=ReceiptStatus.OK,
                wall_ms=1,
                peak_bytes=1,
            )

        return run

    def refused(
        run: Plugin, spec: ActionSpec = SPEC, before=lambda: None, read=invocation, owner=staging
    ) -> str:
        before()
        with pytest.raises(StagingRefused) as error:
            owner.request(spec, read, run).result()
        return error.value.code

    other_code = SPEC.model_copy(update={"code_role_hash": "blake3:" + "2" * 64})
    # A reference the default resolver would fetch: a file holding a schema that accepts all.
    remote = SPEC.model_copy(update={"output_schema": {"$ref": outside.as_uri()}})
    unbound = SPEC.invoke({"window": 20}, CatalogRead(snapshot_id="lake:41", input_versions=latest))
    refusals = [
        refused(
            plugin({"score": 1.0}), SPEC.model_copy(update={"effect_mode": EffectMode.READ_ONLY})
        ),
        refused(plugin({"score": 1.0}), other_code),
        refused(plugin({"score": 1.0}), read=unbound),
        refused(plugin({"score": 1.0}, key="blake3:" + "3" * 64)),
        refused(plugin({"score": "high"})),
        refused(plugin({"score": 1.0}), remote),
        refused(plugin({"score": 1.0}, also="link")),
        refused(plugin({"score": 1.0}, also="provenance")),
        refused(plugin({"score": 1.0}), before=lambda: latest.update(raw_bar=8)),
        refused(plugin({"score": 1.0}), owner=linked_runs),
    ]
    published = artifacts / "plugin-result"
    nothing_committed = not published.exists() or not any(published.iterdir())
    latest["raw_bar"] = 7
    committed = staging.request(SPEC, invocation, plugin({"score": 1.5}, also="attach")).result()
    again = staging.request(SPEC, invocation, plugin({"score": 2.5})).result()
    linked = tmp_path / "linked"
    linked.mkdir()
    os.link(outside, linked / OUTPUT)
    with pytest.raises(WorkspaceSecurityError):  # the artifact owner refuses it unexposed
        resolver.publish_plugin_result(linked, "blake3:" + "4" * 64)
    link_directory(tmp_path / "walk-root", elsewhere)  # a linked walk root over an empty folder
    with pytest.raises(WorkspaceSecurityError):
        resolver.publish_plugin_result(tmp_path / "walk-root", "blake3:" + "5" * 64)

    stored = sorted(path.name for path in next(published.iterdir()).iterdir())
    assert refusals == [
        "task_control.plugin_not_staging_write",
        "task_control.plugin_role_hash_mismatch",
        "task_control.plugin_read_unbound",
        "task_control.plugin_provenance_mismatch",
        "task_control.plugin_output_invalid",
        "task_control.plugin_output_invalid",
        "task_control.plugin_staged_entry_invalid",
        "task_control.plugin_staged_entry_invalid",
        "task_control.plugin_inputs_changed",
        "task_control.plugin_staging_unproved",
    ]
    # Eight runs: an inadmissible request runs no code, and the attached request runs none.
    assert nothing_committed and len(calls) == 8
    assert committed.output == again.output == {"score": 1.5} and stored == [OUTPUT, PROVENANCE]
    assert committed.artifact == again.artifact
    assert committed.artifact.content_hash == invocation.idempotency_key
    assert attached[:2] == ["task_control.plugin_role_hash_mismatch", False]
    running = attached[2]
    assert isinstance(running, Future) and running.result() is committed
    assert len(list(published.iterdir())) == 1 and not any(elsewhere.iterdir())
    assert not any((workspace / "staging" / "plugin-runs").iterdir())


NETWORK = "task_control.plugin_network_refused"
SUBPROCESS = "task_control.plugin_subprocess_refused"
OUTSIDE = "task_control.plugin_write_outside_staging"
IMPORT = "task_control.plugin_import_outside_environment"
W = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
# One case per form the guard refuses, by CPython's own audit event and its arguments, and the
# forms it passes; "{s}" is the staging folder, "{o}" a path outside it, "{l}" a link inside it,
# "{e}" the environment's folder. A "module" case is the spec the child's first finder judges; an
# "import" case's first argument says whether that finder is still first.
GUARD_CASES = [
    ("connect", "socket.connect", ("socket", ("127.0.0.1", 9)), False, NETWORK),
    ("getaddrinfo", "socket.getaddrinfo", ("example.com", 80, 0, 0, 0, 0), False, NETWORK),
    ("gethostbyname", "socket.gethostbyname", ("example.com",), False, NETWORK),
    ("gethostbyaddr", "socket.gethostbyaddr", ("127.0.0.1",), False, NETWORK),
    ("getnameinfo", "socket.getnameinfo", (("127.0.0.1", 80), 0), False, NETWORK),
    ("popen", "subprocess.Popen", ("python", ["python"], None, None), False, SUBPROCESS),
    ("system", "os.system", (b"echo",), False, SUBPROCESS),
    ("exec", "os.exec", ("python", ["python"], None), False, SUBPROCESS),
    ("spawn", "os.spawn", (0, "python", ["python"], None), False, SUBPROCESS),
    ("posix_spawn", "os.posix_spawn", ("python", ["python"], {}), False, SUBPROCESS),
    ("fork", "os.fork", (), False, SUBPROCESS),
    ("open-write", "open", ("{o}/a", "w", W), False, OUTSIDE),
    ("open-append", "open", ("{o}/a", "a", os.O_WRONLY | os.O_APPEND), False, OUTSIDE),
    ("os-open-create", "open", ("{o}/a", None, os.O_RDWR | os.O_CREAT), False, OUTSIDE),
    ("rename-out", "os.rename", ("{s}/a", "{o}/b", -1, -1), False, OUTSIDE),
    ("remove", "os.remove", ("{o}/a", -1), False, OUTSIDE),
    ("rmdir", "os.rmdir", ("{o}/d", -1), False, OUTSIDE),
    ("mkdir", "os.mkdir", ("{o}/d", 511, -1), False, OUTSIDE),
    ("rmtree", "shutil.rmtree", ("{o}", None), False, OUTSIDE),
    ("through-link", "open", ("{l}/a", "w", W), False, OUTSIDE),
    ("relative-to-dir-fd", "os.remove", ("a", 3), False, OUTSIDE),
    ("rename-to-own-dir-fd", "os.rename", ("{s}/a", "b", -1, 3), False, OUTSIDE),
    ("outside-module", "module", ("planted", "{o}/planted.py"), False, IMPORT),
    ("shadowed-stdlib", "module", ("json", "{o}/json/__init__.py"), False, IMPORT),
    ("outside-submodule", "module", ("json.planted", "{o}/planted.py"), False, IMPORT),
    ("outside-namespace", "module", ("spread", None, "{o}/spread"), False, IMPORT),
    ("read-outside", "open", ("{o}/a", "r", os.O_RDONLY), False, None),
    ("write-inside", "open", ("{s}/out/a.parquet", "wb", W), False, None),
    ("rename-inside", "os.rename", ("{s}/a", "{s}/b", -1, -1), False, None),
    ("leased-network", "socket.connect", ("socket", ("127.0.0.1", 9)), True, None),
    ("environment-module", "module", ("json.decoder", "{e}/json/decoder.py"), False, None),
    ("built-in-module", "module", ("sys", "built-in"), False, None),
    ("same-name-outside", "module", ("json.decoder", "{e}/json/decoder.py", "{o}/d.py"), 0, IMPORT),
    ("found-nowhere", "module", ("not_held_anywhere",), False, None),
    ("finder-displaced", "import", ("displaced", None, None, None, None), False, IMPORT),
    ("finder-first", "import", ("first", None, None, None, None), False, None),
    ("other-event", "os.listdir", ("{o}",), False, None),
]


@pytest.mark.proves("task_control.a_guarded_plugin_is_refused_each_wrong_reach_by_name")
@pytest.mark.parametrize(
    ("event", "args", "network", "code"),
    [case[1:] for case in GUARD_CASES],
    ids=[case[0] for case in GUARD_CASES],
)
def test_the_guard_refuses_each_wrong_reach_by_its_audit_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    event: str,
    args: tuple[object, ...],
    network: bool,
    code: str | None,
) -> None:
    """The audit hook refuses each wrong reach by name and passes every allowed form."""
    staging, outside = tmp_path / "staging", tmp_path / "outside"
    staging.mkdir()
    outside.mkdir()
    monkeypatch.chdir(staging)  # the child's own working directory is inside its staging folder
    if "{l}" in str(args):
        link_directory(staging / "link", outside)
    stdlib = os.path.dirname(os.__file__)  # the environment's folders, as the child reads them
    guard = Guard(str(staging), network=network, environment=frozenset({stdlib}))
    places = {"{s}": str(staging), "{o}": str(outside), "{l}": str(staging / "link")}
    places["{e}"] = stdlib

    def placed(value: object) -> object:
        if isinstance(value, str):
            for mark, path in places.items():
                value = value.replace(mark, path)
        return value

    where = tuple(placed(value) for value in args)
    if event == "module":  # the spec the import system found, judged by its own origin
        name, *origin = where
        spec = None if not origin else ModuleSpec(str(name), None, origin=origin[0])
        if len(origin) == 2 and origin[0] is None:
            spec = ModuleSpec(str(name), None, is_package=True)
            spec.submodule_search_locations = [str(origin[1])]
        elif len(origin) == 2:  # the same name approved once, then found again elsewhere
            assert guard.module_refusal(str(name), spec) is None
            spec = ModuleSpec(str(name), None, origin=str(origin[1]))
        found = guard.module_refusal(str(name), spec)
    elif event == "import":  # the child's finder, first in the import system or displaced
        guard.finder = _Finder()
        first = guard.finder if where[0] == "first" else _Finder()
        with monkeypatch.context() as patch:
            patch.setattr(sys, "meta_path", [first, *sys.meta_path])
            found = guard.refusal(event, where)
    else:
        found = guard.refusal(event, where)
    assert (found and found[0]) == code


_Hook = Callable[[str, tuple[object, ...]], None]


class _Finder:
    """A finder that finds nothing, standing in the first place of the import system."""

    @staticmethod
    def find_spec(*_: object) -> None:
        """Find no module."""


def _raising(event: str, _: tuple[object, ...]) -> None:
    raise RuntimeError(f"an earlier hook refuses {event}")


def _heard(hooks: list[_Hook]) -> Callable[[str], None]:
    def audit(event: str) -> None:
        for hook in hooks:
            hook(event, ())

    return audit


@pytest.mark.proves("task_control.a_guarded_plugin_is_refused_each_wrong_reach_by_name")
@pytest.mark.parametrize(
    ("registered", "refused"),
    [
        pytest.param(True, False, id="hook-hears-the-probe"),
        pytest.param(False, True, id="registration-suppressed"),
        pytest.param(None, True, id="registration-refused"),
        pytest.param("after-a-raising-hook", True, id="probe-raised"),
    ],
)
def test_a_child_starts_only_once_its_hook_hears_an_event(
    tmp_path: Path, registered: bool | str | None, refused: bool
) -> None:
    """The child refuses its plugin's start unless its audit hook demonstrably hears events."""
    guard = Guard(str(tmp_path), network=False, environment=frozenset())
    hooks: list[_Hook] = [_raising] if registered == "after-a-raising-hook" else []

    def add(hook: _Hook) -> None:
        if registered is None:
            raise RuntimeError("an earlier hook refused sys.addaudithook")
        if registered:
            hooks.append(hook)

    if refused:
        with pytest.raises(Refused) as error:
            install(guard, add, _heard(hooks))
        assert error.value.code == "task_control.plugin_guard_unavailable"
    else:
        install(guard, add, _heard(hooks))
        assert guard.active


RESULT = b'{"jsonrpc": "2.0", "id": 1, "result": {"wall_ms": 9, "peak_bytes": 36}}\n'
CAUGHT = (
    b'{"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "m", "data": {"code": '
    b'"task_control.plugin_network_refused", "detail": "socket.connect", "refusals": [{"code": '
    b'"task_control.plugin_network_refused", "event": "socket.connect", "detail": "x"}]}}}\n'
)
UNANSWERED = "task_control.plugin_child_unanswered"


@pytest.mark.proves("task_control.a_guarded_childs_result_counts_only_from_its_clean_exit")
@pytest.mark.parametrize(
    ("returncode", "frame", "code"),
    [
        pytest.param(0, RESULT, None, id="result-from-a-clean-exit"),
        pytest.param(1, RESULT, UNANSWERED, id="result-from-a-failed-exit"),
        pytest.param(0, CAUGHT, "task_control.plugin_network_refused", id="named-refusal"),
        pytest.param(
            0, CAUGHT.replace(b'network_refused"', b'unregistered"', 1), UNANSWERED, id="unknown"
        ),
        pytest.param(
            0, RESULT[:-2] + b" " * (1 << 16) + b"}\n", UNANSWERED, id="answer-past-its-bound"
        ),
        pytest.param(0, b"plugin print\n" + RESULT, UNANSWERED, id="two-lines"),
        pytest.param(0, b"", UNANSWERED, id="no-answer"),
    ],
)
def test_the_host_accepts_only_a_clean_childs_one_bounded_answer(
    returncode: int, frame: bytes, code: str | None
) -> None:
    """A result counts only from a child that exited 0 with one bounded answer."""
    if code is None:
        assert child_answer(returncode, frame) == {"wall_ms": 9, "peak_bytes": 36}
        return
    with pytest.raises(StagingRefused) as error:
        child_answer(returncode, frame)
    assert error.value.code == code
    assert "plugin print" not in str(error.value)
    if code != UNANSWERED:
        assert error.value.refusals[0]["event"] == "socket.connect"


@pytest.mark.proves("task_control.a_held_plugin_run_signals_started_only_from_its_childs_mark")
@pytest.mark.parametrize(
    "marked",
    [
        pytest.param(False, id="no-mark-as-in-a-launch-bootstrap"),
        pytest.param(True, id="its-guarded-child-marked"),
    ],
)
def test_a_held_run_signals_started_only_once_its_guarded_child_has_marked(
    tmp_path: Path, marked: bool
) -> None:
    """A held run's started signal comes only from its guarded child's mark, never at launch."""
    mark, signals = tmp_path / STARTED, list[bool]()
    if marked:
        mark.touch()

    signal_started(mark, lambda: signals.append(mark.exists()), threading.Event(), 0.1)

    assert signals == ([True] if marked else [])
