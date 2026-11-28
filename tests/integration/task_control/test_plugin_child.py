"""A user's or an agent's plugin in its own guarded child, through its staging run.

integration: the audit hook and each system's confinement (Low integrity and the Job, Landlock,
rlimits) are the child process's own state, which the test process cannot install without
changing itself for good; the hook's decisions are the staging holder's unit table.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pytest

from alphalattice.control.task_control.plugin_child import GuardedPlugin
from alphalattice.control.task_control.plugin_guard import REDIRECTED
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
)

TEST_SOURCE = "src/alphalattice/control/task_control/plugin_child.py"

READ = CatalogRead(snapshot_id="lake:42", input_versions={"raw_bar": 7})
SPEC = ActionSpec(
    capability_id="factor.custom_momentum",
    plugin_id="sample-factor",
    code_role_hash="blake3:" + "1" * 64,
    effect_mode=EffectMode.STAGING_WRITE,
    execution_mode=ExecutionMode.SANDBOX_SUBPROCESS,
    lease=Lease(memory_bytes=1 << 30, threads=1, timeout_ms=60_000),
    input_schema={"type": "object"},
    output_schema={"type": "object"},
    inputs=("raw_bar",),
)

# A plugin that stages Parquet and reports where its folders point and whether the system
# stopped its native code writing or starting a process outside its staging folder.
CONFINED = r"""
import ctypes, errno, os, sys
import pyarrow as pa
import pyarrow.parquet as pq
from helpers.names import REDIRECTED

def native_escapes(outside):
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.restype = ctypes.c_void_p
        handle = kernel.CreateFileW(os.path.join(outside, "native.txt"), 0x40000000, 0, None,
                                    2, 0x80, None)
        wrote = handle not in (None, ctypes.c_void_p(-1).value)

        class Startup(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("rest", ctypes.c_byte * 100)]

        class Started(ctypes.Structure):
            _fields_ = [("process", ctypes.c_void_p), ("thread", ctypes.c_void_p),
                        ("pid", ctypes.c_uint32), ("tid", ctypes.c_uint32)]

        started = Started()
        command = ctypes.create_unicode_buffer("cmd.exe /c exit 0")
        spawned = bool(kernel.CreateProcessW(None, command, None, None, False, 0x08000000,
                                             None, None, ctypes.byref(Startup(cb=104)),
                                             ctypes.byref(started)))
        return {"native_write": wrote, "native_process": spawned}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if sys.platform != "linux" or libc.syscall(444, None, ctypes.c_size_t(0), 1) < 0:
        return {}  # no Landlock: the hook and the redirection are the guard
    descriptor = libc.open(os.path.join(outside, "native.txt").encode(), 0o101, 0o644)
    return {"native_write": descriptor >= 0}

def run(params, context):
    staging = str(context.staging_dir)
    with open("scratch.txt", "w") as scratch:
        scratch.write("kept out of the commit")
    table = pa.table({"score": [1.0, 2.0]})
    pq.write_table(table, context.staging_dir / "data.parquet")
    inside = [os.environ[name] for name in REDIRECTED] + [os.getcwd()]
    return {
        "redirected": all(os.path.realpath(place).startswith(staging) for place in inside),
        "window": params["window"],
        **native_escapes(params["outside"]),
    }
"""

# A plugin that reaches for each refused form, catching each refusal as a wrong plugin might.
WRONG = r"""
import os, shutil, socket, subprocess, sys

def attempt(call):
    try:
        call()
    except Exception:
        pass

def run(params, context):
    outside, staging = params["outside"], context.staging_dir
    attempt(lambda: socket.socket().connect(("127.0.0.1", 9)))
    attempt(lambda: socket.getaddrinfo("example.com", 80))
    attempt(lambda: socket.gethostbyname("example.com"))
    attempt(lambda: socket.gethostbyaddr("127.0.0.1"))
    attempt(lambda: socket.getnameinfo(("127.0.0.1", 80), 0))
    attempt(lambda: subprocess.run([sys.executable, "-c", "pass"], timeout=5))
    attempt(lambda: os.system("exit 0"))
    attempt(lambda: os.execv(sys.executable, [sys.executable, "-c", "pass"]))
    attempt(lambda: os.spawnv(os.P_WAIT, sys.executable, [sys.executable, "-c", "pass"]))
    if hasattr(os, "posix_spawn"):
        attempt(lambda: os.posix_spawn(sys.executable, [sys.executable, "-c", "pass"], {}))
    attempt(lambda: open(os.path.join(outside, "escaped.txt"), "w"))
    attempt(lambda: os.rename(staging / "a", os.path.join(outside, "b")))
    attempt(lambda: os.remove(os.path.join(outside, "kept.txt")))
    attempt(lambda: os.rmdir(os.path.join(outside, "folder")))
    attempt(lambda: os.mkdir(os.path.join(outside, "made")))
    attempt(lambda: shutil.rmtree(outside))
    attempt(lambda: open(staging / "link" / "through.txt", "w"))
    (staging / "mine").mkdir()
    (staging / "mine" / "planted.py").write_text("X = 1")
    sys.path.append(str(staging / "mine"))
    attempt(lambda: __import__("planted"))
    return {"done": True}
"""


@dataclass(frozen=True)
class _LinkedStaging(GuardedPlugin):
    """A run whose staging folder holds a link out of it, planted before its child starts (a
    directory symlink: Windows makes one in developer mode or with the right, as CI has)."""

    outside: Path = Path()

    def __call__(self, context: ExecutionContext) -> ActionReceipt:
        """Plant the link, then run the plugin in its guarded child."""
        (context.staging_dir / "link").symlink_to(self.outside, target_is_directory=True)
        return super().__call__(context)


def _labelled(path: Path) -> bool:
    """Whether a Windows path carries a mandatory integrity label."""
    advapi = ctypes.WinDLL("advapi32")
    sacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
    # SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION
    assert not advapi.GetNamedSecurityInfoW(
        str(path), 1, 0x10, None, None, None, ctypes.byref(sacl), ctypes.byref(descriptor)
    )
    try:  # an ACL's header: revision, padding, size, then its entry count
        return bool(sacl.value) and ctypes.cast(sacl, ctypes.POINTER(ctypes.c_uint16))[2] > 0
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)


@pytest.mark.proves("task_control.a_guarded_plugin_runs_confined_in_its_own_child")
def test_a_plugin_runs_confined_in_its_child_and_each_wrong_reach_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A plugin runs confined in its own child, and each wrong reach is refused by name."""
    workspace, artifacts, outside = tmp_path / "workspace", tmp_path / "artifacts", tmp_path / "x"
    workspace.mkdir()
    outside.mkdir()
    (outside / "kept.txt").write_text("kept", encoding="utf-8")
    (outside / "folder").mkdir()

    def catalog(tables: Collection[str], snapshot: str | None) -> CatalogRead:
        return READ

    staging = PluginStaging(
        workspace, tmp_path / "data", ArtifactResolver(artifacts), WorkspaceMutationGate(), catalog
    )
    helper = f"REDIRECTED = {REDIRECTED!r}\n".encode()
    confined = SPEC.invoke({"window": 20, "outside": str(outside)}, READ)
    committed = staging.request(
        SPEC,
        confined,
        GuardedPlugin(
            "probe:run",
            {"probe.py": CONFINED.encode(), "helpers/__init__.py": b"", "helpers/names.py": helper},
            confined,
        ),
    ).result()
    folder = next((artifacts / "plugin-result").iterdir())

    wrong = SPEC.invoke({"window": 21, "outside": str(outside)}, READ)
    with pytest.raises(StagingRefused) as refused:
        staging.request(
            SPEC, wrong, _LinkedStaging("wrong:run", {"wrong.py": WRONG.encode()}, wrong, outside)
        ).result()
    events = [refusal["event"] for refusal in refused.value.refusals]

    output = committed.output
    assert isinstance(output, dict)
    expected_native = {
        "win32": {"native_write": False, "native_process": False},
        "linux": {"native_write": False} if "native_write" in output else {},
    }.get(sys.platform, {})
    assert output == {"redirected": True, "window": 20, **expected_native}
    assert sorted(path.name for path in folder.iterdir()) == ["data.parquet", OUTPUT, PROVENANCE]
    if sys.platform == "win32":
        assert not any(_labelled(path) for path in [folder, *folder.iterdir()])
    assert refused.value.code == "task_control.plugin_network_refused"
    spawns = ["os.posix_spawn"] if hasattr(os, "posix_spawn") else []
    assert events == [
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.spawn" if sys.platform == "win32" else "os.fork",  # POSIX spawns by forking
        *spawns,
        "open",
        "os.rename",
        "os.remove",
        "os.rmdir",
        "os.mkdir",
        "shutil.rmtree",
        "open",
        "import",
    ]
    assert refused.value.refusals[-1]["detail"] == "planted"
    assert sorted(path.name for path in outside.iterdir()) == ["folder", "kept.txt"]
    assert len(list((artifacts / "plugin-result").iterdir())) == 1
    assert not any((workspace / "staging" / "plugin-runs").iterdir())
