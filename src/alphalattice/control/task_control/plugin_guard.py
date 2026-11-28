"""The guarded child a user's or an agent's plugin runs in (`python -I -B -m` this module).

Its parent (`plugin_child.GuardedPlugin`) writes one JSON-RPC request to its standard input: the
plugin's entry and verified source bytes, its parameters, its staging folder and its lease. The
child answers once on its standard output, which it first moves aside so the plugin's own prints
go to its error stream. Data stays in files: the plugin reads the read-only data root and writes
Parquet beside its answer in the staging folder.

Before the plugin's code loads, the child confines itself with what its system offers without
administrator rights, then installs a Python audit hook (PEP 578). On Windows it labels its staging
folder Low and lowers its own token to Low integrity, so the system refuses a write anywhere else;
its parent's Job holds one process. On Linux, Landlock confines writes to the staging folder (and
TCP from kernel 6.7) where the kernel has it, and the parent's cgroup scope holds memory. Every
POSIX system bounds its CPU seconds and open files; on macOS those limits, the hook, the
redirected folders, the parent's process group and its polled resident memory are all there is.

The hook refuses, by name and by CPython's own audit event names: a network connect or lookup,
unless the lease grants network; a subprocess or a fork; and a write, rename or delete of a path
that does not resolve inside the staging folder, each path placed by its own directory
descriptor, a link counting as outside. The child's first finder refuses every module, a
submodule included, whose resolved origin lies outside the environment the child started with
and the plugin's own sources; the hook refuses any import once that finder is no longer first.
Every other event passes: it stops a wrong plugin, not a hostile one. A refusal is recorded even
when the plugin catches it, and the answer then names the first. The plugin starts only once the
hook has heard the child's probe event, so a registration an earlier hook refused, loudly or
silently, refuses the start. In a run its parent will cancel (`hold`), the child makes `STARTED`
in its home once its hook has heard that probe, so its parent's started signal comes from this
guarded interpreter and never from a launch bootstrap; it then runs the plugin and holds, alive,
until the cancel ends its tree or its lease's timeout passes.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any

from alphalattice.protocols.plugin_contract import ExecutionContext

HOME = ".plugin-home"
"""The folder in the staging folder that the child's working directory and home point into; its
parent removes it before the staged files are checked."""

STARTED = "started"
"""The file a held child makes in its home once its hook has heard its probe: its started
signal."""

REDIRECTED = ("HOME", "USERPROFILE", "TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME")

NETWORK = frozenset(
    {
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
    }
)
SUBPROCESS = frozenset(
    {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"}
    | {"os.fork", "os.forkpty"}  # os.spawn* forks on POSIX, raising os.fork, not os.spawn
)
WRITES: Mapping[str, tuple[tuple[int, int | None], ...]] = {
    # event: each path's argument position and the position of its own dir_fd
    "open": ((0, None),),
    "os.rename": ((0, 2), (1, 3)),  # os.replace raises it too
    "os.remove": ((0, 1),),  # os.unlink raises it too
    "os.rmdir": ((0, 1),),
    "os.mkdir": ((0, 2),),
    "shutil.rmtree": ((0, 1),),
}
PROBE = "alphalattice.plugin_guard.probe"
"""The event the child raises once its hook is added, so a registration an earlier hook
suppressed is caught before the plugin loads."""
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class Refused(Exception):
    """A guarded run the child does not finish, by its registered code."""

    def __init__(self, code: str, detail: str, refusals: Sequence[Mapping[str, str]] = ()) -> None:
        """Name the refusal, what met it and every refusal the hook recorded."""
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.refusals = code, detail, tuple(refusals)


class Guard:
    """Classify one audit event against a staging folder, a lease and the environment's
    folders."""

    def __init__(
        self,
        staging: str,
        *,
        network: bool,
        environment: frozenset[str],
    ) -> None:
        """Bind the real staging folder, whether the lease grants network and the environment's
        folders (`environment_folders`)."""
        self.root = os.path.normcase(os.path.realpath(staging))
        self._network = network
        self._roots = frozenset(os.path.normcase(os.path.realpath(root)) for root in environment)
        self._busy = threading.local()
        self.finder: object = None
        self.active = False
        self.refusals: list[dict[str, str]] = []

    def refusal(self, event: str, args: tuple[Any, ...]) -> tuple[str, str] | None:
        """The registered code and the detail for an event the guard refuses, else None."""
        if event in NETWORK:
            return None if self._network else ("task_control.plugin_network_refused", event)
        if event in SUBPROCESS:
            return "task_control.plugin_subprocess_refused", event
        if event == "import" and self.finder is not None and sys.meta_path[:1] != [self.finder]:
            return "task_control.plugin_import_outside_environment", "the import system changed"
        if event in WRITES:
            if event == "open" and not _writes(args):
                return None
            for position, descriptor in WRITES[event]:
                directory = None if descriptor is None else args[descriptor]
                if not self._inside(args[position], directory):
                    return "task_control.plugin_write_outside_staging", f"{event} {args[position]}"
        return None

    def module_refusal(
        self, name: str, spec: importlib.machinery.ModuleSpec | None
    ) -> tuple[str, str] | None:
        """Refuse a module whose spec would load it from outside the environment, every spec
        judged by its own origin on every resolution, a submodule's included, never by a name
        approved before; one found nowhere is left to fail as Python fails it, so an optional
        import a library tries is no refusal. The plugin's own modules never reach it: the
        child's first finder serves them."""
        if spec is None:
            return None
        if spec.origin is None or spec.origin in ("built-in", "frozen"):
            places = list(spec.submodule_search_locations or ())
        else:
            places = [spec.origin]
        if all(self._held(place) for place in places):
            return None
        return "task_control.plugin_import_outside_environment", name

    def record(self, code: str, event: str, detail: str) -> None:
        """Keep a refusal, even one the plugin goes on to catch."""
        if len(self.refusals) < 32:
            self.refusals.append({"code": code, "event": event, "detail": detail})

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        """The audit hook: record and raise a refusal; re-entered events pass."""
        if event == PROBE:
            self.active = True
            return
        if getattr(self._busy, "active", False):
            return
        self._busy.active = True
        try:
            found = self.refusal(event, args)
        finally:
            self._busy.active = False
        if found is not None:
            code, detail = found
            self.record(code, event, detail)
            # an import is refused as one that finds nothing would be; the record names it
            refusal = ModuleNotFoundError if event == "import" else PermissionError
            raise refusal(f"{code}: {detail}")

    def _inside(self, path: object, directory: object) -> bool:
        if isinstance(path, int):
            return True  # a descriptor already open; its opening was checked
        if not isinstance(path, str | bytes | os.PathLike):
            return False
        text = os.fsdecode(path)
        if directory not in (None, -1) and not os.path.isabs(text):
            return False  # relative to a directory descriptor the guard cannot place
        absolute = os.path.normcase(os.path.abspath(text))
        real = os.path.normcase(os.path.realpath(absolute))
        return real == absolute and (real == self.root or real.startswith(self.root + os.sep))

    def _held(self, place: str) -> bool:
        real = os.path.normcase(os.path.realpath(place))
        return any(real.startswith(root + os.sep) for root in self._roots)


def environment_folders() -> frozenset[str]:
    """The folders the interpreter loads from now, the environment a guarded plugin imports
    from: its import path and each loaded top-level module's folder, an editable install's
    among them."""
    places = [entry for entry in sys.path if entry]
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if "." not in name and isinstance(file, str):
            folder = os.path.dirname(file)
            places.append(os.path.dirname(folder) if hasattr(module, "__path__") else folder)
    return frozenset(places)


def _writes(args: tuple[Any, ...]) -> bool:
    _, mode, flags = args
    if isinstance(mode, str) and any(sign in mode for sign in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & _WRITE_FLAGS)


class _Sources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """The child's first finder: it serves the plugin's verified source bytes as its modules,
    never re-reading its folder, and finds every other module through the finders after it,
    refusing by name one the guard finds would load from outside the environment."""

    def __init__(self, sources: Mapping[str, bytes]) -> None:
        self.guard: Guard | None = None
        self._modules: dict[str, tuple[str, bytes, bool]] = {}
        for relative, body in sources.items():
            if not relative.endswith(".py"):
                continue
            parts = relative[:-3].split("/")
            package = parts[-1] == "__init__"
            name = ".".join(parts[:-1] if package else parts)
            self._modules[name] = (relative, body, package)

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname in self._modules:
            relative, _, package = self._modules[fullname]
            return importlib.util.spec_from_loader(
                fullname, self, origin=f"plugin:{relative}", is_package=package
            )
        spec = None
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            find = getattr(finder, "find_spec", None)
            spec = None if find is None else find(fullname, path, target)
            if spec is not None:
                break
        if self.guard is not None and (found := self.guard.module_refusal(fullname, spec)):
            code, detail = found
            self.guard.record(code, "import", detail)
            raise ModuleNotFoundError(f"{code}: {detail}")
        return spec

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        relative, body, package = self._modules[module.__name__]
        if package:
            module.__path__ = []
        exec(compile(body, f"plugin:{relative}", "exec"), module.__dict__)


def confine(staging: str, lease: Mapping[str, Any]) -> str:
    """Set this system's confinement for the child before the plugin loads; the layer set."""
    try:
        if sys.platform == "win32":
            label_low(staging)
            _lower_to_low_integrity()
            return "low-integrity"
        _bound(lease)
        if sys.platform == "linux":
            return "landlock" if _landlock(staging, network=bool(lease["network"])) else "rlimits"
        return "rlimits"
    except OSError as error:
        raise Refused("task_control.plugin_confinement_failed", str(error)) from error


def _bound(lease: Mapping[str, Any]) -> None:
    if sys.platform == "win32":
        return
    import resource  # lazy: POSIX only

    seconds = math.ceil(lease["timeout_ms"] / 1000 * lease["threads"]) + 1
    for limit, value in ((resource.RLIMIT_CPU, seconds), (resource.RLIMIT_NOFILE, 1024)):
        _, hard = resource.getrlimit(limit)
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(limit, (value, value))


def _landlock(staging: str, *, network: bool) -> bool:
    """Confine this process's writes to `staging`, and its TCP unless the lease grants network;
    False where the kernel offers no Landlock."""
    if sys.platform != "linux":
        return False

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    create, add_rule, restrict = 444, 445, 446  # the generic syscall table's numbers
    abi = libc.syscall(ctypes.c_long(create), None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 0:
        if ctypes.get_errno() in (errno.ENOSYS, errno.EOPNOTSUPP):
            return False
        raise OSError(ctypes.get_errno(), "landlock version unreadable")
    writes = sum(1 << bit for bit in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12))  # write, remove, make
    writes |= (1 << 13 if abi >= 2 else 0) | (1 << 14 if abi >= 3 else 0)  # refer, truncate

    class Ruleset(ctypes.Structure):
        _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed", ctypes.c_uint64), ("parent", ctypes.c_int32)]

    ruleset = Ruleset(writes, 0 if network or abi < 4 else 0b11)  # bind and connect TCP
    size = ctypes.sizeof(ruleset) if abi >= 4 else 8
    handle = libc.syscall(ctypes.c_long(create), ctypes.byref(ruleset), ctypes.c_size_t(size), 0)
    if handle < 0:
        raise OSError(ctypes.get_errno(), "landlock ruleset refused")
    parent = os.open(staging, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = Beneath(writes, parent)
        if libc.syscall(ctypes.c_long(add_rule), handle, 1, ctypes.byref(rule), 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock rule refused")
        if libc.prctl(38, 1, 0, 0, 0) < 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "no_new_privs refused")
        if libc.syscall(ctypes.c_long(restrict), handle, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock restriction refused")
    finally:
        os.close(parent)
        os.close(handle)
    return True


def _windows_security() -> Any:

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.SetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi.SetNamedSecurityInfoW.restype = ctypes.c_uint32
    return advapi


def _set_label(path: str, sacl: object) -> None:
    advapi = _windows_security()
    # SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION
    result = advapi.SetNamedSecurityInfoW(path, 1, 0x10, None, None, None, sacl)
    if result:
        raise ctypes.WinError(result)


def label_low(path: str) -> None:
    """Label a folder Low, inherited by everything made in it, so a Low process may write there."""
    advapi = _windows_security()
    descriptor, sacl = ctypes.c_void_p(), ctypes.c_void_p()
    present, defaulted = ctypes.c_int(), ctypes.c_int()
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "S:(ML;OICI;NW;;;LW)", 1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not advapi.GetSecurityDescriptorSacl(
            descriptor, ctypes.byref(present), ctypes.byref(sacl), ctypes.byref(defaulted)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        _set_label(path, sacl)
    finally:
        ctypes.windll.kernel32.LocalFree(descriptor)


def clear_labels(path: str) -> None:
    """Remove the integrity label from a folder and every entry in it, links not followed, so
    nothing a Low child left stays writable at Low once it is committed."""
    advapi = _windows_security()
    empty = (ctypes.c_byte * 8)()
    if not advapi.InitializeAcl(empty, ctypes.sizeof(empty), 2):  # ACL_REVISION
        raise ctypes.WinError(ctypes.get_last_error())
    _set_label(path, empty)
    for folder, names, files in os.walk(path, onerror=_raise):
        for name in names + files:
            entry = os.path.join(folder, name)
            if not os.lstat(entry).st_file_attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
                _set_label(entry, empty)  # a link's label call would reach its target


def _raise(error: OSError) -> None:
    raise error


def _lower_to_low_integrity() -> None:

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    advapi.OpenProcessToken.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.ConvertStringSidToSidW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    advapi.SetTokenInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    advapi.GetLengthSid.argtypes = [ctypes.c_void_p]

    class Label(ctypes.Structure):
        _fields_ = [("sid", ctypes.c_void_p), ("attributes", ctypes.c_uint32)]

    token, sid = ctypes.c_void_p(), ctypes.c_void_p()
    # TOKEN_ADJUST_DEFAULT | TOKEN_QUERY
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x88, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not advapi.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(sid)):  # Low
            raise ctypes.WinError(ctypes.get_last_error())
        label = Label(sid, 0x20)  # SE_GROUP_INTEGRITY
        size = ctypes.sizeof(label) + advapi.GetLengthSid(sid)
        if not advapi.SetTokenInformation(token, 25, ctypes.byref(label), size):
            raise ctypes.WinError(ctypes.get_last_error())  # TokenIntegrityLevel
    finally:
        kernel.LocalFree(sid)
        kernel.CloseHandle(token)


def _peak_bytes() -> int:
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("rest", ctypes.c_size_t * 7),
            ]

        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.K32GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        counters = Counters(cb=ctypes.sizeof(Counters))
        if not kernel.K32GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError()
        return int(counters.PeakWorkingSetSize)
    import resource  # lazy: POSIX only

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def install(
    guard: Guard,
    add: Callable[[Callable[[str, tuple[Any, ...]], None]], None] = sys.addaudithook,
    audit: Callable[[str], None] = sys.audit,
) -> None:
    """Add the guard as an audit hook and prove it hears an event, or refuse the start: an
    earlier hook may refuse the registration, CPython may then add nothing silently, and an
    earlier hook may raise on the probe itself."""
    try:
        add(guard)
        audit(PROBE)
    except Exception as error:
        raise Refused("task_control.plugin_guard_unavailable", str(error)) from error
    if not guard.active:
        raise Refused("task_control.plugin_guard_unavailable", "the audit hook heard nothing")


def serve(params: Mapping[str, Any]) -> dict[str, int]:
    """Confine, guard, load and run the plugin; its answer is staged as `output.json`."""
    staging = os.path.realpath(params["staging_dir"])
    lease = params["context"]["lease"]
    confine(staging, lease)
    sources = _Sources({path: base64.b64decode(body) for path, body in params["sources"].items()})
    guard = Guard(
        staging,
        network=bool(lease["network"]),
        environment=environment_folders(),
    )
    sources.guard, guard.finder = guard, sources
    sys.meta_path.insert(0, sources)
    install(guard)
    hold = bool(params.get("hold"))
    if hold:
        open(os.path.join(staging, HOME, STARTED), "x").close()  # a mark, no data
    started = time.perf_counter()
    try:
        context = ExecutionContext.model_validate_json(
            json.dumps({**params["context"], "staging_dir": staging})
        )
        module, _, function = params["entry"].partition(":")
        output = getattr(importlib.import_module(module), function)(params["params"], context)
        if not guard.refusals:
            with open(os.path.join(staging, "output.json"), "x", encoding="utf-8") as answer:
                json.dump(output, answer, sort_keys=True, allow_nan=False)
    except Exception as error:
        if not guard.refusals:
            raise Refused(
                "task_control.plugin_failed", f"{type(error).__name__}: {error}"
            ) from error
    if hold:  # alive, its plugin run, until its parent's cancel ends the tree
        threading.Event().wait(lease["timeout_ms"] / 1000)
    if guard.refusals:
        first = guard.refusals[0]
        raise Refused(first["code"], first["detail"], guard.refusals)
    return {"wall_ms": round((time.perf_counter() - started) * 1000), "peak_bytes": _peak_bytes()}


def main() -> int:
    """Answer the one request on standard input on the standard output this child set aside."""
    request = json.loads(sys.stdin.buffer.readline())
    answer = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": request.get("id")}
    try:
        if request.get("jsonrpc") != "2.0" or request.get("method") != "plugin.run":
            raise Refused("task_control.plugin_child_unanswered", "not a plugin.run request")
        response["result"] = serve(request["params"])
    except Refused as refusal:
        response["error"] = {
            "code": -32000,
            "message": str(refusal),
            "data": {
                "code": refusal.code,
                "detail": refusal.detail,
                "refusals": list(refusal.refusals),
            },
        }
    answer.write(json.dumps(response) + "\n")
    answer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
