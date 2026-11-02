import functools
import json
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path

import pytest

from devtools.pipeline import cache

GIT, KEY = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C"], "a" * 64
LINK = __import__("_winapi").CreateJunction if os.name == "nt" else os.symlink  # unprivileged
WRITER = {"line": "a line", "head": "0" * 40, "invocation": "i", "at": "t"}
READ = {"probes": {"src/a.py": "file"}, "env": {"X": cache.digest("1")}}
READ |= {"results": {"t::x": ("PASSED", 0.1)}, "writer": WRITER}
SPOILS = {"a flipped byte": lambda raw: raw.replace("0.1", "0.2"), "empty": lambda raw: ""}
SPOILS |= {"truncated": lambda raw: raw[:-9], "a directory": None, "schema v2": {"schema": "v2"}}
SPOILS |= {"another key": {"key": "c" * 64}, "a junction out": {"probes": {"out/f": "file"}}}
SPOILS |= {path: {"probes": {path: None}} for path in ("../x", "C:/x", "Src/a.py")}
SPOILS |= {"a writer without its time": {"writer": {k: v for k, v in WRITER.items() if k != "at"}}}
CASES = {"nothing": True, "src/a.py": True, "src/b.py": False, "X": False, "src/c.py": False}
CASES |= {"the breaker after open": False, "src through a junction": False}
CASES |= {"an absent probe": True, "a denied stat": False, "a linked shard": False}
STORES = {"disabled": "result_cache.disabled", "a file": "result_cache.unwritable"}
STORES |= {"a hung git": "result_cache.git_timeout", "bounds": None}
FAULTS = {"a shard that is a file": None, "a held entry": (cache.os, "replace", PermissionError)}
FAULTS |= {"a vanished temporary": (cache.os, "replace", FileNotFoundError)}
FAULTS |= {
    f"a refused {name}": (cache.Path, name, PermissionError) for name in ("write_text", "exists")
}


def _deny(stat, path, *args, **kwargs):
    if "gone" in str(path):
        raise PermissionError(path)
    return stat(path, *args, **kwargs)


def _raw(**fields):
    body = {"schema": cache.SCHEMA, "key": KEY, "files": {}, "listings": {}, **READ, **fields}
    return json.dumps({**body, "digest": cache.digest(body)})


@pytest.mark.proves("pipeline.result_cache_hits_only_on_unchanged_inputs")
@pytest.mark.proves("pipeline.result_cache_never_trusts_a_bad_entry")
@pytest.mark.parametrize("case", CASES | dict.fromkeys(SPOILS, False))
def test_an_entry_hits_only_while_it_and_all_it_read_are_unchanged(tmp_path, monkeypatch, case):
    root, store, spoil = tmp_path / "repo", cache.Store.open(tmp_path / "store"), SPOILS.get(case)
    for path in (root / "src/a.py", root / "src/b.py", tmp_path / "away/f"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, "utf-8")
    for words in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run([*GIT, root, *words], check=True)
    LINK(str(tmp_path / "away"), str(root / "out"))
    monkeypatch.setenv("X", "1")
    files = {"src/b.py": cache.Tree.read(root).blobs["src/b.py"]}
    good = {"key": KEY, "files": files, "listings": {"src": cache.digest(["a.py", "b.py"])}, **READ}
    store.write(cache.Entry.sealed(**good))
    assert store.lookup(KEY, cache.Tree.read(root)) is not None
    if case == "X":
        monkeypatch.setenv("X", "2")
    elif case.startswith("src/"):
        (root / case).write_text("new", "utf-8")
    elif case == "the breaker after open":
        (tmp_path / "store" / "disabled").touch()
    elif case == "src through a junction":  # an entry of files alone, its folder now elsewhere
        store.path(KEY).write_text(_raw(**{**good, "probes": {}, "listings": {}}), "utf-8")
        (root / "src").rename(tmp_path / "moved")
        LINK(str(tmp_path / "moved"), str(root / "src"))
    elif case in ("an absent probe", "a denied stat"):  # absence is known; a denied stat is not
        store.path(KEY).write_text(_raw(**{**good, "probes": {"src/gone.py": None}}), "utf-8")
        if case == "a denied stat":
            monkeypatch.setattr(cache.os, "stat", functools.partial(_deny, os.stat))
    elif case == "a linked shard":  # never read, not merely refused once read
        (shard := store.path(KEY).parent).rename(tmp_path / "shard")
        LINK(str(tmp_path / "shard"), str(shard))
        read, reads = cache.Path.read_bytes, []
        monkeypatch.setattr(cache.Path, "read_bytes", lambda p: reads.append(p) or read(p))
    elif case in SPOILS:
        raw = (path := store.path(KEY)).read_text("utf-8")
        path.unlink()
        if spoil is None:
            path.mkdir()
        else:
            path.write_text(spoil(raw) if callable(spoil) else _raw(**{**good, **spoil}), "utf-8")
    folds = (root / "SRC").exists()  # case-sensitive, Src/a.py is just another absent path
    expected = CASES.get(case, case == "Src/a.py" and not folds)
    assert (store.lookup(KEY, cache.Tree.read(root)) is not None) is expected
    assert case != "a linked shard" or not reads
    assert case != "nothing" or cache.action_key(lock="1") != cache.action_key(lock="2")


@pytest.mark.proves("pipeline.result_cache_never_trusts_a_bad_entry")
@pytest.mark.proves("pipeline.result_cache_store_stays_within_its_bounds")
@pytest.mark.parametrize("case", STORES | FAULTS)
def test_an_unusable_store_is_off_and_a_usable_one_keeps_its_bounds(tmp_path, monkeypatch, case):
    (root := tmp_path / "store").mkdir()
    if case == "disabled":
        (root / "disabled").touch()
    elif case == "a file":
        root.rmdir()
        root.touch()
    timeout = subprocess.TimeoutExpired("git", cache.GIT_SECONDS)
    monkeypatch.setattr(cache.subprocess, "run", lambda *_, **__: (_ for _ in ()).throw(timeout))
    cap = 2 * len(_raw())  # two entries' bytes
    store = cache.Tree.read(tmp_path) if case == "a hung git" else cache.Store.open(root, cap)
    if STORES.get(case):
        assert store == STORES[case]
        return
    if case in FAULTS:  # a write the store cannot take is skipped, never raised into the stage
        if FAULTS[case] is None:
            (root / "ac" / KEY[:2]).write_text("", "utf-8")
        else:
            target, name, error = FAULTS[case]
            monkeypatch.setattr(target, name, lambda *_, **__: (_ for _ in ()).throw(error()))
        store.write(cache.Entry.model_validate_json(_raw()))
        assert not list(root.glob("ac/*/.*.tmp")) and store.lookup(KEY, None) is None
        return
    (away := tmp_path / "away").mkdir()
    (sentinel := away / "old").write_text("outside the store", "utf-8")
    os.utime(sentinel, (0, 0))
    LINK(str(away), str(root / "ac" / "ff"))  # a shard that leads outside the store
    for key, age in {"1": 15 * 86400, "2": 30, "3": 20, "4": 10}.items():  # 1 is idle
        (path := store.path(key * 64)).parent.mkdir()
        path.write_text(_raw(key=key * 64), "utf-8")
        os.utime(path, (time.time() - age, time.time() - age))
    store.evict()
    assert sorted(path.name[0] for path in root.glob("ac/[0-9]*/*")) == ["3", "4"]
    assert sentinel.exists()
    (root / "ac").rename(tmp_path / "ac")  # then ac itself leads outside the store
    LINK(str(away), str(root / "ac"))
    store.write(cache.Entry.model_validate_json(_raw()))
    store.evict(time.time() + 30 * 86400)
    assert sentinel.exists() and sorted(path.name for path in away.iterdir()) == ["old"]
    assert 0 < cache.Store.open(tmp_path / "default").cap <= 4 << 30


OBSERVED = {
    "contents": ("p.read_text(); stale=os.listdir", None),
    "outside": ("outside.read_text()", "undeclared_input"),
    "nested": ("with Recorder(tree,temp,()) as inner: p.read_text()", "undeclared_input"),
    "reentrant": ("with seen: p.read_text()", "undeclared_input"),
    "audit-shape": ("sys.audit('import',42,None)", "undeclared_input"),
    "failed-list": ("os.listdir(temp.parent/'absent')", "undeclared_input"),
    "foreign-list": ("os.listdir(temp)", "undeclared_input"),
    "locked": ("(Path(sys.prefix)/'pyvenv.cfg').read_text()", None),
    "foreign-temp": ("(temp/'foreign').read_text()", "undeclared_input"),
    "untracked": ("(root/'loose').read_text()", "undeclared_input"),
    "write": ("p.write_text('new')", "undeclared_input"),
    "own-temp": ("x=temp/'mine'; x.write_text('made'); x.read_text(); os.listdir(path=root)", None),
    "child": (
        "subprocess.run([sys.executable, '-c', 'pass'], creationflags=flags)",
        "undeclared_input",
    ),
    "socket": ("with socket.socket() as sock: sock.connect(('127.0.0.1', 0))", "undeclared_input"),
    "native": ("import duckdb", "undeclared_input"),
    "scandir": ("list(os.scandir(root))", "metadata_read"),
    "unobserved": ("p.read_text()", "undeclared_input"),
    "borrowed-list": ("saved[3](root)", None),
    "bytecode-import": ("sys.path.insert(0,str(root)); import b", None),
    "bytecode-loose": ("(root/'__pycache__'/'loose.pyc').read_bytes()", "undeclared_input"),
    "bytecode-direct": (
        "Path(cache_from_source(str(root/'b.py'))).read_bytes()",
        "undeclared_input",
    ),
    "bytecode-write": (
        "Path(cache_from_source(str(root/'b.py'))).write_bytes(b'x')",
        "undeclared_input",
    ),
    "bytecode-temp": ("(temp/'made.pyc').write_bytes(b'x')", "undeclared_input"),
}
OBSERVED |= {
    "stat-metadata": (
        "value=os.stat(p); assert type(value) is os.stat_result and value==native_stat",
        "metadata_read",
    ),
    "lstat-metadata": (
        "value=os.lstat(p); assert type(value) is os.stat_result and value==native_lstat",
        "metadata_read",
    ),
    "metadata-missing": ("os.stat(root/'absent')", "metadata_read"),
    "path-probe": ("assert p.exists() and p.is_file() and not p.is_dir()", None),
    "fast-probes": (
        "for f,v in ((os.path.exists,True),(os.path.isfile,True),(os.path.isdir,False)):\n"
        " n=stats.call_count; assert f(p) is v and stats.call_count==n+1",
        None,
    ),
    "lexists-probe": ("assert os.path.lexists(p)", None),
    "absent-probe": ("assert not (root/'later').exists()", None),
    "unreadable-probe": ("p.exists()", "undeclared_input"),
    "unmodelled-kind": ("assert not p.is_file()", "undeclared_input"),
    "environment": ("assert os.getenv('RC_WRAPPER_INPUT')=='first'", None),
    "environment-absent": (
        "assert os.environ.get('RC_WRAPPER_ABSENT','fallback')=='fallback'",
        None,
    ),
    "environment-key-type": (
        "try: os.environ[object()]\nexcept TypeError: pass\nelse: raise AssertionError",
        "undeclared_input",
    ),
    "environment-list": ("assert list(os.environ)", "undeclared_input"),
    "environment-length": ("assert len(os.environ)>0", "undeclared_input"),
    "environment-repr": ("assert bool(repr(os.environ))", "undeclared_input"),
    "environment-write": ("os.environ['RC_WRAPPER_INPUT']='during'", "undeclared_input"),
    "runner-current-test": (
        "item=type('Item',(),{'nodeid':'control'})(); "
        "update_current(item,'call'); update_current(item,None)",
        None,
    ),
    "current-test-write": ("os.environ['PYTEST_CURRENT_TEST']='user'", "undeclared_input"),
    "current-test-delete": ("os.environ.pop('PYTEST_CURRENT_TEST',None)", "undeclared_input"),
    "current-test-lookalike": (
        "os.environ['PYTEST_CURRENT_TEST_OTHER']='user'",
        "undeclared_input",
    ),
    **{
        f"opaque-{event}": (f"sys.audit('{event}',p,1)", "undeclared_input")
        for event in ("socket.sendto", "socket.sendmsg", "ctypes.dlopen", "os.truncate", "os.chown")
    },
}
OBSERVER_CHILD = """import os,sys,json,socket,subprocess,builtins,io; from pathlib import Path
from devtools.pipeline import cache; from devtools.pipeline.observe import Recorder
from unittest.mock import patch
from importlib.util import cache_from_source
os.environ['RC_WRAPPER_INPUT']='first'; os.environ.pop('RC_WRAPPER_ABSENT',None)
os.environ['PYTEST_CURRENT_TEST']='parent'
from _pytest.runner import _update_current_test_var as update_current
root,temp=map(Path,sys.argv[1:3]);p=root/'a.py';outside=temp.parent/'outside'
flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
tree=cache.Tree(Path(sys.prefix).parent,{}) if sys.argv[4]=='locked' else cache.Tree.read(root)
def originals():
 return (os.open,builtins.open,io.open,os.listdir,os.stat,os.lstat,
  os.path.exists,os.path.isfile,os.path.isdir,os.path.lexists,
  *[getattr(type(os.environ),n) for n in ('__getitem__','__iter__','__len__','__repr__')])
saved=originals();native_stat=os.stat(p);native_lstat=os.lstat(p);exception=None
if sys.argv[4]=='unobserved':
 sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError())
                  if e=='sys.addaudithook' else None)
try:
 with patch('sys.addaudithook', wraps=sys.addaudithook) as added, \
      patch('os.stat',wraps=os.stat) as stats:
  if sys.argv[4]=='unreadable-probe':
   stats.side_effect=PermissionError('controlled unreadable path')
  if sys.argv[4]=='unmodelled-kind':
   stats.side_effect=None;stats.return_value=os.stat_result((4096,0,0,0,0,0,0,0,0,0))
  with Recorder(tree,temp,('duckdb','pyarrow.dataset','polars')) as seen: exec(sys.argv[3])
  if sys.argv[4]=='contents':
   with Recorder(tree,temp,()) as again:
    assert p.read_text()=='tracked' and again.reason is None and added.call_count==1
    stale(root)
    assert again.inputs['listings']=={'.':cache.digest(['.git','a.py','loose'])}
except OSError as error: exception=type(error).__name__
item=cache.Entry.sealed(key=cache.action_key(group='test'),**seen.inputs,
 results={'test_x':('PASSED',0.1)},
 writer={'line':'test','head':'0'*40,'invocation':'i','at':'t'}
) if seen.reason is None else None
before=tree.holds(item) if item else False
restored=saved==originals()
after=before
if sys.argv[4]=='absent-probe':
 (root/'later').write_text('appeared')
 after=cache.Tree.read(root).holds(item) if item else False
if sys.argv[4] in ('environment','environment-absent'):
 os.environ['RC_WRAPPER_INPUT' if sys.argv[4]=='environment' else 'RC_WRAPPER_ABSENT']='changed'
 after=cache.Tree.read(root).holds(item) if item else False
print(json.dumps({'reason':seen.reason,'inputs':seen.inputs,'before':before,'after':after,
 'restored':restored,'exception':exception}))
"""
PYTEST_RECORDS = {
    "passing": ("assert True", None),
    "receipt-in-checkout": ("assert True", "request_invalid"),
    "direct-log-stat": (
        "import logging; "
        "h=next(r() for r in logging._handlerList if isinstance(r(),logging.FileHandler)); "
        "__import__('os').stat(h.baseFilename)",
        "metadata_read",
    ),
    "warm-bytecode": ("assert True", None),
    "stale-bytecode": ("assert True", "not_passing"),
    "skipped": ("pytest.skip()", None),
    "failed": ("assert False", "not_passing"),
    "xfailed": ("pytest.xfail()", "not_passing"),
    "xpassed": ("assert True", "not_passing"),
    "native-loaded": ("assert True", "native_reader"),
    "conftest-read": ("assert True", "undeclared_input"),
    "foreign-temp": ("Path('foreign').read_text()", "undeclared_input"),
    "large": ("assert True", "not_passing"),
    "quarantine": ("assert True", "not_passing"),
    "rerun": ("assert True", "not_passing"),
    "e2e": ("assert True", "not_passing"),
    "native-path": ("assert True", "not_passing"),
    "scandir": ("list(__import__('os').scandir('.'))", "metadata_read"),
    "duplicate": ("assert True", "not_passing"),
    "unwritable": ("assert True", "unwritable"),
    **dict.fromkeys(
        ("bad-key", "used-temp", "cacheprovider", "bytecode"), ("assert True", "request_invalid")
    ),
    "off": ("assert True", None),
    "no-git": ("assert True", "tree_unreadable"),
    "session-error": ("assert True", "not_passing"),
    "session-exception": ("assert True", "not_passing"),
    "direct-stat": ("__import__('os').stat('conftest.py')", "metadata_read"),
    "direct-temp-stat": ("__import__('os').stat(tmp_path)", "metadata_read"),
    "direct-parent-stat": ("__import__('os').stat(Path.cwd().parent)", "metadata_read"),
    "framework-outside": (
        "__import__('_pytest.config',fromlist=['_get_directory'])._get_directory(Path.cwd().parent/'unowned')",
        "undeclared_input",
    ),
    "direct-current": (
        "__import__('os').environ['PYTEST_CURRENT_TEST']='test'",
        "undeclared_input",
    ),
    "linecache-stat": (
        "lc=__import__('linecache'); lc.cache[__file__]=(1,1,[],__file__); lc.checkcache(__file__)",
        "metadata_read",
    ),
}
BOUND = {"exists": True, "isfile": True, "isdir": False, "islink": False, "lexists": True}
PYTEST_RECORDS |= {
    "bound-" + name: (
        "assert bound_input.predicate(__file__) is " + str(value),
        "undeclared_input"
        if getattr(sys.modules.get(os.name), "_path_" + name, None) is getattr(os.path, name)
        else None,
    )
    for name, value in BOUND.items()
}


def _record_pytest(tmp_path, case, *, prepare_only=False):
    from devtools.pipeline import observe

    root = tmp_path / "repo"
    relative = "tests/" + ("native/" if case == "native-path" else "") + "test_sample.py"
    for source_relative in (
        "pipeline/observe.py",
        "pipeline/cache.py",
        "pipeline/receipt.py",
        "pipeline/definition.py",
        "testing/sizes.py",
        "testing/requirements.py",
    ):
        target = root / "src/devtools" / source_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(observe.__file__).parents[1] / source_relative).read_bytes())
    (root / "foreign").write_text("untracked", "utf-8")
    (root / "conftest.py").write_text(
        "from pathlib import Path\n"
        + ("Path('foreign').read_text()\n" if case == "conftest-read" else "")
        + (
            "import pytest\n@pytest.hookimpl(wrapper=True)\n"
            "def pytest_runtest_makereport():\n    report=yield\n"
            "    report.rerun=1\n    return report\n"
            if case == "rerun"
            else ""
        ),
        "utf-8",
    )
    if case in {"session-error", "session-exception"}:
        with (root / "conftest.py").open("a", encoding="utf-8") as stream:
            stream.write(
                "import pytest\n@pytest.hookimpl(wrapper=True,tryfirst=True)\n"
                "def pytest_sessionfinish(session):\n    yield\n    "
                + (
                    "session.exitstatus=1"
                    if case == "session-error"
                    else "raise RuntimeError('finish')"
                )
                + "\n"
            )
    body, why = PYTEST_RECORDS[case]
    (root / relative).parent.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n", "utf-8")
    (root / relative).write_text(
        "import pytest\nfrom pathlib import Path\n"
        + ("import bound_input\n" if case.startswith("bound-") else "")
        + ("@pytest.mark.xfail\n" if case == "xpassed" else "")
        + ("@pytest.mark.e2e\n" if case == "e2e" else "")
        + "@pytest.mark.parametrize('value',[0,1],ids=['zero','one'])\n"
        + "def test_sample(tmp_path,value):\n    "
        + body
        + "\n"
        + "    p=tmp_path/'made'; p.write_text('own'); assert p.read_text()=='own'\n",
        "utf-8",
    )
    policy = {
        "schema": "alphalattice.test-sizes.v1",
        "sizes": ["small", "medium", "large"],
        "limits": {"small": 10, "medium": 60, "large": 600},
        "defaults": [{"pattern": "*", "size": "medium", "owner": "devtools.pipeline"}],
        "large": [],
        "quarantine": [],
    }
    if case == "large":
        policy["large"] = [
            {"pattern": "*", "owner": "devtools.pipeline", "reason": "native control"}
        ]
    if case == "quarantine":
        policy["quarantine"] = [
            {
                "node": "tests/test_sample.py::test_sample",
                "owner": "devtools.pipeline",
                "reason": "control",
                "until": "2099-01-01",
            }
        ]
    path = root / "config/registries/test-sizes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(policy), "utf-8")
    if case.startswith("bound-"):
        (root / "bound_input.py").write_text(
            "from os.path import " + case.removeprefix("bound-") + " as predicate\n", "utf-8"
        )
    if case == "duplicate":
        (root / "tests/test_other.py").write_bytes((root / "tests/test_sample.py").read_bytes())
    for words in (
        ["init", "-q"],
        [
            "add",
            "src",
            "conftest.py",
            "tests",
            "pytest.ini",
            "config",
            *(["bound_input.py"] if case.startswith("bound-") else []),
        ],
    ):
        subprocess.run([*GIT, root, *words], check=True)
    if prepare_only:
        return root
    request = {
        "store": str(tmp_path / "store"),
        "key": KEY,
        "writer": WRITER,
        "native_readers": ["json" if case == "native-loaded" else "never_loaded"],
        "basetemp": str(tmp_path / "fresh"),
        "verdict": str(tmp_path / "verdict.json"),
    }
    if case == "direct-log-stat":
        request["log"] = str(tmp_path / "debug.log")
    if case == "receipt-in-checkout":
        request["events"] = str(root / "owned.py")
    if case == "bad-key":
        request["key"] = "not a digest"
    if case == "used-temp":
        Path(request["basetemp"]).mkdir()
    if case == "unwritable":
        Path(request["store"]).write_text("not a directory", "utf-8")
    (tmp_path / "request.json").write_text(json.dumps(request), "utf-8")
    child_env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        ALPHALATTICE_PIPELINE_EVENTS="",
        PYTHONDONTWRITEBYTECODE="" if case == "bytecode" else "1",
        PATH="" if case == "no-git" else os.environ["PATH"],
    )
    if case in {"warm-bytecode", "stale-bytecode"}:
        warm = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(tmp_path / "warm"),
                relative,
            ],
            cwd=root,
            env=dict(child_env, PYTHONDONTWRITEBYTECODE=""),
            capture_output=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
        )
        assert warm.returncode == 0, warm.stderr
    if case == "stale-bytecode":
        source = root / relative
        before = source.stat()
        source.write_bytes(source.read_bytes().replace(b"assert True", b"assert 0   "))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    done = subprocess.run(
        [
            sys.executable,
            *([] if case == "bytecode" else ["-B"]),
            "-m",
            "pytest",
            "-q",
            *(["-o", "log_file=" + request["log"]] if case == "direct-log-stat" else []),
            *(["-p", "bound_input"] if case.startswith("bound-") else []),
            *([] if case == "cacheprovider" else ["-p", "no:cacheprovider"]),
            "-p",
            "devtools.pipeline.observe",
            *([] if case == "off" else ["--result-cache-record", str(tmp_path / "request.json")]),
            "--basetemp",
            request["basetemp"],
            "tests" if case == "duplicate" else relative,
        ],
        cwd=root,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    assert done.returncode == (
        4
        if why == "request_invalid"
        else 1
        if case in {"failed", "session-error", "session-exception", "stale-bytecode"}
        else 0
    ), done.stdout + done.stderr
    if why == "request_invalid" or case == "off":
        assert case == "off" or "result_cache.request_invalid" in done.stderr
        assert not Path(request["verdict"]).exists()
        return
    verdict = json.loads(Path(request["verdict"]).read_text("utf-8"))
    assert verdict["reason"] == ("result_cache." + why if why else None), verdict
    assert verdict["decision"] == ("UNCACHEABLE" if why else "STORED")
    store = cache.Store.open(Path(request["store"]))
    stored = store.lookup(KEY, cache.Tree.read(root)) if isinstance(store, cache.Store) else None
    assert (stored is None) == bool(why)
    if stored:
        assert set(stored.results) == {"test_sample[zero]", "test_sample[one]"}
        assert all(
            r[0] == ("SKIPPED" if case == "skipped" else "PASSED") for r in stored.results.values()
        )
        assert {"conftest.py", "tests/test_sample.py"} <= stored.files.keys()
        assert {
            "src/devtools/pipeline/observe.py",
            "src/devtools/pipeline/cache.py",
        } <= stored.files.keys()
        (root / "tests/test_added.py").touch()
        assert store.lookup(KEY, cache.Tree.read(root)) is None


@pytest.mark.proves("pipeline.result_cache_refuses_undeclared_inputs")
@pytest.mark.parametrize("case", list(OBSERVED) + ["pytest-" + c for c in PYTEST_RECORDS])
def test_a_recorder_keeps_only_inputs_the_shared_cache_can_verify(tmp_path, case):
    """Only observed, complete first-attempt passing groups may publish reusable results."""
    if case.startswith("pytest-"):
        _record_pytest(tmp_path, case.removeprefix("pytest-"))
        return
    from devtools.pipeline import observe

    root, temp = tmp_path / "repo", tmp_path / "temp"
    root.mkdir()
    temp.mkdir()
    (root / "a.py").write_text("tracked")
    (root / "loose").write_text("loose")
    (temp / "foreign").write_text("foreign")
    (tmp_path / "outside").write_text("outside")
    if case.startswith("bytecode-"):
        (root / "b.py").write_text("VALUE = 1\n")
        py_compile.compile(str(root / "b.py"), doraise=True)
        (root / "__pycache__" / "loose.pyc").write_bytes(b"loose")
    for words in (
        ["init", "-q"],
        ["add", "a.py", *(["b.py"] if case.startswith("bytecode-") else [])],
    ):
        subprocess.run([*GIT, root, *words], check=True)
    done = subprocess.run(
        [sys.executable, "-B", "-c", OBSERVER_CHILD, str(root), str(temp), OBSERVED[case][0], case],
        env=dict(os.environ, PYTHONPATH=str(Path(observe.__file__).parents[2])),
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    assert done.returncode == 0, done.stderr
    found, why = json.loads(done.stdout), OBSERVED[case][1]
    assert found["reason"] == ("result_cache." + why if why else None)
    assert found["before"] is (why is None) and found["restored"]
    assert case != "contents" or set(found["inputs"]["files"]) == {"a.py"}
    assert case != "bytecode-import" or set(found["inputs"]["files"]) == {"b.py"}
    assert case not in ("own-temp", "borrowed-list") or found["inputs"]["listings"] == {
        ".": cache.digest([".git", "a.py", "loose"])
    }

    if case in ("path-probe", "fast-probes", "lexists-probe", "absent-probe"):
        assert found["inputs"]["probes"] == (
            {"later": None} if case == "absent-probe" else {"a.py": "file"}
        )
    if case in ("absent-probe", "environment", "environment-absent"):
        assert found["after"] is False
    if case in ("environment", "environment-absent"):
        key = "RC_WRAPPER_INPUT" if case == "environment" else "RC_WRAPPER_ABSENT"
        assert found["inputs"]["env"] == {
            key: cache.digest("first" if case == "environment" else None)
        }
    if case == "runner-current-test":
        assert "PYTEST_CURRENT_TEST" not in found["inputs"]["env"]
    if case in ("metadata-missing", "unreadable-probe", "environment-key-type"):
        assert (
            found["exception"]
            == {
                "metadata-missing": "FileNotFoundError",
                "unreadable-probe": "PermissionError",
                "environment-key-type": None,
            }[case]
        )


@pytest.mark.proves("pipeline.result_cache_runner_serves_verified_groups")
@pytest.mark.parametrize("mode", ["record", "shadow", "on", "on-duplicate", "on-move", "options"])
def test_the_runner_records_checks_and_reuses_one_verified_group(tmp_path, monkeypatch, mode):
    """Equal inputs may hit; shadow executes, and dirty stages never publish."""
    import importlib.util
    import runpy

    assert importlib.util.find_spec("devtools.pipeline.reuse") is not None
    from devtools.pipeline.reuse import Options, Reuse

    if mode == "options":
        assert Options.read(mode="off", cap="1").cap == 1
        for raw in ("0", "-1", "NaN", "true"):
            with pytest.raises(ValueError, match=r"^result_cache\.request_invalid"):
                Options.read(cap=raw)
        with pytest.raises(ValueError, match=r"^result_cache\.request_invalid"):
            Options.read(mode="unknown")
        with pytest.raises(ValueError, match=r"^result_cache\.request_invalid"):
            Options(cap=True)
        return
    root = _record_pytest(tmp_path, "passing", prepare_only=True)
    if mode == "on-duplicate":
        source = root / "tests/test_sample.py"
        source.write_text(
            source.read_text("utf-8").replace(
                "assert True", "assert Path(__file__).name == 'test_sample.py'"
            ),
            "utf-8",
        )
        (root / "tests/test_other.py").write_bytes(source.read_bytes())
    (root / "uv.lock").write_text("locked environment", "utf-8")
    (root / "config/registries/requirements.json").write_text('{"entries":{}}', "utf-8")
    subprocess.run([*GIT, root, "add", "-A"], check=True)
    subprocess.run([*GIT, root, "commit", "-qm", "inputs"], check=True)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    runner = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/check_alpha_verification_impact.py")
    )["_run_group"]
    writer = {**WRITER, "head": cache._git(root, "rev-parse", "HEAD").strip()}
    options = Options(tmp_path / "store", mode.split("-")[0], 1048576)

    def run(number):
        owner = Reuse(root, tmp_path / str(number), options, writer, native_readers=())
        result = runner(
            ("tests/test_sample.py",), "-m", "not real_evidence", capture=True, reuse=owner
        )
        assert result.code == 0, result.stdout + result.stderr
        return owner.decisions

    first, second = run(1), run(2)
    assert first[0]["decision"] == "MISS", first
    assert second[0]["decision"] == "HIT", second
    assert second[0]["executed"] == (not mode.startswith("on"))
    if mode == "on-duplicate":
        owner = Reuse(root, tmp_path / "duplicate", options, writer, native_readers=())
        result = runner(
            ("tests/test_other.py",), "-m", "not real_evidence", capture=True, reuse=owner
        )
        assert result.code == 1, (
            "an equal-body test at another locator must execute its failing assertion"
        )
        assert owner.decisions[0]["decision"] != "HIT"
        return
    if mode == "on-move":
        (root / "tests/test_sample.py").rename(root / "tests/test_moved.py")
        subprocess.run([*GIT, root, "add", "-A"], check=True)
        subprocess.run([*GIT, root, "commit", "-qm", "relocate"], check=True)
        writer = {**writer, "head": cache._git(root, "rev-parse", "HEAD").strip()}
        for number, wanted in ((3, "MISS"), (4, "HIT")):
            owner = Reuse(root, tmp_path / str(number), options, writer, native_readers=())
            result = runner(
                ("tests/test_moved.py",), "-m", "not real_evidence", capture=True, reuse=owner
            )
            assert result.code == 0 and owner.decisions[0]["decision"] == wanted
        return
    source = root / "tests/test_sample.py"
    source.write_text(source.read_text("utf-8") + "\n# changed input\n", "utf-8")
    assert run(3)[0]["decision"] == "MISS"
    assert len(list((tmp_path / "store/ac").glob("*/*.json"))) == 1
