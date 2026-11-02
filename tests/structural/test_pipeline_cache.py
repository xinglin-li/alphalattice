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
    "borrowed-list": ("originals[3](root)", None),
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
}
OBSERVER_CHILD = """import os,sys,json,socket,subprocess,builtins,io; from pathlib import Path
from devtools.pipeline import cache; from devtools.pipeline.observe import Recorder
from unittest.mock import patch
from importlib.util import cache_from_source
root,temp=map(Path,sys.argv[1:3]);p=root/'a.py';outside=temp.parent/'outside'
flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
tree=cache.Tree(Path(sys.prefix).parent,{}) if sys.argv[4]=='locked' else cache.Tree.read(root)
originals=(os.open,builtins.open,io.open,os.listdir)
if sys.argv[4]=='unobserved':
 sys.addaudithook(lambda e,a: (_ for _ in ()).throw(RuntimeError())
                  if e=='sys.addaudithook' else None)
try:
 with patch('sys.addaudithook', wraps=sys.addaudithook) as added:
  with Recorder(tree,temp,('duckdb','pyarrow.dataset','polars')) as seen: exec(sys.argv[3])
  if sys.argv[4]=='contents':
   with Recorder(tree,temp,()) as again:
    assert p.read_text()=='tracked' and again.reason is None and added.call_count==1
    stale(root)
    assert again.inputs['listings']=={'.':cache.digest(['.git','a.py','loose'])}
except OSError: pass
item=cache.Entry.sealed(key=cache.action_key(group='test'),**seen.inputs,
 results={'test_x':('PASSED',0.1)},
 writer={'line':'test','head':'0'*40,'invocation':'i','at':'t'}
) if seen.reason is None else None
before=tree.holds(item) if item else False
restored=originals==(os.open,builtins.open,io.open,os.listdir)
print(json.dumps({'reason':seen.reason,'inputs':seen.inputs,'before':before,'restored':restored}))
"""


@pytest.mark.proves("pipeline.result_cache_refuses_undeclared_inputs")
@pytest.mark.parametrize("case", OBSERVED)
def test_a_recorder_keeps_only_inputs_the_shared_cache_can_verify(tmp_path, case):
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
