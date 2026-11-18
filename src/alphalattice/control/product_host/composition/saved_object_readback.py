"""Read saved workspace objects through their product routes, without test dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alphalattice.control.task_control.upgrade import carry_tasks
from alphalattice.interface.local_application.web import SESSION_COOKIE, SESSION_HEADER
from alphalattice.kernel.data_layer.history import data_root, own_root


def _request(
    session: Any, path: str, *, method: str = "GET", payload: Any = None, timeout: float = 120.0
) -> tuple[int, dict[str, str], bytes]:
    application = session.web.application
    base = session.url.rstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{base}{path}", data=body, method=method)
    request.add_header(SESSION_HEADER, application.session_token)
    request.add_header("Cookie", f"{SESSION_COOKIE}={application.session_token}")
    if method == "POST":
        request.add_header("Origin", base)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def verdict(status: int, body: bytes) -> str:
    """Describe one read by its opening or named refusal."""
    try:
        data: Any = json.loads(body)
    except ValueError:
        text = body[:160].decode("utf-8", "replace").replace("\n", " ")
        return ("OPENS" if status == 200 else f"REFUSES {text}").strip()
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    if not isinstance(data, dict):
        return "OPENS" if status == 200 else f"REFUSES http {status}"
    code = data.get("failure_code")
    if status != 200 or data.get("status") == "REFUSED" or code:
        return f"REFUSES {code or data.get('error') or status}"
    parts = ["OPENS"]
    if data.get("method_standing"):
        parts.append(str(data["method_standing"]))
    if data.get("research_lane"):
        parts.append(str(data["research_lane"]))
    if data.get("recommendation_reproduced") is False:
        parts.append("NOT_REPRODUCED")
    if data.get("review_under_installed_policy") is False:
        parts.append("EARLIER_POLICY")
    disposition = (
        (data.get("evidence") or {}).get("disposition")
        if isinstance(data.get("evidence"), dict)
        else None
    )
    for value in (
        disposition,
        data.get("state"),
        data.get("disposition") if isinstance(data.get("disposition"), str) else None,
    ):
        if value:
            parts.append(str(value))
    return " ".join(parts)


_VOLATILE = re.compile(
    r"^(seconds|bytes|pid|port|elapsed|timing|session_launches|read_count|attempts?)$"
    r"|_(seconds|bytes|ms)$|^elapsed_|^timing_"
)
"""Keys whose numbers the clock, the host or the read decides, never the saved object."""


def numbers(status: int, body: bytes) -> dict[str, str]:
    """Hash numerical leaves by section, preserving their count.

    an OPENS verdict alone did not detect a change from `rank_ic` 0.01 to 0.9.

    A section is an answer's first two keys; list positions are kept in each leaf's path and
    left out of its section. Booleans, strings and the keys `_VOLATILE` names are not numbers of
    the saved object.
    """
    if status != 200:
        return {}
    try:
        data: Any = json.loads(body)
    except ValueError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    sections: dict[str, list[str]] = {}

    def walk(value: Any, path: tuple[str, ...]) -> None:
        """Read the captured source document under its admitted contract."""
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                if not _VOLATILE.search(str(key)):
                    walk(value[key], (*path, str(key)))
        elif isinstance(value, list):
            for position, item in enumerate(value):
                walk(item, (*path, f"[{position}]"))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            keys = [part for part in path if not part.startswith("[")][:2]
            leaves = sections.setdefault(".".join(keys) or "-", [])
            leaves.append(f"{'.'.join(path)}={value!r}")

    walk(data, ())
    return {
        section: hashlib.sha256("\n".join(leaves).encode("utf-8")).hexdigest()[:16]
        + f":{len(leaves)}"
        for section, leaves in sorted(sections.items())
    }


def _copy(source: Path, target: Path) -> None:
    """Copy a workspace keeping its hard links.

    A file the source holds under several names is copied once and
    linked again, as in the source.
    """
    seen: dict[tuple[int, int], str] = {}

    def copy_one(src: str, dst: str) -> str:
        """Read the captured source document under its admitted contract."""
        info = os.stat(src)
        key = (info.st_dev, info.st_ino)
        if info.st_nlink > 1 and key in seen:
            os.link(seen[key], dst)
            return dst
        shutil.copy2(src, dst)
        if info.st_nlink > 1:
            seen[key] = dst
        return dst

    shutil.copytree(source, target, symlinks=True, copy_function=copy_one)


def probe(
    harvest: Path,
    work: Path,
    out: Path,
    *,
    keep: bool = False,
    fallback: Callable[[Path], Any] | None = None,
    harness: Callable[[Path, Path, list[dict[str, Any]]], None] | None = None,
) -> None:
    """Read saved objects through product routes and write the comparison receipt; the copies
    open on a data root of their own under `work`, so their Tasks never reach the machine's log."""
    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    source = data_root()  # where the harvested workspaces' Tasks are, carried into each copy
    # A copy's upgrade backups go beside its root, never to the person's backup root.
    with own_root(work / "data-root"), _environment(ALPHALATTICE_BACKUP_ROOT=str(work / "backups")):
        opener = LocalPortfolioWebSession.from_workspace
        _read_saved(
            harvest, work, out, source, opener, keep=keep, fallback=fallback, harness=harness
        )
    if not keep:  # once the root's log is closed
        shutil.rmtree(work, ignore_errors=True)


def _read_saved(
    harvest: Path,
    work: Path,
    out: Path,
    source: Path,
    opener: Callable[[Path], Any],
    *,
    keep: bool,
    fallback: Callable[[Path], Any] | None,
    harness: Callable[[Path, Path, list[dict[str, Any]]], None] | None,
) -> None:
    """Each harvested workspace, copied with its Tasks, opened and read route by route."""
    roots = sorted({p.parent for p in harvest.rglob("research-workspace.json")})
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, root in enumerate(roots):
        name = str(root.relative_to(harvest))
        copy = work / f"w{index:03d}"
        if index and not keep:  # the last workspace's sessions have stopped
            shutil.rmtree(work / f"w{index - 1:03d}", ignore_errors=True)
        _copy(root, copy)
        carry_tasks(root, copy, root=source)
        opened = _open_copy(copy, name, rows, fallback, opener)
        if opened is None:
            continue
        session, how = opened
        try:
            with session as live:
                rows.append(_workspace_row(name, "OPEN", how))
                _read_routes(_reader(live, name, rows), copy)
        except Exception as error:
            rows.append(
                _workspace_row(name, "SESSION", f"ERROR {type(error).__name__}: {str(error)[:140]}")
            )
    if harness is not None:
        harness(harvest, work, rows)
    _write_receipt(out, len(roots), rows, time.perf_counter() - started)


def _workspace_row(name: str, op: str, verdict: str) -> dict[str, Any]:
    return {"workspace": name, "kind": "workspace", "key": "-", "op": op, "verdict": verdict}


def _open_copy(
    copy: Path,
    name: str,
    rows: list[dict[str, Any]],
    fallback: Callable[[Path], Any] | None,
    opener: Callable[[Path], Any],
) -> tuple[Any, str] | None:
    """The copy's session, opened and stopped once by the product path, with how it opened; a
    harness-built workspace the product path cannot compose (a test strategy package, an absent
    evidence pack) opens the way its builder opened it, on both trees alike. None when neither
    opens it."""
    try:
        session = opener(copy)
        session.start()
        try:
            _upgraded(session)
        finally:
            session.stop()
        return opener(copy), "OPENS"
    except Exception as error:
        rows.append(_workspace_row(name, "PRODUCT_OPEN", f"REFUSES {str(error)[:140]}"))
    if fallback is None:
        return None
    try:
        return fallback(copy), "OPENS (harness)"
    except Exception as refused:
        rows.append(_workspace_row(name, "OPEN", f"REFUSES {str(refused)[:140]}"))
        return None


def _upgraded(session: Any) -> None:
    """A 0.2.7 copy's first 0.2.8 start shows its upgrade plan: acknowledge it as the person
    would, and wait for the run, so the copy's saved objects are read as the upgrade left them."""
    if getattr(session, "first_start", None) is None:
        return
    _status, _headers, body = _request(session, "/api/upgrade")
    acknowledge = json.loads(body)["next_requests"]["acknowledge"]
    _request(session, "/api/upgrade/acknowledge", method="POST", payload=acknowledge)
    if not session.upgraded():
        raise ValueError(f"upgrade.refused: {session.first_start.failure}")


@contextmanager
def _environment(**values: str) -> Iterator[None]:
    before = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _reader(live: Any, name: str, rows: list[dict[str, Any]]) -> Callable[..., Any]:
    """One route's read of the live copy, its verdict appended to `rows`."""

    def read(kind: str, key: str, op: str, path: str, **kwargs: Any) -> Any:
        """Read the captured source document under its admitted contract."""
        row: dict[str, Any] = {"workspace": name, "kind": kind, "key": key, "op": op}
        try:
            status, _headers, body = _request(live, path, timeout=120.0, **kwargs)
            rows.append(row | {"verdict": verdict(status, body), "numbers": numbers(status, body)})
            return json.loads(body) if status == 200 else None
        except Exception as error:
            rows.append(row | {"verdict": f"ERROR {type(error).__name__}: {str(error)[:120]}"})
            return None

    return read


def _read_routes(read: Callable[..., Any], copy: Path) -> None:
    """Every saved object's routes: studies, results and their frozen candidates, feature trials
    and Evidence CRO reviews."""
    tasks = read("tasks", "-", "TASKS", "/api/tasks") or {}
    for task in tasks.get("tasks") or []:
        if task.get("task_kind") == "research_experiment" and task.get("lifecycle") == "SUCCEEDED":
            tid = task["task_id"]
            read("study", tid, "EXPERIMENT_READBACK", f"/api/experiments/readback?task_id={tid}")
            read("study", tid, "EXPERIMENT_EXPORT", f"/api/experiments/export?task_id={tid}")
            draft = {"method": "POST", "payload": {"task_id": tid}}
            read("study", tid, "EXPERIMENT_DRAFT", "/api/experiments/draft", **draft)
            # Replay recomputes nothing: it reuses the sealed evidence or refuses, so a moved
            # implementation that no successor records shows here.
            read("study", tid, "EXPERIMENT_REPLAY", "/api/experiments/replay", **draft)
    results = read("results", "-", "RESULTS", "/api/results") or {}
    listed = results.get("results") or results.get("items") or []
    for result in listed if isinstance(listed, list) else []:
        rh = result.get("result_hash") if isinstance(result, dict) else None
        if rh:
            read("result", rh[:12], "REPORT", f"/api/report?result_hash={rh}")
        frozen = result.get("frozen") if isinstance(result, dict) else None
        candidate = frozen.get("candidate_hash") if isinstance(frozen, dict) else None
        if candidate:
            path = f"/api/finalization?candidate_hash={candidate}"
            read("frozen candidate", candidate[:12], "FINALIZATION", path)
    trials = read("trials", "-", "FEATURE_TRIALS", "/api/feature-trials") or {}
    for trial in trials.get("trials") or []:
        fid = trial.get("feature_trial_id")
        if fid:
            path = f"/api/feature-trials/readback?feature_trial_id={fid}"
            read("trial", fid[:12], "FEATURE_TRIAL_READBACK", path)
    for publication in sorted(copy.rglob("cro-review-publications/*.json")):
        ph = publication.stem
        path = f"/api/evidence-cro/export?review_publication_hash={ph}"
        read("review", ph[:12], "EVIDENCE_CRO_EXPORT", path)


def _write_receipt(out: Path, workspaces: int, rows: list[dict[str, Any]], elapsed: float) -> None:
    """The comparison receipt, and its verdicts counted on standard output."""
    out.write_text(
        json.dumps(
            {"workspaces": workspaces, "seconds": round(elapsed, 1), "rows": rows}, indent=1
        ),
        encoding="utf-8",
        newline="\n",
    )
    table = Counter(
        (
            r["kind"],
            r["op"],
            r["verdict"].split(" ")[0]
            + (" " + r["verdict"].split(" ", 1)[1][:70] if " " in r["verdict"] else ""),
        )
        for r in rows
    )
    print(f"{workspaces} workspaces, {len(rows)} reads, {elapsed:.0f} s")
    for (kind, op, v), n in sorted(table.items()):
        print(f"  {n:4}  {kind:17} {op:20} {v}")
