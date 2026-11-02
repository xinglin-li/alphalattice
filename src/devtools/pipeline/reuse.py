"""Verified group reuse for the impact runner. Observation and publication retain their owners.

Borrowed from Ninja depfiles and Bazel action keys. Recording and shadow never skip execution;
explicit on mode may serve a verified first-attempt entry. Only a clean committed tree writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from devtools.pipeline import cache
from devtools.pipeline.receipt import TestResult
from devtools.testing.requirements import file_holders


@dataclass(frozen=True)
class Options:
    root: Path | None = None
    mode: str = "off"
    cap: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"off", "record", "shadow", "on"} or (
            self.cap is not None and (type(self.cap) is not int or self.cap <= 0)
        ):
            raise ValueError("result_cache.request_invalid: supply a mode and a positive byte cap")

    @classmethod
    def read(
        cls, root: str | None = None, mode: str | None = None, cap: str | None = None
    ) -> Options:
        location = root if root is not None else os.environ.get("ALPHALATTICE_RESULT_CACHE")
        selected = mode or os.environ.get("ALPHALATTICE_RESULT_CACHE_MODE", "off")
        raw = cap if cap is not None else os.environ.get("ALPHALATTICE_RESULT_CACHE_BYTES")
        try:
            return cls(
                Path(location).resolve() if location else None,
                selected,
                int(raw) if raw else None,
            )
        except (ValueError, TypeError):
            raise ValueError(
                "result_cache.request_invalid: supply a mode and a positive byte cap"
            ) from None

    def environment(self) -> dict[str, str]:
        return {
            "ALPHALATTICE_RESULT_CACHE": str(self.root or ""),
            "ALPHALATTICE_RESULT_CACHE_MODE": self.mode,
            "ALPHALATTICE_RESULT_CACHE_BYTES": str(self.cap) if self.cap is not None else "",
        }


class Reuse:
    """One stage's tree and policy inventory, shared by its existing group pool."""

    def __init__(
        self,
        root: Path,
        out: Path,
        options: Options,
        writer: dict[str, str],
        *,
        native_readers: tuple[str, ...],
    ) -> None:
        if (
            not isinstance(writer, dict)
            or set(writer) != {"line", "head", "invocation", "at"}
            or any(not isinstance(v, str) or not v.strip() for v in writer.values())
        ):
            raise ValueError("result_cache.request_invalid: name line, head, invocation and time")
        self.root, self.out, self.options, self.writer = root.resolve(), out, options, writer
        self.native_readers = native_readers
        self.decisions: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.properties: dict[str, dict[str, Any]] = {}
        self.deadline: float | None = None
        self.temp: Path | None = None
        self.out.mkdir(parents=True)
        self.tree = cache.Tree.read(self.root)
        self.store = (
            cache.Store.open(options.root, options.cap)
            if options.root
            and options.mode != "off"
            and not options.root.resolve().is_relative_to(self.root)
            else "result_cache.disabled"
        )
        self.clean = False
        self.unavailable: str | None = None
        try:
            self.clean = (
                not cache._git(root, "status", "--porcelain").strip()
                and cache._git(root, "rev-parse", "HEAD").strip() == writer["head"]
            )
            self.policy = json.loads(
                (root / "config/registries/test-sizes.json").read_text("utf-8")
            )
            self.requirements = json.loads(
                (root / "config/registries/requirements.json").read_text("utf-8")
            )["entries"]
            self.lock = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
        except (OSError, ValueError, subprocess.SubprocessError):
            self.clean, self.policy, self.requirements, self.lock = False, {}, {}, ""
            self.unavailable = "result_cache.tree_unreadable"
        self.recorder = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }

    def identity(
        self, group: tuple[str, ...], arguments: tuple[str, ...]
    ) -> tuple[str, str, str | None]:
        holders = []
        problem = None
        try:
            sources = {name: (self.root / name).read_bytes() for name in group}
            holders = [
                h for name, raw in sources.items() for h in file_holders(name, raw.decode("utf-8"))
            ]
            if not holders or len(group) != 1:
                problem = "result_cache.undeclared_input"
            bound = {k: self.requirements[k] for h in holders for k in h.ids}
            key = cache.action_key(
                recorder=self.recorder,
                native_readers=self.native_readers,
                lock=self.lock,
                python=[sys.implementation.name, sys.version, sys.platform],
                arguments=arguments,
                lane=os.environ.get("ALPHALATTICE_TEST_LANE"),
                holders=sorted((h.ids, h.node.partition("::")[2], h.digest) for h in holders),
                requirements=bound,
                policy=self.policy,
                modules=[hashlib.sha256(sources[n]).hexdigest() for n in group],
            )
        except (OSError, ValueError, KeyError, SyntaxError):
            key, problem = cache.action_key(group=group), "result_cache.undeclared_input"
        group_id = (
            cache.digest([(h.ids, h.node.partition("::")[2]) for h in holders]) if holders else key
        )
        return key, group_id, problem

    def lookup(self, key: str, group: tuple[str, ...]) -> cache.Entry | None:
        entry = (
            self.store.lookup(key, self.tree)
            if isinstance(self.store, cache.Store) and isinstance(self.tree, cache.Tree)
            else None
        )
        if entry:
            try:
                cached_results(entry, group, self.policy)
            except ValueError:
                entry = None
        return entry

    def prepare(
        self, group: tuple[str, ...], arguments: tuple[str, ...], directory: Path
    ) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        reason = self.unavailable or (
            self.tree
            if isinstance(self.tree, str)
            else self.store
            if isinstance(self.store, str)
            else None
        )
        key, group_id, problem = self.identity(group, arguments)
        reason = reason or problem
        entry = self.lookup(key, group) if reason is None else None
        decision = {
            "group": cache.digest(
                (group_id, group)
            ),  # locator disambiguates events, never the action key
            "mode": self.options.mode,
            "decision": "HIT" if entry else "UNCACHEABLE" if reason else "MISS",
            "entry": {"key": entry.key, "digest": entry.digest, "head": entry.writer["head"]}
            if entry
            else None,
            "reason": reason,
            "expected": {group[0] + "::" + n: v[0] for n, v in entry.results.items()}
            if entry
            else {},
            "executed": not (entry and self.options.mode == "on"),
        }
        request = None
        if reason is None and self.clean and isinstance(self.store, cache.Store):
            request = directory / "request.json"
            request.write_text(
                json.dumps(
                    dict(
                        store=str(self.store.root),
                        cap=self.store.cap,
                        key=key,
                        writer=self.writer,
                        native_readers=self.native_readers,
                        events=str(directory / "results.jsonl"),
                        log=str(directory / "debug.log"),
                        basetemp=str(directory / "temp"),
                        verdict=str(directory / "verdict.json"),
                    )
                ),
                "utf-8",
            )
        return dict(
            decision=decision,
            entry=entry,
            request=request,
            directory=directory,
            key=key,
            group=group,
        )

    def finish(self, plan: dict[str, Any], results: list[dict[str, Any]]) -> None:
        entry, decision = plan["entry"], plan["decision"]
        if entry and self.options.mode == "shadow" and isinstance(self.store, cache.Store):
            actual = {r["node"].partition("::")[2]: r["status"] for r in results}
            if actual != {n: value[0] for n, value in entry.results.items()}:
                (self.store.root / "disabled").write_text(
                    "shadow result mismatch; inspect this run before enabling reuse", "utf-8"
                )
                decision["reason"] = "result_cache.not_passing"
        if decision["executed"]:
            try:
                verdict = json.loads((plan["directory"] / "verdict.json").read_text("utf-8"))
                self.properties.update(
                    {
                        plan.get("group", ("",))[0] + "::" + n: p
                        for n, p in verdict.get("properties", {}).items()
                    }
                )
                if verdict["decision"] == "UNCACHEABLE":
                    decision.update(
                        decision="UNCACHEABLE", reason=verdict["reason"], entry=None, expected={}
                    )
            except (OSError, ValueError, KeyError):
                if plan["request"]:
                    decision.update(
                        decision="UNCACHEABLE",
                        reason="result_cache.undeclared_input",
                        entry=None,
                        expected={},
                    )
        if entry and not decision["executed"]:
            self.properties.update(
                {
                    r["node"]: entry.details[r["node"].partition("::")[2]].get("_properties", {})
                    for r in results
                }
            )
        self.decisions.append(decision)
        self.results.extend(results)

    def close(self) -> None:
        if isinstance(self.store, cache.Store) and self.clean:
            self.store.evict()


def cached_results(
    entry: cache.Entry, group: tuple[str, ...], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Restore observed metadata through the current locator without inventing size or limits."""
    if entry.details.keys() != entry.results.keys():
        raise ValueError("result_cache.request_invalid: no complete receipt metadata")
    origin = dict(key=entry.key, digest=entry.digest, head=entry.writer["head"])
    results = [
        TestResult.model_validate(
            {
                **{
                    k: v
                    for k, v in entry.details[name].items()
                    if k not in {"_properties", "_node"}
                },
                "node": group[0] + "::" + name,
                "cached_from": origin,
            }
        )
        for name in entry.results
    ]
    for name, result in zip(entry.results, results, strict=True):
        if entry.details[name].get("_node") != group[0] + "::" + name:
            raise ValueError("result_cache.request_invalid: the test locator changed; run it")
        if (
            (result.status, result.seconds) != entry.results[name]
            or result.attempt != 1
            or result.solo
            or result.size == "large"
            or result.requirement_layer in {"e2e", "native"}
        ):
            raise ValueError("result_cache.request_invalid: run this group without reuse")
    return [r.model_dump(mode="json", by_alias=True) for r in results]


def write_results(owner: Reuse) -> None:
    """The parent emits one event stream and JUnit with explicit cache decisions."""
    stream = [{"cache": c} for c in owner.decisions] + owner.results
    (owner.out / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stream), "utf-8")
    debug = []
    for path in owner.out.glob("*/debug.log"):
        debug.extend(path.read_text("utf-8").splitlines())
    (owner.out / "pytest-debug.log").write_text("\n".join(debug), "utf-8")
    suite = ET.Element("testsuite", tests=str(len(owner.results)))
    for result in owner.results:
        path, _, name = result["node"].partition("::")
        case = ET.SubElement(
            suite,
            "testcase",
            classname=path.removesuffix(".py").replace("/", "."),
            name=name,
            time=str(result["seconds"]),
        )
        props = ET.SubElement(case, "properties")
        for name, value in owner.properties.get(result["node"], {}).items():
            ET.SubElement(props, "property", name=name, value=str(value))
        if result["status"] != "PASSED":
            ET.SubElement(
                case,
                "skipped" if result["status"] == "SKIPPED" else "failure",
                message=result.get("failure", {}).get("message", result["status"]),
            )
    ET.ElementTree(suite).write(owner.out / "junit.xml", encoding="utf-8")
