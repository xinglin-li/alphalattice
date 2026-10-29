"""The pipeline's one definition, config/pipeline.toml: its stages and rows, checked on load.

A stage runs only the rows it names, in order. A row is a port: what it proves, its owner and where
it runs; each host supplies one adapter for it. The test sizes' limits stay in the test registry,
which owns them: a stage names only its sizes and lane, and each must be registered there.
"""

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

from devtools.architecture.test_sizes import load_policy

SCHEMA = "alphalattice.pipeline.v1"
PATH = "config/pipeline.toml"
HOSTS = frozenset({"workshop", "public"})
KINDS = frozenset({"command", "pytest", "pytest-plugin", "ci-job", "workshop"})
TRIGGERS = frozenset({"commit hook", "line", "queue", "develop", "schedule", "person"})
RANGES = frozenset({"staged", "card", "each-card", "set", "tree"})


class DefinitionError(ValueError):
    """A definition the pipeline refuses: its code, then what is wrong and the way on."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class Row:
    """One registered row: a guard, a test step or a CI job, with where it runs."""

    id: str
    kind: str
    owner: str
    reason: str
    hosts: tuple[str, ...]
    run: tuple[str, ...] = ()
    module: str = ""
    plugins: tuple[str, ...] = ()
    selection: dict[str, Any] = field(default_factory=dict)
    public: tuple[str, ...] = ()
    job: str = ""
    escape: str = ""
    review: str = ""


@dataclass(frozen=True)
class Stage:
    """One registered stage and the rows it runs, in order."""

    id: str
    order: int
    reason: str
    trigger: str
    range: str
    bound_minutes: float
    rows: tuple[str, ...]
    lane: str = ""
    sizes: tuple[str, ...] = ()
    place: str = ""


@dataclass(frozen=True)
class Definition:
    """The checked definition, its stages in order, and the digest of the bytes it was read from."""

    sha256: str
    stages: dict[str, Stage]
    rows: dict[str, Row]

    def stage(self, name: str) -> Stage:
        if name not in self.stages:
            raise DefinitionError(
                "pipeline.stage_unknown",
                f"{name!r} is not a stage of {PATH}; its stages are {', '.join(self.stages)}",
            )
        return self.stages[name]


ROW_KEYS = frozenset(f.name for f in fields(Row)) - {"id"}  # the file's keys are the fields
STAGE_KEYS = frozenset(f.name for f in fields(Stage)) - {"id"}


def owners(root: Path) -> frozenset[str]:
    """The owners a row may name: each noun owner in the owners map, and the workshop."""
    document = json.loads((root / "config/registries/owners-map.json").read_text("utf-8"))
    return frozenset({"workshop", *(row["owner"] for row in document["nouns"].values())})


def load(root: Path) -> Definition:
    """Read config/pipeline.toml once and refuse anything malformed, naming its stage or row."""
    try:
        data = (root / PATH).read_bytes()
        document = tomllib.loads(data.decode("utf-8"))
    except FileNotFoundError:
        raise DefinitionError("pipeline.definition_missing", f"{PATH} is absent") from None
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise DefinitionError("pipeline.definition_unreadable", f"{PATH}: {error}") from None
    return check(document, hashlib.sha256(data).hexdigest(), load_policy(root), owners(root))


def check(document: Any, sha256: str, policy: dict[str, Any], named: frozenset[str]) -> Definition:
    """Check a parsed definition against the test registry's lanes and sizes and the owners."""
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or set(document) != {"schema", "owner", "stages", "rows"}
        or not all(isinstance(document[key], dict) and document[key] for key in ("stages", "rows"))
    ):
        raise DefinitionError(
            "pipeline.schema_unknown", f"{PATH} holds schema = {SCHEMA!r}, owner, stages and rows"
        )
    rows = {name: _row(name, value, named) for name, value in document["rows"].items()}
    stages = {name: _stage(name, value, rows, policy) for name, value in document["stages"].items()}
    if len({stage.order for stage in stages.values()}) != len(stages):
        raise DefinitionError("pipeline.stage_invalid", "each stage has its own order")
    plugins = {(row.id, p) for row in rows.values() for p in row.plugins}
    if wrong := sorted(
        (r, p) for r, p in plugins if getattr(rows.get(p), "kind", "") != "pytest-plugin"
    ):
        raise DefinitionError(
            "pipeline.plugin_unknown",
            f"row {wrong[0][0]} loads {wrong[0][1]}, no pytest-plugin row",
        )
    used = {name for stage in stages.values() for name in stage.rows} | {p for _, p in plugins}
    if unused := sorted(set(rows) - used):
        raise DefinitionError(
            "pipeline.row_unused", f"row {unused[0]} is registered but no stage runs it"
        )
    return Definition(sha256, dict(sorted(stages.items(), key=lambda i: i[1].order)), rows)


def _row(name: str, value: Any, named: frozenset[str]) -> Row:
    def refuse(why: str) -> NoReturn:
        raise DefinitionError("pipeline.row_invalid", f"row {name}: {why}")

    if not isinstance(value, dict) or set(value) - ROW_KEYS:
        refuse(f"its keys are among {', '.join(sorted(ROW_KEYS))}")
    if not all(
        _listed(value[key]) for key in ("hosts", "run", "plugins", "public") if key in value
    ):
        refuse("its hosts, run, plugins and public are lists of non-empty strings")
    hosts, kind = tuple(value.get("hosts", ())), value.get("kind")
    if kind not in KINDS:
        refuse(f"kind is one of {', '.join(sorted(KINDS))}")
    if not value.get("reason") or value.get("owner") not in named:
        refuse("it needs a reason, and an owner from config/registries/owners-map.json or workshop")
    if not hosts or set(hosts) - HOSTS:
        refuse("hosts are workshop, public or both")
    if wrong := _kind_problem(kind, hosts, value):
        refuse(wrong)
    if bool(value.get("escape")) != bool(value.get("review")) or not _dated(
        value.get("review", "")
    ):
        refuse("a guard names its escape and its review date (YYYY-MM-DD) together")
    lists = {key: tuple(value.get(key, ())) for key in ("run", "plugins", "public")}
    return Row(**{**value, **lists, "id": name, "hosts": hosts})


def _stage(name: str, value: Any, rows: dict[str, Row], policy: dict[str, Any]) -> Stage:
    def refuse(why: str) -> NoReturn:
        raise DefinitionError("pipeline.stage_invalid", f"stage {name}: {why}")

    if not isinstance(value, dict) or set(value) - STAGE_KEYS:
        refuse(f"its keys are among {', '.join(sorted(STAGE_KEYS))}")
    if not all(_listed(value[key]) for key in ("rows", "sizes") if key in value):
        refuse("its rows and sizes are lists of non-empty strings")
    bound, names = value.get("bound_minutes"), tuple(value.get("rows", ()))
    sizes = tuple(value.get("sizes", ()))
    if type(value.get("order")) is not int or not value.get("reason"):
        refuse("it needs an integer order and a reason")
    if not isinstance(bound, int | float) or isinstance(bound, bool) or not 0 < bound < math.inf:
        refuse("bound_minutes is a positive number")
    if value.get("trigger") not in TRIGGERS or value.get("range") not in RANGES:
        refuse(f"trigger is one of {sorted(TRIGGERS)}, and range one of {sorted(RANGES)}")
    if value.get("lane", "merge") not in policy["lanes"] or set(sizes) - set(policy["sizes"]):
        refuse("its lane and sizes are registered in config/registries/test-sizes.json")
    if value.get("place", "gate") != "gate" or not names or len(set(names)) != len(names):
        refuse("place is gate when given, and it names each of its rows once")
    if missing := [row for row in names if row not in rows]:
        raise DefinitionError(
            "pipeline.row_unregistered",
            f"stage {name} names row {missing[0]}; register it under [rows] with its kind, "
            "owner, reason and hosts",
        )
    return Stage(**{**value, "id": name, "rows": names, "sizes": sizes})


def _kind_problem(kind: str, hosts: tuple[str, ...], value: dict[str, Any]) -> str:
    """What a row of this kind lacks: a command its run, a plugin its module, a CI job its name."""
    needs = {"command": "run", "pytest-plugin": "module", "ci-job": "job"}.get(kind)
    only = {"ci-job": ("public",), "workshop": ("workshop",)}.get(kind, hosts)
    if needs and not value.get(needs):
        return f"a {kind} row names its {needs}"
    return f"a {kind} row runs on {only[0]} only: set hosts = {list(only)}" if hosts != only else ""


def _listed(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _dated(value: str) -> bool:
    try:
        return not value or date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False
