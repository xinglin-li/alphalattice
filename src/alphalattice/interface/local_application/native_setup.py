"""Configure, bind and inspect a project's native Session for research; no hooks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from ctypes.util import find_library
from datetime import UTC, datetime
from functools import cache, partial
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import unquote, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from alphalattice.interface.local_application import client, workbench_assets
from alphalattice.interface.local_application.cli_contract import (
    BINDING_NOTICE,
    REQUEST_PROVENANCE,
    agent_session,
    command,
    command_table,
    refusal_words,
    worded,
)
from alphalattice.interface.local_application.failure_codes import safe_failure_code, setup_failure
from alphalattice.interface.local_application.native_bridge import (
    BINDING_DIRECTORY,
    BINDING_NAME,
    BINDING_RECORDS,
    HOSTS,
    PROJECT_DECLARATION_NAME,
    PROJECT_DECLARATION_SCHEMA,
    SPAWN_HOPS,
    USAGE_READINGS,
    WORKBENCH_BUILD,
    NativeBridgeError,
    NativeResearchBinding,
    agent_setup,
    attachment_preflight,
    autobind_root,
    binding_notice,
    clear_usage_checkpoints,
    codex_queue_readiness,
    command_readiness,
    declares_product,
    fresh_path,
    session_binding,
    session_project,
)
from alphalattice.interface.local_application.native_usage import codex_thread_spawn
from alphalattice.interface.local_application.read_views import emit_owner
from alphalattice.interface.local_application.retrieval_environment import LOCKED_SETUP_WORDS
from alphalattice.kernel.shared_kernel.bounded_memo import BoundedMemo
from alphalattice.kernel.shared_kernel.environment import (
    offline,
    package_versions,
    recorded_environment,
)
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PRODUCT_HOOK_MATCHER = "^alphalattice_.*$"
"""The matcher of the product's retired lifecycle hooks; configure removes groups with it."""
PRODUCT_HOOK_EVENTS = ("SubagentStart", "SubagentStop")
RESOURCE_ROOT = resolve_playpen_root(Path(__file__))
INSTALLED = not (RESOURCE_ROOT / "pyproject.toml").is_file()
ROOT = Path.cwd() if INSTALLED else RESOURCE_ROOT


def _installed_leg() -> dict[str, Any]:
    """Compare the active distribution with this checkout, naming unproved PATH ownership."""
    command = shutil.which("alphalattice", path=fresh_path())
    expected = None
    if (RESOURCE_ROOT / "pyproject.toml").is_file():
        expected = tomllib.loads((RESOURCE_ROOT / "pyproject.toml").read_text("utf-8"))["project"][
            "version"
        ]
    try:
        distribution = importlib.metadata.distribution("alphalattice")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = urlsplit(direct.get("url", ""))
        source = (
            Path(unquote(url.path).lstrip("/") if os.name == "nt" else unquote(url.path))
            if url.scheme == "file"
            else None
        )
        matches = (expected is None or distribution.version == expected) and (
            source is None or source.resolve() == RESOURCE_ROOT.resolve()
        )
        paired = command is not None and Path(command).parent == Path(sys.executable).parent
        return {
            "present": matches if paired else False if command is None or not matches else None,
            "command": command,
            "version": distribution.version,
            "checkout_version": expected,
            "source": None if source is None else str(source),
            "path_owner": "ACTIVE_ENVIRONMENT" if paired else "NOT_PROVED",
        }
    except importlib.metadata.PackageNotFoundError:
        return {"present": False, "command": command, "reason": "DISTRIBUTION_MISSING"}


def _host_facts(binding: NativeResearchBinding | None) -> dict[str, Any]:
    """Read existing Host facts when bound; an absent Host supplies no made-up measurements."""
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchClientError,
    )

    if binding is None:
        return {"scope": "LOCAL_PROCESS", "reason": "SESSION_NOT_BOUND"}
    try:
        client = LocalResearchClient(Path(binding.workspace), timeout=5)
        storage = client.request({"operation": "STORAGE_CAP_SHOW"})
        return {
            "scope": "WORKSPACE_HOST",
            "capacity": storage.get("capacity"),
            "recovery_headroom_bytes": storage.get("recovery_headroom_bytes"),
            "cpu": client.request({"operation": "CPU_BUDGET_SHOW"}),
            "network": client.request({"operation": "NETWORK_ACCESS"}),
            "queue": client.activity(limit=1).get("observer", {}).get("codex_queue"),
        }
    except LocalResearchClientError as error:
        return {"scope": "LOCAL_PROCESS", "reason": str(error).partition(":")[0]}


HOST_DEPENDENCIES = {
    "python": ("Run locked Python", "Use Python 3.12 and uv sync --locked; disclose it.", "AGENT"),
    "uv": (
        "Install the lock's dependencies",
        "Ask once to install uv, then the agent installs it; without it the locked "
        "environment cannot be repaired.",
        "PERSON",
    ),
    "installed_leg": (
        "Resolve the clean alphalattice command",
        "Run uv sync --locked and the documented tool install; disclose it. "
        "Check an unproved PATH owner first. A first use opens nothing until a fresh shell "
        "runs `alphalattice storage cap --help`; off PATH, "
        "`uv tool update-shell` and a restart are the person's.",
        "AGENT",
    ),
    "workbench_build": (
        "Serve the Workbench",
        f"Run the documented Workbench build (`{WORKBENCH_BUILD}`) from the checkout root, "
        "disclosing it; serve refuses until the built files match the sources.",
        "AGENT",
    ),
    "retrieval_runtime": (
        "Read Evidence with its retrieval recipe",
        LOCKED_SETUP_WORDS + " Then use install_retrieval_pack "
        "--install hybrid-v2-minilm. Keep packs outside "
        "the workspace; network acquisition still needs its permission.",
        "AGENT",
    ),
    "background_notices": (
        "Continue after background work ends",
        "Ask once to install the Codex CLI outside the lock; the agent installs it "
        "and checks codex queue --help, then sends the same request again. "
        "Until available, WAIT_IN_THE_TURN occupies this turn until work ends.",
        "PERSON",
    ),
    "disk": (
        "Keep admitted data within disk and storage capacity",
        f"Read `{command({'operation': 'STORAGE_CAP_SHOW'})}` "
        f"and `{command({'operation': 'STORAGE_PLAN'})}`; "
        "ask once before cleanup or changing the cap. "
        "Bind/start the Host to measure its cap; "
        "free space cannot prove it fits.",
        "PERSON",
    ),
    "cpu": (
        "Budget Task workers",
        f"Read `{command({'operation': 'CPU_BUDGET_SHOW'})}`; "
        "the agent sets workers and tasks_waiting within "
        "the reported machine budget.",
        "AGENT",
    ),
    "network": (
        "Know whether acquisition is admitted",
        f"Read `{command({'operation': 'NETWORK_ACCESS'})}`; "
        "relay the person's permission before opening "
        "the network. A closed network remains a valid offline setting.",
        "PERSON",
    ),
    "system_libraries": (
        "Load the system libraries locked packages need",
        "Ask once to install the packages its facts name (`install`), outside the lock, then "
        "run doctor again; never install them unasked.",
        "PERSON",
    ),
    "user_layer": (
        "Local user tuning",
        "Use maintenance Skill; disclose the optional layer. Over a bound or due, review it.",
        "AGENT",
    ),
    "user_backup": (
        "Recover the person's layer before an upgrade",
        "Back up the optional user layer before upgrading, using the maintenance "
        "Skill; disclose it. A missing backup does not refuse research.",
        "AGENT",
    ),
}


SYSTEM_LIBRARIES: dict[str, dict[str, Any]] = {
    # The libraries a locked package loads that no lock can ship: the person installs them once.
    "darwin": {
        "install": "brew install",
        "libraries": (
            {
                "package": "libomp",
                "without_it": "LightGBM cannot load, so no model trains or scores",
                "names": ("omp",),
                "paths": (
                    "/opt/homebrew/opt/libomp/lib/libomp.dylib",
                    "/usr/local/opt/libomp/lib/libomp.dylib",
                ),
            },
        ),
    },
    "linux": {
        "install": "sudo apt install",
        "libraries": (
            {
                "package": "libatomic1",
                "without_it": "the locked Node cannot start, so the Workbench cannot be built",
                "names": ("atomic",),
                "paths": (),
            },
            {
                "package": "libgomp1",
                "without_it": "LightGBM cannot load, so no model trains or scores",
                "names": ("gomp",),
                "paths": (),
            },
        ),
    },
}


def _system_library() -> dict[str, Any]:
    """This OS's libraries that the lock cannot ship, found or not; nothing is installed here."""
    need = next((v for k, v in SYSTEM_LIBRARIES.items() if sys.platform.startswith(k)), None)
    if need is None:
        return {"present": True, "needed": False}
    found, missing = {}, []
    for library in need["libraries"]:
        path = next((p for p in library["paths"] if Path(p).is_file()), None) or next(
            filter(None, map(find_library, library["names"])), None
        )
        if path:
            found[library["package"]] = path
        else:
            missing.append({key: library[key] for key in ("package", "without_it")})
    install = " ".join([need["install"], *(m["package"] for m in missing)])
    return {"present": not missing, "found": found, "missing": missing, "install": install}


USER_LAYER_BOUNDS: Final = {
    "long_term": {"index_lines": 50, "memories": 60},
    "short_term": {"workspace_notes": 20},
}
"""The user layer's two tiers and their bounds, as the maintenance Skill states them."""
MEMORY_DOMAINS: Final = ("operating", "person", "method", "strategy", "state")
"""A memory's `domain` (as `method:<card>`, `strategy:<package>`), each with its review moment."""
REVIEW_DAYS: Final = 30
"""A method memory verified longer ago than this is due for review."""
REVIEW_FIRST: Final = (
    "The person's user layer needs review first ({subject}): start alphalattice_maintainer in "
    "CLEAN mode, re-verifying, revising, merging or deleting each item by the maintenance Skill's "
    "rules within its bounds; then record."
)
_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
type _Fields = dict[str, str]
type _Layer = tuple[dict[str, tuple[_Fields, str]], _Fields, dict[str, _Fields]]


def user_layer(project: Path, workspace: Path | None = None, host: Any = None) -> dict[str, Any]:
    """The person's layer in two tiers and five domains against its bounds, and what is due.

    Each long-term item is due at its domain's moment (`_stale`); a `state` memory and the
    workspace's notes once they name a Goal or Task the Host does not show open.

    Args:
        project: The agent project holding `.alphalattice/user/`.
        workspace: The served workspace whose `agent-notes/` count, when known.
        host: Its Host client, read for the open Goals and Tasks and each package's hash; None
            checks neither.

    Returns:
        Counts by tier and domain, `over`, `due` with reasons, and the review's `next_action`
        and `items`.
    """
    user = project / ".alphalattice/user"
    index, layer = user / "memory/MEMORY.md", _tiers(user, workspace)
    memories, notes, additions = layer
    domains = {name: front.get("domain", "") for name, (front, _) in memories.items()}
    short = {name: memories[name][1] for name, domain in domains.items() if domain == "state"}
    lines = len(index.read_text("utf-8").splitlines()) if index.is_file() else 0
    counts = {
        "long_term": {"index_lines": lines, "memories": len(memories) - len(short)},
        "short_term": {"state_memories": len(short), "workspace_notes": len(notes)},
        "domains": dict(Counter(domain.partition(":")[0] or "none" for domain in domains.values())),
    }
    counts["long_term"]["additions"] = len(additions)
    over = [k for tier, b in USER_LAYER_BOUNDS.items() for k in b if counts[tier][k] > b[k]]
    due = _due(layer, short, host)
    items = ", ".join([*over, *(item["path"] for item in due)])
    review = {"next_action": "REVIEW_THE_USER_LAYER", "items": items} if items else {}
    facts = {"counts": counts, "bounds": USER_LAYER_BOUNDS, "over": over, "due": due}
    return {"exists": user.is_dir(), **facts, **review}


def _due(layer: _Layer, short: _Fields, host: Any) -> list[dict[str, str]]:
    """Each item due and why: long-term at its domain's moment (`_stale`), short-term once it
    names a Goal or Task the Host does not show open."""
    memories, notes, additions = layer
    version, open_ids = package_versions(["alphalattice"])[0][1], _open_ids(host)
    package = None if host is None else cache(partial(_package_hash, host))
    reasons = {n: _stale(f, version, package) for n, (f, _) in memories.items() if n not in short}
    for name, front in additions.items():
        reasons[name] = _stale({"domain": "operating", **front}, version, package)
    for name, text in (short | notes).items():
        closed = [i for i in _ID.findall(text) if open_ids is not None and i not in open_ids]
        reasons[name] = f"points_to_closed:{closed[0]}" if closed else None
    return [{"path": name, "reason": why} for name, why in sorted(reasons.items()) if why]


def _tiers(user: Path, workspace: Path | None) -> _Layer:
    """The memories, the workspace's notes and the additions, each read once."""
    index = user / "memory/MEMORY.md"
    found = {f"memory/{p.name}": _read(p) for p in (user / "memory").glob("*.md") if p != index}
    memories = {name: (front, text) for name, (front, text) in found.items() if front is not None}
    listed = (workspace / "agent-notes").glob("*.md") if workspace else ()
    notes = {f"agent-notes/{p.name}": p.read_text("utf-8") for p in listed if p.name != "INDEX.md"}
    added = (user / "guide.md", *user.glob("cards/*.md"), *user.glob("skills/*.md"))
    additions = {p.relative_to(user).as_posix(): _read(p)[0] or {} for p in added if p.is_file()}
    return memories, notes, additions


def record_offer(answer: dict[str, Any], workspace: Path) -> None:
    """A sealing answer's record offer lists what the person's layer needs reviewed first.

    Args:
        answer: The sealing answer, its detail changed in place; otherwise it stands.
        workspace: Its workspace, whose notes count; the person's layer is the nearest up from
            here.
    """
    here = Path.cwd()
    project = next((p for p in (here, *here.parents) if (p / ".alphalattice/user").is_dir()), here)
    if items := user_layer(project, workspace, _client(workspace)).get("items"):
        review = str(worded(REVIEW_FIRST)).replace("{subject}", items)
        answer["detail"] = f"{review} {answer.get('detail', '')}".rstrip()


def _client(workspace: Path | None) -> Any:
    """The workspace's Host client; None with no Host serving it."""
    from alphalattice.interface.local_application import client

    with suppress(client.LocalResearchClientError, OSError):
        return client.LocalResearchClient(workspace) if workspace else None
    return None


def _open_ids(host: Any) -> frozenset[str] | None:
    """The Goals and Tasks the Host shows open; None with no Host to ask."""
    from alphalattice.interface.local_application.client import LocalResearchClientError

    with suppress(LocalResearchClientError, OSError, KeyError, TypeError, AttributeError):
        goals = host.request({"operation": "GOAL_LIST", "history_limit": 50})["goals"]
        tasks = host.request({"operation": "TASKS"})["tasks"]
        shown = {str(row["goal_id"]) for row in goals if row.get("state") == "OPEN"}
        return frozenset(shown | {str(row["task_id"]) for row in tasks})
    return None


def _package_hash(host: Any, package: str) -> str | None:
    """An installed package's hash, as its controls read it; None when none is installed."""
    from alphalattice.interface.local_application.client import LocalResearchClientError

    with suppress(LocalResearchClientError, OSError, KeyError, TypeError):
        return str(
            host.request({"operation": "CONTROLS", "strategy_package_id": package})["package_hash"]
        )
    return None


def _read(path: Path) -> tuple[_Fields | None, str]:
    """A file's frontmatter fields (None without them) and its text, read once."""
    text = path.read_text("utf-8", errors="replace")
    head, closed, _ = text.removeprefix("---\n").partition("\n---")
    if not text.startswith("---\n") or not closed:
        return None, text
    pairs = (line.partition(":") for line in head.splitlines())
    return {key.strip(): value.strip() for key, _, value in pairs}, text


def _stale(front: _Fields, version: str, package: Callable[[str], str | None] | None) -> str | None:
    """Why a long-term item is due, at its domain's moment: an operating or person one at another
    version, a method one also past `REVIEW_DAYS`, a strategy one once its package's hash moved;
    one with no domain or no `verified: <YYYY-MM-DD> at <version or hash>` line, always."""
    kind, _, subject = front.get("domain", "").partition(":")
    when, _, at = front.get("verified", "").partition(" at ")
    if kind not in MEMORY_DOMAINS:
        return "unclassified"
    try:
        day = datetime.strptime(when.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "unverified"
    if kind == "strategy":
        return "package_changed" if package is not None and package(subject) != at.strip() else None
    if at.strip() != version:
        return "verified_at_another_version"
    old = kind == "method" and (datetime.now(UTC).date() - day).days > REVIEW_DAYS
    return "verified_over_30_days" if old else None


def host_readiness(
    project: Path, host: str, binding: NativeResearchBinding | None
) -> list[dict[str, Any]]:
    """One advisory dependency list: the measured facts, what they enable and who fills them."""
    from alphalattice.kernel.knowledge.model_store import default_store_root, recipe_readiness

    facts = _host_facts(binding)
    # What the agent fills is read from the first use's own source, so the two never disagree.
    fillable = {key: dict(item) for key, item in agent_setup().items()}
    packs = recipe_readiness(default_store_root(), "hybrid-v2-minilm")
    queue = (
        {"present": True, "scope": "HOST_FEATURE"}
        if host == "claude-code"
        else {
            "scope": facts["scope"],
            **(facts.get("queue") or {"present": None, "reason": "HOST_CHECK_NOT_REPORTED"}),
        }
        if facts["scope"] == "WORKSPACE_HOST"
        else {**codex_queue_readiness(), "scope": "LOCAL_PROCESS"}
    )
    capacity = facts.get("capacity")
    headroom = facts.get("recovery_headroom_bytes")
    free = shutil.disk_usage(project).free
    disk_ready = (
        None
        if capacity is None or headroom is None
        else (
            capacity["measured_data_bytes"] <= capacity["cap_bytes"]
            and capacity["free_disk_bytes"] >= headroom
        )
    )
    served = binding.workspace if binding else None
    user = project / ".alphalattice/user"
    backups = sorted(path.name for path in (user / "backups").glob("*") if path.is_dir())
    python_version = str(recorded_environment()["python"])
    specifications = (
        (
            "python",
            python_version.split(".")[:2] == ["3", "12"],
            {"version": python_version, "interpreter": sys.executable},
        ),
        ("uv", (uv := command_readiness([os.environ.get("UV", "uv"), "--version"]))["present"], uv),
        (
            "installed_leg",
            (tool := fillable["installed_leg"]).pop("present"),
            {**{k: v for k, v in _installed_leg().items() if k != "present"}, **tool},
        ),
        ("workbench_build", workbench_assets.workbench_current(), {"command": WORKBENCH_BUILD}),
        (
            "retrieval_runtime",
            (retrieval := fillable["retrieval_runtime"]).pop("present"),
            {**retrieval, "packs": packs},
        ),
        (
            "background_notices",
            queue["present"],
            {"host": host, **queue, "delivery_proved": False},
        ),
        (
            "disk",
            disk_ready,
            {"free_disk_bytes": free, "capacity": capacity, "capacity_scope": facts["scope"]},
        ),
        (
            "cpu",
            (os.cpu_count() or 0) > 0,
            {
                "logical_cores": os.cpu_count(),
                "host": facts.get("cpu"),
                "host_scope": facts["scope"],
            },
        ),
        (
            "network",
            None if facts.get("network") is None else True,
            {
                "scope": facts["scope"],
                "setting": facts.get("network"),
                "operator_offline": offline(),
            },
        ),
        ("system_libraries", (system := _system_library())["present"], system),
        (
            "user_layer",
            "items" not in (layer := user_layer(project, served, _client(served))),
            {"path": str(user), **layer},
        ),
        ("user_backup", bool(backups), {"latest": backups[-1] if backups else None}),
    )
    guidance = dict(HOST_DEPENDENCIES)
    if host == "claude-code":
        guidance["background_notices"] = (
            "Continue after background work ends",
            "Use background Bash completion notices, without another CLI.",
            "AGENT",
        )
    return [
        {
            "key": key,
            "present": present,
            "what_it_enables": guidance[key][0],
            "how_to_fill": guidance[key][1],
            # Who fills an item matters only while it is missing.
            **({} if present is True else {"who_decides": guidance[key][2]}),
            "facts": {"scope": "LOCAL_PROCESS", **measured},
        }
        for key, present, measured in specifications
    ]


def _report_operations(activity: dict[str, Any], operations: set[str]) -> list[dict[str, Any]]:
    """Allow only route tokens and reference kinds out of the activity projection."""
    rows = sorted(
        (item for group in activity.get("groups", []) for item in group["items"]),
        key=lambda item: item["at"],
        reverse=True,
    )[:8]
    recent = []
    for row in rows:
        if row.get("operation") not in operations:
            continue
        item = {"operation": row["operation"]}
        for field, pattern in (
            ("status", r"[A-Z][A-Z_]{0,63}"),
            ("next_action", r"[A-Z][A-Z_]{0,95}"),
        ):
            value = str(row.get(field) or "").partition(":")[0]
            item[field] = value if re.fullmatch(pattern, value) else None
        item.update(
            failure_code=safe_failure_code(str(row.get("failure_code") or "").partition(":")[0]),
            offered_requests=row.get("offered_requests"),
            offers_recorded=row.get("offered_requests") is not None,
            identifier_kinds=row.get("identifier_kinds", []),
        )
        recent.append(item)
    return recent


def _report_readiness(
    project: Path, host: str, binding: NativeResearchBinding | None
) -> dict[str, Any]:
    """Retain readiness decisions without paths, measurements, repair text or raw facts."""
    try:
        items = [
            {key: item[key] for key in ("key", "present", "who_decides")}
            for item in host_readiness(project, host, binding)
        ]
        return {"status": "AVAILABLE", "items": items}
    except Exception as error:
        code = str(setup_failure(error)["failure_code"]).partition(":")[0]
        return {"status": "UNAVAILABLE", "failure_code": code}


def problem_report(
    workspace: Path,
    *,
    project: Path,
    sentence: str | None = None,
    expected_route: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Project existing readiness and activity into a local, reviewable public report."""
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchClientError,
    )

    operations = {op for group in command_table()["commands"].values() for op in group}
    if any(op not in operations for op in expected_route):
        raise NativeBridgeError("local_client.report_route_not_admitted")
    if sentence is not None and sentence not in {
        f"Report the {op} problem to the developers." for op in operations
    }:
        raise NativeBridgeError("local_client.report_sentence_not_admitted")
    session = agent_session(os.environ)
    binding = None
    try:
        found = session_binding(project, workspace, session)
        if found is not None and Path(found[1].workspace).resolve() == workspace.resolve():
            project, binding = found
    except (NativeBridgeError, OSError):
        pass  # A broken binding must not prevent reporting it; doctor states its readiness.
    host = session[0] if session else getattr(binding, "host", None)
    doctor = _report_readiness(project, host or "codex", binding)
    try:
        activity = LocalResearchClient(workspace, timeout=5).activity_recent(limit=8)
    except LocalResearchClientError as error:
        activity = {
            "status": "UNAVAILABLE",
            "failure_code": safe_failure_code(str(error).partition(":")[0]),
        }
    recent = _report_operations(activity, operations)
    last = recent[0]["operation"] if recent else "WORKSPACE_SHOW"
    report = {
        "status": "PROBLEM_REPORT",
        **refusal_words("local_client.problem_report_ready"),
        "included": [
            "versions",
            "host_kind",
            "os_family",
            "doctor_readiness",
            "operation_codes",
            "offer_names",
            "identifier_kinds",
            "scenario_stub",
        ],
        "omitted": [
            "workspace_paths",
            "holdings",
            "figures",
            "file_contents",
            "conversation",
            "identifier_values",
        ],
        "version": dict(package_versions(("alphalattice",)))["alphalattice"],
        "host": {
            "kind": host,
            "version": None
            if host is None
            else command_readiness(
                [{"claude-code": "claude", "codex": "codex"}[host], "--version"], timeout=5
            ).get("version"),
            "version_scope": "CLI_PROBE_NOT_DESKTOP",
        },
        "os": platform.system(),
        "doctor": doctor,
        "activity": {"status": activity["status"], "failure_code": activity.get("failure_code")},
        "operations": recent,
        "scenario": {
            "id": "reported-problem",
            "sentence": sentence or f"Report the {last} problem to the developers.",
            "now": datetime.now(UTC).isoformat(),
            "injected_faults": [],
            "expected_route": [[op] for op in expected_route],
            "decider": "person",
            "forbidden_acts": ["send_without_consent"],
        },
        "route_status": "SUPPLIED" if expected_route else "UNTRIAGED",
        "observed_route": [
            {key: item[key] for key in ("operation", "status", "failure_code")} for item in recent
        ],
    }
    report["issue_url"] = "https://github.com/xinglin-li/alphalattice/issues/new?" + urlencode(
        {
            "template": "bug_report.yml",
            "labels": "bug",
            "title": f"[Bug]: {last}",
            "report": json.dumps(report, ensure_ascii=False, separators=(",", ":")),
        }
    )
    return report


def _absolute_project(value: Path) -> Path:
    """Validate a bounded absolute project path without inspecting its contents.

    Args:
        value: Project path supplied at the typed native session door.

    Returns:
        The unchanged path, ready for the Host's independent admission.

    Raises:
        ValueError: The path is relative, unbounded or contains control characters.
    """
    if (
        not value.is_absolute()
        or len(str(value)) > 512
        or any(ord(character) < 32 for character in str(value))
    ):
        raise ValueError("native_bridge.project_path_invalid")
    return value


class NativeSessionBindRequest(BaseModel):  # type: ignore[misc]
    """A binding request names a project and usage; its session is the request's provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: Path
    usage: Literal["read", "off"] = "read"

    @field_validator("project")  # type: ignore[untyped-decorator]
    @classmethod
    def absolute_project(cls, value: Path) -> Path:
        """Refuse relative or unbounded paths before the Host inspects its own project."""
        return _absolute_project(value)


class NativeEventDeliveryRequest(BaseModel):  # type: ignore[misc]
    """A milestone's request that the Host read the caller's Session usage; it carries none."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: Path
    event: dict[str, Any]

    @field_validator("project")  # type: ignore[untyped-decorator]
    @classmethod
    def absolute_project(cls, value: Path) -> Path:
        """Apply the binding door's path contract to a milestone's reading request."""
        return _absolute_project(value)


def admitted_session_project(workspace: Path, requested: Path, host: str) -> Path:
    """Admit a served workspace's configured project independently of the client's word.

    A workspace inside its agent project (``workspaces/`` of a checkout) finds that project
    up from itself, and the client's project must be it. A workspace kept elsewhere -- a
    second drive, an installed wheel's own folder -- finds none; the client's project is then
    admitted only when it holds this host's configure-owned declaration itself (FLOW-1).

    Args:
        workspace: The Host's own served workspace.
        requested: The client's nominated absolute project path.
        host: The admitted native host identifier.

    Returns:
        The admitted project.

    Raises:
        NativeBridgeError: The project differs, aliases a path, declares nothing or has
            unsafe declarations.
    """
    if requested.is_symlink() or requested.absolute() != requested.resolve():
        raise NativeBridgeError("native_bridge.project_path_invalid")
    try:
        project = session_project(workspace.resolve(), host)
    except NativeBridgeError as error:
        if str(error) != "native_bridge.project_declaration_missing":
            raise
        if not declares_product(requested.resolve(), host):
            raise
        project = requested.resolve()
    if requested.resolve() != project.resolve():
        raise NativeBridgeError("native_bridge.project_mismatch")
    declaration = project / (
        ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    )
    if any(path.is_symlink() for path in (project, declaration.parent, declaration)):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    return project.resolve()


def files_unavailable(
    error: OSError, *, project: Path, workspace: Path | None = None
) -> dict[str, Any]:
    """Name only an admitted local path and its category, never the exception's prose.

    Args:
        error: The failed local file operation, used only for its class and filename.
        project: The admitted absolute project root bounding safe path disclosure.
        workspace: The served workspace, when separately available.

    Returns:
        A refusal with the failure category, admitted path when known and remedy.
    """
    candidate = Path(error.filename) if isinstance(error.filename, str) else None
    safe_path: Path | None = None
    category = "PROJECT_FILES"
    if candidate is not None:
        candidate = candidate.absolute()
        if candidate.is_relative_to(project):
            safe_path = candidate
            category = (
                "PROJECT_BINDING"
                if candidate == project / ".codex" / BINDING_NAME
                else "HOST_DECLARATIONS"
                if candidate.is_relative_to(project / ".codex")
                or candidate.is_relative_to(project / ".claude")
                else "PROJECT_FILES"
            )
        elif workspace is not None and candidate.is_relative_to(workspace):
            safe_path, category = candidate, "WORKSPACE"
    code = f"native_bridge.files_unavailable:{type(error).__name__}"
    return {
        "status": "REFUSED",
        "failure_code": code,
        "reason": code,
        "path_category": category,
        **({"path": str(safe_path)} if safe_path is not None else {}),
        **refusal_words(code),
    }


def _declarations() -> Path:
    return (
        RESOURCE_ROOT
        if INSTALLED
        and not (ROOT / ".codex/config.toml").is_file()
        and not (ROOT / ".claude/settings.json").is_file()
        else ROOT
    )


def _install_declarations(host: str) -> None:
    """Copy shipped guidance unchanged; default host declarations register no product hooks."""
    manifest = json.loads(
        (RESOURCE_ROOT / "config/release/runtime-resources.json").read_text(encoding="utf-8")
    )
    host_directory = ".claude" if host == "claude-code" else ".codex"
    inputs = ["AGENTS.md"]
    if host == "claude-code":
        inputs.append("CLAUDE.md")
    for directory in manifest["directories"]:
        if directory.startswith(host_directory + "/") or directory.startswith(".agents/"):
            inputs.extend(
                path.relative_to(RESOURCE_ROOT).as_posix()
                for path in (RESOURCE_ROOT / directory).rglob("*")
                if path.is_file()
            )
    declaration = ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    inputs.append(declaration)
    documents: dict[Path, bytes] = {}
    for relative in sorted(set(inputs)):
        data = (RESOURCE_ROOT / relative).read_bytes()
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        if relative == declaration and path.exists():
            data = _merge_installed_declaration(data, path.read_bytes(), host)
        elif path.exists() and path.read_bytes() != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
        documents[path] = data
    for path, data in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        elif path == ROOT / declaration and path.read_bytes() != data:
            path.write_bytes(data)


def _merge_installed_declaration(source: bytes, existing: bytes, host: str) -> bytes:
    """Admit product roles without replacing unrelated configuration or existing hooks."""
    if host == "claude-code":
        if not isinstance(json.loads(existing), dict):
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        return existing
    expected = tomllib.loads(source.decode()).get("agents", {})
    current = tomllib.loads(existing.decode()).get("agents", {})
    if not isinstance(current, dict):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    for name, profile in expected.items():
        if name in current and current[name] != profile:
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
    additions = []
    for section in re.split(r"(?=^\[agents\.)", source.decode(), flags=re.MULTILINE):
        if section.startswith("[agents."):
            name = section.split("]", 1)[0].removeprefix("[agents.")
            if name not in current:
                additions.append(section)
    return existing if not additions else existing + b"\n" + "".join(additions).encode()


def strip_product_hooks(project: Path, host: str) -> int:
    """Remove the product's own retired lifecycle hook groups from a project it configures.

    An earlier configure could add ``SubagentStart`` and ``SubagentStop`` groups matching
    `PRODUCT_HOOK_MATCHER`; they called a command that no longer exists and kept the host's
    trust prompts. Only groups with exactly that matcher are removed; every other hook and
    setting stays as written.

    Args:
        project: The agent project being configured.
        host: ``claude-code`` or ``codex``.

    Returns:
        How many product hook groups were removed.

    Raises:
        NativeBridgeError: The declaration is a link or not the expected document.
    """
    path = project / (".claude/settings.json" if host == "claude-code" else ".codex/config.toml")
    if path.is_symlink() or path.parent.is_symlink():
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    if not path.is_file():
        return 0
    removed = 0
    if host == "claude-code":
        document = json.loads(path.read_text(encoding="utf-8"))
        hooks = document.get("hooks") if isinstance(document, dict) else None
        if not isinstance(hooks, dict):
            return 0
        for event in PRODUCT_HOOK_EVENTS:
            groups = hooks.get(event)
            if not isinstance(groups, list):
                continue
            kept = [
                group
                for group in groups
                if not (isinstance(group, dict) and group.get("matcher") == PRODUCT_HOOK_MATCHER)
            ]
            removed += len(groups) - len(kept)
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)
        if removed:
            if not hooks:
                document.pop("hooks", None)
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        return removed
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    kept_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        table = next(
            (name for name in PRODUCT_HOOK_EVENTS if line.strip() == f"[[hooks.{name}]]"), None
        )
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if table is not None and following == f"matcher = {json.dumps(PRODUCT_HOOK_MATCHER)}":
            removed += 1
            index += 2
            # The group's own command tables belong to it until the next other table.
            while index < len(lines) and (
                not lines[index].strip().startswith("[")
                or lines[index].strip() == f"[[hooks.{table}.hooks]]"
            ):
                index += 1
            while kept_lines and not kept_lines[-1].strip():
                kept_lines.pop()
            if kept_lines and index < len(lines):
                kept_lines.append("")
            continue
        kept_lines.append(line)
        index += 1
    if removed:
        result = "\n".join(kept_lines).rstrip("\n") + "\n"
        tomllib.loads(result)
        path.write_text(result, encoding="utf-8", newline="\n")
    return removed


def _roles(host: str = "codex", root: Path | None = None) -> list[str]:
    declarations = _declarations() if root is None else root
    if host == "claude-code":
        # The Claude host's own cards (scripts/materialize_claude_host.py derives them), with
        # the medium-effort card of each evidence specialist; no Codex file is read.
        return sorted(
            path.stem for path in (declarations / ".claude/agents").glob("alphalattice_*.md")
        )
    with (declarations / ".codex/config.toml").open("rb") as stream:
        agents = tomllib.load(stream).get("agents", {})
    return sorted(name for name in agents if name.startswith("alphalattice_"))


def continue_here(project: Path, host: str) -> dict[str, Any]:
    """How the session that installed AlphaLattice continues the research (STOPS-1).

    A host loads AGENTS.md, the research Skill and the specialist cards on its own only in a
    session started in the configured folder. The installing session reads the first two by
    path and starts each specialist as a general subagent from its card, so a card's tool
    limits hold there by instruction; `open_session` is the command for a session where the
    host enforces them, offered as an option and never a step.
    """
    folder = project.resolve()
    claude = host == "claude-code"
    command = f'cd "{folder}"; claude' if claude else f'codex -C "{folder}"'
    skill = folder / (".claude" if claude else ".agents") / "skills/alphalattice-research"
    return {
        "read": [str(folder / "AGENTS.md"), str(skill / "SKILL.md")],
        "specialists": str(folder / (".claude/agents" if claude else ".codex/agents")),
        "open_session": command,
        "disclosure": (
            "AlphaLattice is set up and the research continues in this session. Its specialists "
            "start here from their cards as general subagents, so their tool limits hold by "
            f"instruction; for limits the host enforces, open a session with `{command}`."
        ),
    }


def declare_project(project: Path, host: str) -> None:
    """Write exact host-local project metadata; it grants neither trust nor native authorship."""
    if host not in HOSTS:
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    _create_or_match(
        PROJECT_DECLARATION_NAME,
        {"schema": PROJECT_DECLARATION_SCHEMA, "host": host},
        project,
        directory=".claude" if host == "claude-code" else ".codex",
    )


def bind_session(
    project: Path,
    *,
    host: str,
    session_id: str,
    workspace: Path,
    usage: str = "read",
    roles: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Bind this exact host/Session without replacing another Session's workspace (V568).

    The CLI and independent usage observer find this exact binding within the project.

    Args:
        project: The agent project, a checkout or a configured folder.
        host: The host running the session.
        session_id: The session, as its host names it.
        workspace: The workspace's folder.
        usage: ``off`` prevents native Session usage reads.
        roles: The admitted specialist roles; the project's own cards when omitted.

    Returns:
        The binding as written, with its project.

    Raises:
        NativeBridgeError: For a missing workspace, unsafe records, or a changed scope
            already bound to this exact host/Session.
    """
    if host not in HOSTS or usage not in {"read", "off"}:
        raise NativeBridgeError("native_bridge.binding_invalid")
    if (
        workspace.is_symlink()
        or (os.name == "nt" and workspace.is_junction())
        or workspace.absolute() != workspace.resolve()
    ):
        raise NativeBridgeError("native_bridge.binding_path_invalid")
    if not workspace.is_dir():
        raise NativeBridgeError("native_bridge.workspace_missing")
    folder = str(workspace.resolve())
    document = {
        "session_id": session_id,
        "workspace": folder,
        "roles": list(roles) if roles is not None else _roles(host, project),
        "host": host,
        # Named only when off, so a binding written before the switch reads the same.
        **({"usage": "OFF"} if usage == "off" else {}),
    }
    selected = (host, session_id)
    existing = NativeResearchBinding.read(project, session=selected)
    if existing is None:
        document["observation_started_at"] = datetime.now(UTC).isoformat()
    else:
        if (
            existing.session_id != session_id
            or existing.workspace.resolve() != workspace.resolve()
            or existing.host != host
            or existing.roles != tuple(document["roles"])
            or existing.usage != ("OFF" if usage == "off" else "READ")
        ):
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
        if existing.observation_started_at is not None:
            document["observation_started_at"] = existing.observation_started_at.isoformat()
    NativeResearchBinding.from_document(document)
    if existing is None:
        bindings, refusals = NativeResearchBinding.binding_entries(project)
        if len(bindings) + len(refusals) >= BINDING_RECORDS:
            raise NativeBridgeError("native_bridge.binding_limit_exceeded")
        legacy = project / ".codex" / BINDING_NAME
        # The first binding retains the established compatibility path. A later
        # Session gets its own slot; nothing migrates, replaces or detaches the first.
        if not legacy.exists() and not legacy.is_symlink() and not bindings and not refusals:
            name, directory = BINDING_NAME, ".codex"
        else:
            name = NativeResearchBinding.slot_path(project, selected).name
            directory = f".codex/{BINDING_DIRECTORY}"
        try:
            _create_or_match(name, document, project, directory=directory)
        except NativeBridgeError as error:
            # A concurrent bind of this exact Session may already have admitted the
            # same scope. Keep its checkpoint rather than generating a new one.
            if str(error) != "native_bridge.existing_configuration_differs":
                raise
            winner = NativeResearchBinding.read(project, session=selected)
            if winner is None and name == BINDING_NAME:
                # Two distinct Sessions can both see the project's first empty
                # record. The loser gets its own slot, preserving the winner's file.
                try:
                    _create_or_match(
                        NativeResearchBinding.slot_path(project, selected).name,
                        document,
                        project,
                        directory=f".codex/{BINDING_DIRECTORY}",
                    )
                except NativeBridgeError as slot_error:
                    if str(slot_error) != "native_bridge.existing_configuration_differs":
                        raise
                winner = NativeResearchBinding.read(project, session=selected)
            if winner is None or (
                winner.workspace.resolve() != workspace.resolve()
                or winner.roles != tuple(document["roles"])
                or winner.usage != ("OFF" if usage == "off" else "READ")
            ):
                raise
    binding = NativeResearchBinding.read(project, session=selected)
    preflight = attachment_preflight(project, binding)
    return {
        "status": "BOUND",
        "session_id": session_id,
        "host": host,
        "workspace": folder,
        "project": str(project),
        "attachment_preflight": preflight,
        "detail": preflight["detail"],
        "next_action": preflight["next_action"],
    }


def autobind_session(
    workspace: Path, host: str, session_id: str, *, project: Path | None = None
) -> tuple[Path, str]:
    """Bind the Session a request names to this workspace, with no step and no configure.

    Under the person's hands-off rule a Session is bound by working: the Host writes the
    binding in the workspace's own folder, with the product's shipped specialist roles and
    reading on (the workspace switch still governs reading). A Codex child binds its lead,
    found up its own rollout's spawn chain; a Claude Code subagent carries its lead's id.

    Args:
        workspace: The workspace the Session works on.
        host: Its host.
        session_id: The Session the request names.
        project: The configured project above the workspace, where the binding is kept with
            its own cards so commands from it may omit the workspace; else the
            workspace's own folder with the shipped roles.

    Returns:
        The binding folder and the Session bound.
    """
    lead = session_id
    if host == "codex":
        for _ in range(SPAWN_HOPS):
            spawn = codex_thread_spawn(lead)
            if spawn is None:
                break
            lead = spawn.parent_thread_id
    if project is not None:
        try:
            bind_session(project, host=host, session_id=lead, workspace=workspace)
            return project, lead
        except OSError as error:
            if not _sandboxed_binding(error, project):
                raise
    root = autobind_root(workspace)
    bind_session(root, host=host, session_id=lead, workspace=workspace, roles=tuple(_roles(host)))
    if project is not None:  # kept here because the agent's sandbox refused the project
        _BINDING_MEMO[(host, session_id, workspace.resolve())] = True
        BINDING_NOTICE.set(binding_notice("BOUND_IN_WORKSPACE"))
    return root, lead


_BINDING_MEMO: BoundedMemo[tuple[str, str, Path], bool] = BoundedMemo()
"""Checked Sessions and whether their durable workspace binding needs its notice."""


def configure_binding_memo(*, capacity: int | None = None) -> None:
    """Bound reminder entries; the default is machine-derived and a miss rereads bindings."""
    global _BINDING_MEMO
    _BINDING_MEMO = BoundedMemo(capacity=capacity)


def _sandboxed_binding(error: OSError, project: Path) -> bool:
    """Whether a permission refusal names the project's own binding.

    That is its first file, the Sessions' slot folder or a slot file.
    An agent sandbox (Codex's workspace-write protects a project's top-level `.codex`) refuses
    them while the workspace's own folder stays writable.
    """
    if not isinstance(error, PermissionError) or not isinstance(error.filename, str):
        return False
    path = Path(error.filename).absolute()
    folder = project / ".codex"
    return path == folder / BINDING_NAME or path.is_relative_to(folder / BINDING_DIRECTORY)


def unbound(error: Exception) -> None:
    """Disclose, on this request's answer, a Session the Host could not bind.

    A binding made whose recording failed after it keeps its notice and adds the cause.
    """
    cause = (
        str(error)
        if isinstance(error, NativeBridgeError)
        else f"native_bridge.files_unavailable:{type(error).__name__}"
        if isinstance(error, OSError)
        else type(error).__name__
    )
    cause = safe_failure_code(cause) or cause
    held = BINDING_NOTICE.get()
    status = (
        "BOUND_IN_WORKSPACE" if held and held["status"] == "BOUND_IN_WORKSPACE" else "NOT_BOUND"
    )
    named = Path(error.filename) if isinstance(error, OSError) and error.filename else None
    BINDING_NOTICE.set({**binding_notice(status, cause), **({"path": str(named)} if named else {})})


def remind_binding(host: str, session_id: str, workspace: Path) -> None:
    """Repeat, on this answer, the notice of a Session bound in the workspace's own folder.

    Only beside a configured project; a forgotten memo entry rereads the durable binding.
    """
    key = (host, session_id, workspace.resolve())
    held = _BINDING_MEMO.get(key)
    if held is None:
        try:
            try:
                project = session_project(key[2], host)
                needs_workspace = session_binding(project, None, (host, session_id)) is None
            except NativeBridgeError as error:
                if str(error) != "native_bridge.binding_unreadable":
                    raise
                needs_workspace = True
            held = needs_workspace and bool(session_binding(None, key[2], (host, session_id)))
        except (NativeBridgeError, OSError):
            # Failed reads establish neither presence nor absence; retry after access recovers.
            return
        _BINDING_MEMO[key] = held
    if held and BINDING_NOTICE.get() is None:
        BINDING_NOTICE.set(binding_notice("BOUND_IN_WORKSPACE"))


def bound_in_workspace(
    error: OSError, project: Path | None, workspace: Path, request: NativeSessionBindRequest
) -> dict[str, Any]:
    """Keep an explicit bind the agent's sandbox denies in the project in the workspace instead.

    Every Host reader reads the workspace's own folder, and the answer says so in its own
    words; any other path, or a refusal of the workspace write, stays refused.

    Args:
        error: The refused write.
        project: The admitted project, if one was admitted before the refusal.
        workspace: The served workspace.
        request: The bind request, whose provenance names its host and Session.

    Returns:
        The workspace binding with its `session_binding` notice, or the refusal.
    """
    project = project or workspace
    if not _sandboxed_binding(error, project):
        return files_unavailable(error, project=project, workspace=workspace)
    session = REQUEST_PROVENANCE.get()
    assert session is not None and session.vendor is not None and session.session is not None
    host, roles, root = session.vendor, tuple(_roles(session.vendor)), autobind_root(workspace)
    try:
        bound = bind_session(
            root,
            host=host,
            session_id=session.session,
            workspace=workspace,
            usage=request.usage,
            roles=roles,
        )
    except NativeBridgeError as refused:
        code = str(refused)
        return {"status": "REFUSED", "failure_code": code, **refusal_words(code)}
    except OSError as refused:
        return files_unavailable(refused, project=project, workspace=workspace)
    _BINDING_MEMO[(host, session.session, workspace.resolve())] = True
    notice = binding_notice("BOUND_IN_WORKSPACE")
    words = {k: notice[k] for k in ("detail", "next_action")}
    return {**bound, **words, "session_binding": notice}


def _configuration(host: str = "codex") -> None:
    python = (
        Path(sys.executable)
        if INSTALLED
        else ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not python.is_file():
        raise NativeBridgeError("native_bridge.local_environment_missing")
    if not _roles(host):
        raise NativeBridgeError("native_bridge.roles_missing")


def unbind_session(
    project: Path, *, session_id: str | None, host: str | None = None
) -> dict[str, object]:
    """Detach this Session's own record; other Session bindings and research stay (V586).

    An agent removes only its exact host/Session record. Outside any agent Session,
    a person may detach the sole record; multiple records require an exact owning Session.
    The research is unchanged: a binding is a session's default workspace and the scope of its
    observation, never a record.

    Args:
        project: The agent project holding the binding.
        session_id: The session removing it; None for the person.
        host: The actual host; older shims may omit it only for an unambiguous Session.

    Returns:
        ``NOT_BOUND``, or ``DETACHED`` with the session the removed binding named.

    Raises:
        NativeBridgeError: ``native_bridge.session_mismatch`` when the binding names another
            session than the one removing it; the binding's own read refusals.
    """
    if session_id is None:
        binding = NativeResearchBinding.read(project)
    elif host is not None:
        binding = NativeResearchBinding.read(project, session=(host, session_id))
    else:
        # Compatibility shims did not supply a host. The Session must still name
        # one exact record; an identifier shared by hosts cannot select either.
        candidates = [
            each
            for each in NativeResearchBinding.bindings(project)
            if each.session_id == session_id
        ]
        if len(candidates) > 1:
            raise NativeBridgeError("native_bridge.binding_ambiguous")
        binding = candidates[0] if candidates else None
    if binding is None:
        if session_id is not None and NativeResearchBinding.bindings(project):
            raise NativeBridgeError("native_bridge.session_mismatch")
        return {"status": "NOT_BOUND", "project": str(project)}
    if session_id is not None and binding.session_id != session_id:
        raise NativeBridgeError("native_bridge.session_mismatch")
    binding.record_path(project).unlink()
    clear_usage_checkpoints(binding.workspace, binding)
    return {
        "status": "DETACHED",
        "project": str(project),
        "session_id": binding.session_id,
        "host": binding.host,
        "research_unchanged": True,
    }


def _create_or_match(
    name: str,
    document: dict[str, Any],
    project: Path | None = None,
    *,
    directory: str = ".codex",
) -> None:
    root = ROOT if project is None else project
    path = root / directory / name
    parents = (
        path,
        *(parent for parent in path.parents if parent == root or parent.is_relative_to(root)),
    )
    if any(
        parent.is_symlink() or (os.name == "nt" and parent.is_junction()) for parent in parents
    ) or not path.resolve().is_relative_to(root.resolve()):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    data = (json.dumps(document, indent=2) + "\n").encode()
    # A Claude Code project configured from the installed command holds no `.codex` yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        with path.open("rb") as stream:
            existing = stream.read(len(data) + 1)
        if existing != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs") from None


def main() -> int:
    """Configure, bind or inspect the project: zero on success, two on refusal."""
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, help="The writable agent project; cwd when installed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser(
        "configure", help="Install role guidance and declare a project without product hooks."
    )
    configure.add_argument("--host", choices=HOSTS, default="codex")
    bind = commands.add_parser("bind", help="Bind this bridge to an explicit session/workspace.")
    bind.add_argument("--session-id", required=True)
    bind.add_argument("--workspace", required=True, type=Path)
    bind.add_argument("--host", choices=HOSTS, default="codex", help="The foreground native host.")
    bind.add_argument(
        "--usage",
        choices=[value.lower() for value in USAGE_READINGS],
        default="read",
        help="off: read no native Session file for usage.",
    )
    unbind = commands.add_parser(
        "unbind", help="Detach only the named session; research is unchanged."
    )
    unbind.add_argument("--session-id", required=True)
    unbind.add_argument("--host", choices=HOSTS, help="The exact native host of this Session.")
    doctor = commands.add_parser("doctor", help="Read this Session's binding and readiness.")
    doctor.set_defaults(view="compact")
    doctor.add_argument(
        "--view", choices=("compact", "full", "summary"), help=worded("cli.read.view")
    )
    doctor.add_argument("--output", type=Path)
    doctor.add_argument(
        "--host", choices=HOSTS, help="The host to inspect; by default the bound one, else codex."
    )
    args = parser.parse_args()
    if args.project is not None:
        ROOT = args.project.resolve()
    result: dict[str, Any] = {}
    doctor_command = _doctor_command()
    prefix = (sys.executable, "-m", "alphalattice.interface.local_application.native_setup")
    args.command_line = (*prefix, *sys.argv[1:])
    output = getattr(args, "output", None)
    output_error: client.LocalResearchClientError | None = None
    replace = False
    try:
        replace = client.output_replacement(output, line=args.command_line, prefix=())
        if args.command == "configure":
            if INSTALLED:
                _install_declarations(args.host)
            removed = strip_product_hooks(ROOT, args.host)
            _configuration(args.host)
            declare_project(ROOT, args.host)
            result = {
                "status": "LOCAL_DECLARATIONS_VALIDATED",
                "host": args.host,
                "retired_hook_groups_removed": removed,
                "trust_changed": False,
                "next": (
                    "Continue the research in this session: read the files in `read` by path, "
                    "start each specialist from its card in `specialists`, and tell the person "
                    "`disclosure` in one line."
                ),
                "continue_here": continue_here(ROOT, args.host),
            }
        elif args.command == "bind":
            result = bind_session(
                ROOT,
                host=args.host,
                session_id=args.session_id,
                workspace=args.workspace,
                usage=args.usage,
            )
        elif args.command == "unbind":
            # The checkout's shim of `alphalattice session unbind`, by the session it names.
            result = unbind_session(ROOT, session_id=args.session_id, host=args.host)
        else:
            session = agent_session(os.environ)
            if session is not None and args.host is not None and args.host != session[0]:
                raise NativeBridgeError("native_bridge.session_mismatch")
            binding = NativeResearchBinding.read(ROOT, session=session)
            # An unbound session's declaration is its own host's, as the first use reads it.
            host = args.host or (
                binding.host if binding is not None else session[0] if session else "codex"
            )
            preflight = attachment_preflight(ROOT, binding, host)
            result = {
                **preflight,
                "session_bound": binding is not None,
                "host": binding.host if binding is not None else None,
                "usage_reading": binding.usage if binding is not None else None,
                "roles": _roles(host),
                "attachment_preflight": preflight,
                "host_readiness": host_readiness(ROOT, host, binding),
            }
    except client.LocalResearchClientError as error:
        if str(error).startswith("local_client.output_"):
            output_error = error
        else:
            result = {**setup_failure(error), "next_commands": doctor_command}
    except NativeBridgeError as error:
        result = {**setup_failure(error), "reason": str(error), "next_commands": doctor_command}
    except OSError as error:
        # Its files are the checkout's host directories and the workspace's; one it cannot read
        # or write is said with its way on, never a bare code (V539), and the check its words
        # name is offered (V590).
        result = {
            **setup_failure(error),
            **files_unavailable(error, project=ROOT, workspace=getattr(args, "workspace", None)),
            "next_commands": doctor_command,
        }
    except Exception as error:
        result = {**setup_failure(error), "next_commands": doctor_command}
    return emit_owner(args, result, output_error=output_error, replace=replace)


def _doctor_command() -> dict[str, list[str]]:
    """The binding's own read-only check, every refusal's way on, as the person runs it."""
    command = (
        [
            sys.executable,
            "-m",
            "alphalattice.interface.local_application.native_setup",
            "--project",
            str(ROOT),
            "doctor",
        ]
        if INSTALLED
        else [sys.executable, "scripts/native_research.py", "doctor"]
    )
    return {"doctor": command}


if __name__ == "__main__":
    raise SystemExit(main())
