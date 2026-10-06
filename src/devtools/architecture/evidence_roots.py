"""Typed resolution of real-evidence roots that tests may read but never own.

A real-evidence test reads artifacts that exist only on a dogfood machine: a
sibling research worktree, a frozen closure under an ignored ``workspaces/``
directory, a sealed manifest. Those locations used to be spelled inside the
tests themselves -- twice as drive-letter absolute paths, elsewhere as guesses
relative to the worktree's parent -- so the same test skipped, failed or ran
depending on which checkout it was collected from, and nothing in the tree said
which roots the evidence lane needed.

This module is the one owner of that vocabulary. ``config/evidence-roots.json``
declares every root by name with ordered candidates relative to a stated base;
``config/evidence-roots.local.json`` (ignored) and ``ALPHALATTICE_EVIDENCE_ROOTS``
(a path to another such file) override a machine that keeps its evidence
elsewhere. A per-root ``environment_key`` keeps the operator variables earlier
records documented. Resolution never invents a path: an absent root resolves
to ``None`` and the caller decides whether that is a skip. A root declared ``retired``
resolves to ``None`` wherever it is, and says why: the tree no longer reads what it holds.

A root that exists but cannot be read by this process is not materialized for
this process. Trees written under another security context carry ACLs that
make ``stat`` raise, and ``Path.exists`` propagates that error rather than
answering ``False``; treating it as absence, and saying so in the reason, is
what keeps an evidence test a named skip instead of a setup error.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PLAYPEN_ROOT = Path(__file__).resolve().parents[3]
TRACKED_DECLARATION = PLAYPEN_ROOT / "config" / "evidence-roots.json"
LOCAL_OVERRIDE = PLAYPEN_ROOT / "config" / "evidence-roots.local.json"
OVERRIDE_ENVIRONMENT_KEY = "ALPHALATTICE_EVIDENCE_ROOTS"
DECLARATION_KIND = "PlaypenEvidenceRoots"

type EvidenceRootBase = Literal["worktree", "parent", "absolute"]
type EvidenceRootState = Literal["materialized", "absent", "unreadable", "retired"]


class EvidenceRootError(ValueError):
    """The declaration is malformed or names something it cannot mean."""


@dataclass(frozen=True, slots=True)
class EvidenceRootCandidate:
    base: EvidenceRootBase
    path: str

    def resolve(self, *, worktree: Path) -> Path:
        if self.base == "absolute":
            return Path(self.path)
        anchor = worktree if self.base == "worktree" else worktree.parent
        return anchor / self.path


@dataclass(frozen=True, slots=True)
class EvidenceRootDeclaration:
    name: str
    candidates: tuple[EvidenceRootCandidate, ...]
    marker: str | None
    """A path under the root whose presence proves the root is materialized."""

    environment_key: str | None
    note: str
    retired: str | None = None
    """Why this tree no longer reads the root, when it does not: it never materializes."""

    def locations(self, *, worktree: Path, environment: Mapping[str, str]) -> tuple[Path, ...]:
        """Every place this root may be, the environment override first."""

        found: list[Path] = []
        if self.environment_key is not None:
            override = environment.get(self.environment_key)
            if override:
                found.append(Path(override))
        found.extend(candidate.resolve(worktree=worktree) for candidate in self.candidates)
        return tuple(found)

    def state(self, location: Path) -> EvidenceRootState:
        if self.retired is not None:
            return "retired"
        probe = location / self.marker if self.marker else location
        try:
            if not probe.exists():
                return "absent"
            # `stat` can succeed where reading is denied, so the probe is opened.
            if probe.is_dir():
                with os.scandir(probe):
                    pass
            else:
                probe.open("rb").close()
        except OSError:
            return "unreadable"
        return "materialized"

    def materialized(self, *, worktree: Path, environment: Mapping[str, str]) -> Path | None:
        for location in self.locations(worktree=worktree, environment=environment):
            if self.state(location) == "materialized":
                return location
        return None


def _candidate(value: Any) -> EvidenceRootCandidate:
    if not isinstance(value, dict) or set(value) != {"base", "path"}:
        raise EvidenceRootError("devtools.evidence_root_candidate_invalid")
    base = value["base"]
    path = value["path"]
    if base not in ("worktree", "parent", "absolute") or not isinstance(path, str) or not path:
        raise EvidenceRootError("devtools.evidence_root_candidate_invalid")
    if base != "absolute" and (Path(path).is_absolute() or ".." in Path(path).parts):
        raise EvidenceRootError("devtools.evidence_root_candidate_invalid")
    return EvidenceRootCandidate(base=base, path=path)


def _declaration(name: str, value: Any) -> EvidenceRootDeclaration:
    if not isinstance(value, dict):
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    allowed = {"candidates", "marker", "environment_key", "note", "retired"}
    if set(value) - allowed or "note" not in value:
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    candidates = tuple(_candidate(item) for item in value.get("candidates", ()))
    environment_key = value.get("environment_key")
    if not candidates and environment_key is None:
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    marker = value.get("marker")
    if marker is not None and (not isinstance(marker, str) or not marker):
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    if environment_key is not None and not isinstance(environment_key, str):
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    retired = value.get("retired")
    if retired is not None and (not isinstance(retired, str) or not retired):
        raise EvidenceRootError("devtools.evidence_root_declaration_invalid")
    return EvidenceRootDeclaration(
        name=name,
        candidates=candidates,
        marker=marker,
        environment_key=environment_key,
        note=str(value["note"]),
        retired=retired,
    )


def _load_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != DECLARATION_KIND
        or not isinstance(payload.get("roots"), dict)
    ):
        raise EvidenceRootError(f"devtools.evidence_roots_invalid:{path.name}")
    return dict(payload["roots"])


def _environment_override(environment: Mapping[str, str]) -> tuple[Path, ...]:
    value = environment.get(OVERRIDE_ENVIRONMENT_KEY)
    return (Path(value),) if value else ()


@dataclass(frozen=True, slots=True)
class EvidenceRoots:
    """Every declared root, with overrides already merged by name."""

    worktree: Path
    declarations: Mapping[str, EvidenceRootDeclaration]
    environment: Mapping[str, str]

    @classmethod
    def load(
        cls,
        *,
        worktree: Path = PLAYPEN_ROOT,
        environment: Mapping[str, str] | None = None,
    ) -> EvidenceRoots:
        env = dict(os.environ if environment is None else environment)
        declaration = worktree / "config" / "evidence-roots.json"
        if not declaration.is_file():
            raise EvidenceRootError(
                "devtools.private_evidence_lane_unavailable: config/evidence-roots.json; "
                "the real-evidence lane reads maintainers' private QA roots and is not "
                "part of the public checkout. Use the maintainers' checkout for that lane."
            )
        merged: dict[str, Any] = _load_file(declaration)
        overrides = (worktree / "config" / "evidence-roots.local.json", *_environment_override(env))
        for override in overrides:
            if override.is_file():
                merged.update(_load_file(override))
        return cls(
            worktree=worktree,
            declarations={name: _declaration(name, value) for name, value in merged.items()},
            environment=env,
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.declarations))

    def declaration(self, name: str) -> EvidenceRootDeclaration:
        try:
            return self.declarations[name]
        except KeyError as error:
            raise EvidenceRootError(f"devtools.evidence_root_unknown:{name}") from error

    def path(self, name: str) -> Path | None:
        """The materialized root, or ``None``; never a guess."""

        return self.declaration(name).materialized(
            worktree=self.worktree, environment=self.environment
        )

    def describe(self, name: str) -> str:
        """Why the root resolved as it did, one clause per location tried."""

        declaration = self.declaration(name)
        if declaration.retired is not None:
            return f"evidence root {name!r} is retired: {declaration.retired}"
        states = [
            f"{location} ({declaration.state(location)})"
            for location in declaration.locations(
                worktree=self.worktree, environment=self.environment
            )
        ]
        hint = f"; or set {declaration.environment_key}" if declaration.environment_key else ""
        tried = ", ".join(states) if states else "no location declared"
        return f"evidence root {name!r}: {tried}{hint}"


def resolve_evidence_root(name: str) -> Path | None:
    """Module-scope convenience for tests that gate with ``skipif`` at import time."""

    return EvidenceRoots.load().path(name)


def describe_evidence_root(name: str) -> str:
    """The matching skip reason for ``resolve_evidence_root``."""

    return EvidenceRoots.load().describe(name)


__all__ = [
    "DECLARATION_KIND",
    "LOCAL_OVERRIDE",
    "OVERRIDE_ENVIRONMENT_KEY",
    "PLAYPEN_ROOT",
    "TRACKED_DECLARATION",
    "EvidenceRootBase",
    "EvidenceRootCandidate",
    "EvidenceRootDeclaration",
    "EvidenceRootError",
    "EvidenceRootState",
    "EvidenceRoots",
    "describe_evidence_root",
    "resolve_evidence_root",
]
