"""Adapter families: each kind's built-ins declared through entry points, loaded at first use.

A kind is one entry point group of the installed distribution (`alphalattice.alpha_models`,
`alphalattice.risk_estimators`, `alphalattice.sector_forecasts`): each entry maps an id to the
class its module installs. The group is the family list's one home; setuptools writes it sorted
by name, so a catalog states its own order, the one inside its hash, by id. This loader resolves
an id, importing an adapter's module only when that id is first asked for. A kind with workspace
extensions resolves those, by module name in its extension package, through the same refusals.

An id no entry point declares, a declared target that fails to import, and an object that is not
its kind's adapter, or answers to another id, are refused by the kind's own name for the id.
"""

from __future__ import annotations

import importlib
import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata

_DISTRIBUTION = "alphalattice"
_EXTENSION_ID = re.compile(r"[a-z][a-z0-9_]*")


def declared(group: str) -> dict[str, metadata.EntryPoint]:
    """The installed distribution's entry points in `group`, by id: the routes the loader takes
    and the source identity binds."""
    try:
        installed = metadata.distribution(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return {}  # dependencies alone: nothing is declared, and every id is refused by name
    return {point.name: point for point in installed.entry_points.select(group=group)}


def conforms(protocol: type) -> Callable[[object], bool]:
    """Whether an object has every member its kind's protocol declares, read without calling it.

    What `typing.runtime_checkable` would check, without changing the protocol's module.
    """
    members = typing.get_protocol_members(protocol)
    missing = object()
    return lambda value: all(
        inspect.getattr_static(value, name, missing) is not missing for name in members
    )


@dataclass(frozen=True, slots=True)
class AdapterKind[T]:
    """One kind of adapter: its entry point group and how it admits and refuses one."""

    group: str
    """The entry point group that declares the kind's built-ins."""
    admits: Callable[[object], bool]
    """Whether an object is this kind's adapter."""
    answers_to: Callable[[T], str]
    """The id an admitted adapter answers to."""
    refused: Callable[[str], Exception]
    """The kind's refusal of an id."""
    extensions: tuple[str, type] | None = None
    """The package holding workspace extension modules (each with an `ADAPTER` class) and the
    type an extension's adapter must be, when the kind takes extensions."""

    def ids(self) -> frozenset[str]:
        """The declared ids."""
        return frozenset(declared(self.group))

    def load(self, adapter_id: str) -> T:
        """The adapter a built-in id declares, its module imported now.

        Raises:
            Exception: The kind's refusal of `adapter_id`.
        """
        point = declared(self.group).get(adapter_id)
        if point is None:
            raise self.refused(adapter_id)
        try:
            target = point.load()
        except (ImportError, AttributeError) as error:
            raise self.refused(adapter_id) from error
        return self._admitted(target, adapter_id, self.admits)

    def extension(self, adapter_id: str) -> T:
        """The adapter a workspace extension module installs under `adapter_id`.

        Raises:
            Exception: The kind's refusal of `adapter_id`.
        """
        if self.extensions is None or not _EXTENSION_ID.fullmatch(adapter_id):
            raise self.refused(adapter_id)
        package, kind = self.extensions
        try:
            module = importlib.import_module(f"{package}.{adapter_id}")  # lazy: optional
        except ModuleNotFoundError as error:
            raise self.refused(adapter_id) from error
        target = getattr(module, "ADAPTER", None)
        return self._admitted(target, adapter_id, lambda value: isinstance(value, kind))

    def _admitted(self, target: object, adapter_id: str, admits: Callable[[object], bool]) -> T:
        try:
            instance = target() if isinstance(target, type) else None
        except TypeError as error:  # a class its kind cannot construct bare
            raise self.refused(adapter_id) from error
        if instance is None or not admits(instance):
            raise self.refused(adapter_id)
        adapter: T = instance  # admitted as this kind's adapter
        try:
            answered = self.answers_to(adapter)
        except AttributeError as error:  # an owned member without its id
            raise self.refused(adapter_id) from error
        if answered != adapter_id:
            raise self.refused(adapter_id)
        return adapter


__all__ = ["AdapterKind", "conforms", "declared"]
