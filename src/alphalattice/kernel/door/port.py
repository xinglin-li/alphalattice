"""The boundary for operation admission and dispatch."""

import importlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Protocol, cast


class DoorRefusal(ValueError):
    """A named dispatch refusal, carrying the operation that was requested."""

    def __init__(self, code: str, operation: str, *, module: str | None = None) -> None:
        """Carry the refusal code, requested operation and optional family module."""
        super().__init__(code)
        self.failure_code = code
        self.operation = operation
        self.module = module


class DoorPort[CallT, AnswerT](Protocol):
    """Dispatch an admitted request to its registered family."""

    @property
    def operations(self) -> frozenset[str]:
        """Report which operation names this port serves."""
        ...

    def dispatch(self, operation: str, call: CallT) -> AnswerT:
        """Invoke the registered handler or raise a named refusal."""
        ...


@dataclass(frozen=True, slots=True)
class FamilyRoute:
    """The names and module entry point a family registers at Host composition."""

    operations: tuple[str, ...]
    module: str
    handler: str = "execute"


class _Family[CallT, AnswerT]:
    def __init__(self, route: FamilyRoute) -> None:
        self.route = route
        self.lock = Lock()
        self.handler: Callable[[CallT], AnswerT] | None = None

    def resolve(self, operation: str) -> Callable[[CallT], AnswerT]:
        handler = self.handler
        if handler is not None:
            return handler
        with self.lock:
            if self.handler is None:
                try:
                    target = getattr(importlib.import_module(self.route.module), self.route.handler)
                    if not callable(target):
                        raise TypeError(self.route.handler)
                except Exception as error:
                    raise DoorRefusal(
                        "door.family_import_failed", operation, module=self.route.module
                    ) from error
                self.handler = cast(Callable[[CallT], AnswerT], target)
            return self.handler


class Door[CallT, AnswerT]:
    """A fixed operation table whose family code loads on the first request."""

    def __init__(self, routes: Iterable[FamilyRoute]) -> None:
        """Register the family routes without importing their handlers."""
        self._routes: dict[str, _Family[CallT, AnswerT]] = {}
        for route in routes:
            family = _Family[CallT, AnswerT](route)
            for operation in route.operations:
                if operation in self._routes:
                    raise DoorRefusal("door.operation_duplicate", operation, module=route.module)
                self._routes[operation] = family
        self._operations = frozenset(self._routes)

    @property
    def operations(self) -> frozenset[str]:
        """Return the immutable set of registered operation names."""
        return self._operations

    def dispatch(self, operation: str, call: CallT) -> AnswerT:
        """Resolve and invoke the one family registered for this operation."""
        family = self._routes.get(operation)
        if family is None:
            raise DoorRefusal("door.operation_unknown", operation)
        return family.resolve(operation)(call)
